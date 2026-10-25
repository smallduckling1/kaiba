// SPDX-License-Identifier: GPL-2.0-only WITH LicenseRef-Kaiba-Runtime-Exception
// Copyright (C) The Kaiba Authors
//
// This file is part of the Kaiba compiler toolchain. It is licensed under
// GPL-2.0-only with the Kaiba Runtime/Output Exception; see COPYING and
// LICENSE-EXCEPTION at the repository root.
//! Register allocation over exact live ranges.
//!
//! Each virtual register's liveness is computed per block (dataflow), giving a
//! list of position ranges with holes where the value is dead. Registers are
//! assigned greedily in order of spill-cost density (loop-weighted uses per unit
//! of lifetime), each to the first register whose already-assigned ranges do not
//! overlap. A value that finds no register lives on the stack for its whole lifetime.

use crate::mir::{MFunc, MInst};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Loc<R> {
    Reg(R),
    /// Index of an 8-byte spill slot.
    Spill(usize),
}

pub struct Allocation<R> {
    pub locs: Vec<Option<Loc<R>>>,
    pub num_spills: usize,
    /// Callee-saved registers that were assigned (must be saved/restored).
    pub used_regs: Vec<R>,
}

#[derive(Clone)]
struct BitSet(Vec<u64>);
impl BitSet {
    fn new(n: usize) -> Self { BitSet(vec![0; (n + 63) / 64]) }
    fn set(&mut self, i: usize) { self.0[i / 64] |= 1 << (i % 64); }
    fn clear(&mut self, i: usize) { self.0[i / 64] &= !(1 << (i % 64)); }
    fn get(&self, i: usize) -> bool { self.0[i / 64] >> (i % 64) & 1 == 1 }
    fn iter(&self) -> impl Iterator<Item = usize> + '_ {
        self.0.iter().enumerate().flat_map(|(w, &bits)| (0..64).filter(move |b| bits >> b & 1 == 1).map(move |b| w * 64 + b))
    }
}

type Range = (u32, u32); // inclusive positions

/// Do two sorted range lists share a position?
fn overlaps(a: &[Range], b: &[Range]) -> bool {
    let (mut i, mut j) = (0, 0);
    while i < a.len() && j < b.len() {
        if a[i].0 <= b[j].1 && b[j].0 <= a[i].1 { return true; }
        if a[i].1 < b[j].1 { i += 1; } else { j += 1; }
    }
    false
}

/// Allocate registers. `callee_saved` survive calls; `caller_saved` are only
/// used for live ranges that do not span a call.
pub fn allocate<R: Copy + PartialEq>(f: &MFunc, callee_saved: &[R], caller_saved: &[R]) -> Allocation<R> {
    let nv = f.num_vregs as usize;
    let nb = f.blocks.len();

    // Block-local use (before def) and def sets.
    let mut gen: Vec<BitSet> = (0..nb).map(|_| BitSet::new(nv)).collect();
    let mut kill: Vec<BitSet> = (0..nb).map(|_| BitSet::new(nv)).collect();
    for (b, blk) in f.blocks.iter().enumerate() {
        for inst in &blk.insts {
            inst.for_each_use(|u| { if !kill[b].get(u as usize) { gen[b].set(u as usize); } });
            inst.for_each_def(|d| kill[b].set(d as usize));
        }
    }
    let succs: Vec<Vec<usize>> = f.blocks.iter().map(|b| b.insts.last().map(|t| t.successors()).unwrap_or_default()).collect();
    let mut live_in: Vec<BitSet> = (0..nb).map(|_| BitSet::new(nv)).collect();
    let mut live_out: Vec<BitSet> = (0..nb).map(|_| BitSet::new(nv)).collect();
    loop {
        let mut changed = false;
        for b in (0..nb).rev() {
            let mut out = BitSet::new(nv);
            for &s in &succs[b] { for (w, x) in out.0.iter_mut().zip(&live_in[s].0) { *w |= x; } }
            let mut inn = BitSet::new(nv);
            for w in 0..inn.0.len() { inn.0[w] = gen[b].0[w] | (out.0[w] & !kill[b].0[w]); }
            if inn.0 != live_in[b].0 || out.0 != live_out[b].0 { changed = true; }
            live_in[b] = inn;
            live_out[b] = out;
        }
        if !changed { break; }
    }

    // Approximate loop depth per block: a jump back to an earlier-or-equal block
    // (in layout order) makes every block in between part of a loop.
    let mut depth = vec![0u32; nb];
    for b in 0..nb {
        for &t in &succs[b] {
            if t <= b { for d in depth[t..=b].iter_mut() { *d += 1; } }
        }
    }

    // Positions: instruction n (global index) reads its operands at 2n and writes at 2n+1,
    // so a value last read by an instruction may share a register with one it defines.
    let mut base = vec![0u32; nb];
    let mut total = 0u32;
    for (b, blk) in f.blocks.iter().enumerate() { base[b] = total; total += blk.insts.len() as u32; }
    let mut weight = vec![0f64; nv];
    let mut used = vec![false; nv];
    let mut calls: Vec<u32> = Vec::new();
    for (b, blk) in f.blocks.iter().enumerate() {
        let w = 10f64.powi(depth[b].min(5) as i32);
        for (k, inst) in blk.insts.iter().enumerate() {
            if matches!(inst, MInst::Call { .. }) { calls.push(2 * (base[b] + k as u32)); }
            inst.for_each_use(|u| { weight[u as usize] += w; used[u as usize] = true; });
            inst.for_each_def(|d| weight[d as usize] += w);
        }
    }

    // Build exact ranges walking blocks and instructions backwards (Wimmer & Mossenbock).
    // Ranges are pushed in descending order; the last element is always the lowest.
    let mut ranges: Vec<Vec<Range>> = vec![Vec::new(); nv];
    let add = |ranges: &mut Vec<Vec<Range>>, v: usize, from: u32, to: u32| {
        if let Some(last) = ranges[v].last_mut() {
            if last.0 <= to + 1 { last.0 = last.0.min(from); last.1 = last.1.max(to); return; }
        }
        ranges[v].push((from, to));
    };
    for b in (0..nb).rev() {
        let n = f.blocks[b].insts.len() as u32;
        if n == 0 { continue; }
        let (bs, be) = (2 * base[b], 2 * (base[b] + n) - 1);
        let mut live = live_out[b].clone();
        for v in live_out[b].iter() { add(&mut ranges, v, bs, be); }
        for (k, inst) in f.blocks[b].insts.iter().enumerate().rev() {
            let pos = 2 * (base[b] + k as u32);
            inst.for_each_def(|d| {
                let d = d as usize;
                if live.get(d) { ranges[d].last_mut().unwrap().0 = pos + 1; } else { add(&mut ranges, d, pos + 1, pos + 1); }
                live.clear(d);
            });
            inst.for_each_use(|u| { add(&mut ranges, u as usize, bs, pos); live.set(u as usize); });
        }
    }
    for r in ranges.iter_mut() { r.reverse(); }

    // Allocate in order of spill-cost density: hot, short values first.
    let len_of = |v: usize| ranges[v].iter().map(|r| (r.1 - r.0 + 1) as f64).sum::<f64>();
    let prio = |v: usize| weight[v] / len_of(v);
    let mut order: Vec<usize> = (0..nv).filter(|&v| used[v] && !ranges[v].is_empty()).collect();
    order.sort_by(|&a, &b| prio(b).partial_cmp(&prio(a)).unwrap().then(a.cmp(&b)));

    // A value that crosses a call (is live both before and after it) needs a callee-saved register.
    let crosses_call = |v: usize| ranges[v].iter().any(|&(s, e)| {
        let i = calls.partition_point(|&p| p < s);
        i < calls.len() && calls[i] < e
    });

    let all: Vec<(R, bool)> = caller_saved.iter().map(|&r| (r, false)).chain(callee_saved.iter().map(|&r| (r, true))).collect();
    let mut occupied: Vec<Vec<Range>> = vec![Vec::new(); all.len()];
    let mut locs: Vec<Option<Loc<R>>> = vec![None; nv];
    let mut num_spills = 0;
    for v in order {
        let cross = crosses_call(v);
        let pick = (0..all.len()).find(|&k| (!cross || all[k].1) && !overlaps(&occupied[k], &ranges[v]));
        match pick {
            Some(k) => {
                locs[v] = Some(Loc::Reg(all[k].0));
                occupied[k].extend_from_slice(&ranges[v]);
                occupied[k].sort_unstable();
            }
            None => { locs[v] = Some(Loc::Spill(num_spills)); num_spills += 1; }
        }
    }
    // Only callee-saved registers need saving in the prologue.
    let used_regs: Vec<R> = callee_saved.iter().copied()
        .filter(|r| locs.iter().any(|l| *l == Some(Loc::Reg(*r))))
        .collect();
    Allocation { locs, num_spills, used_regs }
}
