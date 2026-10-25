// SPDX-License-Identifier: GPL-2.0-only WITH LicenseRef-Kaiba-Runtime-Exception
// Copyright (C) The Kaiba Authors
//
// This file is part of the Kaiba compiler toolchain. It is licensed under
// GPL-2.0-only with the Kaiba Runtime/Output Exception; see COPYING and
// LICENSE-EXCEPTION at the repository root.
//! CFG simplification: constant-branch folding, unreachable block removal,
//! trivial phi removal, and merging of straight-line blocks.

use crate::analysis::reverse_postorder;
use crate::entity::{BlockId, Entity};
use crate::function::Function;
use crate::inst::{Constant, InstKind, Operand, Terminator};
use crate::pass::{Changed, FunctionPass};

pub struct SimplifyCfg;

impl FunctionPass for SimplifyCfg {
    fn name(&self) -> &'static str { "simplifycfg" }

    fn run(&mut self, f: &mut Function) -> Changed {
        let mut any = false;
        loop {
            let mut c = false;
            c |= fold_terminators(f);
            c |= remove_unreachable(f);
            c |= simplify_phis(f);
            c |= merge_blocks(f);
            if !c { break; }
            any = true;
        }
        Changed::from_bool(any)
    }
}

fn mask(bits: u32) -> u128 { if bits >= 128 { u128::MAX } else { (1u128 << bits) - 1 } }

/// Drop `pred` from the incoming lists of all phis in `succ`.
fn remove_phi_incoming(f: &mut Function, succ: BlockId, pred: BlockId) {
    for i in f.blocks[succ].insts.clone() {
        if let InstKind::Phi { incoming, .. } = &mut f.insts[i].kind {
            incoming.retain(|(pb, _)| *pb != pred);
        }
    }
}

fn fold_terminators(f: &mut Function) -> bool {
    let mut changed = false;
    for b in f.layout.clone() {
        let Some(t) = f.blocks[b].terminator.clone() else { continue };
        let new = match &t {
            Terminator::CondBr { cond, then_bb, else_bb } => {
                if then_bb == else_bb {
                    Some(Terminator::Br(*then_bb))
                } else if let Operand::Const(Constant::Int { value, .. }) = cond {
                    Some(Terminator::Br(if value & 1 == 1 { *then_bb } else { *else_bb }))
                } else { None }
            }
            Terminator::Switch { val, default, cases } => {
                if let Operand::Const(Constant::Int { bits, value }) = val {
                    let m = mask(*bits);
                    let target = cases.iter().find(|(v, _)| v & m == value & m).map(|(_, b)| *b).unwrap_or(*default);
                    Some(Terminator::Br(target))
                } else if cases.is_empty() {
                    Some(Terminator::Br(*default))
                } else { None }
            }
            _ => None,
        };
        if let Some(n) = new {
            let new_succs = n.successors();
            f.blocks[b].terminator = Some(n);
            for s in t.successors() {
                if !new_succs.contains(&s) { remove_phi_incoming(f, s, b); }
            }
            changed = true;
        }
    }
    changed
}

fn remove_unreachable(f: &mut Function) -> bool {
    if f.entry().is_none() { return false; }
    let mut reach = vec![false; f.blocks.len()];
    for b in reverse_postorder(f) { reach[b.index()] = true; }
    let dead: Vec<BlockId> = f.layout.iter().copied().filter(|b| !reach[b.index()]).collect();
    if dead.is_empty() { return false; }
    for &d in &dead {
        for s in f.successors(d) {
            if reach[s.index()] { remove_phi_incoming(f, s, d); }
        }
    }
    f.layout.retain(|b| reach[b.index()]);
    true
}

fn simplify_phis(f: &mut Function) -> bool {
    let mut changed = false;
    let all: Vec<_> = f.iter_insts().map(|(_, i)| i).collect();
    for i in all {
        let repl = match &f.insts[i].kind {
            InstKind::Phi { incoming, .. } => {
                let r = f.insts[i].result.unwrap();
                let mut uniq: Option<Operand> = None;
                let mut ok = true;
                for (_, o) in incoming {
                    if *o == Operand::Value(r) { continue; }
                    match uniq {
                        None => uniq = Some(*o),
                        Some(u) if u == *o => {}
                        _ => { ok = false; break; }
                    }
                }
                if ok { uniq.map(|u| (r, u)) } else { None }
            }
            _ => None,
        };
        if let Some((r, u)) = repl {
            f.replace_all_uses(r, u);
            f.remove_inst(i);
            changed = true;
        }
    }
    changed
}

/// Merge one block into its unique predecessor (if that predecessor ends in
/// an unconditional branch to it). Returns after the first merge.
fn merge_blocks(f: &mut Function) -> bool {
    let preds = f.predecessors();
    let entry = f.entry();
    for b in f.layout.clone() {
        if Some(b) == entry { continue; }
        let ps = &preds[&b];
        if ps.len() != 1 { continue; }
        let p = ps[0];
        if p == b || !matches!(f.blocks[p].terminator, Some(Terminator::Br(t)) if t == b) { continue; }

        for i in std::mem::take(&mut f.blocks[b].insts) {
            if let InstKind::Phi { incoming, .. } = &f.insts[i].kind {
                let r = f.insts[i].result.unwrap();
                let o = incoming.iter().find(|(pb, _)| *pb == p).map(|(_, o)| *o)
                    .unwrap_or(Operand::Const(Constant::Undef(f.values[r].ty)));
                f.replace_all_uses(r, o);
            } else {
                f.insts[i].block = p;
                f.blocks[p].insts.push(i);
            }
        }
        let term = f.blocks[b].terminator.take();
        if let Some(t) = &term {
            for s in t.successors() {
                for i in f.blocks[s].insts.clone() {
                    if let InstKind::Phi { incoming, .. } = &mut f.insts[i].kind {
                        for (pb, _) in incoming.iter_mut() { if *pb == b { *pb = p; } }
                    }
                }
            }
        }
        f.blocks[p].terminator = term;
        f.layout.retain(|&x| x != b);
        return true;
    }
    false
}
