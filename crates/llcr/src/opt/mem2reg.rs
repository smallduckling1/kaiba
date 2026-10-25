// SPDX-License-Identifier: GPL-2.0-only WITH LicenseRef-Kaiba-Runtime-Exception
// Copyright (C) The Kaiba Authors
//
// This file is part of the Kaiba compiler toolchain. It is licensed under
// GPL-2.0-only with the Kaiba Runtime/Output Exception; see COPYING and
// LICENSE-EXCEPTION at the repository root.
//! mem2reg: promote single-value stack slots to SSA registers using the
//! iterated dominance frontier for phi placement (Cytron et al.) and a
//! dominator-tree renaming walk.
//!
//! A slot is promotable if it is an entry-block `alloca` of one scalar and
//! every use is a non-volatile, same-typed `load` from it or `store` to it
//! (the slot's address never escapes).

use crate::analysis::DomTree;
use crate::entity::{BlockId, Entity, InstId, ValueId};
use crate::function::Function;
use crate::inst::{Constant, InstKind, Operand};
use crate::pass::{Changed, FunctionPass};
use crate::types::Type;
use std::collections::{HashMap, HashSet};

pub struct Mem2Reg;

struct Slot { inst: InstId, value: ValueId, ty: Type }

enum Work { Enter(BlockId), Exit(Vec<usize>) }

fn undef(ty: Type) -> Operand { Operand::Const(Constant::Undef(ty)) }

impl FunctionPass for Mem2Reg {
    fn name(&self) -> &'static str { "mem2reg" }

    fn run(&mut self, f: &mut Function) -> Changed {
        let Some(entry) = f.entry() else { return Changed::No };

        // 1. Candidate slots.
        let mut slots: Vec<Slot> = Vec::new();
        for &i in &f.blocks[entry].insts {
            if let InstKind::Alloca { ty, count, .. } = &f.insts[i].kind {
                let one = matches!(count, Operand::Const(Constant::Int { value: 1, .. }));
                if one && !matches!(ty, Type::Agg(_) | Type::Void) {
                    slots.push(Slot { inst: i, value: f.insts[i].result.unwrap(), ty: *ty });
                }
            }
        }
        if slots.is_empty() { return Changed::No; }

        // 2. Disqualify slots whose address escapes or that are accessed oddly.
        let index: HashMap<ValueId, usize> = slots.iter().enumerate().map(|(k, s)| (s.value, k)).collect();
        let slot_of = |o: &Operand| -> Option<usize> {
            if let Operand::Value(v) = o { index.get(v).copied() } else { None }
        };
        let mut bad = vec![false; slots.len()];
        for (_, i) in f.iter_insts() {
            match &f.insts[i].kind {
                InstKind::Load { ptr, ty, volatile, .. } => {
                    if let Some(k) = slot_of(ptr) { if *volatile || *ty != slots[k].ty { bad[k] = true; } }
                }
                InstKind::Store { val, ptr, volatile, .. } => {
                    if let Some(k) = slot_of(val) { bad[k] = true; }
                    if let Some(k) = slot_of(ptr) {
                        if *volatile || f.operand_type(val) != slots[k].ty { bad[k] = true; }
                    }
                }
                k => k.for_each_operand(|o| { if let Some(s) = slot_of(o) { bad[s] = true; } }),
            }
        }
        for &b in &f.layout {
            if let Some(t) = &f.blocks[b].terminator {
                t.for_each_operand(|o| { if let Some(s) = slot_of(o) { bad[s] = true; } });
            }
        }
        let slots: Vec<Slot> = slots.into_iter().zip(bad).filter(|(_, b)| !b).map(|(s, _)| s).collect();
        if slots.is_empty() { return Changed::No; }
        let index: HashMap<ValueId, usize> = slots.iter().enumerate().map(|(k, s)| (s.value, k)).collect();

        // 3. Phi placement at the iterated dominance frontier of each slot's stores.
        let dt = DomTree::compute(f);
        let df = dt.frontiers(f);
        let mut phi_slot: HashMap<InstId, usize> = HashMap::new();
        let mut block_phis: HashMap<BlockId, Vec<(InstId, usize)>> = HashMap::new();
        for (k, s) in slots.iter().enumerate() {
            let mut defs: Vec<BlockId> = f
                .iter_insts()
                .filter(|&(b, i)| {
                    dt.is_reachable(b)
                        && matches!(&f.insts[i].kind, InstKind::Store { ptr, .. } if *ptr == Operand::Value(s.value))
                })
                .map(|(b, _)| b)
                .collect();
            defs.sort();
            defs.dedup();
            let def_set: HashSet<BlockId> = defs.iter().copied().collect();
            let mut has_phi: HashSet<BlockId> = HashSet::new();
            let mut work = defs;
            while let Some(x) = work.pop() {
                for &y in &df[x.index()] {
                    if has_phi.insert(y) {
                        let (pi, _) = f.prepend_inst(y, InstKind::Phi { ty: s.ty, incoming: vec![] }, Some(s.ty));
                        phi_slot.insert(pi, k);
                        block_phis.entry(y).or_default().push((pi, k));
                        if !def_set.contains(&y) { work.push(y); }
                    }
                }
            }
        }

        // 4. Rename along the dominator tree.
        let mut children: Vec<Vec<BlockId>> = vec![vec![]; f.blocks.len()];
        for &b in dt.rpo() {
            if let Some(p) = dt.idom(b) { children[p.index()].push(b); }
        }
        let mut stacks: Vec<Vec<Operand>> = vec![vec![]; slots.len()];
        let mut repl: HashMap<ValueId, Operand> = HashMap::new();
        let mut dead: Vec<InstId> = slots.iter().map(|s| s.inst).collect();
        let mut work = vec![Work::Enter(entry)];
        while let Some(w) = work.pop() {
            match w {
                Work::Exit(pushed) => { for k in pushed { stacks[k].pop(); } }
                Work::Enter(b) => {
                    let mut pushed = Vec::new();
                    for i in f.blocks[b].insts.clone() {
                        if let Some(&k) = phi_slot.get(&i) {
                            stacks[k].push(Operand::Value(f.insts[i].result.unwrap()));
                            pushed.push(k);
                            continue;
                        }
                        match f.insts[i].kind.clone() {
                            InstKind::Load { ptr: Operand::Value(v), .. } => {
                                if let Some(&k) = index.get(&v) {
                                    let cur = stacks[k].last().copied().unwrap_or(undef(slots[k].ty));
                                    repl.insert(f.insts[i].result.unwrap(), cur);
                                    dead.push(i);
                                }
                            }
                            InstKind::Store { val, ptr: Operand::Value(v), .. } => {
                                if let Some(&k) = index.get(&v) {
                                    let mut val = val;
                                    while let Operand::Value(x) = val {
                                        match repl.get(&x) { Some(n) => val = *n, None => break }
                                    }
                                    stacks[k].push(val);
                                    pushed.push(k);
                                    dead.push(i);
                                }
                            }
                            _ => {}
                        }
                    }
                    let mut succs = f.successors(b);
                    succs.dedup();
                    for s in succs {
                        if let Some(ph) = block_phis.get(&s) {
                            for &(pi, k) in ph {
                                let v = stacks[k].last().copied().unwrap_or(undef(slots[k].ty));
                                if let InstKind::Phi { incoming, .. } = &mut f.insts[pi].kind { incoming.push((b, v)); }
                            }
                        }
                    }
                    work.push(Work::Exit(pushed));
                    for &c in children[b.index()].iter().rev() { work.push(Work::Enter(c)); }
                }
            }
        }

        // Unreachable blocks: loads read undef, stores vanish.
        for b in f.layout.clone() {
            if dt.is_reachable(b) { continue; }
            for i in f.blocks[b].insts.clone() {
                match &f.insts[i].kind {
                    InstKind::Load { ptr: Operand::Value(v), .. } if index.contains_key(v) => {
                        repl.insert(f.insts[i].result.unwrap(), undef(slots[index[v]].ty));
                        dead.push(i);
                    }
                    InstKind::Store { ptr: Operand::Value(v), .. } if index.contains_key(v) => dead.push(i),
                    _ => {}
                }
            }
        }

        // 5. Apply load replacements everywhere, then delete the dead memory ops.
        f.for_each_operand_mut(|o| {
            while let Operand::Value(v) = *o {
                match repl.get(&v) { Some(n) => *o = *n, None => break }
            }
        });
        for i in dead { f.remove_inst(i); }
        Changed::Yes
    }
}
