// SPDX-License-Identifier: GPL-2.0-only WITH LicenseRef-Kaiba-Runtime-Exception
// Copyright (C) The Kaiba Authors
//
// This file is part of the Kaiba compiler toolchain. It is licensed under
// GPL-2.0-only with the Kaiba Runtime/Output Exception; see COPYING and
// LICENSE-EXCEPTION at the repository root.
//! Loop-invariant code motion: hoist pure, non-trapping instructions whose
//! operands are loop-invariant into the loop preheader.

use crate::analysis::DomTree;
use crate::entity::{BlockId, Entity};
use crate::function::{Function, ValueDef};
use crate::inst::{BinOp, Constant, InstKind, Operand, Terminator};
use crate::pass::{Changed, FunctionPass};
use std::collections::{HashMap, HashSet};

pub struct Licm;

/// Natural loops keyed by header (back edges with the same header are merged).
fn find_loops(f: &Function, dt: &DomTree) -> HashMap<BlockId, HashSet<BlockId>> {
    let preds = f.predecessors();
    let mut loops: HashMap<BlockId, HashSet<BlockId>> = HashMap::new();
    for &b in dt.rpo() {
        for s in f.successors(b) {
            if !dt.dominates(s, b) { continue; }
            let body = loops.entry(s).or_default();
            body.insert(s);
            let mut work = vec![b];
            while let Some(x) = work.pop() {
                if body.insert(x) {
                    for &p in &preds[&x] { if dt.is_reachable(p) { work.push(p); } }
                }
            }
        }
    }
    loops
}

fn hoistable(k: &InstKind) -> bool {
    match k {
        InstKind::Binary { op, rhs, .. } => match op {
            // Division can trap; only hoist it when the divisor is a safe constant.
            BinOp::UDiv | BinOp::SDiv | BinOp::URem | BinOp::SRem => {
                matches!(rhs, Operand::Const(Constant::Int { bits, value }) if *value != 0 && *value != (if *bits >= 128 { u128::MAX } else { (1u128 << bits) - 1 }))
            }
            _ => true,
        },
        InstKind::ICmp { .. } | InstKind::FCmp { .. } | InstKind::Cast { .. } | InstKind::Select { .. }
        | InstKind::Gep { .. } | InstKind::ExtractValue { .. } | InstKind::InsertValue { .. } => true,
        _ => false,
    }
}

fn process_loop(f: &mut Function, header: BlockId, body: &HashSet<BlockId>, preds: &HashMap<BlockId, Vec<BlockId>>, dt: &DomTree) -> bool {
    let outside: Vec<BlockId> = preds[&header].iter().copied().filter(|p| !body.contains(p) && dt.is_reachable(*p)).collect();
    let [o] = outside[..] else { return false };
    let mut preheader: Option<BlockId> = None;
    let mut moved_any = false;
    loop {
        let mut moved = false;
        for b in f.layout.clone() {
            if !body.contains(&b) { continue; }
            for i in f.blocks[b].insts.clone() {
                if !hoistable(&f.insts[i].kind) { continue; }
                let mut invariant = true;
                f.insts[i].kind.for_each_operand(|op| {
                    if let Operand::Value(v) = op {
                        if let ValueDef::Inst(d) = f.values[*v].def {
                            if body.contains(&f.insts[d].block) { invariant = false; }
                        }
                    }
                });
                if !invariant { continue; }

                // First hoist for this loop: make sure there is a dedicated preheader.
                let ph = *preheader.get_or_insert_with(|| {
                    if matches!(f.blocks[o].terminator, Some(Terminator::Br(t)) if t == header) { return o; }
                    let p = f.create_block(Some("preheader"));
                    let pos = f.layout.iter().position(|&x| x == o).unwrap();
                    f.layout.pop();
                    f.layout.insert(pos + 1, p);
                    f.set_terminator(p, Terminator::Br(header));
                    f.blocks[o].terminator.as_mut().unwrap().map_successors(|x| if x == header { p } else { x });
                    for phi in f.blocks[header].insts.clone() {
                        if let InstKind::Phi { incoming, .. } = &mut f.insts[phi].kind {
                            for (pb, _) in incoming.iter_mut() { if *pb == o { *pb = p; } }
                        }
                    }
                    p
                });
                f.blocks[b].insts.retain(|&x| x != i);
                f.blocks[ph].insts.push(i);
                f.insts[i].block = ph;
                moved = true;
                moved_any = true;
            }
        }
        if !moved { break; }
    }
    moved_any
}

impl FunctionPass for Licm {
    fn name(&self) -> &'static str { "licm" }

    fn run(&mut self, f: &mut Function) -> Changed {
        let mut changed = false;
        let mut done: HashSet<BlockId> = HashSet::new();
        // Innermost loops first; the CFG may change (preheaders), so recompute each time.
        loop {
            let dt = DomTree::compute(f);
            let loops = find_loops(f, &dt);
            let Some((&h, body)) = loops.iter().filter(|(h, _)| !done.contains(h)).min_by_key(|(h, b)| (b.len(), h.index())) else { break };
            done.insert(h);
            let preds = f.predecessors();
            changed |= process_loop(f, h, body, &preds, &dt);
        }
        Changed::from_bool(changed)
    }
}
