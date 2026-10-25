// SPDX-License-Identifier: GPL-2.0-only WITH LicenseRef-Kaiba-Runtime-Exception
// Copyright (C) The Kaiba Authors
//
// This file is part of the Kaiba compiler toolchain. It is licensed under
// GPL-2.0-only with the Kaiba Runtime/Output Exception; see COPYING and
// LICENSE-EXCEPTION at the repository root.
//! Structural verifier: terminators, phi placement and incoming edges, and
//! SSA dominance (every use is dominated by its definition).

use crate::analysis::DomTree;
use crate::entity::{BlockId, InstId, ValueId};
use crate::function::{Function, ValueDef};
use crate::inst::{InstKind, Operand};
use std::collections::HashMap;

#[derive(Debug)]
pub struct VerifyError(pub String);

fn check_use(
    f: &Function, dt: &DomTree, pos: &HashMap<InstId, usize>,
    v: ValueId, ub: BlockId, upos: Option<usize>,
) -> Option<String> {
    let ValueDef::Inst(d) = f.values[v].def else { return None };
    let Some(&dpos) = pos.get(&d) else { return Some(format!("{v} used but its definition was removed")) };
    let db = f.insts[d].block;
    if db == ub {
        match upos {
            Some(p) if dpos >= p => Some(format!("{v} used before its definition in {ub}")),
            _ => None,
        }
    } else if dt.dominates(db, ub) {
        None
    } else {
        Some(format!("definition of {v} in {db} does not dominate use in {ub}"))
    }
}

pub fn verify_function(f: &Function) -> Result<(), VerifyError> {
    let mut errs: Vec<String> = Vec::new();
    for &b in &f.layout {
        let bd = &f.blocks[b];
        if bd.terminator.is_none() { errs.push(format!("{b} has no terminator")); }
        let mut seen_non_phi = false;
        for &i in &bd.insts {
            let k = &f.insts[i].kind;
            if k.is_phi() && seen_non_phi { errs.push(format!("phi after non-phi in {b}")); }
            if !k.is_phi() { seen_non_phi = true; }
            if f.insts[i].block != b { errs.push(format!("{i} has wrong parent block")); }
        }
        for s in f.successors(b) {
            if !f.layout.contains(&s) { errs.push(format!("{b} branches to dead block {s}")); }
        }
    }
    if errs.is_empty() {
        let dt = DomTree::compute(f);
        let preds = f.predecessors();
        let pos: HashMap<InstId, usize> = f
            .layout.iter()
            .flat_map(|&b| f.blocks[b].insts.iter().enumerate().map(|(p, &i)| (i, p)))
            .collect();
        for &b in &f.layout {
            if !dt.is_reachable(b) { continue; }
            for (p, &i) in f.blocks[b].insts.iter().enumerate() {
                match &f.insts[i].kind {
                    InstKind::Phi { incoming, .. } => {
                        for (pb, op) in incoming {
                            if !preds[&b].contains(pb) {
                                errs.push(format!("phi in {b} has incoming from non-predecessor {pb}"));
                            } else if let Operand::Value(v) = op {
                                if dt.is_reachable(*pb) {
                                    errs.extend(check_use(f, &dt, &pos, *v, *pb, None));
                                }
                            }
                        }
                    }
                    k => k.for_each_operand(|op| {
                        if let Operand::Value(v) = op { errs.extend(check_use(f, &dt, &pos, *v, b, Some(p))); }
                    }),
                }
            }
            if let Some(t) = &f.blocks[b].terminator {
                t.for_each_operand(|op| {
                    if let Operand::Value(v) = op { errs.extend(check_use(f, &dt, &pos, *v, b, None)); }
                });
            }
        }
    }
    if errs.is_empty() { Ok(()) } else { Err(VerifyError(format!("in @{}: {}", f.name, errs.join("; ")))) }
}
