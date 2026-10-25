// SPDX-License-Identifier: GPL-2.0-only WITH LicenseRef-Kaiba-Runtime-Exception
// Copyright (C) The Kaiba Authors
//
// This file is part of the Kaiba compiler toolchain. It is licensed under
// GPL-2.0-only with the Kaiba Runtime/Output Exception; see COPYING and
// LICENSE-EXCEPTION at the repository root.
//! Machine-IR clean-up passes: address-mode folding and dead-code sweep.

use crate::mir::*;

/// Number of definitions and uses of every vreg.
fn counts(f: &MFunc) -> (Vec<u32>, Vec<u32>) {
    let n = f.num_vregs as usize;
    let (mut defs, mut uses) = (vec![0u32; n], vec![0u32; n]);
    for b in &f.blocks {
        for i in &b.insts {
            i.for_each_def(|d| defs[d as usize] += 1);
            i.for_each_use(|u| uses[u as usize] += 1);
        }
    }
    (defs, uses)
}

/// Does `inst` write `v`?
fn writes(inst: &MInst, v: VReg) -> bool {
    let mut hit = false;
    inst.for_each_def(|d| if d == v { hit = true; });
    hit
}

fn vreg(v: &Val) -> Option<VReg> { if let Val::V(x) = v { Some(*x) } else { None } }

/// Position of the (single) definition of `t` before `before` in the same block.
fn def_before(insts: &[MInst], before: usize, t: VReg) -> Option<usize> {
    (0..before).rev().find(|&j| writes(&insts[j], t))
}

/// `base + const` and `base + idx*scale` feeding a load/store are merged into the access.
pub fn fold_addresses(f: &mut MFunc) {
    for b in 0..f.blocks.len() {
        for i in 0..f.blocks[b].insts.len() {
            let mut addr = match &f.blocks[b].insts[i] {
                MInst::Load { addr, .. } | MInst::Store { addr, .. } => addr.clone(),
                _ => continue,
            };
            let mut changed = false;
            loop {
                let insts = &f.blocks[b].insts;
                let Some(t) = vreg(&addr.base) else { break };
                let Some(j) = def_before(insts, i, t) else { break };
                // Look through copies (the folded value must still be the same at the access).
                if let MInst::Mov { src, .. } | MInst::Ext { kind: ExtKind::Zext, from: 32 | 64, to: 64, src, .. } = &insts[j] {
                    if vreg(src).map_or(true, |x| !insts[j + 1..i].iter().any(|k| writes(k, x))) {
                        addr.base = src.clone();
                        changed = true;
                        continue;
                    }
                    break;
                }
                let MInst::Alu { op: Alu::Add, bits: 64, lhs, rhs, .. } = &insts[j] else { break };
                // Operands must hold the same values at the access as at the add.
                let stable = |v: &Val| vreg(v).map_or(true, |x| !insts[j + 1..i].iter().any(|k| writes(k, x)));
                let imm = |v: &Val| if let Val::Imm(c) = v { i32::try_from(*c as i64).ok() } else { None };
                let (new_base, new_index, add_disp);
                if let (Some(c), true) = (imm(rhs), stable(lhs)) {
                    (new_base, new_index, add_disp) = (lhs.clone(), addr.index.clone(), c);
                } else if let (Some(c), true) = (imm(lhs), stable(rhs)) {
                    (new_base, new_index, add_disp) = (rhs.clone(), addr.index.clone(), c);
                } else if addr.index.is_none() && stable(lhs) && stable(rhs) {
                    // base + index: look through a multiply/shift by a power of two.
                    let mut found = None;
                    for (basev, idxv) in [(lhs, rhs), (rhs, lhs)] {
                        let Some(m) = vreg(idxv) else { continue };
                        let Some(k) = def_before(insts, j, m) else { continue };
                        let scaled = match &insts[k] {
                            MInst::Alu { op: Alu::Mul, bits: 64, lhs: x, rhs: Val::Imm(s), .. } if matches!(s, 1 | 2 | 4 | 8) => Some((x.clone(), *s as u8)),
                            MInst::Alu { op: Alu::Shl, bits: 64, lhs: x, rhs: Val::Imm(s), .. } if *s <= 3 => Some((x.clone(), 1u8 << *s)),
                            _ => None,
                        };
                        if let Some((x, s)) = scaled {
                            let ok = vreg(&x).map_or(true, |xv| !insts[k + 1..i].iter().any(|q| writes(q, xv)));
                            if ok { found = Some((basev.clone(), Some((x, s)))); break; }
                        }
                    }
                    let (bv, ix) = found.unwrap_or_else(|| (lhs.clone(), Some((rhs.clone(), 1u8))));
                    (new_base, new_index, add_disp) = (bv, ix, 0);
                } else { break; }
                let Some(d) = addr.disp.checked_add(add_disp) else { break };
                addr = Addr { base: new_base, index: new_index, disp: d };
                changed = true;
            }
            if changed {
                match &mut f.blocks[b].insts[i] {
                    MInst::Load { addr: a, .. } | MInst::Store { addr: a, .. } => *a = addr,
                    _ => unreachable!(),
                }
            }
        }
    }
}

/// Replace uses of a copy's destination by its source when both are defined exactly once
/// (so the source cannot change between the copy and any use).
pub fn propagate_copies(f: &mut MFunc) {
    let (defc, _) = counts(f);
    let n = f.num_vregs as usize;
    let mut alias: Vec<Option<Val>> = vec![None; n];
    for b in &f.blocks {
        for i in &b.insts {
            if let MInst::Mov { dst, src } = i {
                if defc[*dst as usize] != 1 { continue; }
                match src {
                    Val::Imm(_) | Val::Sym { .. } => alias[*dst as usize] = Some(src.clone()),
                    Val::V(u) if defc[*u as usize] == 1 && u != dst => alias[*dst as usize] = Some(src.clone()),
                    _ => {}
                }
            }
        }
    }
    let resolve = |v: &Val| -> Val {
        let mut cur = v.clone();
        for _ in 0..64 {
            match &cur { Val::V(x) => match &alias[*x as usize] { Some(a) => cur = a.clone(), None => break }, _ => break }
        }
        cur
    };
    for b in f.blocks.iter_mut() {
        for i in b.insts.iter_mut() { i.map_uses(|v| *v = resolve(v)); }
    }
}

/// Remove side-effect-free instructions whose results are never read.
pub fn sweep_dead(f: &mut MFunc) {
    loop {
        let (_, uses) = counts(f);
        let mut removed = false;
        for b in f.blocks.iter_mut() {
            let before = b.insts.len();
            b.insts.retain(|i| {
                let pure = matches!(i, MInst::Mov { .. } | MInst::Alu { .. } | MInst::Cmp { .. } | MInst::Select { .. }
                    | MInst::Ext { .. } | MInst::Lea { .. } | MInst::FAlu { .. } | MInst::FCmp { .. } | MInst::FCvt { .. });
                !(pure && i.def().map_or(false, |d| uses[d as usize] == 0))
            });
            removed |= b.insts.len() != before;
        }
        if !removed { break; }
    }
}

/// Run all machine-IR clean-ups.
pub fn optimize(f: &mut MFunc) {
    propagate_copies(f);
    fold_addresses(f);
    sweep_dead(f);
}
