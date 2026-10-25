// SPDX-License-Identifier: GPL-2.0-only WITH LicenseRef-Kaiba-Runtime-Exception
//! Tiny reference interpreter (integers + stack memory) used to check that
//! optimization passes preserve behaviour.
#![allow(dead_code)]
use llcr::*;
use std::collections::HashMap;

fn mask(bits: u32) -> u128 { if bits >= 128 { u128::MAX } else { (1u128 << bits) - 1 } }
fn sx(v: u128, bits: u32) -> i128 {
    if (v >> (bits - 1)) & 1 == 1 { (v | !mask(bits)) as i128 } else { v as i128 }
}

fn ev(vals: &HashMap<ValueId, u128>, o: &Operand) -> u128 {
    match o {
        Operand::Value(v) => *vals.get(v).unwrap_or_else(|| panic!("read of undefined {v}")),
        Operand::Const(Constant::Int { value, .. }) => *value,
        Operand::Const(Constant::Undef(_)) => 0,
        o => panic!("interpreter: unsupported operand {o:?}"),
    }
}

pub fn run(f: &Function, args: &[u128]) -> u128 {
    let mut vals: HashMap<ValueId, u128> = HashMap::new();
    for (p, a) in f.params.iter().zip(args) { vals.insert(*p, *a); }
    let mut mem: HashMap<u128, u128> = HashMap::new();
    let mut next_addr = 0x1000u128;
    let mut b = f.entry().unwrap();
    let mut prev: Option<BlockId> = None;
    for _ in 0..1_000_000 {
        let insts = &f.blocks[b].insts;
        let mut phis = Vec::new();
        for &i in insts {
            if let InstKind::Phi { incoming, .. } = &f.insts[i].kind {
                let (_, o) = incoming.iter().find(|(pb, _)| Some(*pb) == prev).expect("phi without incoming for predecessor");
                phis.push((f.insts[i].result.unwrap(), ev(&vals, o)));
            }
        }
        vals.extend(phis);
        for &i in insts {
            let d = &f.insts[i];
            let out = match &d.kind {
                InstKind::Phi { .. } => None,
                InstKind::Binary { op, lhs, rhs } => {
                    let Type::Int(bits) = f.values[d.result.unwrap()].ty else { panic!() };
                    let (a, c) = (ev(&vals, lhs), ev(&vals, rhs));
                    Some(match op {
                        BinOp::Add => a.wrapping_add(c), BinOp::Sub => a.wrapping_sub(c), BinOp::Mul => a.wrapping_mul(c),
                        BinOp::And => a & c, BinOp::Or => a | c, BinOp::Xor => a ^ c,
                        BinOp::UDiv => a / c, BinOp::URem => a % c,
                        BinOp::SDiv => (sx(a, bits) / sx(c, bits)) as u128, BinOp::SRem => (sx(a, bits) % sx(c, bits)) as u128,
                        BinOp::Shl => a << c, BinOp::LShr => a >> c, BinOp::AShr => (sx(a, bits) >> c) as u128,
                        o => panic!("interpreter: unsupported {o:?}"),
                    } & mask(bits))
                }
                InstKind::ICmp { pred, lhs, rhs } => {
                    let Type::Int(bits) = f.operand_type(lhs) else { panic!() };
                    let (a, c) = (ev(&vals, lhs) & mask(bits), ev(&vals, rhs) & mask(bits));
                    let (x, y) = (sx(a, bits), sx(c, bits));
                    Some(match pred {
                        IntPred::Eq => a == c, IntPred::Ne => a != c,
                        IntPred::Slt => x < y, IntPred::Sgt => x > y, IntPred::Sle => x <= y, IntPred::Sge => x >= y,
                        IntPred::Ult => a < c, IntPred::Ugt => a > c, IntPred::Ule => a <= c, IntPred::Uge => a >= c,
                    } as u128)
                }
                InstKind::Select { cond, then_val, else_val } => {
                    Some(if ev(&vals, cond) & 1 == 1 { ev(&vals, then_val) } else { ev(&vals, else_val) })
                }
                InstKind::Alloca { .. } => { next_addr += 16; Some(next_addr) }
                InstKind::Load { ptr, .. } => Some(*mem.get(&ev(&vals, ptr)).unwrap_or(&0)),
                InstKind::Store { val, ptr, .. } => { mem.insert(ev(&vals, ptr), ev(&vals, val)); None }
                k => panic!("interpreter: unsupported {k:?}"),
            };
            if let (Some(r), Some(v)) = (d.result, out) { vals.insert(r, v); }
        }
        match f.blocks[b].terminator.as_ref().unwrap() {
            Terminator::Ret(Some(o)) => return ev(&vals, o),
            Terminator::Br(t) => { prev = Some(b); b = *t; }
            Terminator::CondBr { cond, then_bb, else_bb } => {
                prev = Some(b);
                b = if ev(&vals, cond) & 1 == 1 { *then_bb } else { *else_bb };
            }
            t => panic!("interpreter: unsupported terminator {t:?}"),
        }
    }
    panic!("interpreter: step limit exceeded");
}
