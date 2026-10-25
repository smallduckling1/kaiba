// SPDX-License-Identifier: GPL-2.0-only WITH LicenseRef-Kaiba-Runtime-Exception
// Copyright (C) The Kaiba Authors
//
// This file is part of the Kaiba compiler toolchain. It is licensed under
// GPL-2.0-only with the Kaiba Runtime/Output Exception; see COPYING and
// LICENSE-EXCEPTION at the repository root.
//! Constant folding of integer arithmetic, comparisons, casts and selects.

use crate::function::Function;
use crate::inst::{BinOp, CastOp, Constant, InstKind, IntPred, Operand};
use crate::pass::{Changed, FunctionPass};
use crate::types::Type;

fn mask(bits: u32) -> u128 { if bits >= 128 { u128::MAX } else { (1u128 << bits) - 1 } }

fn sx(v: u128, bits: u32) -> i128 {
    let v = v & mask(bits);
    if bits < 128 && (v >> (bits - 1)) & 1 == 1 { (v | !mask(bits)) as i128 } else { v as i128 }
}

fn smin(bits: u32) -> i128 { if bits >= 128 { i128::MIN } else { -(1i128 << (bits - 1)) } }

fn fold_bin(op: BinOp, bits: u32, a: u128, b: u128) -> Option<u128> {
    let m = mask(bits);
    let (a, b) = (a & m, b & m);
    let (x, y) = (sx(a, bits), sx(b, bits));
    let overflow = y == -1 && x == smin(bits);
    Some(match op {
        BinOp::Add => a.wrapping_add(b) & m,
        BinOp::Sub => a.wrapping_sub(b) & m,
        BinOp::Mul => a.wrapping_mul(b) & m,
        BinOp::And => a & b,
        BinOp::Or => a | b,
        BinOp::Xor => a ^ b,
        BinOp::UDiv => { if b == 0 { return None; } a / b }
        BinOp::URem => { if b == 0 { return None; } a % b }
        BinOp::SDiv => { if y == 0 || overflow { return None; } (x / y) as u128 & m }
        BinOp::SRem => { if y == 0 || overflow { return None; } (x % y) as u128 & m }
        BinOp::Shl => { if b >= bits as u128 { return None; } (a << b) & m }
        BinOp::LShr => { if b >= bits as u128 { return None; } a >> b }
        BinOp::AShr => { if b >= bits as u128 { return None; } (x >> b) as u128 & m }
        _ => return None,
    })
}

fn fold_icmp(p: IntPred, bits: u32, a: u128, b: u128) -> bool {
    let m = mask(bits);
    let (a, b) = (a & m, b & m);
    let (x, y) = (sx(a, bits), sx(b, bits));
    match p {
        IntPred::Eq => a == b, IntPred::Ne => a != b,
        IntPred::Ugt => a > b, IntPred::Uge => a >= b, IntPred::Ult => a < b, IntPred::Ule => a <= b,
        IntPred::Sgt => x > y, IntPred::Sge => x >= y, IntPred::Slt => x < y, IntPred::Sle => x <= y,
    }
}

fn int_const(o: &Operand) -> Option<(u32, u128)> {
    if let Operand::Const(Constant::Int { bits, value }) = o { Some((*bits, *value)) } else { None }
}

fn fold(kind: &InstKind) -> Option<Operand> {
    match kind {
        InstKind::Binary { op, lhs, rhs } => {
            let ((bits, a), (_, b)) = (int_const(lhs)?, int_const(rhs)?);
            fold_bin(*op, bits, a, b).map(|v| Operand::int(bits, v))
        }
        InstKind::ICmp { pred, lhs, rhs } => {
            let ((bits, a), (_, b)) = (int_const(lhs)?, int_const(rhs)?);
            Some(Operand::int(1, fold_icmp(*pred, bits, a, b) as u128))
        }
        InstKind::Cast { op, val, to: Type::Int(to) } => {
            let (from, v) = int_const(val)?;
            let r = match op {
                CastOp::Trunc => v & mask(*to),
                CastOp::ZExt => v & mask(from),
                CastOp::SExt => (sx(v, from) as u128) & mask(*to),
                _ => return None,
            };
            Some(Operand::int(*to, r))
        }
        InstKind::Select { cond, then_val, else_val } => {
            let (_, c) = int_const(cond)?;
            Some(if c & 1 == 1 { *then_val } else { *else_val })
        }
        _ => None,
    }
}

pub struct ConstantFolding;

impl FunctionPass for ConstantFolding {
    fn name(&self) -> &'static str { "constfold" }

    fn run(&mut self, f: &mut Function) -> Changed {
        let mut changed = false;
        loop {
            let mut round = false;
            let snapshot: Vec<_> = f.iter_insts().map(|(_, i)| i).collect();
            for i in snapshot {
                let Some(result) = f.insts[i].result else { continue };
                if let Some(new) = fold(&f.insts[i].kind) {
                    f.replace_all_uses(result, new);
                    f.remove_inst(i);
                    round = true;
                }
            }
            if !round { break; }
            changed = true;
        }
        Changed::from_bool(changed)
    }
}
