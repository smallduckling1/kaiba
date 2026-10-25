// SPDX-License-Identifier: GPL-2.0-only WITH LicenseRef-Kaiba-Runtime-Exception
// Copyright (C) The Kaiba Authors
//
// This file is part of the Kaiba compiler toolchain. It is licensed under
// GPL-2.0-only with the Kaiba Runtime/Output Exception; see COPYING and
// LICENSE-EXCEPTION at the repository root.

//! Operands, instructions and terminators.

use crate::entity::{BlockId, FuncId, GlobalId, ValueId};
use crate::types::Type;

/// Compile-time constants. Floats are stored as raw bits so `Constant: Eq`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Constant {
    Int { bits: u32, value: u128 },
    F32(u32),
    F64(u64),
    /// Null pointer in address space 0.
    Null,
    Undef(Type),
    Poison(Type),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Operand {
    Value(ValueId),
    Const(Constant),
    /// Address of a global variable.
    Global(GlobalId),
    /// Address of a function.
    Func(FuncId),
}

impl Operand {
    pub fn int(bits: u32, value: u128) -> Self { Operand::Const(Constant::Int { bits, value }) }
    pub fn as_value(&self) -> Option<ValueId> {
        if let Operand::Value(v) = self { Some(*v) } else { None }
    }
}

impl From<ValueId> for Operand {
    fn from(v: ValueId) -> Self { Operand::Value(v) }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BinOp {
    Add, Sub, Mul, UDiv, SDiv, URem, SRem,
    And, Or, Xor, Shl, LShr, AShr,
    FAdd, FSub, FMul, FDiv, FRem,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IntPred { Eq, Ne, Ugt, Uge, Ult, Ule, Sgt, Sge, Slt, Sle }

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FloatPred { Oeq, Ogt, Oge, Olt, Ole, One, Ord, Ueq, Ugt, Uge, Ult, Ule, Une, Uno }

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CastOp {
    Trunc, ZExt, SExt, FPTrunc, FPExt,
    FPToUI, FPToSI, UIToFP, SIToFP,
    PtrToInt, IntToPtr, Bitcast,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Callee {
    Direct(FuncId),
    Indirect(Operand),
}

#[derive(Clone, Debug, PartialEq)]
pub enum InstKind {
    Binary { op: BinOp, lhs: Operand, rhs: Operand },
    ICmp { pred: IntPred, lhs: Operand, rhs: Operand },
    FCmp { pred: FloatPred, lhs: Operand, rhs: Operand },
    Cast { op: CastOp, val: Operand, to: Type },
    Select { cond: Operand, then_val: Operand, else_val: Operand },
    Load { ty: Type, ptr: Operand, align: u32, volatile: bool },
    Store { val: Operand, ptr: Operand, align: u32, volatile: bool },
    /// Stack allocation of `count` objects of `ty`; yields a pointer.
    Alloca { ty: Type, count: Operand, align: u32 },
    /// Address computation over `base_ty` (LLVM-style GEP).
    Gep { base_ty: Type, ptr: Operand, indices: Vec<Operand>, inbounds: bool },
    ExtractValue { agg: Operand, indices: Vec<u32> },
    /// Initialize the target ABI's `va_list` at `list` (variadic functions only).
    VaStart { list: Operand },
    /// Copy a `va_list` (the size is target-defined).
    VaCopy { dst: Operand, src: Operand },
    InsertValue { agg: Operand, val: Operand, indices: Vec<u32> },
    /// Must appear at the start of a block.
    Phi { ty: Type, incoming: Vec<(BlockId, Operand)> },
    Call { callee: Callee, args: Vec<Operand>, ret_ty: Type },
}

impl InstKind {
    /// Does this instruction have effects beyond producing its result?
    pub fn has_side_effects(&self) -> bool {
        match self {
            InstKind::Store { .. } | InstKind::Call { .. } | InstKind::VaStart { .. } | InstKind::VaCopy { .. } => true,
            InstKind::Load { volatile, .. } => *volatile,
            // Division may trap.
            InstKind::Binary { op, .. } => {
                matches!(op, BinOp::UDiv | BinOp::SDiv | BinOp::URem | BinOp::SRem)
            }
            _ => false,
        }
    }

    pub fn is_phi(&self) -> bool { matches!(self, InstKind::Phi { .. }) }

    pub fn for_each_operand(&self, mut f: impl FnMut(&Operand)) {
        use InstKind::*;
        match self {
            Binary { lhs, rhs, .. } | ICmp { lhs, rhs, .. } | FCmp { lhs, rhs, .. } => { f(lhs); f(rhs); }
            Cast { val, .. } => f(val),
            Select { cond, then_val, else_val } => { f(cond); f(then_val); f(else_val); }
            Load { ptr, .. } => f(ptr),
            Store { val, ptr, .. } => { f(val); f(ptr); }
            Alloca { count, .. } => f(count),
            Gep { ptr, indices, .. } => { f(ptr); indices.iter().for_each(f); }
            ExtractValue { agg, .. } => f(agg),
            VaStart { list } => f(list),
            VaCopy { dst, src } => { f(dst); f(src); }
            InsertValue { agg, val, .. } => { f(agg); f(val); }
            Phi { incoming, .. } => incoming.iter().for_each(|(_, o)| f(o)),
            Call { callee, args, .. } => {
                if let Callee::Indirect(o) = callee { f(o); }
                args.iter().for_each(f);
            }
        }
    }

    pub fn for_each_operand_mut(&mut self, mut f: impl FnMut(&mut Operand)) {
        use InstKind::*;
        match self {
            Binary { lhs, rhs, .. } | ICmp { lhs, rhs, .. } | FCmp { lhs, rhs, .. } => { f(lhs); f(rhs); }
            Cast { val, .. } => f(val),
            Select { cond, then_val, else_val } => { f(cond); f(then_val); f(else_val); }
            Load { ptr, .. } => f(ptr),
            Store { val, ptr, .. } => { f(val); f(ptr); }
            Alloca { count, .. } => f(count),
            Gep { ptr, indices, .. } => { f(ptr); indices.iter_mut().for_each(f); }
            ExtractValue { agg, .. } => f(agg),
            VaStart { list } => f(list),
            VaCopy { dst, src } => { f(dst); f(src); }
            InsertValue { agg, val, .. } => { f(agg); f(val); }
            Phi { incoming, .. } => incoming.iter_mut().for_each(|(_, o)| f(o)),
            Call { callee, args, .. } => {
                if let Callee::Indirect(o) = callee { f(o); }
                args.iter_mut().for_each(f);
            }
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum Terminator {
    Ret(Option<Operand>),
    Br(BlockId),
    CondBr { cond: Operand, then_bb: BlockId, else_bb: BlockId },
    Switch { val: Operand, default: BlockId, cases: Vec<(u128, BlockId)> },
    Unreachable,
}

impl Terminator {
    pub fn successors(&self) -> Vec<BlockId> {
        match self {
            Terminator::Ret(_) | Terminator::Unreachable => vec![],
            Terminator::Br(b) => vec![*b],
            Terminator::CondBr { then_bb, else_bb, .. } => vec![*then_bb, *else_bb],
            Terminator::Switch { default, cases, .. } => {
                let mut v = vec![*default];
                v.extend(cases.iter().map(|(_, b)| *b));
                v
            }
        }
    }

    /// Rewrite every successor edge target.
    pub fn map_successors(&mut self, mut f: impl FnMut(BlockId) -> BlockId) {
        match self {
            Terminator::Ret(_) | Terminator::Unreachable => {}
            Terminator::Br(b) => *b = f(*b),
            Terminator::CondBr { then_bb, else_bb, .. } => { *then_bb = f(*then_bb); *else_bb = f(*else_bb); }
            Terminator::Switch { default, cases, .. } => {
                *default = f(*default);
                for (_, b) in cases.iter_mut() { *b = f(*b); }
            }
        }
    }

    pub fn for_each_operand(&self, mut f: impl FnMut(&Operand)) {
        match self {
            Terminator::Ret(Some(o)) => f(o),
            Terminator::CondBr { cond, .. } => f(cond),
            Terminator::Switch { val, .. } => f(val),
            _ => {}
        }
    }

    pub fn for_each_operand_mut(&mut self, mut f: impl FnMut(&mut Operand)) {
        match self {
            Terminator::Ret(Some(o)) => f(o),
            Terminator::CondBr { cond, .. } => f(cond),
            Terminator::Switch { val, .. } => f(val),
            _ => {}
        }
    }
}
