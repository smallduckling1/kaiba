// SPDX-License-Identifier: GPL-2.0-only WITH LicenseRef-Kaiba-Runtime-Exception
// Copyright (C) The Kaiba Authors
//
// This file is part of the Kaiba compiler toolchain. It is licensed under
// GPL-2.0-only with the Kaiba Runtime/Output Exception; see COPYING and
// LICENSE-EXCEPTION at the repository root.
//! Machine IR: three-address instructions over unlimited 64-bit virtual
//! registers. Integer values of width N are kept *canonical*: zero-extended
//! to 64 bits. Operations re-canonicalize their results.

use llcr::{FloatPred, IntPred};

pub type VReg = u32;

#[derive(Clone, Debug, PartialEq)]
pub enum Val {
    V(VReg),
    Imm(u64),
    /// Address of a symbol; `external` symbols are reached through the GOT.
    Sym { name: String, external: bool },
}

/// A memory address: `base + index * scale + disp`.
#[derive(Clone, Debug, PartialEq)]
pub struct Addr {
    pub base: Val,
    /// Optional scaled index; `scale` is 1, 2, 4 or 8.
    pub index: Option<(Val, u8)>,
    pub disp: i32,
}

impl Addr {
    pub fn simple(base: Val) -> Addr { Addr { base, index: None, disp: 0 } }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Alu { Add, Sub, Mul, And, Or, Xor, Shl, LShr, AShr, UDiv, SDiv, URem, SRem }

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExtKind { Zext, Sext, Trunc }

/// How a value travels across a call boundary (System V classification).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ArgClass {
    Int,
    F32,
    F64,
    /// The hidden pointer to a caller-allocated return slot.
    Sret,
    /// A by-value aggregate copied onto the stack (the value is a pointer to the source).
    Mem { size: u64, align: u64 },
}

impl ArgClass {
    pub fn is_fp(self) -> bool { matches!(self, ArgClass::F32 | ArgClass::F64) }
    pub fn is_double(self) -> bool { self == ArgClass::F64 }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ArgLoc {
    /// n-th integer argument register of the target ABI.
    IntReg(usize),
    /// n-th floating-point argument register of the target ABI.
    FpReg(usize),
    /// k-th 8-byte stack argument slot (the first slot, for multi-slot by-value aggregates).
    Stack(usize),
    /// The dedicated struct-return register (AArch64 x8).
    Sret,
}

/// How many argument registers a target's calling convention provides.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Abi {
    pub int_regs: usize,
    pub fp_regs: usize,
    /// Registers available for returning several values (integer / floating point).
    pub int_ret_regs: usize,
    pub fp_ret_regs: usize,
    /// The struct-return pointer is an ordinary first integer argument (x86-64);
    /// otherwise it has its own register (AArch64).
    pub sret_in_arg0: bool,
    /// By-value memory arguments (`byval`) are part of this ABI.
    pub byval: bool,
    /// Size in bytes of the target's `va_list` object.
    pub va_list_size: u64,
}

/// Eight-byte stack slots a by-value aggregate occupies.
fn mem_slots(size: u64) -> usize { ((size + 7) / 8) as usize }

/// Assign each argument (in order) to a register or stack slot.
pub fn classify(classes: &[ArgClass], abi: Abi) -> Vec<ArgLoc> {
    let (mut ints, mut fps, mut stack) = (0, 0, 0);
    classes.iter().map(|c| match *c {
        ArgClass::Sret if !abi.sret_in_arg0 => ArgLoc::Sret,
        ArgClass::Sret | ArgClass::Int => {
            if ints < abi.int_regs { ints += 1; ArgLoc::IntReg(ints - 1) } else { stack += 1; ArgLoc::Stack(stack - 1) }
        }
        ArgClass::F32 | ArgClass::F64 => {
            if fps < abi.fp_regs { fps += 1; ArgLoc::FpReg(fps - 1) } else { stack += 1; ArgLoc::Stack(stack - 1) }
        }
        ArgClass::Mem { size, align } => {
            let a = ((align.max(8)) / 8) as usize;
            stack = (stack + a - 1) / a * a;
            let k = stack;
            stack += mem_slots(size);
            ArgLoc::Stack(k)
        }
    }).collect()
}

/// Number of 8-byte stack slots the stack-passed arguments occupy.
pub fn stack_slots(classes: &[ArgClass], locs: &[ArgLoc]) -> usize {
    classes.iter().zip(locs).filter_map(|(c, l)| match (c, l) {
        (ArgClass::Mem { size, .. }, ArgLoc::Stack(k)) => Some(k + mem_slots(*size)),
        (_, ArgLoc::Stack(k)) => Some(k + 1),
        _ => None,
    }).max().unwrap_or(0)
}

/// Assign each returned value to a return register: the n-th integer value goes to
/// the n-th integer return register, likewise for floating point.
pub fn classify_returns(classes: &[ArgClass], abi: Abi) -> Result<Vec<ArgLoc>, String> {
    let (mut ints, mut fps) = (0, 0);
    classes.iter().map(|c| match c {
        ArgClass::F32 | ArgClass::F64 => {
            if fps >= abi.fp_ret_regs { return Err("too many floating-point values to return in registers".to_string()); }
            fps += 1;
            Ok(ArgLoc::FpReg(fps - 1))
        }
        _ => {
            if ints >= abi.int_ret_regs { return Err("too many integer values to return in registers".to_string()); }
            ints += 1;
            Ok(ArgLoc::IntReg(ints - 1))
        }
    }).collect()
}

#[derive(Clone, Debug, PartialEq)]
pub struct Arg { pub val: Val, pub class: ArgClass }

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FOp { Add, Sub, Mul, Div }

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FCvt {
    /// Integer of `from` bits -> float.
    SIToFP { from: u32, dbl: bool },
    UIToFP { from: u32, dbl: bool },
    /// Float (`dbl` = source is f64) -> integer of `to` bits.
    FPToSI { to: u32, dbl: bool },
    FPToUI { to: u32, dbl: bool },
    FPExt,
    FPTrunc,
}

#[derive(Clone, Debug)]
pub enum CallTarget {
    Direct { name: String, external: bool },
    Indirect(Val),
}

#[derive(Clone, Debug)]
pub enum MInst {
    Param { dst: VReg, class: ArgClass, loc: ArgLoc, bits: u32 },
    Mov { dst: VReg, src: Val },
    Alu { op: Alu, bits: u32, dst: VReg, lhs: Val, rhs: Val },
    Cmp { pred: IntPred, bits: u32, dst: VReg, lhs: Val, rhs: Val },
    Select { dst: VReg, cond: Val, then_val: Val, else_val: Val },
    Ext { kind: ExtKind, from: u32, to: u32, dst: VReg, src: Val },
    Load { dst: VReg, bits: u32, addr: Addr },
    Store { bits: u32, val: Val, addr: Addr },
    FAlu { op: FOp, dbl: bool, dst: VReg, lhs: Val, rhs: Val },
    FCmp { pred: FloatPred, dbl: bool, dst: VReg, lhs: Val, rhs: Val },
    FCvt { kind: FCvt, dst: VReg, src: Val },
    /// Address of a frame slot.
    Lea { dst: VReg, slot: usize },
    /// `dsts` are the returned values (empty for void; several for aggregate returns).
    Call { dsts: Vec<(VReg, u32, ArgClass)>, target: CallTarget, args: Vec<Arg> },
    /// Initialize the `va_list` at `addr` (see [`VaInfo`]).
    VaStart { addr: Val },
    // Terminators (indices are MIR block indices).
    Jmp(usize),
    Br { cond: Val, then_bb: usize, else_bb: usize },
    Switch { val: Val, bits: u32, cases: Vec<(u64, usize)>, default: usize },
    /// Returned values (empty for void).
    Ret(Vec<Arg>),
    Trap,
}

impl MInst {
    pub fn is_terminator(&self) -> bool {
        matches!(self, MInst::Jmp(_) | MInst::Br { .. } | MInst::Switch { .. } | MInst::Ret(_) | MInst::Trap)
    }

    pub fn def(&self) -> Option<VReg> {
        match self {
            MInst::Param { dst, .. } | MInst::Mov { dst, .. } | MInst::Alu { dst, .. } | MInst::Cmp { dst, .. }
            | MInst::Select { dst, .. } | MInst::Ext { dst, .. } | MInst::Load { dst, .. } | MInst::Lea { dst, .. }
            | MInst::FAlu { dst, .. } | MInst::FCmp { dst, .. } | MInst::FCvt { dst, .. } => Some(*dst),
            _ => None,
        }
    }

    /// All vregs this instruction defines (calls may define several).
    pub fn for_each_def(&self, mut f: impl FnMut(VReg)) {
        if let MInst::Call { dsts, .. } = self { dsts.iter().for_each(|d| f(d.0)); }
        else if let Some(d) = self.def() { f(d); }
    }

    pub fn for_each_use(&self, mut f: impl FnMut(VReg)) {
        let mut v = |x: &Val| if let Val::V(r) = x { f(*r) };
        match self {
            MInst::Param { .. } | MInst::Lea { .. } | MInst::Jmp(_) | MInst::Trap => {}
            MInst::Mov { src, .. } | MInst::Ext { src, .. } | MInst::FCvt { src, .. } => v(src),
            MInst::Alu { lhs, rhs, .. } | MInst::Cmp { lhs, rhs, .. } | MInst::FAlu { lhs, rhs, .. } | MInst::FCmp { lhs, rhs, .. } => { v(lhs); v(rhs); }
            MInst::Select { cond, then_val, else_val, .. } => { v(cond); v(then_val); v(else_val); }
            MInst::VaStart { addr } => v(addr),
            MInst::Load { addr, .. } => { v(&addr.base); if let Some((i, _)) = &addr.index { v(i); } }
            MInst::Store { val, addr, .. } => { v(val); v(&addr.base); if let Some((i, _)) = &addr.index { v(i); } }
            MInst::Call { target, args, .. } => {
                if let CallTarget::Indirect(t) = target { v(t); }
                args.iter().for_each(|a| v(&a.val));
            }
            MInst::Br { cond, .. } => v(cond),
            MInst::Switch { val, .. } => v(val),
            MInst::Ret(xs) => xs.iter().for_each(|x| v(&x.val)),
        }
    }

    /// Rewrite every value this instruction reads.
    pub fn map_uses(&mut self, mut f: impl FnMut(&mut Val)) {
        match self {
            MInst::Param { .. } | MInst::Lea { .. } | MInst::Jmp(_) | MInst::Trap => {}
            MInst::Mov { src, .. } | MInst::Ext { src, .. } | MInst::FCvt { src, .. } => f(src),
            MInst::Alu { lhs, rhs, .. } | MInst::Cmp { lhs, rhs, .. } | MInst::FAlu { lhs, rhs, .. } | MInst::FCmp { lhs, rhs, .. } => { f(lhs); f(rhs); }
            MInst::Select { cond, then_val, else_val, .. } => { f(cond); f(then_val); f(else_val); }
            MInst::VaStart { addr } => f(addr),
            MInst::Load { addr, .. } => { f(&mut addr.base); if let Some((i, _)) = &mut addr.index { f(i); } }
            MInst::Store { val, addr, .. } => { f(val); f(&mut addr.base); if let Some((i, _)) = &mut addr.index { f(i); } }
            MInst::Call { target, args, .. } => {
                if let CallTarget::Indirect(t) = target { f(t); }
                args.iter_mut().for_each(|a| f(&mut a.val));
            }
            MInst::Br { cond, .. } => f(cond),
            MInst::Switch { val, .. } => f(val),
            MInst::Ret(xs) => xs.iter_mut().for_each(|x| f(&mut x.val)),
        }
    }

    pub fn successors(&self) -> Vec<usize> {
        match self {
            MInst::Jmp(b) => vec![*b],
            MInst::Br { then_bb, else_bb, .. } => vec![*then_bb, *else_bb],
            MInst::Switch { cases, default, .. } => {
                let mut v = vec![*default];
                v.extend(cases.iter().map(|(_, b)| *b));
                v
            }
            _ => vec![],
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct MBlock { pub insts: Vec<MInst> }

/// What a variadic function needs to set up `va_list`s: how many of the argument
/// registers its *named* parameters use, and where the register save area is.
#[derive(Clone, Copy, Debug)]
pub struct VaInfo {
    pub named_gp: usize,
    pub named_fp: usize,
    /// 8-byte stack slots used by named stack parameters.
    pub named_stack: usize,
    /// Index into `MFunc::slots` of the register save area.
    pub save_slot: usize,
}

#[derive(Clone, Debug)]
pub struct MFunc {
    pub name: String,
    pub global: bool,
    pub blocks: Vec<MBlock>,
    /// Frame slots for allocas: (size, align).
    pub slots: Vec<(u64, u64)>,
    pub num_vregs: u32,
    /// Present when the function calls `va_start`.
    pub va: Option<VaInfo>,
}
