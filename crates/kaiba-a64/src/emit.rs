// SPDX-License-Identifier: GPL-2.0-only WITH LicenseRef-Kaiba-Runtime-Exception
// Copyright (C) The Kaiba Authors
//
// This file is part of the Kaiba compiler toolchain. It is licensed under
// GPL-2.0-only with the Kaiba Runtime/Output Exception; see COPYING and
// LICENSE-EXCEPTION at the repository root.
//! AArch64 assembly emission (GNU as) from machine IR + allocation.
//!
//! Conventions: integers of N bits are kept zero-extended in 64-bit registers;
//! floats live in general registers as bit patterns and pass through d0/d1 (or
//! s0/s1) for each operation. x9-x11, x16 are scratch; x0-x7 carry arguments.

use crate::regs::*;
use kaiba_mir::mir::*;
use kaiba_mir::regalloc::{Allocation, Loc};
use llcr::{FloatPred, IntPred};
use std::fmt::Write;

fn align_up(x: u64, a: u64) -> u64 { (x + a - 1) / a * a }

fn cc_of(p: IntPred) -> &'static str {
    match p {
        IntPred::Eq => "eq", IntPred::Ne => "ne", IntPred::Ugt => "hi", IntPred::Uge => "hs", IntPred::Ult => "lo",
        IntPred::Ule => "ls", IntPred::Sgt => "gt", IntPred::Sge => "ge", IntPred::Slt => "lt", IntPred::Sle => "le",
    }
}

fn cc_inverse(cc: &str) -> &'static str {
    match cc {
        "eq" => "ne", "ne" => "eq", "hi" => "ls", "ls" => "hi", "hs" => "lo", "lo" => "hs",
        "gt" => "le", "le" => "gt", "ge" => "lt", _ => "ge",
    }
}

struct Emitter<'a> {
    out: String,
    f: &'a MFunc,
    alloc: &'a Allocation<R>,
    spill_off: Vec<u64>,
    slot_off: Vec<u64>,
    saved: Vec<R>,
    uses: Vec<u32>,
}

impl<'a> Emitter<'a> {
    fn ln(&mut self, s: impl AsRef<str>) { let _ = writeln!(self.out, "    {}", s.as_ref()); }
    fn label(&self, b: usize) -> String { format!(".L{}_{}", self.f.name, b) }

    // ---------- constants, memory, symbols ----------

    fn mov_imm(&mut self, d: R, v: u64) {
        if v == 0 { self.ln(format!("mov {}, xzr", d.x())); return; }
        if v <= 0xffff { self.ln(format!("mov {}, #{v}", d.x())); return; }
        if !v <= 0xffff { self.ln(format!("movn {}, #{}", d.x(), !v & 0xffff)); return; }
        let mut first = true;
        for k in 0..4 {
            let chunk = (v >> (16 * k)) & 0xffff;
            if chunk == 0 { continue; }
            if first { self.ln(format!("movz {}, #{chunk}, lsl #{}", d.x(), 16 * k)); first = false; }
            else { self.ln(format!("movk {}, #{chunk}, lsl #{}", d.x(), 16 * k)); }
        }
    }

    /// Load/store `reg` at `[x29 - off]`.
    fn fp_mem(&mut self, mnem: &str, reg: &str, off: u64) {
        if off <= 255 {
            self.ln(format!("{mnem} {reg}, [x29, #-{off}]"));
        } else {
            self.fp_addr(IP0, off);
            self.ln(format!("{mnem} {reg}, [{}]", IP0.x()));
        }
    }

    /// d = x29 - off
    fn fp_addr(&mut self, d: R, off: u64) {
        if off <= 4095 { self.ln(format!("sub {}, x29, #{off}", d.x())); }
        else { self.mov_imm(d, off); self.ln(format!("sub {0}, x29, {0}", d.x())); }
    }

    fn load_sym(&mut self, d: R, name: &str, external: bool) {
        if external {
            self.ln(format!("adrp {}, :got:\"{name}\"", d.x()));
            self.ln(format!("ldr {0}, [{0}, :got_lo12:\"{name}\"]", d.x()));
        } else {
            self.ln(format!("adrp {}, \"{name}\"", d.x()));
            self.ln(format!("add {0}, {0}, :lo12:\"{name}\"", d.x()));
        }
    }

    fn load_into(&mut self, d: R, v: &Val) {
        match v {
            Val::V(x) => match self.alloc.locs[*x as usize] {
                Some(Loc::Reg(r)) => { if r != d { self.ln(format!("mov {}, {}", d.x(), r.x())); } }
                Some(Loc::Spill(i)) => { let off = self.spill_off[i]; self.fp_mem("ldr", &d.x(), off); }
                None => self.ln(format!("mov {}, xzr", d.x())),
            },
            Val::Imm(i) => self.mov_imm(d, *i),
            Val::Sym { name, external } => self.load_sym(d, name, *external),
        }
    }

    /// A register holding `v`: its own register, or `scratch` after loading it there.
    fn use_reg(&mut self, v: &Val, scratch: R) -> R {
        if let Val::V(x) = v {
            if let Some(Loc::Reg(r)) = self.alloc.locs[*x as usize] { return r; }
        }
        self.load_into(scratch, v);
        scratch
    }

    fn def_reg(&self, dst: VReg) -> R {
        match self.alloc.locs[dst as usize] { Some(Loc::Reg(r)) => r, _ => S0 }
    }

    /// Commit `r` (holding the value of `dst`) to its home.
    fn finish(&mut self, dst: VReg, r: R) {
        match self.alloc.locs[dst as usize] {
            Some(Loc::Reg(d)) => { if d != r { self.ln(format!("mov {}, {}", d.x(), r.x())); } }
            Some(Loc::Spill(i)) => { let off = self.spill_off[i]; self.fp_mem("str", &r.x(), off); }
            None => {}
        }
    }

    // ---------- integer helpers ----------

    /// Re-canonicalize `r` (zero-extend from `bits`).
    fn mask(&mut self, r: R, bits: u32) {
        match bits {
            64 => {}
            32 => self.ln(format!("mov {0}, {0}", r.w())),
            16 => self.ln(format!("uxth {0}, {0}", r.w())),
            8 => self.ln(format!("uxtb {0}, {0}", r.w())),
            _ => self.ln(format!("and {0}, {0}, #1", r.w())),
        }
    }

    /// d = sign-extension (to 64 bits) of the low `bits` of s.
    fn sext(&mut self, d: R, s: R, bits: u32) {
        match bits {
            64 => { if d != s { self.ln(format!("mov {}, {}", d.x(), s.x())); } }
            32 => self.ln(format!("sxtw {}, {}", d.x(), s.w())),
            16 => self.ln(format!("sxth {}, {}", d.x(), s.w())),
            8 => self.ln(format!("sxtb {}, {}", d.x(), s.w())),
            _ => self.ln(format!("neg {}, {}", d.x(), s.x())),
        }
    }

    fn to_fp(&mut self, n: usize, r: R, dbl: bool) {
        if dbl { self.ln(format!("fmov d{n}, {}", r.x())); } else { self.ln(format!("fmov s{n}, {}", r.w())); }
    }

    fn from_fp(&mut self, r: R, n: usize, dbl: bool) {
        if dbl { self.ln(format!("fmov {}, d{n}", r.x())); } else { self.ln(format!("fmov {}, s{n}", r.w())); }
    }

    /// Emit a compare and return the condition meaning "pred holds".
    fn emit_cmp(&mut self, pred: IntPred, bits: u32, lhs: &Val, rhs: &Val) -> &'static str {
        let signed = matches!(pred, IntPred::Sgt | IntPred::Sge | IntPred::Slt | IntPred::Sle);
        let narrow_signed = signed && bits < 32;
        let w64 = bits == 64 || narrow_signed;
        let mut a = self.use_reg(lhs, S0);
        if narrow_signed { self.sext(S0, a, bits); a = S0; }
        if let (Val::Imm(i), false) = (rhs, narrow_signed) {
            if *i <= 4095 {
                self.ln(format!("cmp {}, #{i}", a.name(if w64 { 64 } else { 32 })));
                return cc_of(pred);
            }
        }
        let mut b = self.use_reg(rhs, S1);
        if narrow_signed { self.sext(S1, b, bits); b = S1; }
        let w = if w64 { 64 } else { 32 };
        self.ln(format!("cmp {}, {}", a.name(w), b.name(w)));
        cc_of(pred)
    }

    /// A memory operand `[...]` for an access of `size` bytes. Scratch: x10 and x11.
    fn addr_op(&mut self, a: &Addr, size: i64) -> String {
        let mut base = self.use_reg(&a.base, S1);
        if let Some((iv, sc)) = &a.index {
            let i = self.use_reg(iv, S2);
            self.ln(format!("add {}, {}, {}, lsl #{}", S1.x(), base.x(), i.x(), sc.trailing_zeros()));
            base = S1;
        }
        let d = a.disp as i64;
        if d == 0 { return format!("[{}]", base.x()); }
        if (-256..=255).contains(&d) || (d > 0 && d % size == 0 && d / size <= 4095) {
            return format!("[{}, #{d}]", base.x());
        }
        self.mov_imm(S2, d as u64);
        self.ln(format!("add {}, {}, {}", S1.x(), base.x(), S2.x()));
        format!("[{}]", S1.x())
    }

    fn epilogue(&mut self) {
        for (i, r) in self.saved.clone().iter().enumerate() {
            self.ln(format!("ldr {}, [x29, #-{}]", r.x(), 8 * (i + 1)));
        }
        self.ln("mov sp, x29");
        self.ln("ldp x29, x30, [sp], #16");
        self.ln("ret");
    }

    // ---------- instructions ----------

    fn inst(&mut self, inst: &MInst, next: Option<usize>) {
        match inst {
            MInst::Param { dst, class, loc, bits } => {
                let r = match loc {
                    ArgLoc::IntReg(i) => { let r = R(*i as u8); self.mask(r, *bits); r }
                    ArgLoc::FpReg(i) => { self.from_fp(S0, *i, class.is_double()); S0 }
                    ArgLoc::Stack(k) => {
                        self.ln(format!("ldr x9, [x29, #{}]", 16 + 8 * k));
                        match class { ArgClass::F32 => self.mask(S0, 32), ArgClass::F64 => {} _ => self.mask(S0, *bits) }
                        S0
                    }
                    // The struct-return pointer arrives in x8.
                    ArgLoc::Sret => R(8),
                };
                self.finish(*dst, r);
            }
            MInst::Mov { dst, src } => {
                let d = self.def_reg(*dst);
                self.load_into(d, src);
                self.finish(*dst, d);
            }
            MInst::Alu { op, bits, dst, lhs, rhs } => self.alu(*op, *bits, *dst, lhs, rhs),
            MInst::Cmp { pred, bits, dst, lhs, rhs } => {
                let cc = self.emit_cmp(*pred, *bits, lhs, rhs);
                let d = self.def_reg(*dst);
                self.ln(format!("cset {}, {cc}", d.w()));
                self.finish(*dst, d);
            }
            MInst::Select { dst, cond, then_val, else_val } => {
                let a = self.use_reg(then_val, S0);
                let b = self.use_reg(else_val, S1);
                let c = self.use_reg(cond, S2);
                self.ln(format!("cmp {}, #0", c.x()));
                let d = self.def_reg(*dst);
                self.ln(format!("csel {}, {}, {}, ne", d.x(), a.x(), b.x()));
                self.finish(*dst, d);
            }
            MInst::Ext { kind, from, to, dst, src } => {
                let d = self.def_reg(*dst);
                let a = self.use_reg(src, S0);
                match kind {
                    ExtKind::Zext => { if d != a { self.ln(format!("mov {}, {}", d.x(), a.x())); } }
                    ExtKind::Sext => { self.sext(d, a, *from); self.mask(d, *to); }
                    ExtKind::Trunc => { if d != a { self.ln(format!("mov {}, {}", d.x(), a.x())); } self.mask(d, *to); }
                }
                self.finish(*dst, d);
            }
            MInst::Load { dst, bits, addr } => {
                let (size, mn) = match bits { 64 => (8, "ldr"), 32 => (4, "ldr"), 16 => (2, "ldrh"), _ => (1, "ldrb") };
                let a = self.addr_op(addr, size);
                let d = self.def_reg(*dst);
                self.ln(format!("{mn} {}, {a}", if *bits == 64 { d.x() } else { d.w() }));
                if *bits == 1 { self.mask(d, 1); }
                self.finish(*dst, d);
            }
            MInst::Store { bits, val, addr } => {
                let v = self.use_reg(val, S0);
                let (size, mn) = match bits { 64 => (8, "str"), 32 => (4, "str"), 16 => (2, "strh"), _ => (1, "strb") };
                let a = self.addr_op(addr, size);
                self.ln(format!("{mn} {}, {a}", if *bits == 64 { v.x() } else { v.w() }));
            }
            MInst::Lea { dst, slot } => {
                let d = self.def_reg(*dst);
                let off = self.slot_off[*slot];
                self.fp_addr(d, off);
                self.finish(*dst, d);
            }
            MInst::Call { dsts, target, args } => self.call(dsts, target, args),
            MInst::FAlu { op, dbl, dst, lhs, rhs } => {
                let a = self.use_reg(lhs, S0);
                self.to_fp(0, a, *dbl);
                let b = self.use_reg(rhs, S1);
                self.to_fp(1, b, *dbl);
                let (m, p) = (match op { FOp::Add => "fadd", FOp::Sub => "fsub", FOp::Mul => "fmul", FOp::Div => "fdiv" }, if *dbl { "d" } else { "s" });
                self.ln(format!("{m} {p}0, {p}0, {p}1"));
                let d = self.def_reg(*dst);
                self.from_fp(d, 0, *dbl);
                self.finish(*dst, d);
            }
            MInst::FCmp { pred, dbl, dst, lhs, rhs } => {
                let a = self.use_reg(lhs, S0);
                self.to_fp(0, a, *dbl);
                let b = self.use_reg(rhs, S1);
                self.to_fp(1, b, *dbl);
                let p = if *dbl { "d" } else { "s" };
                self.ln(format!("fcmp {p}0, {p}1"));
                use FloatPred::*;
                // (first condition, optional second condition OR-ed in)
                let (c1, c2): (&str, Option<&str>) = match pred {
                    Oeq => ("eq", None), Ogt => ("gt", None), Oge => ("ge", None), Olt => ("mi", None), Ole => ("ls", None),
                    One => ("mi", Some("gt")), Ord => ("vc", None), Uno => ("vs", None), Ueq => ("eq", Some("vs")),
                    Ugt => ("hi", None), Uge => ("pl", None), Ult => ("lt", None), Ule => ("le", None), Une => ("ne", None),
                };
                let d = self.def_reg(*dst);
                match c2 {
                    None => self.ln(format!("cset {}, {c1}", d.w())),
                    Some(c2) => {
                        self.ln(format!("cset {}, {c1}", S1.w()));
                        self.ln(format!("cset {}, {c2}", S2.w()));
                        self.ln(format!("orr {}, {}, {}", d.w(), S1.w(), S2.w()));
                    }
                }
                self.finish(*dst, d);
            }
            MInst::FCvt { kind, dst, src } => {
                let a = self.use_reg(src, S0);
                match *kind {
                    FCvt::SIToFP { from, dbl } => {
                        let mut a = a;
                        if from < 64 { self.sext(S0, a, from); a = S0; }
                        self.ln(format!("scvtf {}0, {}", if dbl { "d" } else { "s" }, a.x()));
                        self.from_fp(S0, 0, dbl);
                    }
                    FCvt::UIToFP { dbl, .. } => {
                        self.ln(format!("ucvtf {}0, {}", if dbl { "d" } else { "s" }, a.x()));
                        self.from_fp(S0, 0, dbl);
                    }
                    FCvt::FPToSI { to, dbl } | FCvt::FPToUI { to, dbl } => {
                        let signed = matches!(kind, FCvt::FPToSI { .. });
                        self.to_fp(0, a, dbl);
                        self.ln(format!("{} x9, {}0", if signed { "fcvtzs" } else { "fcvtzu" }, if dbl { "d" } else { "s" }));
                        self.mask(S0, to);
                    }
                    FCvt::FPExt => { self.to_fp(0, a, false); self.ln("fcvt d0, s0"); self.from_fp(S0, 0, true); }
                    FCvt::FPTrunc => { self.to_fp(0, a, true); self.ln("fcvt s0, d0"); self.from_fp(S0, 0, false); }
                }
                let d = self.def_reg(*dst);
                if d != S0 { self.ln(format!("mov {}, x9", d.x())); }
                self.finish(*dst, d);
            }
            MInst::Jmp(t) => { if next != Some(*t) { let l = self.label(*t); self.ln(format!("b {l}")); } }
            MInst::Br { cond, then_bb, else_bb } => {
                let c = self.use_reg(cond, S0);
                let (lt, le) = (self.label(*then_bb), self.label(*else_bb));
                if next == Some(*then_bb) { self.ln(format!("cbz {}, {le}", c.x())); }
                else {
                    self.ln(format!("cbnz {}, {lt}", c.x()));
                    if next != Some(*else_bb) { self.ln(format!("b {le}")); }
                }
            }
            MInst::Switch { val, cases, default, .. } => {
                let v = self.use_reg(val, S0);
                for (c, t) in cases {
                    if *c <= 4095 { self.ln(format!("cmp {}, #{c}", v.x())); }
                    else { self.mov_imm(S1, *c); self.ln(format!("cmp {}, {}", v.x(), S1.x())); }
                    let l = self.label(*t);
                    self.ln(format!("b.eq {l}"));
                }
                if next != Some(*default) { let l = self.label(*default); self.ln(format!("b {l}")); }
            }
            MInst::Ret(vals) => {
                let classes: Vec<ArgClass> = vals.iter().map(|a| a.class).collect();
                let locs = classify_returns(&classes, AARCH64_ABI).expect("return classification was checked during lowering");
                for (a, l) in vals.iter().zip(locs) {
                    match l {
                        ArgLoc::IntReg(i) => self.load_into(R(i as u8), &a.val),
                        ArgLoc::FpReg(i) => { let r = self.use_reg(&a.val, S0); self.to_fp(i, r, a.class.is_double()); }
                        _ => unreachable!(),
                    }
                }
                self.epilogue();
            }
            MInst::VaStart { addr } => {
                let va = self.f.va.expect("va_start without a register save area");
                let p = self.use_reg(addr, S1).x();
                // AAPCS64 va_list: { void *__stack; void *__gr_top; void *__vr_top; int __gr_offs; int __vr_offs; }
                self.ln(format!("add x9, x29, #{}", 16 + 8 * va.named_stack));
                self.ln(format!("str x9, [{p}]"));
                let off = self.slot_off[va.save_slot];
                self.fp_addr(S0, off);
                self.ln("add x11, x9, #64");
                self.ln(format!("str x11, [{p}, #8]"));
                self.ln("add x11, x9, #192");
                self.ln(format!("str x11, [{p}, #16]"));
                self.ln(format!("mov w11, #-{}", 8 * (8 - va.named_gp)));
                self.ln(format!("str w11, [{p}, #24]"));
                self.ln(format!("mov w11, #-{}", 16 * (8 - va.named_fp)));
                self.ln(format!("str w11, [{p}, #28]"));
            }
            MInst::Trap => self.ln("brk #0"),
        }
    }

    fn alu(&mut self, op: Alu, bits: u32, dst: VReg, lhs: &Val, rhs: &Val) {
        let w = if bits == 64 { 64 } else { 32 };
        let a = self.use_reg(lhs, S0);
        match op {
            Alu::Add | Alu::Sub if matches!(rhs, Val::Imm(i) if *i <= 4095) => {
                let Val::Imm(i) = rhs else { unreachable!() };
                let d = self.def_reg(dst);
                self.ln(format!("{} {}, {}, #{i}", if op == Alu::Add { "add" } else { "sub" }, d.name(w), a.name(w)));
                if bits < 32 { self.mask(d, bits); }
                self.finish(dst, d);
            }
            Alu::Add | Alu::Sub | Alu::Mul | Alu::And | Alu::Or | Alu::Xor => {
                let b = self.use_reg(rhs, S1);
                let d = self.def_reg(dst);
                let m = match op { Alu::Add => "add", Alu::Sub => "sub", Alu::Mul => "mul", Alu::And => "and", Alu::Or => "orr", _ => "eor" };
                self.ln(format!("{m} {}, {}, {}", d.name(w), a.name(w), b.name(w)));
                if bits < 32 { self.mask(d, bits); }
                self.finish(dst, d);
            }
            Alu::Shl | Alu::LShr | Alu::AShr => {
                let mut a = a;
                let d = self.def_reg(dst);
                let ashr = op == Alu::AShr;
                if ashr && bits < 32 { self.sext(S0, a, bits); a = S0; }
                let m = match op { Alu::Shl => "lsl", Alu::LShr => "lsr", _ => "asr" };
                // Narrow arithmetic shifts operate on the sign-extended value, 64 bits wide.
                let sw = if ashr && bits < 32 { 64 } else { w };
                match rhs {
                    Val::Imm(k) if *k < 64 => self.ln(format!("{m} {}, {}, #{k}", d.name(sw), a.name(sw))),
                    _ => {
                        let b = self.use_reg(rhs, S1);
                        self.ln(format!("{m} {}, {}, {}", d.name(sw), a.name(sw), b.name(sw)));
                    }
                }
                if bits < 32 { self.mask(d, bits); }
                self.finish(dst, d);
            }
            Alu::UDiv | Alu::URem | Alu::SDiv | Alu::SRem => {
                let signed = matches!(op, Alu::SDiv | Alu::SRem);
                let (mut a, mut b) = (a, self.use_reg(rhs, S1));
                let mut dw = w;
                if signed && bits < 32 {
                    self.sext(S0, a, bits); a = S0;
                    self.sext(S1, b, bits); b = S1;
                    dw = 64;
                }
                let d = self.def_reg(dst);
                let (div, rem) = (if signed { "sdiv" } else { "udiv" }, matches!(op, Alu::URem | Alu::SRem));
                if rem {
                    self.ln(format!("{div} {}, {}, {}", S2.name(dw), a.name(dw), b.name(dw)));
                    self.ln(format!("msub {}, {}, {}, {}", d.name(dw), S2.name(dw), b.name(dw), a.name(dw)));
                } else {
                    self.ln(format!("{div} {}, {}, {}", d.name(dw), a.name(dw), b.name(dw)));
                }
                if bits < 32 { self.mask(d, bits); }
                self.finish(dst, d);
            }
        }
    }

    fn call(&mut self, dsts: &[(VReg, u32, ArgClass)], target: &CallTarget, args: &[Arg]) {
        let classes: Vec<ArgClass> = args.iter().map(|a| a.class).collect();
        let locs = classify(&classes, AARCH64_ABI);
        let total = align_up(8 * stack_slots(&classes, &locs) as u64, 16);
        if total > 0 { self.ln(format!("sub sp, sp, #{total}")); }
        for (a, l) in args.iter().zip(&locs) {
            match l {
                ArgLoc::Stack(k) => { let r = self.use_reg(&a.val, S0); self.ln(format!("str {}, [sp, #{}]", r.x(), 8 * k)); }
                ArgLoc::FpReg(i) => { let r = self.use_reg(&a.val, S0); self.to_fp(*i, r, a.class.is_double()); }
                ArgLoc::IntReg(i) => self.load_into(R(*i as u8), &a.val),
                ArgLoc::Sret => self.load_into(R(8), &a.val),
            }
        }
        match target {
            CallTarget::Direct { name, .. } => self.ln(format!("bl \"{name}\"")),
            CallTarget::Indirect(t) => { self.load_into(IP0, t); self.ln(format!("blr {}", IP0.x())); }
        }
        if total > 0 { self.ln(format!("add sp, sp, #{total}")); }
        let rcls: Vec<ArgClass> = dsts.iter().map(|d| d.2).collect();
        let rlocs = classify_returns(&rcls, AARCH64_ABI).expect("return classification was checked during lowering");
        for ((d, bits, class), loc) in dsts.iter().zip(rlocs) {
            match loc {
                ArgLoc::IntReg(i) => { let r = R(i as u8); self.mask(r, *bits); self.finish(*d, r); }
                ArgLoc::FpReg(i) => { self.from_fp(S0, i, class.is_double()); self.finish(*d, S0); }
                _ => unreachable!(),
            }
        }
    }
}

pub fn emit_function(f: &MFunc, alloc: &Allocation<R>) -> String {
    // Frame, addressed from x29: [saved callee-saved regs][alloca slots][spill slots].
    let saved = alloc.used_regs.clone();
    let mut cur = 8 * saved.len() as u64;
    let mut slot_off = Vec::new();
    for &(size, align) in &f.slots {
        cur = align_up(cur + size.max(1), align.max(1));
        slot_off.push(cur);
    }
    let mut spill_off = Vec::new();
    for _ in 0..alloc.num_spills { cur = align_up(cur + 8, 8); spill_off.push(cur); }
    let frame = align_up(cur, 16);

    let mut uses = vec![0u32; f.num_vregs as usize];
    for b in &f.blocks { for i in &b.insts { i.for_each_use(|u| uses[u as usize] += 1); } }
    let mut e = Emitter { out: String::new(), f, alloc, spill_off, slot_off, saved, uses };

    let _ = writeln!(e.out, ".text");
    let _ = writeln!(e.out, ".balign 4");
    if f.global { let _ = writeln!(e.out, ".globl \"{}\"", f.name); }
    let _ = writeln!(e.out, ".type \"{0}\", %function\n\"{0}\":", f.name);
    e.ln("stp x29, x30, [sp, #-16]!");
    e.ln("mov x29, sp");
    if frame > 0 {
        if frame <= 4095 { e.ln(format!("sub sp, sp, #{frame}")); }
        else { e.mov_imm(IP0, frame); e.ln(format!("sub sp, sp, {}", IP0.x())); }
    }
    for (i, r) in e.saved.clone().iter().enumerate() {
        e.ln(format!("str {}, [x29, #-{}]", r.x(), 8 * (i + 1)));
    }
    if let Some(va) = f.va {
        // Variadic prologue: spill x0-x7 and q0-q7 into the save area (GR part first).
        let off = e.slot_off[va.save_slot];
        e.fp_addr(IP0, off);
        let b = IP0.x();
        for i in 0..4 { e.ln(format!("stp x{}, x{}, [{b}, #{}]", 2 * i, 2 * i + 1, 16 * i)); }
        for i in 0..8 { e.ln(format!("str q{i}, [{b}, #{}]", 64 + 16 * i)); }
    }
    for (bi, blk) in f.blocks.iter().enumerate() {
        let l = e.label(bi);
        let _ = writeln!(e.out, "{l}:");
        let next = if bi + 1 < f.blocks.len() { Some(bi + 1) } else { None };
        let mut k = 0;
        while k < blk.insts.len() {
            // Fuse `cmp` + conditional branch when the flag value has no other use.
            if let (MInst::Cmp { pred, bits, dst, lhs, rhs }, Some(MInst::Br { cond: Val::V(c), then_bb, else_bb })) =
                (&blk.insts[k], blk.insts.get(k + 1))
            {
                if c == dst && e.uses[*dst as usize] == 1 {
                    let cc = e.emit_cmp(*pred, *bits, lhs, rhs);
                    let (lt, le) = (e.label(*then_bb), e.label(*else_bb));
                    if next == Some(*then_bb) {
                        e.ln(format!("b.{} {le}", cc_inverse(cc)));
                    } else {
                        e.ln(format!("b.{cc} {lt}"));
                        if next != Some(*else_bb) { e.ln(format!("b {le}")); }
                    }
                    k += 2;
                    continue;
                }
            }
            // Fuse `cmp` + `select` into `cmp` + `csel`.
            if let (MInst::Cmp { pred, bits, dst, lhs, rhs }, Some(MInst::Select { dst: sd, cond: Val::V(c), then_val, else_val })) =
                (&blk.insts[k], blk.insts.get(k + 1))
            {
                if c == dst && e.uses[*dst as usize] == 1 {
                    let cc = e.emit_cmp(*pred, *bits, lhs, rhs);
                    let a = e.use_reg(then_val, S0);
                    let b = e.use_reg(else_val, S1);
                    let d = e.def_reg(*sd);
                    e.ln(format!("csel {}, {}, {}, {cc}", d.x(), a.x(), b.x()));
                    e.finish(*sd, d);
                    k += 2;
                    continue;
                }
            }
            e.inst(&blk.insts[k], next);
            k += 1;
        }
    }
    let _ = writeln!(e.out, ".size \"{0}\", .-\"{0}\"", f.name);
    e.out
}
