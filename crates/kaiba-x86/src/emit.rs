// SPDX-License-Identifier: GPL-2.0-only WITH LicenseRef-Kaiba-Runtime-Exception
// Copyright (C) The Kaiba Authors
//
// This file is part of the Kaiba compiler toolchain. It is licensed under
// GPL-2.0-only with the Kaiba Runtime/Output Exception; see COPYING and
// LICENSE-EXCEPTION at the repository root.
//! Assembly emission (GNU as, Intel syntax) from machine IR + allocation.
//!
//! Register conventions: rax/rcx/rdx/r10/r11 and the argument registers are
//! scratch (never allocated); the allocator only hands out callee-saved
//! registers, so values survive calls without any save/restore around them.

use crate::mir::*;
use crate::regalloc::{Allocation, Loc};
use crate::regs::{PReg, ARG_REGS};
use llcr::{FloatPred, IntPred};
use std::fmt::Write;

fn align_up(x: u64, a: u64) -> u64 { (x + a - 1) / a * a }

/// Native operand width (in bits) used for a value of `bits` bits.
fn native_w(bits: u32) -> u32 { match bits { 64 => 64, 32 => 32, 16 => 16, _ => 8 } }

fn ptr_kw(w: u32) -> &'static str { match w { 64 => "QWORD", 32 => "DWORD", 16 => "WORD", _ => "BYTE" } }

fn imm_ok(i: u64, w: u32) -> bool {
    match w { 64 => i <= i32::MAX as u64, 32 => i <= u32::MAX as u64, 16 => i <= 0xffff, _ => i <= 0xff }
}

fn cc_of(p: IntPred) -> &'static str {
    match p {
        IntPred::Eq => "e", IntPred::Ne => "ne", IntPred::Ugt => "a", IntPred::Uge => "ae", IntPred::Ult => "b",
        IntPred::Ule => "be", IntPred::Sgt => "g", IntPred::Sge => "ge", IntPred::Slt => "l", IntPred::Sle => "le",
    }
}

fn cc_inverse(cc: &str) -> &'static str {
    match cc { "e" => "ne", "ne" => "e", "a" => "be", "be" => "a", "ae" => "b", "b" => "ae", "g" => "le", "le" => "g", "ge" => "l", _ => "ge" }
}

/// Condition after swapping the operands of a comparison.
fn cc_mirror(cc: &str) -> &'static str {
    match cc { "e" => "e", "ne" => "ne", "a" => "b", "b" => "a", "ae" => "be", "be" => "ae", "g" => "l", "l" => "g", "ge" => "le", _ => "ge" }
}

struct Emitter<'a> {
    out: String,
    f: &'a MFunc,
    alloc: &'a Allocation<PReg>,
    spill_off: Vec<u64>,
    slot_off: Vec<u64>,
    saved: Vec<PReg>,
    lbl: usize,
    uses: Vec<u32>,
}

impl<'a> Emitter<'a> {
    fn ln(&mut self, s: impl AsRef<str>) { let _ = writeln!(self.out, "    {}", s.as_ref()); }
    fn label(&self, b: usize) -> String { format!(".L{}_{}", self.f.name, b) }

    fn spill_mem(&self, i: usize) -> String { format!("QWORD PTR [rbp-{}]", self.spill_off[i]) }

    /// Load `v` into `reg` (64-bit). Never modifies flags.
    fn src_to(&mut self, reg: PReg, v: &Val) {
        let r = reg.name(64);
        match v {
            Val::V(x) => match self.alloc.locs[*x as usize] {
                Some(Loc::Reg(s)) => { if s != reg { self.ln(format!("mov {r}, {}", s.name(64))); } }
                Some(Loc::Spill(i)) => { let m = self.spill_mem(i); self.ln(format!("mov {r}, {m}")); }
                None => self.ln(format!("mov {}, 0", reg.name(32))),
            },
            Val::Imm(i) => {
                if *i <= u32::MAX as u64 { self.ln(format!("mov {}, {i}", reg.name(32))); } else { self.ln(format!("mov {r}, {i}")); }
            }
            Val::Sym { name, external } => {
                if *external { self.ln(format!("mov {r}, QWORD PTR [rip + \"{name}\"@GOTPCREL]")); }
                else { self.ln(format!("lea {r}, [rip + \"{name}\"]")); }
            }
        }
    }

    fn store_dst(&mut self, dst: VReg, reg: PReg) {
        match self.alloc.locs[dst as usize] {
            Some(Loc::Reg(d)) => { if d != reg { self.ln(format!("mov {}, {}", d.name(64), reg.name(64))); } }
            Some(Loc::Spill(i)) => { let m = self.spill_mem(i); self.ln(format!("mov {m}, {}", reg.name(64))); }
            None => {}
        }
    }

    /// Re-canonicalize `r` (zero-extend from `bits`).
    fn mask(&mut self, r: PReg, bits: u32) {
        match bits {
            64 => {}
            32 => self.ln(format!("mov {0}, {0}", r.name(32))),
            16 | 8 => self.ln(format!("movzx {}, {}", r.name(32), r.name(bits))),
            _ => self.ln(format!("and {}, 1", r.name(32))),
        }
    }

    /// Sign-extend the low `bits` of `r` to 64 bits (no-op if `signed` is false).
    fn ext(&mut self, r: PReg, bits: u32, signed: bool) {
        if !signed { return; }
        match bits {
            64 => {}
            32 => self.ln(format!("movsxd {}, {}", r.name(64), r.name(32))),
            16 | 8 => self.ln(format!("movsx {}, {}", r.name(64), r.name(bits))),
            _ => self.ln(format!("neg {}", r.name(64))),
        }
    }

    /// Perform register-to-register moves "simultaneously" (`(dst, src)` pairs),
    /// ordering them so no source is clobbered and breaking cycles through rax.
    fn parallel_moves(&mut self, mut moves: Vec<(PReg, PReg)>) {
        moves.retain(|(d, s)| d != s);
        while !moves.is_empty() {
            let free = (0..moves.len()).find(|&i| !moves.iter().any(|(_, s)| *s == moves[i].0));
            if let Some(i) = free {
                let (d, s) = moves.remove(i);
                self.ln(format!("mov {}, {}", d.name(64), s.name(64)));
            } else {
                // Only cycles remain: park one destination's current value in r11.
                let d = moves[0].0;
                self.ln(format!("mov rax, {}", d.name(64)));
                for m in moves.iter_mut() { if m.1 == d { m.1 = PReg::Rax; } }
            }
        }
    }

    /// Function-entry parameters. Integer-register parameters may land in
    /// registers that other parameters still occupy, so they move in parallel.
    fn emit_params(&mut self, ps: &[MInst]) {
        let (mut moves, mut masks) = (Vec::new(), Vec::new());
        let mut rest = Vec::new();
        for p in ps {
            match p {
                MInst::Param { dst, class: ArgClass::Int | ArgClass::Sret, loc: ArgLoc::IntReg(i), bits } => {
                    let src = ARG_REGS[*i];
                    match self.alloc.locs[*dst as usize] {
                        Some(Loc::Reg(d)) => { moves.push((d, src)); masks.push((d, *bits)); }
                        Some(Loc::Spill(_)) => { self.mask(src, *bits); self.store_dst(*dst, src); }
                        None => {}
                    }
                }
                other => rest.push(other.clone()),
            }
        }
        self.parallel_moves(moves);
        for (d, bits) in masks { self.mask(d, bits); }
        for p in &rest { self.inst(p, None); }
    }

    fn spill_at(&self, i: usize, w: u32) -> String { format!("{} PTR [rbp-{}]", ptr_kw(w), self.spill_off[i]) }

    /// `v` as a register or memory operand of width `w`, if it lives in one.
    fn rm(&self, v: &Val, w: u32) -> Option<String> {
        match v {
            Val::V(x) => match self.alloc.locs[*x as usize] {
                Some(Loc::Reg(r)) => Some(r.name(w).to_string()),
                Some(Loc::Spill(i)) => Some(self.spill_at(i, w)),
                None => None,
            },
            _ => None,
        }
    }

    /// Like `rm`, but also accepts an immediate that fits the instruction encoding.
    fn rmi(&self, v: &Val, w: u32) -> Option<String> {
        if let Some(s) = self.rm(v, w) { return Some(s); }
        if let Val::Imm(i) = v { if imm_ok(*i, w) { return Some(i.to_string()); } }
        None
    }

    fn in_reg(&self, v: &Val, r: PReg) -> bool {
        matches!(v, Val::V(x) if self.alloc.locs[*x as usize] == Some(Loc::Reg(r)))
    }

    /// Destination register for a result: its own register, or `r10` when spilled.
    fn dst_reg(&self, dst: VReg) -> PReg {
        match self.alloc.locs[dst as usize] { Some(Loc::Reg(r)) => r, _ => PReg::R10 }
    }

    /// The inside of a `[...]` memory operand. Scratch: r11 (base) and rax (index).
    fn addr_str(&mut self, a: &Addr) -> String {
        let reg_of = |e: &Self, v: &Val| e.rm(v, 64).filter(|s| !s.contains("PTR"));
        let mut s = match reg_of(self, &a.base) {
            Some(r) => r,
            None => { self.src_to(PReg::R11, &a.base); "r11".to_string() }
        };
        if let Some((iv, sc)) = &a.index {
            let ir = match reg_of(self, iv) {
                Some(r) => r,
                None => { self.src_to(PReg::Rax, iv); "rax".to_string() }
            };
            s.push_str(&format!("+{ir}*{sc}"));
        }
        if a.disp != 0 { s.push_str(&format!("{:+}", a.disp)); }
        s
    }

    /// Emit a compare and return the condition code that means "pred holds".
    fn emit_cmp(&mut self, pred: IntPred, bits: u32, lhs: &Val, rhs: &Val) -> &'static str {
        let w = native_w(bits);
        let mut cc = cc_of(pred);
        let (mut l, mut r) = (lhs.clone(), rhs.clone());
        if self.rm(&l, w).is_none() && self.rm(&r, w).is_some() && matches!(l, Val::Imm(_)) {
            std::mem::swap(&mut l, &mut r);
            cc = cc_mirror(cc);
        }
        let ls = match self.rm(&l, w) { Some(s) => s, None => { self.src_to(PReg::R10, &l); PReg::R10.name(w).to_string() } };
        let lmem = ls.contains("PTR");
        let rs = match self.rmi(&r, w) {
            Some(s) if !(lmem && s.contains("PTR")) => s,
            _ => { self.src_to(PReg::R11, &r); PReg::R11.name(w).to_string() }
        };
        self.ln(format!("cmp {ls}, {rs}"));
        cc
    }

    fn new_label(&mut self) -> String { self.lbl += 1; format!(".Lf{}_{}", self.f.name, self.lbl) }

    /// GPR bits -> xmm`n` (movq for f64, movd for f32).
    fn to_xmm(&mut self, n: usize, r: PReg, dbl: bool) {
        if dbl { self.ln(format!("movq xmm{n}, {}", r.name(64))); } else { self.ln(format!("movd xmm{n}, {}", r.name(32))); }
    }

    /// xmm`n` -> GPR bits (f32 results are zero-extended, keeping values canonical).
    fn from_xmm(&mut self, r: PReg, n: usize, dbl: bool) {
        if dbl { self.ln(format!("movq {}, xmm{n}", r.name(64))); } else { self.ln(format!("movd {}, xmm{n}", r.name(32))); }
    }

    fn epilogue(&mut self) {
        for (i, r) in self.saved.clone().iter().enumerate() {
            self.ln(format!("mov {}, QWORD PTR [rbp-{}]", r.name(64), 8 * (i + 1)));
        }
        self.ln("leave");
        self.ln("ret");
    }

    fn inst(&mut self, inst: &MInst, next: Option<usize>) {
        use PReg::*;
        match inst {
            MInst::Param { dst, class, loc, bits } => match loc {
                ArgLoc::IntReg(i) => {
                    let r = ARG_REGS[*i];
                    self.mask(r, *bits);
                    self.store_dst(*dst, r);
                }
                ArgLoc::FpReg(i) => {
                    self.from_xmm(R10, *i, class.is_double());
                    self.store_dst(*dst, R10);
                }
                ArgLoc::Stack(k) => {
                    if let ArgClass::Mem { .. } = class {
                        // A by-value aggregate: the parameter is the address of the caller's copy.
                        self.ln(format!("lea rax, [rbp+{}]", 16 + 8 * k));
                    } else {
                        self.ln(format!("mov rax, QWORD PTR [rbp+{}]", 16 + 8 * k));
                        match class { ArgClass::F32 => self.mask(Rax, 32), ArgClass::F64 => {} _ => self.mask(Rax, *bits) }
                    }
                    self.store_dst(*dst, Rax);
                }
                ArgLoc::Sret => unreachable!("x86-64 passes the struct-return pointer as an ordinary argument"),
            },
            MInst::Mov { dst, src } => match self.alloc.locs[*dst as usize] {
                Some(Loc::Reg(d)) => self.src_to(d, src),
                _ => { self.src_to(R10, src); self.store_dst(*dst, R10); }
            },
            MInst::Alu { op, bits, dst, lhs, rhs } => {
                let bits = *bits;
                match op {
                    Alu::Add | Alu::Sub | Alu::Mul | Alu::And | Alu::Or | Alu::Xor => {
                        let w = if bits == 64 { 64 } else { 32 };
                        let m = match op { Alu::Add => "add", Alu::Sub => "sub", Alu::Mul => "imul", Alu::And => "and", Alu::Or => "or", _ => "xor" };
                        let d = self.dst_reg(*dst);
                        let (l, r) = if *op != Alu::Sub && self.in_reg(rhs, d) && !self.in_reg(lhs, d) {
                            (rhs.clone(), lhs.clone())
                        } else { (lhs.clone(), rhs.clone()) };
                        let rs = if self.in_reg(&r, d) && !self.in_reg(&l, d) {
                            self.src_to(R11, &r);
                            R11.name(w).to_string()
                        } else if let Some(s) = self.rmi(&r, w) { s } else {
                            self.src_to(R11, &r);
                            R11.name(w).to_string()
                        };
                        self.src_to(d, &l);
                        self.ln(format!("{m} {}, {rs}", d.name(w)));
                        if bits < 32 { self.mask(d, bits); }
                        self.store_dst(*dst, d);
                    }
                    Alu::Shl | Alu::LShr | Alu::AShr if matches!(rhs, Val::Imm(k) if *k < 64) => {
                        let Val::Imm(k) = rhs else { unreachable!() };
                        let d = self.dst_reg(*dst);
                        self.src_to(d, lhs);
                        match op {
                            Alu::Shl | Alu::LShr => {
                                let w = if bits == 64 { 64 } else { 32 };
                                self.ln(format!("{} {}, {k}", if *op == Alu::Shl { "shl" } else { "shr" }, d.name(w)));
                                if bits < 32 { self.mask(d, bits); }
                            }
                            _ if bits >= 32 => self.ln(format!("sar {}, {k}", d.name(bits))),
                            _ => { self.ext(d, bits, true); self.ln(format!("sar {}, {k}", d.name(64))); self.mask(d, bits); }
                        }
                        self.store_dst(*dst, d);
                    }
                    Alu::Shl | Alu::LShr | Alu::AShr => {
                        self.src_to(R10, lhs);
                        self.src_to(Rcx, rhs);
                        if *op == Alu::AShr { self.ext(R10, bits, true); }
                        let m = match op { Alu::Shl => "shl", Alu::LShr => "shr", _ => "sar" };
                        self.ln(format!("{m} r10, cl"));
                        self.mask(R10, bits);
                        self.store_dst(*dst, R10);
                    }
                    Alu::UDiv | Alu::URem => {
                        self.src_to(Rax, lhs);
                        self.src_to(Rcx, rhs);
                        self.ln("xor edx, edx");
                        self.ln("div rcx");
                        self.store_dst(*dst, if *op == Alu::UDiv { Rax } else { Rdx });
                    }
                    Alu::SDiv | Alu::SRem => {
                        self.src_to(Rax, lhs);
                        self.ext(Rax, bits, true);
                        self.src_to(Rcx, rhs);
                        self.ext(Rcx, bits, true);
                        self.ln("cqo");
                        self.ln("idiv rcx");
                        let r = if *op == Alu::SDiv { Rax } else { Rdx };
                        self.mask(r, bits);
                        self.store_dst(*dst, r);
                    }
                }
            }
            MInst::Cmp { pred, bits, dst, lhs, rhs } => {
                let cc = self.emit_cmp(*pred, *bits, lhs, rhs);
                let d = self.dst_reg(*dst);
                self.ln(format!("set{cc} {}", d.name(8)));
                self.ln(format!("movzx {}, {}", d.name(32), d.name(8)));
                self.store_dst(*dst, d);
            }
            MInst::Select { dst, cond, then_val, else_val } => {
                self.src_to(R11, cond);
                self.ln("test r11, r11");
                self.src_to(R10, else_val);
                self.src_to(R11, then_val);
                self.ln("cmovne r10, r11");
                self.store_dst(*dst, R10);
            }
            MInst::Ext { kind, from, to, dst, src } => {
                let d = self.dst_reg(*dst);
                self.src_to(d, src);
                match kind {
                    ExtKind::Zext => {}
                    ExtKind::Sext => { self.ext(d, *from, true); self.mask(d, *to); }
                    ExtKind::Trunc => self.mask(d, *to),
                }
                self.store_dst(*dst, d);
            }
            MInst::Load { dst, bits, addr } => {
                let a = self.addr_str(addr);
                let d = self.dst_reg(*dst);
                match bits {
                    64 => self.ln(format!("mov {}, QWORD PTR [{a}]", d.name(64))),
                    32 => self.ln(format!("mov {}, DWORD PTR [{a}]", d.name(32))),
                    16 => self.ln(format!("movzx {}, WORD PTR [{a}]", d.name(32))),
                    _ => self.ln(format!("movzx {}, BYTE PTR [{a}]", d.name(32))),
                }
                if *bits == 1 { self.mask(d, 1); }
                self.store_dst(*dst, d);
            }
            MInst::Store { bits, val, addr } => {
                let a = self.addr_str(addr);
                let w = native_w(*bits);
                let src = match val {
                    Val::V(x) => match self.alloc.locs[*x as usize] {
                        Some(Loc::Reg(r)) => r.name(w).to_string(),
                        _ => { self.src_to(R10, val); R10.name(w).to_string() }
                    },
                    Val::Imm(i) if imm_ok(*i, w) => i.to_string(),
                    _ => { self.src_to(R10, val); R10.name(w).to_string() }
                };
                self.ln(format!("mov {} PTR [{a}], {src}", ptr_kw(w)));
            }
            MInst::Lea { dst, slot } => {
                self.ln(format!("lea r10, [rbp-{}]", self.slot_off[*slot]));
                self.store_dst(*dst, R10);
            }
            MInst::Call { dsts, target, args } => {
                let classes: Vec<ArgClass> = args.iter().map(|a| a.class).collect();
                let locs = classify(&classes, crate::X86_ABI);
                let slots = stack_slots(&classes, &locs);
                let total = align_up(8 * slots as u64, 16);
                if total > 0 { self.ln(format!("sub rsp, {total}")); }
                for (a, l) in args.iter().zip(&locs) {
                    let ArgLoc::Stack(k) = l else { continue };
                    if let ArgClass::Mem { size, .. } = a.class {
                        // Copy the aggregate into the outgoing argument area, 8 bytes at a time.
                        self.src_to(R11, &a.val);
                        let mut off = 0u64;
                        while off < size {
                            let rem = size - off;
                            let (w, r, kw) = if rem >= 8 { (8, "r10", "QWORD") } else if rem >= 4 { (4, "r10d", "DWORD") } else if rem >= 2 { (2, "r10w", "WORD") } else { (1, "r10b", "BYTE") };
                            if w == 8 { self.ln(format!("mov r10, QWORD PTR [r11+{off}]")); }
                            else if w == 4 { self.ln(format!("mov r10d, DWORD PTR [r11+{off}]")); }
                            else { self.ln(format!("movzx r10d, {kw} PTR [r11+{off}]")); }
                            self.ln(format!("mov {kw} PTR [rsp+{}], {r}", 8 * *k as u64 + off));
                            off += w;
                        }
                    } else {
                        self.src_to(R10, &a.val);
                        self.ln(format!("mov QWORD PTR [rsp+{}], r10", 8 * k));
                    }
                }
                let mut nfp = 0;
                for (a, l) in args.iter().zip(&locs) {
                    if let ArgLoc::FpReg(i) = l {
                        self.src_to(R10, &a.val);
                        self.to_xmm(*i, R10, a.class.is_double());
                        nfp += 1;
                    }
                }
                // Integer arguments: register-held sources move in parallel, the rest are loaded afterwards.
                let (mut moves, mut loads) = (Vec::new(), Vec::new());
                for (a, l) in args.iter().zip(&locs) {
                    if let ArgLoc::IntReg(i) = l {
                        let t = ARG_REGS[*i];
                        match &a.val {
                            Val::V(x) => match self.alloc.locs[*x as usize] {
                                Some(Loc::Reg(r)) => moves.push((t, r)),
                                _ => loads.push((t, a.val.clone())),
                            },
                            _ => loads.push((t, a.val.clone())),
                        }
                    }
                }
                // The indirect target may sit in an argument register: read it first.
                if let CallTarget::Indirect(t) = target { self.src_to(R11, t); }
                self.parallel_moves(moves);
                for (t, v) in &loads { self.src_to(*t, v); }
                // For variadic callees `al` holds the number of vector registers used.
                self.ln(format!("mov eax, {nfp}"));
                match target {
                    CallTarget::Direct { name, external } => {
                        if *external { self.ln(format!("call \"{name}\"@PLT")); } else { self.ln(format!("call \"{name}\"")); }
                    }
                    CallTarget::Indirect(_) => self.ln("call r11"),
                }
                if total > 0 { self.ln(format!("add rsp, {total}")); }
                let rcls: Vec<ArgClass> = dsts.iter().map(|d| d.2).collect();
                let rlocs = classify_returns(&rcls, crate::X86_ABI).expect("return classification was checked during lowering");
                for ((d, bits, class), loc) in dsts.iter().zip(rlocs) {
                    match loc {
                        ArgLoc::IntReg(i) => { let r = [Rax, Rdx][i]; self.mask(r, *bits); self.store_dst(*d, r); }
                        ArgLoc::FpReg(i) => { self.from_xmm(R10, i, class.is_double()); self.store_dst(*d, R10); }
                        _ => unreachable!(),
                    }
                }
            }
            MInst::FAlu { op, dbl, dst, lhs, rhs } => {
                self.src_to(R10, lhs);
                self.src_to(R11, rhs);
                self.to_xmm(0, R10, *dbl);
                self.to_xmm(1, R11, *dbl);
                let m = match op { FOp::Add => "add", FOp::Sub => "sub", FOp::Mul => "mul", FOp::Div => "div" };
                self.ln(format!("{m}{} xmm0, xmm1", if *dbl { "sd" } else { "ss" }));
                self.from_xmm(R10, 0, *dbl);
                self.store_dst(*dst, R10);
            }
            MInst::FCmp { pred, dbl, dst, lhs, rhs } => {
                use FloatPred::*;
                // (swap operands?, condition, optional second condition + combiner)
                let (swap, cc, extra): (bool, &str, Option<(&str, &str)>) = match pred {
                    Ogt => (false, "a", None), Oge => (false, "ae", None),
                    Olt => (true, "a", None), Ole => (true, "ae", None),
                    Oeq => (false, "e", Some(("np", "and"))), One => (false, "ne", Some(("np", "and"))),
                    Ord => (false, "np", None), Uno => (false, "p", None),
                    Ueq => (false, "e", None), Une => (false, "ne", Some(("p", "or"))),
                    Ugt => (true, "b", None), Uge => (true, "be", None),
                    Ult => (false, "b", None), Ule => (false, "be", None),
                };
                self.src_to(R10, lhs);
                self.src_to(R11, rhs);
                self.to_xmm(0, R10, *dbl);
                self.to_xmm(1, R11, *dbl);
                let (a, b) = if swap { (1, 0) } else { (0, 1) };
                self.ln(format!("ucomis{} xmm{a}, xmm{b}", if *dbl { "d" } else { "s" }));
                self.ln(format!("set{cc} r10b"));
                if let Some((cc2, comb)) = extra {
                    self.ln(format!("set{cc2} r11b"));
                    self.ln(format!("{comb} r10b, r11b"));
                }
                self.ln("movzx r10d, r10b");
                self.store_dst(*dst, R10);
            }
            MInst::FCvt { kind, dst, src } => {
                self.src_to(R10, src);
                match *kind {
                    FCvt::SIToFP { from, dbl } => {
                        self.ext(R10, from, true);
                        self.ln("pxor xmm0, xmm0");
                        self.ln(format!("cvtsi2{} xmm0, r10", if dbl { "sd" } else { "ss" }));
                        self.from_xmm(R10, 0, dbl);
                    }
                    FCvt::UIToFP { from, dbl } => {
                        let s = if dbl { "sd" } else { "ss" };
                        self.ln("pxor xmm0, xmm0");
                        if from < 64 {
                            self.ln(format!("cvtsi2{s} xmm0, r10"));
                        } else {
                            let (neg, done) = (self.new_label(), self.new_label());
                            self.ln("test r10, r10");
                            self.ln(format!("js {neg}"));
                            self.ln(format!("cvtsi2{s} xmm0, r10"));
                            self.ln(format!("jmp {done}"));
                            let _ = writeln!(self.out, "{neg}:");
                            self.ln("mov r11, r10");
                            self.ln("shr r11, 1");
                            self.ln("and r10d, 1");
                            self.ln("or r11, r10");
                            self.ln(format!("cvtsi2{s} xmm0, r11"));
                            self.ln(format!("add{s} xmm0, xmm0"));
                            let _ = writeln!(self.out, "{done}:");
                        }
                        self.from_xmm(R10, 0, dbl);
                    }
                    FCvt::FPToSI { to, dbl } => {
                        self.to_xmm(0, R10, dbl);
                        self.ln(format!("cvtt{}2si r10, xmm0", if dbl { "sd" } else { "ss" }));
                        self.mask(R10, to);
                    }
                    FCvt::FPToUI { to, dbl } => {
                        self.to_xmm(0, R10, dbl);
                        let s = if dbl { "sd" } else { "ss" };
                        if to < 64 {
                            self.ln(format!("cvtt{s}2si r10, xmm0"));
                            self.mask(R10, to);
                        } else {
                            // Values >= 2^63 are converted as (x - 2^63) ^ 2^63.
                            let (big, done) = (self.new_label(), self.new_label());
                            self.ln(format!("mov r11, {}", if dbl { "0x43E0000000000000" } else { "0x5F000000" }));
                            self.to_xmm(1, R11, dbl);
                            self.ln(format!("ucomis{} xmm0, xmm1", if dbl { "d" } else { "s" }));
                            self.ln(format!("jae {big}"));
                            self.ln(format!("cvtt{s}2si r10, xmm0"));
                            self.ln(format!("jmp {done}"));
                            let _ = writeln!(self.out, "{big}:");
                            self.ln(format!("sub{s} xmm0, xmm1"));
                            self.ln(format!("cvtt{s}2si r10, xmm0"));
                            self.ln("mov r11, 0x8000000000000000");
                            self.ln("xor r10, r11");
                            let _ = writeln!(self.out, "{done}:");
                        }
                    }
                    FCvt::FPExt => {
                        self.to_xmm(0, R10, false);
                        self.ln("cvtss2sd xmm0, xmm0");
                        self.from_xmm(R10, 0, true);
                    }
                    FCvt::FPTrunc => {
                        self.to_xmm(0, R10, true);
                        self.ln("cvtsd2ss xmm0, xmm0");
                        self.from_xmm(R10, 0, false);
                    }
                }
                self.store_dst(*dst, R10);
            }
            MInst::Jmp(t) => { if next != Some(*t) { let l = self.label(*t); self.ln(format!("jmp {l}")); } }
            MInst::Br { cond, then_bb, else_bb } => {
                match self.rm(cond, 64).filter(|s| !s.contains("PTR")) {
                    Some(r) => self.ln(format!("test {r}, {r}")),
                    None => { self.src_to(R10, cond); self.ln("test r10, r10"); }
                }
                let (lt, le) = (self.label(*then_bb), self.label(*else_bb));
                if next == Some(*then_bb) {
                    self.ln(format!("je {le}"));
                } else {
                    self.ln(format!("jne {lt}"));
                    if next != Some(*else_bb) { self.ln(format!("jmp {le}")); }
                }
            }
            MInst::Switch { val, cases, default, .. } => {
                self.src_to(R10, val);
                for (v, t) in cases {
                    if *v <= i32::MAX as u64 { self.ln(format!("cmp r10, {v}")); }
                    else { self.ln(format!("mov r11, {v}")); self.ln("cmp r10, r11"); }
                    let l = self.label(*t);
                    self.ln(format!("je {l}"));
                }
                if next != Some(*default) { let l = self.label(*default); self.ln(format!("jmp {l}")); }
            }
            MInst::Ret(vals) => {
                let classes: Vec<ArgClass> = vals.iter().map(|a| a.class).collect();
                let locs = classify_returns(&classes, crate::X86_ABI).expect("return classification was checked during lowering");
                for (a, l) in vals.iter().zip(locs) {
                    match l {
                        ArgLoc::IntReg(i) => self.src_to([Rax, Rdx][i], &a.val),
                        ArgLoc::FpReg(i) => { self.src_to(R10, &a.val); self.to_xmm(i, R10, a.class.is_double()); }
                        _ => unreachable!(),
                    }
                }
                self.epilogue();
            }
            MInst::VaStart { addr } => {
                let va = self.f.va.expect("va_start without a register save area");
                let b = match self.rm(addr, 64) {
                    Some(r) if !r.contains("PTR") => r,
                    _ => { self.src_to(R11, addr); "r11".to_string() }
                };
                self.ln(format!("mov DWORD PTR [{b}], {}", 8 * va.named_gp));
                self.ln(format!("mov DWORD PTR [{b}+4], {}", 48 + 16 * va.named_fp));
                self.ln(format!("lea r10, [rbp+{}]", 16 + 8 * va.named_stack));
                self.ln(format!("mov QWORD PTR [{b}+8], r10"));
                self.ln(format!("lea r10, [rbp-{}]", self.slot_off[va.save_slot]));
                self.ln(format!("mov QWORD PTR [{b}+16], r10"));
            }
            MInst::Trap => self.ln("ud2"),
        }
    }
}

pub fn emit_function(f: &MFunc, alloc: &Allocation<PReg>) -> String {
    // Frame: [saved callee-saved regs][alloca slots][spill slots], rbp-relative.
    let saved = alloc.used_regs.clone();
    let mut cur = 8 * saved.len() as u64;
    let mut slot_off = Vec::new();
    for &(size, align) in &f.slots {
        cur = align_up(cur + size.max(1), align.max(1));
        slot_off.push(cur);
    }
    let mut spill_off = Vec::new();
    for _ in 0..alloc.num_spills {
        cur = align_up(cur + 8, 8);
        spill_off.push(cur);
    }
    let frame = align_up(cur, 16);

    let mut uses = vec![0u32; f.num_vregs as usize];
    for b in &f.blocks { for i in &b.insts { i.for_each_use(|u| uses[u as usize] += 1); } }
    let mut e = Emitter { out: String::new(), f, alloc, spill_off, slot_off, saved, lbl: 0, uses };
    let _ = writeln!(e.out, ".text");
    if f.global { let _ = writeln!(e.out, ".globl \"{}\"", f.name); }
    let _ = writeln!(e.out, ".type \"{0}\", @function\n\"{0}\":", f.name);
    e.ln("push rbp");
    e.ln("mov rbp, rsp");
    if frame > 0 { e.ln(format!("sub rsp, {frame}")); }
    for (i, r) in e.saved.clone().iter().enumerate() {
        e.ln(format!("mov QWORD PTR [rbp-{}], {}", 8 * (i + 1), r.name(64)));
    }
    if let Some(va) = f.va {
        // Variadic prologue: spill the argument registers into the save area
        // (6 GPRs, then 8 XMM registers) before anything can clobber them.
        let base = -(e.slot_off[va.save_slot] as i64);
        for (i, r) in ["rdi", "rsi", "rdx", "rcx", "r8", "r9"].iter().enumerate() {
            e.ln(format!("mov QWORD PTR [rbp{:+}], {r}", base + 8 * i as i64));
        }
        for i in 0..8 { e.ln(format!("movups XMMWORD PTR [rbp{:+}], xmm{i}", base + 48 + 16 * i as i64)); }
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
                        e.ln(format!("j{} {le}", cc_inverse(cc)));
                    } else {
                        e.ln(format!("j{cc} {lt}"));
                        if next != Some(*else_bb) { e.ln(format!("jmp {le}")); }
                    }
                    k += 2;
                    continue;
                }
            }
            if bi == 0 && k == 0 && matches!(blk.insts[0], MInst::Param { .. }) {
                let n = blk.insts.iter().take_while(|i| matches!(i, MInst::Param { .. })).count();
                e.emit_params(&blk.insts[..n]);
                k += n;
                continue;
            }
            // Fuse `cmp` + `select` into `cmp` + `cmov`.
            if let (MInst::Cmp { pred, bits, dst, lhs, rhs }, Some(MInst::Select { dst: sd, cond: Val::V(c), then_val, else_val })) =
                (&blk.insts[k], blk.insts.get(k + 1))
            {
                if c == dst && e.uses[*dst as usize] == 1 {
                    let cc = e.emit_cmp(*pred, *bits, lhs, rhs);
                    e.src_to(PReg::R10, else_val);
                    e.src_to(PReg::R11, then_val);
                    e.ln(format!("cmov{cc} r10, r11"));
                    e.store_dst(*sd, PReg::R10);
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
