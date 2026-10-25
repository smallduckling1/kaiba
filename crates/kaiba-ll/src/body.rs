// SPDX-License-Identifier: GPL-2.0-only WITH LicenseRef-Kaiba-Runtime-Exception
// Copyright (C) The Kaiba Authors
//
// This file is part of the Kaiba compiler toolchain. It is licensed under
// GPL-2.0-only with the Kaiba Runtime/Output Exception; see COPYING and
// LICENSE-EXCEPTION at the repository root.
//! Lowering of `.ll` function bodies into llcr.
//!
//! Value names are interned to *provisional* ids while parsing, which makes
//! forward references (phis, out-of-order blocks) trivial; a final pass maps
//! provisional ids onto the real arena values.

use crate::cur::Cur;
use crate::{skip_attrs, Ctx, LlError, ParamGroup};
use llcr::layout::size_align;
use llcr::{
    BinOp, BlockId, Callee, CastOp, Constant, Entity, FloatPred, FuncId, Function, InstKind, IntPred,
    Operand, Signature, Terminator, Type, ValueId,
};
use std::collections::HashMap;

/// `Tuple` models the `{iN, i1}` pair returned by `*.with.overflow` until `extractvalue` splits it.
enum Def { Real(ValueId), Alias(Operand), Tuple(Vec<Operand>) }

struct Body<'a> {
    cx: &'a mut Ctx,
    f: Function,
    names: Vec<String>,
    name_ix: HashMap<String, u32>,
    defs: HashMap<u32, Def>,
    blocks: HashMap<String, BlockId>,
    cur: BlockId,
    fresh: usize,
}

fn mask128(bits: u32) -> u128 { if bits >= 128 { u128::MAX } else { (1u128 << bits) - 1 } }

const FLAGS: &[&str] = &[
    "nsw", "nuw", "exact", "inbounds", "nusw", "nneg", "disjoint", "samesign", "fast", "nnan", "ninf", "nsz", "arcp",
    "contract", "afn", "reassoc",
];

fn binop(m: &str) -> Option<BinOp> {
    Some(match m {
        "add" => BinOp::Add, "sub" => BinOp::Sub, "mul" => BinOp::Mul, "udiv" => BinOp::UDiv, "sdiv" => BinOp::SDiv,
        "urem" => BinOp::URem, "srem" => BinOp::SRem, "shl" => BinOp::Shl, "lshr" => BinOp::LShr, "ashr" => BinOp::AShr,
        "and" => BinOp::And, "or" => BinOp::Or, "xor" => BinOp::Xor, "fadd" => BinOp::FAdd, "fsub" => BinOp::FSub,
        "fmul" => BinOp::FMul, "fdiv" => BinOp::FDiv, "frem" => BinOp::FRem,
        _ => return None,
    })
}

fn castop(m: &str) -> Option<CastOp> {
    Some(match m {
        "trunc" => CastOp::Trunc, "zext" => CastOp::ZExt, "sext" => CastOp::SExt, "fptrunc" => CastOp::FPTrunc,
        "fpext" => CastOp::FPExt, "fptoui" => CastOp::FPToUI, "fptosi" => CastOp::FPToSI, "uitofp" => CastOp::UIToFP,
        "sitofp" => CastOp::SIToFP, "ptrtoint" => CastOp::PtrToInt, "inttoptr" => CastOp::IntToPtr,
        "bitcast" => CastOp::Bitcast,
        _ => return None,
    })
}

fn ipred(m: &str) -> Option<IntPred> {
    Some(match m {
        "eq" => IntPred::Eq, "ne" => IntPred::Ne, "ugt" => IntPred::Ugt, "uge" => IntPred::Uge, "ult" => IntPred::Ult,
        "ule" => IntPred::Ule, "sgt" => IntPred::Sgt, "sge" => IntPred::Sge, "slt" => IntPred::Slt, "sle" => IntPred::Sle,
        _ => return None,
    })
}

fn fpred(m: &str) -> Option<FloatPred> {
    Some(match m {
        "oeq" => FloatPred::Oeq, "ogt" => FloatPred::Ogt, "oge" => FloatPred::Oge, "olt" => FloatPred::Olt,
        "ole" => FloatPred::Ole, "one" => FloatPred::One, "ord" => FloatPred::Ord, "ueq" => FloatPred::Ueq,
        "ugt" => FloatPred::Ugt, "uge" => FloatPred::Uge, "ult" => FloatPred::Ult, "ule" => FloatPred::Ule,
        "une" => FloatPred::Une, "uno" => FloatPred::Uno,
        _ => return None,
    })
}

/// A block label at column 0, e.g. `for.cond:   ; preds = ...`.
fn parse_label(line: &str) -> Option<String> {
    if line.is_empty() || line.starts_with(|c: char| c == ' ' || c == '\t' || c == ';') { return None; }
    if line.starts_with('"') {
        let mut c = Cur::new(line);
        let s = c.quoted().ok()?;
        return c.eat(":").then(|| String::from_utf8_lossy(&s).into_owned());
    }
    let end = line.find(':')?;
    Some(line[..end].to_string())
}

impl<'a> Body<'a> {
    fn prov(&mut self, name: &str) -> Operand {
        let n = self.names.len() as u32;
        let ix = *self.name_ix.entry(name.to_string()).or_insert(n);
        if ix == n { self.names.push(name.to_string()); }
        Operand::Value(ValueId::new(ix as usize))
    }

    fn define(&mut self, name: &str, d: Def) -> Result<Operand, String> {
        let p = self.prov(name);
        let Operand::Value(v) = p else { unreachable!() };
        if self.defs.insert(v.index() as u32, d).is_some() { return Err(format!("%{name} defined twice")); }
        Ok(p)
    }

    /// Append an instruction; a result gets `res` as its name (or a fresh one).
    fn emit(&mut self, res: Option<String>, kind: InstKind, rty: Option<Type>) -> Result<Option<Operand>, String> {
        let (_, v) = self.f.append_inst(self.cur, kind, rty);
        match v {
            Some(v) => {
                let name = res.unwrap_or_else(|| { self.fresh += 1; format!("\0t{}", self.fresh) });
                Ok(Some(self.define(&name, Def::Real(v))?))
            }
            None => Ok(None),
        }
    }

    fn block(&self, name: &str) -> Result<BlockId, String> {
        self.blocks.get(name).copied().ok_or_else(|| format!("unknown label %{name}"))
    }

    fn label(&mut self, c: &mut Cur) -> Result<BlockId, String> {
        if !c.kw("label") { return Err("expected `label`".into()); }
        let n = c.name('%')?;
        self.block(&n)
    }

    fn value(&mut self, c: &mut Cur, ty: Type) -> Result<Operand, String> {
        c.ws();
        match c.peek() {
            Some('%') => { let n = c.name('%')?; Ok(self.prov(&n)) }
            Some('@') => {
                let n = c.name('@')?;
                if let Some(&f) = self.cx.funcs.get(&n) { Ok(Operand::Func(f)) }
                else if let Some(&g) = self.cx.globals.get(&n) { Ok(Operand::Global(g)) }
                else { Err(format!("unknown symbol @{n}")) }
            }
            Some(ch) if ch.is_ascii_digit() || ch == '-' => match ty {
                Type::Int(bits) => Ok(Operand::int(bits, c.int()? & mask128(bits))),
                Type::F32 => Ok(Operand::Const(Constant::F32(c.float_bits(true)? as u32))),
                Type::F64 => Ok(Operand::Const(Constant::F64(c.float_bits(false)?))),
                t => Err(format!("numeric literal for non-numeric type {t:?}")),
            },
            _ => {
                let w = c.ident()?;
                match w {
                    "true" => Ok(Operand::int(1, 1)),
                    "false" => Ok(Operand::int(1, 0)),
                    "null" => Ok(Operand::Const(Constant::Null)),
                    "undef" => Ok(Operand::Const(Constant::Undef(ty))),
                    "poison" => Ok(Operand::Const(Constant::Poison(ty))),
                    "zeroinitializer" => match ty {
                        Type::Int(b) => Ok(Operand::int(b, 0)),
                        Type::Ptr(_) => Ok(Operand::Const(Constant::Null)),
                        Type::F32 => Ok(Operand::Const(Constant::F32(0))),
                        Type::F64 => Ok(Operand::Const(Constant::F64(0))),
                        _ => Err("aggregate zeroinitializer operands are not supported".into()),
                    },
                    "getelementptr" => {
                        while c.peek_ident().map_or(false, |w| FLAGS.contains(&w)) { c.ident()?; }
                        c.expect("(")?;
                        let (base_ty, ptr, indices) = self.gep_args(c)?;
                        c.expect(")")?;
                        let r = self.emit(None, InstKind::Gep { base_ty, ptr, indices, inbounds: true }, Some(Type::PTR))?;
                        Ok(r.unwrap())
                    }
                    w if castop(w).is_some() && c.peek() == Some('(') => {
                        c.expect("(")?;
                        let (_, v) = self.typed(c)?;
                        if c.ident()? != "to" { return Err("expected `to` in constant cast".into()); }
                        let to = self.cx.ty(c)?;
                        c.expect(")")?;
                        let r = self.emit(None, InstKind::Cast { op: castop(w).unwrap(), val: v, to }, Some(to))?;
                        Ok(r.unwrap())
                    }
                    other => Err(format!("unsupported operand `{other}`")),
                }
            }
        }
    }

    /// A zero/undef/poison constant for a scalar leaf type.
    fn leaf_const(kind: &str, ty: Type) -> Operand {
        match kind {
            "undef" => Operand::Const(Constant::Undef(ty)),
            "poison" => Operand::Const(Constant::Poison(ty)),
            _ => match ty {
                Type::Int(b) => Operand::int(b, 0),
                Type::F32 => Operand::Const(Constant::F32(0)),
                Type::F64 => Operand::Const(Constant::F64(0)),
                _ => Operand::Const(Constant::Null),
            },
        }
    }

    /// A first-class aggregate value as a list of scalar leaf operands.
    fn agg_value(&mut self, c: &mut Cur, ty: Type) -> Result<Vec<Operand>, String> {
        let leaves = self.cx.leaves(ty)?;
        if c.peek() == Some('%') {
            let n = c.name('%')?;
            let Operand::Value(v) = self.prov(&n) else { unreachable!() };
            return match self.defs.get(&(v.index() as u32)) {
                Some(Def::Tuple(t)) => Ok(t.clone()),
                _ => Err(format!("aggregate value %{n} is not available here (aggregate phis and forward references are not supported)")),
            };
        }
        let w = c.ident()?;
        if !matches!(w, "zeroinitializer" | "undef" | "poison") {
            return Err(format!("unsupported aggregate constant `{w}`"));
        }
        Ok(leaves.iter().map(|l| Self::leaf_const(w, l.ty)).collect())
    }

    /// Address of the byte at `off` from `ptr`.
    fn offset_ptr(&mut self, ptr: Operand, off: u64) -> Result<Operand, String> {
        if off == 0 { return Ok(ptr); }
        let r = self.emit(None, InstKind::Gep { base_ty: Type::I8, ptr, indices: vec![Operand::int(64, off as u128)], inbounds: true }, Some(Type::PTR))?;
        Ok(r.unwrap())
    }

    /// (first leaf, count) of the sub-aggregate at `indices` within `ty`.
    fn leaf_range(&self, ty: Type, indices: &[u32]) -> Result<(usize, usize), String> {
        let leaves = self.cx.leaves(ty)?;
        let hits: Vec<usize> = leaves.iter().enumerate().filter(|(_, l)| l.path.starts_with(indices)).map(|(i, _)| i).collect();
        match (hits.first(), hits.last()) {
            (Some(&a), Some(&b)) => Ok((a, b - a + 1)),
            _ => Err("aggregate index out of range".into()),
        }
    }

    fn typed(&mut self, c: &mut Cur) -> Result<(Type, Operand), String> {
        let t = self.cx.ty(c)?;
        skip_attrs(c)?;
        let v = self.value(c, t)?;
        Ok((t, v))
    }

    fn gep_args(&mut self, c: &mut Cur) -> Result<(Type, Operand, Vec<Operand>), String> {
        let base_ty = self.cx.ty(c)?;
        c.expect(",")?;
        let (_, ptr) = self.typed(c)?;
        let mut idx = Vec::new();
        while c.eat(",") {
            if c.kw("inrange") { if c.peek() == Some('(') { skip_attrs_paren(c)?; } }
            let (_, i) = self.typed(c)?;
            idx.push(i);
        }
        Ok((base_ty, ptr, idx))
    }

    fn skip_flags(&mut self, c: &mut Cur) {
        while c.peek_ident().map_or(false, |w| FLAGS.contains(&w)) { let _ = c.ident(); }
    }

    fn align(&mut self, c: &mut Cur, default: Type) -> Result<u32, String> {
        let dflt = size_align(&self.cx.module, default).map(|(_, a)| a as u32).unwrap_or(1);
        let save = c.p;
        if c.eat(",") && c.kw("align") { return Ok(c.int()? as u32); }
        c.p = save;
        Ok(dflt)
    }

    fn inst(&mut self, c: &mut Cur, res: Option<String>, mn: &str) -> Result<(), String> {
        if let Some(op) = binop(mn) {
            self.skip_flags(c);
            let ty = self.cx.ty(c)?;
            let lhs = self.value(c, ty)?;
            c.expect(",")?;
            let rhs = self.value(c, ty)?;
            self.emit(res, InstKind::Binary { op, lhs, rhs }, Some(ty))?;
            return Ok(());
        }
        if let Some(op) = castop(mn) {
            self.skip_flags(c);
            let (_, val) = self.typed(c)?;
            if c.ident()? != "to" { return Err("expected `to`".into()); }
            let to = self.cx.ty(c)?;
            self.emit(res, InstKind::Cast { op, val, to }, Some(to))?;
            return Ok(());
        }
        match mn {
            "icmp" => {
                self.skip_flags(c);
                let p = c.ident()?;
                let pred = ipred(p).ok_or_else(|| format!("unknown icmp predicate `{p}`"))?;
                let ty = self.cx.ty(c)?;
                let lhs = self.value(c, ty)?;
                c.expect(",")?;
                let rhs = self.value(c, ty)?;
                self.emit(res, InstKind::ICmp { pred, lhs, rhs }, Some(Type::I1))?;
            }
            "fcmp" => {
                self.skip_flags(c);
                let p = c.ident()?;
                let pred = fpred(p).ok_or_else(|| format!("unsupported fcmp predicate `{p}`"))?;
                let ty = self.cx.ty(c)?;
                let lhs = self.value(c, ty)?;
                c.expect(",")?;
                let rhs = self.value(c, ty)?;
                self.emit(res, InstKind::FCmp { pred, lhs, rhs }, Some(Type::I1))?;
            }
            "select" => {
                self.skip_flags(c);
                let (_, cond) = self.typed(c)?;
                c.expect(",")?;
                let ty = self.cx.ty(c)?;
                if let Type::Agg(_) = ty {
                    let a = self.agg_value(c, ty)?;
                    c.expect(",")?;
                    self.cx.ty(c)?;
                    let b = self.agg_value(c, ty)?;
                    let mut outs = Vec::new();
                    for ((l, x), y) in self.cx.leaves(ty)?.iter().zip(a).zip(b) {
                        outs.push(self.emit(None, InstKind::Select { cond, then_val: x, else_val: y }, Some(l.ty))?.unwrap());
                    }
                    let name = res.ok_or("select needs a result")?;
                    self.define(&name, Def::Tuple(outs))?;
                } else {
                    let then_val = self.value(c, ty)?;
                    c.expect(",")?;
                    self.cx.ty(c)?;
                    let else_val = self.value(c, ty)?;
                    self.emit(res, InstKind::Select { cond, then_val, else_val }, Some(ty))?;
                }
            }
            "fneg" => {
                self.skip_flags(c);
                let (ty, v) = self.typed(c)?;
                let neg_zero = match ty {
                    Type::F32 => Constant::F32(0x8000_0000),
                    Type::F64 => Constant::F64(0x8000_0000_0000_0000),
                    t => return Err(format!("fneg on non-float type {t:?}")),
                };
                self.emit(res, InstKind::Binary { op: BinOp::FSub, lhs: Operand::Const(neg_zero), rhs: v }, Some(ty))?;
            }
            "freeze" => {
                let ty = self.cx.ty(c)?;
                let name = res.ok_or("freeze needs a result")?;
                if let Type::Agg(_) = ty {
                    let t = self.agg_value(c, ty)?;
                    self.define(&name, Def::Tuple(t))?;
                } else {
                    let v = self.value(c, ty)?;
                    self.define(&name, Def::Alias(v))?;
                }
            }
            "phi" => {
                self.skip_flags(c);
                let ty = self.cx.ty(c)?;
                if let Type::Agg(_) = ty { return Err("aggregate-typed phi nodes are not supported".into()); }
                let mut incoming = Vec::new();
                loop {
                    c.expect("[")?;
                    let v = self.value(c, ty)?;
                    c.expect(",")?;
                    let b = c.name('%')?;
                    c.expect("]")?;
                    incoming.push((self.block(&b)?, v));
                    if !c.eat(",") { break; }
                    if c.peek() == Some('!') { break; }
                }
                self.emit(res, InstKind::Phi { ty, incoming }, Some(ty))?;
            }
            "alloca" => {
                c.kw("inalloca");
                let ty = self.cx.ty(c)?;
                let mut count = Operand::int(32, 1);
                let mut align = None;
                while c.eat(",") {
                    if c.kw("align") { align = Some(c.int()? as u32); }
                    else if c.kw("addrspace") { skip_attrs_paren(c)?; }
                    else if c.peek() == Some('!') { break; }
                    else { count = self.typed(c)?.1; }
                }
                let align = match align { Some(a) => a, None => size_align(&self.cx.module, ty)?.1 as u32 };
                self.emit(res, InstKind::Alloca { ty, count, align }, Some(Type::PTR))?;
            }
            "load" => {
                let volatile = c.kw("volatile");
                if c.kw("atomic") { return Err("atomic loads are not supported".into()); }
                let ty = self.cx.ty(c)?;
                c.expect(",")?;
                let (_, ptr) = self.typed(c)?;
                if let Type::Agg(_) = ty {
                    // Aggregate load: one scalar load per leaf.
                    let mut outs = Vec::new();
                    for l in self.cx.leaves(ty)? {
                        let addr = self.offset_ptr(ptr, l.off)?;
                        let align = size_align(&self.cx.module, l.ty)?.1 as u32;
                        outs.push(self.emit(None, InstKind::Load { ty: l.ty, ptr: addr, align, volatile }, Some(l.ty))?.unwrap());
                    }
                    let name = res.ok_or("aggregate load needs a result")?;
                    self.define(&name, Def::Tuple(outs))?;
                    return Ok(());
                }
                let align = self.align(c, ty)?;
                self.emit(res, InstKind::Load { ty, ptr, align, volatile }, Some(ty))?;
            }
            "store" => {
                let volatile = c.kw("volatile");
                if c.kw("atomic") { return Err("atomic stores are not supported".into()); }
                let ty = self.cx.ty(c)?;
                if let Type::Agg(_) = ty {
                    // Aggregate store: one scalar store per leaf.
                    let vals = self.agg_value(c, ty)?;
                    c.expect(",")?;
                    let (_, ptr) = self.typed(c)?;
                    for (l, v) in self.cx.leaves(ty)?.iter().zip(vals) {
                        let addr = self.offset_ptr(ptr, l.off)?;
                        let align = size_align(&self.cx.module, l.ty)?.1 as u32;
                        self.emit(None, InstKind::Store { val: v, ptr: addr, align, volatile }, None)?;
                    }
                    return Ok(());
                }
                skip_attrs(c)?;
                let val = self.value(c, ty)?;
                c.expect(",")?;
                let (_, ptr) = self.typed(c)?;
                let align = self.align(c, ty)?;
                self.emit(None, InstKind::Store { val, ptr, align, volatile }, None)?;
            }
            "getelementptr" => {
                let mut inbounds = false;
                while c.peek_ident().map_or(false, |w| FLAGS.contains(&w)) { inbounds |= c.ident()? == "inbounds"; }
                let (base_ty, ptr, indices) = self.gep_args(c)?;
                self.emit(res, InstKind::Gep { base_ty, ptr, indices, inbounds }, Some(Type::PTR))?;
            }
            "extractvalue" => {
                let ty = self.cx.ty(c)?;
                let vals = self.agg_value(c, ty)?;
                let mut indices = Vec::new();
                while c.eat(",") { if c.peek() == Some('!') { break; } indices.push(c.int()? as u32); }
                let (start, count) = self.leaf_range(ty, &indices)?;
                let rty = self.cx.type_at(ty, &indices)?;
                let name = res.ok_or("extractvalue needs a result")?;
                if let Type::Agg(_) = rty {
                    self.define(&name, Def::Tuple(vals[start..start + count].to_vec()))?;
                } else {
                    self.define(&name, Def::Alias(vals[start]))?;
                }
            }
            "insertvalue" => {
                let ty = self.cx.ty(c)?;
                let mut vals = self.agg_value(c, ty)?;
                c.expect(",")?;
                let vty = self.cx.ty(c)?;
                let sub = if let Type::Agg(_) = vty { self.agg_value(c, vty)? } else { vec![self.value(c, vty)?] };
                let mut indices = Vec::new();
                while c.eat(",") { if c.peek() == Some('!') { break; } indices.push(c.int()? as u32); }
                let (start, count) = self.leaf_range(ty, &indices)?;
                if sub.len() != count { return Err("insertvalue operand does not match the member it replaces".into()); }
                vals.splice(start..start + count, sub);
                let name = res.ok_or("insertvalue needs a result")?;
                self.define(&name, Def::Tuple(vals))?;
            }
            "call" => self.call(c, res)?,
            "br" => {
                let t = if c.peek_ident() == Some("label") {
                    Terminator::Br(self.label(c)?)
                } else {
                    let (_, cond) = self.typed(c)?;
                    c.expect(",")?;
                    let then_bb = self.label(c)?;
                    c.expect(",")?;
                    let else_bb = self.label(c)?;
                    Terminator::CondBr { cond, then_bb, else_bb }
                };
                self.f.set_terminator(self.cur, t);
            }
            "switch" => {
                let (ty, val) = self.typed(c)?;
                c.expect(",")?;
                let default = self.label(c)?;
                c.expect("[")?;
                let mut cases = Vec::new();
                while !c.eat("]") {
                    let (cty, cv) = self.typed(c)?;
                    let Operand::Const(Constant::Int { value, .. }) = cv else { return Err("switch case must be a constant".into()) };
                    let _ = cty;
                    c.expect(",")?;
                    cases.push((value, self.label(c)?));
                }
                let _ = ty;
                self.f.set_terminator(self.cur, Terminator::Switch { val, default, cases });
            }
            "ret" => {
                let t = if c.kw("void") { Terminator::Ret(None) } else {
                    let ty = self.cx.ty(c)?;
                    if let Type::Agg(_) = ty {
                        // Rebuild the aggregate value the backend returns in registers.
                        let vals = self.agg_value(c, ty)?;
                        let mut agg = Operand::Const(Constant::Undef(ty));
                        for (l, v) in self.cx.leaves(ty)?.iter().zip(vals) {
                            agg = self.emit(None, InstKind::InsertValue { agg, val: v, indices: l.path.clone() }, Some(ty))?.unwrap();
                        }
                        Terminator::Ret(Some(agg))
                    } else {
                        Terminator::Ret(Some(self.value(c, ty)?))
                    }
                };
                self.f.set_terminator(self.cur, t);
            }
            "unreachable" => self.f.set_terminator(self.cur, Terminator::Unreachable),
            other => return Err(format!("unsupported instruction `{other}`")),
        }
        Ok(())
    }

    fn call(&mut self, c: &mut Cur, res: Option<String>) -> Result<(), String> {
        skip_attrs(c)?;
        let ret = self.cx.ty(c)?;
        if c.peek() == Some('(') {
            // Explicit function type (variadic callee): `i32 (ptr, ...)`.
            c.expect("(")?;
            if !c.eat(")") {
                loop {
                    if !c.eat("...") { self.cx.ty(c)?; skip_attrs(c)?; }
                    if !c.eat(",") { break; }
                }
                c.expect(")")?;
            }
        }
        c.ws();
        let callee_name = if c.peek() == Some('@') { Some(c.name('@')?) } else { None };
        let indirect = if callee_name.is_none() { Some(self.value(c, Type::PTR)?) } else { None };
        c.expect("(")?;
        let mut args: Vec<(Type, Operand)> = Vec::new();
        if !c.eat(")") {
            loop {
                let t = self.cx.ty(c)?;
                skip_attrs(c)?;
                if let Type::Agg(_) = t {
                    // First-class aggregate arguments are passed as their scalar leaves.
                    let vals = self.agg_value(c, t)?;
                    for (l, v) in self.cx.leaves(t)?.iter().zip(vals) { args.push((l.ty, v)); }
                } else {
                    let v = self.value(c, t)?;
                    args.push((t, v));
                }
                if !c.eat(",") { break; }
            }
            c.expect(")")?;
        }
        let rty = if ret == Type::Void { None } else { Some(ret) };

        if let Some(name) = &callee_name {
            if name.starts_with("llvm.") { return self.intrinsic(name, ret, &args, res); }
        }
        let callee = match (callee_name, indirect) {
            (Some(n), _) => Callee::Direct(*self.cx.funcs.get(&n).ok_or_else(|| format!("call to undeclared function @{n}"))?),
            (None, Some(o)) => Callee::Indirect(o),
            _ => unreachable!(),
        };
        let args = args.into_iter().map(|(_, v)| v).collect();
        if let Type::Agg(_) = ret {
            // Aggregate result: the call returns it in registers; split it into leaves.
            let agg = self.emit(None, InstKind::Call { callee, args, ret_ty: ret }, Some(ret))?.unwrap();
            let mut outs = Vec::new();
            for l in self.cx.leaves(ret)? {
                outs.push(self.emit(None, InstKind::ExtractValue { agg, indices: l.path.clone() }, Some(l.ty))?.unwrap());
            }
            let name = res.ok_or("aggregate call result must be named")?;
            self.define(&name, Def::Tuple(outs))?;
            return Ok(());
        }
        self.emit(res, InstKind::Call { callee, args, ret_ty: ret }, rty)?;
        Ok(())
    }

    fn libc(&mut self, name: &str, params: Vec<Type>, ret: Type) -> FuncId {
        self.cx.ensure_func(name, Signature::new(params, ret))
    }

    fn intrinsic(&mut self, name: &str, ret: Type, args: &[(Type, Operand)], res: Option<String>) -> Result<(), String> {
        let has = |p: &str| name.starts_with(p);
        if has("llvm.lifetime.") || has("llvm.dbg.") || has("llvm.assume") || has("llvm.invariant.")
            || has("llvm.donothing") || has("llvm.experimental.noalias") || has("llvm.prefetch") {
            return Ok(());
        }
        if has("llvm.va_start") {
            self.emit(None, InstKind::VaStart { list: args[0].1 }, None)?;
            return Ok(());
        }
        if has("llvm.va_end") { return Ok(()); }
        if has("llvm.va_copy") {
            self.emit(None, InstKind::VaCopy { dst: args[0].1, src: args[1].1 }, None)?;
            return Ok(());
        }
        if has("llvm.memcpy.") || has("llvm.memmove.") {
            let f = self.libc(if has("llvm.memcpy.") { "memcpy" } else { "memmove" }, vec![Type::PTR, Type::PTR, Type::I64], Type::PTR);
            let a = vec![args[0].1, args[1].1, args[2].1];
            self.emit(None, InstKind::Call { callee: Callee::Direct(f), args: a, ret_ty: Type::PTR }, Some(Type::PTR))?;
            return Ok(());
        }
        if has("llvm.memset.") {
            let f = self.libc("memset", vec![Type::PTR, Type::I32, Type::I64], Type::PTR);
            let v = self.emit(None, InstKind::Cast { op: CastOp::ZExt, val: args[1].1, to: Type::I32 }, Some(Type::I32))?.unwrap();
            let a = vec![args[0].1, v, args[2].1];
            self.emit(None, InstKind::Call { callee: Callee::Direct(f), args: a, ret_ty: Type::PTR }, Some(Type::PTR))?;
            return Ok(());
        }
        if has("llvm.expect.") {
            let r = res.ok_or("llvm.expect needs a result")?;
            self.define(&r, Def::Alias(args[0].1))?;
            return Ok(());
        }
        let minmax = [("llvm.smax.", IntPred::Sgt), ("llvm.smin.", IntPred::Slt), ("llvm.umax.", IntPred::Ugt), ("llvm.umin.", IntPred::Ult)];
        for (p, pred) in minmax {
            if has(p) {
                let (x, y) = (args[0].1, args[1].1);
                let cond = self.emit(None, InstKind::ICmp { pred, lhs: x, rhs: y }, Some(Type::I1))?.unwrap();
                self.emit(res, InstKind::Select { cond, then_val: x, else_val: y }, Some(ret))?;
                return Ok(());
            }
        }
        if has("llvm.abs.") {
            let (t, x) = args[0];
            let Type::Int(b) = t else { return Err("llvm.abs on non-integer".into()) };
            let neg = self.emit(None, InstKind::Binary { op: BinOp::Sub, lhs: Operand::int(b, 0), rhs: x }, Some(t))?.unwrap();
            let isneg = self.emit(None, InstKind::ICmp { pred: IntPred::Slt, lhs: x, rhs: Operand::int(b, 0) }, Some(Type::I1))?.unwrap();
            self.emit(res, InstKind::Select { cond: isneg, then_val: neg, else_val: x }, Some(t))?;
            return Ok(());
        }
        if let Some((base, suffix)) = name["llvm.".len()..].rsplit_once('.') {
            let fty = match suffix { "f64" => Some(Type::F64), "f32" => Some(Type::F32), _ => None };
            if let Some(t) = fty {
                let a: Vec<Operand> = args.iter().map(|x| x.1).collect();
                let bits = if t == Type::F64 { 64 } else { 32 };
                match base {
                    "fabs" => {
                        let it = Type::Int(bits);
                        let i = self.emit(None, InstKind::Cast { op: CastOp::Bitcast, val: a[0], to: it }, Some(it))?.unwrap();
                        let m = self.bin(BinOp::And, it, i, Operand::int(bits, mask128(bits) >> 1))?;
                        self.emit(res, InstKind::Cast { op: CastOp::Bitcast, val: m, to: t }, Some(t))?;
                        return Ok(());
                    }
                    "fmuladd" => {
                        let p = self.bin(BinOp::FMul, t, a[0], a[1])?;
                        self.emit(res, InstKind::Binary { op: BinOp::FAdd, lhs: p, rhs: a[2] }, Some(t))?;
                        return Ok(());
                    }
                    _ => {}
                }
                let libm = match base {
                    "sqrt" | "floor" | "ceil" | "trunc" | "rint" | "nearbyint" | "round" | "exp" | "exp2" | "log"
                    | "log2" | "log10" | "sin" | "cos" | "pow" | "copysign" | "fma" => Some(base),
                    "minnum" => Some("fmin"),
                    "maxnum" => Some("fmax"),
                    _ => None,
                };
                if let Some(l) = libm {
                    let cname = if t == Type::F32 { format!("{l}f") } else { l.to_string() };
                    let f = self.cx.ensure_func(&cname, Signature::new(vec![t; a.len()], t));
                    self.emit(res, InstKind::Call { callee: Callee::Direct(f), args: a, ret_ty: t }, Some(t))?;
                    return Ok(());
                }
            }
        }
        // Saturating arithmetic.
        for (prefix, kind) in [("llvm.uadd.sat.", 0u8), ("llvm.usub.sat.", 1), ("llvm.sadd.sat.", 3), ("llvm.ssub.sat.", 4)] {
            if has(prefix) {
                let (t, a) = args[0];
                let b = args[1].1;
                let Type::Int(n) = t else { return Err(format!("@{name} on non-integer")) };
                let (r, ov) = self.with_overflow(kind, t, n, a, b)?;
                let sat = match kind {
                    0 => Operand::int(n, mask128(n)),
                    1 => Operand::int(n, 0),
                    _ => {
                        // Signed overflow saturates toward the sign of the left operand.
                        let neg = self.icmp(IntPred::Slt, a, Operand::int(n, 0))?;
                        self.select(t, neg, Operand::int(n, 1u128 << (n - 1)), Operand::int(n, mask128(n) >> 1))?
                    }
                };
                self.emit(res, InstKind::Select { cond: ov, then_val: sat, else_val: r }, Some(t))?;
                return Ok(());
            }
        }
        for (prefix, kind) in [("llvm.uadd.with.overflow.", 0), ("llvm.usub.with.overflow.", 1), ("llvm.umul.with.overflow.", 2),
                               ("llvm.sadd.with.overflow.", 3), ("llvm.ssub.with.overflow.", 4), ("llvm.smul.with.overflow.", 5)] {
            if has(prefix) {
                let (t, a) = args[0];
                let b = args[1].1;
                let Type::Int(n) = t else { return Err(format!("@{name} on non-integer")) };
                let (r, ov) = self.with_overflow(kind, t, n, a, b)?;
                let name = res.ok_or("with.overflow needs a result")?;
                self.define(&name, Def::Tuple(vec![r, ov]))?;
                return Ok(());
            }
        }
        if has("llvm.ctpop.") || has("llvm.ctlz.") || has("llvm.cttz.") || has("llvm.bswap.")
            || has("llvm.fshl.") || has("llvm.fshr.") {
            let (t, x) = args[0];
            let Type::Int(n) = t else { return Err(format!("@{name} on non-integer")) };
            if !matches!(n, 8 | 16 | 32 | 64) { return Err(format!("@{name}: unsupported width i{n}")); }
            let out = if has("llvm.ctpop.") { self.ctpop(t, n, x)? }
            else if has("llvm.ctlz.") {
                let mut v = x;
                let mut sh = 1;
                while sh < n {
                    let t1 = self.bin(BinOp::LShr, t, v, Operand::int(n, sh as u128))?;
                    v = self.bin(BinOp::Or, t, v, t1)?;
                    sh *= 2;
                }
                let pc = self.ctpop(t, n, v)?;
                self.bin(BinOp::Sub, t, Operand::int(n, n as u128), pc)?
            } else if has("llvm.cttz.") {
                let neg = self.bin(BinOp::Sub, t, Operand::int(n, 0), x)?;
                let low = self.bin(BinOp::And, t, x, neg)?;
                let m1 = self.bin(BinOp::Sub, t, low, Operand::int(n, 1))?;
                self.ctpop(t, n, m1)?
            } else if has("llvm.bswap.") {
                if n == 8 { x } else {
                    let mut acc = Operand::int(n, 0);
                    for i in 0..n / 8 {
                        let sh = self.bin(BinOp::LShr, t, x, Operand::int(n, (8 * i) as u128))?;
                        let byte = self.bin(BinOp::And, t, sh, Operand::int(n, 0xff))?;
                        let up = self.bin(BinOp::Shl, t, byte, Operand::int(n, (n - 8 - 8 * i) as u128))?;
                        acc = self.bin(BinOp::Or, t, acc, up)?;
                    }
                    acc
                }
            } else {
                // Funnel shifts, written so no shift amount ever reaches the bit width.
                let (a, b, s) = (args[0].1, args[1].1, args[2].1);
                let s = self.bin(BinOp::And, t, s, Operand::int(n, (n - 1) as u128))?;
                let rest = self.bin(BinOp::Sub, t, Operand::int(n, (n - 1) as u128), s)?;
                if has("llvm.fshl.") {
                    let l = self.bin(BinOp::Shl, t, a, s)?;
                    let r1 = self.bin(BinOp::LShr, t, b, Operand::int(n, 1))?;
                    let r = self.bin(BinOp::LShr, t, r1, rest)?;
                    self.bin(BinOp::Or, t, l, r)?
                } else {
                    let l1 = self.bin(BinOp::Shl, t, a, Operand::int(n, 1))?;
                    let l = self.bin(BinOp::Shl, t, l1, rest)?;
                    let r = self.bin(BinOp::LShr, t, b, s)?;
                    self.bin(BinOp::Or, t, l, r)?
                }
            };
            if let Some(r) = res { self.define(&r, Def::Alias(out))?; }
            return Ok(());
        }
        Err(format!("unsupported intrinsic @{name}"))
    }

    fn icmp(&mut self, pred: IntPred, l: Operand, r: Operand) -> Result<Operand, String> {
        Ok(self.emit(None, InstKind::ICmp { pred, lhs: l, rhs: r }, Some(Type::I1))?.unwrap())
    }

    fn select(&mut self, ty: Type, cond: Operand, t: Operand, e: Operand) -> Result<Operand, String> {
        Ok(self.emit(None, InstKind::Select { cond, then_val: t, else_val: e }, Some(ty))?.unwrap())
    }

    /// (result, overflow flag) for `{u,s}{add,sub,mul}.with.overflow`.
    fn with_overflow(&mut self, kind: u8, t: Type, n: u32, a: Operand, b: Operand) -> Result<(Operand, Operand), String> {
        let (zero, one, all1) = (Operand::int(n, 0), Operand::int(n, 1), Operand::int(n, mask128(n)));
        let i1 = Type::I1;
        Ok(match kind {
            0 => { let r = self.bin(BinOp::Add, t, a, b)?; let ov = self.icmp(IntPred::Ult, r, a)?; (r, ov) }
            1 => { let r = self.bin(BinOp::Sub, t, a, b)?; let ov = self.icmp(IntPred::Ult, a, b)?; (r, ov) }
            2 => {
                let r = self.bin(BinOp::Mul, t, a, b)?;
                let az = self.icmp(IntPred::Eq, a, zero)?;
                let a1 = self.select(t, az, one, a)?;
                let q = self.bin(BinOp::UDiv, t, r, a1)?;
                let qne = self.icmp(IntPred::Ne, q, b)?;
                let nz = self.icmp(IntPred::Ne, a, zero)?;
                (r, self.bin(BinOp::And, i1, nz, qne)?)
            }
            3 | 4 => {
                let (op, x1, x2) = if kind == 3 {
                    let r = self.bin(BinOp::Add, t, a, b)?;
                    (r, self.bin(BinOp::Xor, t, a, r)?, self.bin(BinOp::Xor, t, b, r)?)
                } else {
                    let r = self.bin(BinOp::Sub, t, a, b)?;
                    (r, self.bin(BinOp::Xor, t, a, b)?, self.bin(BinOp::Xor, t, a, r)?)
                };
                let x3 = self.bin(BinOp::And, t, x1, x2)?;
                (op, self.icmp(IntPred::Slt, x3, zero)?)
            }
            _ => {
                let r = self.bin(BinOp::Mul, t, a, b)?;
                let a0 = self.icmp(IntPred::Eq, a, zero)?;
                let am1 = self.icmp(IntPred::Eq, a, all1)?;
                let skip = self.bin(BinOp::Or, i1, a0, am1)?;
                let a1 = self.select(t, skip, one, a)?;
                let q = self.bin(BinOp::SDiv, t, r, a1)?;
                let qne = self.icmp(IntPred::Ne, q, b)?;
                let nz = self.icmp(IntPred::Ne, a, zero)?;
                let nm1 = self.icmp(IntPred::Ne, a, all1)?;
                let c1 = self.bin(BinOp::And, i1, nz, nm1)?;
                let chk = self.bin(BinOp::And, i1, c1, qne)?;
                let bmin = self.icmp(IntPred::Eq, b, Operand::int(n, 1u128 << (n - 1)))?;
                let both = self.bin(BinOp::And, i1, am1, bmin)?;
                (r, self.bin(BinOp::Or, i1, chk, both)?)
            }
        })
    }

    fn bin(&mut self, op: BinOp, ty: Type, lhs: Operand, rhs: Operand) -> Result<Operand, String> {
        Ok(self.emit(None, InstKind::Binary { op, lhs, rhs }, Some(ty))?.unwrap())
    }

    /// Population count via the classic SWAR sequence.
    fn ctpop(&mut self, t: Type, n: u32, x: Operand) -> Result<Operand, String> {
        let rep = |byte: u128| Operand::int(n, byte * (mask128(n) / 0xff));
        let s1 = self.bin(BinOp::LShr, t, x, Operand::int(n, 1))?;
        let s1 = self.bin(BinOp::And, t, s1, rep(0x55))?;
        let x1 = self.bin(BinOp::Sub, t, x, s1)?;
        let a = self.bin(BinOp::And, t, x1, rep(0x33))?;
        let s2 = self.bin(BinOp::LShr, t, x1, Operand::int(n, 2))?;
        let s2 = self.bin(BinOp::And, t, s2, rep(0x33))?;
        let x2 = self.bin(BinOp::Add, t, a, s2)?;
        let s3 = self.bin(BinOp::LShr, t, x2, Operand::int(n, 4))?;
        let s3 = self.bin(BinOp::Add, t, x2, s3)?;
        let x3 = self.bin(BinOp::And, t, s3, rep(0x0f))?;
        if n == 8 { return Ok(x3); }
        let m = self.bin(BinOp::Mul, t, x3, rep(0x01))?;
        self.bin(BinOp::LShr, t, m, Operand::int(n, (n - 8) as u128))
    }
}

fn skip_attrs_paren(c: &mut Cur) -> Result<(), String> {
    c.expect("(")?;
    let mut depth = 1;
    while depth > 0 {
        match c.rest().chars().next() {
            Some('(') => depth += 1,
            Some(')') => depth -= 1,
            None => return Err("unbalanced parentheses".into()),
            _ => {}
        }
        c.p += 1;
    }
    Ok(())
}

pub(crate) fn build_body(cx: &mut Ctx, fid: FuncId, groups: &[ParamGroup], lines: &[&str], start: usize, end: usize) -> Result<(), LlError> {
    let err = |ln: usize, m: String| LlError { line: ln + 1, message: m };
    let f = cx.module.functions[fid].clone();
    let fname = f.name.clone();
    let mut b = Body { cx, f, names: vec![], name_ix: HashMap::new(), defs: HashMap::new(), blocks: HashMap::new(), cur: BlockId::new(0), fresh: 0 };

    let pnames: Vec<String> = groups.iter().map(|g| g.name.clone()).collect();
    for g in groups {
        if g.agg.is_some() {
            // A first-class aggregate parameter arrives as `count` scalar parameters.
            let mut leaves = Vec::new();
            for i in 0..g.count {
                let p = b.f.params[g.first + i];
                let synth = format!("\0param{}.{}", g.first, i);
                leaves.push(b.define(&synth, Def::Real(p)).map_err(|m| err(start, m))?);
            }
            b.define(&g.name, Def::Tuple(leaves)).map_err(|m| err(start, m))?;
        } else {
            b.define(&g.name, Def::Real(b.f.params[g.first])).map_err(|m| err(start, m))?;
        }
    }

    // Create blocks up front so branches can refer forward.
    let mut first = true;
    for ln in start..end {
        let l = lines[ln];
        if l.trim().is_empty() || l.trim_start().starts_with(';') { continue; }
        match parse_label(l) {
            Some(name) => {
                let id = b.f.create_block(Some(&name));
                b.blocks.insert(name, id);
            }
            None if first => {
                // LLVM numbers unnamed values in order: unnamed params, then the entry block.
                let n = pnames.iter().filter_map(|p| p.parse::<u32>().ok()).max().map_or(0, |m| m + 1).to_string();
                let id = b.f.create_block(Some("entry"));
                b.blocks.insert(n, id);
            }
            None => {}
        }
        first = false;
    }
    if b.f.layout.is_empty() { return Err(err(start, "function has no blocks".into())); }
    b.cur = b.f.layout[0];

    let mut ln = start;
    while ln < end {
        let raw = lines[ln];
        let t = raw.trim();
        if t.is_empty() || t.starts_with(';') { ln += 1; continue; }
        if let Some(name) = parse_label(raw) {
            b.cur = b.blocks[&name];
            ln += 1;
            continue;
        }
        // `switch` cases are printed one per line; join up to the closing `]`.
        let mut text = t.to_string();
        let first_ln = ln;
        if t.starts_with("switch ") {
            while !text.contains(']') && ln + 1 < end { ln += 1; text.push(' '); text.push_str(lines[ln].trim()); }
        }
        ln += 1;
        let mut c = Cur::new(&text);
        let res = if c.peek() == Some('%') {
            let n = c.name('%').map_err(|m| err(first_ln, m))?;
            c.expect("=").map_err(|m| err(first_ln, m))?;
            Some(n)
        } else { None };
        let run = || -> Result<(), String> {
            while matches!(c.peek_ident(), Some("tail" | "musttail" | "notail")) { c.ident()?; }
            let mn = c.ident()?;
            b.inst(&mut c, res, mn)
        };
        run().map_err(|m| err(first_ln, format!("{m} (in @{fname})")))?;
    }

    // Resolve provisional ids to real values.
    fn resolve(defs: &HashMap<u32, Def>, o: Operand) -> Result<Operand, u32> {
        match o {
            Operand::Value(v) => match defs.get(&(v.index() as u32)) {
                Some(Def::Real(r)) => Ok(Operand::Value(*r)),
                Some(Def::Alias(a)) => resolve(defs, *a),
                Some(Def::Tuple(_)) => Err(v.index() as u32),
                None => Err(v.index() as u32),
            },
            other => Ok(other),
        }
    }
    let mut missing = None;
    let defs = std::mem::take(&mut b.defs);
    b.f.for_each_operand_mut(|o| match resolve(&defs, *o) {
        Ok(n) => *o = n,
        Err(i) => { missing.get_or_insert(i); }
    });
    if let Some(i) = missing {
        return Err(err(start, format!("use of undefined (or unsupported aggregate) value %{} in @{fname}", b.names[i as usize].replace('\0', "tmp"))));
    }
    for &blk in &b.f.layout {
        if b.f.blocks[blk].terminator.is_none() {
            return Err(err(end, format!("block {} has no terminator in @{fname}", b.f.blocks[blk].name.clone().unwrap_or_default())));
        }
    }
    let f = b.f;
    b.cx.module.functions[fid] = f;
    Ok(())
}
