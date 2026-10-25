// SPDX-License-Identifier: GPL-2.0-only WITH LicenseRef-Kaiba-Runtime-Exception
// Copyright (C) The Kaiba Authors
//
// This file is part of the Kaiba compiler toolchain. It is licensed under
// GPL-2.0-only with the Kaiba Runtime/Output Exception; see COPYING and
// LICENSE-EXCEPTION at the repository root.
//! Parser for the textual `.llcr` format produced by [`crate::print`].
//!
//! The format is line-oriented. Value and block numbers in the text are
//! *labels*: they are remapped onto fresh arena ids, so gaps and out-of-order
//! numbering are accepted, and forward references (loops, phis) just work.
//! `@name` resolves to a function first, then to a global.

use crate::entity::{BlockId, Entity, FuncId, GlobalId, ValueId};
use crate::function::{Function, Linkage};
use crate::inst::*;
use crate::module::{GlobalVar, Module, Reloc, RelocTarget};
use crate::types::{AggregateType, Signature, Type};
use std::collections::HashMap;
use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParseError {
    /// 1-based line number.
    pub line: usize,
    pub message: String,
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { write!(f, "line {}: {}", self.line, self.message) }
}
impl std::error::Error for ParseError {}

const BINOPS: &[BinOp] = &[
    BinOp::Add, BinOp::Sub, BinOp::Mul, BinOp::UDiv, BinOp::SDiv, BinOp::URem, BinOp::SRem,
    BinOp::And, BinOp::Or, BinOp::Xor, BinOp::Shl, BinOp::LShr, BinOp::AShr,
    BinOp::FAdd, BinOp::FSub, BinOp::FMul, BinOp::FDiv, BinOp::FRem,
];
const IPREDS: &[IntPred] = &[
    IntPred::Eq, IntPred::Ne, IntPred::Ugt, IntPred::Uge, IntPred::Ult, IntPred::Ule,
    IntPred::Sgt, IntPred::Sge, IntPred::Slt, IntPred::Sle,
];
const FPREDS: &[FloatPred] = &[
    FloatPred::Oeq, FloatPred::Ogt, FloatPred::Oge, FloatPred::Olt, FloatPred::Ole, FloatPred::One,
    FloatPred::Ord, FloatPred::Ueq, FloatPred::Ugt, FloatPred::Uge, FloatPred::Ult, FloatPred::Ule,
    FloatPred::Une, FloatPred::Uno,
];
const CASTS: &[CastOp] = &[
    CastOp::Trunc, CastOp::ZExt, CastOp::SExt, CastOp::FPTrunc, CastOp::FPExt, CastOp::FPToUI,
    CastOp::FPToSI, CastOp::UIToFP, CastOp::SIToFP, CastOp::PtrToInt, CastOp::IntToPtr, CastOp::Bitcast,
];
const LINKAGES: &[Linkage] = &[Linkage::External, Linkage::Internal, Linkage::Private, Linkage::Declaration];

fn by_name<T: Copy + fmt::Debug>(all: &[T], name: &str) -> Option<T> {
    all.iter().copied().find(|v| format!("{v:?}").to_lowercase() == name)
}

struct Ctx {
    module: Module,
    funcs: HashMap<String, FuncId>,
    globals: HashMap<String, GlobalId>,
    blocks: HashMap<u32, BlockId>,
}

struct Cur<'a> { s: &'a str, p: usize }

impl<'a> Cur<'a> {
    fn new(s: &'a str) -> Self { Cur { s, p: 0 } }
    fn rest(&self) -> &'a str { &self.s[self.p..] }
    fn ws(&mut self) { while self.rest().starts_with(' ') { self.p += 1; } }
    fn eat(&mut self, t: &str) -> bool {
        self.ws();
        if self.rest().starts_with(t) { self.p += t.len(); true } else { false }
    }
    fn expect(&mut self, t: &str) -> Result<(), String> {
        if self.eat(t) { Ok(()) } else { Err(format!("expected `{t}` before `{}`", self.rest())) }
    }
    fn at_end(&mut self) -> bool { self.ws(); self.rest().is_empty() }
    fn take_while(&mut self, f: impl Fn(char) -> bool) -> &'a str {
        let r = self.rest();
        let n = r.find(|c: char| !f(c)).unwrap_or(r.len());
        self.p += n;
        &r[..n]
    }
    fn ident(&mut self) -> Result<&'a str, String> {
        self.ws();
        let t = self.take_while(|c| c.is_ascii_alphanumeric() || c == '_' || c == '.');
        if t.is_empty() { Err(format!("expected identifier before `{}`", self.rest())) } else { Ok(t) }
    }
    fn peek_ident(&mut self) -> Option<&'a str> {
        self.ws();
        let save = self.p;
        let r = self.ident().ok();
        self.p = save;
        r
    }
    fn kw(&mut self, k: &str) -> Result<(), String> {
        let i = self.ident()?;
        if i == k { Ok(()) } else { Err(format!("expected `{k}`, found `{i}`")) }
    }
    fn number(&mut self) -> Result<u128, String> {
        self.ws();
        if self.rest().starts_with("0x") {
            self.p += 2;
            let d = self.take_while(|c| c.is_ascii_hexdigit());
            u128::from_str_radix(d, 16).map_err(|e| format!("bad hex number: {e}"))
        } else {
            let d = self.take_while(|c| c.is_ascii_digit());
            d.parse::<u128>().map_err(|_| format!("expected number before `{}`", self.rest()))
        }
    }
    fn uint(&mut self) -> Result<u32, String> {
        u32::try_from(self.number()?).map_err(|_| "number out of range".to_string())
    }
    fn name(&mut self) -> Result<&'a str, String> {
        self.expect("@")?;
        let t = self.take_while(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '.' | '$'));
        if t.is_empty() { Err("empty symbol name".into()) } else { Ok(t) }
    }
    fn quoted(&mut self) -> Result<&'a str, String> {
        self.expect("\"")?;
        let t = self.take_while(|c| c != '"');
        self.expect("\"")?;
        Ok(t)
    }

    fn ty(&mut self, ctx: &mut Ctx) -> Result<Type, String> {
        self.ws();
        if self.eat("<{") {
            let fields = self.type_list(ctx, "}>")?;
            return Ok(ctx.module.intern_aggregate(AggregateType::Struct { fields, packed: true }));
        }
        if self.eat("{") {
            let fields = self.type_list(ctx, "}")?;
            return Ok(ctx.module.intern_aggregate(AggregateType::Struct { fields, packed: false }));
        }
        if self.eat("<") {
            let len = self.uint()?;
            self.kw("x")?;
            let elem = self.ty(ctx)?;
            self.expect(">")?;
            return Ok(ctx.module.intern_aggregate(AggregateType::Vector { elem, len }));
        }
        if self.eat("[") {
            let len = self.number()? as u64;
            self.kw("x")?;
            let elem = self.ty(ctx)?;
            self.expect("]")?;
            return Ok(ctx.module.intern_aggregate(AggregateType::Array { elem, len }));
        }
        let id = self.ident()?;
        Ok(match id {
            "void" => Type::Void,
            "f32" => Type::F32,
            "f64" => Type::F64,
            "ptr" => {
                if self.peek_ident() == Some("addrspace") {
                    self.ident()?;
                    self.expect("(")?;
                    let a = self.uint()?;
                    self.expect(")")?;
                    Type::Ptr(a)
                } else { Type::PTR }
            }
            s => match s.strip_prefix('i').and_then(|d| d.parse::<u32>().ok()) {
                Some(n) if n > 0 => Type::Int(n),
                _ => return Err(format!("unknown type `{s}`")),
            },
        })
    }

    fn type_list(&mut self, ctx: &mut Ctx, close: &str) -> Result<Vec<Type>, String> {
        let mut v = Vec::new();
        if self.eat(close) { return Ok(v); }
        loop {
            v.push(self.ty(ctx)?);
            if !self.eat(",") { break; }
        }
        self.expect(close)?;
        Ok(v)
    }

    fn block(&mut self, ctx: &Ctx) -> Result<BlockId, String> {
        self.expect("bb")?;
        let n = self.uint()?;
        ctx.blocks.get(&n).copied().ok_or_else(|| format!("unknown block bb{n}"))
    }

    fn operand(&mut self, ctx: &mut Ctx) -> Result<Operand, String> {
        self.ws();
        let r = self.rest();
        if r.starts_with("%v") {
            self.p += 2;
            let n = self.uint()?;
            return Ok(Operand::Value(ValueId::new(n as usize)));
        }
        if r.starts_with('@') {
            let n = self.name()?;
            if let Some(&f) = ctx.funcs.get(n) { return Ok(Operand::Func(f)); }
            if let Some(&g) = ctx.globals.get(n) { return Ok(Operand::Global(g)); }
            return Err(format!("unknown symbol @{n}"));
        }
        if r.starts_with(|c: char| c.is_ascii_digit()) {
            let v = self.number()?;
            self.expect(":")?;
            let t = self.ident()?;
            let bits = t.strip_prefix('i').and_then(|d| d.parse::<u32>().ok()).ok_or_else(|| format!("bad constant type `{t}`"))?;
            return Ok(Operand::int(bits, v));
        }
        match self.ident()? {
            "null" => Ok(Operand::Const(Constant::Null)),
            "undef" => { self.expect(":")?; Ok(Operand::Const(Constant::Undef(self.ty(ctx)?))) }
            "poison" => { self.expect(":")?; Ok(Operand::Const(Constant::Poison(self.ty(ctx)?))) }
            "f32" => { self.expect(":")?; let n = self.number()?; Ok(Operand::Const(Constant::F32(u32::try_from(n).map_err(|_| "f32 bits out of range")?))) }
            "f64" => { self.expect(":")?; let n = self.number()?; Ok(Operand::Const(Constant::F64(u64::try_from(n).map_err(|_| "f64 bits out of range")?))) }
            other => Err(format!("unexpected `{other}` in operand position")),
        }
    }

    fn comma(&mut self) -> Result<(), String> { self.expect(",") }

    fn index_list(&mut self) -> Result<Vec<u32>, String> {
        self.expect("[")?;
        let mut v = Vec::new();
        if self.eat("]") { return Ok(v); }
        loop {
            v.push(self.uint()?);
            if !self.eat(",") { break; }
        }
        self.expect("]")?;
        Ok(v)
    }
}

/// Returns (result type, instruction).
fn parse_inst(c: &mut Cur, ctx: &mut Ctx) -> Result<(Option<Type>, InstKind), String> {
    let mn = c.ident()?;
    if let Some(op) = by_name(BINOPS, mn) {
        let t = c.ty(ctx)?;
        let lhs = c.operand(ctx)?; c.comma()?;
        let rhs = c.operand(ctx)?;
        return Ok((Some(t), InstKind::Binary { op, lhs, rhs }));
    }
    if let Some(op) = by_name(CASTS, mn) {
        let _from = c.ty(ctx)?;
        let val = c.operand(ctx)?;
        c.kw("to")?;
        let to = c.ty(ctx)?;
        return Ok((Some(to), InstKind::Cast { op, val, to }));
    }
    Ok(match mn {
        "icmp" => {
            let pred = by_name(IPREDS, c.ident()?).ok_or("unknown icmp predicate")?;
            let _t = c.ty(ctx)?;
            let lhs = c.operand(ctx)?; c.comma()?;
            let rhs = c.operand(ctx)?;
            (Some(Type::I1), InstKind::ICmp { pred, lhs, rhs })
        }
        "fcmp" => {
            let pred = by_name(FPREDS, c.ident()?).ok_or("unknown fcmp predicate")?;
            let _t = c.ty(ctx)?;
            let lhs = c.operand(ctx)?; c.comma()?;
            let rhs = c.operand(ctx)?;
            (Some(Type::I1), InstKind::FCmp { pred, lhs, rhs })
        }
        "select" => {
            let t = c.ty(ctx)?;
            let cond = c.operand(ctx)?; c.comma()?;
            let then_val = c.operand(ctx)?; c.comma()?;
            let else_val = c.operand(ctx)?;
            (Some(t), InstKind::Select { cond, then_val, else_val })
        }
        "load" => {
            let volatile = c.peek_ident() == Some("volatile");
            if volatile { c.ident()?; }
            let ty = c.ty(ctx)?; c.comma()?;
            let ptr = c.operand(ctx)?; c.comma()?;
            c.kw("align")?;
            let align = c.uint()?;
            (Some(ty), InstKind::Load { ty, ptr, align, volatile })
        }
        "store" => {
            let volatile = c.peek_ident() == Some("volatile");
            if volatile { c.ident()?; }
            let _t = c.ty(ctx)?;
            let val = c.operand(ctx)?; c.comma()?;
            let ptr = c.operand(ctx)?; c.comma()?;
            c.kw("align")?;
            let align = c.uint()?;
            (None, InstKind::Store { val, ptr, align, volatile })
        }
        "va_start" => {
            let list = c.operand(ctx)?;
            (None, InstKind::VaStart { list })
        }
        "va_copy" => {
            let dst = c.operand(ctx)?;
            c.comma()?;
            let src = c.operand(ctx)?;
            (None, InstKind::VaCopy { dst, src })
        }
        "alloca" => {
            let ty = c.ty(ctx)?; c.comma()?;
            let count = c.operand(ctx)?; c.comma()?;
            c.kw("align")?;
            let align = c.uint()?;
            (Some(Type::PTR), InstKind::Alloca { ty, count, align })
        }
        "gep" => {
            let inbounds = c.peek_ident() == Some("inbounds");
            if inbounds { c.ident()?; }
            let base_ty = c.ty(ctx)?; c.comma()?;
            let ptr = c.operand(ctx)?; c.comma()?;
            c.expect("[")?;
            let mut indices = Vec::new();
            if !c.eat("]") {
                loop {
                    indices.push(c.operand(ctx)?);
                    if !c.eat(",") { break; }
                }
                c.expect("]")?;
            }
            (Some(Type::PTR), InstKind::Gep { base_ty, ptr, indices, inbounds })
        }
        "extractvalue" => {
            let t = c.ty(ctx)?;
            let agg = c.operand(ctx)?; c.comma()?;
            let indices = c.index_list()?;
            (Some(t), InstKind::ExtractValue { agg, indices })
        }
        "insertvalue" => {
            let t = c.ty(ctx)?;
            let agg = c.operand(ctx)?; c.comma()?;
            let val = c.operand(ctx)?; c.comma()?;
            let indices = c.index_list()?;
            (Some(t), InstKind::InsertValue { agg, val, indices })
        }
        "phi" => {
            let ty = c.ty(ctx)?;
            let mut incoming = Vec::new();
            loop {
                c.expect("[")?;
                let b = c.block(ctx)?;
                c.expect(":")?;
                let o = c.operand(ctx)?;
                c.expect("]")?;
                incoming.push((b, o));
                if !c.eat(",") { break; }
            }
            (Some(ty), InstKind::Phi { ty, incoming })
        }
        "call" => {
            let ret_ty = c.ty(ctx)?;
            let callee = match c.operand(ctx)? { Operand::Func(f) => Callee::Direct(f), o => Callee::Indirect(o) };
            c.expect("(")?;
            let mut args = Vec::new();
            if !c.eat(")") {
                loop {
                    args.push(c.operand(ctx)?);
                    if !c.eat(",") { break; }
                }
                c.expect(")")?;
            }
            ((ret_ty != Type::Void).then_some(ret_ty), InstKind::Call { callee, args, ret_ty })
        }
        other => return Err(format!("unknown instruction `{other}`")),
    })
}

fn parse_term(c: &mut Cur, ctx: &mut Ctx) -> Result<Terminator, String> {
    Ok(match c.ident()? {
        "unreachable" => Terminator::Unreachable,
        "ret" => {
            if c.at_end() { Terminator::Ret(None) } else {
                let _t = c.ty(ctx)?;
                Terminator::Ret(Some(c.operand(ctx)?))
            }
        }
        "br" => Terminator::Br(c.block(ctx)?),
        "condbr" => {
            let cond = c.operand(ctx)?; c.comma()?;
            let then_bb = c.block(ctx)?; c.comma()?;
            let else_bb = c.block(ctx)?;
            Terminator::CondBr { cond, then_bb, else_bb }
        }
        "switch" => {
            let _t = c.ty(ctx)?;
            let val = c.operand(ctx)?; c.comma()?;
            c.kw("default")?;
            let default = c.block(ctx)?;
            let mut cases = Vec::new();
            while c.eat(",") {
                if c.at_end() { break; }
                c.expect("[")?;
                let v = c.number()?;
                c.expect(":")?;
                let b = c.block(ctx)?;
                c.expect("]")?;
                cases.push((v, b));
            }
            Terminator::Switch { val, default, cases }
        }
        other => return Err(format!("unknown terminator `{other}`")),
    })
}

fn label_number(l: &str) -> Result<u32, String> {
    l[2..].split(':').next().unwrap_or("").trim().parse::<u32>().map_err(|_| format!("bad block label `{l}`"))
}

fn parse_body(ctx: &mut Ctx, f: &mut Function, param_nums: &[u32], lines: &[&str], start: usize) -> Result<usize, ParseError> {
    let err = |ln: usize, m: String| ParseError { line: ln + 1, message: m };
    let end = (start..lines.len())
        .find(|&i| lines[i].trim_end() == "}")
        .ok_or_else(|| err(start.saturating_sub(1), "unterminated function body".into()))?;
    ctx.blocks.clear();
    for ln in start..end {
        let l = lines[ln];
        if l.starts_with("bb") {
            let n = label_number(l).map_err(|m| err(ln, m))?;
            let name = l.find(';').map(|i| l[i + 1..].trim()).filter(|s| !s.is_empty());
            let b = f.create_block(name);
            if ctx.blocks.insert(n, b).is_some() { return Err(err(ln, format!("duplicate label bb{n}"))); }
        }
    }
    let mut map: HashMap<u32, ValueId> = HashMap::new();
    for (k, &n) in param_nums.iter().enumerate() { map.insert(n, f.params[k]); }
    let mut cur_block: Option<BlockId> = None;
    for ln in start..end {
        let l = lines[ln].trim_end();
        let t = l.trim_start();
        if t.is_empty() || t.starts_with(';') { continue; }
        if l.starts_with("bb") {
            let n = label_number(l).map_err(|m| err(ln, m))?;
            cur_block = Some(ctx.blocks[&n]);
            continue;
        }
        let b = cur_block.ok_or_else(|| err(ln, "instruction outside of a block".into()))?;
        if f.blocks[b].terminator.is_some() { return Err(err(ln, "instruction after terminator".into())); }
        let mut c = Cur::new(t);
        if matches!(c.peek_ident(), Some("ret" | "br" | "condbr" | "switch" | "unreachable")) {
            let term = parse_term(&mut c, ctx).map_err(|m| err(ln, m))?;
            if !c.at_end() { return Err(err(ln, format!("trailing text `{}`", c.rest()))); }
            f.set_terminator(b, term);
            continue;
        }
        let lhs = if c.rest().starts_with("%v") {
            c.p += 2;
            let n = c.uint().map_err(|m| err(ln, m))?;
            c.expect("=").map_err(|m| err(ln, m))?;
            Some(n)
        } else { None };
        let (rty, kind) = parse_inst(&mut c, ctx).map_err(|m| err(ln, m))?;
        if !c.at_end() { return Err(err(ln, format!("trailing text `{}`", c.rest()))); }
        match (lhs, rty) {
            (Some(_), None) => return Err(err(ln, "instruction produces no result".into())),
            (None, Some(_)) => return Err(err(ln, "instruction result must be named".into())),
            _ => {}
        }
        let (_, v) = f.append_inst(b, kind, rty);
        if let (Some(n), Some(v)) = (lhs, v) {
            if map.insert(n, v).is_some() { return Err(err(ln, format!("%v{n} defined twice"))); }
        }
    }
    let mut missing = None;
    f.for_each_operand_mut(|o| {
        if let Operand::Value(v) = *o {
            match map.get(&(v.index() as u32)) {
                Some(&n) => *o = Operand::Value(n),
                None => { missing.get_or_insert(v.index()); }
            }
        }
    });
    if let Some(m) = missing { return Err(err(start, format!("use of undefined value %v{m} in @{}", f.name))); }
    for &b in &f.layout.clone() {
        if f.blocks[b].terminator.is_none() { return Err(err(end, format!("{b} has no terminator in @{}", f.name))); }
    }
    Ok(end + 1)
}

pub fn parse_module(src: &str) -> Result<Module, ParseError> {
    let lines: Vec<&str> = src.lines().collect();
    let err = |ln: usize, m: String| ParseError { line: ln + 1, message: m };

    // Pre-scan: symbol names -> ids (ids follow order of appearance).
    let mut ctx = Ctx { module: Module::new(""), funcs: HashMap::new(), globals: HashMap::new(), blocks: HashMap::new() };
    for (ln, l) in lines.iter().enumerate() {
        let sym = if l.starts_with("define ") || l.starts_with("declare ") || l.starts_with("global @") {
            let a = l.find('@').unwrap();
            let rest = &l[a + 1..];
            let n = rest.find(|c: char| !(c.is_ascii_alphanumeric() || matches!(c, '_' | '.' | '$'))).unwrap_or(rest.len());
            Some((l.starts_with("global"), rest[..n].to_string()))
        } else { None };
        if let Some((is_global, name)) = sym {
            let dup = if is_global {
                let id = GlobalId::new(ctx.globals.len());
                ctx.globals.insert(name.clone(), id).is_some()
            } else {
                let id = FuncId::new(ctx.funcs.len());
                ctx.funcs.insert(name.clone(), id).is_some()
            };
            if dup { return Err(err(ln, format!("duplicate symbol @{name}"))); }
        }
    }

    let mut i = 0;
    while i < lines.len() {
        let t = lines[i].trim();
        if t.is_empty() || t.starts_with(';') { i += 1; continue; }
        let mut c = Cur::new(t);
        let kw = c.ident().map_err(|m| err(i, m))?;
        match kw {
            "module" => { ctx.module.name = c.quoted().map_err(|m| err(i, m))?.to_string(); }
            "target" => { ctx.module.target_triple = Some(c.quoted().map_err(|m| err(i, m))?.to_string()); }
            "global" => {
                let g = (|| -> Result<GlobalVar, String> {
                    let name = c.name()?.to_string();
                    c.expect(":")?;
                    let ty = c.ty(&mut ctx)?;
                    let linkage = by_name(LINKAGES, c.ident()?).ok_or("unknown linkage")?;
                    let is_const = c.peek_ident() == Some("const");
                    if is_const { c.ident()?; }
                    c.kw("align")?;
                    let align = c.uint()?;
                    let mut init = None;
                    if c.peek_ident() == Some("init") {
                        c.ident()?;
                        c.expect("[")?;
                        let mut bytes = Vec::new();
                        while !c.eat("]") {
                            let h = c.ident()?;
                            bytes.push(u8::from_str_radix(h, 16).map_err(|_| format!("bad byte `{h}`"))?);
                        }
                        init = Some(bytes);
                    }
                    let mut relocs = Vec::new();
                    if c.peek_ident() == Some("reloc") {
                        c.ident()?;
                        c.expect("[")?;
                        loop {
                            let offset = c.number()? as u64;
                            c.expect(":")?;
                            let sym = c.name()?;
                            let target = if let Some(&f) = ctx.funcs.get(sym) { RelocTarget::Func(f) }
                                else if let Some(&g) = ctx.globals.get(sym) { RelocTarget::Global(g) }
                                else { return Err(format!("unknown symbol @{sym}")); };
                            let addend = if c.eat("+") { c.number()? as i64 } else if c.eat("-") { -(c.number()? as i64) } else { 0 };
                            relocs.push(Reloc { offset, target, addend });
                            if !c.eat(",") { break; }
                        }
                        c.expect("]")?;
                    }
                    Ok(GlobalVar { name, ty, linkage, is_const, init, relocs, align })
                })().map_err(|m| err(i, m))?;
                ctx.module.add_global(g);
            }
            "declare" | "define" => {
                let is_def = kw == "define";
                let mut param_nums = Vec::new();
                let (name, sig, linkage) = (|| -> Result<(String, Signature, Linkage), String> {
                    let linkage = if is_def { by_name(LINKAGES, c.ident()?).ok_or("unknown linkage")? } else { Linkage::Declaration };
                    let name = c.name()?.to_string();
                    c.expect("(")?;
                    let (mut params, mut variadic, mut sret, mut byval) = (Vec::new(), false, false, Vec::new());
                    if !c.eat(")") {
                        loop {
                            if c.eat("...") { variadic = true; } else {
                                if c.peek_ident() == Some("sret") { c.ident()?; sret = true; }
                                else if c.peek_ident() == Some("byval") {
                                    c.ident()?;
                                    c.expect("(")?;
                                    let size = c.number()? as u64;
                                    c.expect(",")?;
                                    let align = c.number()? as u64;
                                    c.expect(")")?;
                                    byval.push((params.len(), size, align));
                                }
                                params.push(c.ty(&mut ctx)?);
                                if is_def { c.expect("%v")?; param_nums.push(c.uint()?); }
                            }
                            if !c.eat(",") { break; }
                        }
                        c.expect(")")?;
                    }
                    c.expect("->")?;
                    let ret = c.ty(&mut ctx)?;
                    if is_def { c.expect("{")?; }
                    if !c.at_end() { return Err(format!("trailing text `{}`", c.rest())); }
                    Ok((name, Signature { params, ret, variadic, sret, byval }, linkage))
                })().map_err(|m| err(i, m))?;
                let mut f = Function::new(name.clone(), sig, linkage);
                if is_def { i = parse_body(&mut ctx, &mut f, &param_nums, &lines, i + 1)?; } else { i += 1; }
                let id = ctx.module.add_function(f);
                if ctx.funcs.get(&name) != Some(&id) { return Err(err(i, format!("internal error: id mismatch for @{name}"))); }
                continue;
            }
            other => return Err(err(i, format!("unexpected `{other}` at top level"))),
        }
        i += 1;
    }
    Ok(ctx.module)
}
