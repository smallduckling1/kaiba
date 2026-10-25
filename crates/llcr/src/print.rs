// SPDX-License-Identifier: GPL-2.0-only WITH LicenseRef-Kaiba-Runtime-Exception
// Copyright (C) The Kaiba Authors
//
// This file is part of the Kaiba compiler toolchain. It is licensed under
// GPL-2.0-only with the Kaiba Runtime/Output Exception; see COPYING and
// LICENSE-EXCEPTION at the repository root.
//! Textual form of llcr (the `.llcr` syntax). Designed to be parsed back.

use crate::function::{Function, Linkage};
use crate::inst::{Callee, Constant, InstKind, Operand, Terminator};
use crate::module::{Module, RelocTarget};
use crate::types::{AggregateType, Type};
use std::fmt::{self, Write};

pub fn type_to_string(m: &Module, t: Type) -> String {
    match t {
        Type::Void => "void".into(),
        Type::Int(n) => format!("i{n}"),
        Type::F32 => "f32".into(),
        Type::F64 => "f64".into(),
        Type::Ptr(0) => "ptr".into(),
        Type::Ptr(a) => format!("ptr addrspace({a})"),
        Type::Agg(id) => match m.aggregate(id) {
            AggregateType::Struct { fields, packed } => {
                let inner = fields.iter().map(|t| type_to_string(m, *t)).collect::<Vec<_>>().join(", ");
                if *packed { format!("<{{ {inner} }}>") } else { format!("{{ {inner} }}") }
            }
            AggregateType::Array { elem, len } => format!("[{len} x {}]", type_to_string(m, *elem)),
            AggregateType::Vector { elem, len } => format!("<{len} x {}>", type_to_string(m, *elem)),
        },
    }
}

pub fn operand_to_string(m: &Module, o: &Operand) -> String {
    match o {
        Operand::Value(v) => v.to_string(),
        Operand::Const(Constant::Int { bits, value }) => format!("{value}:i{bits}"),
        Operand::Const(Constant::F32(b)) => format!("f32:{b:#010x}"),
        Operand::Const(Constant::F64(b)) => format!("f64:{b:#018x}"),
        Operand::Const(Constant::Null) => "null".into(),
        Operand::Const(Constant::Undef(t)) => format!("undef:{}", type_to_string(m, *t)),
        Operand::Const(Constant::Poison(t)) => format!("poison:{}", type_to_string(m, *t)),
        Operand::Global(g) => format!("@{}", m.globals[*g].name),
        Operand::Func(f) => format!("@{}", m.functions[*f].name),
    }
}

fn lower<T: fmt::Debug>(x: &T) -> String { format!("{x:?}").to_lowercase() }

fn linkage(l: Linkage) -> String { lower(&l) }

fn param_marker(sig: &crate::types::Signature, k: usize) -> String {
    if sig.sret && k == 0 { return "sret ".into(); }
    match sig.byval.iter().find(|(i, _, _)| *i == k) { Some((_, s, a)) => format!("byval({s},{a}) "), None => String::new() }
}

pub fn print_function(m: &Module, f: &Function) -> String {
    let ty = |t: Type| type_to_string(m, t);
    let op = |o: &Operand| operand_to_string(m, o);
    let mut s = String::new();
    if f.is_declaration() {
        let ps: Vec<_> = f.sig.params.iter().enumerate().map(|(k, t)| format!("{}{}", param_marker(&f.sig, k), ty(*t)))
            .chain(f.sig.variadic.then(|| "...".to_string())).collect();
        let _ = writeln!(s, "declare @{}({}) -> {}", f.name, ps.join(", "), ty(f.sig.ret));
        return s;
    }
    let ps: Vec<_> = f.params.iter().enumerate().map(|(k, &p)| format!("{}{} {}", param_marker(&f.sig, k), ty(f.values[p].ty), p))
        .chain(f.sig.variadic.then(|| "...".to_string())).collect();
    let _ = writeln!(s, "define {} @{}({}) -> {} {{", linkage(f.linkage), f.name, ps.join(", "), ty(f.sig.ret));
    for &b in &f.layout {
        let bd = &f.blocks[b];
        match &bd.name {
            Some(n) => { let _ = writeln!(s, "{b}:  ; {n}"); }
            None => { let _ = writeln!(s, "{b}:"); }
        }
        for &i in &bd.insts {
            let d = &f.insts[i];
            let lhs = d.result.map(|r| format!("{r} = ")).unwrap_or_default();
            let rty = d.result.map(|r| ty(f.values[r].ty)).unwrap_or_default();
            let body = match &d.kind {
                InstKind::Binary { op: o, lhs, rhs } => format!("{} {rty} {}, {}", lower(o), op(lhs), op(rhs)),
                InstKind::ICmp { pred, lhs, rhs } => format!("icmp {} {} {}, {}", lower(pred), ty(f.operand_type(lhs)), op(lhs), op(rhs)),
                InstKind::FCmp { pred, lhs, rhs } => format!("fcmp {} {} {}, {}", lower(pred), ty(f.operand_type(lhs)), op(lhs), op(rhs)),
                InstKind::Cast { op: c, val, to } => format!("{} {} {} to {}", lower(c), ty(f.operand_type(val)), op(val), ty(*to)),
                InstKind::Select { cond, then_val, else_val } => format!("select {rty} {}, {}, {}", op(cond), op(then_val), op(else_val)),
                InstKind::Load { ty: t, ptr, align, volatile } => format!("load {}{}, {}, align {align}", if *volatile { "volatile " } else { "" }, ty(*t), op(ptr)),
                InstKind::Store { val, ptr, align, volatile } => format!("store {}{} {}, {}, align {align}", if *volatile { "volatile " } else { "" }, ty(f.operand_type(val)), op(val), op(ptr)),
                InstKind::VaStart { list } => format!("va_start {}", op(list)),
                InstKind::VaCopy { dst, src } => format!("va_copy {}, {}", op(dst), op(src)),
                InstKind::Alloca { ty: t, count, align } => format!("alloca {}, {}, align {align}", ty(*t), op(count)),
                InstKind::Gep { base_ty, ptr, indices, inbounds } => {
                    let idx: Vec<_> = indices.iter().map(&op).collect();
                    format!("gep {}{}, {}, [{}]", if *inbounds { "inbounds " } else { "" }, ty(*base_ty), op(ptr), idx.join(", "))
                }
                InstKind::ExtractValue { agg, indices } => format!("extractvalue {rty} {}, {indices:?}", op(agg)),
                InstKind::InsertValue { agg, val, indices } => format!("insertvalue {rty} {}, {}, {indices:?}", op(agg), op(val)),
                InstKind::Phi { ty: t, incoming } => {
                    let inc: Vec<_> = incoming.iter().map(|(b, o)| format!("[{b}: {}]", op(o))).collect();
                    format!("phi {} {}", ty(*t), inc.join(", "))
                }
                InstKind::Call { callee, args, ret_ty } => {
                    let c = match callee { Callee::Direct(fid) => format!("@{}", m.functions[*fid].name), Callee::Indirect(o) => op(o) };
                    let a: Vec<_> = args.iter().map(&op).collect();
                    format!("call {} {c}({})", ty(*ret_ty), a.join(", "))
                }
            };
            let _ = writeln!(s, "  {lhs}{body}");
        }
        let t = match &bd.terminator {
            None => "<missing terminator>".to_string(),
            Some(Terminator::Ret(None)) => "ret".into(),
            Some(Terminator::Ret(Some(o))) => format!("ret {} {}", ty(f.operand_type(o)), op(o)),
            Some(Terminator::Br(t)) => format!("br {t}"),
            Some(Terminator::CondBr { cond, then_bb, else_bb }) => format!("condbr {}, {then_bb}, {else_bb}", op(cond)),
            Some(Terminator::Switch { val, default, cases }) => {
                let cs: Vec<_> = cases.iter().map(|(v, b)| format!("[{v}: {b}]")).collect();
                format!("switch {} {}, default {default}, {}", ty(f.operand_type(val)), op(val), cs.join(", "))
            }
            Some(Terminator::Unreachable) => "unreachable".into(),
        };
        let _ = writeln!(s, "  {t}");
    }
    s.push_str("}\n");
    s
}

impl fmt::Display for Module {
    fn fmt(&self, out: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(out, "module \"{}\"", self.name)?;
        if let Some(t) = &self.target_triple { writeln!(out, "target \"{t}\"")?; }
        for (_, g) in self.globals.iter() {
            write!(out, "global @{}: {} {}{} align {}", g.name, type_to_string(self, g.ty), linkage(g.linkage),
                   if g.is_const { " const" } else { "" }, g.align)?;
            if let Some(b) = &g.init {
                let hex: Vec<_> = b.iter().map(|x| format!("{x:02x}")).collect();
                write!(out, " init [{}]", hex.join(" "))?;
            }
            if !g.relocs.is_empty() {
                let rs: Vec<String> = g.relocs.iter().map(|r| {
                    let name = match r.target { RelocTarget::Global(id) => &self.globals[id].name, RelocTarget::Func(id) => &self.functions[id].name };
                    let add = if r.addend == 0 { String::new() } else { format!("{:+}", r.addend) };
                    format!("{}: @{name}{add}", r.offset)
                }).collect();
                write!(out, " reloc [{}]", rs.join(", "))?;
            }
            writeln!(out)?;
        }
        for (_, f) in self.functions.iter() {
            writeln!(out)?;
            write!(out, "{}", print_function(self, f))?;
        }
        Ok(())
    }
}
