// SPDX-License-Identifier: GPL-2.0-only WITH LicenseRef-Kaiba-Runtime-Exception
// Copyright (C) The Kaiba Authors
//
// This file is part of the Kaiba compiler toolchain. It is licensed under
// GPL-2.0-only with the Kaiba Runtime/Output Exception; see COPYING and
// LICENSE-EXCEPTION at the repository root.
//! Frontend for LLVM textual IR (`.ll`): lowers a practical subset into llcr.
//!
//! Aimed at what clang emits for integer/pointer C at `-O0`..`-O2`:
//! globals with constant initializers, named struct types, phis, switch,
//! GEP (incl. constant-expression GEPs), calls (direct, indirect, varargs),
//! and the common intrinsics (memcpy/memset/lifetime/expect/min/max/abs).
//! Unsupported constructs produce an error naming the line.

mod body;
mod cur;

use cur::Cur;
use llcr::layout::{size_align, struct_layout};
use llcr::{
    AggregateType, FuncId, Function, GlobalId, GlobalVar, Linkage, Module, Reloc, RelocTarget, Signature, Type,
};
use std::collections::HashMap;
use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LlError { pub line: usize, pub message: String }

impl fmt::Display for LlError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { write!(f, "line {}: {}", self.line, self.message) }
}
impl std::error::Error for LlError {}

const LINKAGE_WORDS: &[&str] = &[
    "private", "internal", "available_externally", "linkonce", "linkonce_odr", "weak", "weak_odr", "common",
    "appending", "extern_weak", "external", "dso_local", "dso_preemptable", "hidden", "protected", "default",
    "unnamed_addr", "local_unnamed_addr", "externally_initialized", "thread_local",
];

const ATTR_WORDS: &[&str] = &[
    "noundef", "zeroext", "signext", "inreg", "noalias", "nocapture", "nonnull", "readonly", "readnone", "writeonly",
    "returned", "nofree", "nest", "align", "dereferenceable", "dereferenceable_or_null", "immarg", "noext",
    "captures", "range", "initializes", "allocalign", "allocptr", "noreturn", "nounwind", "willreturn", "memory",
    "tail", "musttail", "notail", "fastcc", "ccc", "coldcc", "cc", "fast", "nnan", "ninf", "nsz", "arcp", "contract",
    "afn", "reassoc", "nofpclass", "noundef", "speculatable", "norecurse", "nosync", "mustprogress",
    "sret", "byval", "dead_on_unwind", "writable", "alignstack", "dead_on_return",
];

/// Attributes that change the calling convention in ways we don't implement.
const UNSUPPORTED_ATTRS: &[&str] = &["inalloca", "byref", "preallocated"];

/// One scalar leaf of an aggregate.
#[derive(Clone, Debug)]
pub(crate) struct Leaf { pub ty: Type, pub off: u64, pub path: Vec<u32> }

/// How one source-level parameter maps onto llcr parameters.
pub(crate) struct ParamGroup { pub name: String, pub first: usize, pub count: usize, pub agg: Option<Type> }

pub(crate) struct Ctx {
    pub module: Module,
    named_raw: HashMap<String, String>,
    named: HashMap<String, Type>,
    pub funcs: HashMap<String, FuncId>,
    pub globals: HashMap<String, GlobalId>,
}

impl Ctx {
    fn new() -> Self {
        Ctx { module: Module::new("ll"), named_raw: HashMap::new(), named: HashMap::new(), funcs: HashMap::new(), globals: HashMap::new() }
    }

    pub fn ty(&mut self, c: &mut Cur) -> Result<Type, String> {
        let list = |cx: &mut Ctx, c: &mut Cur, close: &str| -> Result<Vec<Type>, String> {
            let mut v = Vec::new();
            if c.eat(close) { return Ok(v); }
            loop {
                v.push(cx.ty(c)?);
                if !c.eat(",") { break; }
            }
            c.expect(close)?;
            Ok(v)
        };
        match c.peek() {
            Some('{') => {
                c.expect("{")?;
                let fields = list(self, c, "}")?;
                Ok(self.module.intern_aggregate(AggregateType::Struct { fields, packed: false }))
            }
            Some('[') => {
                c.expect("[")?;
                let len = c.int()? as u64;
                if c.ident()? != "x" { return Err("expected `x` in array type".into()); }
                let elem = self.ty(c)?;
                c.expect("]")?;
                Ok(self.module.intern_aggregate(AggregateType::Array { elem, len }))
            }
            Some('<') => {
                if c.eat("<{") {
                    let fields = list(self, c, "}>")?;
                    return Ok(self.module.intern_aggregate(AggregateType::Struct { fields, packed: true }));
                }
                c.expect("<")?;
                let len = c.int()? as u32;
                if c.ident()? != "x" { return Err("scalable vectors are not supported".into()); }
                let elem = self.ty(c)?;
                c.expect(">")?;
                Ok(self.module.intern_aggregate(AggregateType::Vector { elem, len }))
            }
            Some('%') => { let n = c.name('%')?; self.named_type(&n) }
            _ => {
                let id = c.ident()?;
                Ok(match id {
                    "void" => Type::Void,
                    "float" => Type::F32,
                    "double" => Type::F64,
                    "ptr" => {
                        if c.kw("addrspace") { c.expect("(")?; let a = c.int()? as u32; c.expect(")")?; Type::Ptr(a) } else { Type::PTR }
                    }
                    s => match s.strip_prefix('i').and_then(|d| d.parse::<u32>().ok()) {
                        Some(n) if n > 0 => Type::Int(n),
                        _ => return Err(format!("unsupported type `{s}`")),
                    },
                })
            }
        }
    }

    fn named_type(&mut self, name: &str) -> Result<Type, String> {
        if let Some(&t) = self.named.get(name) { return Ok(t); }
        let raw = self.named_raw.get(name).cloned().ok_or_else(|| format!("unknown type %{name}"))?;
        let mut c = Cur::new(&raw);
        let t = if c.kw("opaque") {
            self.module.intern_aggregate(AggregateType::Struct { fields: vec![], packed: false })
        } else {
            self.ty(&mut c)?
        };
        self.named.insert(name.to_string(), t);
        Ok(t)
    }

    fn symbol(&self, name: &str) -> Result<RelocTarget, String> {
        if let Some(&f) = self.funcs.get(name) { Ok(RelocTarget::Func(f)) }
        else if let Some(&g) = self.globals.get(name) { Ok(RelocTarget::Global(g)) }
        else { Err(format!("unknown symbol @{name}")) }
    }

    /// `getelementptr [inbounds] (T, ptr @sym, T idx, ...)` with constant indices -> (symbol, byte offset).
    fn const_gep(&mut self, c: &mut Cur) -> Result<(RelocTarget, i64), String> {
        while matches!(c.peek_ident(), Some("inbounds" | "nusw" | "nuw")) { c.ident()?; }
        c.expect("(")?;
        let mut cur = self.ty(c)?;
        c.expect(",")?;
        c.ident()?; // `ptr`
        let target = self.symbol(&c.name('@')?)?;
        let mut off: i64 = 0;
        let mut first = true;
        while c.eat(",") {
            let t = self.ty(c)?;
            let Type::Int(bits) = t else { return Err("GEP index must be an integer".into()) };
            let raw = c.int()?;
            let idx = if bits >= 64 { raw as i64 } else {
                let v = raw & ((1u128 << bits) - 1);
                if (v >> (bits - 1)) & 1 == 1 { (v as i128 - (1i128 << bits)) as i64 } else { v as i64 }
            };
            if first {
                off += idx * size_align(&self.module, cur)?.0 as i64;
                first = false;
                continue;
            }
            let Type::Agg(id) = cur else { return Err("GEP into a non-aggregate".into()) };
            match self.module.aggregate(id).clone() {
                AggregateType::Struct { fields, packed } => {
                    let (offs, _, _) = struct_layout(&self.module, &fields, packed)?;
                    off += *offs.get(idx as usize).ok_or("struct index out of range")? as i64;
                    cur = fields[idx as usize];
                }
                AggregateType::Array { elem, .. } | AggregateType::Vector { elem, .. } => {
                    off += idx * size_align(&self.module, elem)?.0 as i64;
                    cur = elem;
                }
            }
        }
        c.expect(")")?;
        Ok((target, off))
    }

    /// Scalar leaves of a (possibly nested) aggregate: type, byte offset, index path.
    pub fn leaves(&self, t: Type) -> Result<Vec<Leaf>, String> {
        let mut out = Vec::new();
        self.leaves_into(t, 0, &mut Vec::new(), &mut out)?;
        if out.len() > 16 { return Err("aggregate too large to pass in registers".into()); }
        Ok(out)
    }

    fn leaves_into(&self, t: Type, base: u64, path: &mut Vec<u32>, out: &mut Vec<Leaf>) -> Result<(), String> {
        match t {
            Type::Agg(id) => match self.module.aggregate(id).clone() {
                AggregateType::Struct { fields, packed } => {
                    let (offs, _, _) = struct_layout(&self.module, &fields, packed)?;
                    for (i, f) in fields.iter().enumerate() {
                        path.push(i as u32);
                        self.leaves_into(*f, base + offs[i], path, out)?;
                        path.pop();
                    }
                }
                AggregateType::Array { elem, len } => {
                    if len > 16 { return Err("aggregate too large to pass in registers".into()); }
                    let es = size_align(&self.module, elem)?.0;
                    for i in 0..len {
                        path.push(i as u32);
                        self.leaves_into(elem, base + i * es, path, out)?;
                        path.pop();
                    }
                }
                AggregateType::Vector { .. } => return Err("vector types are not supported".into()),
            },
            Type::Void => return Err("void aggregate member".into()),
            t => out.push(Leaf { ty: t, off: base, path: path.clone() }),
        }
        Ok(())
    }

    /// The type reached by following `path` into `t`.
    pub fn type_at(&self, mut t: Type, path: &[u32]) -> Result<Type, String> {
        for &i in path {
            let Type::Agg(id) = t else { return Err("index into a non-aggregate".into()) };
            t = match self.module.aggregate(id) {
                AggregateType::Struct { fields, .. } => *fields.get(i as usize).ok_or("aggregate index out of range")?,
                AggregateType::Array { elem, .. } | AggregateType::Vector { elem, .. } => *elem,
            };
        }
        Ok(t)
    }

    pub fn ensure_func(&mut self, name: &str, sig: Signature) -> FuncId {
        if let Some(&f) = self.funcs.get(name) { return f; }
        let id = self.module.add_function(Function::new(name, sig, Linkage::Declaration));
        self.funcs.insert(name.to_string(), id);
        id
    }

    /// Parse a constant of type `ty` and write its bytes into `buf[off..]`.
    fn const_into(&mut self, ty: Type, c: &mut Cur, buf: &mut [u8], off: usize, rel: &mut Vec<Reloc>) -> Result<(), String> {
        if c.kw("zeroinitializer") || c.kw("undef") || c.kw("poison") { return Ok(()); }
        match ty {
            Type::Int(n) => {
                let v = if c.kw("true") { 1 } else if c.kw("false") { 0 } else { c.int()? };
                let (size, _) = size_align(&self.module, Type::Int(n))?;
                for i in 0..size as usize { buf[off + i] = (v >> (8 * i)) as u8; }
            }
            Type::F32 => { let b = c.float_bits(true)? as u32; buf[off..off + 4].copy_from_slice(&b.to_le_bytes()); }
            Type::F64 => { let b = c.float_bits(false)?; buf[off..off + 8].copy_from_slice(&b.to_le_bytes()); }
            Type::Ptr(_) => {
                if c.kw("null") {
                } else if c.peek() == Some('@') {
                    let (target, addend) = (self.symbol(&c.name('@')?)?, 0);
                    rel.push(Reloc { offset: off as u64, target, addend });
                } else if c.kw("getelementptr") {
                    let (target, addend) = self.const_gep(c)?;
                    rel.push(Reloc { offset: off as u64, target, addend });
                } else {
                    return Err("unsupported pointer constant in global initializer".into());
                }
            }
            Type::Agg(id) => match self.module.aggregate(id).clone() {
                AggregateType::Array { elem, len } => {
                    if elem == Type::I8 && c.peek() == Some('c') {
                        c.expect("c")?;
                        let s = c.quoted()?;
                        if s.len() as u64 > len { return Err("string initializer is longer than its array".into()); }
                        buf[off..off + s.len()].copy_from_slice(&s);
                    } else {
                        let es = size_align(&self.module, elem)?.0 as usize;
                        c.expect("[")?;
                        for k in 0..len as usize {
                            let t = self.ty(c)?;
                            if t != elem { return Err("array element type mismatch".into()); }
                            self.const_into(elem, c, buf, off + k * es, rel)?;
                            if k + 1 < len as usize { c.expect(",")?; }
                        }
                        c.expect("]")?;
                    }
                }
                AggregateType::Vector { elem, len } => {
                    let es = size_align(&self.module, elem)?.0 as usize;
                    c.expect("<")?;
                    for k in 0..len as usize {
                        self.ty(c)?;
                        self.const_into(elem, c, buf, off + k * es, rel)?;
                        if k + 1 < len as usize { c.expect(",")?; }
                    }
                    c.expect(">")?;
                }
                AggregateType::Struct { fields, packed } => {
                    let (offs, _, _) = struct_layout(&self.module, &fields, packed)?;
                    let (open, close) = if packed { ("<{", "}>") } else { ("{", "}") };
                    c.expect(open)?;
                    for (k, &f) in fields.iter().enumerate() {
                        let t = self.ty(c)?;
                        if t != f { return Err("struct field type mismatch".into()); }
                        self.const_into(f, c, buf, off + offs[k] as usize, rel)?;
                        if k + 1 < fields.len() { c.expect(",")?; }
                    }
                    c.expect(close)?;
                }
            },
            Type::Void => return Err("void constant".into()),
        }
        Ok(())
    }
}

/// Skip attribute/linkage words. Returns the linkage implied, if any.
pub(crate) fn skip_attrs(c: &mut Cur) -> Result<Option<Linkage>, String> {
    let mut link = None;
    while let Some(w) = c.peek_ident() {
        let is_link = LINKAGE_WORDS.contains(&w);
        if UNSUPPORTED_ATTRS.contains(&w) {
            return Err(format!("`{w}` (aggregate by-value arguments) is not supported"));
        }
        if !is_link && !ATTR_WORDS.contains(&w) { break; }
        c.ident()?;
        if is_link {
            link = link.or(match w { "internal" => Some(Linkage::Internal), "private" => Some(Linkage::Private), _ => Some(Linkage::External) });
        }
        if c.peek() == Some('(') {
            let open = c.p;
            let mut depth = 0;
            loop {
                match c.rest().chars().next() {
                    Some('(') => depth += 1,
                    Some(')') => depth -= 1,
                    None => return Err("unbalanced parentheses".into()),
                    _ => {}
                }
                c.p += 1;
                if depth == 0 { break; }
            }
            let inner = c.s[open + 1..c.p - 1].to_string();
            if w == "sret" { c.sret_ty = Some(inner); } else if w == "byval" { c.byval_ty = Some(inner); }
        } else if matches!(w, "align" | "cc") {
            c.int()?;
        }
    }
    Ok(link)
}

struct Header { name: String, sig: Signature, linkage: Linkage, groups: Vec<ParamGroup> }

fn parse_header(cx: &mut Ctx, c: &mut Cur, is_def: bool) -> Result<Header, String> {
    let linkage = skip_attrs(c)?.unwrap_or(Linkage::External);
    let ret = cx.ty(c)?;
    skip_attrs(c)?;
    let name = c.name('@')?;
    c.expect("(")?;
    let (mut params, mut variadic, mut sret, mut byval) = (vec![], false, false, vec![]);
    let mut groups: Vec<ParamGroup> = vec![];
    if !c.eat(")") {
        loop {
            if c.eat("...") { variadic = true; } else {
                let t = cx.ty(c)?;
                c.sret_ty = None;
                c.byval_ty = None;
                skip_attrs(c)?;
                let pname = if c.peek() == Some('%') { c.name('%')? } else { String::new() };
                let first = params.len();
                if c.sret_ty.take().is_some() {
                    if first != 0 { return Err("sret must be the first parameter".into()); }
                    sret = true;
                    params.push(Type::PTR);
                    groups.push(ParamGroup { name: pname, first, count: 1, agg: None });
                } else if let Some(text) = c.byval_ty.take() {
                    let bt = cx.ty(&mut Cur::new(&text))?;
                    let (size, align) = size_align(&cx.module, bt)?;
                    byval.push((first, size, align.max(8)));
                    params.push(Type::PTR);
                    groups.push(ParamGroup { name: pname, first, count: 1, agg: None });
                } else if let Type::Agg(_) = t {
                    // First-class aggregates are passed as their scalar leaves.
                    let leaves = cx.leaves(t)?;
                    for l in &leaves { params.push(l.ty); }
                    groups.push(ParamGroup { name: pname, first, count: leaves.len(), agg: Some(t) });
                } else {
                    params.push(t);
                    groups.push(ParamGroup { name: pname, first, count: 1, agg: None });
                }
            }
            if !c.eat(",") { break; }
        }
        c.expect(")")?;
    }
    Ok(Header { name, sig: Signature { params, ret, variadic, sret, byval }, linkage: if is_def { linkage } else { Linkage::Declaration }, groups })
}

struct PendingInit { gid: GlobalId, ty: Type, rest: String, is_const: bool, has_init: bool }

/// Register a global (name, kind, type, linkage); its initializer is parsed later,
/// once every symbol it might point to exists.
fn parse_global_head(cx: &mut Ctx, line: &str) -> Result<PendingInit, String> {
    let mut c = Cur::new(line);
    let name = c.name('@')?;
    c.expect("=")?;
    let mut linkage = None;
    while let Some(w) = c.peek_ident() {
        if !LINKAGE_WORDS.contains(&w) { break; }
        c.ident()?;
        linkage = linkage.or(Some(w));
        if c.peek() == Some('(') { skip_attrs(&mut c)?; }
    }
    let is_const = if c.kw("constant") { true } else if c.kw("global") { false } else {
        return Err("aliases, ifuncs and other global kinds are not supported".into());
    };
    let ty = cx.ty(&mut c)?;
    let has_init = !matches!(c.peek(), None | Some(',') | Some(';'));
    let (_, tyalign) = size_align(&cx.module, ty)?;
    let external = matches!(linkage, Some("external") | Some("extern_weak")) && !has_init;
    let lk = match linkage {
        Some("internal") => Linkage::Internal,
        Some("private") => Linkage::Private,
        _ if external => Linkage::Declaration,
        _ => Linkage::External,
    };
    let gid = cx.module.add_global(GlobalVar { name: name.clone(), ty, linkage: lk, is_const, init: None, relocs: vec![], align: tyalign as u32 });
    cx.globals.insert(name, gid);
    Ok(PendingInit { gid, ty, rest: c.rest().to_string(), is_const, has_init: has_init && lk != Linkage::Declaration })
}

fn parse_global_init(cx: &mut Ctx, p: &PendingInit) -> Result<(), String> {
    let mut c = Cur::new(&p.rest);
    let (size, tyalign) = size_align(&cx.module, p.ty)?;
    let (mut init, mut relocs) = (None, Vec::new());
    if p.has_init {
        let mut buf = vec![0u8; size as usize];
        let zero = { let mut q = Cur::new(c.rest()); q.kw("zeroinitializer") || q.kw("undef") || q.kw("poison") };
        cx.const_into(p.ty, &mut c, &mut buf, 0, &mut relocs)?;
        init = if zero && !p.is_const { None } else if zero { Some(vec![]) } else { Some(buf) };
    }
    let mut align = tyalign as u32;
    while c.eat(",") {
        if c.kw("align") { align = c.int()? as u32; } else { break; }
    }
    let g = &mut cx.module.globals[p.gid];
    g.init = init;
    g.relocs = relocs;
    g.align = align;
    Ok(())
}

/// Parse LLVM textual IR into an llcr module.
pub fn parse_ll(src: &str) -> Result<Module, LlError> {
    let lines: Vec<&str> = src.lines().collect();
    let err = |ln: usize, m: String| LlError { line: ln + 1, message: m };
    let mut cx = Ctx::new();

    // Pass 0: named types and module-level info.
    for l in lines.iter() {
        let t = l.trim_start();
        if t.starts_with('%') {
            let mut c = Cur::new(t);
            if let Ok(name) = c.name('%') {
                if c.eat("=") && c.kw("type") { cx.named_raw.insert(name, c.rest().to_string()); }
            }
        } else if let Some(r) = t.strip_prefix("target triple") {
            let mut c = Cur::new(r);
            if c.eat("=") { if let Ok(s) = c.quoted() { cx.module.target_triple = Some(String::from_utf8_lossy(&s).into_owned()); } }
        } else if let Some(r) = t.strip_prefix("source_filename") {
            let mut c = Cur::new(r);
            if c.eat("=") { if let Ok(s) = c.quoted() { cx.module.name = String::from_utf8_lossy(&s).into_owned(); } }
        }
    }

    // Pass A: globals, function shells; remember bodies.
    struct BodyRef { fid: FuncId, groups: Vec<ParamGroup>, start: usize, end: usize }
    let mut bodies = Vec::new();
    let mut pending: Vec<(usize, PendingInit)> = Vec::new();
    let mut i = 0;
    while i < lines.len() {
        let t = lines[i].trim_start();
        if t.starts_with("define ") || t.starts_with("declare ") {
            let is_def = t.starts_with("define ");
            let mut c = Cur::new(&t[if is_def { 7 } else { 8 }..]);
            let h = parse_header(&mut cx, &mut c, is_def).map_err(|m| err(i, m))?;
            let skip = h.name.starts_with("llvm.") && !is_def;
            if !skip {
                if cx.funcs.contains_key(&h.name) { return Err(err(i, format!("duplicate function @{}", h.name))); }
                let id = cx.module.add_function(Function::new(h.name.clone(), h.sig, h.linkage));
                cx.funcs.insert(h.name, id);
                if is_def {
                    let end = (i + 1..lines.len()).find(|&j| lines[j].trim_end() == "}")
                        .ok_or_else(|| err(i, "unterminated function body".into()))?;
                    bodies.push(BodyRef { fid: id, groups: h.groups, start: i + 1, end });
                    i = end;
                }
            } else if is_def {
                return Err(err(i, "cannot define an intrinsic".into()));
            }
        } else if t.starts_with('@') {
            let p = parse_global_head(&mut cx, t).map_err(|m| err(i, m))?;
            pending.push((i, p));
        }
        i += 1;
    }

    // Pass A2: global initializers (all symbols are now known).
    for (ln, p) in &pending {
        parse_global_init(&mut cx, p).map_err(|m| err(*ln, m))?;
    }

    // Pass B: function bodies.
    for b in bodies {
        body::build_body(&mut cx, b.fid, &b.groups, &lines, b.start, b.end)?;
    }
    Ok(cx.module)
}
