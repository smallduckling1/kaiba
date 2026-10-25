// SPDX-License-Identifier: GPL-2.0-only WITH LicenseRef-Kaiba-Runtime-Exception
// Copyright (C) The Kaiba Authors
//
// This file is part of the Kaiba compiler toolchain. It is licensed under
// GPL-2.0-only with the Kaiba Runtime/Output Exception; see COPYING and
// LICENSE-EXCEPTION at the repository root.
//! Instruction selection: llcr -> machine IR. Also performs out-of-SSA
//! conversion (critical-edge splitting + parallel-safe phi copies).

use crate::layout::{size_align, struct_layout};
use crate::mir::*;
use llcr::{
    AggregateType, BinOp, BlockId, Callee, CastOp, Constant, Entity, Function, InstId, InstKind, Module, Operand,
    Terminator, Type, ValueId,
};
use std::collections::HashMap;

fn mask64(bits: u32) -> u64 { if bits >= 64 { u64::MAX } else { (1u64 << bits) - 1 } }

fn sx64(v: u64, bits: u32) -> i64 {
    if bits >= 64 { v as i64 } else if (v >> (bits - 1)) & 1 == 1 { (v | !mask64(bits)) as i64 } else { v as i64 }
}

pub fn bits_of(t: Type) -> Result<u32, String> {
    match t {
        Type::Int(n) if matches!(n, 1 | 8 | 16 | 32 | 64) => Ok(n),
        Type::Ptr(_) => Ok(64),
        Type::F32 => Ok(32),
        Type::F64 => Ok(64),
        t => Err(format!("unsupported type {t:?}")),
    }
}

pub fn class_of(t: Type) -> Result<ArgClass, String> {
    match t {
        Type::F32 => Ok(ArgClass::F32),
        Type::F64 => Ok(ArgClass::F64),
        t => bits_of(t).map(|_| ArgClass::Int),
    }
}

/// Flatten an aggregate type into its scalar leaves (depth-first, declaration order).
fn leaf_types(m: &Module, t: Type, out: &mut Vec<Type>) -> Result<(), String> {
    match t {
        Type::Agg(id) => match m.aggregate(id) {
            AggregateType::Struct { fields, .. } => { for f in fields { leaf_types(m, *f, out)?; } }
            AggregateType::Array { elem, len } => {
                if *len > 16 { return Err("aggregate too large to pass in registers".into()); }
                for _ in 0..*len { leaf_types(m, *elem, out)?; }
            }
            AggregateType::Vector { .. } => return Err("vector types are not supported".into()),
        },
        Type::Void => return Err("void aggregate leaf".into()),
        t => { bits_of(t)?; out.push(t); }
    }
    if out.len() > 16 { return Err("aggregate too large to pass in registers".into()); }
    Ok(())
}

fn leaf_count(m: &Module, t: Type) -> Result<usize, String> {
    let mut v = Vec::new();
    leaf_types(m, t, &mut v)?;
    Ok(v.len())
}

/// For an index path into an aggregate: (first leaf, number of leaves, type at the path).
fn leaf_index(m: &Module, mut t: Type, indices: &[u32]) -> Result<(usize, usize, Type), String> {
    let mut start = 0;
    for &i in indices {
        let Type::Agg(id) = t else { return Err("index into a non-aggregate".into()) };
        match m.aggregate(id) {
            AggregateType::Struct { fields, .. } => {
                let i = i as usize;
                if i >= fields.len() { return Err("aggregate index out of range".into()); }
                for f in &fields[..i] { start += leaf_count(m, *f)?; }
                t = fields[i];
            }
            AggregateType::Array { elem, len } => {
                if i as u64 >= *len { return Err("aggregate index out of range".into()); }
                start += i as usize * leaf_count(m, *elem)?;
                t = *elem;
            }
            AggregateType::Vector { .. } => return Err("vector types are not supported".into()),
        }
    }
    Ok((start, leaf_count(m, t)?, t))
}

/// Split every edge out of a multi-way terminator into a block that has phis,
/// so phi copies can be placed at the end of a single-successor predecessor.
fn split_critical_edges(f: &mut Function) {
    for p in f.layout.clone() {
        let same = match &f.blocks[p].terminator {
            Some(Terminator::CondBr { then_bb, else_bb, .. }) if then_bb == else_bb => Some(*then_bb),
            _ => None,
        };
        if let Some(t) = same { f.blocks[p].terminator = Some(Terminator::Br(t)); }
        if matches!(f.blocks[p].terminator, Some(Terminator::Br(_)) | None) { continue; }
        let mut succs = f.successors(p);
        succs.sort();
        succs.dedup();
        for s in succs {
            if !f.blocks[s].insts.iter().any(|&i| f.insts[i].kind.is_phi()) { continue; }
            let e = f.create_block(None);
            // Keep the new block next to its predecessor so live ranges stay short.
            let pos = f.layout.iter().position(|&b| b == p).unwrap();
            f.layout.pop();
            f.layout.insert(pos + 1, e);
            f.set_terminator(e, Terminator::Br(s));
            f.blocks[p].terminator.as_mut().unwrap().map_successors(|b| if b == s { e } else { b });
            for i in f.blocks[s].insts.clone() {
                if let InstKind::Phi { incoming, .. } = &mut f.insts[i].kind {
                    for (pb, _) in incoming.iter_mut() { if *pb == p { *pb = e; } }
                }
            }
        }
    }
}

struct Lowerer<'a> {
    m: &'a Module,
    f: Function,
    next_vreg: u32,
    slots: Vec<(u64, u64)>,
    is_entry: bool,
    abi: Abi,
    /// Aggregate-typed SSA values, as lists of their scalar leaves.
    aggs: HashMap<ValueId, Vec<Val>>,
    /// Vreg holding the hidden struct-return pointer, if any.
    sret_vreg: Option<VReg>,
    va: Option<VaInfo>,
    /// Named-parameter register/stack usage (for `va_start`).
    named: (usize, usize, usize),
}

impl<'a> Lowerer<'a> {
    fn tmp(&mut self) -> VReg { let v = self.next_vreg; self.next_vreg += 1; v }

    fn val(&self, o: &Operand) -> Result<Val, String> {
        Ok(match o {
            Operand::Value(v) => Val::V(v.index() as u32),
            Operand::Const(Constant::Int { bits, value }) => {
                if *bits > 64 { return Err(format!("unsupported integer width i{bits}")); }
                Val::Imm(*value as u64 & mask64(*bits))
            }
            Operand::Const(Constant::Null | Constant::Undef(_) | Constant::Poison(_)) => Val::Imm(0),
            Operand::Const(Constant::F32(b)) => Val::Imm(*b as u64),
            Operand::Const(Constant::F64(b)) => Val::Imm(*b),
            Operand::Global(g) => {
                let g = &self.m.globals[*g];
                Val::Sym { name: g.name.clone(), external: g.linkage == llcr::Linkage::Declaration }
            }
            Operand::Func(fid) => {
                let f = &self.m.functions[*fid];
                Val::Sym { name: f.name.clone(), external: f.is_declaration() }
            }
        })
    }

    fn agg_leaves(&self, o: &Operand) -> Result<Vec<Val>, String> {
        match o {
            Operand::Value(v) => self.aggs.get(v).cloned().ok_or_else(|| "aggregate value used in an unsupported way".to_string()),
            Operand::Const(Constant::Undef(t) | Constant::Poison(t)) => {
                let mut lt = Vec::new();
                leaf_types(self.m, *t, &mut lt)?;
                Ok(vec![Val::Imm(0); lt.len()])
            }
            _ => Err("unsupported aggregate operand".into()),
        }
    }

    fn bits(&self, o: &Operand) -> Result<u32, String> { bits_of(self.f.operand_type(o)) }

    fn lower_inst(&mut self, i: InstId, out: &mut Vec<MInst>) -> Result<(), String> {
        let kind = self.f.insts[i].kind.clone();
        let res = self.f.insts[i].result;
        let dst = || res.map(|r| r.index() as u32).ok_or_else(|| "instruction has no result".to_string());
        match kind {
            InstKind::Phi { .. } => {}
            InstKind::Binary { op, lhs, rhs } => {
                let rty = self.f.values[res.unwrap()].ty;
                if matches!(rty, Type::F32 | Type::F64) {
                    let (l, r) = (self.val(&lhs)?, self.val(&rhs)?);
                    return self.float_binary(op, rty, dst()?, l, r, out);
                }
                let alu = match op {
                    BinOp::Add => Alu::Add, BinOp::Sub => Alu::Sub, BinOp::Mul => Alu::Mul,
                    BinOp::And => Alu::And, BinOp::Or => Alu::Or, BinOp::Xor => Alu::Xor,
                    BinOp::Shl => Alu::Shl, BinOp::LShr => Alu::LShr, BinOp::AShr => Alu::AShr,
                    BinOp::UDiv => Alu::UDiv, BinOp::SDiv => Alu::SDiv, BinOp::URem => Alu::URem, BinOp::SRem => Alu::SRem,
                    _ => return Err("floating point is not supported yet".into()),
                };
                let bits = bits_of(self.f.values[res.unwrap()].ty)?;
                out.push(MInst::Alu { op: alu, bits, dst: dst()?, lhs: self.val(&lhs)?, rhs: self.val(&rhs)? });
            }
            InstKind::ICmp { pred, lhs, rhs } => {
                let bits = self.bits(&lhs)?;
                out.push(MInst::Cmp { pred, bits, dst: dst()?, lhs: self.val(&lhs)?, rhs: self.val(&rhs)? });
            }
            InstKind::FCmp { pred, lhs, rhs } => {
                let dbl = self.f.operand_type(&lhs) == Type::F64;
                out.push(MInst::FCmp { pred, dbl, dst: dst()?, lhs: self.val(&lhs)?, rhs: self.val(&rhs)? });
            }
            InstKind::Cast { op, val, to } => {
                let (d, src) = (dst()?, self.val(&val)?);
                let to_bits = bits_of(to)?;
                match op {
                    CastOp::Trunc => out.push(MInst::Ext { kind: ExtKind::Trunc, from: self.bits(&val)?, to: to_bits, dst: d, src }),
                    CastOp::ZExt => out.push(MInst::Ext { kind: ExtKind::Zext, from: self.bits(&val)?, to: to_bits, dst: d, src }),
                    CastOp::SExt => out.push(MInst::Ext { kind: ExtKind::Sext, from: self.bits(&val)?, to: to_bits, dst: d, src }),
                    CastOp::PtrToInt if to_bits < 64 => out.push(MInst::Ext { kind: ExtKind::Trunc, from: 64, to: to_bits, dst: d, src }),
                    CastOp::PtrToInt | CastOp::IntToPtr => out.push(MInst::Mov { dst: d, src }),
                    CastOp::SIToFP => out.push(MInst::FCvt { kind: FCvt::SIToFP { from: self.bits(&val)?, dbl: to == Type::F64 }, dst: d, src }),
                    CastOp::UIToFP => out.push(MInst::FCvt { kind: FCvt::UIToFP { from: self.bits(&val)?, dbl: to == Type::F64 }, dst: d, src }),
                    CastOp::FPToSI => out.push(MInst::FCvt { kind: FCvt::FPToSI { to: to_bits, dbl: self.f.operand_type(&val) == Type::F64 }, dst: d, src }),
                    CastOp::FPToUI => out.push(MInst::FCvt { kind: FCvt::FPToUI { to: to_bits, dbl: self.f.operand_type(&val) == Type::F64 }, dst: d, src }),
                    CastOp::FPExt => out.push(MInst::FCvt { kind: FCvt::FPExt, dst: d, src }),
                    CastOp::FPTrunc => out.push(MInst::FCvt { kind: FCvt::FPTrunc, dst: d, src }),
                    CastOp::Bitcast if self.bits(&val)? == to_bits => out.push(MInst::Mov { dst: d, src }),
                    _ => return Err(format!("unsupported cast {op:?}")),
                }
            }
            InstKind::Select { cond, then_val, else_val } => {
                out.push(MInst::Select { dst: dst()?, cond: self.val(&cond)?, then_val: self.val(&then_val)?, else_val: self.val(&else_val)? });
            }
            InstKind::Load { ty, ptr, .. } => out.push(MInst::Load { dst: dst()?, bits: bits_of(ty)?, addr: Addr::simple(self.val(&ptr)?) }),
            InstKind::Store { val, ptr, .. } => {
                out.push(MInst::Store { bits: self.bits(&val)?, val: self.val(&val)?, addr: Addr::simple(self.val(&ptr)?) });
            }
            InstKind::Alloca { ty, count, align } => {
                if !self.is_entry { return Err("alloca outside the entry block is not supported yet".into()); }
                let Operand::Const(Constant::Int { value, .. }) = count else {
                    return Err("dynamic alloca is not supported yet".into());
                };
                let (size, al) = size_align(self.m, ty)?;
                self.slots.push((size * value as u64, al.max(align as u64).max(1)));
                out.push(MInst::Lea { dst: dst()?, slot: self.slots.len() - 1 });
            }
            InstKind::Gep { base_ty, ptr, indices, .. } => self.lower_gep(dst()?, base_ty, &ptr, &indices, out)?,
            InstKind::InsertValue { agg, val, indices } => {
                let ty = self.f.values[res.unwrap()].ty;
                let mut leaves = self.agg_leaves(&agg)?;
                let (start, count, vty) = leaf_index(self.m, ty, &indices)?;
                if count == 1 && !matches!(vty, Type::Agg(_)) {
                    leaves[start] = self.val(&val)?;
                } else {
                    let sub = self.agg_leaves(&val)?;
                    leaves.splice(start..start + count, sub);
                }
                self.aggs.insert(res.unwrap(), leaves);
            }
            InstKind::VaStart { list } => {
                if !self.f.sig.variadic { return Err("va_start in a non-variadic function".into()); }
                let (named_gp, named_fp, named_stack) = self.named;
                if self.va.is_none() {
                    // Register save area: 6 GPRs + 8 XMM (x86-64) or 8 GPRs + 8 Q regs (AArch64).
                    let size = if self.abi.va_list_size == 24 { 176 } else { 192 };
                    self.slots.push((size, 16));
                    self.va = Some(VaInfo { named_gp, named_fp, named_stack, save_slot: self.slots.len() - 1 });
                }
                out.push(MInst::VaStart { addr: self.val(&list)? });
            }
            InstKind::VaCopy { dst, src } => {
                let (d, s) = (self.val(&dst)?, self.val(&src)?);
                for off in (0..self.abi.va_list_size).step_by(8) {
                    let (da, sa, t) = (self.tmp(), self.tmp(), self.tmp());
                    out.push(MInst::Alu { op: Alu::Add, bits: 64, dst: sa, lhs: s.clone(), rhs: Val::Imm(off) });
                    out.push(MInst::Load { dst: t, bits: 64, addr: Addr::simple(Val::V(sa)) });
                    out.push(MInst::Alu { op: Alu::Add, bits: 64, dst: da, lhs: d.clone(), rhs: Val::Imm(off) });
                    out.push(MInst::Store { bits: 64, val: Val::V(t), addr: Addr::simple(Val::V(da)) });
                }
            }
            InstKind::ExtractValue { agg, indices } => {
                let ty = self.f.operand_type(&agg);
                let leaves = self.agg_leaves(&agg)?;
                let (start, count, rty) = leaf_index(self.m, ty, &indices)?;
                if matches!(rty, Type::Agg(_)) {
                    self.aggs.insert(res.unwrap(), leaves[start..start + count].to_vec());
                } else {
                    out.push(MInst::Mov { dst: dst()?, src: leaves[start].clone() });
                }
            }
            InstKind::Call { callee, args, ret_ty } => {
                let sig = if let Callee::Direct(fid) = &callee { Some(self.m.functions[*fid].sig.clone()) } else { None };
                let target = match callee {
                    Callee::Direct(fid) => {
                        let f = &self.m.functions[fid];
                        CallTarget::Direct { name: f.name.clone(), external: f.is_declaration() }
                    }
                    Callee::Indirect(o) => CallTarget::Indirect(self.val(&o)?),
                };
                let mut targs = Vec::new();
                for (i, a) in args.iter().enumerate() {
                    let class = match &sig {
                        Some(sg) if sg.sret && i == 0 => ArgClass::Sret,
                        Some(sg) if sg.byval.iter().any(|b| b.0 == i) => {
                            let &(_, size, align) = sg.byval.iter().find(|b| b.0 == i).unwrap();
                            if !self.abi.byval { return Err("by-value memory arguments are not supported on this target".into()); }
                            ArgClass::Mem { size, align }
                        }
                        _ => class_of(self.f.operand_type(a))?,
                    };
                    targs.push(Arg { val: self.val(a)?, class });
                }
                let dsts = match ret_ty {
                    Type::Void => vec![],
                    Type::Agg(_) => {
                        let mut lt = Vec::new();
                        leaf_types(self.m, ret_ty, &mut lt)?;
                        let classes = lt.iter().map(|t| class_of(*t)).collect::<Result<Vec<_>, _>>()?;
                        classify_returns(&classes, self.abi)?;
                        let temps: Vec<VReg> = lt.iter().map(|_| self.tmp()).collect();
                        if let Some(r) = res { self.aggs.insert(r, temps.iter().map(|t| Val::V(*t)).collect()); }
                        let mut v = Vec::new();
                        for ((t, tmp), c) in lt.iter().zip(&temps).zip(classes) { v.push((*tmp, bits_of(*t)?, c)); }
                        v
                    }
                    t => vec![(dst()?, bits_of(t)?, class_of(t)?)],
                };
                out.push(MInst::Call { dsts, target, args: targs });
            }
        }
        Ok(())
    }

    fn float_binary(&mut self, op: BinOp, ty: Type, dst: VReg, l: Val, r: Val, out: &mut Vec<MInst>) -> Result<(), String> {
        let dbl = ty == Type::F64;
        let fop = match op {
            BinOp::FAdd => FOp::Add, BinOp::FSub => FOp::Sub, BinOp::FMul => FOp::Mul, BinOp::FDiv => FOp::Div,
            BinOp::FRem => {
                let class = if dbl { ArgClass::F64 } else { ArgClass::F32 };
                out.push(MInst::Call {
                    dsts: vec![(dst, if dbl { 64 } else { 32 }, class)],
                    target: CallTarget::Direct { name: (if dbl { "fmod" } else { "fmodf" }).into(), external: true },
                    args: vec![Arg { val: l, class }, Arg { val: r, class }],
                });
                return Ok(());
            }
            _ => return Err(format!("integer operation {op:?} on floating-point type")),
        };
        out.push(MInst::FAlu { op: fop, dbl, dst, lhs: l, rhs: r });
        Ok(())
    }

    fn lower_gep(&mut self, dst: VReg, base_ty: Type, ptr: &Operand, indices: &[Operand], out: &mut Vec<MInst>) -> Result<(), String> {
        let mut acc = self.val(ptr)?;
        let mut const_off: i64 = 0;
        let mut cur = base_ty;
        for (n, idx) in indices.iter().enumerate() {
            let scale: u64;
            if n == 0 {
                scale = size_align(self.m, cur)?.0;
            } else {
                match cur {
                    Type::Agg(id) => match self.m.aggregate(id) {
                        AggregateType::Struct { fields, packed } => {
                            let Operand::Const(Constant::Int { value, .. }) = idx else {
                                return Err("struct GEP index must be constant".into());
                            };
                            let (offs, _, _) = struct_layout(self.m, fields, *packed)?;
                            let k = *value as usize;
                            let field = *fields.get(k).ok_or("struct GEP index out of range")?;
                            const_off += offs[k] as i64;
                            cur = field;
                            continue;
                        }
                        AggregateType::Array { elem, .. } | AggregateType::Vector { elem, .. } => {
                            scale = size_align(self.m, *elem)?.0;
                            cur = *elem;
                        }
                    },
                    t => return Err(format!("cannot index into {t:?} in GEP")),
                }
            }
            match idx {
                Operand::Const(Constant::Int { bits, value }) => {
                    const_off = const_off.wrapping_add(sx64(*value as u64, *bits).wrapping_mul(scale as i64));
                }
                other => {
                    let bits = self.bits(other)?;
                    let mut v = self.val(other)?;
                    if bits < 64 {
                        let t = self.tmp();
                        out.push(MInst::Ext { kind: ExtKind::Sext, from: bits, to: 64, dst: t, src: v });
                        v = Val::V(t);
                    }
                    if scale != 1 {
                        let t = self.tmp();
                        out.push(MInst::Alu { op: Alu::Mul, bits: 64, dst: t, lhs: v, rhs: Val::Imm(scale) });
                        v = Val::V(t);
                    }
                    let t = self.tmp();
                    out.push(MInst::Alu { op: Alu::Add, bits: 64, dst: t, lhs: acc, rhs: v });
                    acc = Val::V(t);
                }
            }
        }
        if const_off != 0 {
            let t = self.tmp();
            out.push(MInst::Alu { op: Alu::Add, bits: 64, dst: t, lhs: acc, rhs: Val::Imm(const_off as u64) });
            acc = Val::V(t);
        }
        out.push(MInst::Mov { dst, src: acc });
        Ok(())
    }
}

pub fn lower_function(m: &Module, src: &Function, abi: Abi) -> Result<MFunc, String> {
    let mut f = src.clone();
    split_critical_edges(&mut f);
    let bidx: HashMap<BlockId, usize> = f.layout.iter().enumerate().map(|(i, &b)| (b, i)).collect();
    let layout = f.layout.clone();
    let sret_vreg = if f.sig.sret { f.params.first().map(|p| p.index() as u32) } else { None };
    let mut lw = Lowerer { m, next_vreg: f.values.len() as u32, f, slots: vec![], is_entry: true, abi, aggs: HashMap::new(), sret_vreg, va: None, named: (0, 0, 0) };
    let mut blocks = Vec::new();
    for (bi, &b) in layout.iter().enumerate() {
        lw.is_entry = bi == 0;
        let mut out = Vec::new();
        if bi == 0 {
            let ptys: Vec<Type> = lw.f.params.iter().map(|&p| lw.f.values[p].ty).collect();
            let mut classes = ptys.iter().map(|&t| class_of(t)).collect::<Result<Vec<_>, _>>()?;
            if lw.f.sig.sret { classes[0] = ArgClass::Sret; }
            for &(k, size, align) in &lw.f.sig.byval.clone() {
                if !abi.byval { return Err(format!("in @{}: by-value memory parameters are not supported on this target", src.name)); }
                classes[k] = ArgClass::Mem { size, align };
            }
            let locs = classify(&classes, abi);
            lw.named = (
                locs.iter().filter(|l| matches!(l, ArgLoc::IntReg(_))).count(),
                locs.iter().filter(|l| matches!(l, ArgLoc::FpReg(_))).count(),
                stack_slots(&classes, &locs),
            );
            for (k, &p) in lw.f.params.clone().iter().enumerate() {
                out.push(MInst::Param { dst: p.index() as u32, class: classes[k], loc: locs[k], bits: bits_of(ptys[k])? });
            }
        }
        for i in lw.f.blocks[b].insts.clone() {
            lw.lower_inst(i, &mut out).map_err(|e| format!("in @{}: {e}", src.name))?;
        }
        let term = lw.f.blocks[b].terminator.clone().ok_or_else(|| format!("in @{}: {b} has no terminator", src.name))?;

        // Phi copies for the (single) successor, via temporaries so that
        // phis reading each other's old values stay correct.
        let mut succs = term.successors();
        succs.dedup();
        if let [s] = succs[..] {
            let mut pairs = Vec::new();
            for &i in &lw.f.blocks[s].insts.clone() {
                if let InstKind::Phi { incoming, .. } = &lw.f.insts[i].kind {
                    let (_, o) = incoming.iter().find(|(pb, _)| *pb == b)
                        .ok_or_else(|| format!("in @{}: phi in {s} has no incoming value for {b}", src.name))?;
                    pairs.push((lw.f.insts[i].result.unwrap().index() as u32, lw.val(o)?));
                }
            }
            pairs.retain(|(d, v)| *v != Val::V(*d)); // self-copies are no-ops
            let hazard = pairs.iter().any(|(_, v)| matches!(v, Val::V(x) if pairs.iter().any(|(d, _)| d == x)));
            if !hazard {
                for (d, v) in &pairs { out.push(MInst::Mov { dst: *d, src: v.clone() }); }
            } else {
                // Some copy reads a register another copy writes: go through temporaries.
                let tmps: Vec<VReg> = pairs.iter().map(|_| lw.tmp()).collect();
                for (t, (_, v)) in tmps.iter().zip(&pairs) { out.push(MInst::Mov { dst: *t, src: v.clone() }); }
                for (t, (p, _)) in tmps.iter().zip(&pairs) { out.push(MInst::Mov { dst: *p, src: Val::V(*t) }); }
            }
        }

        out.push(match term {
            // A struct-returning function hands the hidden pointer back (required by the x86-64 ABI).
            Terminator::Ret(None) => MInst::Ret(match lw.sret_vreg {
                Some(v) => vec![Arg { val: Val::V(v), class: ArgClass::Int }],
                None => vec![],
            }),
            Terminator::Ret(Some(o)) => {
                let ty = lw.f.operand_type(&o);
                if let Type::Agg(_) = ty {
                    let mut lt = Vec::new();
                    leaf_types(m, ty, &mut lt).map_err(|e| format!("in @{}: {e}", src.name))?;
                    let leaves = lw.agg_leaves(&o).map_err(|e| format!("in @{}: {e}", src.name))?;
                    let args: Vec<Arg> = lt.iter().zip(leaves).map(|(t, v)| Ok(Arg { val: v, class: class_of(*t)? })).collect::<Result<_, String>>()?;
                    let classes: Vec<ArgClass> = args.iter().map(|a| a.class).collect();
                    classify_returns(&classes, abi).map_err(|e| format!("in @{}: {e}", src.name))?;
                    MInst::Ret(args)
                } else {
                    MInst::Ret(vec![Arg { val: lw.val(&o)?, class: class_of(ty)? }])
                }
            }
            Terminator::Br(t) => MInst::Jmp(bidx[&t]),
            Terminator::CondBr { cond, then_bb, else_bb } => MInst::Br { cond: lw.val(&cond)?, then_bb: bidx[&then_bb], else_bb: bidx[&else_bb] },
            Terminator::Switch { val, default, cases } => {
                let bits = lw.bits(&val)?;
                MInst::Switch {
                    val: lw.val(&val)?, bits, default: bidx[&default],
                    cases: cases.iter().map(|(v, b)| (*v as u64 & mask64(bits), bidx[b])).collect(),
                }
            }
            Terminator::Unreachable => MInst::Trap,
        });
        blocks.push(MBlock { insts: out });
    }
    Ok(MFunc {
        name: src.name.clone(),
        global: src.linkage == llcr::Linkage::External,
        blocks, slots: lw.slots, num_vregs: lw.next_vreg, va: lw.va,
    })
}
