// SPDX-License-Identifier: GPL-2.0-only WITH LicenseRef-Kaiba-Runtime-Exception
// Copyright (C) The Kaiba Authors
//
// This file is part of the Kaiba compiler toolchain. It is licensed under
// GPL-2.0-only with the Kaiba Runtime/Output Exception; see COPYING and
// LICENSE-EXCEPTION at the repository root.
//! Inliner for small, non-recursive direct calls, plus removal of internal
//! functions that become unreferenced.

use crate::entity::{BlockId, FuncId, InstId, ValueId};
use crate::function::{Function, Linkage};
use crate::inst::{Callee, InstKind, Operand, Terminator};
use crate::module::{Module, RelocTarget};
use crate::pass::{Changed, ModulePass};
use std::collections::{HashMap, HashSet};

pub struct Inline {
    /// Largest callee (in instructions) that will be inlined.
    pub threshold: usize,
    /// Safety valve on growth: inlined calls per caller per run.
    pub max_per_caller: usize,
}

impl Default for Inline {
    fn default() -> Self { Inline { threshold: 40, max_per_caller: 64 } }
}

fn inst_count(f: &Function) -> usize { f.layout.iter().map(|&b| f.blocks[b].insts.len()).sum() }

fn direct_callees(f: &Function) -> HashSet<FuncId> {
    f.iter_insts()
        .filter_map(|(_, i)| match &f.insts[i].kind { InstKind::Call { callee: Callee::Direct(g), .. } => Some(*g), _ => None })
        .collect()
}

/// Functions that can (transitively) call themselves.
fn recursive_set(m: &Module) -> HashSet<FuncId> {
    let graph: HashMap<FuncId, HashSet<FuncId>> = m.functions.iter().map(|(id, f)| (id, direct_callees(f))).collect();
    let mut rec = HashSet::new();
    for &start in graph.keys() {
        let (mut seen, mut stack) = (HashSet::new(), vec![start]);
        while let Some(x) = stack.pop() {
            for &y in &graph[&x] {
                if y == start { rec.insert(start); }
                if seen.insert(y) { stack.push(y); }
            }
        }
    }
    rec
}

fn eligible(m: &Module, g: FuncId, rec: &HashSet<FuncId>, threshold: usize) -> bool {
    let f = &m.functions[g];
    if f.is_declaration() || f.sig.variadic || rec.contains(&g) || inst_count(f) > threshold { return false; }
    // Special calling conventions (hidden return pointer, by-value memory args, aggregate returns) stay as calls.
    if f.sig.sret || !f.sig.byval.is_empty() || matches!(f.sig.ret, crate::types::Type::Agg(_)) { return false; }
    // Allocas must stay in the entry block of the caller; only the callee's entry allocas can be hoisted there.
    f.layout.iter().skip(1).all(|&b| f.blocks[b].insts.iter().all(|&i| !matches!(f.insts[i].kind, InstKind::Alloca { .. })))
}

/// Replace the call `call` in `f` with a copy of `callee`'s body.
fn inline_call(f: &mut Function, call: InstId, callee: &Function) {
    let InstKind::Call { args, ret_ty, .. } = f.insts[call].kind.clone() else { unreachable!() };
    let call_result = f.insts[call].result;
    let b = f.insts[call].block;
    let f_entry = f.layout[0];
    let old_len = f.layout.len();

    // 1. Split `b` at the call: instructions after it, and b's terminator, move to `cont`.
    let pos = f.blocks[b].insts.iter().position(|&i| i == call).unwrap();
    let tail = f.blocks[b].insts.split_off(pos + 1);
    f.blocks[b].insts.pop(); // the call itself
    let cont = f.create_block(Some("inline.cont"));
    for &i in &tail { f.insts[i].block = cont; }
    f.blocks[cont].insts = tail;
    f.blocks[cont].terminator = f.blocks[b].terminator.take();
    let succs = f.successors(cont);
    for s in succs {
        for i in f.blocks[s].insts.clone() {
            if let InstKind::Phi { incoming, .. } = &mut f.insts[i].kind {
                for (pb, _) in incoming.iter_mut() { if *pb == b { *pb = cont; } }
            }
        }
    }

    // 2. Clone the callee's blocks and instructions.
    let mut bmap: HashMap<BlockId, BlockId> = HashMap::new();
    for &cb in &callee.layout { bmap.insert(cb, f.create_block(Some("inline"))); }
    let mut vmap: HashMap<ValueId, Operand> = HashMap::new();
    for (k, &p) in callee.params.iter().enumerate() { vmap.insert(p, args[k]); }
    let mut cloned: Vec<InstId> = Vec::new();
    for &cb in &callee.layout {
        for &ci in &callee.blocks[cb].insts {
            let d = &callee.insts[ci];
            let rty = d.result.map(|r| callee.values[r].ty);
            let (ni, nv) = f.append_inst(bmap[&cb], d.kind.clone(), rty);
            if let (Some(r), Some(nv)) = (d.result, nv) { vmap.insert(r, Operand::Value(nv)); }
            cloned.push(ni);
        }
    }
    let remap = |o: &mut Operand| {
        if let Operand::Value(v) = *o { *o = *vmap.get(&v).expect("callee value without a definition"); }
    };
    for &ni in &cloned {
        f.insts[ni].kind.for_each_operand_mut(|o| remap(o));
        if let InstKind::Phi { incoming, .. } = &mut f.insts[ni].kind {
            for (pb, _) in incoming.iter_mut() { *pb = bmap[pb]; }
        }
    }

    // 3. Terminators: returns become branches to `cont`.
    let mut rets: Vec<(BlockId, Operand)> = Vec::new();
    for &cb in &callee.layout {
        let nb = bmap[&cb];
        let mut t = callee.blocks[cb].terminator.clone().expect("callee block without terminator");
        t.for_each_operand_mut(|o| remap(o));
        match t {
            Terminator::Ret(v) => {
                if let Some(v) = v { rets.push((nb, v)); }
                f.set_terminator(nb, Terminator::Br(cont));
            }
            mut t => { t.map_successors(|x| bmap[&x]); f.set_terminator(nb, t); }
        }
    }
    f.set_terminator(b, Terminator::Br(bmap[&callee.layout[0]]));

    // 4. The call's result.
    if let Some(r) = call_result {
        let value = match rets.len() {
            0 => Operand::Const(crate::inst::Constant::Undef(ret_ty)),
            1 => rets[0].1,
            _ => {
                let (_, pv) = f.prepend_inst(cont, InstKind::Phi { ty: ret_ty, incoming: rets.clone() }, Some(ret_ty));
                Operand::Value(pv.unwrap())
            }
        };
        f.replace_all_uses(r, value);
    }

    // 5. Hoist the callee's entry-block allocas into the caller's entry block.
    let clone_entry = bmap[&callee.layout[0]];
    let allocas: Vec<InstId> = f.blocks[clone_entry].insts.iter().copied()
        .filter(|&i| matches!(f.insts[i].kind, InstKind::Alloca { .. })).collect();
    if !allocas.is_empty() {
        f.blocks[clone_entry].insts.retain(|i| !allocas.contains(i));
        for &a in &allocas { f.insts[a].block = f_entry; }
        let entry_insts = &mut f.blocks[f_entry].insts;
        for (k, &a) in allocas.iter().enumerate() { entry_insts.insert(k, a); }
    }

    // 6. Layout: keep the new blocks right after `b` (short live ranges downstream).
    let mut new_blocks: Vec<BlockId> = f.layout.split_off(old_len);
    // `cont` was created first; it must come after the inlined body.
    let cont_pos = new_blocks.iter().position(|&x| x == cont).unwrap();
    new_blocks.remove(cont_pos);
    new_blocks.push(cont);
    let at = f.layout.iter().position(|&x| x == b).unwrap() + 1;
    f.layout.splice(at..at, new_blocks);
}

fn referenced_functions(m: &Module) -> HashSet<FuncId> {
    let mut r = HashSet::new();
    for (_, f) in m.functions.iter() {
        for (_, i) in f.iter_insts() {
            if let InstKind::Call { callee: Callee::Direct(g), .. } = &f.insts[i].kind { r.insert(*g); }
            f.insts[i].kind.for_each_operand(|o| { if let Operand::Func(g) = o { r.insert(*g); } });
        }
        for &b in &f.layout {
            if let Some(t) = &f.blocks[b].terminator { t.for_each_operand(|o| { if let Operand::Func(g) = o { r.insert(*g); } }); }
        }
    }
    for (_, g) in m.globals.iter() {
        for rel in &g.relocs { if let RelocTarget::Func(id) = rel.target { r.insert(id); } }
    }
    r
}

impl ModulePass for Inline {
    fn name(&self) -> &'static str { "inline" }

    fn run(&mut self, m: &mut Module) -> Changed {
        let mut changed = false;
        let rec = recursive_set(m);
        let ids: Vec<FuncId> = m.functions.keys().collect();
        for &fid in &ids {
            if m.functions[fid].is_declaration() { continue; }
            let mut budget = self.max_per_caller;
            'again: while budget > 0 {
                let site = {
                    let f = &m.functions[fid];
                    f.iter_insts().find_map(|(_, i)| match &f.insts[i].kind {
                        InstKind::Call { callee: Callee::Direct(g), .. } if *g != fid && eligible(m, *g, &rec, self.threshold) => Some((i, *g)),
                        _ => None,
                    })
                };
                let Some((call, g)) = site else { break 'again };
                let callee = m.functions[g].clone();
                inline_call(&mut m.functions[fid], call, &callee);
                budget -= 1;
                changed = true;
            }
        }
        // Drop internal functions nobody refers to any more (keep the symbol as a body-less declaration).
        loop {
            let used = referenced_functions(m);
            let dead: Vec<FuncId> = m.functions.iter()
                .filter(|(id, f)| matches!(f.linkage, Linkage::Internal | Linkage::Private) && !f.is_declaration() && !used.contains(id))
                .map(|(id, _)| id).collect();
            if dead.is_empty() { break; }
            for id in dead {
                let f = &mut m.functions[id];
                f.layout.clear();
                f.linkage = Linkage::Declaration;
                changed = true;
            }
        }
        Changed::from_bool(changed)
    }
}
