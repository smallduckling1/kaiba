// SPDX-License-Identifier: GPL-2.0-only WITH LicenseRef-Kaiba-Runtime-Exception
// Copyright (C) The Kaiba Authors
//
// This file is part of the Kaiba compiler toolchain. It is licensed under
// GPL-2.0-only with the Kaiba Runtime/Output Exception; see COPYING and
// LICENSE-EXCEPTION at the repository root.

//! Functions: arenas of blocks, instructions and values plus block layout.

use crate::entity::{BlockId, InstId, PrimaryMap, ValueId};
use crate::inst::{InstKind, Operand, Terminator};
use crate::types::{Signature, Type};
use std::collections::HashMap;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Linkage { External, Internal, Private, Declaration }

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ValueDef { Param(u32), Inst(InstId) }

#[derive(Clone, Debug)]
pub struct ValueData {
    pub ty: Type,
    pub def: ValueDef,
}

#[derive(Clone, Debug)]
pub struct InstData {
    pub kind: InstKind,
    pub result: Option<ValueId>,
    pub block: BlockId,
}

#[derive(Clone, Debug, Default)]
pub struct BlockData {
    pub name: Option<String>,
    /// Non-terminator instructions in program order (phis first).
    pub insts: Vec<InstId>,
    pub terminator: Option<Terminator>,
}

#[derive(Clone, Debug)]
pub struct Function {
    pub name: String,
    pub sig: Signature,
    pub linkage: Linkage,
    pub params: Vec<ValueId>,
    pub blocks: PrimaryMap<BlockId, BlockData>,
    pub insts: PrimaryMap<InstId, InstData>,
    pub values: PrimaryMap<ValueId, ValueData>,
    /// Block order; the first entry is the entry block. Blocks removed from
    /// here are dead (arena slots are tombstoned, never reused).
    pub layout: Vec<BlockId>,
}

impl Function {
    pub fn new(name: impl Into<String>, sig: Signature, linkage: Linkage) -> Self {
        let mut values = PrimaryMap::new();
        let params = sig
            .params
            .iter()
            .enumerate()
            .map(|(i, &ty)| values.push(ValueData { ty, def: ValueDef::Param(i as u32) }))
            .collect();
        Function {
            name: name.into(), sig, linkage, params,
            blocks: PrimaryMap::new(), insts: PrimaryMap::new(), values,
            layout: Vec::new(),
        }
    }

    pub fn is_declaration(&self) -> bool { self.layout.is_empty() }
    pub fn entry(&self) -> Option<BlockId> { self.layout.first().copied() }

    pub fn create_block(&mut self, name: Option<&str>) -> BlockId {
        let b = self.blocks.push(BlockData { name: name.map(str::to_owned), ..Default::default() });
        self.layout.push(b);
        b
    }

    /// Append an instruction to `block`. Returns its result value (if the
    /// instruction produces one) alongside the instruction id.
    pub fn append_inst(&mut self, block: BlockId, kind: InstKind, result_ty: Option<Type>) -> (InstId, Option<ValueId>) {
        let inst = self.insts.next_key();
        let result = result_ty
            .filter(|t| *t != Type::Void)
            .map(|ty| self.values.push(ValueData { ty, def: ValueDef::Inst(inst) }));
        let id = self.insts.push(InstData { kind, result, block });
        debug_assert_eq!(id, inst);
        self.blocks[block].insts.push(id);
        (id, result)
    }

    pub fn set_terminator(&mut self, block: BlockId, term: Terminator) {
        self.blocks[block].terminator = Some(term);
    }

    pub fn remove_inst(&mut self, inst: InstId) {
        let b = self.insts[inst].block;
        self.blocks[b].insts.retain(|&i| i != inst);
    }

    pub fn value_type(&self, v: ValueId) -> Type { self.values[v].ty }

    pub fn successors(&self, b: BlockId) -> Vec<BlockId> {
        self.blocks[b].terminator.as_ref().map(Terminator::successors).unwrap_or_default()
    }

    /// Compute predecessor lists for all live blocks (not cached).
    pub fn predecessors(&self) -> HashMap<BlockId, Vec<BlockId>> {
        let mut preds: HashMap<BlockId, Vec<BlockId>> = self.layout.iter().map(|&b| (b, vec![])).collect();
        for &b in &self.layout {
            for s in self.successors(b) {
                preds.entry(s).or_default().push(b);
            }
        }
        preds
    }

    /// Iterate live instructions in layout order.
    pub fn iter_insts(&self) -> impl Iterator<Item = (BlockId, InstId)> + '_ {
        self.layout.iter().flat_map(move |&b| self.blocks[b].insts.iter().map(move |&i| (b, i)))
    }

    /// Visit every operand of every live instruction and terminator.
    pub fn for_each_operand_mut(&mut self, mut f: impl FnMut(&mut Operand)) {
        for bi in 0..self.layout.len() {
            let b = self.layout[bi];
            for ii in 0..self.blocks[b].insts.len() {
                let i = self.blocks[b].insts[ii];
                self.insts[i].kind.for_each_operand_mut(&mut f);
            }
            if let Some(t) = self.blocks[b].terminator.as_mut() {
                t.for_each_operand_mut(&mut f);
            }
        }
    }

    /// Replace all uses of `old` with `new`. Returns the number of uses rewritten.
    pub fn replace_all_uses(&mut self, old: ValueId, new: Operand) -> usize {
        let mut n = 0;
        self.for_each_operand_mut(|o| {
            if *o == Operand::Value(old) { *o = new; n += 1; }
        });
        n
    }

    /// Number of uses of each value, across live code.
    pub fn use_counts(&self) -> HashMap<ValueId, usize> {
        let mut m = HashMap::new();
        let mut bump = |o: &Operand| {
            if let Operand::Value(v) = o { *m.entry(*v).or_insert(0) += 1; }
        };
        for (_, i) in self.iter_insts() { self.insts[i].kind.for_each_operand(&mut bump); }
        for &b in &self.layout {
            if let Some(t) = &self.blocks[b].terminator { t.for_each_operand(&mut bump); }
        }
        m
    }
}

impl Function {
    /// The type of an operand. Globals and function addresses are pointers.
    pub fn operand_type(&self, op: &Operand) -> Type {
        use crate::inst::Constant;
        match op {
            Operand::Value(v) => self.values[*v].ty,
            Operand::Const(Constant::Int { bits, .. }) => Type::Int(*bits),
            Operand::Const(Constant::F32(_)) => Type::F32,
            Operand::Const(Constant::F64(_)) => Type::F64,
            Operand::Const(Constant::Null) => Type::PTR,
            Operand::Const(Constant::Undef(t)) | Operand::Const(Constant::Poison(t)) => *t,
            Operand::Global(_) | Operand::Func(_) => Type::PTR,
        }
    }
}

impl Function {
    /// Insert an instruction at the very start of `block` (used for phis).
    pub fn prepend_inst(&mut self, block: BlockId, kind: InstKind, result_ty: Option<Type>) -> (InstId, Option<ValueId>) {
        let r = self.append_inst(block, kind, result_ty);
        let v = &mut self.blocks[block].insts;
        let id = v.pop().unwrap();
        v.insert(0, id);
        r
    }
}
