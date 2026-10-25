// SPDX-License-Identifier: GPL-2.0-only WITH LicenseRef-Kaiba-Runtime-Exception
// Copyright (C) The Kaiba Authors
//
// This file is part of the Kaiba compiler toolchain. It is licensed under
// GPL-2.0-only with the Kaiba Runtime/Output Exception; see COPYING and
// LICENSE-EXCEPTION at the repository root.

//! Module: top-level container of functions, globals and interned aggregates.

use crate::entity::{AggId, FuncId, GlobalId, PrimaryMap};
use crate::function::{Function, Linkage};
use crate::types::{AggregateType, Type};
use std::collections::HashMap;

/// The symbol a pointer-sized relocation in a global initializer refers to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RelocTarget { Global(GlobalId), Func(FuncId) }

/// An 8-byte absolute address (`symbol + addend`) stored at `offset` in a global.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Reloc { pub offset: u64, pub target: RelocTarget, pub addend: i64 }

#[derive(Clone, Debug)]
pub struct GlobalVar {
    pub name: String,
    pub ty: Type,
    pub linkage: Linkage,
    pub is_const: bool,
    /// Raw initializer bytes; `None` means external / zero-initialised by linkage.
    pub init: Option<Vec<u8>>,
    /// Address relocations overlaid on `init` (the bytes at these offsets are ignored).
    pub relocs: Vec<Reloc>,
    pub align: u32,
}

#[derive(Clone, Debug, Default)]
pub struct Module {
    pub name: String,
    /// Target triple, e.g. `x86_64-unknown-linux-gnu`.
    pub target_triple: Option<String>,
    pub functions: PrimaryMap<FuncId, Function>,
    pub globals: PrimaryMap<GlobalId, GlobalVar>,
    aggregates: PrimaryMap<AggId, AggregateType>,
    agg_lookup: HashMap<AggregateType, AggId>,
}

impl Module {
    pub fn new(name: impl Into<String>) -> Self { Module { name: name.into(), ..Default::default() } }

    /// Intern an aggregate type; structurally equal types share an id.
    pub fn intern_aggregate(&mut self, agg: AggregateType) -> Type {
        if let Some(&id) = self.agg_lookup.get(&agg) { return Type::Agg(id); }
        let id = self.aggregates.push(agg.clone());
        self.agg_lookup.insert(agg, id);
        Type::Agg(id)
    }

    pub fn aggregate(&self, id: AggId) -> &AggregateType { &self.aggregates[id] }

    pub fn add_function(&mut self, f: Function) -> FuncId { self.functions.push(f) }
    pub fn add_global(&mut self, g: GlobalVar) -> GlobalId { self.globals.push(g) }

    pub fn find_function(&self, name: &str) -> Option<FuncId> {
        self.functions.iter().find(|(_, f)| f.name == name).map(|(id, _)| id)
    }
}
