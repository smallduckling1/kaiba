// SPDX-License-Identifier: GPL-2.0-only WITH LicenseRef-Kaiba-Runtime-Exception
// Copyright (C) The Kaiba Authors
//
// This file is part of the Kaiba compiler toolchain. It is licensed under
// GPL-2.0-only with the Kaiba Runtime/Output Exception; see COPYING and
// LICENSE-EXCEPTION at the repository root.

//! llcr type system. Scalars are `Copy` and inline; aggregates are interned
//! in the [`Module`](crate::Module) and referenced by [`AggId`].

use crate::entity::AggId;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Type {
    Void,
    /// Integer of the given bit width (`Int(1)` is the boolean type).
    Int(u32),
    F32,
    F64,
    /// Opaque pointer in the given address space.
    Ptr(u32),
    /// Struct, array or vector, defined in the module's aggregate table.
    Agg(AggId),
}

impl Type {
    pub const I1: Type = Type::Int(1);
    pub const I8: Type = Type::Int(8);
    pub const I16: Type = Type::Int(16);
    pub const I32: Type = Type::Int(32);
    pub const I64: Type = Type::Int(64);
    pub const PTR: Type = Type::Ptr(0);

    pub fn is_int(self) -> bool { matches!(self, Type::Int(_)) }
    pub fn is_float(self) -> bool { matches!(self, Type::F32 | Type::F64) }
    pub fn is_ptr(self) -> bool { matches!(self, Type::Ptr(_)) }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum AggregateType {
    Struct { fields: Vec<Type>, packed: bool },
    Array { elem: Type, len: u64 },
    Vector { elem: Type, len: u32 },
}

/// Function signature.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Signature {
    pub params: Vec<Type>,
    pub ret: Type,
    pub variadic: bool,
    /// Parameter 0 is the hidden pointer to the caller-allocated return slot
    /// (the function itself returns void; the ABI decides where the pointer lives).
    pub sret: bool,
    /// Parameters passed by value in memory (`ptr` params whose pointee is copied
    /// onto the stack at the call): (param index, size in bytes, alignment).
    pub byval: Vec<(usize, u64, u64)>,
}

impl Signature {
    pub fn new(params: Vec<Type>, ret: Type) -> Self { Self { params, ret, variadic: false, sret: false, byval: Vec::new() } }
}
