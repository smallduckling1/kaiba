// SPDX-License-Identifier: GPL-2.0-only WITH LicenseRef-Kaiba-Runtime-Exception
// Copyright (C) The Kaiba Authors
//
// This file is part of the Kaiba compiler toolchain. It is licensed under
// GPL-2.0-only with the Kaiba Runtime/Output Exception; see COPYING and
// LICENSE-EXCEPTION at the repository root.
//! Data layout (x86_64 System V): sizes, alignments and struct offsets.

use crate::module::Module;
use crate::types::{AggregateType, Type};

fn align_up(x: u64, a: u64) -> u64 { (x + a - 1) / a * a }

pub fn size_align(m: &Module, t: Type) -> Result<(u64, u64), String> {
    Ok(match t {
        Type::Void => (0, 1),
        Type::Int(n) if matches!(n, 1 | 8) => (1, 1),
        Type::Int(16) => (2, 2),
        Type::Int(32) => (4, 4),
        Type::Int(64) => (8, 8),
        Type::Int(n) => return Err(format!("unsupported integer width i{n}")),
        Type::F32 => (4, 4),
        Type::F64 => (8, 8),
        Type::Ptr(_) => (8, 8),
        Type::Agg(id) => match m.aggregate(id) {
            AggregateType::Struct { fields, packed } => {
                let (_, size, align) = struct_layout(m, fields, *packed)?;
                (size, align)
            }
            AggregateType::Array { elem, len } => {
                let (s, a) = size_align(m, *elem)?;
                (s * len, a)
            }
            AggregateType::Vector { elem, len } => {
                let (s, a) = size_align(m, *elem)?;
                (s * *len as u64, a)
            }
        },
    })
}

/// Returns (field offsets, total size, alignment).
pub fn struct_layout(m: &Module, fields: &[Type], packed: bool) -> Result<(Vec<u64>, u64, u64), String> {
    let (mut off, mut max_align) = (0u64, 1u64);
    let mut offs = Vec::new();
    for &f in fields {
        let (s, a) = size_align(m, f)?;
        let a = if packed { 1 } else { a };
        off = align_up(off, a);
        offs.push(off);
        off += s;
        max_align = max_align.max(a);
    }
    Ok((offs, align_up(off, max_align), max_align))
}
