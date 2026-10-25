// SPDX-License-Identifier: GPL-2.0-only WITH LicenseRef-Kaiba-Runtime-Exception
// Copyright (C) The Kaiba Authors
//
// This file is part of the Kaiba compiler toolchain. It is licensed under
// GPL-2.0-only with the Kaiba Runtime/Output Exception; see COPYING and
// LICENSE-EXCEPTION at the repository root.
//! AArch64 (AAPCS64, Linux/ELF) code generator for llcr.
//!
//! Shares instruction selection and register allocation with the x86 backend
//! (`kaiba-mir`). Supports integers, pointers, scalar f32/f64, static allocas,
//! GEP, calls (any arity, direct/indirect), switch/select, globals (with
//! relocations). Not supported yet: aggregates in registers, dynamic alloca.

pub mod emit;
pub mod regs;

use kaiba_mir::{lower, regalloc};
use llcr::Module;
use std::fmt::{self, Write};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CodegenError(pub String);

impl fmt::Display for CodegenError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { f.write_str(&self.0) }
}
impl std::error::Error for CodegenError {}
impl From<String> for CodegenError {
    fn from(s: String) -> Self { CodegenError(s) }
}

/// Compile a module to AArch64 assembly (GNU as syntax).
pub fn compile_module(m: &Module) -> Result<String, CodegenError> {
    let mut out = String::new();
    let _ = writeln!(out, ".file \"{}\"", m.name);
    out.push_str(&kaiba_mir::data::emit_globals(m, '%')?);
    for (_, f) in m.functions.iter() {
        if f.is_declaration() { continue; }
        let mut mf = lower::lower_function(m, f, regs::AARCH64_ABI)?;
        kaiba_mir::opt::optimize(&mut mf);
        let alloc = regalloc::allocate(&mf, &regs::CALLEE_SAVED, &regs::CALLER_SAVED);
        out.push_str(&emit::emit_function(&mf, &alloc));
    }
    let _ = writeln!(out, ".section .note.GNU-stack,\"\",%progbits");
    Ok(out)
}
