// SPDX-License-Identifier: GPL-2.0-only WITH LicenseRef-Kaiba-Runtime-Exception
// Copyright (C) The Kaiba Authors
//
// This file is part of the Kaiba compiler toolchain. It is licensed under
// GPL-2.0-only with the Kaiba Runtime/Output Exception; see COPYING and
// LICENSE-EXCEPTION at the repository root.
//! x86_64 System V (Linux, ELF) code generator for llcr.
//!
//! Pipeline: llcr -> [critical-edge split, phi copies] -> machine IR ->
//! linear-scan register allocation -> GNU as (Intel syntax) text.
//!
//! Supported: integers (i1/i8/i16/i32/i64), pointers, loads/stores, static
//! allocas, GEP over structs/arrays, calls (direct/indirect, any arity),
//! all integer arithmetic/compares/casts, scalar float arithmetic/compares/
//! conversions, select, switch, globals.
//! Floating point (f32/f64) uses scalar SSE: values live in the integer
//! register file as bit patterns and pass through xmm0/xmm1 for each operation.
//! Not yet supported: aggregates in registers, dynamic alloca, vectors.

pub mod emit;
pub mod regs;

pub use kaiba_mir::{layout, lower, mir, regalloc};

/// System V x86_64: 6 integer and 8 SSE argument registers.
pub const X86_ABI: mir::Abi = mir::Abi {
    int_regs: 6, fp_regs: 8, int_ret_regs: 2, fp_ret_regs: 2, sret_in_arg0: true, byval: true, va_list_size: 24,
};

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

/// Compile a module to x86_64 assembly (GNU as, Intel syntax).
pub fn compile_module(m: &Module) -> Result<String, CodegenError> {
    let mut out = String::new();
    let _ = writeln!(out, ".intel_syntax noprefix");
    let _ = writeln!(out, ".file \"{}\"", m.name);
    out.push_str(&kaiba_mir::data::emit_globals(m, '@')?);
    for (_, f) in m.functions.iter() {
        if f.is_declaration() { continue; }
        let mut mf = lower::lower_function(m, f, X86_ABI)?;
        kaiba_mir::opt::optimize(&mut mf);
        let alloc = regalloc::allocate(&mf, &regs::CALLEE_SAVED, &regs::CALLER_SAVED);
        out.push_str(&emit::emit_function(&mf, &alloc));
    }
    let _ = writeln!(out, ".section .note.GNU-stack,\"\",@progbits");
    Ok(out)
}
