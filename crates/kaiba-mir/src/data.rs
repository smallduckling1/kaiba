// SPDX-License-Identifier: GPL-2.0-only WITH LicenseRef-Kaiba-Runtime-Exception
// Copyright (C) The Kaiba Authors
//
// This file is part of the Kaiba compiler toolchain. It is licensed under
// GPL-2.0-only with the Kaiba Runtime/Output Exception; see COPYING and
// LICENSE-EXCEPTION at the repository root.
//! Emission of global variables as GNU-as data directives (shared by all backends).

use crate::layout::size_align;
use llcr::{Linkage, Module, RelocTarget};
use std::fmt::Write;

/// `type_char` introduces the `.type` kind: `@` on x86, `%` on ARM-family targets
/// (where `@` starts a comment).
pub fn emit_globals(m: &Module, type_char: char) -> Result<String, String> {
    let mut out = String::new();
    for (_, g) in m.globals.iter() {
        if g.linkage == Linkage::Declaration { continue; }
        let (size, align) = size_align(m, g.ty)?;
        let init = g.init.clone().unwrap_or_default();
        if init.len() as u64 > size {
            return Err(format!("initializer of @{} is larger than its type", g.name));
        }
        let has_relocs = !g.relocs.is_empty();
        let section = if g.is_const {
            if has_relocs { ".section .data.rel.ro,\"aw\"" } else { ".section .rodata" }
        } else if g.init.is_some() || has_relocs { ".data" } else { ".bss" };
        let _ = writeln!(out, "{section}");
        let _ = writeln!(out, ".balign {}", align.max(g.align as u64).max(1));
        if g.linkage == Linkage::External { let _ = writeln!(out, ".globl \"{}\"", g.name); }
        let _ = writeln!(out, ".type \"{0}\", {type_char}object\n.size \"{0}\", {size}\n\"{0}\":", g.name);
        if g.init.is_some() || g.is_const || has_relocs {
            let (mut i, mut zeros) = (0u64, 0u64);
            let flush = |out: &mut String, zeros: &mut u64| {
                if *zeros > 0 { let _ = writeln!(out, "    .zero {zeros}"); *zeros = 0; }
            };
            while i < size {
                if let Some(r) = g.relocs.iter().find(|r| r.offset == i) {
                    flush(&mut out, &mut zeros);
                    let name = match r.target {
                        RelocTarget::Global(id) => &m.globals[id].name,
                        RelocTarget::Func(id) => &m.functions[id].name,
                    };
                    let add = if r.addend == 0 { String::new() } else { format!("{:+}", r.addend) };
                    let _ = writeln!(out, "    .quad \"{name}\"{add}");
                    i += 8;
                } else {
                    let b = init.get(i as usize).copied().unwrap_or(0);
                    if b == 0 { zeros += 1; } else { flush(&mut out, &mut zeros); let _ = writeln!(out, "    .byte {b}"); }
                    i += 1;
                }
            }
            flush(&mut out, &mut zeros);
        } else {
            let _ = writeln!(out, "    .zero {size}");
        }
    }
    Ok(out)
}
