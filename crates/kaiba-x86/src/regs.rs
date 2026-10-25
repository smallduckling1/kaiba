// SPDX-License-Identifier: GPL-2.0-only WITH LicenseRef-Kaiba-Runtime-Exception
// Copyright (C) The Kaiba Authors
//
// This file is part of the Kaiba compiler toolchain. It is licensed under
// GPL-2.0-only with the Kaiba Runtime/Output Exception; see COPYING and
// LICENSE-EXCEPTION at the repository root.
//! x86_64 register definitions.

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PReg { Rax, Rcx, Rdx, Rbx, Rsp, Rbp, Rsi, Rdi, R8, R9, R10, R11, R12, R13, R14, R15 }

const NAMES: [[&str; 4]; 16] = [
    ["rax", "eax", "ax", "al"], ["rcx", "ecx", "cx", "cl"], ["rdx", "edx", "dx", "dl"], ["rbx", "ebx", "bx", "bl"],
    ["rsp", "esp", "sp", "spl"], ["rbp", "ebp", "bp", "bpl"], ["rsi", "esi", "si", "sil"], ["rdi", "edi", "di", "dil"],
    ["r8", "r8d", "r8w", "r8b"], ["r9", "r9d", "r9w", "r9b"], ["r10", "r10d", "r10w", "r10b"], ["r11", "r11d", "r11w", "r11b"],
    ["r12", "r12d", "r12w", "r12b"], ["r13", "r13d", "r13w", "r13b"], ["r14", "r14d", "r14w", "r14b"], ["r15", "r15d", "r15w", "r15b"],
];

impl PReg {
    /// Name of the register viewed as `bits` wide (1 is treated as 8).
    pub fn name(self, bits: u32) -> &'static str {
        let col = match bits { 64 => 0, 32 => 1, 16 => 2, _ => 3 };
        NAMES[self as usize][col]
    }
}

/// Callee-saved registers: values in them survive calls.
pub const CALLEE_SAVED: [PReg; 5] = [PReg::Rbx, PReg::R12, PReg::R13, PReg::R14, PReg::R15];

/// Caller-saved registers the allocator may use, but only for live ranges that
/// do not span a call. (rax/rcx/rdx/r10/r11 stay reserved as emitter scratch.)
pub const CALLER_SAVED: [PReg; 4] = [PReg::Rsi, PReg::Rdi, PReg::R8, PReg::R9];

/// System V integer argument registers, in order.
pub const ARG_REGS: [PReg; 6] = [PReg::Rdi, PReg::Rsi, PReg::Rdx, PReg::Rcx, PReg::R8, PReg::R9];
