// SPDX-License-Identifier: GPL-2.0-only WITH LicenseRef-Kaiba-Runtime-Exception
// Copyright (C) The Kaiba Authors
//
// This file is part of the Kaiba compiler toolchain. It is licensed under
// GPL-2.0-only with the Kaiba Runtime/Output Exception; see COPYING and
// LICENSE-EXCEPTION at the repository root.
//! AArch64 register definitions.

use kaiba_mir::mir::Abi;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct R(pub u8);

impl R {
    /// Name as a 64-bit (`x`) register.
    pub fn x(self) -> String { format!("x{}", self.0) }
    /// Name as a 32-bit (`w`) register.
    pub fn w(self) -> String { format!("w{}", self.0) }
    /// 64-bit name for `bits == 64`, otherwise the 32-bit name.
    pub fn name(self, bits: u32) -> String { if bits == 64 { self.x() } else { self.w() } }
}

/// Emitter scratch registers (never allocated, never live across instructions).
pub const S0: R = R(9);
pub const S1: R = R(10);
pub const S2: R = R(11);
/// Intra-procedure-call scratch: indirect call targets and large offsets.
pub const IP0: R = R(16);

pub const FP: R = R(29);

/// Callee-saved: x19..x28.
pub const CALLEE_SAVED: [R; 10] = [R(19), R(20), R(21), R(22), R(23), R(24), R(25), R(26), R(27), R(28)];

/// Caller-saved temporaries the allocator may use for call-free live ranges.
/// The argument registers x0..x7 are deliberately *not* allocatable, so argument
/// setup and parameter receipt never need parallel moves.
pub const CALLER_SAVED: [R; 4] = [R(12), R(13), R(14), R(15)];

/// AAPCS64: eight integer and eight floating-point argument registers.
pub const AARCH64_ABI: Abi = Abi {
    int_regs: 8, fp_regs: 8, int_ret_regs: 2, fp_ret_regs: 4, sret_in_arg0: false, byval: false, va_list_size: 32,
};
