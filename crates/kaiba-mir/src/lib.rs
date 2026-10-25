// SPDX-License-Identifier: GPL-2.0-only WITH LicenseRef-Kaiba-Runtime-Exception
// Copyright (C) The Kaiba Authors
//
// This file is part of the Kaiba compiler toolchain. It is licensed under
// GPL-2.0-only with the Kaiba Runtime/Output Exception; see COPYING and
// LICENSE-EXCEPTION at the repository root.
//! Target-independent backend pieces: machine IR (`mir`), llcr -> MIR
//! instruction selection with out-of-SSA conversion (`lower`), and linear-scan
//! register allocation generic over the register type (`regalloc`).

pub mod data;
pub mod layout;
pub mod lower;
pub mod mir;
pub mod opt;
pub mod regalloc;
