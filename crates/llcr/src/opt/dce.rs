// SPDX-License-Identifier: GPL-2.0-only WITH LicenseRef-Kaiba-Runtime-Exception
// Copyright (C) The Kaiba Authors
//
// This file is part of the Kaiba compiler toolchain. It is licensed under
// GPL-2.0-only with the Kaiba Runtime/Output Exception; see COPYING and
// LICENSE-EXCEPTION at the repository root.

//! Dead code elimination: the reference `FunctionPass`.

use crate::function::Function;
use crate::pass::{Changed, FunctionPass};

pub struct DeadCodeElimination;

impl FunctionPass for DeadCodeElimination {
    fn name(&self) -> &'static str { "dce" }

    fn run(&mut self, f: &mut Function) -> Changed {
        let mut changed = false;
        loop {
            let uses = f.use_counts();
            let dead: Vec<_> = f
                .iter_insts()
                .filter(|&(_, i)| {
                    let d = &f.insts[i];
                    !d.kind.has_side_effects()
                        && d.result.map_or(true, |v| uses.get(&v).copied().unwrap_or(0) == 0)
                })
                .map(|(_, i)| i)
                .collect();
            if dead.is_empty() { break; }
            for i in dead { f.remove_inst(i); }
            changed = true;
        }
        Changed::from_bool(changed)
    }
}
