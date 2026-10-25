// SPDX-License-Identifier: GPL-2.0-only WITH LicenseRef-Kaiba-Runtime-Exception
// Copyright (C) The Kaiba Authors
//
// This file is part of the Kaiba compiler toolchain. It is licensed under
// GPL-2.0-only with the Kaiba Runtime/Output Exception; see COPYING and
// LICENSE-EXCEPTION at the repository root.

//! Optimizer pass infrastructure.

use crate::function::Function;
use crate::module::Module;
use crate::verify::verify_function;

/// Whether a pass modified the IR.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Changed { Yes, No }

impl Changed {
    pub fn from_bool(b: bool) -> Self { if b { Changed::Yes } else { Changed::No } }
    pub fn is_changed(self) -> bool { self == Changed::Yes }
    pub fn merge(self, o: Changed) -> Changed { Changed::from_bool(self.is_changed() || o.is_changed()) }
}

/// A transformation or analysis over a single function.
pub trait FunctionPass {
    fn name(&self) -> &'static str;
    fn run(&mut self, func: &mut Function) -> Changed;
}

/// A transformation over a whole module (e.g. inlining, global DCE).
pub trait ModulePass {
    fn name(&self) -> &'static str;
    fn run(&mut self, module: &mut Module) -> Changed;
}

impl FunctionPass for Box<dyn FunctionPass> {
    fn name(&self) -> &'static str { (**self).name() }
    fn run(&mut self, func: &mut Function) -> Changed { (**self).run(func) }
}

impl ModulePass for Box<dyn ModulePass> {
    fn name(&self) -> &'static str { (**self).name() }
    fn run(&mut self, module: &mut Module) -> Changed { (**self).run(module) }
}

enum Entry {
    Function(Box<dyn FunctionPass>),
    Module(Box<dyn ModulePass>),
}

/// Runs an ordered pipeline of passes over a module.
#[derive(Default)]
pub struct PassManager {
    passes: Vec<Entry>,
    /// Run the verifier after every pass (default: on in debug builds).
    pub verify_each: bool,
    /// Optional trace of `(pass name, changed)` pairs.
    pub trace: Vec<(&'static str, bool)>,
}

impl PassManager {
    pub fn new() -> Self {
        PassManager { verify_each: cfg!(debug_assertions), ..Default::default() }
    }

    pub fn add_function_pass(&mut self, p: impl FunctionPass + 'static) -> &mut Self {
        self.passes.push(Entry::Function(Box::new(p)));
        self
    }

    pub fn add_module_pass(&mut self, p: impl ModulePass + 'static) -> &mut Self {
        self.passes.push(Entry::Module(Box::new(p)));
        self
    }

    pub fn run(&mut self, module: &mut Module) -> Changed {
        let mut any = Changed::No;
        let verify = self.verify_each;
        for entry in &mut self.passes {
            let (name, changed) = match entry {
                Entry::Function(p) => {
                    let mut c = Changed::No;
                    let ids: Vec<_> = module.functions.keys().collect();
                    for id in ids {
                        let f = &mut module.functions[id];
                        if f.is_declaration() { continue; }
                        c = c.merge(p.run(f));
                    }
                    (p.name(), c)
                }
                Entry::Module(p) => (p.name(), p.run(module)),
            };
            if verify && changed.is_changed() {
                for (_, f) in module.functions.iter().filter(|(_, f)| !f.is_declaration()) {
                    if let Err(e) = verify_function(f) {
                        panic!("IR verification failed after pass `{name}`: {}", e.0);
                    }
                }
            }
            self.trace.push((name, changed.is_changed()));
            any = any.merge(changed);
        }
        any
    }
}
