// SPDX-License-Identifier: GPL-2.0-only WITH LicenseRef-Kaiba-Runtime-Exception
// Copyright (C) The Kaiba Authors
//
// This file is part of the Kaiba compiler toolchain. It is licensed under
// GPL-2.0-only with the Kaiba Runtime/Output Exception; see COPYING and
// LICENSE-EXCEPTION at the repository root.
//! Optimization passes, plus the name registry and standard pipelines.

pub mod constfold;
pub mod dce;
pub mod inline;
pub mod licm;
pub mod mem2reg;
pub mod simplifycfg;

use crate::pass::{FunctionPass, ModulePass, PassManager};

/// Names accepted by [`create_pass`].
pub const PASS_NAMES: &[&str] = &["inline", "mem2reg", "simplifycfg", "constfold", "licm", "dce"];

pub enum AnyPass {
    Function(Box<dyn FunctionPass>),
    Module(Box<dyn ModulePass>),
}

pub fn create_pass(name: &str) -> Option<AnyPass> {
    Some(match name {
        "inline" => AnyPass::Module(Box::new(inline::Inline::default())),
        "mem2reg" => AnyPass::Function(Box::new(mem2reg::Mem2Reg)),
        "simplifycfg" => AnyPass::Function(Box::new(simplifycfg::SimplifyCfg)),
        "constfold" => AnyPass::Function(Box::new(constfold::ConstantFolding)),
        "licm" => AnyPass::Function(Box::new(licm::Licm)),
        "dce" => AnyPass::Function(Box::new(dce::DeadCodeElimination)),
        _ => return None,
    })
}

pub fn add_pass(pm: &mut PassManager, p: AnyPass) {
    match p {
        AnyPass::Function(f) => { pm.add_function_pass(f); }
        AnyPass::Module(m) => { pm.add_module_pass(m); }
    }
}

/// Names making up the `-O1` pipeline, in order.
pub const O1_PIPELINE: &[&str] = &["inline", "mem2reg", "simplifycfg", "constfold", "simplifycfg", "licm", "dce"];

pub fn add_o1_pipeline(pm: &mut PassManager) {
    for n in O1_PIPELINE { add_pass(pm, create_pass(n).expect("registered pass")); }
}
