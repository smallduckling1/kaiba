// SPDX-License-Identifier: GPL-2.0-only WITH LicenseRef-Kaiba-Runtime-Exception
// Copyright (C) The Kaiba Authors
//
// This file is part of the Kaiba compiler toolchain. It is licensed under
// GPL-2.0-only with the Kaiba Runtime/Output Exception; see COPYING and
// LICENSE-EXCEPTION at the repository root.

//! llcr: Low-Level Code Representation.
//!
//! A strongly-typed SSA IR. Hierarchy: [`Module`] -> [`Function`] -> blocks ->
//! instructions + terminator. Everything is arena-allocated and referenced by
//! typed ids (see [`entity`]).

pub mod analysis;
pub mod entity;
pub mod function;
pub mod inst;
pub mod layout;
pub mod module;
pub mod opt;
pub mod parse;
pub mod pass;
pub mod print;
pub mod types;
pub mod verify;

pub use analysis::{reverse_postorder, DomTree};
pub use entity::{Entity, AggId, BlockId, FuncId, GlobalId, InstId, ValueId};
pub use function::{BlockData, Function, InstData, Linkage, ValueData, ValueDef};
pub use inst::{BinOp, Callee, CastOp, Constant, FloatPred, InstKind, IntPred, Operand, Terminator};
pub use module::{GlobalVar, Module, Reloc, RelocTarget};
pub use parse::{parse_module, ParseError};
pub use pass::{Changed, FunctionPass, ModulePass, PassManager};
pub use types::{AggregateType, Signature, Type};

#[cfg(test)]
mod tests {
    use super::*;
    use opt::dce::DeadCodeElimination;

    /// fn f(a: i32) -> i32 { let _unused = a * 2; a + 1 }
    #[test]
    fn build_and_dce() {
        let mut m = Module::new("t");
        let mut f = Function::new("f", Signature::new(vec![Type::I32], Type::I32), Linkage::External);
        let a = f.params[0];
        let bb = f.create_block(Some("entry"));
        let (_, _unused) = f.append_inst(bb, InstKind::Binary { op: BinOp::Mul, lhs: a.into(), rhs: Operand::int(32, 2) }, Some(Type::I32));
        let (_, sum) = f.append_inst(bb, InstKind::Binary { op: BinOp::Add, lhs: a.into(), rhs: Operand::int(32, 1) }, Some(Type::I32));
        f.set_terminator(bb, Terminator::Ret(Some(sum.unwrap().into())));
        let id = m.add_function(f);

        let mut pm = PassManager::new();
        pm.add_function_pass(DeadCodeElimination);
        assert!(pm.run(&mut m).is_changed());
        assert_eq!(m.functions[id].blocks[bb].insts.len(), 1);
    }

    #[test]
    fn aggregate_interning() {
        let mut m = Module::new("t");
        let s = AggregateType::Struct { fields: vec![Type::I32, Type::F64], packed: false };
        assert_eq!(m.intern_aggregate(s.clone()), m.intern_aggregate(s));
    }
}
