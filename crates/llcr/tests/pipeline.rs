// SPDX-License-Identifier: GPL-2.0-only WITH LicenseRef-Kaiba-Runtime-Exception
use llcr::opt::{constfold::ConstantFolding, dce::DeadCodeElimination};
use llcr::*;
use llcr::verify::verify_function;

/// entry: condbr c -> then/else; both -> merge: phi; ret
fn diamond() -> (Function, [BlockId; 4], ValueId, ValueId) {
    let mut f = Function::new("d", Signature::new(vec![Type::I1, Type::I32], Type::I32), Linkage::External);
    let (c, x) = (f.params[0], f.params[1]);
    let e = f.create_block(Some("entry"));
    let t = f.create_block(Some("then"));
    let el = f.create_block(Some("else"));
    let m = f.create_block(Some("merge"));
    f.set_terminator(e, Terminator::CondBr { cond: c.into(), then_bb: t, else_bb: el });
    let (_, tv) = f.append_inst(t, InstKind::Binary { op: BinOp::Add, lhs: x.into(), rhs: Operand::int(32, 1) }, Some(Type::I32));
    let (_, ev) = f.append_inst(el, InstKind::Binary { op: BinOp::Sub, lhs: x.into(), rhs: Operand::int(32, 1) }, Some(Type::I32));
    f.set_terminator(t, Terminator::Br(m));
    f.set_terminator(el, Terminator::Br(m));
    let (_, p) = f.append_inst(m, InstKind::Phi { ty: Type::I32, incoming: vec![(t, tv.unwrap().into()), (el, ev.unwrap().into())] }, Some(Type::I32));
    f.set_terminator(m, Terminator::Ret(Some(p.unwrap().into())));
    (f, [e, t, el, m], tv.unwrap(), ev.unwrap())
}

#[test]
fn dominators_and_frontiers() {
    let (f, [e, t, el, m], _, _) = diamond();
    let dt = DomTree::compute(&f);
    assert_eq!(dt.idom(m), Some(e));
    assert_eq!(dt.idom(t), Some(e));
    assert!(dt.dominates(e, m) && !dt.dominates(t, m) && !dt.dominates(el, m));
    let df = dt.frontiers(&f);
    assert_eq!(df[t.index()], vec![m]);
    assert_eq!(df[el.index()], vec![m]);
    assert!(df[e.index()].is_empty());
    assert!(verify_function(&f).is_ok());
}

#[test]
fn verifier_rejects_non_dominating_use() {
    let (mut f, [_, _, el, _], tv, _) = diamond();
    // `else` block uses a value defined in `then`.
    f.append_inst(el, InstKind::Cast { op: CastOp::Bitcast, val: tv.into(), to: Type::I32 }, Some(Type::I32));
    let err = verify_function(&f).unwrap_err();
    assert!(err.0.contains("does not dominate"), "{}", err.0);
}

#[test]
fn fold_then_dce_and_print() {
    let mut m = Module::new("t");
    let mut f = Function::new("k", Signature::new(vec![Type::I32], Type::I32), Linkage::External);
    let bb = f.create_block(None);
    let (_, a) = f.append_inst(bb, InstKind::Binary { op: BinOp::Add, lhs: Operand::int(32, 2), rhs: Operand::int(32, 3) }, Some(Type::I32));
    let (_, c) = f.append_inst(bb, InstKind::ICmp { pred: IntPred::Slt, lhs: a.unwrap().into(), rhs: Operand::int(32, 10) }, Some(Type::I1));
    let (_, s) = f.append_inst(bb, InstKind::Select { cond: c.unwrap().into(), then_val: a.unwrap().into(), else_val: f.params[0].into() }, Some(Type::I32));
    f.set_terminator(bb, Terminator::Ret(Some(s.unwrap().into())));
    let id = m.add_function(f);

    let mut pm = PassManager::new();
    pm.add_function_pass(ConstantFolding).add_function_pass(DeadCodeElimination);
    assert!(pm.run(&mut m).is_changed());
    let f = &m.functions[id];
    assert!(f.blocks[bb].insts.is_empty());
    assert_eq!(f.blocks[bb].terminator, Some(Terminator::Ret(Some(Operand::int(32, 5)))));
    let text = m.to_string();
    assert!(text.contains("define external @k(i32 %v0) -> i32"), "{text}");
    assert!(text.contains("ret i32 5:i32"), "{text}");
}

#[test]
fn signed_folding_edge_cases() {
    let mut f = Function::new("e", Signature::new(vec![], Type::I8), Linkage::External);
    let bb = f.create_block(None);
    // -128 / -1 must NOT fold (overflow); 200 ashr 1 on i8 = 0xE4.
    let (_, d) = f.append_inst(bb, InstKind::Binary { op: BinOp::SDiv, lhs: Operand::int(8, 0x80), rhs: Operand::int(8, 0xFF) }, Some(Type::I8));
    let (_, sh) = f.append_inst(bb, InstKind::Binary { op: BinOp::AShr, lhs: Operand::int(8, 200), rhs: Operand::int(8, 1) }, Some(Type::I8));
    let (_, r) = f.append_inst(bb, InstKind::Binary { op: BinOp::Add, lhs: d.unwrap().into(), rhs: sh.unwrap().into() }, Some(Type::I8));
    f.set_terminator(bb, Terminator::Ret(Some(r.unwrap().into())));
    ConstantFolding.run(&mut f);
    let remaining: Vec<_> = f.blocks[bb].insts.clone();
    assert_eq!(remaining.len(), 2); // sdiv + add stay
    match &f.insts[remaining[1]].kind {
        InstKind::Binary { rhs, .. } => assert_eq!(*rhs, Operand::int(8, 0xE4)),
        k => panic!("{k:?}"),
    }
}
