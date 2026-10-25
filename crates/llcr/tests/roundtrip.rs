// SPDX-License-Identifier: GPL-2.0-only WITH LicenseRef-Kaiba-Runtime-Exception
use llcr::opt::{constfold::ConstantFolding, dce::DeadCodeElimination};
use llcr::verify::verify_function;
use llcr::*;

const LOOP: &str = r#"module "loop"
target "x86_64-unknown-linux-gnu"

define external @sum(i32 %v0) -> i32 {
bb0:  ; entry
  br bb1
bb1:  ; header
  %v1 = phi i32 [bb0: 0:i32], [bb2: %v4]
  %v2 = phi i32 [bb0: 0:i32], [bb2: %v3]
  %v5 = icmp slt i32 %v1, %v0
  condbr %v5, bb2, bb3
bb2:  ; body
  %v3 = add i32 %v2, %v1
  %v4 = add i32 %v1, 1:i32
  br bb1
bb3:  ; exit
  ret i32 %v2
}
"#;

#[test]
fn parses_loop_with_forward_refs_and_gaps() {
    let m = parse_module(LOOP).unwrap();
    assert_eq!(m.name, "loop");
    assert_eq!(m.target_triple.as_deref(), Some("x86_64-unknown-linux-gnu"));
    let f = &m.functions[m.find_function("sum").unwrap()];
    assert_eq!(f.layout.len(), 4);
    verify_function(f).unwrap();
    // Reprinting renumbers compactly, and is then stable.
    let t2 = m.to_string();
    let t3 = parse_module(&t2).unwrap().to_string();
    assert_eq!(t2, t3);
}

fn rich_module() -> Module {
    let mut m = Module::new("rich");
    m.target_triple = Some("aarch64-unknown-linux-gnu".into());
    let st = m.intern_aggregate(AggregateType::Struct { fields: vec![Type::I32, Type::F64], packed: false });
    let arr = m.intern_aggregate(AggregateType::Array { elem: Type::I8, len: 16 });
    let g = m.add_global(GlobalVar { name: "tbl".into(), ty: arr, linkage: Linkage::Internal, is_const: true, init: Some(vec![1, 0xab, 255]), relocs: vec![], align: 8 });
    let mut decl = Function::new("puts", Signature::new(vec![Type::PTR], Type::I32), Linkage::Declaration);
    decl.sig.variadic = true;
    let puts = m.add_function(decl);

    let mut f = Function::new("main", Signature::new(vec![Type::I32, Type::PTR], Type::I32), Linkage::External);
    let (a, p) = (f.params[0], f.params[1]);
    let e = f.create_block(Some("entry"));
    let l = f.create_block(None);
    let r = f.create_block(None);
    let (_, slot) = f.append_inst(e, InstKind::Alloca { ty: st, count: Operand::int(32, 1), align: 8 }, Some(Type::PTR));
    let (_, gp) = f.append_inst(e, InstKind::Gep { base_ty: st, ptr: slot.unwrap().into(), indices: vec![Operand::int(32, 0), Operand::int(32, 1)], inbounds: true }, Some(Type::PTR));
    f.append_inst(e, InstKind::Store { val: Operand::Const(Constant::F64(0x4000000000000000)), ptr: gp.unwrap().into(), align: 8, volatile: true }, None);
    let (_, ld) = f.append_inst(e, InstKind::Load { ty: Type::F64, ptr: gp.unwrap().into(), align: 8, volatile: false }, Some(Type::F64));
    let (_, fc) = f.append_inst(e, InstKind::FCmp { pred: FloatPred::Olt, lhs: ld.unwrap().into(), rhs: Operand::Const(Constant::F64(0)) }, Some(Type::I1));
    let (_, w) = f.append_inst(e, InstKind::Cast { op: CastOp::ZExt, val: fc.unwrap().into(), to: Type::I32 }, Some(Type::I32));
    let (_, call) = f.append_inst(e, InstKind::Call { callee: Callee::Direct(puts), args: vec![Operand::Global(g)], ret_ty: Type::I32 }, Some(Type::I32));
    f.append_inst(e, InstKind::Call { callee: Callee::Indirect(p.into()), args: vec![], ret_ty: Type::Void }, None);
    let (_, ev) = f.append_inst(e, InstKind::ExtractValue { agg: Operand::Const(Constant::Undef(st)), indices: vec![0] }, Some(Type::I32));
    let (_, c) = f.append_inst(e, InstKind::ICmp { pred: IntPred::Ne, lhs: a.into(), rhs: w.unwrap().into() }, Some(Type::I1));
    f.set_terminator(e, Terminator::CondBr { cond: c.unwrap().into(), then_bb: l, else_bb: r });
    f.set_terminator(l, Terminator::Switch { val: ev.unwrap().into(), default: r, cases: vec![(1, r), (2, e)] });
    let (_, s) = f.append_inst(r, InstKind::Select { cond: Operand::int(1, 1), then_val: call.unwrap().into(), else_val: Operand::Const(Constant::Poison(Type::I32)) }, Some(Type::I32));
    f.set_terminator(r, Terminator::Ret(Some(s.unwrap().into())));
    m.add_function(f);
    m
}

#[test]
fn print_parse_print_is_identity() {
    let m = rich_module();
    let t1 = m.to_string();
    let m2 = parse_module(&t1).unwrap_or_else(|e| panic!("{e}\n{t1}"));
    assert_eq!(t1, m2.to_string());
    let g = &m2.globals[GlobalId::new(0)];
    assert_eq!(g.init.as_deref(), Some(&[1u8, 0xab, 255][..]));
    assert!(m2.functions[m2.find_function("puts").unwrap()].sig.variadic);
}

#[test]
fn optimized_output_reparses() {
    let mut m = parse_module(LOOP).unwrap();
    let mut pm = PassManager::new();
    pm.add_function_pass(ConstantFolding).add_function_pass(DeadCodeElimination);
    pm.run(&mut m);
    let t = m.to_string();
    let t2 = parse_module(&t).unwrap().to_string();
    assert_eq!(t, t2);
}

#[test]
fn reports_errors_with_line_numbers() {
    let bad_use = LOOP.replace("%v1, %v0", "%v9, %v0");
    let e = parse_module(&bad_use).unwrap_err();
    assert!(e.message.contains("undefined value %v9"), "{e}");

    let no_term = "module \"x\"\ndefine external @f() -> void {\nbb0:\n}\n";
    assert!(parse_module(no_term).unwrap_err().message.contains("no terminator"));

    let bad_inst = "module \"x\"\ndefine external @f() -> void {\nbb0:\n  frobnicate\n  ret\n}\n";
    let e = parse_module(bad_inst).unwrap_err();
    assert_eq!(e.line, 4);
    assert!(e.message.contains("unknown instruction"));
}

#[test]
fn global_relocations_round_trip() {
    let src = r#"module "r"
global @a: i32 internal align 4 init [01 00 00 00]
global @tbl: [3 x ptr] internal const align 8 init [00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00] reloc [0: @a, 8: @f, 16: @a+4]

define external @f() -> void {
bb0:
  ret
}
"#;
    let m = parse_module(src).unwrap_or_else(|e| panic!("{e}"));
    let g = &m.globals[GlobalId::new(1)];
    assert_eq!(g.relocs.len(), 3);
    assert_eq!(g.relocs[2].addend, 4);
    assert_eq!(g.relocs[1].target, RelocTarget::Func(m.find_function("f").unwrap()));
    let t = m.to_string();
    assert!(t.contains("reloc [0: @a, 8: @f, 16: @a+4]"), "{t}");
    assert_eq!(t, parse_module(&t).unwrap().to_string());
}

#[test]
fn special_abi_signatures_round_trip() {
    let src = r#"module "abi"
declare @fill(sret ptr, i64) -> void
declare @eat(byval(32,8) ptr, i32) -> i64

define external @pair(i64 %v0) -> { i64, i64 } {
bb0:
  ret { i64, i64 } undef:{ i64, i64 }
}

define external @mk(sret ptr %v0, byval(16,8) ptr %v1) -> void {
bb0:
  ret
}
"#;
    let m = parse_module(src).unwrap_or_else(|e| panic!("{e}"));
    let fill = &m.functions[m.find_function("fill").unwrap()];
    assert!(fill.sig.sret && fill.sig.byval.is_empty());
    let eat = &m.functions[m.find_function("eat").unwrap()];
    assert_eq!(eat.sig.byval, vec![(0, 32, 8)]);
    let mk = &m.functions[m.find_function("mk").unwrap()];
    assert!(mk.sig.sret && mk.sig.byval == vec![(1, 16, 8)]);
    assert!(matches!(m.functions[m.find_function("pair").unwrap()].sig.ret, Type::Agg(_)));
    let t = m.to_string();
    assert!(t.contains("declare @fill(sret ptr, i64) -> void"), "{t}");
    assert!(t.contains("@eat(byval(32,8) ptr, i32)"), "{t}");
    assert_eq!(t, parse_module(&t).unwrap().to_string());
}
