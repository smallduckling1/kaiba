// SPDX-License-Identifier: GPL-2.0-only WITH LicenseRef-Kaiba-Runtime-Exception
mod common;
use llcr::opt::{dce::DeadCodeElimination, mem2reg::Mem2Reg, simplifycfg::SimplifyCfg};
use llcr::verify::verify_function;
use llcr::*;

const SUM: &str = r#"module "m"
define external @sum(i32 %v0) -> i32 {
bb0:
  %v1 = alloca i32, 1:i32, align 4
  %v2 = alloca i32, 1:i32, align 4
  store i32 0:i32, %v1, align 4
  store i32 0:i32, %v2, align 4
  br bb1
bb1:
  %v3 = load i32, %v2, align 4
  %v4 = icmp slt i32 %v3, %v0
  condbr %v4, bb2, bb3
bb2:
  %v5 = load i32, %v1, align 4
  %v6 = load i32, %v2, align 4
  %v7 = add i32 %v5, %v6
  store i32 %v7, %v1, align 4
  %v8 = add i32 %v6, 1:i32
  store i32 %v8, %v2, align 4
  br bb1
bb3:
  %v9 = load i32, %v1, align 4
  ret i32 %v9
}
"#;

const PICK: &str = r#"module "m"
define external @pick(i32 %v0) -> i32 {
bb0:
  %v1 = alloca i32, 1:i32, align 4
  %v2 = icmp ne i32 %v0, 0:i32
  condbr %v2, bb1, bb2
bb1:
  store i32 10:i32, %v1, align 4
  br bb3
bb2:
  store i32 20:i32, %v1, align 4
  br bb3
bb3:
  %v3 = load i32, %v1, align 4
  ret i32 %v3
}
"#;

fn optimize(m: &mut Module) -> Changed {
    let mut pm = PassManager::new();
    pm.add_function_pass(Mem2Reg).add_function_pass(SimplifyCfg).add_function_pass(DeadCodeElimination);
    pm.run(m)
}

fn first(m: &Module) -> &Function { &m.functions[FuncId::new(0)] }

#[test]
fn mem2reg_loop_preserves_semantics() {
    let mut m = parse_module(SUM).unwrap();
    let before: Vec<u128> = (0..9).map(|n| common::run(first(&m), &[n])).collect();
    assert_eq!(before[8], 28);
    assert!(optimize(&mut m).is_changed());
    let text = m.to_string();
    assert!(!text.contains("alloca") && !text.contains("load") && !text.contains("store"), "{text}");
    assert_eq!(text.matches("phi").count(), 2, "{text}");
    verify_function(first(&m)).unwrap();
    let after: Vec<u128> = (0..9).map(|n| common::run(first(&m), &[n])).collect();
    assert_eq!(before, after);
    // Optimized text still round-trips (after one renumbering round).
    let t2 = parse_module(&text).unwrap().to_string();
    assert_eq!(t2, parse_module(&t2).unwrap().to_string());
}

#[test]
fn mem2reg_diamond_places_phi_at_merge() {
    let mut m = parse_module(PICK).unwrap();
    optimize(&mut m);
    let text = m.to_string();
    assert_eq!(text.matches("phi").count(), 1, "{text}");
    assert_eq!(common::run(first(&m), &[1]), 10);
    assert_eq!(common::run(first(&m), &[0]), 20);
}

#[test]
fn mem2reg_leaves_escaping_alloca_alone() {
    let src = r#"module "m"
declare @use(ptr) -> void
define external @g() -> i32 {
bb0:
  %v0 = alloca i32, 1:i32, align 4
  call void @use(%v0)
  %v1 = load i32, %v0, align 4
  ret i32 %v1
}
"#;
    let mut m = parse_module(src).unwrap();
    let mut pm = PassManager::new();
    pm.add_function_pass(Mem2Reg);
    assert!(!pm.run(&mut m).is_changed());
    assert!(m.to_string().contains("alloca"));
}

#[test]
fn mem2reg_uninitialized_read_becomes_undef() {
    let src = r#"module "m"
define external @h() -> i32 {
bb0:
  %v0 = alloca i32, 1:i32, align 4
  %v1 = load i32, %v0, align 4
  ret i32 %v1
}
"#;
    let mut m = parse_module(src).unwrap();
    optimize(&mut m);
    assert!(m.to_string().contains("ret i32 undef:i32"), "{m}");
}

#[test]
fn simplifycfg_folds_constant_branch_and_merges() {
    let src = r#"module "m"
define external @f(i32 %v0) -> i32 {
bb0:
  condbr 1:i1, bb1, bb2
bb1:
  %v1 = add i32 %v0, 1:i32
  br bb3
bb2:
  %v2 = add i32 %v0, 2:i32
  br bb3
bb3:
  %v3 = phi i32 [bb1: %v1], [bb2: %v2]
  ret i32 %v3
}
"#;
    let mut m = parse_module(src).unwrap();
    let mut pm = PassManager::new();
    pm.add_function_pass(SimplifyCfg);
    assert!(pm.run(&mut m).is_changed());
    let f = first(&m);
    assert_eq!(f.layout.len(), 1, "{m}");
    assert!(!m.to_string().contains("phi"));
    assert_eq!(common::run(f, &[5]), 6);
}

#[test]
fn simplifycfg_keeps_real_diamond() {
    let mut m = parse_module(PICK).unwrap();
    let mut pm = PassManager::new();
    pm.add_function_pass(SimplifyCfg);
    assert!(!pm.run(&mut m).is_changed());
    assert_eq!(first(&m).layout.len(), 4);
}

fn run_passes(m: &mut Module, names: &[&str]) {
    let mut pm = PassManager::new();
    for n in names { llcr::opt::add_pass(&mut pm, llcr::opt::create_pass(n).unwrap()); }
    pm.run(m);
}

fn func<'a>(m: &'a Module, name: &str) -> &'a Function { &m.functions[m.find_function(name).unwrap()] }

fn count_calls(f: &Function) -> usize {
    f.iter_insts().filter(|&(_, i)| matches!(f.insts[i].kind, InstKind::Call { .. })).count()
}

const INLINE_SRC: &str = r#"module "m"
define internal @add3(i32 %v0, i32 %v1) -> i32 {
bb0:
  %v2 = add i32 %v0, %v1
  %v3 = add i32 %v2, 3:i32
  ret i32 %v3
}
define internal @absdiff(i32 %v0, i32 %v1) -> i32 {
bb0:
  %v2 = icmp sgt i32 %v0, %v1
  condbr %v2, bb1, bb2
bb1:
  %v3 = sub i32 %v0, %v1
  ret i32 %v3
bb2:
  %v4 = sub i32 %v1, %v0
  ret i32 %v4
}
define internal @withslot(i32 %v0) -> i32 {
bb0:
  %v1 = alloca i32, 1:i32, align 4
  store i32 %v0, %v1, align 4
  %v2 = load i32, %v1, align 4
  %v3 = mul i32 %v2, 2:i32
  ret i32 %v3
}
define external @chain(i32 %v0, i32 %v1) -> i32 {
bb0:
  %v2 = call i32 @add3(%v0, %v1)
  %v3 = call i32 @absdiff(%v2, 10:i32)
  %v4 = call i32 @withslot(%v3)
  ret i32 %v4
}
define external @fact(i32 %v0) -> i32 {
bb0:
  %v1 = icmp slt i32 %v0, 2:i32
  condbr %v1, bb1, bb2
bb1:
  ret i32 1:i32
bb2:
  %v2 = sub i32 %v0, 1:i32
  %v3 = call i32 @fact(%v2)
  %v4 = mul i32 %v0, %v3
  ret i32 %v4
}
define external @callfact(i32 %v0) -> i32 {
bb0:
  %v1 = call i32 @fact(%v0)
  %v2 = call i32 @add3(%v1, 1:i32)
  ret i32 %v2
}
"#;

#[test]
fn inliner_removes_calls_hoists_allocas_and_skips_recursion() {
    let mut m = parse_module(INLINE_SRC).unwrap();
    run_passes(&mut m, &["inline", "mem2reg", "simplifycfg", "dce"]);
    let chain = func(&m, "chain");
    assert_eq!(count_calls(chain), 0, "{m}");
    assert!(!m.to_string().contains("alloca"), "callee alloca should be hoisted and promoted\n{m}");
    verify_function(chain).unwrap();
    for a in [-20i32, -3, 0, 4, 9, 50] {
        for b in [-7i32, 0, 1, 13] {
            let t = a.wrapping_add(b).wrapping_add(3);
            let expect = (t - 10).abs().wrapping_mul(2);
            assert_eq!(common::run(chain, &[a as u32 as u128, b as u32 as u128]) as u32, expect as u32, "a={a} b={b}");
        }
    }
    // Recursive callee is left alone; the non-recursive helper inside callfact is inlined.
    assert_eq!(count_calls(func(&m, "fact")), 1);
    let cf = func(&m, "callfact");
    assert_eq!(count_calls(cf), 1, "{m}");
    // Unreferenced internal helpers become declarations.
    for dead in ["add3", "absdiff", "withslot"] { assert!(func(&m, dead).is_declaration(), "{dead} should be dropped"); }
}

const LICM_SRC: &str = r#"module "m"
define external @loopinv(i32 %v0, i32 %v1, i32 %v2) -> i32 {
bb0:
  br bb1
bb1:
  %v3 = phi i32 [bb0: 0:i32], [bb2: %v8]
  %v4 = phi i32 [bb0: 0:i32], [bb2: %v7]
  %v5 = icmp slt i32 %v3, %v0
  condbr %v5, bb2, bb3
bb2:
  %v6 = mul i32 %v1, %v2
  %v9 = add i32 %v6, 5:i32
  %v10 = sdiv i32 %v1, %v2
  %v11 = add i32 %v9, %v10
  %v7 = add i32 %v4, %v11
  %v8 = add i32 %v3, 1:i32
  br bb1
bb3:
  ret i32 %v4
}
define external @guarded(i32 %v0, i32 %v1, i32 %v2) -> i32 {
bb0:
  %v3 = icmp sgt i32 %v0, 0:i32
  condbr %v3, bb1, bb2
bb1:
  %v4 = phi i32 [bb0: 0:i32], [bb1: %v8]
  %v5 = phi i32 [bb0: 0:i32], [bb1: %v7]
  %v6 = mul i32 %v1, %v2
  %v7 = add i32 %v5, %v6
  %v8 = add i32 %v4, 1:i32
  %v9 = icmp slt i32 %v8, %v0
  condbr %v9, bb1, bb2
bb2:
  %v10 = phi i32 [bb0: 0:i32], [bb1: %v7]
  ret i32 %v10
}
"#;

#[test]
fn licm_hoists_invariants_but_not_trapping_division() {
    let mut m = parse_module(LICM_SRC).unwrap();
    let before: Vec<u128> = [(5u128, 3u128, 4u128), (0, 3, 4), (7, 100, 7)].iter().map(|&(n, a, b)| common::run(func(&m, "loopinv"), &[n, a, b])).collect();
    run_passes(&mut m, &["licm"]);
    let f = func(&m, "loopinv");
    verify_function(f).unwrap();
    let kinds = |b: BlockId| -> Vec<String> { f.blocks[b].insts.iter().map(|&i| format!("{:?}", f.insts[i].kind)).collect() };
    let (bb0, bb2) = (f.layout[0], f.layout[2]);
    assert_eq!(f.blocks[bb0].insts.len(), 2, "mul and add should move to the preheader: {:?}", kinds(bb0));
    assert!(kinds(bb2).iter().any(|k| k.contains("SDiv")), "variable divisor must stay in the loop");
    let after: Vec<u128> = [(5u128, 3u128, 4u128), (0, 3, 4), (7, 100, 7)].iter().map(|&(n, a, b)| common::run(f, &[n, a, b])).collect();
    assert_eq!(before, after);
}

#[test]
fn licm_creates_a_preheader_when_the_entry_edge_is_conditional() {
    let mut m = parse_module(LICM_SRC).unwrap();
    let args = |n: u128| [n, 6u128, 7u128];
    let before: Vec<u128> = (0..6).map(|n| common::run(func(&m, "guarded"), &args(n))).collect();
    assert_eq!(before[3], 126);
    run_passes(&mut m, &["licm"]);
    let f = func(&m, "guarded");
    verify_function(f).unwrap();
    assert_eq!(f.layout.len(), 4, "a preheader block should have been added\n{m}");
    let after: Vec<u128> = (0..6).map(|n| common::run(f, &args(n))).collect();
    assert_eq!(before, after);
}
