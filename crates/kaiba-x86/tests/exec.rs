// SPDX-License-Identifier: GPL-2.0-only WITH LicenseRef-Kaiba-Runtime-Exception
//! End-to-end tests: llcr -> x86_64 asm -> gcc (assemble+link with a C driver) -> run.
//! Each program is built both unoptimized and with the -O1 pipeline.

use llcr::{parse_module, PassManager};
use std::process::Command;
use std::sync::atomic::{AtomicUsize, Ordering};

static N: AtomicUsize = AtomicUsize::new(0);

fn have_gcc() -> bool {
    Command::new("gcc").arg("--version").output().map(|o| o.status.success()).unwrap_or(false)
}

fn build_and_run(ir: &str, c: &str, optimize: bool) -> String {
    let mut m = parse_module(ir).unwrap_or_else(|e| panic!("parse: {e}\n{ir}"));
    if optimize {
        let mut pm = PassManager::new();
        llcr::opt::add_o1_pipeline(&mut pm);
        pm.run(&mut m);
    }
    let asm = kaiba_x86::compile_module(&m).unwrap_or_else(|e| panic!("codegen: {e}"));
    let dir = std::env::temp_dir().join(format!("kaiba-x86-{}-{}", std::process::id(), N.fetch_add(1, Ordering::SeqCst)));
    std::fs::create_dir_all(&dir).unwrap();
    let (s, cfile, exe) = (dir.join("out.s"), dir.join("main.c"), dir.join("prog"));
    std::fs::write(&s, &asm).unwrap();
    std::fs::write(&cfile, c).unwrap();
    let o = Command::new("gcc").arg("-o").arg(&exe).arg(&cfile).arg(&s).output().unwrap();
    assert!(o.status.success(), "gcc failed:\n{}\n--- asm ---\n{asm}", String::from_utf8_lossy(&o.stderr));
    let r = Command::new(&exe).output().unwrap();
    assert!(r.status.success(), "program crashed ({:?})\n--- asm ---\n{asm}", r.status);
    let _ = std::fs::remove_dir_all(&dir);
    String::from_utf8(r.stdout).unwrap()
}

fn check(ir: &str, c: &str, expected: &str) {
    if !have_gcc() { eprintln!("gcc not found; skipping"); return; }
    for opt in [false, true] {
        let got = build_and_run(ir, c, opt);
        assert_eq!(got.trim(), expected.trim(), "optimize={opt}\n{ir}");
    }
}

fn lines<T: ToString>(v: impl IntoIterator<Item = T>) -> String {
    v.into_iter().map(|x| x.to_string()).collect::<Vec<_>>().join("\n")
}

#[test]
fn stack_slots_and_loops() {
    let ir = r#"module "t"
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
    let c = r#"#include <stdio.h>
int sum(int);
int main(){ for(int i=0;i<12;i++) printf("%d\n", sum(i)); return 0; }"#;
    check(ir, c, &lines((0..12).map(|n: i32| (0..n).sum::<i32>())));
}

#[test]
fn recursion_and_iterative_fib_with_phi_swap() {
    let ir = r#"module "t"
define external @fib(i32 %v0) -> i32 {
bb0:
  %v1 = icmp slt i32 %v0, 2:i32
  condbr %v1, bb1, bb2
bb1:
  ret i32 %v0
bb2:
  %v2 = sub i32 %v0, 1:i32
  %v3 = call i32 @fib(%v2)
  %v4 = sub i32 %v0, 2:i32
  %v5 = call i32 @fib(%v4)
  %v6 = add i32 %v3, %v5
  ret i32 %v6
}
define external @fibi(i32 %v0) -> i64 {
bb0:
  br bb1
bb1:
  %v1 = phi i32 [bb0: 0:i32], [bb2: %v6]
  %v2 = phi i64 [bb0: 0:i64], [bb2: %v3]
  %v3 = phi i64 [bb0: 1:i64], [bb2: %v5]
  %v4 = icmp slt i32 %v1, %v0
  condbr %v4, bb2, bb3
bb2:
  %v5 = add i64 %v2, %v3
  %v6 = add i32 %v1, 1:i32
  br bb1
bb3:
  ret i64 %v2
}
"#;
    let c = r#"#include <stdio.h>
int fib(int); long fibi(int);
int main(){ for(int i=0;i<20;i++) printf("%d %ld\n", fib(i), fibi(i)); return 0; }"#;
    let mut f = vec![0i64, 1];
    for i in 2..20 { let n = f[i - 1] + f[i - 2]; f.push(n); }
    check(ir, c, &lines((0..20).map(|i| format!("{} {}", f[i], f[i]))));
}

#[test]
fn lost_copy_on_critical_edge() {
    let ir = r#"module "t"
define external @dowhile(i32 %v0) -> i32 {
bb0:
  br bb1
bb1:
  %v1 = phi i32 [bb0: 0:i32], [bb1: %v4]
  %v2 = phi i32 [bb0: 0:i32], [bb1: %v3]
  %v3 = add i32 %v2, %v1
  %v4 = add i32 %v1, 1:i32
  %v5 = icmp slt i32 %v4, %v0
  condbr %v5, bb1, bb2
bb2:
  ret i32 %v2
}
"#;
    let c = r#"#include <stdio.h>
int dowhile(int);
int main(){ for(int i=0;i<10;i++) printf("%d\n", dowhile(i)); return 0; }"#;
    let exp = (0..10).map(|n: i32| {
        let (mut i, mut s) = (0i32, 0i32);
        loop { let s1 = s + i; let i1 = i + 1; if i1 < n { i = i1; s = s1; } else { break; } }
        s
    });
    check(ir, c, &lines(exp));
}

#[test]
fn register_pressure_forces_spills() {
    // 14 values live at once (only 5 allocatable registers).
    let mut s = String::from("module \"t\"\ndefine external @pressure(i32 %v0, i32 %v1) -> i32 {\nbb0:\n");
    let mut n = 2;
    let mut ts = vec![];
    for i in 0..14 {
        s += &format!("  %v{n} = mul i32 %v0, {}:i32\n  %v{} = add i32 %v{n}, %v1\n", i + 1, n + 1);
        ts.push(n + 1);
        n += 2;
    }
    let mut acc = ts[0];
    for t in &ts[1..] {
        s += &format!("  %v{n} = add i32 %v{acc}, %v{t}\n");
        acc = n;
        n += 1;
    }
    s += &format!("  ret i32 %v{acc}\n}}\n");
    let c = r#"#include <stdio.h>
int pressure(int,int);
int main(){ for(int a=-3;a<4;a++) printf("%d\n", pressure(a, 1000*a+7)); return 0; }"#;
    let exp = (-3i32..4).map(|a| {
        let b = 1000 * a + 7;
        (0..14).fold(0i32, |s, i| s.wrapping_add(a.wrapping_mul(i + 1).wrapping_add(b)))
    });
    check(&s, c, &lines(exp));
}

#[test]
fn narrow_integers() {
    let ir = r#"module "t"
define external @narrow(i32 %v0) -> i32 {
bb0:
  %v1 = trunc i32 %v0 to i8
  %v2 = add i8 %v1, 100:i8
  %v3 = sext i8 %v2 to i32
  %v4 = zext i8 %v2 to i32
  %v5 = icmp slt i8 %v2, 0:i8
  %v6 = zext i1 %v5 to i32
  %v7 = ashr i8 %v2, 1:i8
  %v8 = sext i8 %v7 to i32
  %v9 = mul i32 %v4, 1000:i32
  %v10 = mul i32 %v6, 100000:i32
  %v11 = mul i32 %v8, 7:i32
  %v12 = add i32 %v3, %v9
  %v13 = add i32 %v12, %v10
  %v14 = add i32 %v13, %v11
  ret i32 %v14
}
"#;
    let c = r#"#include <stdio.h>
int narrow(int);
int main(){ int xs[]={-128,-100,-1,0,1,27,28,100,127,300,-300};
 for(int i=0;i<11;i++) printf("%d\n", narrow(xs[i])); return 0; }"#;
    let xs = [-128i32, -100, -1, 0, 1, 27, 28, 100, 127, 300, -300];
    let exp = xs.iter().map(|&x| {
        let t = (x as i8).wrapping_add(100);
        (t as i32).wrapping_add((t as u8 as i32) * 1000).wrapping_add((t < 0) as i32 * 100000).wrapping_add(((t >> 1) as i32) * 7)
    });
    check(ir, c, &lines(exp));
}

#[test]
fn division_and_remainder() {
    let ir = r#"module "t"
define external @divs(i32 %v0, i32 %v1) -> i32 {
bb0:
  %v2 = sdiv i32 %v0, %v1
  %v3 = srem i32 %v0, %v1
  %v4 = udiv i32 %v0, %v1
  %v5 = urem i32 %v0, %v1
  %v6 = mul i32 %v3, 7:i32
  %v7 = mul i32 %v4, 13:i32
  %v8 = mul i32 %v5, 31:i32
  %v9 = add i32 %v2, %v6
  %v10 = add i32 %v9, %v7
  %v11 = add i32 %v10, %v8
  ret i32 %v11
}
"#;
    let c = r#"#include <stdio.h>
int divs(int,int);
int main(){ int as[]={100,-100,7,-7,2147483647,-2147483647,0}; int bs[]={3,-3,5,-5,1,-1,2};
 for(int i=0;i<7;i++) for(int j=0;j<7;j++) printf("%d\n", divs(as[i],bs[j])); return 0; }"#;
    let (a, b) = ([100i32, -100, 7, -7, i32::MAX, -i32::MAX, 0], [3i32, -3, 5, -5, 1, -1, 2]);
    let mut exp = vec![];
    for x in a { for y in b {
        let v = x.wrapping_div(y).wrapping_add(x.wrapping_rem(y).wrapping_mul(7))
            .wrapping_add(((x as u32) / (y as u32)).wrapping_mul(13) as i32)
            .wrapping_add(((x as u32) % (y as u32)).wrapping_mul(31) as i32);
        exp.push(v);
    } }
    check(ir, c, &lines(exp));
}

#[test]
fn arrays_and_dynamic_gep() {
    let ir = r#"module "t"
define external @squares(i32 %v0) -> i32 {
bb0:
  %v1 = alloca [8 x i32], 1:i32, align 16
  br bb1
bb1:
  %v2 = phi i32 [bb0: 0:i32], [bb2: %v6]
  %v3 = icmp slt i32 %v2, %v0
  condbr %v3, bb2, bb3
bb2:
  %v4 = gep inbounds [8 x i32], %v1, [0:i32, %v2]
  %v5 = mul i32 %v2, %v2
  store i32 %v5, %v4, align 4
  %v6 = add i32 %v2, 1:i32
  br bb1
bb3:
  br bb4
bb4:
  %v8 = phi i32 [bb3: 0:i32], [bb5: %v13]
  %v9 = phi i32 [bb3: 0:i32], [bb5: %v12]
  %v10 = icmp slt i32 %v8, %v0
  condbr %v10, bb5, bb6
bb5:
  %v11 = gep inbounds [8 x i32], %v1, [0:i32, %v8]
  %v14 = load i32, %v11, align 4
  %v12 = add i32 %v9, %v14
  %v13 = add i32 %v8, 1:i32
  br bb4
bb6:
  ret i32 %v9
}
"#;
    let c = r#"#include <stdio.h>
int squares(int);
int main(){ for(int i=0;i<=8;i++) printf("%d\n", squares(i)); return 0; }"#;
    check(ir, c, &lines((0..=8).map(|n: i32| (0..n).map(|i| i * i).sum::<i32>())));
}

#[test]
fn struct_layout_and_gep() {
    let ir = r#"module "t"
define external @st(i32 %v0) -> i64 {
bb0:
  %v1 = alloca { i8, i32, i64 }, 1:i32, align 8
  %v2 = gep inbounds { i8, i32, i64 }, %v1, [0:i32, 0:i32]
  %v3 = gep inbounds { i8, i32, i64 }, %v1, [0:i32, 1:i32]
  %v4 = gep inbounds { i8, i32, i64 }, %v1, [0:i32, 2:i32]
  %v5 = trunc i32 %v0 to i8
  %v6 = mul i32 %v0, 2:i32
  %v7 = sext i32 %v0 to i64
  %v8 = mul i64 %v7, 3:i64
  store i8 %v5, %v2, align 1
  store i32 %v6, %v3, align 4
  store i64 %v8, %v4, align 8
  %v9 = load i8, %v2, align 1
  %v10 = load i32, %v3, align 4
  %v11 = load i64, %v4, align 8
  %v12 = zext i8 %v9 to i64
  %v13 = sext i32 %v10 to i64
  %v14 = add i64 %v12, %v13
  %v15 = add i64 %v14, %v11
  ret i64 %v15
}
"#;
    let c = r#"#include <stdio.h>
long st(int);
int main(){ int xs[]={0,1,-1,200,-200,100000}; for(int i=0;i<6;i++) printf("%ld\n", st(xs[i])); return 0; }"#;
    let exp = [0i32, 1, -1, 200, -200, 100000].iter().map(|&x| {
        (x as i8 as u8 as i64) + (x.wrapping_mul(2) as i64) + (x as i64) * 3
    });
    check(ir, c, &lines(exp));
}

#[test]
fn many_arguments_in_and_out() {
    let ir = r#"module "t"
declare @sum7(i64, i64, i64, i64, i64, i64, i64) -> i64
declare @sum8(i64, i64, i64, i64, i64, i64, i64, i64) -> i64
define external @call7(i64 %v0) -> i64 {
bb0:
  %v1 = call i64 @sum7(%v0, 2:i64, 3:i64, 4:i64, 5:i64, 6:i64, 7:i64)
  ret i64 %v1
}
define external @call8(i64 %v0) -> i64 {
bb0:
  %v1 = call i64 @sum8(%v0, 2:i64, 3:i64, 4:i64, 5:i64, 6:i64, 7:i64, 8:i64)
  ret i64 %v1
}
define external @pick(i64 %v0, i64 %v1, i64 %v2, i64 %v3, i64 %v4, i64 %v5, i64 %v6, i64 %v7) -> i64 {
bb0:
  %v8 = mul i64 %v7, 100:i64
  %v9 = add i64 %v8, %v6
  ret i64 %v9
}
"#;
    let c = r#"#include <stdio.h>
long sum7(long a,long b,long c,long d,long e,long f,long g){ return ((((((a*10+b)*10+c)*10+d)*10+e)*10+f)*10+g); }
long sum8(long a,long b,long c,long d,long e,long f,long g,long h){ return sum7(a,b,c,d,e,f,g)*10+h; }
long call7(long), call8(long), pick(long,long,long,long,long,long,long,long);
int main(){ printf("%ld\n%ld\n%ld\n", call7(1), call8(1), pick(1,2,3,4,5,6,7,8)); return 0; }"#;
    check(ir, c, "1234567\n12345678\n807");
}

#[test]
fn globals_select_switch_indirect_and_wide_math() {
    let ir = r#"module "t"
global @tbl: [4 x i32] internal const align 4 init [0a 00 00 00 14 00 00 00 1e 00 00 00 28 00 00 00]
global @counter: i32 external align 4 init [00 00 00 00]
define external @lookup(i32 %v0) -> i32 {
bb0:
  %v1 = gep inbounds [4 x i32], @tbl, [0:i32, %v0]
  %v2 = load i32, %v1, align 4
  ret i32 %v2
}
define external @bump() -> i32 {
bb0:
  %v0 = load i32, @counter, align 4
  %v1 = add i32 %v0, 1:i32
  store i32 %v1, @counter, align 4
  ret i32 %v1
}
define external @classify(i32 %v0) -> i32 {
bb0:
  switch i32 %v0, default bb3, [1: bb1], [2: bb2]
bb1:
  ret i32 10:i32
bb2:
  ret i32 20:i32
bb3:
  ret i32 0:i32
}
define external @max(i32 %v0, i32 %v1) -> i32 {
bb0:
  %v2 = icmp sgt i32 %v0, %v1
  %v3 = select i32 %v2, %v0, %v1
  ret i32 %v3
}
define external @apply(ptr %v0, i32 %v1) -> i32 {
bb0:
  %v2 = call i32 %v0(%v1)
  ret i32 %v2
}
define internal @dbl(i32 %v0) -> i32 {
bb0:
  %v1 = mul i32 %v0, 2:i32
  ret i32 %v1
}
define external @use_fp(i32 %v0) -> i32 {
bb0:
  %v1 = call i32 @apply(@dbl, %v0)
  ret i32 %v1
}
define external @mix(i64 %v0) -> i64 {
bb0:
  %v1 = mul i64 %v0, 6364136223846793005:i64
  %v2 = add i64 %v1, 1442695040888963407:i64
  %v3 = lshr i64 %v2, 33:i64
  %v4 = xor i64 %v2, %v3
  %v5 = shl i64 %v4, 7:i64
  %v6 = ashr i64 %v5, 3:i64
  %v7 = sub i64 %v6, %v0
  ret i64 %v7
}
"#;
    let c = r#"#include <stdio.h>
extern int counter;
int lookup(int), bump(void), classify(int), max(int,int), apply(int(*)(int),int), use_fp(int);
long mix(long);
static int twice1(int x){ return 2*x+1; }
int main(){
  for(int i=0;i<4;i++) printf("%d\n", lookup(i));
  bump(); bump(); bump(); printf("%d\n", counter);
  for(int i=0;i<4;i++) printf("%d\n", classify(i));
  printf("%d %d\n", max(3,9), max(-3,-9));
  printf("%d %d\n", apply(twice1, 5), use_fp(21));
  printf("%ld %ld\n", mix(1), mix(-123456789));
  return 0; }"#;
    let mix = |x: i64| -> i64 {
        let v2 = (x as u64).wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let v4 = v2 ^ (v2 >> 33);
        (((v4 << 7) as i64) >> 3).wrapping_sub(x)
    };
    let exp = format!("10\n20\n30\n40\n3\n0\n10\n20\n0\n9 -3\n11 42\n{} {}", mix(1), mix(-123456789));
    check(ir, c, &exp);
}

#[test]
fn symbols_named_like_registers_and_mnemonics() {
    let ir = r#"module "t"
define external @st(i32 %v0) -> i32 {
bb0:
  %v1 = add i32 %v0, 1:i32
  ret i32 %v1
}
define external @rax(i32 %v0) -> i32 {
bb0:
  %v1 = call i32 @st(%v0)
  %v2 = call i32 @st(%v1)
  ret i32 %v2
}
"#;
    let c = r#"#include <stdio.h>
int rax(int);
int main(){ printf("%d\n", rax(40)); return 0; }"#;
    check(ir, c, "42");
}

fn have_libm_build(ir: &str, c: &str, expected: &str) {
    // Like `check`, but links libm and runs only when gcc is present.
    if !have_gcc() { return; }
    for opt in [false, true] {
        let mut m = parse_module(ir).unwrap_or_else(|e| panic!("parse: {e}"));
        if opt { let mut pm = PassManager::new(); llcr::opt::add_o1_pipeline(&mut pm); pm.run(&mut m); }
        let asm = kaiba_x86::compile_module(&m).unwrap_or_else(|e| panic!("codegen: {e}"));
        let dir = std::env::temp_dir().join(format!("kaiba-x86f-{}-{}", std::process::id(), N.fetch_add(1, Ordering::SeqCst)));
        std::fs::create_dir_all(&dir).unwrap();
        let (s, cf, exe) = (dir.join("out.s"), dir.join("main.c"), dir.join("prog"));
        std::fs::write(&s, &asm).unwrap();
        std::fs::write(&cf, c).unwrap();
        let o = Command::new("gcc").arg("-o").arg(&exe).arg(&cf).arg(&s).arg("-lm").output().unwrap();
        assert!(o.status.success(), "gcc failed:\n{}\n{asm}", String::from_utf8_lossy(&o.stderr));
        let r = Command::new(&exe).output().unwrap();
        assert!(r.status.success(), "crashed\n{asm}");
        assert_eq!(String::from_utf8(r.stdout).unwrap().trim(), expected.trim(), "optimize={opt}");
        let _ = std::fs::remove_dir_all(&dir);
    }
}

#[test]
fn every_float_predicate_including_nan() {
    // Bit i of the result is the i-th fcmp predicate.
    let preds = ["oeq", "ogt", "oge", "olt", "ole", "one", "ord", "ueq", "ugt", "uge", "ult", "ule", "une", "uno"];
    let mut ir = String::from("module \"t\"\n");
    for (name, ty) in [("cmpd", "f64"), ("cmpf", "f32")] {
        ir += &format!("define external @{name}({ty} %v0, {ty} %v1) -> i32 {{\nbb0:\n");
        let mut n = 2;
        let mut acc = None;
        for (i, p) in preds.iter().enumerate() {
            ir += &format!("  %v{n} = fcmp {p} {ty} %v0, %v1\n  %v{} = zext i1 %v{n} to i32\n  %v{} = shl i32 %v{}, {i}:i32\n", n + 1, n + 2, n + 1);
            let bit = n + 2;
            n += 3;
            acc = Some(match acc {
                None => bit,
                Some(a) => { ir += &format!("  %v{n} = or i32 %v{a}, %v{bit}\n"); n += 1; n - 1 }
            });
        }
        ir += &format!("  ret i32 %v{}\n}}\n", acc.unwrap());
    }
    let c = r#"#include <stdio.h>
#include <math.h>
int cmpd(double,double); int cmpf(float,float);
static int ref(double a,double b){
  int o=!isunordered(a,b);
  int r[14]={ a==b&&o, a>b&&o, a>=b&&o, a<b&&o, a<=b&&o, a!=b&&o, o, a==b||!o, a>b||!o, a>=b||!o, a<b||!o, a<=b||!o, a!=b||!o, !o };
  int m=0; for(int i=0;i<14;i++) m|=r[i]<<i; return m; }
int main(){
  double v[]={0.0,-0.0,1.5,-2.25,1e300,-1e300,INFINITY,-INFINITY,NAN,3.0};
  int bad=0, n=0;
  for(int i=0;i<10;i++) for(int j=0;j<10;j++){ n++;
    if(cmpd(v[i],v[j])!=ref(v[i],v[j])) { bad++; printf("D mismatch %d %d: %x vs %x\n",i,j,cmpd(v[i],v[j]),ref(v[i],v[j])); }
    if(cmpf((float)v[i],(float)v[j])!=ref((float)v[i],(float)v[j])) { bad++; printf("F mismatch %d %d\n",i,j); } }
  printf("checked %d, mismatches %d\n", n, bad); return 0; }
"#;
    have_libm_build(&ir, c, "checked 100, mismatches 0");
}

#[test]
fn float_arithmetic_conversions_and_mixed_arguments() {
    let ir = r#"module "t"
declare @sink(i32, f64, i64, f32, i32, i32, i32, i32, i32, f64, f64, f64, f64, f64, f64, f64, f64, f64, i32) -> i64
define external @poly(f64 %v0) -> f64 {
bb0:
  %v1 = fmul f64 %v0, f64:0x4008000000000000
  %v2 = fadd f64 %v1, f64:0xc000000000000000
  %v3 = fmul f64 %v2, %v0
  %v4 = fdiv f64 %v3, f64:0x4014000000000000
  %v5 = fsub f64 %v4, %v0
  ret f64 %v5
}
define external @conv(i64 %v0, f64 %v1) -> f64 {
bb0:
  %v2 = sitofp i64 %v0 to f64
  %v3 = uitofp i64 %v0 to f64
  %v4 = fptosi f64 %v1 to i32
  %v5 = fptoui f64 %v1 to i64
  %v6 = sitofp i32 %v4 to f32
  %v7 = fpext f32 %v6 to f64
  %v8 = fptrunc f64 %v1 to f32
  %v9 = fpext f32 %v8 to f64
  %v10 = uitofp i64 %v5 to f64
  %v11 = fadd f64 %v2, %v3
  %v12 = fadd f64 %v11, %v7
  %v13 = fadd f64 %v12, %v9
  %v14 = fadd f64 %v13, %v10
  ret f64 %v14
}
define external @callmix(i32 %v0, f64 %v1) -> i64 {
bb0:
  %v2 = sext i32 %v0 to i64
  %v3 = fptrunc f64 %v1 to f32
  %v4 = fadd f64 %v1, f64:0x3ff0000000000000
  %v5 = call i64 @sink(%v0, %v1, %v2, %v3, 4:i32, 5:i32, 6:i32, 7:i32, 8:i32, %v4, %v4, %v4, %v4, %v4, %v4, %v4, %v1, %v1, 99:i32)
  ret i64 %v5
}
define external @bigu(f64 %v0) -> i64 {
bb0:
  %v1 = fptoui f64 %v0 to i64
  ret i64 %v1
}
define external @bigf(f32 %v0) -> i64 {
bb0:
  %v1 = fptoui f32 %v0 to i64
  ret i64 %v1
}
define external @u2f(i64 %v0) -> f64 {
bb0:
  %v1 = uitofp i64 %v0 to f64
  ret f64 %v1
}
define external @fsum(f32 %v0, f32 %v1) -> f32 {
bb0:
  %v2 = fadd f32 %v0, %v1
  %v3 = fmul f32 %v2, f32:0x40000000
  ret f32 %v3
}
"#;
    let c = r#"#include <stdio.h>
#include <math.h>
unsigned long bigu(double); unsigned long bigf(float); double u2f(unsigned long);
double poly(double); double conv(long,double); long callmix(int,double); float fsum(float,float);
long sink(int a,double b,long c,float d,int e,int f,int g,int h,int i,double j,double k,double l,double m,double n,double o,double p,double q,double r,int s){
  return a*1000L + (long)(b*10) + c + (long)(d*100) + e+f+g+h+i + (long)(j+k+l+m+n+o+p+q+r) + s; }
int main(){
  for(double x=-3; x<=3; x+=1.5) printf("%.6f\n", poly(x));
  printf("%.3f %.3f\n", conv(-7, 2.75), conv(123456789012345L, -9.5+20));
  printf("%.3f\n", conv(-1, 1e9));
  printf("%ld %ld\n", callmix(7, 2.5), callmix(-3, 100.25));
  printf("%.3f\n", fsum(1.25f, 2.5f));
  double bs[]={3.0, 9.2e18, 9223372036854775808.0, 1.5e19, 1.8e19};
  for(int i=0;i<5;i++) printf("%lu %lu\n", bigu(bs[i]), bigf((float)bs[i]));
  unsigned long us[]={0, 1, 9223372036854775807UL, 9223372036854775808UL, 18446744073709551615UL, 12345678901234567891UL};
  for(int i=0;i<6;i++) printf("%.1f\n", u2f(us[i]));
  return 0; }
"#;
    // Expected values computed with the same formulas in Rust.
    let poly = |x: f64| ((x * 3.0 - 2.0) * x) / 5.0 - x;
    let conv = |a: i64, b: f64| -> f64 {
        let s = a as f64; let u = (a as u64) as f64;
        let i = b as i32; let f = i as f32 as f64; let t = (b as f32) as f64; let w = (b as u64) as f64;
        s + u + f + t + w
    };
    let sink = |a: i32, b: f64, c: i64, d: f32, rest: i64, js: f64| -> i64 {
        a as i64 * 1000 + (b * 10.0) as i64 + c + (d * 100.0) as i64 + rest + js as i64 + 99
    };
    let cm = |a: i32, x: f64| -> i64 { sink(a, x, a as i64, x as f32, 4 + 5 + 6 + 7 + 8, (x + 1.0) * 7.0 + x + x) };
    let mut exp = vec![];
    let mut x = -3.0f64; while x <= 3.0 { exp.push(format!("{:.6}", poly(x))); x += 1.5; }
    exp.push(format!("{:.3} {:.3}", conv(-7, 2.75), conv(123456789012345, 10.5)));
    exp.push(format!("{:.3}", conv(-1, 1e9)));
    exp.push(format!("{} {}", cm(7, 2.5), cm(-3, 100.25)));
    exp.push(format!("{:.3}", (1.25f32 + 2.5f32) * 2.0));
    for b in [3.0f64, 9.2e18, 9223372036854775808.0, 1.5e19, 1.8e19] { exp.push(format!("{} {}", b as u64, (b as f32) as u64)); }
    for u in [0u64, 1, 9223372036854775807, 9223372036854775808, u64::MAX, 12345678901234567891] { exp.push(format!("{:.1}", u as f64)); }
    have_libm_build(ir, c, &exp.join("\n"));
}

#[test]
fn argument_register_cycles_and_indirect_target_in_arg_register() {
    let ir = r#"module "t"
declare @take6(i64, i64, i64, i64, i64, i64) -> i64
define external @rot(i64 %v0, i64 %v1, i64 %v2, i64 %v3, i64 %v4, i64 %v5) -> i64 {
bb0:
  %v6 = call i64 @take6(%v1, %v2, %v0, %v5, %v3, %v4)
  ret i64 %v6
}
define external @swap2(i64 %v0, i64 %v1) -> i64 {
bb0:
  %v2 = call i64 @take6(%v1, %v0, 3:i64, 4:i64, 5:i64, 6:i64)
  ret i64 %v2
}
define external @icall(ptr %v0, i64 %v1, i64 %v2) -> i64 {
bb0:
  %v3 = call i64 %v0(%v2, %v1)
  ret i64 %v3
}
define external @icall3(i64 %v0, i64 %v1, ptr %v2) -> i64 {
bb0:
  %v3 = call i64 %v2(%v1, %v0, 7:i64)
  ret i64 %v3
}
"#;
    let c = r#"#include <stdio.h>
long take6(long a,long b,long c,long d,long e,long f){ return ((((a*10+b)*10+c)*10+d)*10+e)*10+f; }
long f2(long x,long y){ return x*10+y; }
long f3(long x,long y,long z){ return x*100+y*10+z; }
long rot(long,long,long,long,long,long), swap2(long,long), icall(long(*)(long,long),long,long), icall3(long,long,long(*)(long,long,long));
int main(){ printf("%ld %ld %ld %ld\n", rot(1,2,3,4,5,6), swap2(1,2), icall(f2,1,2), icall3(8,9,f3)); return 0; }
"#;
    check(ir, c, "231645 213456 21 987");
}

#[test]
fn unused_parameters_do_not_clobber_live_ones() {
    // Regression: a dead leading parameter used to share a register with the next live one
    // (found compiling zlib's `zcalloc(opaque, items, size)`).
    let ir = r#"module "t"
define external @prod(ptr %v0, i32 %v1, i32 %v2) -> i64 {
bb0:
  %v3 = mul i32 %v2, %v1
  %v4 = zext i32 %v3 to i64
  ret i64 %v4
}
define external @third(i64 %v0, i64 %v1, i64 %v2, i64 %v3) -> i64 {
bb0:
  %v4 = sub i64 %v3, %v2
  ret i64 %v4
}
define external @all_dead(i64 %v0, i64 %v1) -> i64 {
bb0:
  ret i64 7:i64
}
"#;
    let c = r#"#include <stdio.h>
long prod(void*, unsigned, unsigned), third(long,long,long,long), all_dead(long,long);
int main(){ printf("%ld %ld %ld %ld\n", prod(0, 1000, 7), prod((void*)1, 65536, 3), third(1,2,10,25), all_dead(5,6)); return 0; }
"#;
    check(ir, c, "7000 196608 15 7");
}
