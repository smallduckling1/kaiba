// SPDX-License-Identifier: GPL-2.0-only WITH LicenseRef-Kaiba-Runtime-Exception
use std::io::Write;
use std::process::{Command, Stdio};

const SUM: &str = r#"module "m"
define external @sum(i32 %v0) -> i32 {
bb0:
  %v1 = alloca i32, 1:i32, align 4
  store i32 7:i32, %v1, align 4
  %v2 = load i32, %v1, align 4
  %v3 = add i32 %v2, %v0
  ret i32 %v3
}
"#;

const BAD: &str = r#"module "m"
define external @f(i1 %v0) -> i32 {
bb0:
  condbr %v0, bb1, bb2
bb1:
  %v1 = add i32 1:i32, 2:i32
  br bb3
bb2:
  br bb3
bb3:
  ret i32 %v1
}
"#;

fn kaiba(args: &[&str], stdin: Option<&str>) -> (bool, String, String) {
    let mut c = Command::new(env!("CARGO_BIN_EXE_kaiba"));
    c.args(args).stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::piped());
    let mut child = c.spawn().unwrap();
    if let Some(s) = stdin { child.stdin.as_mut().unwrap().write_all(s.as_bytes()).unwrap(); }
    drop(child.stdin.take());
    let o = child.wait_with_output().unwrap();
    (o.status.success(), String::from_utf8_lossy(&o.stdout).into(), String::from_utf8_lossy(&o.stderr).into())
}

#[test]
fn opt_o1_from_stdin() {
    let (ok, out, err) = kaiba(&["opt", "-O1", "-"], Some(SUM));
    assert!(ok, "{err}");
    assert!(!out.contains("alloca") && !out.contains("load"), "{out}");
    assert!(out.contains("add i32 7:i32") || out.contains(", 7:i32"), "{out}");
}

#[test]
fn opt_o0_keeps_memory_ops() {
    let (ok, out, _) = kaiba(&["opt", "-O0", "-"], Some(SUM));
    assert!(ok);
    assert!(out.contains("alloca"));
}

#[test]
fn explicit_passes_and_stats() {
    let (ok, out, err) = kaiba(&["opt", "--passes=mem2reg,dce", "--stats", "-"], Some(SUM));
    assert!(ok, "{err}");
    assert!(err.contains("mem2reg: changed"), "{err}");
    assert!(!out.contains("alloca"));
}

#[test]
fn unknown_pass_is_an_error() {
    let (ok, _, err) = kaiba(&["opt", "--passes=bogus", "-"], Some(SUM));
    assert!(!ok);
    assert!(err.contains("unknown pass `bogus`") && err.contains("mem2reg"), "{err}");
}

#[test]
fn writes_output_file() {
    let path = std::env::temp_dir().join(format!("kaiba-cli-test-{}.llcr", std::process::id()));
    let (ok, out, err) = kaiba(&["opt", "-o", path.to_str().unwrap(), "-"], Some(SUM));
    assert!(ok, "{err}");
    assert!(out.is_empty());
    assert!(std::fs::read_to_string(&path).unwrap().contains("define external @sum"));
    let _ = std::fs::remove_file(path);
}

#[test]
fn verify_accepts_good_and_rejects_bad() {
    let dir = std::env::temp_dir();
    let (good, bad) = (dir.join(format!("kaiba-good-{}.llcr", std::process::id())), dir.join(format!("kaiba-bad-{}.llcr", std::process::id())));
    std::fs::write(&good, SUM).unwrap();
    std::fs::write(&bad, BAD).unwrap();
    let (ok, out, _) = kaiba(&["verify", good.to_str().unwrap()], None);
    assert!(ok && out.contains("ok"));
    let (ok, _, err) = kaiba(&["verify", bad.to_str().unwrap()], None);
    assert!(!ok);
    assert!(err.contains("does not dominate"), "{err}");
    // `opt` also refuses invalid input.
    let (ok, _, err) = kaiba(&["opt", bad.to_str().unwrap()], None);
    assert!(!ok && err.contains("input IR is invalid"), "{err}");
    let _ = std::fs::remove_file(good);
    let _ = std::fs::remove_file(bad);
}

#[test]
fn parse_errors_include_location() {
    let (ok, _, err) = kaiba(&["opt", "-"], Some("module \"x\"\ndefine external @f() -> void {\nbb0:\n  frobnicate\n  ret\n}\n"));
    assert!(!ok);
    assert!(err.contains("line 4") && err.contains("unknown instruction"), "{err}");
}

fn have_cc() -> bool { Command::new("cc").arg("--version").output().map(|o| o.status.success()).unwrap_or(false) }

#[test]
fn asm_subcommand_emits_x86() {
    let (ok, out, err) = kaiba(&["asm", "-"], Some(SUM));
    assert!(ok, "{err}");
    assert!(out.contains(".intel_syntax noprefix") && out.contains("\"sum\":") && out.contains("ret"), "{out}");
}

#[test]
fn asm_reports_unsupported_features() {
    let src = "module \"m\"\ndefine external @f(i32 %v0) -> i32 {\nbb0:\n  %v1 = alloca i32, %v0, align 4\n  ret i32 0:i32\n}\n";
    let (ok, _, err) = kaiba(&["asm", "-O0", "-"], Some(src));
    assert!(!ok);
    assert!(err.contains("codegen:") && err.contains("dynamic alloca"), "{err}");
}

#[test]
fn build_links_and_runs() {
    if !have_cc() { return; }
    let dir = std::env::temp_dir().join(format!("kaiba-build-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let (ir, c, exe) = (dir.join("sum.llcr"), dir.join("main.c"), dir.join("prog"));
    std::fs::write(&ir, SUM).unwrap();
    std::fs::write(&c, "#include <stdio.h>\nint sum(int);\nint main(){printf(\"%d\\n\", sum(5));return 0;}\n").unwrap();
    let (ok, _, err) = kaiba(&["build", "-O1", "-o", exe.to_str().unwrap(), ir.to_str().unwrap(), c.to_str().unwrap()], None);
    assert!(ok, "{err}");
    let out = Command::new(&exe).output().unwrap();
    assert_eq!(String::from_utf8_lossy(&out.stdout).trim(), "12"); // 7 + 5
    let _ = std::fs::remove_dir_all(&dir);
}

const LL: &str = r#"; ModuleID = 'x.c'
target triple = "x86_64-pc-linux-gnu"
define dso_local i32 @addmul(i32 noundef %a, i32 noundef %b) #0 {
entry:
  %add = add nsw i32 %a, %b
  %mul = mul nsw i32 %add, 3
  ret i32 %mul
}
attributes #0 = { noinline nounwind }
"#;

#[test]
fn accepts_llvm_ir_and_converts_to_llcr() {
    let (ok, out, err) = kaiba(&["opt", "-O0", "-"], Some(LL));
    assert!(ok, "{err}");
    assert!(out.contains("target \"x86_64-pc-linux-gnu\"") && out.contains("define external @addmul(i32 %v0, i32 %v1) -> i32"), "{out}");
    assert!(out.contains("mul i32"), "{out}");
}

#[test]
fn builds_llvm_ir_into_an_executable() {
    if !have_cc() { return; }
    let dir = std::env::temp_dir().join(format!("kaiba-ll-build-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let (ir, c, exe) = (dir.join("x.ll"), dir.join("main.c"), dir.join("prog"));
    std::fs::write(&ir, LL).unwrap();
    std::fs::write(&c, "#include <stdio.h>\nint addmul(int,int);\nint main(){printf(\"%d\\n\", addmul(2,5));return 0;}\n").unwrap();
    let (ok, _, err) = kaiba(&["build", "-o", exe.to_str().unwrap(), ir.to_str().unwrap(), c.to_str().unwrap()], None);
    assert!(ok, "{err}");
    let out = Command::new(&exe).output().unwrap();
    assert_eq!(String::from_utf8_lossy(&out.stdout).trim(), "21");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn llvm_ir_errors_name_the_line() {
    let bad = "define i32 @f() {\nentry:\n  fence seq_cst\n  ret i32 0\n}\n";
    let (ok, _, err) = kaiba(&["verify", "-"], Some(bad));
    assert!(!ok);
    assert!(err.contains("line 3") && err.contains("unsupported instruction"), "{err}");
}

#[test]
fn asm_target_flag_and_triple_inference() {
    let (ok, out, err) = kaiba(&["asm", "--target=aarch64", "-"], Some(SUM));
    assert!(ok, "{err}");
    assert!(out.contains("stp x29, x30") && !out.contains("intel_syntax"), "{out}");
    let (ok, out, _) = kaiba(&["asm", "--target=x86_64", "-"], Some(SUM));
    assert!(ok && out.contains(".intel_syntax"));
    // The module's target triple selects the backend when no flag is given.
    let ll = LL.replace("x86_64-pc-linux-gnu", "aarch64-unknown-linux-gnu");
    let (ok, out, err) = kaiba(&["asm", "-"], Some(&ll));
    assert!(ok, "{err}");
    assert!(out.contains("ret") && out.contains("x29"), "{out}");
    let (ok, _, err) = kaiba(&["asm", "--target=mips", "-"], Some(SUM));
    assert!(!ok && err.contains("unknown target"), "{err}");
}

#[test]
fn builds_and_runs_aarch64_under_qemu() {
    let have = |c: &str| Command::new(c).arg("--version").output().map(|o| o.status.success()).unwrap_or(false);
    if !have("aarch64-linux-gnu-gcc") || !have("qemu-aarch64") { return; }
    let dir = std::env::temp_dir().join(format!("kaiba-a64-build-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let (ir, c, exe) = (dir.join("sum.llcr"), dir.join("main.c"), dir.join("prog"));
    std::fs::write(&ir, SUM).unwrap();
    std::fs::write(&c, "#include <stdio.h>\nint sum(int);\nint main(){printf(\"%d\\n\", sum(5));return 0;}\n").unwrap();
    let (ok, _, err) = kaiba(&["build", "--target=aarch64", "-o", exe.to_str().unwrap(), ir.to_str().unwrap(), c.to_str().unwrap()], None);
    assert!(ok, "{err}");
    let out = Command::new("qemu-aarch64").args(["-L", "/usr/aarch64-linux-gnu"]).arg(&exe).output().unwrap();
    assert_eq!(String::from_utf8_lossy(&out.stdout).trim(), "12");
    let _ = std::fs::remove_dir_all(&dir);
}
