// SPDX-License-Identifier: GPL-2.0-only WITH LicenseRef-Kaiba-Runtime-Exception
// Copyright (C) The Kaiba Authors
//
// This file is part of the Kaiba compiler toolchain. It is licensed under
// GPL-2.0-only with the Kaiba Runtime/Output Exception; see COPYING and
// LICENSE-EXCEPTION at the repository root.
//! `kaiba`: command-line driver for the Kaiba compiler toolchain.

use llcr::opt::{self, PASS_NAMES};
use llcr::verify::verify_function;
use llcr::{parse_module, Module, PassManager};
use std::io::{Read, Write};
use std::process::Command;

const USAGE: &str = "\
kaiba: llcr compiler driver

Inputs may be llcr (.llcr) or LLVM IR text (.ll, subset); the format is
detected from the extension or the contents.

USAGE:
    kaiba opt    [OPTIONS] <input.llcr | ->        optimize, print llcr
    kaiba asm    [OPTIONS] <input.llcr | ->        optimize, print assembly (x86_64 or aarch64)
    kaiba build  [OPTIONS] -o <exe> <input.llcr> [cc args...]
                                                   compile and link with the system `cc`
    kaiba verify <input.llcr | ->                  check that the IR is well formed
    kaiba passes                                   list available passes

OPTIONS:
    -O0, -O1             optimization level (default -O1)
    --passes=a,b,c       run exactly these passes (overrides -O)
    -o <file>            output file (stdout if omitted, except for `build`)
    --target=T           x86_64 or aarch64 (default: from the module's target triple, else x86_64)
    --no-verify          do not verify the IR after each pass
    --stats              print which passes changed the IR to stderr
";

#[derive(Default)]
struct Opts {
    level: u8,
    passes: Option<Vec<String>>,
    out: Option<String>,
    verify: bool,
    stats: bool,
    input: Option<String>,
    target: Option<String>,
    /// Arguments after the input file (only collected when `trailing` is set).
    rest: Vec<String>,
}

fn parse_opts(args: &[String], trailing: bool) -> Result<Opts, String> {
    let mut o = Opts { level: 1, verify: true, ..Default::default() };
    let mut i = 0;
    while i < args.len() {
        if o.input.is_some() && trailing {
            o.rest.extend_from_slice(&args[i..]);
            break;
        }
        let a = &args[i];
        match a.as_str() {
            "-O0" => o.level = 0,
            "-O1" => o.level = 1,
            "-o" => { i += 1; o.out = Some(args.get(i).ok_or("-o requires a file name")?.clone()); }
            "--no-verify" => o.verify = false,
            s if s.starts_with("--target=") => o.target = Some(s["--target=".len()..].to_string()),
            "--stats" => o.stats = true,
            s if s.starts_with("--passes=") => {
                o.passes = Some(s["--passes=".len()..].split(',').filter(|p| !p.is_empty()).map(str::to_owned).collect());
            }
            s if s.starts_with('-') && s != "-" => return Err(format!("unknown option `{s}`\n\n{USAGE}")),
            s => { if o.input.replace(s.to_string()).is_some() { return Err("only one input file is supported".into()); } }
        }
        i += 1;
    }
    Ok(o)
}

fn read_input(path: &str) -> Result<String, String> {
    if path == "-" {
        let mut s = String::new();
        std::io::stdin().read_to_string(&mut s).map_err(|e| format!("reading stdin: {e}"))?;
        Ok(s)
    } else {
        std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))
    }
}

/// `.ll` (LLVM IR) vs `.llcr`: by extension, else by sniffing the first line.
fn is_llvm_ir(path: &str, src: &str) -> bool {
    if path.ends_with(".ll") { return true; }
    if path.ends_with(".llcr") { return false; }
    let first = src.lines().map(str::trim).find(|l| !l.is_empty() && !l.starts_with(';'));
    !matches!(first, Some(l) if l.starts_with("module \""))
}

fn load(path: &str) -> Result<Module, String> {
    let src = read_input(path)?;
    if is_llvm_ir(path, &src) {
        kaiba_ll::parse_ll(&src).map_err(|e| format!("{path}: {e}"))
    } else {
        parse_module(&src).map_err(|e| format!("{path}: {e}"))
    }
}

fn verify_module(m: &Module) -> Result<(), String> {
    let errs: Vec<String> = m
        .functions.iter()
        .filter(|(_, f)| !f.is_declaration())
        .filter_map(|(_, f)| verify_function(f).err().map(|e| e.0))
        .collect();
    if errs.is_empty() { Ok(()) } else { Err(errs.join("\n")) }
}

/// Load, verify and optimize according to `o`.
fn prepare(o: &Opts) -> Result<Module, String> {
    let input = o.input.as_deref().ok_or_else(|| format!("missing input file\n\n{USAGE}"))?;
    let mut m = load(input)?;
    verify_module(&m).map_err(|e| format!("input IR is invalid:\n{e}"))?;
    let mut pm = PassManager::new();
    pm.verify_each = o.verify;
    match &o.passes {
        Some(list) => {
            for n in list {
                let p = opt::create_pass(n)
                    .ok_or_else(|| format!("unknown pass `{n}` (available: {})", PASS_NAMES.join(", ")))?;
                opt::add_pass(&mut pm, p);
            }
        }
        None if o.level >= 1 => opt::add_o1_pipeline(&mut pm),
        None => {}
    }
    pm.run(&mut m);
    verify_module(&m).map_err(|e| format!("internal error: optimizer produced invalid IR:\n{e}"))?;
    if o.stats {
        for (name, changed) in &pm.trace { eprintln!("{name}: {}", if *changed { "changed" } else { "no change" }); }
    }
    Ok(m)
}

fn write_output(out: &Option<String>, text: &str) -> Result<(), String> {
    match out {
        Some(p) => std::fs::write(p, text).map_err(|e| format!("{p}: {e}")),
        None => std::io::stdout().write_all(text.as_bytes()).map_err(|e| e.to_string()),
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Target { X86, A64 }

/// Explicit `--target=`, else the module's target triple, else x86_64.
fn resolve_target(m: &Module, o: &Opts) -> Result<Target, String> {
    let from_name = |n: &str| -> Result<Target, String> {
        match n {
            "x86_64" | "x86-64" | "x86" | "amd64" => Ok(Target::X86),
            "aarch64" | "arm64" | "a64" => Ok(Target::A64),
            other => Err(format!("unknown target `{other}` (expected x86_64 or aarch64)")),
        }
    };
    if let Some(t) = &o.target { return from_name(t); }
    Ok(match m.target_triple.as_deref() {
        Some(t) if t.starts_with("aarch64") || t.starts_with("arm64") => Target::A64,
        _ => Target::X86,
    })
}

fn compile_asm(m: &Module, t: Target) -> Result<String, String> {
    match t {
        Target::X86 => kaiba_x86::compile_module(m).map_err(|e| format!("codegen: {e}")),
        Target::A64 => kaiba_a64::compile_module(m).map_err(|e| format!("codegen: {e}")),
    }
}

fn cmd_build(args: &[String]) -> Result<(), String> {
    let o = parse_opts(args, true)?;
    let exe = o.out.clone().ok_or("`build` requires -o <exe>")?;
    let m = prepare(&o)?;
    let target = resolve_target(&m, &o)?;
    let asm = compile_asm(&m, target)?;
    let tmp = std::env::temp_dir().join(format!("kaiba-{}.s", std::process::id()));
    std::fs::write(&tmp, asm).map_err(|e| format!("{}: {e}", tmp.display()))?;
    let default_cc = if target == Target::A64 { "aarch64-linux-gnu-gcc" } else { "cc" };
    let cc = std::env::var("KAIBA_CC").unwrap_or_else(|_| default_cc.to_string());
    let status = Command::new(&cc).arg("-o").arg(&exe).arg(&tmp).args(&o.rest).status();
    let _ = std::fs::remove_file(&tmp);
    match status {
        Ok(s) if s.success() => Ok(()),
        Ok(s) => Err(format!("linker driver `{cc}` failed ({s})")),
        Err(e) => Err(format!("could not run `{cc}`: {e}")),
    }
}

fn real_main(args: Vec<String>) -> Result<(), String> {
    match args.first().map(String::as_str) {
        Some("opt") => {
            let o = parse_opts(&args[1..], false)?;
            write_output(&o.out, &prepare(&o)?.to_string())
        }
        Some("asm") => {
            let o = parse_opts(&args[1..], false)?;
            let m = prepare(&o)?;
            let t = resolve_target(&m, &o)?;
            write_output(&o.out, &compile_asm(&m, t)?)
        }
        Some("build") => cmd_build(&args[1..]),
        Some("verify") => {
            let path = args.get(1).ok_or_else(|| format!("missing input file\n\n{USAGE}"))?;
            verify_module(&load(path)?)?;
            println!("{path}: ok");
            Ok(())
        }
        Some("passes") => { for p in PASS_NAMES { println!("{p}"); } Ok(()) }
        Some("--version") | Some("-V") => { println!("kaiba {}", env!("CARGO_PKG_VERSION")); Ok(()) }
        Some("--help") | Some("-h") | None => { print!("{USAGE}"); Ok(()) }
        Some(c) => Err(format!("unknown command `{c}`\n\n{USAGE}")),
    }
}

fn main() {
    if let Err(e) = real_main(std::env::args().skip(1).collect()) {
        eprintln!("kaiba: error: {e}");
        std::process::exit(1);
    }
}
