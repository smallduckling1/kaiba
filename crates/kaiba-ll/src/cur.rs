// SPDX-License-Identifier: GPL-2.0-only WITH LicenseRef-Kaiba-Runtime-Exception
// Copyright (C) The Kaiba Authors
//
// This file is part of the Kaiba compiler toolchain. It is licensed under
// GPL-2.0-only with the Kaiba Runtime/Output Exception; see COPYING and
// LICENSE-EXCEPTION at the repository root.
//! Tiny scanner for LLVM textual IR lines.

pub struct Cur<'a> {
    pub s: &'a str,
    pub p: usize,
    /// Type text of the last `sret(T)` / `byval(T)` attribute seen by `skip_attrs`.
    pub sret_ty: Option<String>,
    pub byval_ty: Option<String>,
}

fn ident_char(c: char) -> bool { c.is_ascii_alphanumeric() || matches!(c, '_' | '.' | '$' | '-') }

impl<'a> Cur<'a> {
    pub fn new(s: &'a str) -> Self { Cur { s, p: 0, sret_ty: None, byval_ty: None } }
    pub fn rest(&self) -> &'a str { &self.s[self.p..] }
    pub fn ws(&mut self) { while matches!(self.rest().chars().next(), Some(' ' | '\t')) { self.p += 1; } }
    pub fn peek(&mut self) -> Option<char> { self.ws(); self.rest().chars().next() }
    pub fn eat(&mut self, t: &str) -> bool {
        self.ws();
        if self.rest().starts_with(t) { self.p += t.len(); true } else { false }
    }
    pub fn expect(&mut self, t: &str) -> Result<(), String> {
        if self.eat(t) { Ok(()) } else { Err(format!("expected `{t}` before `{}`", self.rest())) }
    }
    pub fn ident(&mut self) -> Result<&'a str, String> {
        self.ws();
        let r = self.rest();
        let n = r.find(|c: char| !ident_char(c)).unwrap_or(r.len());
        if n == 0 { return Err(format!("expected identifier before `{r}`")); }
        self.p += n;
        Ok(&r[..n])
    }
    pub fn peek_ident(&mut self) -> Option<&'a str> {
        let save = self.p;
        let r = self.ident().ok();
        self.p = save;
        r
    }
    /// Consume the keyword `k` if it is next.
    pub fn kw(&mut self, k: &str) -> bool {
        if self.peek_ident() == Some(k) { let _ = self.ident(); true } else { false }
    }
    /// Integer literal; negatives wrap to two's complement in u128.
    pub fn int(&mut self) -> Result<u128, String> {
        self.ws();
        let neg = self.rest().starts_with('-');
        if neg { self.p += 1; }
        let r = self.rest();
        let n = r.find(|c: char| !c.is_ascii_digit()).unwrap_or(r.len());
        if n == 0 { return Err(format!("expected integer before `{r}`")); }
        self.p += n;
        let v: u128 = r[..n].parse().map_err(|_| "integer literal too large".to_string())?;
        Ok(if neg { v.wrapping_neg() } else { v })
    }
    /// Quoted string with LLVM `\XX` escapes (cursor must be at the opening quote).
    pub fn quoted(&mut self) -> Result<Vec<u8>, String> {
        self.expect("\"")?;
        let b = self.rest().as_bytes();
        let (mut i, mut out) = (0, Vec::new());
        while i < b.len() {
            match b[i] {
                b'"' => { self.p += i + 1; return Ok(out); }
                b'\\' if i + 2 < b.len() && b[i + 1] == b'\\' => { out.push(b'\\'); i += 2; }
                b'\\' if i + 2 < b.len() => {
                    let h = std::str::from_utf8(&b[i + 1..i + 3]).unwrap_or("");
                    out.push(u8::from_str_radix(h, 16).map_err(|_| "bad escape in string".to_string())?);
                    i += 3;
                }
                c => { out.push(c); i += 1; }
            }
        }
        Err("unterminated string".into())
    }
    /// `%name` / `@name` (sigil required), plain or quoted.
    pub fn name(&mut self, sigil: char) -> Result<String, String> {
        self.ws();
        if !self.rest().starts_with(sigil) { return Err(format!("expected `{sigil}name` before `{}`", self.rest())); }
        self.p += 1;
        if self.rest().starts_with('"') {
            Ok(String::from_utf8_lossy(&self.quoted()?).into_owned())
        } else {
            let r = self.rest();
            let n = r.find(|c: char| !ident_char(c)).unwrap_or(r.len());
            if n == 0 { return Err("empty name".into()); }
            self.p += n;
            Ok(r[..n].to_string())
        }
    }
    /// Floating-point literal as raw bits of the requested width.
    pub fn float_bits(&mut self, is32: bool) -> Result<u64, String> {
        self.ws();
        let r = self.rest();
        let n = r.find(|c: char| !(c.is_ascii_alphanumeric() || matches!(c, '.' | '+' | '-'))).unwrap_or(r.len());
        let tok = &r[..n];
        self.p += n;
        let v: f64 = if let Some(h) = tok.strip_prefix("0x").or_else(|| tok.strip_prefix("0X")) {
            f64::from_bits(u64::from_str_radix(h, 16).map_err(|_| format!("bad hex float `{tok}`"))?)
        } else {
            tok.parse().map_err(|_| format!("bad float literal `{tok}`"))?
        };
        Ok(if is32 { (v as f32).to_bits() as u64 } else { v.to_bits() })
    }
}
