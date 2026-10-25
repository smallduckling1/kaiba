#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-2.0-only WITH LicenseRef-Kaiba-Runtime-Exception
#
# Real-world stress test: compile zlib with Kaiba and compare with a gcc-built zlib.
#
#   tools/zlib-check.sh [x86|a64|both]      (default: x86)
#
# For each target and clang optimization level it checks that
#   1. zlib's own test program (test/example.c) prints exactly what the gcc build prints, and
#   2. tools/zlib-stress.c produces byte-identical compressed streams (every level/strategy).
# Needs network access (downloads zlib 1.3.1), clang, gcc, and for a64 also
# aarch64-linux-gnu-gcc + qemu-aarch64.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WORK="${WORK:-/tmp/kaiba-zlib}"
KAIBA_OPTS="${KAIBA_OPTS:--O1}"
SRCS="adler32 compress crc32 deflate gzclose gzlib gzread gzwrite infback inffast inflate inftrees trees uncompr zutil"
DEFS="-DHAVE_UNISTD_H=1 -D_LARGEFILE64_SOURCE=1"

mkdir -p "$WORK"
if [ ! -d "$WORK/zlib-1.3.1" ]; then
  curl -sSL -o "$WORK/zlib.tar.gz" https://codeload.github.com/madler/zlib/tar.gz/refs/tags/v1.3.1
  tar xzf "$WORK/zlib.tar.gz" -C "$WORK"
fi
(cd "$ROOT" && cargo build --release -q -p kaiba)
KAIBA="$ROOT/target/release/kaiba"
cd "$WORK/zlib-1.3.1"

check_target() {
  local tgt="$1" triple cc run
  if [ "$tgt" = a64 ]; then triple=aarch64-linux-gnu; cc=aarch64-linux-gnu-gcc; run="qemu-aarch64 -L /usr/aarch64-linux-gnu"
  else triple=x86_64-linux-gnu; cc=gcc; run=""; fi
  local ref="$WORK/ref-$tgt"; mkdir -p "$ref"
  $cc -O2 -w -I. $DEFS -o "$ref/example" test/example.c $(for s in $SRCS; do echo $s.c; done)
  $cc -O2 -w -I. $DEFS -o "$ref/stress" "$ROOT/tools/zlib-stress.c" $(for s in $SRCS; do echo $s.c; done)
  (cd "$ref" && $run ./example > example.out 2>&1 && $run ./stress > stress.out 2>/dev/null)
  for copt in -O0 -O1 -O2; do
    local out="$WORK/kaiba-$tgt$copt"; rm -rf "$out"; mkdir -p "$out"
    for s in $SRCS; do
      clang --target=$triple -S -emit-llvm $copt -w -I. $DEFS -fno-vectorize -fno-slp-vectorize -o "$out/$s.ll" $s.c
      "$KAIBA" asm $KAIBA_OPTS "$out/$s.ll" > "$out/$s.s"
    done
    $cc -w -I. $DEFS -o "$out/example" test/example.c "$out"/*.s
    $cc -O2 -w -I. $DEFS -o "$out/stress" "$ROOT/tools/zlib-stress.c" "$out"/*.s
    (cd "$out" && $run ./example > example.out 2>&1 && $run ./stress > stress.out 2>/dev/null)
    cmp -s "$out/example.out" "$ref/example.out" && cmp -s "$out/stress.out" "$ref/stress.out" \
      && echo "OK   $tgt clang$copt kaiba$KAIBA_OPTS: example output and $(grep -c level "$ref/stress.out") compressed streams identical" \
      || { echo "FAIL $tgt clang$copt kaiba$KAIBA_OPTS"; diff "$out/example.out" "$ref/example.out" | head -5; diff "$out/stress.out" "$ref/stress.out" | head -5; return 1; }
  done
}

case "${1:-x86}" in
  x86) check_target x86 ;;
  a64) check_target a64 ;;
  both) check_target x86; check_target a64 ;;
  *) echo "usage: $0 [x86|a64|both]" >&2; exit 2 ;;
esac
