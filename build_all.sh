#!/usr/bin/env bash
# Build the distributables — the `gui` app plus one CLI per model — for Linux
# x86-64, Windows x86-64 and macOS universal2 (arm64 + x86_64).
#
# `gui` runs all three models behind one page; the CLIs (`fold`, `fold_standalone`,
# `mpnn`, `rfd2`) are the same ports driven from a terminal, for scripting and
# batch work. Pure Rust, no C dependencies, no Python, no installer. The four
# ProteinMPNN checkpoints, the RFdiffusion2 chemical database / AF2 frame tables
# / SE(3) Clebsch-Gordan bases / IGSO(3) noise tables / ligand-topology library,
# and the ESMFold residue constants are all compiled in. Only the ESMFold and
# RFdiffusion2 model weights are fetched at first run.
#
# Prerequisites (one time):
#   rustup target add x86_64-pc-windows-gnu aarch64-apple-darwin x86_64-apple-darwin
#   cargo install --locked cargo-zigbuild
#   zig 0.11+ on PATH   (https://ziglang.org/download/)
#
# Usage: ./build_all.sh [--gui-only]
#   --gui-only   build just the `gui` app, skipping the four CLIs
set -euo pipefail
cd "$(dirname "$0")"

GUI_ONLY=0
[ "${1:-}" = "--gui-only" ] && GUI_ONLY=1
export PATH="$HOME/zig:$PATH"

# .cargo/config.toml pins target-cpu=native so *development* builds are fast on
# this machine. That is wrong for anything shipped, so distribution builds use a
# portable baseline: x86-64-v3 (AVX2/FMA — Haswell 2013+, Zen 1+). This is what
# the ESMFold and RFdiffusion2 repos already ship with, and it costs nothing in
# correctness: Rust never contracts a*b+c into an FMA on its own, and both ports'
# numerics are accumulate-in-f64-then-round, so they are independent of
# instruction selection. Measured on the RFdiffusion2 demo design, the more
# conservative x86-64-v2 is 1.9x slower for no portability that matters.
PORTABLE="-C target-cpu=x86-64-v3"

# The app, then one CLI per model: ESMFold1, ESMFold2, ProteinMPNN, RFdiffusion2.
BINS=(gui fold fold_standalone mpnn rfd2)
[ "$GUI_ONLY" = 1 ] && BINS=(gui)
# `cargo build` takes one --bin per binary; expand the array into those flags.
BIN_FLAGS=(); for b in "${BINS[@]}"; do BIN_FLAGS+=(--bin "$b"); done
mkdir -p dist

echo "==> Linux x86-64"
RUSTFLAGS="$PORTABLE" cargo build --release --target x86_64-unknown-linux-gnu "${BIN_FLAGS[@]}"
mkdir -p dist/linux-x86_64
for b in "${BINS[@]}"; do
  cp "target/x86_64-unknown-linux-gnu/release/$b" "dist/linux-x86_64/$b"
  chmod +x "dist/linux-x86_64/$b"
done

if command -v zig >/dev/null 2>&1 && command -v cargo-zigbuild >/dev/null 2>&1; then
  echo "==> Windows x86-64"
  RUSTFLAGS="$PORTABLE" cargo zigbuild --release --target x86_64-pc-windows-gnu "${BIN_FLAGS[@]}"
  mkdir -p dist/windows-x86_64
  for b in "${BINS[@]}"; do cp "target/x86_64-pc-windows-gnu/release/$b.exe" "dist/windows-x86_64/$b.exe"; done

  # Remove previous universal2 output first: lipo does not always overwrite a
  # stale file in place, and a leftover from an earlier build silently produces
  # a binary with the wrong embedded page. Cheap to delete; expensive to notice.
  echo "==> macOS universal2 (arm64 + x86_64)"
  mkdir -p dist/macos-universal
  # ONE --bin per invocation. Passing several to a universal2 build lets the lipo
  # step pair the wrong slices: measured with all five at once, the x86_64 halves
  # were right but the arm64 halves of gui/mpnn/rfd2 were rotated between binaries,
  # so `gui` on Apple Silicon actually ran mpnn. Nothing warns you -- `file` still
  # reports a valid 2-architecture binary. Build them one at a time and check the
  # slice sizes against target/{aarch64,x86_64}-apple-darwin/release/.
  #
  # Removing the previous output first is also required: lipo does not always
  # overwrite a stale file in place, and a leftover from an earlier build silently
  # produces a binary with the wrong embedded page.
  for b in "${BINS[@]}"; do
    rm -f "target/universal2-apple-darwin/release/$b"
    # A single space = "no rustflags"; target-cpu=x86-64-v3 is invalid for arm64.
    RUSTFLAGS=" " cargo zigbuild --release --target universal2-apple-darwin --bin "$b"
    cp "target/universal2-apple-darwin/release/$b" "dist/macos-universal/$b"
    chmod +x "dist/macos-universal/$b"
  done
else
  echo "!! zig / cargo-zigbuild not on PATH — skipping Windows and macOS builds"
fi

echo
echo "Built:"
find dist -type f | sort | while read -r f; do
  printf "  %-40s %6.1f MB\n" "$f" "$(stat -c%s "$f" | awk '{print $1/1048576}')"
done
