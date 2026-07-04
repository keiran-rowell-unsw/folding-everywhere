#!/usr/bin/env bash
# Build the macOS binaries (fold, fold_gui, fold_standalone) as a single
# **universal2** Mach-O each — one file that runs on both Apple Silicon (arm64)
# and Intel (x86_64) Macs.
#
# This cross-compiles from Linux (or macOS) using cargo-zigbuild + zig, so no
# Xcode/macOS SDK is required. The app is pure Rust + a local web server
# (tiny_http); it links only libSystem, which zig supplies.
#
# Prerequisites (one time):
#   rustup target add aarch64-apple-darwin x86_64-apple-darwin
#   cargo install cargo-zigbuild
#   # zig 0.13+ on PATH (https://ziglang.org/download/)
#
# Usage:  ./build_macos.sh
set -euo pipefail
cd "$(dirname "$0")"

# The repo's .cargo/config.toml pins target-cpu=native for fast *host* builds.
# That is invalid/non-portable when cross-compiling to macOS, so override it with
# a portable per-arch baseline (a single space = "no rustflags").
RUSTFLAGS=" " cargo zigbuild --release --target universal2-apple-darwin

OUT=target/universal2-apple-darwin/release
mkdir -p dist/macos
for b in fold fold_gui fold_standalone; do
  cp "$OUT/$b" "dist/macos/$b"
  chmod +x "dist/macos/$b"
done

echo "Built universal2 (arm64 + x86_64) macOS binaries into dist/macos/:"
file dist/macos/*
