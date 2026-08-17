# Deployment — what ships, and how to run it

Folding Everywhere v2 ships **one app for all three models** plus **one CLI per model**,
prebuilt for the three desktop platforms in [`../../dist/`](../../dist/). Nothing is
installed, no Python, no GPU, no C dependencies.

| Platform | App | ProteinMPNN CLI |
|---|---|---|
| Windows x86-64 | `gui.exe` | `mpnn.exe` |
| macOS universal (arm64 + Intel) | `gui` | `mpnn` |
| Linux x86-64 | `gui` | `mpnn` |

**ProteinMPNN needs no download at all.** All four published checkpoints
(`v_48_002/010/020/030`) are embedded with `include_bytes!` (~6.7 MB each), so both the app and
the CLI are self-contained: copy one file to a machine with no network and it designs
sequences immediately. That is why `mpnn` is ~27 MB where the other CLIs are a few MB.

The app is documented in [`../../docs/GUI.md`](../../docs/GUI.md); this file covers the
command-line binary.

## Running

```bash
# 8 designs at T = 0.1, seed 37 — the same sequences protein_mpnn_run.py gives
mpnn --pdb backbone.pdb --num_seq_per_target 8 --sampling_temp 0.1 --seed 37

mpnn --pdb backbone.pdb --model_name v_48_002 --out designs.fa
mpnn --pdb backbone.pdb --score_only              # just score the native sequence
```

Output is FASTA on stdout (or `--out`), in the reference's own layout. `mpnn` with no
arguments prints the full option list.

On macOS the binaries are unsigned, so clear the Gatekeeper quarantine once:

```bash
xattr -dr com.apple.quarantine gui mpnn && chmod +x gui mpnn
```

## End-user machine requirements

| | |
|---|---|
| OS | Windows 10+, macOS 11+, any current Linux |
| CPU | x86-64 with AVX2 (any PC from ~2013 on), or Apple Silicon |
| RAM | ~2 GB |
| Disk | the executable only — no weights to cache |
| Network | none |

## Building the distributables

```bash
./build_all.sh            # from the repo root: all three platforms, into dist/
./build_all.sh --gui-only # just the app
```

Cross-compiling needs [`cargo-zigbuild`](https://github.com/rust-cross/cargo-zigbuild) and
zig 0.11+; the full prerequisites are in [`../../docs/BUILD.md`](../../docs/BUILD.md).

A shipped binary should reproduce the reference exactly, so it is worth checking one after a
release build:

```bash
./dist/linux-x86_64/mpnn --pdb 5L33.pdb --num_seq_per_target 4 --seed 37 > portable.fa
diff portable.fa results/reference.fa
```

### Why distribution builds override `target-cpu`

The workspace's `.cargo/config.toml` pins `target-cpu=native` so *development* builds are fast
on the build machine. That is wrong for anything shipped, so `build_all.sh` overrides it with a
portable baseline, `x86-64-v3` (AVX2/FMA — Haswell 2013+, Zen 1+).

This costs nothing in correctness: Rust never contracts `a*b+c` into an FMA on its own, and the
few deliberate `mul_add` calls are correctly rounded whether they lower to a hardware FMA or the
libm fallback. The macOS universal2 build passes no `target-cpu` at all, because `x86-64-v3` is
not a valid value for arm64.
