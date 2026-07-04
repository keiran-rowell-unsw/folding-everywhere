# Deployment — standalone executable for Windows / macOS / Linux

## Short answer

**Yes — this folds proteins from a single, no-install native executable on Windows,
macOS, and Linux.** The code is pure Rust with **zero OS-specific or C dependencies**
(no Python, no PyTorch, no CUDA, no DLLs/shared libs to install). It compiles to one
self-contained binary per platform:

| Platform | Executable |
|---|---|
| Windows (x86-64) | `fold.exe` |
| macOS (Intel x86-64 **and** Apple Silicon arm64) | `fold` |
| Linux (x86-64) | `fold` (already built here) |

The residue constants are baked into the binary (`include_bytes!`), so the executable
itself is small (a few MB) and self-contained.

## The one catch: the model weights are a separate ~8.4 GB data file

The trained network weights (`model.safetensors`, **8.44 GB**) are **data, not code** —
they are *not* inside the executable. So the "download and run" package is two files:

```
fold(.exe)            # tiny (~few MB) — the program
model.safetensors     # 8.44 GB — the trained weights (download once)
```

Run:

```bash
# Windows
fold.exe --seq MQIFVKTLTGKTITLEV... --weights model.safetensors -o out.pdb

# macOS / Linux
./fold --seq MQIFVKTLTGKTITLEV... --weights model.safetensors -o out.pdb

# or a FASTA file with multiple sequences
./fold --fasta seqs.fasta --weights model.safetensors
```

Output: a PDB file (pLDDT confidence in the B-factor column).

> It could technically be made *literally one file* by embedding the 8.4 GB weights
> into the exe (≈8.5 GB executable), but keeping weights separate is the normal,
> easier-to-update approach.

## End-user machine requirements

| Resource | Needed |
|---|---|
| RAM | ~10 GB free (weights are memory-mapped). 16 GB laptop is fine; 8 GB is too tight. |
| Disk | ~9 GB for the weights file |
| CPU | any modern x86-64 (Windows / Intel Mac) or ARM64 (Apple Silicon M1/M2/M3); **no GPU** |
| Time | CPU-only → minutes per protein (e.g. flgM ≈ 7 min on 4 cores; faster with more cores) |
| Install | **nothing** — no Python / PyTorch / CUDA / runtime libraries |

CPU-speed is the deliberate tradeoff for being dependency-free and fp32-accurate.

## Getting the weights file

`model.safetensors` is the `facebook/esmfold_v1` checkpoint (ESM-2 weights stored
F16, folding head F32). Download once from Hugging Face, e.g.:

```bash
pip install huggingface_hub        # only on a machine that has Python; one-time
hf download facebook/esmfold_v1 model.safetensors --local-dir ./weights
```

(If the repo only ships `pytorch_model.bin`, convert it to safetensors once with
`safetensors.torch.save_file`; this project already used a cached safetensors copy.)

## How the Windows / macOS executables are produced

The code needs **no changes** to target other OSes — it's portable pure Rust.

### Option A (simplest, most reliable): build on each target OS
On a Windows PC or a Mac, install Rust once (`rustup`), then:

```bash
cargo build --release --bin fold
# Windows -> target\release\fold.exe
# macOS   -> target/release/fold
```

Only the *person building* needs `rustup`; the *end user* needs nothing.

### Option B: cross-compile from Linux
- **Windows:** works via the `zig` toolchain (no admin rights):
  ```bash
  # one-time: download zig, add to PATH
  cargo install --locked cargo-zigbuild
  rustup target add x86_64-pc-windows-gnu
  RUSTFLAGS="-C target-cpu=x86-64-v2" \
    cargo zigbuild --release --bin fold --target x86_64-pc-windows-gnu
  # -> target/x86_64-pc-windows-gnu/release/fold.exe
  ```
  (`target-cpu=x86-64-v2` keeps the exe portable across CPUs rather than tuned to the
  build host. Use `aarch64-pc-windows-...` only for ARM Windows.)
- **macOS:** cross-compiling from Linux needs Apple's SDK and is fiddly — **build on a
  Mac** (Option A). For Apple Silicon use the default target on an M-series Mac; for
  Intel Macs use `x86_64-apple-darwin`.

## Portability notes
- All crates used (`memmap2`, `safetensors`, `half`, `rayon`, `libm`, `serde_json`)
  are cross-platform; file I/O and threading use only Rust `std`.
- The only Linux/macOS-ism is the *auto-detect-weights-in-HF-cache* convenience path;
  on Windows just always pass `--weights <path>` (the normal usage anyway).
- For widest CPU compatibility, build distribution binaries with a baseline
  `target-cpu` (e.g. `x86-64-v2`) instead of `native`.
