# Building

One Cargo workspace holds the app and all four model crates. There are no C dependencies, no
build scripts and no code generation — `cargo build` is the whole story.

```bash
cargo build --release --bin gui   # just the app
cargo build --release             # also the CLIs: fold, fold_standalone, mpnn, rfd2
cargo test  --release             # the parity suites
```

Toolchain: **Rust 1.75+** (developed on 1.95). Build-time crates: `memmap2`, `safetensors`,
`half`, `bytemuck`, `rayon`, `matrixmultiply`, `libm`, `serde_json`, `tiny_http`.

## Workspace members

| Member | Library | Binaries |
|---|---|---|
| `gui` | — | **`gui`** ← the shipped app |
| `esmfold/esmfold1` | `esmfold` | `fold` |
| `esmfold/esmfold2` | `esmfold2` | `fold_standalone` |
| `proteinmpnn/mpnn` | `proteinmpnn` | `mpnn` |
| `rfdiffusion2/rfd2` | `rfd2` | `rfd2` |

`[profile.release]` is `opt-level = 3`, `lto = false`, `codegen-units = 16` — deliberately
identical to the three source repos, so the merged build's numerics match what each port was
validated with.

## Data compiled into the binary

Several files must be present at build time because `include_bytes!` reads them, and their
paths are **relative to the source file**. This is why each model keeps its own subtree
rather than sharing one flat `fixtures/` directory.

| Compiled-in file | Read by |
|---|---|
| `esmfold/esmfold1/fixtures/constants/residue_constants.safetensors` | `esmfold1/src/constants.rs` |
| `esmfold/esmfold2/src/featurize_tables.json` | `esmfold2/src/featurize.rs` |
| `proteinmpnn/weights/v_48_{002,010,020,030}.pt` (27 MB) | `mpnn/src/embedded.rs` (`../../weights/`) |
| `rfdiffusion2/rfd2/data/{chemical,openfold,se3_cg}.safetensors` | `rfd2/src/{chemical,openfold,model/se3}.rs` |
| `gui/data/{igso3,ligand_library}.safetensors`, `ligand_library.json` | `gui/src/rfd2.rs` |
| `gui/data/example_M0584_1ldm.pdb`, `example_6EKB.pdb` | `gui/src/{rfd2,mpnn}.rs` |

Likewise `proteinmpnn/mpnn/tests/` and `rfdiffusion2/rfd2/tests/` resolve fixtures as
`{CARGO_MANIFEST_DIR}/../fixtures`, i.e. the subtree root.

> Note for anyone porting a change back to the v1 repo: in `folding-everywhere` the
> ESMFold1 constants file exists only at the repo root, so `cargo build -p esmfold1` fails
> there. This repo carries the copy at the path `constants.rs` actually asks for.

## Cross-compiling

`build_all.sh` produces all three distributables:

```bash
./build_all.sh              # the app plus one CLI per model, for all three platforms
./build_all.sh --gui-only   # just the app
# dist/linux-x86_64/{gui,fold,fold_standalone,mpnn,rfd2}
# dist/windows-x86_64/{gui,fold,fold_standalone,mpnn,rfd2}.exe
# dist/macos-universal/{gui,fold,fold_standalone,mpnn,rfd2}   (arm64 + x86_64 in one file)
```

One-time prerequisites:

```bash
rustup target add x86_64-pc-windows-gnu aarch64-apple-darwin x86_64-apple-darwin
cargo install --locked cargo-zigbuild
# zig 0.11+ on PATH — https://ziglang.org/download/
```

No Xcode, no macOS SDK, no Windows toolchain: zig supplies the linkers and the system
libraries, and the app links only libc / libSystem / the Win32 API.

### `target-cpu`, and why it does not change the answer

`.cargo/config.toml` pins `target-cpu=native` so *host* development builds are fast on the
machine you are on. That is wrong for anything shipped, so `build_all.sh` overrides it:

| Target | `RUSTFLAGS` |
|---|---|
| linux-x86_64, windows-x86_64 | `-C target-cpu=x86-64-v3` (AVX2/FMA — Haswell 2013+, Zen 1+) |
| universal2-apple-darwin | `" "` (a single space = *no* rustflags; `x86-64-v3` is invalid for arm64) |

The choice is a pure speed knob. Rust never contracts `a*b + c` into an FMA on its own and
never reassociates a float reduction, so instruction selection cannot change a result; the
few deliberate `mul_add` calls are correctly rounded either way, and the RFdiffusion2 port's
numerics are accumulate-in-f64-then-round, which is order-independent by construction.
Measured on the RFdiffusion2 demo design, all three of `native`, `x86-64-v3` and `x86-64-v2`
produce the byte-identical output file (sha256 `ec256c46…`) — but `x86-64-v2` takes 173.5 s
where `v3` takes 91.2 s and `native` 92.1 s. The SSE-only baseline costs a 1.9× slowdown on
this GEMM-bound workload for portability to pre-2013 CPUs that could not run ESMFold in
useful time anyway.

If you need a build for such a CPU, set `PORTABLE="-C target-cpu=x86-64-v2"` in
`build_all.sh`.

### Sizes

The app is dominated by the 27 MB of embedded ProteinMPNN checkpoints; the macOS universal
binary carries two architectures, so it is roughly twice the size.

## Disk and memory while building

The three-target build writes ~4 GB into `target/`. Actually running the models additionally
needs their weights (8.4 GB for ESMFold1, 1.34 GB for RFdiffusion2). Budget ~15 GB free
to do both in one pass, or run the checks and then `cargo clean`.
