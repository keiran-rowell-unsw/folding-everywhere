# Deployment — what ships, and how to run it

Folding Everywhere v2 ships **one app for all three models** plus **one CLI per model**,
prebuilt for the three desktop platforms in [`../../dist/`](../../dist/). Nothing is
installed, no Python, no GPU, no C dependencies.

| Platform | App | ESMFold1 CLI | ESMFold2 CLI |
|---|---|---|---|
| Windows x86-64 | `gui.exe` | `fold.exe` | `fold_standalone.exe` |
| macOS universal (arm64 + Intel) | `gui` | `fold` | `fold_standalone` |
| Linux x86-64 | `gui` | `fold` | `fold_standalone` |

The app is documented in [`../../docs/GUI.md`](../../docs/GUI.md); this file covers the
command-line binaries and what both need from the machine.

## Running

```bash
# ESMFold1 — sequence in, PDB out
fold --seq MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG \
     -o ubiquitin.pdb
fold --fasta proteins.fasta -o out.pdb        # or a whole FASTA

# ESMFold2 — seeded; same seed gives the same structure.
# Positional: <SEQUENCE> [seed] [out.npy] [num_loops] [num_sampling_steps]
# Writes out.npy and the matching out.pdb, and prints a metrics JSON line.
# loops/steps default to 3/14 (the fast bit-exact benchmark setting); pass 20 68
# for the official release depth.
fold_standalone MQIFVKT... 0 out.npy 20 68
```

On macOS the binaries are unsigned, so clear the Gatekeeper quarantine once:

```bash
xattr -dr com.apple.quarantine gui fold fold_standalone
chmod +x gui fold fold_standalone
```

## Weights

The residue constants are baked into the binaries with `include_bytes!`, but the **model
weights are not** — they are 8.4 GB (ESMFold1) and ~30 GB (ESMFold2), far too large to embed.

The app downloads them automatically on first use. The CLIs take a path, and otherwise fall
back to the Hugging Face cache:

```
fold             --weights PATH | $ESMFOLD_WEIGHTS | the Hugging Face cache
fold_standalone  the Hugging Face cache for biohub/ESMC-6B and biohub/ESMFold2
```

Once a model has been downloaded, the CLI reuses the same file — nothing is duplicated.

## End-user machine requirements

| | |
|---|---|
| OS | Windows 10+, macOS 11+, any current Linux |
| CPU | x86-64 with AVX2 (any PC from ~2013 on), or Apple Silicon |
| RAM | ~10 GB for ESMFold1, ~25 GB for ESMFold2 |
| Disk | ~9 GB (ESMFold1) or ~30 GB (ESMFold2) of cached weights |
| Network | first run only, to fetch weights (`curl`, which all three OSes ship) |

## Building the distributables

```bash
./build_all.sh            # from the repo root: all three platforms, into dist/
./build_all.sh --gui-only # just the app
```

Cross-compiling needs [`cargo-zigbuild`](https://github.com/rust-cross/cargo-zigbuild) and
zig 0.11+; the full prerequisites are in [`../../docs/BUILD.md`](../../docs/BUILD.md).

### Why distribution builds override `target-cpu`

The workspace's `.cargo/config.toml` pins `target-cpu=native` so *development* builds are fast
on the build machine. That is wrong for anything shipped, so `build_all.sh` overrides it with a
portable baseline, `x86-64-v3` (AVX2/FMA — Haswell 2013+, Zen 1+).

This costs nothing in correctness: Rust never contracts `a*b+c` into an FMA on its own, and the
port's numerics are accumulate-in-f64-then-round, so results are independent of instruction
selection. The macOS universal2 build passes no `target-cpu` at all, because `x86-64-v3` is not
a valid value for arm64.
