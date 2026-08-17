# Deployment — what ships, and how to run it

Folding Everywhere v2 ships **one app for all three models** plus **one CLI per model**,
prebuilt for the three desktop platforms in [`../../dist/`](../../dist/). Nothing is
installed, no Python, no GPU, no C dependencies.

| Platform | App | RFdiffusion2 CLI |
|---|---|---|
| Windows x86-64 | `gui.exe` | `rfd2.exe` |
| macOS universal (arm64 + Intel) | `gui` | `rfd2` |
| Linux x86-64 | `gui` | `rfd2` |

The chemical database, AF2 frame tables, SE(3) Clebsch-Gordan bases, IGSO(3) noise tables and
the ligand-topology library are all compiled in. Only the 1.34 GB model checkpoint is fetched
at first run.

The app is documented in [`../../docs/GUI.md`](../../docs/GUI.md); this file covers the
command-line binary.

## Running

```bash
rfd2 --input-pdb M0584_1ldm.pdb \
     --contigs '10,A106-106,10' \
     --ligand NAD,OXM \
     --ligand-topology fixtures/ligand/M0584_1ldm.safetensors \
     --weights RFD_173.pt \
     --igso3 fixtures/noiser/stages.safetensors \
     --T 100 --output-prefix out/design
```

`rfd2 --help` prints the full option list. The contig is a comma list: a number is that many
*designed* residues, `A106-106` keeps motif residue 106 of chain A. Files are written as
`<prefix>_<i>-atomized-bb-False.pdb`.

Two inputs are worth calling out:

* **`--ligand-topology`** is a per-PDB sidecar produced by `python/gen_ligand_bonds.py`. It is
  not optional and it is not interchangeable between structures — see
  [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md).
* **`--weights`** takes the official `RFD_173.pt` directly, or a safetensors export of it.

The port **refuses** configurations it has not been validated on (atomization, guideposts,
`partial_T`) with a message naming what was not measured, rather than guessing.

On macOS the binaries are unsigned, so clear the Gatekeeper quarantine once:

```bash
xattr -dr com.apple.quarantine gui rfd2 && chmod +x gui rfd2
```

## End-user machine requirements

| | |
|---|---|
| OS | Windows 10+, macOS 11+, any current Linux |
| CPU | x86-64 with AVX2 (any PC from ~2013 on), or Apple Silicon |
| RAM | ~2 GB |
| Disk | ~1.4 GB for the cached checkpoint |
| Network | first run only, to fetch the checkpoint |

Runtime is the thing to plan for, not memory: on 4 CPU cores the bundled example takes ~1.5 min
at `T = 2` and ~70 min at `T = 100`. RFdiffusion2 is normally a GPU model; this is a CPU port.

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

This costs nothing in correctness — and here that matters more than usual, since the port's
claim is bit-exactness: Rust never contracts `a*b+c` into an FMA on its own, and every reduction
accumulates in f64 before rounding, so the answer does not depend on instruction selection.
Measured on the demo design, the more conservative `x86-64-v2` is 1.9x slower for no portability
that matters. The macOS universal2 build passes no `target-cpu` at all, because `x86-64-v3` is
not a valid value for arm64.
