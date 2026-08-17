# Code structure & logic — RFdiffusion2

The crate is `rfd2/`, built bottom-up: `tensor` → `ops` → `weights`/`pth` → `rng` → the model
modules → the CLI. Nothing above a rung was written until the rung below was green; the
numerical target is in [`BITEXACT.md`](BITEXACT.md), the reconnaissance behind each design
decision in [`RECON.md`](RECON.md), and the measured result in
[`../results/README.md`](../results/README.md).

`tensor`, `ops`, `pth`, `weights`, `parity` and `rng::torch` are carried over from the
ProteinMPNN port, where they were already validated against PyTorch fp32. `rng::numpy` and
`rng::pyrandom` are new, because RFdiffusion2 draws from **three** generators rather than one.

## 1. End-to-end data flow

```
  PDB + ligand names + contig string
        │
        ▼
  pdb.rs        parse ATOM/HETATM
  ligand.rs     topology, loaded from a per-PDB sidecar (never perceived)
  contig.rs     "10,A106-106,10" -> which rows are motif, which are designed
  indep.rs      make_indep: the structure every later stage transforms
        │
        ▼
  sample_init.rs   seed the three RNGs, build the starting noise
        │
        ▼
  ┌── sampler.rs ──────────────────────────── T steps ──────────────┐
  │  prepro.rs / featurize.rs   Indep + timestep -> Rfi             │
  │  model/  (rf, iterblock, attention, se3, str2str, track, ...)   │
  │  score.rs                   trunk quaternion updates -> frames  │
  │  noiser.rs                  the flow-matching interpolant       │
  └─────────────────────────────────────────────────────────────────┘
        │
        ▼
  torsions.rs   psi -> the backbone carbonyl O
  output.rs     save_outputs: ATOM / HETATM / CONECT, byte-for-byte
        │
        ▼
  design_<i>-atomized-bb-False.pdb
```

`design.rs` is the top of that chain — `run_inference.py:main` for one design — and
`bin/rfd2.rs` is the CLI wrapper around it.

## 2. Module-by-module

### Foundation

- **`tensor.rs`** — the dense fp32 tensor type and its views.
- **`ops/`** — `matmul`, `reduce`, `elem`, `activation`, and `acc.rs`, the double-double
  accumulator behind the `exact` feature.
- **`pth.rs` / `weights.rs`** — read the official `RFD_173.pt` (ZIP + pickle) or a safetensors
  export, exposing EMA and final parameter sets.
- **`parity.rs`** — the comparison helpers the fixture tests use.

### RNG (`rng/`)

Three generators, because the reference draws from three: `rng::torch` (the Mersenne
Twister behind `torch.randn`/`rand`), `rng::numpy` (`numpy.random.normal`), and
`rng::pyrandom` (CPython's `random`). `dropout.rs` matters here — the model runs **with dropout
live at inference**, ~2.64 M draws per forward, so the draw *order* is part of the answer.

### Structure preparation

- **`pdb.rs`** — parsing and writing; `output.rs` owns the exact output layout.
- **`ligand.rs`** — topology from a sidecar. Deliberately a loader, not a port: upstream
  perceives bonds through OpenBabel, whose tie-breaking is not reproducible from Rust.
- **`contig.rs`**, **`indep.rs`**, **`insert.rs`** — the contig map and the `Indep` structure.
- **`atom_frames.rs`**, **`atom_frames_priority.rs`**, **`chiral.rs`**, **`chemical.rs`**,
  **`chemical_gen.rs`**, **`lj.rs`** — the chemical database and per-atom frames.

### The network (`model/`)

- **`rf.rs`** — the RoseTTAFold trunk; **`iterblock.rs`** the 36 repeated blocks.
- **`attention.rs`**, **`track.rs`**, **`t2d.rs`**, **`embeddings.rs`** — attention, the
  1D/2D/3D tracks, and template/timestep embeddings.
- **`se3.rs`** — the SE(3)-equivariant layers and their Clebsch-Gordan bases (compiled in).
- **`str2str.rs`**, **`xyzconv.rs`**, **`xyzconv_bwd.rs`** — frame updates and the coordinate
  conversion, including the backward pass the sampler needs.
- **`openfold.rs`**, **`geom.rs`**, **`nn.rs`** — the AF2 frame algebra, geometry helpers and
  the small nn primitives.

### Sampling

- **`noiser.rs`** — `NormalizingFlow` and the `se3_flow_matching` interpolant.
- **`sample_init.rs`**, **`sampler.rs`** — the starting state and `sample_step`. The demo runs
  `NRBStyleSelfCond`, not `FlowMatching`.
- **`score.rs`** — the layer between the network and the sampler.
- **`torsions.rs`** — the ψ torsion, which places the backbone carbonyl O. This is where the
  benchmark's only residuals live; see [`../results/README.md`](../results/README.md).

## 3. Numerics: what is bit-exact and what is not

Full argument in [`BITEXACT.md`](BITEXACT.md). In short: every reduction accumulates in f64 and
rounds once to fp32, which makes the port's answer independent of instruction selection and, in
the `exact` feature, provably the correctly-rounded one. Bit-exactness is claimed against a
**pinned** PyTorch reference (`python/pinned.py`), not against stock PyTorch — the conditions
are in [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md), and they are all mandatory.

## 4. Reference harness (`python/`)

Imports the **unmodified** upstream modules, so fixtures cannot drift from the published model.
`run_reference.py` runs it, `pinned.py` patches the ~100 entry points that make it
deterministic, and the `gen_*.py` / `dump_*.py` scripts write the safetensors fixtures the Rust
tests read.

## 5. Tests (`rfd2/tests/`)

Fixture-driven, one per rung, each skipping cleanly when its fixture is absent. They resolve
fixtures as `{CARGO_MANIFEST_DIR}/../fixtures`, which is why this model keeps its own subtree
rather than sharing one flat `fixtures/` directory with the other two.

**A fixture older than the last change to `python/pinned.py` or to the noiser silently measures
the wrong target, and the tests still pass.** Check mtimes before believing a rung.
