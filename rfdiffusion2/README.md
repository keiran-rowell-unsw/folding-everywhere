# RFdiffusion2 — motif → designed backbone

**A pure-Rust, fp32, CPU re-implementation of RFdiffusion2** (Ahern et al., bioRxiv 2025)
— atom-level enzyme active-site scaffolding.

**[Project page](https://github.com/lingxusb/folding-everywhere)** · **[Author](https://lingxusb.github.io)**

> This is the **RFdiffusion2** part of *Folding Everywhere v2*. It is one of the three models
> in the single app the repo ships — see the [top-level README](../README.md) and
> **[docs/GUI.md](../docs/GUI.md)**.
>
> ### **[Download the app](../dist/)** — `gui.exe` (Windows) · `gui` (macOS universal / Linux)
> Double-click it, open the **RFdiffusion2** tab, click *Load example*, then *Design
> backbone*. The first design downloads the official checkpoint (RFD_173.pt, 1.34 GB, one
> time) from the Institute for Protein Design.

## Quick start

### App (recommended)

Download it from [`../dist/`](../dist/), run it, open the **RFdiffusion2** tab, click *Load
example* and then *Design backbone*. The example is 1LDM lactate dehydrogenase with NAD and
OXM, contig `10,A106-106,10`.

### Command line

`./build_all.sh` ships an `rfd2` CLI for each platform alongside the app, and
`cargo build --release` builds it from source:

```bash
rfd2 --input-pdb motif.pdb --contigs '10,A106-106,10' --ligand NAD,OXM \
     --ligand-topology M0584_1ldm.safetensors \
     --weights ~/.rfdiffusion2/RFD_173.pt --igso3 ../gui/data/igso3.safetensors \
     --T 100 --output-prefix out/design
```

The contig is a comma list: a bare number is that many DESIGNED residues, `A106-106` keeps
motif residue 106 of chain A. Same seed → identical design, byte for byte. `rfd2 --help`
prints the full option list, and [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md) covers what ships
and what each input is.

## The one limitation, stated plainly

Ligand bond orders and aromaticity are **perceived from 3D coordinates by OpenBabel** inside
the reference pipeline — 3 of 4 demo inputs carry no CONECT records at all — so they are an
**input** to this port, not something it computes. Ten ligand sets (56 ligands) ship inside
the app and are matched to your file **by atom name**, so they work on any structure:

```
M0584_1ldm  NAD,OXM        M0636_1uaq  ZN,DUC       M0710_1ra0  FE,FPY
M0054_1qfe  DHS            M0097_1ctt  ZN,DHZ       M0375_4ts9  FMC,PO4
M0179_1q3s  MG,ADP         M0365_1pfk  FBP,MG,ADP   M0315_1ey3  DAK
M0093_1dqa  NAP,COA
```

A PDB whose ligands are not in that list needs one run of
[`python/gen_ligand_bonds.py`](python/gen_ligand_bonds.py) first. The program refuses rather
than guessing, so it will never silently return a wrong answer.

## How long a design takes

CPU only; RFdiffusion2 is normally run on GPU. Measured with the shipped binary on 4 cores,
for the bundled example (1LDM lactate dehydrogenase, contig `10,A106-106,10` — 21 designed
residues + 50 ligand atoms = 71 tokens):

| steps `T` | time |
|---|---|
| 2 (quick sanity check) | ~1.5 min |
| 20 | ~15 min |
| 100 (the GUI default, the real setting) | ~70 min |

Cost is ~42 s per denoising step at this size plus ~9 s fixed, and scales roughly as
L^1.8: the full production contig (180 residues, L = 230) is ~13 min per step.

## Does it really reproduce RFdiffusion2?

**Bit-exact end-to-end is the target, and the route to it is measured.** It is
not reached the way one might first assume, so read `docs/BITEXACT.md` — this is
the short version.

Stock PyTorch's fp32 GEMM **cannot** be reproduced by choosing a reduction
order. Measured at RFdiffusion2's real shapes (`python/probe_gemm_order.py`):
the best candidate agrees on 99.1 % of outputs at K = 32–65 and ~10 % at
K >= 192, never 100 %. Adding FMA makes it worse, which also rules out fusion as
the explanation. Matching MKL would mean reimplementing MKL.

The route that does work is to make the reduction order **irrelevant**: both
sides accumulate in f64 and round to f32 exactly once. The f64 error (~1e-16
relative) sits ~9 orders of magnitude below an f32 ULP, so the f32 result is the
correctly-rounded one and is independent of blocking, SIMD width and thread
count. Measured (`python/probe_f64_pinning.py`): four deliberately different f64
summation orders over 299 200 values -> **0 disagreements**.

Applied to this port, rung 1 becomes exactly 0:

```
linear     (f64-pinned): 45 cases, 1640040 values BIT-IDENTICAL
layer_norm (f64-pinned):            106176 values BIT-IDENTICAL
softmax    (f64-pinned):            130666 values BIT-IDENTICAL
```

So there are two modes, and the distinction matters:

| | **stock mode** | **pinned mode (bit-exact)** |
|---|---|---|
| Reference | unmodified PyTorch + MKL | PyTorch with `python/pinned.py` (100 patched entry points) |
| Agreement | ~1e-6, cosine 1.0, discrete decisions identical | **bit-identical** |
| Cost | fast | ~2x slower (f64 accumulate) |

**The one caveat, stated plainly:** pinned mode pins the *reference* too. A
pinned run is RFdiffusion2 with identical weights, architecture, algorithm, RNG
and discrete decisions, but with intermediates rounded once at f64 rather than
accumulated in fp32 in MKL's order — it differs from a stock-MKL run by fp32
round-off (measured max |D| 1.4e-6 at a single linear layer). It is in fact the
*more* accurate of the two. What is not claimed is bit-identity with an
unmodified `run_inference.py` on stock PyTorch+MKL; §1 of `docs/BITEXACT.md`
shows no non-MKL implementation can achieve that.

Discrete outputs — every RNG draw, contig sample, decoding order, top-k index,
atomization choice, argmax token — are bit-exact/integer-identical in **both**
modes.


Full argument in [`docs/BITEXACT.md`](docs/BITEXACT.md); the environment conditions a pinned
reference run needs are in [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md), and they are
all mandatory.

## Benchmarking summary

The whole inference path is ported — chemical database, featurisation, contig sampling,
IGSO(3) noiser, the 36-block RoseTTAFold trunk, SE(3) equivariant refinement, the diffusion
sampler and PDB output. Measured over a **29-case benchmark** (10 proteins, L = 30–230
tokens, T = 2/10/100, self-conditioning, variable-length contigs, multiple designs) against
a pinned PyTorch reference on the same 4-core CPU — see
[`results/README.md`](results/README.md):

```
29 cases
  byte-identical output file      22 / 29   (76 %)
  ligand atoms bit-exact          29 / 29   (100 %)
  CONECT bond records exact       29 / 29   (100 %)
  speedup (Rust vs pinned PyTorch)  mean 1.62x, range 1.28-2.15x
```

Not one ligand coordinate and not one bond record differed, in any case — including a
61-atom ligand, a 3-ligand system and single-atom metals (ZN, FE, MG). Every residual is in
the designed protein backbone:

| case | L | protein atoms exact | max &#124;Δ&#124; |
|---|---|---|---|
| `p01_L30` | 30 | 105/109 | **0.192 Å** |
| `du_mm4_prod` *(production config)* | 230 | 863/920 | 0.118 Å |
| `len25_L101` | 101 | 247/261 | 0.042 Å |
| `len40_L131` | 131 | 402/411 | 0.033 Å |
| `du_T100_mid` | 49 | 105/108 | 0.002 Å |
| `p09_L82` | 82 | 101/105 | 0.001 Å |
| `cfg_T10s` | 49 | 104/108 | 0.001 Å |

### The differences

The largest disagreement anywhere in the benchmark is **0.192 Å** on a single atom — and
three of the seven cases differ by 0.001–0.002 Å, i.e. **one unit in the last printed
decimal** (the PDB `%8.3f` format resolves 5e-4 Å). For scale, a C–O bond is ~1.23 Å and a
C–C bond ~1.5 Å, and experimental structures are solved at 1–2 Å resolution: the worst
residual here is roughly **one sixth of a bond length**, and the typical one is a thousandth
of it. It is also vastly below RFdiffusion2's own inference noise — the model runs with
**dropout live**, ~2.64 M draws per forward pass, so changing the seed changes the design
completely.

### Why it is always the carbonyl oxygen

The seven cases are not seven different problems. In **every one** the worst-displaced atom
is a backbone **carbonyl O**, and that residue's CA is bit-identical.

N, CA, C and CB are placed directly by the residue's rigid frame. **O is the only backbone
atom placed through the ψ torsion** — rotated about the CA–C bond after the frame is already
fixed — so it carries one extra degree of freedom the others do not, and it is the last atom
placed in the backbone-building chain. Across all seven cases the displacement is simply
`lever arm × Δψ`, with the ~1.1 Å perpendicular distance from O to the CA–C axis as the
slope (`results/figures/fig6_carbonyl_oxygen.png`).

The worst case, `p01_L30`, is therefore not "a protein with a large error": it is **one
carbonyl group rotated by 9.7°** at the N-terminus of a 21-residue design. Its CA and CB
agree to the last bit, N and C move ~0.03 Å, and 105 of its 109 protein atoms, all 9 ligand
atoms and all 8 bonds are exact.

**What is not yet established**, stated as a hypothesis rather than a finding: why Δψ reaches
9.7° when the ψ values come from an RNG verified bit-identical (generator state after the psi
draw: 32/32 identical), and a 20 mrad frame difference alone accounts for only ~1.1°. The
working hypothesis is ill-conditioned normalisation — ψ enters as a 2-vector that is
normalised, so when its magnitude happens to be small a last-bit difference swings the angle
a long way. That would also explain why the effect is sporadic rather than length-dependent.
The test — dump the ψ pair's magnitude for these residues and check it is near zero on
exactly the large-Δψ cases — has not been run.

Two further things the headline number does not say, both in
[`results/README.md`](results/README.md): byte-identity **does not survive to
production scale** (L = 230 agrees to 0.118 Å with 93.8 % of atoms bit-identical), and
agreement is **not monotone in the port's correctness** — a provably correct fix once made
T = 2 byte-identical and simultaneously broke T = 100, because the reference is not correctly
rounded either.

The rung-by-rung parity numbers underneath that result are in
[`results/README.md`](results/README.md). Reproduce the unit ladder with
`cargo test --release -p rfd2 -- --nocapture` (after `.venv/bin/python python/gen_*.py` — the
large fixtures are regenerated, not shipped).

## Repository layout

```
rfdiffusion2/              (this subtree)
├── README.md           this file
├── docs/
│   ├── CODE_STRUCTURE.md  module-by-module map of the crate
│   ├── DEPLOYMENT.md      what ships, how to run it
│   ├── RECON.md           reconnaissance: inference path, RNG, dtypes, checkpoint
│   ├── BITEXACT.md        why f64-pinning is the route to bit-exactness
│   └── REPRODUCIBILITY.md the env conditions a pinned reference run needs
├── python/             reference harness (imports unmodified upstream)
├── rfd2/               the Rust crate (+ the `rfd2` CLI) and its embedded data/
├── fixtures/           safetensors fixtures written by python/ (regenerated, not shipped)
├── bench/              the 29-case benchmark: cases, runner, comparison, figures
└── results/            measured parity + benchmark numbers and figures
```

The **app that drives this crate lives at [`../gui/`](../gui/)** and is shared with ESMFold
and ProteinMPNN; it also holds the IGSO(3) tables, the ligand-topology library and the
example structure. Prebuilt binaries are in [`../dist/`](../dist/); the workspace manifest is
[`../Cargo.toml`](../Cargo.toml).

`rfd2/tests/` reach `fixtures/` with `{CARGO_MANIFEST_DIR}/../fixtures`, which is why each
model keeps its own subtree here rather than sharing one flat `fixtures/` directory.

## Building from source

```bash
cargo build --release --bin rfd2   # the CLI
cargo build --release --bin gui    # the app, all three models
./build_all.sh                     # Linux + Windows + macOS universal, into ../dist/
```

### The upstream reference

Pinned at `RosettaCommons/RFdiffusion2` commit
`d365cbf4db3958814a9f8e4f6f94fa309dfebc2b` (2026-04-03), cloned to
`../ref_RFdiffusion2` relative to this subtree (i.e. `rfdiffusion2/ref_RFdiffusion2`). Weights: `RFD_173.pt` / `RFD_140.pt` from
`https://files.ipd.uw.edu/pub/rfdiffusion2/model_weights/` (1.34 GB each).

The Python harness in `python/` imports the **unmodified** upstream modules, so
fixtures cannot drift from the published model.


## Licence & credit

The model architecture and the trained weights are the work of the Institute for Protein
Design and co-authors; the checkpoint is downloaded at runtime from `files.ipd.uw.edu` under
its own licence and is **not** redistributed here. This is an independent reimplementation of
the inference path — no training code, no new weights.

> Ahern, W. et al. *Atom level enzyme active site scaffolding using RFdiffusion2.*
> bioRxiv (2025). doi:[10.1101/2025.04.09.648075](https://doi.org/10.1101/2025.04.09.648075)

Designs are computational hypotheses and should be validated experimentally.
