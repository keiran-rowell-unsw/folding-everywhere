# Code structure & logic — esmfold-fp32

A pure-Rust, fp32 reimplementation of **ESMFold v1**. Input: an amino-acid
sequence. Output: 3D all-atom coordinates (PDB) + pLDDT / pTM / PAE. Everything is
deterministic and validated layer-by-layer against PyTorch fp32 (cosine = 1.0 at
every stage; end-to-end ~0.0001 Å RMSD).

---

## 1. End-to-end data flow

```
 sequence "MQIF..."
   │  tokenizer.rs            (chars -> ESM 33-token ids, +<cls>/<eos>)
   ▼
 ESM-2 3B backbone  esm2.rs   (36 pre-LN transformer layers, rotary attention)
   │  -> 37 hidden states  [L+2, 2560]
   ▼
 LM→trunk glue  pipeline.rs   (softmax(esm_s_combine)·states -> MLP -> +aa embedding)
   │  -> s_s_0 [L,1024],  s_z_0 = 0 [L,L,128]
   ▼
 Folding trunk  trunk.rs      (×4 recycles):
   │     relative-position embed -> 48 × TriangularSelfAttentionBlock
   │     each block: pair→seq bias, gated seq attention, seq MLP,
   │                 seq→pair, tri-mul(out/in), tri-attention(start/end), pair MLP
   ▼  s_s [L,1024], s_z [L,L,128]
 Structure module  structure.rs (8 shared iterations):
   │     IPA -> backbone-frame update (quaternions) -> angle resnet ->
   │     torsion→frames -> frames+literature→atom14 coords
   │     (recycling distogram fed back to the next trunk recycle)
   ▼  atom14 [L,14,3]
 Heads  heads.rs               (distogram, pLDDT, pTM, PAE) + atom14→atom37
   ▼
 PDB  pdb.rs                   (ATOM records, pLDDT in B-factor)
```

The whole thing is orchestrated by `pipeline::fold()` and driven by `bin/fold.rs`.

---

## 2. Module-by-module

### Foundation
- **`tensor.rs`** — `Tensor { data: Vec<f32>, shape: Vec<usize> }`. Always
  row-major & contiguous; `permute`/`t` materialize a fresh buffer so no op depends
  on stride tricks. This is deliberate: it lets every reduction live in one place
  with a pinned accumulation order (key to matching PyTorch numerically).
- **`weights.rs`** — memory-maps the `model.safetensors` file, parses the header
  once, and serves tensors **by name** as fp32 (`F16`→`F32` upcast is lossless).
  Because it's mmap'd and we fetch one layer at a time, peak RAM is a couple of GB
  even though the model is ~3.5 B params.
- **`parity.rs`** — comparison primitives used by the tests: `max_abs`, `max_rel`,
  `max_ulp` (bit-level distance), `cosine`.

### Core ops (`ops/`)
- **`matmul.rs`** — `linear(x, w, b)` = PyTorch `F.linear` (`w` is `[out,in]`). The
  hot kernel is a vectorizable 8-lane dot (`dot8`) with a fixed reduction order;
  rayon parallelizes over **output rows only**, so results are identical regardless
  of thread count. A `linear_f64` (f64 accumulation) exists as a diagnostic to tell
  rounding noise from real bugs.
- **`reduce.rs`** — `layer_norm` (biased variance, eps inside sqrt — matches ATen)
  and `softmax_last` (max-subtracted).
- **`activation.rs`** — `gelu_erf` (`x·0.5·(1+erf(x/√2))` via `libm::erff`, the exact
  ESM variant — not the tanh approx), `relu`, `sigmoid`, `softplus`.
- **`rotary.rs`** — rotary position embeddings: `inv_freq` table, `cos`/`sin`,
  `rotate_half` apply. Tables computed in f64→f32 to minimize transcendental error.

### Language model
- **`tokenizer.rs`** — the fixed ESM-2 33-token alphabet; `tokenize` adds `<cls>`/`<eos>`.
- **`esm2.rs`** — ESM-2 3B: token embedding ×0.88 (token-dropout at inference),
  then 36 pre-LN blocks. Each block = LayerNorm → multi-head attention (q pre-scaled
  by `head^-0.5` **before** rotary, eager softmax) → residual → LayerNorm → FFN
  (Linear→erf-GELU→Linear) → residual. Returns the **37-state stack**
  `[emb, L1…L35, LN_after(L36)]` that ESMFold consumes.

### Folding trunk
- **`trunk.rs`** — `relative_position` embedding and one `block` (the 9 sub-steps in
  exact source order) plus `trunk_iter` (relpos + 48 blocks). All the Evoformer-style
  pieces are here: gated sequence attention with a pair-derived bias, the outer-product
  `sequence_to_pair`, the two triangular multiplicative updates
  (`out`: Σ_k a[i,k]·b[j,k]; `in`: Σ_k a[k,i]·b[k,j]), the two triangular attentions
  (start/end), and the residue MLPs.

### Structure module
- **`rigid.rs`** — frame math, all hand-written algebra (no eigendecomposition on the
  inference path): `quat_to_rot`, `rot_vec_mul`, `rot_matmul`, `compose_q_update`
  (quaternion backbone update + L2 normalize), and a 3×3 `Frame` (compose/apply/from_4x4)
  for the side-chain frames.
- **`constants.rs`** — residue constants (rigid-group default frames, atom14↔atom37
  maps, atom masks, literature atom positions). Embedded in the binary via
  `include_bytes!` (`Constants::embedded()`), so no external constants file is needed.
- **`structure.rs`** — the 8-iteration loop: **Invariant Point Attention** (scalar +
  point + pair attention terms with the AF2 scale constants √(1/48), √(1/3), √(1/54)
  and softplus head weights), backbone-frame update, `angle_resnet` (7 torsion angles),
  `torsion_to_frames`, and `frames_to_atom14` (places literature atom positions into
  the predicted side-chain frames).

### Heads & output
- **`heads.rs`** — `distogram` (symmetrized logits), `plddt` (categorical mean over
  50 bins), `compute_ptm`/`compute_pae` (d0 formula, 64 bins), and `atom14_to_atom37`.
- **`pdb.rs`** — writes standard PDB ATOM records, pLDDT (×100) in the B-factor column;
  `mean_plddt` averages pLDDT over existing atoms only (0–100), matching upstream ESMFold's
  `output["mean_plddt"]`.

### Orchestration
- **`pipeline.rs`** — `lm_to_trunk` (the LM→trunk glue), `distogram_bins` (CB-distogram
  for recycling), and `fold()` which runs the whole pipeline including the 4-recycle
  loop (each recycle: LayerNorm the recycled s/z, add the recycle distogram embedding,
  run the trunk, run the structure module, recompute the recycle distogram).
- **`bin/fold.rs`** — the CLI: parses `--seq`/`--fasta`, finds weights, folds, writes
  PDB (and optional `--dump` of raw atom37 for benchmarking).

---

## 3. Numerics / parity strategy

- **fp32 everywhere.** ESM weights are stored F16 in the checkpoint and upcast
  losslessly to F32; folding weights are already F32.
- **Deterministic ops.** One reduction order, thread-count-independent. This is why
  the port reproduces PyTorch fp32 to cosine = 1.0 at every layer.
- **Why not literal bit-for-bit:** PyTorch's fp32 GEMM uses MKL/oneDNN blocked-SIMD
  accumulation (a different summation order) and libm transcendentals differ in the
  last bit. These keep agreement at fp32 epsilon (final coords within ~1e-3 Å);
  discretized outputs (pTM, atom37 gather) are already bit-exact.

## 4. Reference harness (`python/`)

Generates the ground-truth fixtures the Rust tests check against, all in pinned,
single-thread, deterministic fp32: `gen_op_fixtures.py` (per-op), `ref_lm.py`
(ESM-2 per layer), `ref_trunk.py` (trunk + structure + heads), `export_constants.py`,
`ref_fold.py` + `benchmark.py` (full reference + timing/memory). The PyTorch fp32
reference is run **decomposed** (ESM ~11 GB, then the folding head ~2.8 GB) so it fits
in 15 GB; the Rust binary does the whole thing in one process via weight streaming.

## 5. Tests (`tests/`)

`parity_ops` (ops), `parity_lm` (ESM-2), `parity_trunk` (48 blocks), `parity_structure`
(IPA + all-atom), `parity_heads` (LM→trunk glue + heads), `parity_e2e` (full fold).
All 16 pass. Run with `cargo test --release`.
