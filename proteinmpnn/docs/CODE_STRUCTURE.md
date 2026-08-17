# Code structure & logic — ProteinMPNN

A pure-Rust, fp32 reimplementation of **ProteinMPNN** (Dauparas et al., *Science*
2022). Input: a protein backbone (PDB). Output: designed amino-acid sequences and
their scores — the same ones `protein_mpnn_run.py` produces at the same seed.

Everything is deterministic and validated layer-by-layer against PyTorch fp32.

---

## 1. End-to-end data flow

```
 backbone.pdb
   │  pdb.rs            (ATOM records -> per-chain N/CA/C/O + 1-letter sequence)
   ▼
 featurize.rs           (chain ordering, residue_idx with +100 chain offsets,
   │                     mask, chain_M, native sequence indices S)
   ▼
 features.rs            virtual Cb -> Ca-Ca distances -> top-k (k=48) graph
   │                    25 atom-pair RBF blocks x 16 bins        = 400
   │                    relative-position one-hot(66) -> Linear   =  16
   │                    concat 416 -> Linear(416,128) -> LayerNorm
   ▼  E [L,K,128],  E_idx [L,K]
 W_e -> h_E                                                    (model.rs)
   │
   ▼
 encoder  layers.rs     3 x EncLayer:
   │                      node msg  = MLP(cat[h_V, h_E, h_V[nbr]]) -> mask -> sum/30
   │                      -> +norm1 -> dense FFN -> +norm2 -> mask
   │                      edge msg  = MLP(cat[h_V, h_E, h_V[nbr]]) -> +norm3
   ▼  h_V [L,128], h_E [L,K,128]        <- depends ONLY on the backbone
   │
   ├─ forward()  teacher-forced, all positions at once -> log-probs -> score
   │
   └─ sample()   autoregressive over a random decoding order:
                   for each position t, for each of 3 DecLayers:
                     h_ESV = mask_bw * cat[h_V, h_E, h_S[nbr]] + encoder-only part
                   -> W_out -> softmax(f64) -> multinomial -> residue
   ▼
 designed sequence + score                                       (bin/mpnn.rs)
```

`ProteinMpnn::encode()` is the shared prefix and is computed **once per
structure**; the reference recomputes it inside both `sample()` and `forward()`,
which is most of its work when designing many sequences for one backbone.

---

## 2. Module-by-module

### Foundation
- **`tensor.rs`** — `Tensor { data: Vec<f32>, shape: Vec<usize> }`, always
  row-major and contiguous. `permute` materializes a fresh buffer so no op
  depends on stride tricks; every reduction lives in `ops` with a pinned
  accumulation order. Concatenation helpers use `Vec::with_capacity` + `extend`
  rather than zero-filling, because the decoder's `[L,K,512]` buffers are large
  enough that the extra page-fault pass shows up in the profile.
- **`weights.rs`** — serves tensors by name as fp32 from either an mmap'd file
  or a `&'static [u8]` baked into the binary (`Backing`). Auto-detects PyTorch
  `.pt` (ZIP+pickle) vs safetensors from the magic bytes.
- **`pth.rs`** — minimal ZIP64 + pickle reader for `torch.save` files. Walks the
  nested `{'num_edges', 'noise_level', 'model_state_dict'}` dict and records
  every (str -> tensor) pair, so state-dict entries come out under plain names.
- **`embedded.rs`** — the four published vanilla checkpoints via
  `include_bytes!`. At ~6.7 MB each they fit comfortably in the executable, so
  there is no weights download and no companion data file.
- **`parity.rs`** — comparison primitives for the tests: `max_abs`, `max_rel`,
  `max_ulp` (bit-level distance), cosine, and the fraction of bit-identical
  values.

### Core ops (`ops/`)
- **`matmul.rs`** — `linear(x, w, b)` = PyTorch `F.linear` (`w` is `[out,in]`).
  The kernel is a fixed 8-lane dot (`dot8`) with a pinned reduction order;
  `dot8x4` runs four of them over one pass of `x` for instruction-level
  parallelism while keeping each output element's order *identical*. Rayon
  parallelizes over output rows only, so results never depend on thread count.
  `linear_f64` (f64 accumulation) is a diagnostic that separates rounding noise
  from real bugs.
- **`reduce.rs`** — `layer_norm` (biased variance, eps inside sqrt, matching
  ATen), `softmax_last`, `log_softmax_last`.
- **`activation.rs`** — erf-GELU (`x*0.5*(1+erf(x/√2))`), the exact form
  `torch.nn.GELU()` uses by default — not the tanh approximation.

### Model
- **`pdb.rs`** — a faithful port of `parse_PDB_biounits`, including its quirks:
  MSE→MET rewriting, `resSeq - 1` keying over a *dense* range (so numbering gaps
  become masked residues), insertion codes ordered by code string, first-atom-
  wins, and f64 parsing narrowed to f32 only at the end.
- **`featurize.rs`** — the single-protein path through `tied_featurize`: designed
  chains first (sorted) then fixed chains, `residue_idx = 100*(c-1) + global`,
  `chain_encoding`, and the mask marking residues with all four backbone atoms.
- **`features.rs`** — `ProteinFeatures`. Virtual Cb, the masked kNN graph, the 25
  ordered atom-pair RBF blocks, the relative-position encoding, and the
  `Linear(416,128) → LayerNorm` projection. `edge_input()` exposes the 416-wide
  pre-projection tensor so tests can check the geometry separately from the GEMM.
- **`layers.rs`** — `EncLayer`, `DecLayer`, `PositionWiseFeedForward`, and the
  gather/concat plumbing (`gather_nodes`, `cat_neighbors_nodes`,
  `cat_self_edge_neighbor`). Dropout is identity at inference and omitted.
- **`model.rs`** — `encode` / `forward` / `sample`, the decoding-order masks, the
  score, and `torch_init_draws` (see below).

### RNG (`rng.rs`)
The piece that makes `--seed N` mean the same thing in Rust as in PyTorch. It
reimplements, bit-exactly:

- `at::mt19937` (the 32-bit-state MT19937 variant in `MT19937RNGEngine.h`);
- `uniform_real_distribution<float>` / `<double>`;
- `torch.randn` on a contiguous fp32 tensor — the `normal_fill` path, which
  fills the whole buffer with uniforms first and then Box-Muller-transforms
  pairs `(j, j+8)` sixteen at a time, **redrawing the last 16 values** when the
  size is not a multiple of 16;
- the Box-Muller transcendentals themselves, which are the Cephes fp32
  polynomials from `ATen/native/cpu/avx_mathfun.h` (`log256_ps`, `sincos256_ps`)
  — *not* libm — compiled by PyTorch with FMA contraction enabled;
- `exponential_(1)` = `-log1p(-uniform_double)`;
- `torch.multinomial(p, 1)`, which does not walk a CDF but computes
  `argmax(p / Exp(1))`.

Two details were load-bearing and are pinned by tests: using `mul_add` in exactly
the places GCC contracts (it fuses the *first* operand of an add, so
`y*z + e*q1` becomes `fma(y, z, fl(e*q1))`, not `fma(e, q1, fl(y*z))`), and
carrying the full-precision Cephes constants.

### `torch_init_draws` — the surprising part of seeding
`protein_mpnn_run.py` calls `torch.manual_seed(seed)` and *then* constructs
`ProteinMPNN`. Construction initialises every parameter — `nn.Linear`'s
kaiming-uniform weight and uniform bias, `nn.Embedding`'s `normal_`, and the
explicit `xavier_uniform_` loop in `__init__` — before `load_state_dict`
overwrites all of it. The values are discarded but the generator has moved: for
`v_48_020`, by exactly **3,305,317** draws. Reproducing `--seed N` therefore
means advancing the stream by the same amount. `torch_init_draws` derives that
count from the checkpoint's own tensor list, so it stays correct across model
variants instead of being a magic constant.

### Sampling runs in float64
`bias_AAs_np = np.zeros(21)` in the reference is a **float64** array, which
promotes the whole `logits - omit*1e8 + bias/T` expression, the softmax, *and*
the exponential draws inside `multinomial` to double precision. That is
incidental to the reference's design but it determines the output, so
`model.rs` reproduces it exactly.

---

## 3. Numerics: what is bit-exact and what is not

| Quantity | Agreement with PyTorch |
|---|---|
| featurization (X, S, mask, residue_idx, chain encoding) | **bit-identical** |
| virtual Cb | **bit-identical** |
| kNN neighbour indices `E_idx` | **integer-identical** |
| decoding order, attention masks | **integer-identical** |
| `randn`, `exponential_`, `multinomial` draws | **bit-identical** |
| **designed sequences** | **identical, residue for residue** |
| distances / RBF / activations / logits | fp32 round-off (~1e-5), cosine 1.0 |

Literal bit-exactness of the *continuous* values is not achievable, for two
reasons that are properties of PyTorch rather than of this port:

1. **PyTorch's fp32 `sqrt` is not correctly rounded.** It returns 1 ULP below
   the IEEE result for ~0.6% of inputs (measured: 1116 of 200000 random values,
   always low, never high). Every Ca-Ca distance flows through it.
2. **fp32 GEMM accumulation order.** PyTorch dispatches to MKL/oneDNN, whose
   blocked-SIMD summation order differs from any scalar fold. Matching it would
   mean reimplementing MKL.

Both are bounded, non-accumulating disagreements — cosine similarity stays at
1.0 to twelve decimals through the whole network, and every discrete decision
(which neighbour, which decoding step, which residue) is identical. The
benchmark measures this directly: see `results/`.

---

## 4. Reference harness (`python/`)

Imports the **unmodified** upstream `protein_mpnn_utils`, so fixtures cannot
drift from the published model.

| script | purpose |
|---|---|
| `common.py` | determinism pinning (single thread, fp32), paths, fixture writer |
| `gen_op_fixtures.py` | per-op fixtures at the widths the model actually uses |
| `gen_rng_fixtures.py` | `randn` / `exponential_` / `multinomial` / decoding order |
| `gen_weight_fixture.py` | state dict → safetensors, to validate the `.pt` reader |
| `ref_dump.py` | every intermediate of a full run (features → encoder → decoder → sampling) |
| `select_proteins.py` | uniform random draw of benchmark structures from the PDB |
| `run_benchmark.py` | drives both CLIs, records timing/memory/agreement |
| `compare_logprobs.py` | full-precision log-probability comparison |
| `make_figures.py` | the five benchmark figures |

## 5. Tests (`mpnn/tests/`)

| file | what it pins |
|---|---|
| `parity_ops.rs` | linear at 6 widths, LayerNorm, GELU, softmax, log-softmax, embedding, neighbour-sum |
| `parity_rng.rs` | 15,185 `randn` values over 70 (seed, size) cases; exponentials; 1,200 sequential multinomial draws; decoding orders — all bit-exact |
| `pt_loader.rs` | all 118 tensors / 1,660,485 parameters bit-identical to `torch.load` |
| `parity_model.rs` | featurization → geometry → graph → edges → 3 encoder layers → 3 decoder layers → logits → log-probs → sampled sequence |

`parity_model.rs` also writes `results/stage_parity/*.json`, so the benchmark's
stage figure is generated from the same numbers the tests assert on.

Run everything with `cargo test --release`.
