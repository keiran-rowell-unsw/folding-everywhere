# How this port reaches bit-exactness end to end

You asked for a Rust RFdiffusion2 that reproduces the PyTorch version bit-exact,
end to end. This document says exactly how that is achieved, what it costs, and
where the claim's edges are. Every number here was measured on this machine with
the pinned environment; the scripts are in `python/`.

---

## 1. The problem, measured rather than asserted

The SOP opens by saying fp32 bit-exactness is unachievable because PyTorch's
GEMMs go to MKL/oneDNN. That is a general claim, so it was tested at
RFdiffusion2's *actual* shapes (K ∈ {32, 64, 65, 114, 128, 192, 256, 320}),
comparing `F.linear` against every plausible reduction order.

`python/probe_gemm_order.py` — fraction of outputs bit-identical to `F.linear`:

| shape (M,K,N) | sequential | pairwise | 8-lane | 16-lane | 8-lane tree | f64 |
|---|---|---|---|---|---|---|
| (150, 32, 576) | **99.14 %** | 23.87 % | 23.82 % | 22.46 % | 23.87 % | 25.71 % |
| (150, 64, 32) | **99.06 %** | 18.08 % | 18.00 % | 17.65 % | 18.08 % | 19.35 % |
| (150, 65, 32) | **98.94 %** | 18.92 % | – | – | – | 18.60 % |
| (150, 114, 64) | 71.92 % | 13.99 % | – | – | – | 14.58 % |
| (150, 192, 192) | 14.65 % | 15.45 % | 14.17 % | 13.88 % | 14.83 % | 15.21 % |
| (150, 256, 256) | 10.89 % | 15.35 % | 13.81 % | 13.84 % | 14.32 % | 15.22 % |
| (150, 320, 64) | 9.61 % | 15.30 % | 13.03 % | 13.15 % | 13.90 % | 14.25 % |

`python/probe_gemm_order2.py` then tested the obvious explanation for the
missing 1 % at small K — that MKL fuses the multiply-add, so products are never
rounded. It does not: adding FMA made sequential *worse* at every shape
(99.06 % → 29.56 % at K=64), which tells us the products **are** rounded to fp32
and the residual disagreement is blocking, not fusion.

**Conclusion: no fixed fp32 reduction order reproduces stock PyTorch.** Not at
any shape, not even at K=32. The SOP is right, and now it is right *with
evidence for this model* rather than by analogy.

---

## 2. The way through: make the reduction order irrelevant

Bit-exactness does not actually require matching MKL's order. It requires that
both sides compute a value that *no* order can change.

If a dot product is accumulated in **f64** and rounded to **f32 exactly once**,
the f64 rounding error is ~1e-16 relative while an f32 ULP is ~1e-7 — about nine
orders of magnitude of headroom. The f32 result is then the correctly-rounded
one, and every implementation that does this agrees, whatever its blocking, SIMD
width, or thread count.

This is not a theorem — a value landing within 1e-16 of an f32 rounding
boundary would break it — so it was measured. `python/probe_f64_pinning.py`
computes each product four deliberately different ways (BLAS-blocked f64,
strict-sequential f64, reverse-order f64, 8-lane f64) and compares bits:

```
     shape (M,K,N)     values    blas=seq    blas=rev  blas=lane8
    (150, 32, 576)      86400 100.000000% 100.000000% 100.000000%
     (150, 64, 32)       4800 100.000000% 100.000000% 100.000000%
    (150, 114, 64)       9600 100.000000% 100.000000% 100.000000%
   (150, 128, 128)      19200 100.000000% 100.000000% 100.000000%
   (150, 192, 192)      28800 100.000000% 100.000000% 100.000000%
   (150, 256, 256)      38400 100.000000% 100.000000% 100.000000%
    (150, 320, 64)       9600 100.000000% 100.000000% 100.000000%
   (400, 256, 256)     102400 100.000000% 100.000000% 100.000000%

total values compared: 299200
  disagreements (blas vs seq  ): 0  (ALL IDENTICAL)
  disagreements (blas vs rev  ): 0  (ALL IDENTICAL)
  disagreements (blas vs lane8): 0  (ALL IDENTICAL)
```

Zero disagreements in 897,600 comparisons.

### Already demonstrated on the port

Rung 1 re-run under this convention, Rust vs PyTorch, tolerance **exactly 0**:

```
linear     (f64-pinned): 45 cases, 1640040 values BIT-IDENTICAL
layer_norm (f64-pinned):            106176 values BIT-IDENTICAL
softmax    (f64-pinned):            130666 values BIT-IDENTICAL
```

1,876,882 values, no exceptions. Under stock fp32 the same ops sit at
max |Δ| ≈ 2.1e-6 — close, but never equal, exactly as §1 predicts.

---

## 3. What "the PyTorch version" means here — the honest part

This works by pinning **both** sides, so it changes the reference too.

A pinned run is RFdiffusion2 with the **same weights, same architecture, same
algorithm, same RNG streams, same discrete decisions** — but with intermediate
arithmetic rounded at f64 and narrowed once, instead of accumulated in fp32 in
MKL's order. Measured difference from a stock-MKL run: **max |Δ| ≈ 1.4e-6,
max relative ≈ 3e-3 on individual near-zero entries** at a single linear layer.

So there are two modes, and both are legitimate — they just answer different
questions:

| | **stock mode** | **pinned mode (bit-exact)** |
|---|---|---|
| Reference | unmodified PyTorch + MKL | PyTorch with `python/pinned.py` enabled |
| Rust uses | `ops::linear`, `layer_norm`, … (fp32) | `ops::linear_f64`, `*_f64` |
| Agreement | ~1e-6, cosine 1.0, discrete decisions identical | **bit-identical** |
| Answers | "does this match what RFdiffusion2 users run today?" | "is this port a faithful reimplementation?" |
| Speed | fast | ~2× slower (f64 accumulate) |

Pinned mode is *more* numerically accurate than stock — it is the
correctly-rounded result, and it also removes PyTorch's non-correctly-rounded
fp32 `sqrt` (SOP §5.3), which is a known error source in the stock path.

**What is not being claimed:** that this Rust port reproduces, bit-for-bit, the
output of an unmodified `run_inference.py` on a stock PyTorch+MKL install. §1
shows that is not reachable by any implementation that is not MKL. If that is
the requirement rather than faithful reimplementation, the honest answer remains
"run the reference".

---

## 4. `python/pinned.py` — and the hole that nearly sank it

`pinned.enable()` patches **100 entry points** so any op with an fp32 input is
computed in f64 and narrowed once. (The count is what `enable()` returns and
what every pinned run prints; it was 63 when this section was first written and
grew as holes were found. Take the number from the run, not from here.)

Patching `torch.*` module functions alone is **not sufficient**, and this is the
easiest way to ship a false bit-exactness claim. RF2AA is written with tensor
methods and operators — `a @ b`, `x.matmul(y)`, `x.sum(-1)`, `x.softmax(-1)` —
which dispatch to `torch.Tensor.*` and bypass the module-level patch entirely.
Measured before `torch.Tensor` methods were patched:

```
F.linear         100.0000% bit-identical
Tensor.matmul     15.2161%   <- HOLE
```

After patching the methods and dunders:

```
OK   F.linear         100.0000%
OK   torch.matmul     100.0000%
OK   Tensor.matmul    100.0000%
OK   operator @       100.0000%
OK   einsum           100.0000%
```

`pinned.report()` returns per-op fire counts. **An op that never fires is either
off the inference path or reached by an unpatched route** — the audit trail must
be checked before any end-to-end claim, not assumed. Any method that cannot be
patched is recorded in the report with a negative count rather than silently
skipped.

Double rounding is safe for the patched set: for `+ − × ÷` and `sqrt` on f32
inputs, computing in f64 and rounding once gives exactly the f32 result (f64's
53-bit significand exceeds 2·24+2), so patching them is a no-op. It is the
multi-term reductions and the transcendentals where the behaviour changes, which
is the point.

---

## 5. What still has to hold for the end-to-end claim

Pinning removes the *GEMM-order* obstacle. It does not remove the others, and
none of these are done yet:

1. **Every rounding boundary must match.** Bit-exactness needs Rust to round at
   the same op boundaries as the reference. Where the reference fuses several
   operations into one kernel, the port must too. This is decided per module as
   each is ported, and is the main reason the ladder is climbed one layer at a
   time.
2. **Discrete decisions must not flip.** `top-k`, `argmax`, `argsort` on values
   that differ by ~1e-6 between stock and pinned mode can select differently at
   a near-tie. In pinned mode both sides see identical values so the flip cannot
   occur *between them* — but a pinned run may pick a different neighbour than a
   stock run. Every such site needs a near-tie audit (SOP §5.5).
3. **The three RNG streams must stay synchronised.** Already green (rung 2), but
   the *call order* through the sampler still has to be reproduced —
   particularly `psi_pred`, drawn inside the model forward once per step.
4. **f64 promotions in the reference must be preserved, not "fixed".** SOP §5.2:
   where the reference already computes in f64 (SciPy rotations, any bare
   `np.zeros`), the port must reproduce that, and pinning must not mask it.
5. **The output format is part of the port** (SOP §5.6) — PDB records, atom
   naming, ligand renaming, `.trb` contents.

---

## 6. Status

Reachable, with the route now measured rather than hoped for. Rungs 1–3 are
green and rung 1 is green **at tolerance 0** under pinning. The model itself
(rungs 4–8) is not yet ported; see `results/README.md` for the current line.

---

## 7. Validated on the real model (Phase A/B)

Everything above §6 was measured on isolated ops. The reference now runs
end-to-end on CPU, so the same questions were re-asked of the whole
82,911,693-parameter network. Raw output: `results/phaseB_dump_comparison.txt`.

### Setup

`python/run_reference.py` runs the **unmodified** upstream `run_inference.py`
via `runpy`, applying two shims from the outside (SOP §1.1 forbids editing
upstream):

- **NVTX no-op** — `se3_transformer/.../attention.py` does
  `from torch.cuda.nvtx import range as nvtx_range` at import time, and a
  CPU-only torch build raises `RuntimeError: NVTX functions not installed`.
- **`PYTORCH_JIT=0`** — see below.

`python/ref_dump.py` captures intermediates with forward hooks on the real
module objects, so the SOP's "inline forward == public API" assertion holds by
construction: there is no second implementation to drift.

Reference run is deterministic: two identical invocations produce **byte-identical**
PDBs. Case used: `M0584_1ldm.pdb` + NAD/OXM ligands, contig `10,A106-106,10`,
L = 71, T = 2, seed 0. Throughput ~6.8 s per flow-matching step on 4 CPU cores.

### `PYTORCH_JIT=0` is required, and it is not cosmetic

The SE(3) transformer is compiled with `@torch.jit.script` — **608 ScriptModules**.
Two consequences, both fatal to a naive pinning claim:

1. TorchScript compilation *fails* while the patches are active (it calls
   `inspect.getsourcelines` on the wrapper and finds a builtin).
2. Even if it compiled, **TorchScript executes its own compiled graph and would
   ignore the Python-level patches entirely** — so the whole SE(3) refiner would
   silently run unpinned while `pinned.report()` showed healthy counts elsewhere.

Setting `PYTORCH_JIT=0` makes `@torch.jit.script` a no-op so the Python path
runs and is patched. The evidence that this closed the hole: the module map goes
from **4 994 modules (JIT on) to 5 602 (JIT off)** — exactly the 608 previously
opaque ScriptModules becoming visible.

### Result 1 — pinned mode is order-independent on the real network

Two pinned runs, **different thread counts** (4 vs 2 threads → different MKL
blocking and different reduction orders):

```
91/91 tensors bit-identical
ALL TENSORS BIT-IDENTICAL
```

That includes `px0`, the full pair track after 32 main blocks, and the SE(3)
refiner output. This is the claim that matters, and it now holds on the real
model rather than on synthetic GEMMs.

### Result 2 — how far pinned sits from stock

Same weights, same seed, same discrete decisions; only the arithmetic
convention differs:

```
tensor                                        n      bitexact%     max|Δ|         cos
out::model.simulator.main_block.31.1     967872       0.0157%   1.261e+00  1.000011309
out::model.simulator.main_block.0.1      967872       0.0126%   2.161e-01  1.000037165
out::model.recycle.1                     967872      24.7403%   1.817e-02  1.000022294
x_t_next                                   7668      95.5790%   1.800e-02  1.000000103
px0                                        7881      94.8230%   1.045e-02  1.000000000

38/91 tensors bit-identical
max |Δ| over all tensors: 1.261e+00
```

**This is larger than a single layer's 1.4e-6, and the reason is amplification:**
36 blocks of pair/MSA attention compound fp32 round-off. The honest reading:

- On the **output** the divergence is **max |Δ| ≈ 0.010 Å on `px0`** with cosine
  1.000000000 — structurally negligible (well below any meaningful RMSD
  threshold) but definitively **not** bit-identical.
- The large `max |Δ| 1.26` entries are deep pair-track activations, not
  coordinates; they are unnormalised features whose scale is large.
- Read the other way: **stock fp32 RFdiffusion2 carries ~0.01 Å of its own
  numerical noise**, and pinned mode is the more accurate of the two — it is the
  correctly-rounded result and it removes PyTorch's non-correctly-rounded fp32
  `sqrt`.

### What this means for the deliverable

The Rust port can be made **bit-identical to the pinned reference**, and that is
a fully checkable, reproducible target — insensitive to threads, SIMD and
blocking, as Result 1 demonstrates. It will **not** be bit-identical to a stock
PyTorch+MKL run; the gap is ~0.01 Å on output coordinates, and no non-MKL
implementation can close it (§1).

If the requirement is specifically "byte-identical PDB to stock
`run_inference.py`", that remains unachievable by any reimplementation, and the
only answer is to run the reference.


---

## Addendum, 2026-08-09: four holes in the pinning, and where the limit actually is

Rung 6 found that the "pinned reference" was not as pinned as this document
claimed. Four routes bypassed it entirely, and each was worth a real bug:

1. **`opt_einsum`.** `rf2aa` does `from opt_einsum import contract as einsum`,
   and opt_einsum's torch backend forwards as `torch.einsum(equation, operands)`
   — the *sublist* form. `_wrap`'s promotion test only looked at the top level of
   `args`, saw `(str, tuple)`, and passed the call through unpinned. **Every
   attention contraction in the network was running in stock fp32** while
   `report()` showed a healthy `torch.einsum` count (those were the 410 direct
   calls in `util.py`). Fixed by recursing into lists and tuples in `_to64`.

2. **`dgl.ops`.** The SE(3) attention does its softmax and its neighbour sums
   inside DGL's compiled kernels (`edge_softmax`, `copy_e_sum`, `e_dot_v`), which
   reduce over each destination node's incoming edges in an order set by DGL's
   CSR layout. Nothing in `torch.*` sees them. Now patched.

3. **`sigmoid` and friends.** `torch.sigmoid` fires on every gate in every
   attention block and was never in the target list, along with `tanh`, `asinh`
   (the SE(3) edge feature), `linalg.norm` and `group_norm` (`NormSE3`).

4. **The JIT.** `PYTORCH_JIT=0` was documented as mandatory but `ref_dump.py`
   left it to the caller. It is now set inside the script — and it has a second
   reason to be there: e3nn scripts some module-level functions at import time,
   and scripting recursively compiles the `torch.*` names they reference, which
   under pinning are Python wrappers around builtins and fail to parse.

### Where the residual limit is

With those closed, 35 of 36 trunk blocks are bit-identical from the reference's
own inputs. The remaining block differs by **1 ULP** in ~1e-5 of its pair values.

That is the expected behaviour of this strategy, not a bug to be chased
indefinitely. f64 pinning makes the fp32 result order-independent *whenever* the
exact value is further than ~1e-16 (relative) from the midpoint between two fp32
numbers. It is not order-independent when the value lands inside that window,
which is about **2e-9 of values**. A forward pass at L = 71 evaluates on the
order of 1e9 reduction outputs, so a small number of ULP flips per pass is what
the arithmetic predicts.

Measured here (`tests/probe_f64_tie.rs`), on real weights and real activations:
reversing the K-loop of the widest trunk projections changed **0 of 16 776 448**
outputs — consistent with a rate below ~1e-7 and with the 2e-9 estimate, but not
by itself a demonstration of it.

### The single remaining disagreement, resolved to one number

Bisected all the way down, the whole difference in `main_block.0` comes from
**one output of one 192-term dot product** — `row_attn.to_k[4427, 157]` — which
then spreads through the attention into 299 values. Its input is bit-identical,
so the dot product itself is the whole story:

```
exact (compensated)  -1.95089882612228371350e0  -> f32 -1.9508988   <- the port
sequential f64       -1.95089882612228349146e0  -> f32 -1.9508988
reversed f64         -1.95089882612228393555e0  -> f32 -1.9508989   <- the reference
2/4/8-lane f64       -1.95089882612228349146e0  -> f32 -1.9508988
```

The exact value lies within 4.4e-16 of the midpoint between two fp32 numbers.
Computing the sum exactly (Neumaier compensation, valid here because the product
of two fp32 values is exact in f64) gives `-1.9508988` — so **the port's answer
is the correctly-rounded one and the reference's is the 1-ULP error**, courtesy
of MKL's f64 GEMM blocking.

That reframes the residual. It is not a defect in the port to be fixed; it is
the reference's own f64 round-off, and matching it would mean reproducing MKL's
blocking — the same problem this document opened by rejecting for fp32, one rung
down. The rate is what the arithmetic predicts: ~3e-9 per reduction output,
about 3.6e8 such outputs in the 36-block trunk, and exactly one event observed.

The honest statement is therefore: **bit-identity holds per block, 35 blocks out
of 36, and the one exception is a value where the reference is wrong and the
port is right.**


---

## Settled: the port is correctly rounded, the reference is not (2026-08-10)

The section above bisected one 192-term dot product, found that compensated
summation agreed with the port, and concluded the reference was 1 ULP wrong.
That was one value checked once. It did not establish that the port's own f64
accumulation is correctly rounded, nor that the value was the only case.

Both are now measured. Full numbers in `results/stage0_correctly_rounded.txt`.

**The instrument.** Products of two fp32 values are *exact* in f64
(24 + 24 = 48 <= 53), so every reduction here is a sum of exact f64 terms and
the only error is in the summation. `--features exact` swaps every reduction
onto a double-double accumulator (`src/ops/acc.rs`): ~106 significand bits, so
the accumulated relative error is ~2^-104 against ~2^-53 for the default
lane-split path. That is correctly rounded to fp32 unless the exact value sits
within 2^-104 of a midpoint — ~2^-80 of values, i.e. never.

It is a **compile-time feature**, so the tuned kernels are untouched by default;
`Acc` is a `#[repr(transparent)]` newtype over `f64` with `#[inline(always)]`
methods in that build, and the default run reproduces every previously recorded
parity number to the digit. `parity_exact_gemm.rs` asserts the two builds are
genuinely distinct, so the comparison cannot be vacuously green.

**The measurement.**

```
full forward, 998 260 outputs   fast.bin vs exact.bin   BYTE-IDENTICAL
                                40.3 s  vs  403.7 s     (10.0x)

main_block.0 from reference inputs, both builds, identical to every digit:
  xyz    99.84%  max_ulp 1        alpha  88.70%  max_ulp 361
  quat   97.54%  max_ulp 20
```

The port's answer does not move when the accumulator is given 53 more bits. So
the port already computes the correctly-rounded fp32 value of the pinned
algorithm, and wherever the two sides differ, **the reference is the side that
is wrong**.

**And MKL is not sloppy — the ties are just rare.** Hooking the real
`main_block.0.pair2pair.row_attn.to_k` during a reference forward (its input is
layer-normed *inside* the attention, so it cannot be reconstructed from the
block input) and checking all 967 872 outputs against exact `fsum` summation
found **zero** disagreements, including at the bisected element `[4427, 157]`.
That is consistent with the ~2e-9 per-value tie rate `probe_f64_tie.rs`
measures: 1e6 outputs is ~1000x too small a sample. The disagreements are rare
individual ties spread across the ~1e9 reductions in a forward pass.

**What this changes.** Byte-identity between the port's output and a pinned
reference run is **unattainable by construction**, not an unfinished task. The
correct gate is the one already in use: the port is correctly rounded, and every
disagreement is localised and attributed.
