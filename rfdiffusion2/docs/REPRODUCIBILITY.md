# Will the Rust port and PyTorch produce identical output?

The one question this project exists to answer, stated precisely: **given the
same input and the same settings, do `rfd2` and RFdiffusion2 write the same
file?**

Short answer: **yes, byte for byte — against a *pinned* PyTorch, on the
configurations measured so far.** Every qualifier in that sentence is load
bearing. This document says exactly what each one means.

Last measured 2026-08-11. Numbers come from `results/layerwise_M0584_1ldm.tsv`
and `results/README.md`; nothing here is claimed that a test does not assert.

---

## 1. What is verified

| configuration | result |
|---|---|
| `M0584_1ldm`, contig `10,A106-106,10`, **T = 2** | **byte-identical**, sha256 `ec256c46…` |
| `M0584_1ldm`, same contig, **T = 100** (the demo's real setting) | 263/267 lines; 4 atoms differ by **0.001 Å**, one unit in the last printed decimal. All 50 HETATM and all 106 CONECT identical. |
| `M0584_1ldm`, variable-length contig `5-15,A106-106,5-15` + `contigmap.length=25-25`, T = 2 | **byte-identical**, sha256 `d8d96ef4…` (re-verified post-fix, 287 lines) |

"Byte-identical" means `cmp` returns 0 on the written `.pdb`: identical atom
records, identical ligand block, identical `CONECT` records, identical
formatting.

### Agreement is NOT monotone in the port's correctness

This is the single most counter-intuitive fact in this project, and it is
measured, not argued. On 2026-08-11 a **provably correct** fix to the port
(`ops::reduce::sum_compensated` — verified against 400-bit exact arithmetic)
moved the two configurations in **opposite directions**:

| | T = 2 | T = 100 |
|---|---|---|
| before the fix | 258/267, max 0.048 Å | **byte-identical** |
| after the fix | **byte-identical** | 263/267, max 0.001 Å |

The reason: **the reference is not correctly rounded either.** It carries its own
1-ULP errors — proven at `main_block.23.row_attn.norm_pair`, where exact
arithmetic puts the true value 5.57e-9 of an fp32 ULP on the *port's* side of a
midpoint. So the port had two errors that happened to cancel over the T = 100
trajectory; removing one uncovered the other.

Consequences, stated because they govern how to read any future benchmark:

* **Byte-identity is achievable but fragile.** It is not a guarantee that
  survives an arbitrary input, and improving the port can break it.
* **Do not optimise for byte-identity.** The fix was kept: it lowers the
  worst-case disagreement across both tested configurations from **0.048 Å to
  0.001 Å**, a 48x improvement, even though it costs one exact match. Reverting
  it would leave the port knowingly mis-rounding `layer_norm`.
* **The robust route is to make BOTH sides correctly rounded** — i.e. widen
  `python/pinned.py`'s reductions past f64 for the ops where this bites
  (`layer_norm` first: both known disagreements originated there). That changes
  the reference and invalidates every fixture, so it is a deliberate decision,
  not a patch.

## 2. The qualifier that matters most: *pinned*, not stock

**"PyTorch" above means PyTorch running under `python/pinned.py`**, a shim that
patches **100 entry points** so every fp32 op computes its interior in f64 and
rounds to fp32 exactly once.

Against **stock, unmodified PyTorch + MKL the port is NOT bit-identical, and
cannot be made so.** This is a property of MKL, not a gap in the port:

* Stock MKL's fp32 GEMM uses a reduction order that no other implementation
  reproduces. Measured at RFdiffusion2's real shapes, the best-matching
  candidate order agreed on **99.1 % of outputs at K ≤ 65 and ~10 % at K ≥ 192**,
  never 100 %.
* The route that works is to make order *irrelevant* — accumulate in f64 on both
  sides and round once. That is what pinning does, and it was verified
  order-independent on the real 82.9 M-parameter model (two runs at different
  thread counts, 91/91 tensors bit-identical).
* A pinned run differs from a stock run by **~0.01 Å on `px0`**.

So:

> If the expectation is *"someone runs stock RFdiffusion2 and gets our file"* —
> that is **not** demonstrated and is not achievable.
>
> If the expectation is *"the two implementations agree exactly under a defined,
> reproducible numerical convention"* — that is what is demonstrated.

### What "pinned" does mechanically — and what it does NOT do

Both sides run the **same convention**, which is what makes agreement possible:

| side | how |
|---|---|
| PyTorch | `python/pinned.py` wraps ~100 op entry points: fp32 inputs -> promote to f64 -> compute -> narrow to fp32 **once** |
| Rust | ops accumulate natively in f64 (`ops::acc::Acc`, `dot_f64`, `layer_norm_f64`, `softmax_last_f64`, ...) -> narrow to fp32 **once** |

**This is not "run the model in f64".** Tensors stay fp32 *between* ops; only
each op's interior is f64. So every fp32 rounding happens at exactly the same
place as in stock RFdiffusion2 — each one just lands on the correctly-rounded
value instead of an order-dependent one. The op boundaries, and therefore the
model, are unchanged.

Why order matters at all: fp32 addition is not associative. Same 192 fp32
values, same dot product, only the accumulation order changed —

```
fp32 sequential     0xbf716e2c
fp32  4-lane SIMD   0xbf716e34
fp32  8-lane SIMD   0xbf716e40      <- 20 ULP from sequential
fp32 16-lane SIMD   0xbf716e40

f64 accumulate -> fp32 once
  forward / reverse / shuffled      0xbf716e3e   <- all identical
```

`0xbf716e3e` is the correctly-rounded answer; stock's sequential fp32 is 18 ULP
away from it. So matching stock would mean **reproducing its error**.

### Why stock (A) is not the target

1. **It is an artifact, not a specification.** MKL's blocking, SIMD lane count
   and combination tree are proprietary, chosen by runtime CPU dispatch, and
   change between MKL versions. There is nothing to implement against — which is
   why Intel ships `MKL_CBWR` at all.
2. **Measured unreachable.** Every plausible fp32 order was tried (§1 of
   `docs/BITEXACT.md`): best 99.14 % at K=32, 9.61 % at K=320, never 100 %. The
   obvious explanation — fused multiply-add — was tested and *rejected*: adding
   FMA made it worse (99.06 % -> 29.56 % at K=64), so products are rounded and
   the residual is blocking.
3. **It would be per-CPU.** AVX2 and AVX-512 dispatch different kernels; the
   deliverable is portable Windows/Linux/macOS binaries.
4. **It would make the port less accurate**, deliberately.

Neither A nor B is "the truth" — both approximate the same exact function, and
**B is closer to it**. The port computes that function more accurately; B is a
referee both implementations can independently agree on.

For scale: the A-B gap is ~0.01 A on `px0`, while RFdiffusion2 runs **with
dropout live at inference** (~2.64 M draws per forward, nothing calls `.eval()`),
so the seed changes the design far more than the arithmetic does. Byte-
reproducing a published design is therefore not achievable by anyone — including
a rerun of the official pipeline on different hardware.

## 3. The other conditions, all mandatory

| condition | why |
|---|---|
| `RFD2_PINNED=1` | §2 — the numerical convention |
| `PYTORCH_JIT=0` | without it the SE(3) transformer's **608 ScriptModules** run their own compiled graph, ignore the Python-level pinning, and cannot be hooked. The audit counts still look healthy, so this fails silently. |
| `PYTHONHASHSEED=1` | **the reference's own output is otherwise non-deterministic.** `dev/idealize_backbone.rewrite` recovers the ligand list from a Python `set`; CPython randomises string hashes, so the HETATM block order follows the interpreter's hash seed. `inference.deterministic=True` does *not* cover this. |
| `OMP/MKL_NUM_THREADS=1`, `MKL_CBWR=COMPATIBLE` | the documented environment. Thread count is *not* expected to matter under pinning (measured order-independent), but this is the configuration every number was taken in. |
| a ligand-topology sidecar | 3 of 4 demo inputs have no `CONECT` records, so OpenBabel perceives connectivity *and* aromaticity from coordinates. `python/gen_ligand_bonds.py` runs the reference's own path once per PDB; `src/ligand.rs` hard-errors on anything uncovered. **A novel ligand needs one Python invocation before `rfd2` can run on it.** |

## 4. What is NOT yet verified

* **Only one protein** (`M0584_1ldm`) and two contigs.
* `num_designs > 1` — the per-design reseed.
* The `.trb` output file — the port does not write one.
* Every rung-8 configuration: atomization, guideposts, `partial_T`, self-
  conditioning beyond the one T=2 check, any second input or ligand.
* Anything on GPU (this machine has none).

These are the subject of the **planned benchmarking experiment** (multiple
proteins, multiple configurations, more designs). Until then, note the port
**refuses** rather than guesses on most of them — `sample_init::Options` rejects
`preserve_motif_sidechains` / `independently_center_diffuseds` / `partial_T`,
`mask_indep` rejects a non-protein masked row, `writepdb_file` rejects a residue
that would trigger `fix_null_sidechains`, and `ContigMap::parse` refuses a range
rather than pretending to be deterministic. Each refusal names what was not
measured, so a wrong answer is never returned silently.

## 5. One known internal disagreement, invisible in the output

`main_block.23`'s `row_attn.norm_pair` still differs from the reference in
**one element of 967 872, by 1 ULP**. Adjudicated at 400-bit precision: the exact
value sits 5.57e-9 of an fp32 ULP *above* the midpoint, so **the port is
correctly rounded and the reference is the wrong side**. Its mean and variance
already agree to float80, so no summation change applies; matching it would mean
reproducing ATen's rounding error deliberately.

It no longer shows in the written file — but that is because it is *below the
PDB's print resolution*, not because it is absent. `%8.3f` resolves 5e-4 Å; the
perturbation is ~1e-5 Å. **This is the honest reason not to promise byte-identity
on an arbitrary input from two configurations.** A different trajectory could
amplify it above 5e-4 Å.

The mirror-image case in `main_block.2` *was* a real port defect (naive f64
summation in `layer_norm`) and is fixed — see
`results/layerwise_M0584_1ldm.tsv`.

## 6. How to check it yourself

Full invocations: `results/README.md`. The one-line gate:

```bash
cmp runs/<case>/ref/design_0-atomized-bb-False.pdb \
    runs/<case>/rs/design_0-atomized-bb-False.pdb && echo IDENTICAL
```

**Before believing any of this, check the fixtures are not stale** — a fixture
older than the last change to `python/pinned.py` *or* to the noiser measures the
wrong target, and the tests still pass:

```bash
stat -c '%y  %n' python/pinned.py fixtures/*/*.safetensors | sort
```

That failure mode cost a full investigation on 2026-08-11, and the end-to-end
byte-diff **cannot** detect it: the two reference generations differed by
1.5e-5 Å on `px0`, far below what the PDB format can express.
