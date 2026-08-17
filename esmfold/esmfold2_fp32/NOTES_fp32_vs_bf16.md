# Why the bf16 atom attention isn't reproduced bit-for-bit (and what actually is)

> Context: ESMFold2's released inference path hard-casts the SWA atom attention to
> **bfloat16**. The natural question — "bf16 is deterministic, so why can't you
> reproduce it in Rust?" — has a precise answer. There are two different things,
> and only one of them is hard.

## The bf16 *cast* — yes, I reproduce it exactly

`x.to(torch.bfloat16)` is round-to-nearest-even on the mantissa, fully
deterministic. My Rust uses the `half` crate's `bf16::from_f32`, which is the same
operation. I cast cos/sin and q/k/v to bf16 at the exact same points the reference
does. **That part matches bit-for-bit.** So "can't reproduce the cast" was the
wrong way to put it.

## What I haven't reproduced is the *fused bf16 attention kernel's internal arithmetic*

The hard part isn't the cast — it's everything torch does *between* the casts
inside `scaled_dot_product_attention`: `QKᵀ` (a bf16 matmul), `·scale`, `softmax`,
`·V` (another bf16 matmul). Each of those is also deterministic, but the result
depends on torch's **specific CPU kernel**: the fp32 accumulation order of the
matmul reductions, and exactly where it rounds intermediates back to bf16 (after
the QKᵀ? after softmax? at tile boundaries in a fused flash-style kernel?). Those
choices are an undocumented, version- and CPU-ISA-dependent implementation detail
in oneDNN/ATen — not a simple formula.

My experiment is the evidence. I tried the three natural recipes against torch's
actual bf16 `F.scaled_dot_product_attention` (random q,k,v) and none nails it:

```
fp32 math, round only output   3.9e-3   ("torch is fp32-opmath-then-round"? → no)
round scores+probs to bf16     7.8e-3
bf16 per-stage matmul          7.8e-3
```

Notably the first uses torch's *own* fp32 matmul and *still* differs by 3.9e-3
from torch's bf16 SDPA — so the fused bf16 kernel is doing its reductions/roundings
in an order none of those three reproduce. It's deterministic, just not a recipe
I've matched.

## So: reproducible in principle, but the wrong battle

I *could* reverse-engineer torch's exact CPU bf16 SDPA kernel and hard-code its
reduction/rounding order in Rust — but that's brittle (breaks across torch versions
and CPU instruction sets) and is the same class of problem as matching MKL's sgemm
to 0 ULP.

The clean and robust fix is the one in `RESULTS.md` §5A: **make both sides fp32**
by deleting the two hard-coded bf16 casts from the reference (they're a
speed/memory optimization, not part of the trained weights). Then there's no bf16
kernel to match at all — the attention is plain fp32, and the residual drops from
bf16-ULP (~4e-3) to fp32-accumulation level (~1e-5), which is already handled with
f64 accumulation. That's the path to sub-mÅ.

## Two options to push further

- **(a) fp32-both-sides re-validation** — patch the reference to drop the bf16
  casts (`build_3d_rope` `cos/sin.to(bfloat16)`; the `q,k,v = .bfloat16()` branch
  in `SWA3DRoPEAttention`), regenerate fixtures, re-run a protein, and demonstrate
  the sub-mÅ match. (Recommended; clean and version-independent.)
- **(b) reverse-engineer torch's bf16 SDPA kernel** so the *released* bf16 path
  matches bit-for-bit. (Possible but brittle.)
