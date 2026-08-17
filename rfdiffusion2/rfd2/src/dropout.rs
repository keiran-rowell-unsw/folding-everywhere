//! Dropout — which RFdiffusion2 runs **at inference**.
//!
//! This is not a training-only path that the port can skip. Nothing on the
//! inference path ever calls `.eval()`: measured on the real sampler, all
//! **5 758** modules report `training == True`, including **184 `nn.Dropout`**
//! and **112 `rf2aa.util_module.Dropout`** instances. So every published
//! RFdiffusion2 design carries dropout noise, drawn from the torch RNG inside
//! the forward pass — the same situation as `psi_pred`, but ~300 draws per
//! forward instead of one.
//!
//! Reproducing it means reproducing two *different* generators, because the two
//! dropout classes reach ATen through two different kernels:
//!
//! | class | call | kernel | RNG |
//! |---|---|---|---|
//! | `rf2aa` `Dropout` | `Bernoulli(tensor([1-p])).sample(shape)` | `bernoulli_tensor_cpu_` | the torch MT19937, one fp32 uniform per element |
//! | `nn.Dropout` | `F.dropout` -> `mask.bernoulli_(1-p)` | `bernoulli_scalar_cpu_` | **MKL VSL**, one MT19937 draw total |
//!
//! The second one is the trap. A scalar-`p` `bernoulli_` on an MKL-enabled build
//! does *not* walk the torch stream per element: it takes a single
//! `generator->random()` as a seed, opens an MKL `VSL_BRNG_MCG31` stream, and
//! draws from that. Assuming it behaves like `rand() < p` gets both the mask and
//! every subsequent draw in the stream wrong.
//!
//! Both are reproduced here and both are verified against torch:
//! `Bernoulli(tensor).sample == rand < p` on 36/36 shape/seed/p combinations,
//! and the MKL path on **288/288** seed x p x n combinations
//! (n up to 5000, i.e. across MKL's 800-element `parallel_for` grain, which does
//! not perturb the sequence because skip-ahead on a multiplicative congruential
//! generator is just repeated multiplication).

use crate::rng::torch::Mt19937;
use crate::tensor::Tensor;

/// MKL's `VSL_BRNG_MCG31` (MCG31m1): `x <- 1132489760 * x mod (2^31 - 1)`,
/// with `x0 = seed mod m` and `0` mapped to `1`.
///
/// Note the sequence **starts at `x0` itself**, before any multiplication —
/// advancing first shifts every mask by one element, which is exactly the kind
/// of off-by-one that still produces a plausible-looking structure.
pub struct Mcg31 {
    x: u64,
}

impl Mcg31 {
    const M: u64 = (1 << 31) - 1;
    const A: u64 = 1_132_489_760;

    pub fn new(seed: u64) -> Self {
        let mut x = seed % Self::M;
        if x == 0 {
            x = 1;
        }
        Mcg31 { x }
    }

    /// The current state, then advance.
    #[inline]
    pub fn next_u31(&mut self) -> u64 {
        let v = self.x;
        self.x = (Self::A * self.x) % Self::M;
        v
    }

    /// The uniform variate, **in single precision**.
    ///
    /// This is not a detail. MKL's `viRngBernoulli`/ICDF compares an fp32
    /// uniform against an fp32 `p`, so the effective threshold is
    /// `f32(x) * f32(1/M) < f32(p)` — not the f64 comparison it looks like.
    /// The two agree except within ~50 counts of `p·M`, which is roughly one
    /// draw in 4e7; RFdiffusion2 takes ~7e7 MKL draws per forward, so it hits
    /// the boundary a couple of times *per run*. Observed as exactly one dropout
    /// mask element flipping in `extra_block.2`'s `pair2pair.ff`, which moved a
    /// single pair cell (42, 39) by 1.03 — a difference that looks structural
    /// and is really one bit of one mask.
    #[inline]
    pub fn next_uniform_f32(&mut self) -> f32 {
        self.next_u31() as f32 * (1.0f32 / Self::M as f32)
    }
}

/// `torch.Tensor.bernoulli_(p)` with a **scalar** `p` — the `nn.Dropout` path.
///
/// Consumes exactly one `u32` from the torch generator no matter how many
/// elements are produced.
pub fn bernoulli_scalar(gen: &mut Mt19937, n: usize, p: f64) -> Vec<f32> {
    let seed = gen.random() as u64;
    let mut m = Mcg31::new(seed);
    let pf = p as f32;
    (0..n).map(|_| if m.next_uniform_f32() < pf { 1.0 } else { 0.0 }).collect()
}

/// `torch.distributions.Bernoulli(torch.tensor([p])).sample(shape)` — the
/// `rf2aa` `Dropout` path. One fp32 uniform per element, in order.
pub fn bernoulli_tensor(gen: &mut Mt19937, n: usize, p: f32) -> Vec<f32> {
    (0..n)
        .map(|_| if gen.uniform_f32() < p { 1.0 } else { 0.0 })
        .collect()
}

/// `nn.Dropout(p)` in training mode.
///
/// The scaling is **`noise.div_(1 - p)` followed by `input * noise`**, not a
/// multiply by a precomputed `1/(1-p)`. That is `at::_dropout_impl`, the CPU
/// path: `native_dropout` (which does use a reciprocal) is gated behind
/// `is_fused_kernel_acceptable`, which requires a CUDA/XPU/lazy tensor and is
/// therefore never taken here.
///
/// The distinction is not cosmetic. At **p = 0.15**,
/// `f32(1.0 / 0.85) = 1.1764706` but `f32(1.0) / f32(0.85) = 1.1764705` — one
/// ULP apart, and `nn.Dropout(0.15)` is on `msa2msa.ff` and both of `Str2Str`'s
/// feed-forwards, so the wrong one leaves the whole block ~1e-6 off with
/// `cos = 1.0000000000`: a discrepancy that looks like harmless round-off and is
/// actually a reproducible bug. (At p = 0.1 and p = 0.25 the two agree, so a
/// spot check on those two would have missed it.)
pub fn nn_dropout(gen: &mut Mt19937, x: &Tensor, p: f64) -> Tensor {
    let p1m = 1.0 - p;
    let mask = bernoulli_scalar(gen, x.numel(), p1m);
    let denom = p1m as f32;
    Tensor::new(
        x.data.iter().zip(&mask).map(|(v, m)| v * (m / denom)).collect(),
        x.shape.clone(),
    )
}

/// `rf2aa.util_module.Dropout(broadcast_dim, p_drop)` in training mode.
///
/// The mask has the broadcast axis collapsed to length 1, so the *number of
/// draws* depends on `broadcast_dim` — getting that wrong desynchronises every
/// later draw in the stream, including `psi_pred`.
pub fn rf_dropout(gen: &mut Mt19937, x: &Tensor, broadcast_dim: Option<usize>, p: f64) -> Tensor {
    let mut shape = x.shape.clone();
    if let Some(d) = broadcast_dim {
        shape[d] = 1;
    }
    let n: usize = shape.iter().product();
    // `1 - p` is evaluated in Python floats (f64) and only then narrowed for
    // the tensor op. Doing it in fp32 instead lands on a rounding tie for
    // p = 0.15 and can pick the other neighbour.
    let p1m = (1.0 - p) as f32;
    let mask = bernoulli_tensor(gen, n, p1m);
    let inv = p1m;
    let mut out = vec![0.0f32; x.numel()];
    let strides = x.strides();
    let mstrides = {
        let mut s = vec![1usize; shape.len()];
        for i in (0..shape.len().saturating_sub(1)).rev() {
            s[i] = s[i + 1] * shape[i + 1];
        }
        s
    };
    let nd = shape.len();
    let mut idx = vec![0usize; nd];
    for (o, dst) in out.iter_mut().enumerate() {
        let mut mi = 0usize;
        for d in 0..nd {
            if shape[d] != 1 {
                mi += idx[d] * mstrides[d];
            }
        }
        *dst = mask[mi] * x.data[o] / inv;
        // odometer over the OUTPUT shape
        for d in (0..nd).rev() {
            idx[d] += 1;
            if idx[d] < x.shape[d] {
                break;
            }
            idx[d] = 0;
        }
        let _ = strides;
    }
    Tensor::new(out, x.shape.clone())
}
