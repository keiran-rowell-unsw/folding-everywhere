//! LayerNorm, softmax and log-softmax over the last dimension.
//!
//! Each op comes in two variants:
//!
//! * **stock** (`layer_norm`, `softmax_last`, …) — fp32 throughout with a pinned
//!   accumulation order. Tracks stock PyTorch to ~1e-6.
//! * **pinned** (`*_f64`) — every intermediate in f64, rounded to f32 exactly
//!   once at the end. This is the bit-exact mode described in
//!   `docs/BITEXACT.md`: because the f64 rounding error (~1e-16 relative) is far
//!   below an f32 ULP (~1e-7), the f32 result is the correctly-rounded one and
//!   therefore does not depend on summation order, blocking, SIMD width or
//!   thread count. Measured, not assumed — see `python/probe_f64_pinning.py`.

use rayon::prelude::*;
use crate::tensor::Tensor;

/// LayerNorm over the last dim. Biased variance, eps inside sqrt (matches ATen).
pub fn layer_norm(x: &Tensor, weight: &Tensor, bias: &Tensor, eps: f32) -> Tensor {
    let c = x.last();
    assert_eq!(weight.numel(), c);
    assert_eq!(bias.numel(), c);
    let rows = x.numel() / c;
    let mut out = vec![0.0f32; x.numel()];
    let w = &weight.data;
    let b = &bias.data;
    for r in 0..rows {
        let xr = &x.data[r * c..r * c + c];
        let or = &mut out[r * c..r * c + c];
        let mut mean = 0.0f32;
        for &v in xr {
            mean += v;
        }
        mean /= c as f32;
        let mut var = 0.0f32;
        for &v in xr {
            let d = v - mean;
            var += d * d;
        }
        var /= c as f32;
        let rstd = 1.0f32 / (var + eps).sqrt();
        for i in 0..c {
            or[i] = (xr[i] - mean) * rstd * w[i] + b[i];
        }
    }
    Tensor::new(out, x.shape.clone())
}

/// Softmax over the last dimension (max-subtracted).
pub fn softmax_last(x: &Tensor) -> Tensor {
    let c = x.last();
    let rows = x.numel() / c;
    let mut out = vec![0.0f32; x.numel()];
    for r in 0..rows {
        let xr = &x.data[r * c..r * c + c];
        let or = &mut out[r * c..r * c + c];
        let mut m = f32::NEG_INFINITY;
        for &v in xr {
            if v > m {
                m = v;
            }
        }
        let mut s = 0.0f32;
        for i in 0..c {
            let e = libm::expf(xr[i] - m);
            or[i] = e;
            s += e;
        }
        let inv = 1.0f32 / s;
        for i in 0..c {
            or[i] *= inv;
        }
    }
    Tensor::new(out, x.shape.clone())
}

/// log_softmax over the last dimension: (x - max) - log(sum(exp(x - max))).
/// Matches ATen's `_log_softmax` (which also subtracts the max and takes the log
/// of the summed exponentials rather than log(softmax)).
pub fn log_softmax_last(x: &Tensor) -> Tensor {
    let c = x.last();
    let rows = x.numel() / c;
    let mut out = vec![0.0f32; x.numel()];
    for r in 0..rows {
        let xr = &x.data[r * c..r * c + c];
        let or = &mut out[r * c..r * c + c];
        let mut m = f32::NEG_INFINITY;
        for &v in xr {
            if v > m {
                m = v;
            }
        }
        let mut s = 0.0f32;
        for i in 0..c {
            s += libm::expf(xr[i] - m);
        }
        let logsum = libm::logf(s);
        for i in 0..c {
            or[i] = (xr[i] - m) - logsum;
        }
    }
    Tensor::new(out, x.shape.clone())
}

// ---------------------------------------------------------------------------
// Bit-exact ("pinned") variants: f64 intermediates, one rounding to f32.
// ---------------------------------------------------------------------------

/// `F.layer_norm(x.double(), ..., eps).float()`.
///
/// Mean and (biased) variance are accumulated in f64, `eps` is added inside the
/// sqrt as ATen does, and the result is narrowed to f32 only at the store.
/// Neumaier compensated summation: the running error term is carried alongside
/// and folded in at the end, which makes the result correctly rounded for any
/// input this network produces.
///
/// Why here and not in the GEMM: a `layer_norm` row is an O(n) reduction over
/// 192 values and the compensation costs ~3 flops per element on a path that is
/// nowhere near the bottleneck. The GEMM's inner loop is O(n*k) at 17 GFLOP/s
/// and cannot pay it — see `ops::acc`, where the same problem is solved with a
/// build feature instead.
///
/// This is load-bearing, not hygiene. Measured on `main_block.2`'s
/// `tri_mul_out.norm`: naive sequential summation gives `var = 162.5444035746675`
/// where the correctly-rounded value is `162.54440357466754` — one f64 ULP. The
/// exact `layer_norm` output for one element of that row sits 3.1e-9 of an fp32
/// ULP below a midpoint, so that single f64 bit flipped the narrowed fp32 result
/// and put the port 1 ULP away from the reference. ATen's own reduction is
/// blocked/vectorised and does not have the error, so the port was the wrong
/// side. See `results/layerwise_M0584_1ldm.tsv`.
#[inline]
fn sum_compensated(xs: impl Iterator<Item = f64>) -> f64 {
    let mut s = 0.0f64;
    let mut c = 0.0f64;
    for v in xs {
        let t = s + v;
        // whichever operand is larger keeps its low bits; the other's are lost
        c += if s.abs() >= v.abs() { (s - t) + v } else { (v - t) + s };
        s = t;
    }
    s + c
}

pub fn layer_norm_f64(x: &Tensor, weight: &Tensor, bias: &Tensor, eps: f64) -> Tensor {
    let c = x.last();
    assert_eq!(weight.numel(), c);
    assert_eq!(bias.numel(), c);
    let rows = x.numel() / c;
    let mut out = vec![0.0f32; x.numel()];
    // Row-wise and independent, so parallelising cannot change a value: each
    // output row is a function of its own input row alone.
    let _ = rows;
    out.par_chunks_mut(c).enumerate().for_each(|(r, or)| {
        let xr = &x.data[r * c..r * c + c];
        let mut mean = sum_compensated(xr.iter().map(|&v| v as f64));
        mean /= c as f64;
        let mut var = sum_compensated(xr.iter().map(|&v| {
            let d = v as f64 - mean;
            d * d
        }));
        var /= c as f64;
        let rstd = 1.0f64 / (var + eps).sqrt();
        for i in 0..c {
            or[i] = ((xr[i] as f64 - mean) * rstd * weight.data[i] as f64
                + bias.data[i] as f64) as f32;
        }
    });
    Tensor::new(out, x.shape.clone())
}

/// `F.softmax(x.double(), dim=-1).float()`.
pub fn softmax_last_f64(x: &Tensor) -> Tensor {
    let c = x.last();
    let _rows = x.numel() / c;
    let mut out = vec![0.0f32; x.numel()];
    out.par_chunks_mut(c).enumerate().for_each(|(r, or)| {
        let xr = &x.data[r * c..r * c + c];
        let mut buf = vec![0.0f64; c];
        let mut m = f64::NEG_INFINITY;
        for &v in xr {
            let v = v as f64;
            if v > m {
                m = v;
            }
        }
        let mut s = 0.0f64;
        for i in 0..c {
            let e = (xr[i] as f64 - m).exp();
            buf[i] = e;
            s += e;
        }
        for i in 0..c {
            or[i] = (buf[i] / s) as f32;
        }
    });
    Tensor::new(out, x.shape.clone())
}

/// `F.log_softmax(x.double(), dim=-1).float()`.
pub fn log_softmax_last_f64(x: &Tensor) -> Tensor {
    let c = x.last();
    let rows = x.numel() / c;
    let mut out = vec![0.0f32; x.numel()];
    for r in 0..rows {
        let xr = &x.data[r * c..r * c + c];
        let or = &mut out[r * c..r * c + c];
        let mut m = f64::NEG_INFINITY;
        for &v in xr {
            let v = v as f64;
            if v > m {
                m = v;
            }
        }
        let mut s = 0.0f64;
        for &v in xr {
            s += (v as f64 - m).exp();
        }
        let logsum = s.ln();
        for i in 0..c {
            or[i] = ((xr[i] as f64 - m) - logsum) as f32;
        }
    }
    Tensor::new(out, x.shape.clone())
}
