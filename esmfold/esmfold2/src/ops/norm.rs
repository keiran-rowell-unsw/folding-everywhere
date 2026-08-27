//! LayerNorm and RMSNorm over the last axis, matching PyTorch's biased-variance
//! formulation. Two-pass fp32 (mean then variance) in ascending order.

use crate::tensor::Tensor;
#[cfg(feature = "native")]
use rayon::prelude::*;

/// finfo(float32).eps == 2^-23 (the default eps for F.rms_norm when eps=None).
pub const RMS_DEFAULT_EPS: f32 = 1.1920928955078125e-07;
/// nn.LayerNorm default eps.
pub const LN_DEFAULT_EPS: f32 = 1e-5;

/// LayerNorm over the last axis: (x-mean)/sqrt(var+eps) * weight + bias.
/// `weight`/`bias` are length-`C` (last dim). `bias` optional (bias=False norms).
pub fn layernorm(x: &Tensor, weight: &[f32], bias: Option<&[f32]>, eps: f32) -> Tensor {
    let c = x.last();
    assert_eq!(weight.len(), c);
    if let Some(b) = bias { assert_eq!(b.len(), c); }
    let rows = x.rows();
    let mut out = vec![0.0f32; rows * c];
    let process = |r: usize, orow: &mut [f32]| {
        let xr = &x.data[r * c..r * c + c];
        let mut mean = 0.0f32;
        for &v in xr { mean += v; }
        mean /= c as f32;
        let mut var = 0.0f32;
        for &v in xr { let d = v - mean; var += d * d; }
        var /= c as f32;
        let inv = 1.0f32 / (var + eps).sqrt();
        for j in 0..c {
            let normed = (xr[j] - mean) * inv;
            orow[j] = normed * weight[j] + bias.map(|b| b[j]).unwrap_or(0.0);
        }
    };
    #[cfg(feature = "native")]
    out.par_chunks_mut(c).enumerate().for_each(|(r, orow)| process(r, orow));
    #[cfg(not(feature = "native"))]
    out.chunks_mut(c).enumerate().for_each(|(r, orow)| process(r, orow));
    Tensor::new(out, x.shape.clone())
}

/// RMSNorm over the last axis: x / sqrt(mean(x^2)+eps) * weight.
/// `weight=None` means pure normalization (no affine), as in `qk_norm`.
pub fn rmsnorm(x: &Tensor, weight: Option<&[f32]>, eps: f32) -> Tensor {
    let c = x.last();
    if let Some(w) = weight { assert_eq!(w.len(), c); }
    let rows = x.rows();
    let mut out = vec![0.0f32; rows * c];
    let process = |r: usize, orow: &mut [f32]| {
        let xr = &x.data[r * c..r * c + c];
        let mut ms = 0.0f32;
        for &v in xr { ms += v * v; }
        ms /= c as f32;
        let inv = 1.0f32 / (ms + eps).sqrt();
        for j in 0..c {
            let normed = xr[j] * inv;
            orow[j] = match weight { Some(w) => normed * w[j], None => normed };
        }
    };
    #[cfg(feature = "native")]
    out.par_chunks_mut(c).enumerate().for_each(|(r, orow)| process(r, orow));
    #[cfg(not(feature = "native"))]
    out.chunks_mut(c).enumerate().for_each(|(r, orow)| process(r, orow));
    Tensor::new(out, x.shape.clone())
}
