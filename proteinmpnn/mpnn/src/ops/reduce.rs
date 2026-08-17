//! LayerNorm, softmax and log-softmax over the last dimension. fp32, pinned order.

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
