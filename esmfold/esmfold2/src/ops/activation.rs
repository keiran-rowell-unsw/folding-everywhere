//! Activations and softmax. Element-wise functions match PyTorch's definitions.

use crate::tensor::Tensor;

/// SiLU / swish: x * sigmoid(x).
#[inline]
pub fn silu_scalar(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

pub fn silu(t: &Tensor) -> Tensor {
    Tensor::new(t.data.iter().map(|&x| silu_scalar(x)).collect(), t.shape.clone())
}

#[inline]
pub fn sigmoid_scalar(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

pub fn sigmoid(t: &Tensor) -> Tensor {
    Tensor::new(t.data.iter().map(|&x| sigmoid_scalar(x)).collect(), t.shape.clone())
}

/// Exact (erf-based) GELU, matching nn.GELU() default (approximate='none').
/// gelu(x) = x * 0.5 * (1 + erf(x / sqrt(2))).
#[inline]
pub fn gelu_scalar(x: f32) -> f32 {
    // libm::erff for the fp32 erf; torch CPU uses a similar erf.
    0.5 * x * (1.0 + libm::erff(x * std::f32::consts::FRAC_1_SQRT_2))
}

pub fn gelu(t: &Tensor) -> Tensor {
    Tensor::new(t.data.iter().map(|&x| gelu_scalar(x)).collect(), t.shape.clone())
}

/// SwiGLU: split last dim in half -> silu(x1) * x2.  (gate first, value second)
pub fn swiglu_split(t: &Tensor) -> Tensor {
    let c = t.last();
    assert_eq!(c % 2, 0);
    let h = c / 2;
    let rows = t.rows();
    let mut out = vec![0.0f32; rows * h];
    for r in 0..rows {
        let xr = &t.data[r * c..r * c + c];
        let orow = &mut out[r * h..r * h + h];
        for j in 0..h {
            orow[j] = silu_scalar(xr[j]) * xr[h + j];
        }
    }
    let mut shape = t.shape.clone();
    let n = shape.len();
    shape[n - 1] = h;
    Tensor::new(out, shape)
}

/// In-place softmax over the last axis (subtract max, exp, normalize).
pub fn softmax_last(t: &Tensor) -> Tensor {
    let c = t.last();
    let rows = t.rows();
    let mut out = vec![0.0f32; rows * c];
    for r in 0..rows {
        let xr = &t.data[r * c..r * c + c];
        let orow = &mut out[r * c..r * c + c];
        let mut m = f32::NEG_INFINITY;
        for &v in xr { if v > m { m = v; } }
        let mut s = 0.0f32;
        for j in 0..c {
            let e = (xr[j] - m).exp();
            orow[j] = e;
            s += e;
        }
        let inv = 1.0 / s;
        for j in 0..c { orow[j] *= inv; }
    }
    Tensor::new(out, t.shape.clone())
}
