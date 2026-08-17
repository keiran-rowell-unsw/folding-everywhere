//! Pointwise activations.
//!
//! ProteinMPNN uses `torch.nn.GELU()` with the default `approximate='none'`,
//! i.e. the exact erf form `x * 0.5 * (1 + erf(x/sqrt(2)))` — NOT the tanh
//! approximation.

use crate::tensor::Tensor;

const SQRT1_2: f32 = std::f32::consts::FRAC_1_SQRT_2;

#[inline]
pub fn gelu_erf_scalar(x: f32) -> f32 {
    0.5 * x * (1.0 + libm::erff(x * SQRT1_2))
}

pub fn gelu(x: &Tensor) -> Tensor {
    Tensor::new(x.data.iter().map(|&v| gelu_erf_scalar(v)).collect(), x.shape.clone())
}

/// In-place GELU — avoids one full-size allocation in the hot message MLPs.
pub fn gelu_(x: &mut Tensor) {
    for v in x.data.iter_mut() {
        *v = gelu_erf_scalar(*v);
    }
}

#[inline]
pub fn relu_scalar(x: f32) -> f32 {
    if x > 0.0 {
        x
    } else {
        0.0
    }
}

pub fn relu(x: &Tensor) -> Tensor {
    Tensor::new(x.data.iter().map(|&v| relu_scalar(v)).collect(), x.shape.clone())
}
