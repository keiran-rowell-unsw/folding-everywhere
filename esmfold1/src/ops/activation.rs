//! Pointwise activations. erf-GELU matches ESM's explicit `x*0.5*(1+erf(x/sqrt2))`
//! (NOT the tanh approximation). Folding MLPs use ReLU; gates use sigmoid; IPA
//! head weights use softplus.

use crate::tensor::Tensor;

const SQRT1_2: f32 = std::f32::consts::FRAC_1_SQRT_2; // 1/sqrt(2)

#[inline]
pub fn gelu_erf_scalar(x: f32) -> f32 {
    0.5 * x * (1.0 + libm::erff(x * SQRT1_2))
}

pub fn gelu_erf(x: &Tensor) -> Tensor {
    let data = x.data.iter().map(|&v| gelu_erf_scalar(v)).collect();
    Tensor::new(data, x.shape.clone())
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
    let data = x.data.iter().map(|&v| relu_scalar(v)).collect();
    Tensor::new(data, x.shape.clone())
}

#[inline]
pub fn sigmoid_scalar(x: f32) -> f32 {
    1.0 / (1.0 + libm::expf(-x))
}

pub fn sigmoid(x: &Tensor) -> Tensor {
    let data = x.data.iter().map(|&v| sigmoid_scalar(v)).collect();
    Tensor::new(data, x.shape.clone())
}

/// softplus(x) = log(1+exp(x)); matches nn.Softplus (threshold 20 -> identity).
#[inline]
pub fn softplus_scalar(x: f32) -> f32 {
    if x > 20.0 {
        x
    } else {
        libm::log1pf(libm::expf(x))
    }
}

pub fn softplus(x: &Tensor) -> Tensor {
    let data = x.data.iter().map(|&v| softplus_scalar(v)).collect();
    Tensor::new(data, x.shape.clone())
}
