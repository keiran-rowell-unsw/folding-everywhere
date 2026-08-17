//! Pointwise activations.
//!
//! RFdiffusion2's network uses **ReLU** (13 `F.relu` + 4 `nn.ReLU`) and **ELU**
//! (3 `nn.ELU`) — measured by grepping `rf2aa/model/` and the SE3 transformer.
//! It uses no GELU; `gelu` is retained from the ProteinMPNN port because it is
//! validated and costs nothing to keep.

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

/// ATen's `relu` is `clamp_min(x, 0)`, whose predicate is `x >= 0` — so it
/// **passes `-0.0` through unchanged** rather than normalising it to `+0.0`.
/// Measured: `torch.relu(-0.0)` has bits `0x80000000` at every tensor length
/// (scalar and vectorized kernels agree).
///
/// `x > 0.0` would be wrong only for `-0.0`, which `parity::compare` cannot see
/// (its ordered key maps both zeros to 0) — the strict bit comparison in
/// `tests/parity_ops.rs` is what catches it.
#[inline]
pub fn relu_scalar(x: f32) -> f32 {
    if x >= 0.0 {
        x
    } else {
        0.0
    }
}

pub fn relu(x: &Tensor) -> Tensor {
    Tensor::new(x.data.iter().map(|&v| relu_scalar(v)).collect(), x.shape.clone())
}

/// `nn.ELU(alpha)`: `x` for `x > 0`, else `(exp(x) - 1) * alpha`.
///
/// Three things here are measured, not assumed (torch 2.4.0+cpu):
///
/// 1. The negative branch is **`exp(x) - 1`, not `expm1(x)`**. At `x = -1e-8`
///    torch returns exactly `0.0` while `expm1` returns `-1e-8`.
/// 2. The `exp` must be **correctly rounded in fp32**. At `x = -1e-6` torch
///    returns `-1.013278961e-06`; libm's `expf` (as numpy uses) is 1 ULP high
///    there and yields `-9.536743164e-07` — a 6 % error, because subtracting 1
///    from a value just under 1 cancels catastrophically and promotes that one
///    bit into the leading digits. This is SOP §5.3 with an amplifier attached.
///    Computing `exp` in f64 and rounding once reproduces torch's result.
/// 3. `elu(-0.0)` is `+0.0` (bits `0x00000000`), which falls out of (1)
///    automatically: `exp(-0.0) - 1.0 == +0.0`. Note this differs from `relu`,
///    which preserves `-0.0`.
#[inline]
pub fn elu_scalar(x: f32, alpha: f32) -> f32 {
    if x > 0.0 {
        x
    } else {
        let e = (x as f64).exp() as f32;
        (e - 1.0) * alpha
    }
}

pub fn elu(x: &Tensor, alpha: f32) -> Tensor {
    Tensor::new(
        x.data.iter().map(|&v| elu_scalar(v, alpha)).collect(),
        x.shape.clone(),
    )
}

pub fn relu_(x: &mut Tensor) {
    for v in x.data.iter_mut() {
        *v = relu_scalar(*v);
    }
}
