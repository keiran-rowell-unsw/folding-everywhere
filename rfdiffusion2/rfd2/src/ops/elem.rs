//! Elementwise transcendentals, in the pinned (f64-then-round-once) convention.
//!
//! These are here rather than inlined because they are *not* exactly rounded in
//! fp32 ATen — the CPU kernels evaluate through SLEEF's vectorized `exp`/`tanh`,
//! so the last bit depends on the vector width the kernel happened to pick.
//! Computing in f64 and rounding once makes the result the correctly-rounded
//! fp32 value and therefore width- and thread-independent, which is the whole
//! convention (`docs/BITEXACT.md`). `python/pinned.py` applies the same
//! treatment to the reference, so both sides land on the same bits.
//!
//! `relu` and `elu` live in `activation.rs`; they were pinned in rung 1.

use rayon::prelude::*;
use crate::tensor::Tensor;

#[inline]
pub fn sigmoid_scalar(x: f32) -> f32 {
    (1.0f64 / (1.0f64 + (-(x as f64)).exp())) as f32
}

#[inline]
pub fn tanh_scalar(x: f32) -> f32 {
    (x as f64).tanh() as f32
}

#[inline]
pub fn asinh_scalar(x: f32) -> f32 {
    (x as f64).asinh() as f32
}

/// `torch.exp` under pinning.
#[inline]
pub fn exp_scalar(x: f32) -> f32 {
    (x as f64).exp() as f32
}

pub fn map<F: Fn(f32) -> f32>(t: &Tensor, f: F) -> Tensor {
    Tensor::new(t.data.iter().map(|&v| f(v)).collect(), t.shape.clone())
}

pub fn map_<F: Fn(f32) -> f32>(t: &mut Tensor, f: F) {
    for v in t.data.iter_mut() {
        *v = f(*v);
    }
}

pub fn sigmoid(t: &Tensor) -> Tensor {
    map(t, sigmoid_scalar)
}

pub fn sigmoid_(t: &mut Tensor) {
    map_(t, sigmoid_scalar);
}

/// Softmax over an arbitrary axis, f64 intermediates, one rounding.
///
/// The reference reaches `F.softmax(x, dim=-2)` in three places (triangle
/// attention, biased axial attention, `SequenceWeight` over dim 1), and the
/// non-last axis is exactly where a "reshape so it's the last dim" shortcut
/// would silently transpose the wrong pair of axes.
pub fn softmax_dim(x: &Tensor, dim: usize) -> Tensor {
    let nd = x.shape.len();
    assert!(dim < nd);
    let strides = x.strides();
    let n = x.shape[dim];
    let stride = strides[dim];
    // outer = product of dims before `dim`, inner = product of dims after
    let outer: usize = x.shape[..dim].iter().product();
    let inner: usize = x.shape[dim + 1..].iter().product();
    let mut out = vec![0.0f32; x.numel()];
    // Each `outer` slab is disjoint in `out`, and within it every lane touches a
    // distinct set of offsets, so this is a pure scheduling change.
    let slab = x.shape[dim] * inner;
    out.par_chunks_mut(slab).enumerate().for_each(|(o, oslab)| {
        let mut buf = vec![0.0f64; n];
        for i in 0..inner {
            let base = o * slab + i;
            let mut m = f64::NEG_INFINITY;
            for k in 0..n {
                let v = x.data[base + k * stride] as f64;
                if v > m {
                    m = v;
                }
            }
            let mut s = 0.0f64;
            for k in 0..n {
                let e = (x.data[base + k * stride] as f64 - m).exp();
                buf[k] = e;
                s += e;
            }
            for k in 0..n {
                oslab[i + k * stride] = (buf[k] / s) as f32;
            }
        }
    });
    let _ = outer;
    Tensor::new(out, x.shape.clone())
}
