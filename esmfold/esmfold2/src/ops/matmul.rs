//! Matmul / Linear with a pinned accumulation order (deterministic, thread-count
//! independent). Parallelism is over OUTPUT ROWS; each output element is a
//! sequential 8-lane fp32 fold. Mirrors the validated v1 kernels.

use crate::tensor::Tensor;
#[cfg(feature = "native")]
use rayon::prelude::*;

/// Vectorizable fp32 dot product with a fixed 8-lane partial-sum order.
#[inline]
pub fn dot8(a: &[f32], b: &[f32], k: usize) -> f32 {
    let mut acc = [0.0f32; 8];
    let nchunks = k / 8;
    for c in 0..nchunks {
        let ai = &a[c * 8..c * 8 + 8];
        let bi = &b[c * 8..c * 8 + 8];
        for l in 0..8 {
            acc[l] += ai[l] * bi[l];
        }
    }
    let mut s = ((acc[0] + acc[1]) + (acc[2] + acc[3])) + ((acc[4] + acc[5]) + (acc[6] + acc[7]));
    for kk in (nchunks * 8)..k {
        s += a[kk] * b[kk];
    }
    s
}

/// PyTorch-style Linear: x[..,K] @ w[O,K]^T (+ b[O]) -> [..,O].
/// `w` is row-major [O,K] (the torch `nn.Linear.weight` layout).
/// Uses a tuned f32 GEMM (matrixmultiply::sgemm) on native; falls back to
/// sequential dot products on wasm.
pub fn linear(x: &Tensor, w: &Tensor, b: Option<&Tensor>) -> Tensor {
    let k = x.last();
    let m = x.rows();
    assert_eq!(w.shape[1], k, "linear K mismatch x{:?} w{:?}", x.shape, w.shape);
    let o = w.shape[0];
    let mut out = vec![0.0f32; m * o];

    #[cfg(feature = "native")]
    {
        // C[m,o] = x[m,k] @ (w^T)[k,o].  w is [O,K] row-major -> B[k,o]=w[o,k]: rsb=1, csb=K.
        unsafe {
            matrixmultiply::sgemm(
                m, k, o, 1.0,
                x.data.as_ptr(), k as isize, 1,
                w.data.as_ptr(), 1, k as isize,
                0.0, out.as_mut_ptr(), o as isize, 1,
            );
        }
        if let Some(bb) = b {
            out.par_chunks_mut(o).for_each(|orow| { for oi in 0..o { orow[oi] += bb.data[oi]; } });
        }
    }

    #[cfg(not(feature = "native"))]
    {
        for i in 0..m {
            let xrow = &x.data[i * k..i * k + k];
            for oi in 0..o {
                let wrow = &w.data[oi * k..oi * k + k];
                out[i * o + oi] = dot8(xrow, wrow, k);
            }
        }
        if let Some(bb) = b {
            for orow in out.chunks_mut(o) {
                for oi in 0..o { orow[oi] += bb.data[oi]; }
            }
        }
    }

    let mut shape = x.shape.clone();
    let n = shape.len();
    shape[n - 1] = o;
    Tensor::new(out, shape)
}

/// General 2D matmul a[M,K] @ b[K,N] -> [M,N] using dot8 over an explicit
/// transpose-free path: we read b column-major by pre-transposing when needed.
/// Here `b` is [K,N] row-major; we accumulate a[i,:]·b[:,j].
pub fn matmul2d(a: &Tensor, b: &Tensor) -> Tensor {
    assert_eq!(a.ndim(), 2);
    assert_eq!(b.ndim(), 2);
    let (m, k) = (a.shape[0], a.shape[1]);
    let (k2, n) = (b.shape[0], b.shape[1]);
    assert_eq!(k, k2);
    let ad = &a.data;
    let bd = &b.data;
    let mut out = vec![0.0f32; m * n];

    #[cfg(feature = "native")]
    out.par_chunks_mut(n).enumerate().for_each(|(i, orow)| {
        let arow = &ad[i * k..i * k + k];
        for kk in 0..k {
            let aik = arow[kk];
            let brow = &bd[kk * n..kk * n + n];
            for j in 0..n { orow[j] += aik * brow[j]; }
        }
    });

    #[cfg(not(feature = "native"))]
    for (i, orow) in out.chunks_mut(n).enumerate() {
        let arow = &ad[i * k..i * k + k];
        for kk in 0..k {
            let aik = arow[kk];
            let brow = &bd[kk * n..kk * n + n];
            for j in 0..n { orow[j] += aik * brow[j]; }
        }
    }

    Tensor::new(out, vec![m, n])
}

/// Linear with **f64 accumulation** via a tuned GEMM (matrixmultiply::dgemm) on
/// native, or sequential f64 dot products on wasm. Rounded to f32 at the end.
/// Used for chaos-sensitive pair trunk / parcae / diffusion-pair ops.
pub fn linear_f64(x: &Tensor, w: &Tensor, b: Option<&Tensor>) -> Tensor {
    let k = x.last();
    let m = x.rows();
    let o = w.shape[0];
    let mut cf = vec![0.0f64; m * o];

    #[cfg(feature = "native")]
    {
        let xf: Vec<f64> = x.data.iter().map(|&v| v as f64).collect();
        let wf: Vec<f64> = w.data.iter().map(|&v| v as f64).collect();
        unsafe {
            matrixmultiply::dgemm(
                m, k, o, 1.0,
                xf.as_ptr(), k as isize, 1,
                wf.as_ptr(), 1, k as isize,
                0.0, cf.as_mut_ptr(), o as isize, 1,
            );
        }
    }

    #[cfg(not(feature = "native"))]
    {
        for i in 0..m {
            let xrow = &x.data[i * k..i * k + k];
            for oi in 0..o {
                let wrow = &w.data[oi * k..oi * k + k];
                let mut acc = 0.0f64;
                for kk in 0..k { acc += xrow[kk] as f64 * wrow[kk] as f64; }
                cf[i * o + oi] = acc;
            }
        }
    }

    if let Some(bb) = b {
        for row in 0..m { for oi in 0..o { cf[row * o + oi] += bb.data[oi] as f64; } }
    }
    let out: Vec<f32> = cf.iter().map(|&v| v as f32).collect();
    let mut shape = x.shape.clone();
    let n = shape.len();
    shape[n - 1] = o;
    Tensor::new(out, shape)
}
