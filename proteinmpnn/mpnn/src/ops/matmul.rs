//! Linear / matmul with a pinned accumulation order.
//!
//! Parallelism (rayon) is over OUTPUT ROWS only; the K-reduction for any single
//! output element is always a sequential fp32 fold in ascending k, so results are
//! independent of thread count -> deterministic.
//!
//! `linear_f64` accumulates in f64 as a diagnostic: it lets a test decide whether
//! a divergence from PyTorch is fp32-accumulation noise or a real bug.

use crate::tensor::Tensor;
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

/// Four independent `dot8`s sharing one pass over `a`.
///
/// Each output element keeps *exactly* the accumulation order `dot8` uses, so
/// this is bit-identical to calling `dot8` four times — but the four chains are
/// independent, which is what lets the FMA units stay busy. With a single chain
/// the kernel is latency-bound at ~1 FMA per 4 cycles.
#[inline]
fn dot8x4(a: &[f32], w: &[f32], k: usize) -> [f32; 4] {
    let mut acc = [[0.0f32; 8]; 4];
    let nchunks = k / 8;
    for c in 0..nchunks {
        let ai = &a[c * 8..c * 8 + 8];
        for (r, accr) in acc.iter_mut().enumerate() {
            let bi = &w[r * k + c * 8..r * k + c * 8 + 8];
            for l in 0..8 {
                accr[l] += ai[l] * bi[l];
            }
        }
    }
    let mut out = [0.0f32; 4];
    for (r, o) in out.iter_mut().enumerate() {
        let ac = &acc[r];
        let mut s =
            ((ac[0] + ac[1]) + (ac[2] + ac[3])) + ((ac[4] + ac[5]) + (ac[6] + ac[7]));
        for kk in (nchunks * 8)..k {
            s += a[kk] * w[r * k + kk];
        }
        *o = s;
    }
    out
}

/// Below this many output rows, rayon's task overhead exceeds the work.
const PAR_MIN_ROWS: usize = 8;

fn linear_row(xrow: &[f32], wd: &[f32], k: usize, o: usize, bd: Option<&[f32]>, orow: &mut [f32]) {
    let mut oi = 0;
    while oi + 4 <= o {
        let r = dot8x4(xrow, &wd[oi * k..], k);
        orow[oi..oi + 4].copy_from_slice(&r);
        oi += 4;
    }
    while oi < o {
        orow[oi] = dot8(xrow, &wd[oi * k..oi * k + k], k);
        oi += 1;
    }
    if let Some(bb) = bd {
        for oi in 0..o {
            orow[oi] += bb[oi];
        }
    }
}

/// PyTorch `F.linear`: x[..., K] @ w[O, K]^T + b[O] -> [..., O].
pub fn linear(x: &Tensor, w: &Tensor, b: Option<&Tensor>) -> Tensor {
    let k = x.last();
    let rows = x.numel() / k;
    assert_eq!(w.shape[1], k, "linear K mismatch x{:?} w{:?}", x.shape, w.shape);
    let o = w.shape[0];
    if let Some(bb) = b {
        assert_eq!(bb.numel(), o);
    }
    let mut out = vec![0.0f32; rows * o];
    let xd = &x.data;
    let wd = &w.data;
    let bd = b.map(|t| t.data.as_slice());
    if rows >= PAR_MIN_ROWS {
        out.par_chunks_mut(o).enumerate().for_each(|(row, orow)| {
            linear_row(&xd[row * k..row * k + k], wd, k, o, bd, orow);
        });
    } else {
        for (row, orow) in out.chunks_mut(o).enumerate() {
            linear_row(&xd[row * k..row * k + k], wd, k, o, bd, orow);
        }
    }
    let mut shape = x.shape.clone();
    let n = shape.len();
    shape[n - 1] = o;
    Tensor::new(out, shape)
}

/// Diagnostic: `linear` with f64 accumulation, rounded to f32 at the end.
pub fn linear_f64(x: &Tensor, w: &Tensor, b: Option<&Tensor>) -> Tensor {
    let k = x.last();
    let rows = x.numel() / k;
    let o = w.shape[0];
    let mut out = vec![0.0f32; rows * o];
    let xd = &x.data;
    let wd = &w.data;
    let bd = b.map(|t| t.data.as_slice());
    out.par_chunks_mut(o).enumerate().for_each(|(row, orow)| {
        let xrow = &xd[row * k..row * k + k];
        for oi in 0..o {
            let wrow = &wd[oi * k..oi * k + k];
            let mut acc = 0.0f64;
            for kk in 0..k {
                acc += xrow[kk] as f64 * wrow[kk] as f64;
            }
            let bias = bd.map(|bb| bb[oi] as f64).unwrap_or(0.0);
            orow[oi] = (acc + bias) as f32;
        }
    });
    let mut shape = x.shape.clone();
    let n = shape.len();
    shape[n - 1] = o;
    Tensor::new(out, shape)
}

/// `nn.Embedding` lookup: rows of `w` [V, C] gathered by `ids` -> [..., C].
pub fn embedding(ids: &[i64], w: &Tensor, out_shape: &[usize]) -> Tensor {
    let c = w.shape[1];
    let mut out = vec![0.0f32; ids.len() * c];
    for (i, &id) in ids.iter().enumerate() {
        let r = id as usize;
        assert!(r < w.shape[0], "embedding id {r} out of range {}", w.shape[0]);
        out[i * c..i * c + c].copy_from_slice(&w.data[r * c..r * c + c]);
    }
    Tensor::new(out, out_shape.to_vec())
}
