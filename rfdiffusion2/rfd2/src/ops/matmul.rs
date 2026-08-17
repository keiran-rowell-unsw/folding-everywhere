//! Linear / matmul with a pinned accumulation order.
//!
//! Parallelism (rayon) is over OUTPUT ROWS only; the K-reduction for any single
//! output element is always a sequential fp32 fold in ascending k, so results are
//! independent of thread count -> deterministic.
//!
//! `linear_f64` accumulates in f64 as a diagnostic: it lets a test decide whether
//! a divergence from PyTorch is fp32-accumulation noise or a real bug.

use crate::ops::acc::Acc;
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

/// The number of independent f64 accumulator chains a dot product is split
/// into. Four, because LLVM emits 256-bit vectors for this target even with
/// AVX-512 available (`prefer-256-bit`), so four f64 lanes is exactly one
/// register and an `RBLK x 4` tile keeps its accumulators resident.
///
/// **Why this is allowed to change the summation order.** `docs/BITEXACT.md`
/// measures it: four deliberately different f64 orders over 299 200 values gave
/// zero disagreements after narrowing to fp32, because an f64 rounding error is
/// ~9 orders of magnitude below an fp32 ULP. Order-independence is the whole
/// point of accumulating in double — so the fast kernel and the naive one are
/// interchangeable, and `tests/parity_ops.rs` is the standing proof.
const LANES: usize = 4;

/// One f64 dot product over fp32 inputs, split into `LANES` independent chains.
///
/// The naive single-accumulator loop is *latency*-bound, not throughput-bound:
/// each `acc += a*b` waits on the previous add, so it retires one FMA every
/// ~4 cycles regardless of how many execution ports are idle. Measured at the
/// model's real shapes that was 4.1-4.5 GFLOP/s across 4 cores.
#[inline]
pub fn dot_f64(a: &[f32], b: &[f32], k: usize) -> f64 {
    let mut acc = [Acc::new(); LANES];
    let nch = k / LANES;
    for c in 0..nch {
        let ai = &a[c * LANES..c * LANES + LANES];
        let bi = &b[c * LANES..c * LANES + LANES];
        for l in 0..LANES {
            acc[l].add(ai[l] as f64 * bi[l] as f64);
        }
    }
    let mut s = Acc::new();
    for a in acc {
        s.merge(a);
    }
    for kk in (nch * LANES)..k {
        s.add(a[kk] as f64 * b[kk] as f64);
    }
    s.get()
}

/// Four output columns sharing one pass over the input row, so `x` is read once
/// per four weight rows instead of once per row.
#[inline]
fn dot_f64x4(a: &[f32], w: &[f32], k: usize) -> [f64; 4] {
    let mut acc = [[Acc::new(); LANES]; 4];
    let nch = k / LANES;
    for c in 0..nch {
        let ai = &a[c * LANES..c * LANES + LANES];
        for (r, accr) in acc.iter_mut().enumerate() {
            let bi = &w[r * k + c * LANES..r * k + c * LANES + LANES];
            for l in 0..LANES {
                accr[l].add(ai[l] as f64 * bi[l] as f64);
            }
        }
    }
    let mut out = [0.0f64; 4];
    for (r, o) in out.iter_mut().enumerate() {
        let mut s = Acc::new();
        for a in acc[r] {
            s.merge(a);
        }
        for kk in (nch * LANES)..k {
            s.add(a[kk] as f64 * w[r * k + kk] as f64);
        }
        *o = s.get();
    }
    out
}

/// Rows per register tile. With `LANES = 8` and four output columns this holds
/// `RBLK * 4` accumulator vectors live — 16 AVX-512 registers of the 32
/// available, leaving room for the operands.
const RBLK: usize = 4;

/// The `RBLK x 4` tile with both bounds constant, so it unrolls and the weight
/// vectors are loaded once per four input rows instead of once per row.
///
/// That reuse is the point. The one-row-at-a-time kernel streams the whole
/// weight matrix once per row: at `[5041, 192] x [192, 192]` that is 740 MB of
/// reads for 372 MFLOP, and the measurement said so — 8.2 GFLOP/s, right at
/// this machine's memory bandwidth rather than anywhere near its FMA rate.
#[inline]
fn tile_full(xd: &[f32], wd: &[f32], k: usize, nch: usize, xs: [usize; RBLK], ws: [usize; 4])
    -> [[f64; 4]; RBLK]
{
    let mut acc = [[[Acc::new(); LANES]; 4]; RBLK];
    for c in 0..nch {
        let off = c * LANES;
        for r in 0..RBLK {
            let ai = &xd[xs[r] + off..xs[r] + off + LANES];
            for j in 0..4 {
                let bi = &wd[ws[j] + off..ws[j] + off + LANES];
                for l in 0..LANES {
                    acc[r][j][l].add(ai[l] as f64 * bi[l] as f64);
                }
            }
        }
    }
    let mut out = [[0.0f64; 4]; RBLK];
    for r in 0..RBLK {
        for j in 0..4 {
            let mut s = Acc::new();
            for l in acc[r][j] {
                s.merge(l);
            }
            for kk in (nch * LANES)..k {
                s.add(xd[xs[r] + kk] as f64 * wd[ws[j] + kk] as f64);
            }
            out[r][j] = s.get();
        }
    }
    out
}

fn linear_f64_row(xrow: &[f32], wd: &[f32], k: usize, o: usize, bd: Option<&[f32]>, orow: &mut [f32]) {
    let mut oi = 0;
    while oi + 4 <= o {
        let r = dot_f64x4(xrow, &wd[oi * k..], k);
        for j in 0..4 {
            let bias = bd.map(|bb| bb[oi + j] as f64).unwrap_or(0.0);
            orow[oi + j] = (r[j] + bias) as f32;
        }
        oi += 4;
    }
    while oi < o {
        let s = dot_f64(xrow, &wd[oi * k..oi * k + k], k);
        let bias = bd.map(|bb| bb[oi] as f64).unwrap_or(0.0);
        orow[oi] = (s + bias) as f32;
        oi += 1;
    }
}

/// A weight matrix already widened to f64.
///
/// The kernel converts every operand to f64 before multiplying, and on this
/// machine `vcvtps2pd` issues on a single port — so converting the *weights*
/// on every call costs as much as the arithmetic does. Weights never change,
/// so they are widened once at load time and the inner loop reads them
/// directly. Inputs still convert per call, but a `RBLK x 4` tile amortises
/// each input conversion over four output columns.
pub struct WeightsF64 {
    pub w: Vec<f64>,
    pub b: Option<Vec<f64>>,
    pub o: usize,
    pub k: usize,
}

impl WeightsF64 {
    pub fn new(w: &Tensor, b: Option<&Tensor>) -> Self {
        WeightsF64 {
            w: w.data.iter().map(|v| *v as f64).collect(),
            b: b.map(|t| t.data.iter().map(|v| *v as f64).collect()),
            o: w.shape[0],
            k: w.shape[1],
        }
    }
}

#[inline]
fn dot_pre(a: &[f32], b: &[f64], k: usize) -> f64 {
    let mut acc = [Acc::new(); LANES];
    let nch = k / LANES;
    for c in 0..nch {
        let ai = &a[c * LANES..c * LANES + LANES];
        let bi = &b[c * LANES..c * LANES + LANES];
        for l in 0..LANES {
            acc[l].add(ai[l] as f64 * bi[l]);
        }
    }
    let mut s = Acc::new();
    for a in acc {
        s.merge(a);
    }
    for kk in (nch * LANES)..k {
        s.add(a[kk] as f64 * b[kk]);
    }
    s.get()
}

#[inline]
fn tile_pre(xd: &[f32], wd: &[f64], k: usize, nch: usize, xs: [usize; RBLK], ws: [usize; 4])
    -> [[f64; 4]; RBLK]
{
    let mut acc = [[[Acc::new(); LANES]; 4]; RBLK];
    for c in 0..nch {
        let off = c * LANES;
        let mut xv = [[0.0f64; LANES]; RBLK];
        for r in 0..RBLK {
            let ai = &xd[xs[r] + off..xs[r] + off + LANES];
            for l in 0..LANES {
                xv[r][l] = ai[l] as f64;
            }
        }
        for j in 0..4 {
            let bi = &wd[ws[j] + off..ws[j] + off + LANES];
            for r in 0..RBLK {
                for l in 0..LANES {
                    acc[r][j][l].add(xv[r][l] * bi[l]);
                }
            }
        }
    }
    let mut out = [[0.0f64; 4]; RBLK];
    for r in 0..RBLK {
        for j in 0..4 {
            let mut s = Acc::new();
            for l in acc[r][j] {
                s.merge(l);
            }
            for kk in (nch * LANES)..k {
                s.add(xd[xs[r] + kk] as f64 * wd[ws[j] + kk]);
            }
            out[r][j] = s.get();
        }
    }
    out
}

/// `F.linear` under pinning, against pre-widened weights.
///
/// Parallelism is one rayon task per `RBLK` rows. A coarser row *tile* (128
/// rows, sweeping columns inside it to keep the weight slice in L1) was tried
/// and measured **slower** — 15.0 vs 17.1 GFLOP/s on the pair projection, and
/// 8.0 vs 15.5 on the 71-row MSA projection, where a 128-row tile leaves a
/// single task and no parallelism at all. So the kernel is not weight-bandwidth
/// bound the way the arithmetic suggested; recording the negative result here
/// so it is not retried.
pub fn linear_pre(x: &Tensor, w: &WeightsF64) -> Tensor {
    let (k, o) = (w.k, w.o);
    assert_eq!(x.last(), k, "linear K mismatch x{:?} K={k}", x.shape);
    let rows = x.numel() / k;
    let mut out = vec![0.0f32; rows * o];
    let xd = &x.data;
    let wd = &w.w;
    let bd = w.b.as_deref();
    let nch = k / LANES;

    // Blocking is pure scheduling: every output element still gets its own
    // `LANES`-way f64 sum in the same order, so the result does not depend on
    // `RBLK` or on the thread count. `tests/debug_blocks.rs` is the proof.
    let block = |base: usize, nrows: usize, obuf: &mut [f32]| {
        if nrows == RBLK {
            let xs: [usize; RBLK] = std::array::from_fn(|r| (base + r) * k);
            let mut oi = 0;
            while oi + 4 <= o {
                let ws: [usize; 4] = std::array::from_fn(|j| (oi + j) * k);
                let t = tile_pre(xd, wd, k, nch, xs, ws);
                for r in 0..RBLK {
                    for j in 0..4 {
                        let bias = bd.map(|bb| bb[oi + j]).unwrap_or(0.0);
                        obuf[r * o + oi + j] = (t[r][j] + bias) as f32;
                    }
                }
                oi += 4;
            }
            for r in 0..RBLK {
                for j in oi..o {
                    let s = dot_pre(&xd[xs[r]..xs[r] + k], &wd[j * k..j * k + k], k);
                    let bias = bd.map(|bb| bb[j]).unwrap_or(0.0);
                    obuf[r * o + j] = (s + bias) as f32;
                }
            }
        } else {
            for r in 0..nrows {
                let xr = &xd[(base + r) * k..(base + r) * k + k];
                for j in 0..o {
                    let s = dot_pre(xr, &wd[j * k..j * k + k], k);
                    let bias = bd.map(|bb| bb[j]).unwrap_or(0.0);
                    obuf[r * o + j] = (s + bias) as f32;
                }
            }
        }
    };

    if rows >= PAR_MIN_ROWS {
        out.par_chunks_mut(o * RBLK).enumerate().for_each(|(bi, obuf)| {
            block(bi * RBLK, obuf.len() / o, obuf);
        });
    } else {
        for (bi, obuf) in out.chunks_mut(o * RBLK).enumerate() {
            let n = obuf.len() / o;
            block(bi * RBLK, n, obuf);
        }
    }
    let mut shape = x.shape.clone();
    let n = shape.len();
    shape[n - 1] = o;
    Tensor::new(out, shape)
}

/// `F.linear` under pinning: accumulate in f64, add the bias in f64, round to
/// f32 exactly once.
pub fn linear_f64(x: &Tensor, w: &Tensor, b: Option<&Tensor>) -> Tensor {
    let k = x.last();
    let rows = x.numel() / k;
    assert_eq!(w.shape[1], k, "linear K mismatch x{:?} w{:?}", x.shape, w.shape);
    let o = w.shape[0];
    let mut out = vec![0.0f32; rows * o];
    let xd = &x.data;
    let wd = &w.data;
    let bd = b.map(|t| t.data.as_slice());
    let nch = k / LANES;

    // One rayon task per block of `RBLK` rows. Blocking is a pure scheduling
    // choice: every output element still gets its own `LANES`-way f64 sum in
    // the same order, so the result does not depend on `RBLK`, on the thread
    // count, or on where the blocks fall.
    let block = |base: usize, nrows: usize, obuf: &mut [f32]| {
        if nrows == RBLK {
            let xs: [usize; RBLK] = std::array::from_fn(|r| (base + r) * k);
            let mut oi = 0;
            while oi + 4 <= o {
                let ws: [usize; 4] = std::array::from_fn(|j| (oi + j) * k);
                let t = tile_full(xd, wd, k, nch, xs, ws);
                for r in 0..RBLK {
                    for j in 0..4 {
                        let bias = bd.map(|bb| bb[oi + j] as f64).unwrap_or(0.0);
                        obuf[r * o + oi + j] = (t[r][j] + bias) as f32;
                    }
                }
                oi += 4;
            }
            // the ragged output tail
            for r in 0..RBLK {
                for j in oi..o {
                    let s = dot_f64(&xd[xs[r]..xs[r] + k], &wd[j * k..j * k + k], k);
                    let bias = bd.map(|bb| bb[j] as f64).unwrap_or(0.0);
                    obuf[r * o + j] = (s + bias) as f32;
                }
            }
        } else {
            for r in 0..nrows {
                let row = base + r;
                linear_f64_row(&xd[row * k..row * k + k], wd, k, o, bd, &mut obuf[r * o..(r + 1) * o]);
            }
        }
    };

    if rows >= PAR_MIN_ROWS {
        out.par_chunks_mut(o * RBLK).enumerate().for_each(|(bi, obuf)| {
            block(bi * RBLK, obuf.len() / o, obuf);
        });
    } else {
        for (bi, obuf) in out.chunks_mut(o * RBLK).enumerate() {
            let n = obuf.len() / o;
            block(bi * RBLK, n, obuf);
        }
    }
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
