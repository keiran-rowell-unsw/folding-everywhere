//! Throughput of the primitives, at the shapes the real model uses.
//! Not a parity test — a stopwatch, so the optimisation pass has a baseline.
use rfd2::ops::{linear_f64, linear_pre, WeightsF64};
use rfd2::tensor::Tensor;
use std::time::Instant;

fn t(shape: &[usize], seed: u64) -> Tensor {
    let n: usize = shape.iter().product();
    let mut s = seed;
    let d: Vec<f32> = (0..n)
        .map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((s >> 33) as f32 / u32::MAX as f32) - 0.5
        })
        .collect();
    Tensor::new(d, shape.to_vec())
}

fn bench(name: &str, rows: usize, k: usize, o: usize, iters: usize) {
    let x = t(&[rows, k], 1);
    let w = t(&[o, k], 2);
    let b = t(&[o], 3);
    let pre = WeightsF64::new(&w, Some(&b));
    let flops = 2.0 * rows as f64 * k as f64 * o as f64 * iters as f64;
    let mut sink = 0.0f32;
    let t0 = Instant::now();
    for _ in 0..iters {
        sink += linear_f64(&x, &w, Some(&b)).data[0];
    }
    let s_naive = t0.elapsed().as_secs_f64();
    let t1 = Instant::now();
    for _ in 0..iters {
        sink += linear_pre(&x, &pre).data[0];
    }
    let s_pre = t1.elapsed().as_secs_f64();
    // both kernels must agree bit for bit, or the speed is meaningless
    let a = linear_f64(&x, &w, Some(&b));
    let c = linear_pre(&x, &pre);
    let bad = a.data.iter().zip(&c.data).filter(|(p, q)| p.to_bits() != q.to_bits()).count();
    println!(
        "{name:<28} rows={rows:<6} K={k:<4} O={o:<4}  f32-w {:5.2} GFLOP/s -> f64-w {:5.2} GFLOP/s  ({:.1}x)  mismatch {bad}  (sink {sink:.3})",
        flops / s_naive / 1e9,
        flops / s_pre / 1e9,
        s_naive / s_pre
    );
}

#[test]
fn linear_throughput() {
    let l = 71;
    bench("pair proj (L*L, 192->192)", l * l, 192, 192, 20);
    bench("pair tri (L*L, 192->32)", l * l, 192, 32, 40);
    bench("msa proj (L, 256->256)", l, 256, 256, 200);
    bench("se3 radial (E, 64->128)", l * l, 64, 128, 20);
    bench("attn qk (L*L, 192->6)", l * l, 192, 6, 100);
}

/// Both operands already f64 — but WITHOUT the output blocking, one row per
/// rayon task. It was written to test whether `vcvtps2pd` is what limits the
/// mixed kernel; the answer is no, and emphatically so: at 5.2 GFLOP/s it is
/// three times SLOWER than the mixed kernel that converts on the fly. What
/// governs is the `RBLK x 4` register tile, not the operand type. Kept as the
/// negative control.
#[test]
fn f64_unblocked_control() {
    let (rows, k, o) = (5041usize, 192usize, 192usize);
    let x: Vec<f64> = t(&[rows, k], 1).data.iter().map(|v| *v as f64).collect();
    let w: Vec<f64> = t(&[o, k], 2).data.iter().map(|v| *v as f64).collect();
    let mut out = vec![0.0f64; rows * o];
    let iters = 20;
    let t0 = Instant::now();
    for _ in 0..iters {
        use rayon::prelude::*;
        out.par_chunks_mut(o).enumerate().for_each(|(r, orow)| {
            let xr = &x[r * k..r * k + k];
            for j in 0..o {
                let wr = &w[j * k..j * k + k];
                let mut acc = [0.0f64; 4];
                for c in 0..k / 4 {
                    for l in 0..4 {
                        acc[l] += xr[c * 4 + l] * wr[c * 4 + l];
                    }
                }
                orow[j] = (acc[0] + acc[1]) + (acc[2] + acc[3]);
            }
        });
    }
    let secs = t0.elapsed().as_secs_f64();
    let flops = 2.0 * rows as f64 * k as f64 * o as f64 * iters as f64;
    println!("f64, unblocked (control)   rows={rows} K={k} O={o}  {:.2} GFLOP/s  (sink {:.3})",
             flops / secs / 1e9, out[0]);
}
