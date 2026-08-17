//! SOP §4 rung 1 — every op at every width RFdiffusion2 uses.
//!
//! Tolerances, per the SOP's rung-1 row:
//!   * reductions (linear, layernorm, softmax): ~sqrt(K)·eps
//!   * gathers (embedding): **exactly 0**
//!
//! Widths come from the checkpoint's config, not from guesswork — see the
//! header of `python/gen_op_fixtures.py`.

use rfd2::ops;
use rfd2::parity;
use rfd2::tensor::Tensor;
use rfd2::weights::Weights;

fn fixture() -> Weights {
    let root = env!("CARGO_MANIFEST_DIR");
    let path = format!("{root}/../fixtures/ops/ops.safetensors");
    Weights::open(&path)
        .unwrap_or_else(|e| panic!("open {path}: {e}\nrun python/gen_op_fixtures.py first"))
}

const LINEAR_WIDTHS: [(usize, usize); 15] = [
    (256, 256), (192, 192), (256, 256), (192, 192), (320, 64), (80, 64),
    (114, 64), (64, 64), (64, 32), (32, 32), (32, 576), (32, 3328), (65, 32),
    (192, 37), (64, 1),
];
const ROWS: [usize; 3] = [1, 7, 150];

/// sqrt(K) * eps, the accumulation-noise budget for a K-long reduction.
fn reduction_tol(k: usize) -> f32 {
    (k as f32).sqrt() * f32::EPSILON * 8.0
}

#[test]
fn linear_at_every_width() {
    let f = fixture();
    let mut worst = 0.0f32;
    let mut worst_tag = String::new();
    let mut n = 0usize;

    for &(din, dout) in LINEAR_WIDTHS.iter() {
        for rows in ROWS {
            let tag = format!("lin_{din}x{dout}_r{rows}");
            if !f.has(&format!("{tag}_x")) {
                continue;
            }
            let x = f.get(&format!("{tag}_x"));
            let w = f.get(&format!("{tag}_w"));
            let b = f.get(&format!("{tag}_b"));
            let want = f.get(&format!("{tag}_y"));

            let got = ops::linear(&x, &w, Some(&b));
            assert_eq!(got.shape, want.shape, "{tag}: shape");
            let s = parity::compare(&got.data, &want.data);
            let tol = reduction_tol(din);
            assert!(
                s.max_abs <= tol,
                "{tag}: max|Δ| {:.3e} > tol {:.3e} (cos {:.12})",
                s.max_abs, tol, s.cosine
            );
            assert!(s.cosine > 0.999999, "{tag}: cosine {:.12}", s.cosine);
            if s.max_abs > worst {
                worst = s.max_abs;
                worst_tag = tag.clone();
            }
            n += 1;

            // and the no-bias path, which F.linear takes when bias is None
            let want_nb = f.get(&format!("{tag}_y_nobias"));
            let got_nb = ops::linear(&x, &w, None);
            let s = parity::compare(&got_nb.data, &want_nb.data);
            assert!(s.max_abs <= tol, "{tag} (no bias): max|Δ| {:.3e}", s.max_abs);
        }
    }
    println!("linear: {n} cases, worst max|Δ| {worst:.3e} at {worst_tag}");
    assert!(n >= 30, "expected the full width sweep, ran {n}");
}

#[test]
fn layer_norm_at_every_width() {
    let f = fixture();
    let mut worst = 0.0f32;
    for c in [256usize, 192, 128, 64, 32] {
        for rows in ROWS {
            let tag = format!("ln_{c}_r{rows}");
            let x = f.get(&format!("{tag}_x"));
            let w = f.get(&format!("{tag}_w"));
            let b = f.get(&format!("{tag}_b"));
            let want = f.get(&format!("{tag}_y"));
            let got = ops::layer_norm(&x, &w, &b, 1e-5);
            let s = parity::compare(&got.data, &want.data);
            let tol = reduction_tol(c) * 4.0; // mean + var + rsqrt
            assert!(s.max_abs <= tol, "{tag}: max|Δ| {:.3e} > {:.3e}", s.max_abs, tol);
            worst = worst.max(s.max_abs);
        }
    }
    println!("layer_norm: worst max|Δ| {worst:.3e}");
}

#[test]
fn softmax_at_every_width() {
    let f = fixture();
    let mut worst = 0.0f32;
    for c in [37usize, 64, 128, 150, 192, 256] {
        for rows in ROWS {
            let tag = format!("sm_{c}_r{rows}");
            let x = f.get(&format!("{tag}_x"));
            let want = f.get(&format!("{tag}_y"));
            let got = ops::softmax_last(&x);
            let s = parity::compare(&got.data, &want.data);
            assert!(s.max_abs <= 1e-6, "{tag}: max|Δ| {:.3e}", s.max_abs);
            worst = worst.max(s.max_abs);
        }
    }
    println!("softmax: worst max|Δ| {worst:.3e}");
}

/// Pointwise functions are pure, so the SOP budget is <= 1 ULP (§5.3), not a
/// reduction tolerance. ELU's negative branch is the interesting one: ATen uses
/// `expm1`, and `alpha * (exp(x) - 1)` would be visibly wrong near zero.
#[test]
fn activations_within_one_ulp() {
    let f = fixture();
    let x = f.get("act_x");

    let want = f.get("act_relu");
    let got = ops::relu(&x);
    for (i, (g, w)) in got.data.iter().zip(&want.data).enumerate() {
        assert_eq!(g.to_bits(), w.to_bits(), "relu[{i}] x={}", x.data[i]);
    }
    println!("relu: {} values bit-identical", got.data.len());

    let want = f.get("act_elu_1.0");
    let got = ops::elu(&x, 1.0);
    // The negative branch is `exp(x) - 1` with exp(x) in (0, 1]. torch's exp is
    // SLEEF u10 (<=1 ULP), not correctly rounded, so a few values sit one bit
    // away in exp -- and because subtracting 1 moves the result down by one or
    // more binades, that single bit shows up as 2, 4, ... ULP *of the output*.
    // The invariant that is actually stable is therefore stated on exp itself:
    //
    //     |got - want|  <=  1 ULP of exp(x)  <=  2^-24
    //
    // SOP §5.3: bound the last bit of a transcendental and record it; do not
    // chase it. Reproducing torch exactly here would mean porting SLEEF's expf.
    const ULP_OF_EXP: f32 = 1.0 / (1u32 << 24) as f32; // exp(x) <= 1 on this branch
    let mut n_off = 0usize;
    let mut max_abs = 0.0f32;
    let mut worst_x = 0.0f32;
    for (i, (g, w)) in got.data.iter().zip(&want.data).enumerate() {
        // Signed zeros and the identity branch are pure sign/copy, not
        // transcendental rounding, so they must be exact to the bit.
        if x.data[i] >= 0.0 {
            assert_eq!(
                g.to_bits(), w.to_bits(),
                "elu[{i}] x={:e}: non-negative branch must be bit-exact", x.data[i]
            );
            continue;
        }
        let d = (g - w).abs();
        assert!(
            d <= ULP_OF_EXP,
            "elu[{i}] x={:e}: |Δ| {d:e} > 1 ULP of exp ({ULP_OF_EXP:e})",
            x.data[i]
        );
        if g.to_bits() != w.to_bits() {
            n_off += 1;
        }
        if d > max_abs {
            max_abs = d;
            worst_x = x.data[i];
        }
    }
    println!(
        "elu: {} values, {} bit-identical, {n_off} within 1 ULP of exp \
(max |Δ| {max_abs:e} at x={worst_x:e})",
        got.data.len(),
        got.data.len() - n_off
    );
    // Measured: 65 / 6009 = 1.08 % of values sit 1 ULP of exp away. That is the
    // normal density for an fp32 transcendental (the ProteinMPNN port measured
    // 0.56 % for PyTorch's sqrt). The guard exists to catch a *systematic*
    // divergence -- a wrong constant or the wrong branch -- which would put the
    // rate near 100 %, not near 1 %.
    assert!(
        n_off * 20 <= got.data.len(),
        "elu: {n_off}/{} values disagree -- systematic, not SLEEF rounding",
        got.data.len()
    );
}

/// Gathers must be exact — a nonzero difference here is always a real bug
/// (wrong index, wrong stride), never rounding.
#[test]
fn embedding_is_exact() {
    let f = fixture();
    for (vocab, dim) in [(80usize, 256usize), (83, 64), (164, 256)] {
        let tag = format!("emb_{vocab}x{dim}");
        let table = f.get(&format!("{tag}_w"));
        let (idx, _) = f.get_i64(&format!("{tag}_idx"));
        let want = f.get(&format!("{tag}_y"));

        let mut out = Vec::with_capacity(idx.len() * dim);
        for &i in &idx {
            let s = i as usize * dim;
            out.extend_from_slice(&table.data[s..s + dim]);
        }
        let got = Tensor::new(out, vec![idx.len(), dim]);
        assert_eq!(got.shape, want.shape, "{tag}: shape");
        for (i, (g, w)) in got.data.iter().zip(&want.data).enumerate() {
            assert_eq!(g.to_bits(), w.to_bits(), "{tag}[{i}]");
        }
        println!("embedding {tag}: {} values exact", got.data.len());
    }
}

/// **Bit-exact mode.** See `docs/BITEXACT.md`.
///
/// Stock PyTorch's fp32 GEMM cannot be reproduced by choosing a reduction order
/// — `python/probe_gemm_order.py` measured the best candidate at 99.1 % (small
/// K) and ~10 % (K >= 192), never 100 %. But accumulating in f64 and rounding to
/// f32 exactly once yields the *correctly rounded* dot product, which does not
/// depend on the summation order at all: `python/probe_f64_pinning.py` compared
/// four deliberately different f64 orders over 299 200 values and found zero
/// disagreements.
///
/// So when the reference is pinned to that convention, this rung's tolerance
/// stops being sqrt(K)·eps and becomes **exactly 0** — which is what makes an
/// end-to-end bit-exact port reachable.
#[test]
fn linear_bit_exact_under_f64_pinning() {
    let f = fixture();
    let mut n_values = 0usize;
    let mut n_cases = 0usize;
    for &(din, dout) in LINEAR_WIDTHS.iter() {
        for rows in ROWS {
            let tag = format!("lin_{din}x{dout}_r{rows}");
            if !f.has(&format!("{tag}_y_pinned")) {
                continue;
            }
            let x = f.get(&format!("{tag}_x"));
            let w = f.get(&format!("{tag}_w"));
            let b = f.get(&format!("{tag}_b"));

            let want = f.get(&format!("{tag}_y_pinned"));
            let got = ops::linear_f64(&x, &w, Some(&b));
            assert_eq!(got.shape, want.shape, "{tag}: shape");
            for (i, (g, wv)) in got.data.iter().zip(&want.data).enumerate() {
                assert_eq!(
                    g.to_bits(), wv.to_bits(),
                    "{tag}[{i}]: got {g:e} want {wv:e} -- f64 pinning must be exact"
                );
            }

            let want_nb = f.get(&format!("{tag}_y_pinned_nobias"));
            let got_nb = ops::linear_f64(&x, &w, None);
            for (i, (g, wv)) in got_nb.data.iter().zip(&want_nb.data).enumerate() {
                assert_eq!(g.to_bits(), wv.to_bits(), "{tag} (no bias)[{i}]");
            }

            n_values += got.data.len() + got_nb.data.len();
            n_cases += 1;
        }
    }
    println!("linear (f64-pinned): {n_cases} cases, {n_values} values BIT-IDENTICAL");
    assert!(n_cases >= 30, "expected the full width sweep, ran {n_cases}");
}

/// Bit-exact mode for the remaining reductions: layernorm and softmax.
/// Both involve an f64 sum whose order differs between torch and this port, and
/// a transcendental (`sqrt`, `exp`) evaluated in f64. If the f64-pinning
/// argument holds, all of that is absorbed by the single f32 rounding.
#[test]
fn layer_norm_and_softmax_bit_exact_under_f64_pinning() {
    let f = fixture();

    let mut n = 0usize;
    for c in [256usize, 192, 128, 64, 32] {
        for rows in ROWS {
            let tag = format!("ln_{c}_r{rows}");
            let x = f.get(&format!("{tag}_x"));
            let w = f.get(&format!("{tag}_w"));
            let b = f.get(&format!("{tag}_b"));
            let want = f.get(&format!("{tag}_y_pinned"));
            let got = ops::layer_norm_f64(&x, &w, &b, 1e-5);
            for (i, (g, wv)) in got.data.iter().zip(&want.data).enumerate() {
                assert_eq!(g.to_bits(), wv.to_bits(), "{tag}[{i}] got {g:e} want {wv:e}");
            }
            n += got.data.len();
        }
    }
    println!("layer_norm (f64-pinned): {n} values BIT-IDENTICAL");

    let mut n = 0usize;
    for c in [37usize, 64, 128, 150, 192, 256] {
        for rows in ROWS {
            let tag = format!("sm_{c}_r{rows}");
            let x = f.get(&format!("{tag}_x"));
            let want = f.get(&format!("{tag}_y_pinned"));
            let got = ops::softmax_last_f64(&x);
            for (i, (g, wv)) in got.data.iter().zip(&want.data).enumerate() {
                assert_eq!(g.to_bits(), wv.to_bits(), "{tag}[{i}] got {g:e} want {wv:e}");
            }
            n += got.data.len();
        }
    }
    println!("softmax (f64-pinned): {n} values BIT-IDENTICAL");
}
