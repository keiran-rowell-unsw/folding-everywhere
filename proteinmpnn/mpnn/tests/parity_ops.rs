//! Per-op parity vs PyTorch fixtures (`fixtures/ops/*.safetensors`).
//!
//! Tolerances scale with the reduction width: a K-wide fp32 dot product has
//! ~sqrt(K)*eps expected disagreement with a differently-blocked GEMM, so wide
//! layers get looser bounds. Ops with no reduction must be bit-exact.

use proteinmpnn::ops;
use proteinmpnn::parity::compare;
use proteinmpnn::tensor::Tensor;
use proteinmpnn::weights::Weights;

fn fx(name: &str) -> Weights {
    let p = format!("{}/../fixtures/ops/{}.safetensors", env!("CARGO_MANIFEST_DIR"), name);
    Weights::open(&p).unwrap_or_else(|e| panic!("open {p}: {e}\nrun python/gen_op_fixtures.py"))
}

fn check(label: &str, got: &Tensor, want: &Tensor, max_abs: f32) {
    assert_eq!(got.shape, want.shape, "{label} shape");
    let s = compare(&got.data, &want.data);
    println!("{label:22} {}", s.summary());
    assert!(!s.any_nan, "{label}: NaN");
    assert!(s.max_abs <= max_abs, "{label}: max_abs {:.3e} > {:.3e}", s.max_abs, max_abs);
}

#[test]
fn op_linear_all_widths() {
    // (fixture, tolerance) — tolerance grows with K.
    for (name, tol) in [
        ("linear_66x16", 1e-6f32),
        ("linear_384x128", 1e-5),
        ("linear_416x128", 1e-5),
        ("linear_512x128", 1e-5),
        ("linear_128x512", 1e-5),
        ("linear_128x21", 1e-5),
    ] {
        let f = fx(name);
        let (x, w, b, want) = (f.get("x"), f.get("w"), f.get("b"), f.get("y"));
        check(name, &ops::linear(&x, &w, Some(&b)), &want, tol);

        // f64 accumulation should be at least as close: if it is much closer,
        // the fp32 gap is accumulation order rather than a wrong formula.
        let yf = ops::linear_f64(&x, &w, Some(&b));
        let s = compare(&yf.data, &want.data);
        println!("  {name} (f64-acc)     {}", s.summary());
    }
}

#[test]
fn op_layer_norm() {
    for name in ["layernorm_128", "layernorm_512"] {
        let f = fx(name);
        let y = ops::layer_norm(&f.get("x"), &f.get("w"), &f.get("b"), 1e-5);
        // Welford/cascade moments over C vs our two-pass fp32 mean+var.
        check(name, &y, &f.get("y"), 5e-6);
    }
}

#[test]
fn op_gelu() {
    let f = fx("gelu");
    // Inputs run to +/-20, where 1 ULP is already 1.9e-6; libm erff vs SLEEF
    // erff differ below that.
    check("gelu", &ops::gelu(&f.get("x")), &f.get("y"), 2e-6);
}

#[test]
fn op_softmax_and_log_softmax() {
    let f = fx("softmax_last");
    check("softmax_last", &ops::softmax_last(&f.get("x")), &f.get("y"), 1e-6);
    let f = fx("log_softmax_last");
    check("log_softmax_last", &ops::log_softmax_last(&f.get("x")), &f.get("y"), 2e-6);
}

/// A gather — must be bit-exact.
#[test]
fn op_embedding() {
    let f = fx("embedding");
    let w = f.get("w");
    let (ids, _) = f.get_i64("ids");
    let y = ops::embedding(&ids, &w, &[ids.len(), w.shape[1]]);
    check("embedding", &y, &f.get("y"), 0.0);
}

/// `sum(x, dim=-2) / 30`, the message pooling in both layer types.
#[test]
fn op_sum_neighbors() {
    let f = fx("sum_neighbors");
    let x = f.get("x");
    let (l, k, c) = (x.shape[0], x.shape[1], x.shape[2]);
    let mut out = vec![0.0f32; l * c];
    for i in 0..l {
        let o = &mut out[i * c..i * c + c];
        for t in 0..k {
            let row = &x.data[(i * k + t) * c..(i * k + t) * c + c];
            for ci in 0..c {
                o[ci] += row[ci];
            }
        }
        for v in o.iter_mut() {
            *v /= 30.0;
        }
    }
    check("sum_neighbors", &Tensor::new(out, vec![l, c]), &f.get("y"), 1e-6);
}
