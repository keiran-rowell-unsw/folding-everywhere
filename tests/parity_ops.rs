//! P1 per-op parity vs PyTorch fixtures (fixtures/ops/*.safetensors).

use esmfold::ops;
use esmfold::parity::compare;
use esmfold::tensor::Tensor;
use esmfold::weights::Weights;

fn fx(name: &str) -> Weights {
    let p = format!("{}/fixtures/ops/{}.safetensors", env!("CARGO_MANIFEST_DIR"), name);
    Weights::open(&p).unwrap_or_else(|e| panic!("open {p}: {e}"))
}

fn check(label: &str, got: &Tensor, want: &Tensor, max_abs: f32) {
    assert_eq!(got.shape, want.shape, "{label} shape");
    let s = compare(&got.data, &want.data);
    println!("{label:24} {}", s.summary());
    assert!(!s.any_nan, "{label}: NaN");
    assert!(s.max_abs <= max_abs, "{label}: max_abs {:.3e} > {:.3e}", s.max_abs, max_abs);
}

#[test]
fn op_linear_small() {
    let f = fx("linear_small");
    let y = ops::linear(&f.get("x"), &f.get("w"), Some(&f.get("b")));
    check("linear_small", &y, &f.get("y"), 1e-6);
}

#[test]
fn op_linear_bigk() {
    let f = fx("linear_bigK");
    let (x, w, b, want) = (f.get("x"), f.get("w"), f.get("b"), f.get("y"));
    let y = ops::linear(&x, &w, Some(&b));
    check("linear_bigK (f32)", &y, &want, 1e-3);
    let yf = ops::linear_f64(&x, &w, Some(&b));
    let s = compare(&yf.data, &want.data);
    println!("linear_bigK (f64-acc)   {}", s.summary());
}

#[test]
fn op_matmul_bigk() {
    let f = fx("matmul_bigK");
    let y = ops::matmul2d(&f.get("a"), &f.get("b"));
    check("matmul_bigK", &y, &f.get("y"), 1e-3);
}

#[test]
fn op_layernorm() {
    let f = fx("layernorm");
    let y = ops::layer_norm(&f.get("x"), &f.get("w"), &f.get("b"), 1e-5);
    check("layernorm", &y, &f.get("y"), 1e-5);
}

#[test]
fn op_gelu_erf() {
    let f = fx("gelu_erf");
    check("gelu_erf", &ops::gelu_erf(&f.get("x")), &f.get("y"), 1e-6);
}

#[test]
fn op_sigmoid() {
    let f = fx("sigmoid");
    check("sigmoid", &ops::sigmoid(&f.get("x")), &f.get("y"), 1e-6);
}

#[test]
fn op_softplus() {
    let f = fx("softplus");
    check("softplus", &ops::softplus(&f.get("x")), &f.get("y"), 1e-6);
}

#[test]
fn op_relu() {
    let f = fx("relu");
    check("relu", &ops::relu(&f.get("x")), &f.get("y"), 0.0);
}

#[test]
fn op_softmax_last() {
    let f = fx("softmax_last");
    check("softmax_last", &ops::softmax_last(&f.get("x")), &f.get("y"), 1e-6);
}

#[test]
fn op_rotary() {
    let f = fx("rotary");
    let (cos_w, sin_w) = (f.get("cos"), f.get("sin"));
    let (l, dim) = (cos_w.shape[0], cos_w.shape[1]);
    let (cos, sin) = ops::build_cos_sin(l, dim);
    // rotary is transcendental-limited (libm cos/sin vs torch): ~1e-6 fp32 noise
    check("rotary.cos", &Tensor::new(cos.clone(), vec![l, dim]), &cos_w, 2e-6);
    check("rotary.sin", &Tensor::new(sin.clone(), vec![l, dim]), &sin_w, 2e-6);

    // apply to q and k ([H, L, dim])
    let q = f.get("q");
    let (h, ll, dd) = (q.shape[0], q.shape[1], q.shape[2]);
    let mut qd = q.data.clone();
    ops::apply_rotary_inplace(&mut qd, h, ll, dd, &cos, &sin);
    check("rotary.q", &Tensor::new(qd, q.shape.clone()), &f.get("q_rot"), 5e-6);

    let k = f.get("k");
    let mut kd = k.data.clone();
    ops::apply_rotary_inplace(&mut kd, k.shape[0], k.shape[1], k.shape[2], &cos, &sin);
    check("rotary.k", &Tensor::new(kd, k.shape.clone()), &f.get("k_rot"), 5e-6);
}
