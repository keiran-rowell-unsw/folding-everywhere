//! How order-independent is the f64 pinning, really?
//!
//! `docs/BITEXACT.md` argues that accumulating in f64 and rounding once makes
//! the fp32 result independent of the reduction order, because an f64 rounding
//! error (~1e-16 relative) is ~9 orders of magnitude below an fp32 ULP
//! (~6e-8 relative). That is true *per value* but it is a probability, not a
//! guarantee: two f64 orders disagree whenever the exact value happens to lie
//! within ~1e-16 of the midpoint between two fp32 numbers, which is about
//! **2e-9 of values**.
//!
//! At RFdiffusion2's scale that is not negligible. One forward pass evaluates
//! on the order of 1e9 reduction outputs, so a handful of ULP flips per pass is
//! the *expected* behaviour — which is exactly what the block-by-block
//! comparison shows (35 of 36 blocks bit-identical, one differing by 1 ULP in
//! ~1e-5 of its values).
//!
//! This test measures the rate directly on real weights and real activations,
//! so the number in `docs/BITEXACT.md` is measured rather than argued.

use rfd2::parity;
use rfd2::tensor::Tensor;
use rfd2::weights::Weights;
use std::path::Path;

fn open(rel: &str) -> Option<Weights> {
    let path = format!("{}/../{rel}", env!("CARGO_MANIFEST_DIR"));
    if !Path::new(&path).exists() {
        eprintln!("SKIP: {path} missing");
        return None;
    }
    Some(Weights::open(&path).expect("open"))
}

/// `linear` with the K-loop run backwards. Same f64 arithmetic, different order.
fn linear_f64_reversed(x: &Tensor, w: &Tensor, b: Option<&Tensor>) -> Tensor {
    let k = x.last();
    let rows = x.numel() / k;
    let o = w.shape[0];
    let mut out = vec![0.0f32; rows * o];
    for r in 0..rows {
        for oi in 0..o {
            let mut acc = 0.0f64;
            for kk in (0..k).rev() {
                acc += x.data[r * k + kk] as f64 * w.data[oi * k + kk] as f64;
            }
            let bias = b.map(|bb| bb.data[oi] as f64).unwrap_or(0.0);
            out[r * o + oi] = (acc + bias) as f32;
        }
    }
    let mut shape = x.shape.clone();
    let n = shape.len();
    shape[n - 1] = o;
    Tensor::new(out, shape)
}

#[test]
fn f64_pinning_is_order_independent_but_not_perfectly() {
    let Some(f) = open("fixtures/blocks_io/io.safetensors") else { return };
    let Some(w) = open("fixtures/weights/model_state_dict.safetensors") else { return };

    let mut total = 0usize;
    let mut diff = 0usize;
    // the widest reductions in the trunk: pair -> pair projections (K = 192)
    // and the MSA2Pair output projection (K = 256)
    for i in 0..8 {
        let pair = f.get(&format!("in::model.simulator.main_block.{i}.1"));
        for name in [
            "pair2pair.tri_mul_out.left_proj",
            "pair2pair.row_attn.to_q",
            "pair2pair.col_attn.to_v",
        ] {
            let base = format!("model.simulator.main_block.{i}.{name}");
            if !w.has(&format!("{base}.weight")) {
                continue;
            }
            let weight = w.get(&format!("{base}.weight"));
            let bias =
                if w.has(&format!("{base}.bias")) { Some(w.get(&format!("{base}.bias"))) } else { None };
            let a = rfd2::ops::linear_f64(&pair, &weight, bias.as_ref());
            let b = linear_f64_reversed(&pair, &weight, bias.as_ref());
            let s = parity::compare(&a.data, &b.data);
            total += s.n;
            diff += s.n - s.exact;
        }
    }
    let rate = diff as f64 / total as f64;
    println!(
        "f64 order sensitivity: {diff} of {total} outputs change when the K-loop \
         is reversed  ({:.3e} of values, max 1 ULP)",
        rate
    );
    // The point of the test is the number, not a threshold; but a rate far above
    // ~1e-7 would mean something other than tie-straddling is going on.
    assert!(rate < 1e-6, "order sensitivity {rate:.3e} is too high to be tie-straddling");
}
