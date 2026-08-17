//! One level below `debug_pair2pair.rs`: which op inside `tri_mul_out` /
//! `row_attn` first disagrees, and — for the single worst element — what the
//! two sides' bit patterns are, so the value can be adjudicated in exact
//! arithmetic outside Rust.
//!
//! The triangle contraction is an `einsum`, not a `nn.Module`, so it has no
//! forward hook of its own. It is captured anyway: its output is exactly
//! `in::…tri_mul_out.norm_out.0`, the input of the module that follows it.
//!
//! Fixture: `python/dump_io.py --pinned --out tri_io --match
//!   'model\.simulator\.main_block\.(1|2|23)\.pair2pair\.(tri_mul_out|row_attn)(\.[a-z_0-9]+)?$'`

use rfd2::model::attention::TriangleMultiplication;
use rfd2::nn::{LayerNorm, Linear, Params};
use rfd2::ops::acc::Acc;
use rfd2::ops::elem::sigmoid_scalar;
use rfd2::parity;
use rfd2::tensor::Tensor;
use rfd2::weights::Weights;
use std::path::Path;

fn open(rel: &str) -> Option<Weights> {
    let p = format!("{}/../{rel}", env!("CARGO_MANIFEST_DIR"));
    if !Path::new(&p).exists() {
        eprintln!("SKIP: {p} missing — see the module header");
        return None;
    }
    Some(Weights::open(&p).expect("open"))
}

#[test]
fn which_op_inside_tri_mul_out_diverges() {
    let Some(io) = open("fixtures/tri_io/io.safetensors") else { return };
    let Some(w) = open("fixtures/weights/model_state_dict.safetensors") else { return };
    let root = Params::root(&w, "model").sub("simulator").sub("main_block");

    for blk in [1usize, 2] {
        let key = format!("model.simulator.main_block.{blk}.pair2pair.tri_mul_out");
        if !io.has(&format!("in::{key}.0")) {
            continue;
        }
        let p = root.idx(blk).sub("pair2pair").sub("tri_mul_out");
        println!("\n=== main_block.{blk}.tri_mul_out {} ===",
                 if blk == 1 { "(control)" } else { "(diverges)" });

        let mut report = |name: &str, got: &[f32], want_key: String| {
            let want = io.get(&want_key).data;
            let s = parity::compare(got, &want);
            println!(
                "  {name:<24} {:>7}/{:<7} {:6.2}%  max|d| {:.3e}",
                s.exact, s.n, 100.0 * s.exact_frac(), s.max_abs
            );
            if s.exact != s.n {
                // Print the single worst element as raw bits. Two fp32 values one
                // ULP apart differ by exactly 1 in the integer view, which is what
                // makes the exact-arithmetic adjudication outside Rust meaningful.
                let (mut idx, mut worst) = (0usize, 0.0f32);
                for (i, (g, wv)) in got.iter().zip(&want).enumerate() {
                    let d = (g - wv).abs();
                    if d > worst {
                        worst = d;
                        idx = i;
                    }
                }
                let (g, wv) = (got[idx], want[idx]);
                println!(
                    "      worst element [{idx}]  port {:.9e} ({:#010x})  ref {:.9e} ({:#010x})  \
                     int gap {}",
                    g, g.to_bits(), wv, wv.to_bits(),
                    (g.to_bits() as i64 - wv.to_bits() as i64).abs()
                );
                let n_diff =
                    got.iter().zip(&want).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
                println!("      {n_diff} of {} values differ", got.len());
            }
        };

        // ---- the chain, in the order TriangleMultiplication::forward runs it --
        let pair_in = io.get(&format!("in::{key}.0"));
        let normed = LayerNorm::load(&p.sub("norm")).forward(&pair_in);
        report("norm", &normed.data, format!("out::{key}.norm"));

        let mut left = Linear::load(&p.sub("left_proj")).forward(&normed);
        report("left_proj", &left.data, format!("out::{key}.left_proj"));
        let lg = Linear::load(&p.sub("left_gate")).forward(&normed);
        report("left_gate", &lg.data, format!("out::{key}.left_gate"));
        for (i, x) in left.data.iter_mut().enumerate() {
            *x *= sigmoid_scalar(lg.data[i]);
        }

        let mut right = Linear::load(&p.sub("right_proj")).forward(&normed);
        report("right_proj", &right.data, format!("out::{key}.right_proj"));
        let rg = Linear::load(&p.sub("right_gate")).forward(&normed);
        report("right_gate", &rg.data, format!("out::{key}.right_gate"));
        for (i, x) in right.data.iter_mut().enumerate() {
            *x *= sigmoid_scalar(rg.data[i]);
        }

        // the einsum — no module, so compared against norm_out's captured INPUT
        let (b, l) = (pair_in.shape[0], pair_in.shape[1]);
        let dh = left.last();
        for x in right.data.iter_mut() {
            *x /= l as f32;
        }
        let mut ein = vec![0.0f32; b * l * l * dh];
        for bi in 0..b {
            for i in 0..l {
                for j in 0..l {
                    for d in 0..dh {
                        let mut acc = Acc::new();
                        for k in 0..l {
                            let li = ((bi * l + i) * l + k) * dh + d; // 'bikd,bjkd->bijd'
                            let rj = ((bi * l + j) * l + k) * dh + d;
                            acc.add(left.data[li] as f64 * right.data[rj] as f64);
                        }
                        ein[((bi * l + i) * l + j) * dh + d] = acc.get() as f32;
                    }
                }
            }
        }
        report("einsum (bikd,bjkd)", &ein, format!("in::{key}.norm_out.0"));

        let ein_t = Tensor::new(ein, vec![b, l, l, dh]);
        let no = LayerNorm::load(&p.sub("norm_out")).forward(&ein_t);
        report("norm_out", &no.data, format!("out::{key}.norm_out"));
        let op = Linear::load(&p.sub("out_proj")).forward(&no);
        report("out_proj", &op.data, format!("out::{key}.out_proj"));
    }
}
