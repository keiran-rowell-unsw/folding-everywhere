//! Which child of `row_attn` diverges in `main_block.23` — the last block that
//! is not bit-identical after the `layer_norm` summation fix.
//!
//! `main_block.2`'s divergence was traced to `tri_mul_out.norm`, i.e. naive
//! sequential f64 summation inside `layer_norm` losing one f64 ULP of `var`
//! (`ops::reduce::sum_compensated` documents the measurement). `row_attn`
//! contains two `layer_norm`s of its own, so this asks whether block 23 has the
//! same cause or a different one.
//!
//! Fixture: `python/dump_io.py --pinned --out tri_io --match
//!   'model\.simulator\.main_block\.(1|2|23)\.pair2pair\.(tri_mul_out|row_attn)(\.[a-z_0-9]+)?$'`

use rfd2::nn::{LayerNorm, Linear, Params};
use rfd2::parity;
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
fn which_child_of_row_attn_diverges() {
    let Some(io) = open("fixtures/tri_io/io.safetensors") else { return };
    let Some(w) = open("fixtures/weights/model_state_dict.safetensors") else { return };
    let root = Params::root(&w, "model").sub("simulator").sub("main_block");

    for blk in [1usize, 23] {
        let key = format!("model.simulator.main_block.{blk}.pair2pair.row_attn");
        if !io.has(&format!("in::{key}.0")) {
            eprintln!("SKIP block {blk}: no capture");
            continue;
        }
        let p = root.idx(blk).sub("pair2pair").sub("row_attn");
        println!("\n=== main_block.{blk}.row_attn {} ===",
                 if blk == 1 { "(control)" } else { "(diverges)" });

        let mut report = |name: &str, got: &[f32], want_key: String| {
            if !io.has(&want_key) {
                println!("  {name:<12} (not captured)");
                return;
            }
            let want = io.get(&want_key).data;
            let s = parity::compare(got, &want);
            print!(
                "  {name:<12} {:>7}/{:<7} {:6.2}%  max|d| {:.3e}",
                s.exact, s.n, 100.0 * s.exact_frac(), s.max_abs
            );
            if s.exact == s.n {
                println!();
            } else {
                let (mut idx, mut worst) = (0usize, 0.0f32);
                for (i, (g, wv)) in got.iter().zip(&want).enumerate() {
                    let d = (g - wv).abs();
                    if d > worst {
                        worst = d;
                        idx = i;
                    }
                }
                let n_diff =
                    got.iter().zip(&want).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
                println!("   <-- {n_diff} differ");
                println!(
                    "      worst [{idx}]  port {:.9e} ({:#010x})  ref {:.9e} ({:#010x})  int gap {}",
                    got[idx], got[idx].to_bits(), want[idx], want[idx].to_bits(),
                    (got[idx].to_bits() as i64 - want[idx].to_bits() as i64).abs()
                );
            }
        };

        // `row_attn` permutes its inputs before the two layer_norms; the capture
        // is of the module's own arguments, so the permute is replayed here.
        let pair_in = io.get(&format!("in::{key}.0"));
        let bias_in = io.get(&format!("in::{key}.1"));
        let pair = pair_in.permute(&[0, 2, 1, 3]); // is_row = true
        let bias = bias_in.permute(&[0, 2, 1, 3]);

        let np = LayerNorm::load(&p.sub("norm_pair")).forward(&pair);
        report("norm_pair", &np.data, format!("out::{key}.norm_pair"));
        let nb = LayerNorm::load(&p.sub("norm_bias")).forward(&bias);
        report("norm_bias", &nb.data, format!("out::{key}.norm_bias"));

        for nm in ["to_q", "to_k", "to_v", "to_g"] {
            let lin = Linear::load(&p.sub(nm));
            report(nm, &lin.forward(&np).data, format!("out::{key}.{nm}"));
        }
        let tob = Linear::load(&p.sub("to_b"));
        report("to_b", &tob.forward(&nb).data, format!("out::{key}.to_b"));
    }
}
