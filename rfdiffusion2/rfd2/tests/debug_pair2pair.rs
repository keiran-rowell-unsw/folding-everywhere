//! Bisection inside `pair2pair` — the open defect from the layer-by-layer run.
//!
//! `tests/parity_layerwise.rs` and `tests/debug_blocks.rs` both localise the
//! disagreement to `main_block.2` and `main_block.23`, and in both the first
//! module to break is `pair2pair` while `msa2msa` and `msa2pair` above it are
//! bit-identical. This test goes one level deeper: every one of `pair2pair`'s
//! twelve children is driven from **its own** captured input and compared to its
//! own captured output, so a failing row is that child and nothing else.
//!
//! `main_block.1` is included as a control. It is bit-identical at block level,
//! so every one of its rows must be exact; if a `main_block.1` row fails, the
//! harness is wrong rather than the port.
//!
//! Fixture: `python/dump_io.py --pinned --out p2p_io
//!          --match 'model\.simulator\.main_block\.(1|2|23)\.pair2pair(\.[a-z_]+)?$'`

use rfd2::model::attention::{BiasedAxialAttention, TriangleMultiplication};
use rfd2::model::rf::Arch;
use rfd2::nn::{Ctx, FeedForward, LayerNorm, Linear, Params};
use rfd2::parity;
use rfd2::rng::torch::Mt19937;
use rfd2::weights::Weights;
use std::path::Path;

fn open(rel: &str) -> Option<Weights> {
    let p = format!("{}/../{rel}", env!("CARGO_MANIFEST_DIR"));
    if !Path::new(&p).exists() {
        eprintln!("SKIP: {p} missing — run python/dump_io.py (see the module header)");
        return None;
    }
    Some(Weights::open(&p).expect("open"))
}

#[test]
fn which_child_of_pair2pair_diverges() {
    let Some(io) = open("fixtures/p2p_io/io.safetensors") else { return };
    let Some(w) = open("fixtures/weights/model_state_dict.safetensors") else { return };
    let arch = Arch::rfd173();
    let root = Params::root(&w, "model").sub("simulator").sub("main_block");

    let mut worst: Vec<(String, parity::Stats)> = Vec::new();
    for blk in [1usize, 2, 23] {
        let key = format!("model.simulator.main_block.{blk}.pair2pair");
        if !io.has(&format!("in::{key}.0")) {
            eprintln!("SKIP block {blk}: no capture");
            continue;
        }
        let p = root.idx(blk).sub("pair2pair");
        println!("\n=== main_block.{blk}.pair2pair {} ===",
                 if blk == 1 { "(control — must be exact)" } else { "(diverges at block level)" });

        let mut row = |name: &str, got: &[f32]| {
            let want = io.get(&format!("out::{key}.{name}")).data;
            if want.len() != got.len() {
                println!("  {name:<14} LEN {} vs {}", got.len(), want.len());
                return;
            }
            let s = parity::compare(got, &want);
            let flag = if s.exact == s.n { "" } else { "   <-- DIFFERS" };
            println!(
                "  {name:<14} {:>7}/{:<7} {:6.2}%  max|d| {:.3e}  max_ulp {}{flag}",
                s.exact, s.n, 100.0 * s.exact_frac(), s.max_abs, s.max_ulp
            );
            if s.exact != s.n {
                // The magnitude of the values that disagree decides whether this
                // is a rounding tie or a formula error: a tie perturbs the last
                // bit of whatever it lands on, so |value| at the worst site is
                // the scale the error should be read against.
                let (mut n_diff, mut at, mut worst) = (0usize, 0.0f32, 0.0f32);
                for (g, wv) in got.iter().zip(&want) {
                    if g.to_bits() != wv.to_bits() {
                        n_diff += 1;
                        let d = (g - wv).abs();
                        if d > worst {
                            worst = d;
                            at = *wv;
                        }
                    }
                }
                println!(
                    "                 {n_diff} values differ ({:.4}% of the tensor); worst site \
                     value {at:.6e}, i.e. {:.1} ULP of that magnitude",
                    100.0 * n_diff as f64 / s.n as f64,
                    worst as f64 / (at.abs() as f64 * f32::EPSILON as f64).max(f64::MIN_POSITIVE)
                );
            }
            if s.exact != s.n {
                worst.push((format!("main_block.{blk}.{name}"), s));
            }
        };

        // ---- the parameter-free projections, each from its own input --------
        let inp = |name: &str, i: usize| io.get(&format!("in::{key}.{name}.{i}"));

        row("emb_rbf", &Linear::load(&p.sub("emb_rbf")).forward(&inp("emb_rbf", 0)).data);
        row("norm_state",
            &LayerNorm::load(&p.sub("norm_state")).forward(&inp("norm_state", 0)).data);
        row("proj_left", &Linear::load(&p.sub("proj_left")).forward(&inp("proj_left", 0)).data);
        row("proj_right", &Linear::load(&p.sub("proj_right")).forward(&inp("proj_right", 0)).data);
        row("to_gate", &Linear::load(&p.sub("to_gate")).forward(&inp("to_gate", 0)).data);

        // ---- the two triangle multiplications --------------------------------
        for (nm, outgoing) in [("tri_mul_out", true), ("tri_mul_in", false)] {
            let m = TriangleMultiplication::load(&p.sub(nm), outgoing);
            row(nm, &m.forward(&inp(nm, 0)).data);
        }

        // ---- the two biased axial attentions ---------------------------------
        for (nm, is_row) in [("row_attn", true), ("col_attn", false)] {
            let m = BiasedAxialAttention::load(&p.sub(nm), arch.n_head_pair, arch.d_hidden, is_row);
            row(nm, &m.forward(&inp(nm, 0), &inp(nm, 1)).data);
        }

        // ---- the feed-forward: the only child that draws from the RNG --------
        {
            let bytes: Vec<u8> =
                io.get_i64(&format!("rng::{key}.ff")).0.into_iter().map(|v| v as u8).collect();
            let mut ctx = Ctx::new(Mt19937::from_torch_state(&bytes));
            // NOT `arch.p_drop`. `PairStr2Pair.ff = FeedForwardLayer(d_pair, 2)`
            // passes no p_drop, so this one keeps the 0.1 default while the
            // block's own dropout is 0.15 — see PairStr2Pair::load. Getting it
            // wrong changes the mask AND the 1/(1-p) scale, which is what the
            // control block caught.
            let m = FeedForward::load(&p.sub("ff"), 0.1);
            row("ff", &m.forward(&inp("ff", 0), &mut ctx).data);
        }
    }

    println!("\n{} child rows differ", worst.len());
    for (n, s) in &worst {
        println!("  {n:<34} {}", s.summary());
    }
    // The control block must be clean, whatever the other two do. If it is not,
    // the disagreement is in this harness (wrong weights, wrong arch constants,
    // a stale fixture) and nothing else here can be believed.
    let control: Vec<&String> =
        worst.iter().map(|(n, _)| n).filter(|n| n.starts_with("main_block.1.")).collect();
    assert!(control.is_empty(), "the CONTROL block disagrees — the harness is wrong: {control:?}");
}
