//! `main_block.0` is the one trunk block that is not bit-identical: it differs
//! by 1 ULP in ~1e-5 of its pair values. This walks its MSA and pair tracks to
//! find which stage introduces it — and, importantly, how many values it
//! introduces, since a handful of ULP flips is the predicted behaviour of f64
//! pinning (see `docs/BITEXACT.md`) rather than a bug.

use rfd2::model::attention::{BiasedAxialAttention, TriangleMultiplication};
use rfd2::model::track::{MSA2Pair, MSAPairStr2MSA};
use rfd2::nn::{Ctx, Params};
use rfd2::parity;
use rfd2::rng::torch::Mt19937;
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

fn chk_detail(label: &str, got: &[f32], want: &[f32], l: usize, c: usize) {
    chk(label, got, want);
    let bad: Vec<usize> = got
        .iter()
        .zip(want)
        .enumerate()
        .filter(|(_, (g, w))| g.to_bits() != w.to_bits())
        .map(|(i, _)| i)
        .collect();
    if bad.is_empty() {
        return;
    }
    let cells: std::collections::BTreeSet<(usize, usize)> =
        bad.iter().map(|i| ((i / c) / l, (i / c) % l)).collect();
    let chans: std::collections::BTreeSet<usize> = bad.iter().map(|i| i % c).collect();
    println!(
        "      {} cells, {} distinct channels; first idx {} got {:e} want {:e}",
        cells.len(),
        chans.len(),
        bad[0],
        got[bad[0]],
        want[bad[0]]
    );
    println!("      cells: {:?}", cells.iter().take(10).collect::<Vec<_>>());
}

fn chk(label: &str, got: &[f32], want: &[f32]) {
    let s = parity::compare(got, want);
    println!(
        "{:<24} {:>9} values, {:>5} differ, max|d| {:.3e}, max {} ULP",
        label,
        s.n,
        s.n - s.exact,
        s.max_abs,
        s.max_ulp
    );
}

const B: &str = "model.simulator.main_block.0";

#[test]
fn main_block0_tracks() {
    let Some(io) = open("fixtures/m0_io/io.safetensors") else { return };
    let Some(w) = open("fixtures/weights/model_state_dict.safetensors") else { return };
    let p = Params::root(&w, "model").sub("simulator").sub("main_block").sub("0");

    // ---- msa2msa ----------------------------------------------------------
    if io.has(&format!("in::{B}.msa2msa.0")) {
        let msa = io.get(&format!("in::{B}.msa2msa.0"));
        let pair = io.get(&format!("in::{B}.msa2msa.1"));
        let rbf = io.get(&format!("in::{B}.msa2msa.2"));
        let state = io.get(&format!("in::{B}.msa2msa.3"));
        let bytes: Vec<u8> =
            io.get_i64(&format!("rng::{B}.msa2msa")).0.into_iter().map(|v| v as u8).collect();
        let mut ctx = Ctx::new(Mt19937::from_torch_state(&bytes));
        let m = MSAPairStr2MSA::load(&p.sub("msa2msa"), 8, 32, false, 0.15f64);
        let out = m.forward(&msa, &pair, &rbf, &state, &mut ctx);
        chk("msa2msa", &out.data, &io.get(&format!("out::{B}.msa2msa")).data);
    }

    // ---- msa2pair ---------------------------------------------------------
    let msa = io.get(&format!("in::{B}.msa2pair.0"));
    let pair_in = io.get(&format!("in::{B}.msa2pair.1"));
    let m2p = MSA2Pair::load(&p.sub("msa2pair"));
    let pair = m2p.forward(&msa, &pair_in);
    chk("msa2pair", &pair.data, &io.get(&format!("out::{B}.msa2pair")).data);

    // ---- pair2pair, stage by stage ---------------------------------------
    let pp = p.sub("pair2pair");
    let pair0 = io.get(&format!("in::{B}.pair2pair.0"));
    let rbf = io.get(&format!("in::{B}.pair2pair.1"));

    let r = rfd2::nn::Linear::load(&pp.sub("emb_rbf")).forward(&rbf);
    chk("emb_rbf", &r.data, &io.get(&format!("out::{B}.pair2pair.emb_rbf")).data);

    let tmo = TriangleMultiplication::load(&pp.sub("tri_mul_out"), true);
    chk(
        "tri_mul_out",
        &tmo.forward(&pair0).data,
        &io.get(&format!("out::{B}.pair2pair.tri_mul_out")).data,
    );

    let tmi = TriangleMultiplication::load(&pp.sub("tri_mul_in"), false);
    chk(
        "tri_mul_in",
        &tmi.forward(&io.get(&format!("in::{B}.pair2pair.tri_mul_in.0"))).data,
        &io.get(&format!("out::{B}.pair2pair.tri_mul_in")).data,
    );

    let ra = BiasedAxialAttention::load(&pp.sub("row_attn"), 6, 32, true);
    let ra_out = ra.forward(
        &io.get(&format!("in::{B}.pair2pair.row_attn.0")),
        &io.get(&format!("in::{B}.pair2pair.row_attn.1")),
    );
    chk_detail(
        "row_attn",
        &ra_out.data,
        &io.get(&format!("out::{B}.pair2pair.row_attn")).data,
        71,
        192,
    );

    let ca = BiasedAxialAttention::load(&pp.sub("col_attn"), 6, 32, false);
    chk(
        "col_attn",
        &ca.forward(
            &io.get(&format!("in::{B}.pair2pair.col_attn.0")),
            &io.get(&format!("in::{B}.pair2pair.col_attn.1")),
        )
        .data,
        &io.get(&format!("out::{B}.pair2pair.col_attn")).data,
    );

    // Is the row_attn difference order sensitivity, or a real disagreement?
    // Recompute its tied contraction with the n-loop reversed: same f64
    // arithmetic, different order. If the answer moves by the same amount, the
    // 299 values are the f64 tie-straddle limit and not a bug.
    {
        let pair = io.get(&format!("in::{B}.pair2pair.row_attn.0"));
        let l = pair.shape[1];
        let (h, dim) = (6usize, 32usize);
        let pr = pair.permute(&[0, 2, 1, 3]);
        let pn = rfd2::nn::LayerNorm::load(&pp.sub("row_attn").sub("norm_pair")).forward(&pr);
        let mut q = rfd2::nn::Linear::load_nobias(&pp.sub("row_attn").sub("to_q")).forward(&pn);
        let mut k = rfd2::nn::Linear::load_nobias(&pp.sub("row_attn").sub("to_k")).forward(&pn);
        let sc = rfd2::model::attention::scaling(dim);
        for v in q.data.iter_mut() {
            *v *= sc;
        }
        for v in k.data.iter_mut() {
            *v /= l as f32;
        }
        let mut fwd = vec![0.0f32; l * l * h];
        let mut rev = vec![0.0f32; l * l * h];
        for i in 0..l {
            for j in 0..l {
                for hh in 0..h {
                    let mut a = 0.0f64;
                    for ni in 0..l {
                        let qo = ((ni * l + i) * h + hh) * dim;
                        let ko = ((ni * l + j) * h + hh) * dim;
                        for d in 0..dim {
                            a += q.data[qo + d] as f64 * k.data[ko + d] as f64;
                        }
                    }
                    let mut b = 0.0f64;
                    for ni in (0..l).rev() {
                        let qo = ((ni * l + i) * h + hh) * dim;
                        let ko = ((ni * l + j) * h + hh) * dim;
                        for d in (0..dim).rev() {
                            b += q.data[qo + d] as f64 * k.data[ko + d] as f64;
                        }
                    }
                    fwd[(i * l + j) * h + hh] = a as f32;
                    rev[(i * l + j) * h + hh] = b as f32;
                }
            }
        }
        let s = parity::compare(&fwd, &rev);
        println!(
            "row_attn tied logits: {} of {} change when the 2272-term f64 sum is \
             reversed ({:.2e})",
            s.n - s.exact,
            s.n,
            (s.n - s.exact) as f64 / s.n as f64
        );

        // Now the value contraction, which is where the sensitivity should be:
        // 71-term f64 sums whose terms partly cancel.
        let bias = io.get(&format!("in::{B}.pair2pair.row_attn.1"));
        let bn = rfd2::nn::LayerNorm::load(&pp.sub("row_attn").sub("norm_bias"))
            .forward(&bias.permute(&[0, 2, 1, 3]));
        let bb = rfd2::nn::Linear::load_nobias(&pp.sub("row_attn").sub("to_b")).forward(&bn);
        let mut logits = fwd.clone();
        for (i, v) in logits.iter_mut().enumerate() {
            *v += bb.data[i];
        }
        let attn = rfd2::ops::elem::softmax_dim(
            &rfd2::tensor::Tensor::new(logits, vec![1, l, l, h]),
            2,
        );
        let v = rfd2::nn::Linear::load_nobias(&pp.sub("row_attn").sub("to_v")).forward(&pn);
        let mut o_f = vec![0.0f32; l * l * h * dim];
        let mut o_r = vec![0.0f32; l * l * h * dim];
        for ni in 0..l {
            for i in 0..l {
                for hh in 0..h {
                    for d in 0..dim {
                        let mut a = 0.0f64;
                        for j in 0..l {
                            a += attn.data[((i * l) + j) * h + hh] as f64
                                * v.data[(((ni * l) + j) * h + hh) * dim + d] as f64;
                        }
                        let mut b = 0.0f64;
                        for j in (0..l).rev() {
                            b += attn.data[((i * l) + j) * h + hh] as f64
                                * v.data[(((ni * l) + j) * h + hh) * dim + d] as f64;
                        }
                        o_f[(((ni * l) + i) * h + hh) * dim + d] = a as f32;
                        o_r[(((ni * l) + i) * h + hh) * dim + d] = b as f32;
                    }
                }
            }
        }
        let s = parity::compare(&o_f, &o_r);
        println!(
            "row_attn value contraction: {} of {} change when the 71-term f64 sum \
             is reversed ({:.2e})",
            s.n - s.exact,
            s.n,
            (s.n - s.exact) as f64 / s.n as f64
        );
    }

    let bytes: Vec<u8> =
        io.get_i64(&format!("rng::{B}.pair2pair.ff")).0.into_iter().map(|v| v as u8).collect();
    let mut ctx = Ctx::new(Mt19937::from_torch_state(&bytes));
    let ff = rfd2::nn::FeedForward::load(&pp.sub("ff"), 0.1);
    chk(
        "ff",
        &ff.forward(&io.get(&format!("in::{B}.pair2pair.ff.0")), &mut ctx).data,
        &io.get(&format!("out::{B}.pair2pair.ff")).data,
    );
}
