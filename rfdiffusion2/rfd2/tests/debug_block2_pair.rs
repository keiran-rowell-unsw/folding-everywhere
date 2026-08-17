//! `extra_block.2`'s pair track, stage by stage. The whole block is exact
//! except a single pair cell (42, 39), which differs by ~1.0 — far too much to
//! be round-off, so one stage is doing something structurally different for
//! that one cell.

use rfd2::model::attention::{BiasedAxialAttention, TriangleMultiplication};
use rfd2::model::track::MSA2Pair;
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

fn chk(label: &str, got: &[f32], want: &[f32], l: usize, c: usize) {
    let s = parity::compare(got, want);
    print!("{:<26} {}", label, s.summary());
    if s.exact != s.n {
        let bad: Vec<usize> = got
            .iter()
            .zip(want)
            .enumerate()
            .filter(|(_, (g, w))| g.to_bits() != w.to_bits())
            .map(|(i, _)| i)
            .collect();
        let cells: std::collections::BTreeSet<(usize, usize)> =
            bad.iter().map(|i| ((i / c) / l, (i / c) % l)).collect();
        print!("   cells {:?}", cells.iter().take(5).collect::<Vec<_>>());
        let i = bad[0];
        print!("  first: got {} want {}", got[i], want[i]);
    }
    println!();
}

const B: &str = "model.simulator.extra_block.2";

#[test]
fn block2_pair_track() {
    let Some(io) = open("fixtures/b2p_io/io.safetensors") else { return };
    let Some(w) = open("fixtures/weights/model_state_dict.safetensors") else { return };
    let p = Params::root(&w, "model").sub("simulator").sub("extra_block").sub("2");

    let msa = io.get(&format!("in::{B}.msa2pair.0"));
    let pair_in = io.get(&format!("in::{B}.msa2pair.1"));
    let l = pair_in.shape[1];
    let c = pair_in.last();

    let m2p = MSA2Pair::load(&p.sub("msa2pair"));
    let pair = m2p.forward(&msa, &pair_in);
    chk("msa2pair", &pair.data, &io.get(&format!("out::{B}.msa2pair")).data, l, c);

    let pp = p.sub("pair2pair");
    let pair0 = io.get(&format!("in::{B}.pair2pair.0"));
    let rbf = io.get(&format!("in::{B}.pair2pair.1"));
    let state = io.get(&format!("in::{B}.pair2pair.2"));

    let emb_rbf = rfd2::nn::Linear::load(&pp.sub("emb_rbf"));
    let r = emb_rbf.forward(&rbf);
    chk("emb_rbf", &r.data, &io.get(&format!("out::{B}.pair2pair.emb_rbf")).data, l, c);

    let tmo = TriangleMultiplication::load(&pp.sub("tri_mul_out"), true);
    let d = tmo.forward(&pair0);
    chk("tri_mul_out", &d.data, &io.get(&format!("out::{B}.pair2pair.tri_mul_out")).data, l, c);

    let tin = io.get(&format!("in::{B}.pair2pair.tri_mul_in.0"));
    let tmi = TriangleMultiplication::load(&pp.sub("tri_mul_in"), false);
    let d = tmi.forward(&tin);
    chk("tri_mul_in", &d.data, &io.get(&format!("out::{B}.pair2pair.tri_mul_in")).data, l, c);

    let ra_in = io.get(&format!("in::{B}.pair2pair.row_attn.0"));
    let ra_bias = io.get(&format!("in::{B}.pair2pair.row_attn.1"));
    let ra = BiasedAxialAttention::load(&pp.sub("row_attn"), 6, 32, true);
    let d = ra.forward(&ra_in, &ra_bias);
    chk("row_attn", &d.data, &io.get(&format!("out::{B}.pair2pair.row_attn")).data, l, c);

    let ca_in = io.get(&format!("in::{B}.pair2pair.col_attn.0"));
    let ca_bias = io.get(&format!("in::{B}.pair2pair.col_attn.1"));
    let ca = BiasedAxialAttention::load(&pp.sub("col_attn"), 6, 32, false);
    let d = ca.forward(&ca_in, &ca_bias);
    chk("col_attn", &d.data, &io.get(&format!("out::{B}.pair2pair.col_attn")).data, l, c);

    let ff_in = io.get(&format!("in::{B}.pair2pair.ff.0"));
    let bytes: Vec<u8> = io
        .get_i64(&format!("rng::{B}.pair2pair.ff"))
        .0
        .into_iter()
        .map(|v| v as u8)
        .collect();
    let mut ctx = Ctx::new(Mt19937::from_torch_state(&bytes));
    let ff = rfd2::nn::FeedForward::load(&pp.sub("ff"), 0.1);
    let d = ff.forward(&ff_in, &mut ctx);
    chk("ff", &d.data, &io.get(&format!("out::{B}.pair2pair.ff")).data, l, c);

    let _ = state;
}
