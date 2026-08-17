//! Bisection harness for `simulator.extra_block.0` — the first block that
//! exercises the SE(3) transformer, the chiral gradients and the graph
//! construction, i.e. everything rung 6b and 6c rest on.
//!
//! Every stage is fed the reference's own captured input, and the RNG is
//! restarted from the generator state captured on entry to that same module, so
//! a failure localises to one stage rather than to "somewhere upstream".

use rfd2::model::embeddings::{PairStr2Pair, PositionalEncoding2D};
use rfd2::model::iterblock::{BlockCfg, BlockInputs, IterBlock, TrackState};
use rfd2::model::str2str::Str2Str;
use rfd2::model::track::{MSA2Pair, MSAPairStr2MSA};
use rfd2::nn::{Ctx, Params};
use rfd2::rng::torch::Mt19937;
use rfd2::{geom, parity};
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

fn ctx_at(io: &Weights, module: &str) -> Ctx {
    let bytes: Vec<u8> = io
        .get_i64(&format!("rng::{module}"))
        .0
        .into_iter()
        .map(|v| v as u8)
        .collect();
    Ctx::new(Mt19937::from_torch_state(&bytes))
}

fn chk(label: &str, got: &[f32], want: &[f32]) -> bool {
    let s = parity::compare(got, want);
    println!("{:<52} {}", label, s.summary());
    s.exact == s.n && got.len() == want.len()
}

const B: &str = "model.simulator.extra_block.0";

fn cfg() -> BlockCfg {
    BlockCfg {
        n_head_msa: 8,
        d_hidden_msa: 8,
        n_head_pair: 6,
        d_hidden: 32,
        use_global_attn: true,
        enable_same_chain: true,
        p_drop: 0.15f64,
        se3_num_layers: 1,
        l0_in: 64,
        l1_in: 6,
        num_channels: 32,
        num_degrees: 2,
        l0_out: 64,
        l1_out: 2,
        n_heads: 4,
        div: 4,
        top_k: -1,
        n_extra_l1: 3,
    }
}

#[test]
fn extra_block0_bisect() {
    let Some(io) = open("fixtures/extra0_io/io.safetensors") else { return };
    let Some(w) = open("fixtures/weights/model_state_dict.safetensors") else { return };
    let Some(step) = open("fixtures/model_pinned/step0.safetensors") else { return };
    let p = Params::root(&w, "model").sub("simulator").sub("extra_block").sub("0");

    // block inputs, in IterBlock.forward's positional order
    let msa = io.get(&format!("in::{B}.0"));
    let pair = io.get(&format!("in::{B}.1"));
    let xyz = io.get(&format!("in::{B}.2")); // [1,L,3,3]
    let state = io.get(&format!("in::{B}.3"));
    let (seq_unmasked, s) = io.get_i64(&format!("in::{B}.4"));
    let l = s[s.len() - 1];
    let idx = io.get_i64(&format!("in::{B}.5")).0;
    let bond_feats = io.get_i64(&format!("in::{B}.6")).0;
    let same_chain: Vec<bool> =
        io.get_i64(&format!("in::{B}.7")).0.into_iter().map(|v| v != 0).collect();
    let dist_matrix = step.get("rfi.dist_matrix").data;
    let chirals = step.get("rfi.chirals").data;
    let atom_frames = step.get_i64("rfi.atom_frames").0;
    let is_motif: Vec<bool> =
        step.get_i64("rfi.is_motif").0.into_iter().map(|v| v != 0).collect();
    let rotation_mask: Vec<bool> = seq_unmasked.iter().map(|&t| geom::is_atom(t)).collect();

    // ---- rbf_feat + positional encoding ----------------------------------
    let ca: Vec<f32> = (0..l).flat_map(|i| xyz.data[i * 9 + 3..i * 9 + 6].to_vec()).collect();
    let mut rbf = geom::rbf_ca(&ca, l).reshape(&[1, l, l, geom::D_COUNT]);
    let pos = PositionalEncoding2D::load(&p.sub("pos"), true);
    let pe = pos.forward(&seq_unmasked, &idx, &bond_feats, &dist_matrix, &same_chain);
    for (i, v) in rbf.data.iter_mut().enumerate() {
        *v += pe.data[i];
    }
    chk("rbf_feat (vs msa2msa arg 2)", &rbf.data, &io.get(&format!("in::{B}.msa2msa.2")).data);

    // ---- msa2msa ----------------------------------------------------------
    let mut ctx = ctx_at(&io, &format!("{B}.msa2msa"));
    let m = MSAPairStr2MSA::load(&p.sub("msa2msa"), 8, 8, true, 0.15f64);
    let msa_o = m.forward(&msa, &pair, &rbf, &state, &mut ctx);
    chk("msa2msa", &msa_o.data, &io.get(&format!("out::{B}.msa2msa")).data);

    // ---- msa2pair ---------------------------------------------------------
    let m2 = MSA2Pair::load(&p.sub("msa2pair"));
    let pair_o = m2.forward(&msa_o, &pair);
    chk("msa2pair", &pair_o.data, &io.get(&format!("out::{B}.msa2pair")).data);

    // ---- pair2pair --------------------------------------------------------
    let mut ctx = ctx_at(&io, &format!("{B}.pair2pair"));
    let pp = PairStr2Pair::load(&p.sub("pair2pair"), 6, 32, 0.15f64);
    let pair_o2 = pp.forward(&pair_o, &rbf, &state, &mut ctx);
    chk("pair2pair", &pair_o2.data, &io.get(&format!("out::{B}.pair2pair")).data);

    // ---- str2str (SE3 transformer) ---------------------------------------
    let mut ctx = ctx_at(&io, &format!("{B}.str2str"));
    let c = cfg();
    let s2s = Str2Str::load(
        &p.sub("str2str"),
        c.se3_num_layers,
        c.l0_in,
        c.l1_in,
        c.num_channels,
        c.num_degrees,
        c.l0_out,
        c.l1_out,
        c.n_heads,
        c.div,
        c.p_drop,
    );
    let extra_l1 = rfd2::chiral::chiral_grads(&xyz.data, l, 3, &chirals);
    let out = s2s.forward(
        &msa_o,
        &pair_o2,
        &xyz.data,
        3,
        &state,
        &idx,
        &rotation_mask,
        &bond_feats,
        &dist_matrix,
        &atom_frames,
        &is_motif,
        None,
        &extra_l1,
        c.n_extra_l1,
        c.top_k,
        &mut ctx,
    );
    chk("str2str.xyz", &out.xyz, &io.get(&format!("out::{B}.str2str.0")).data);
    chk("str2str.state", &out.state.data, &io.get(&format!("out::{B}.str2str.1")).data);
    chk("str2str.alpha", &out.alpha.data, &io.get(&format!("out::{B}.str2str.2")).data);
    chk("str2str.quat", &out.quat, &io.get(&format!("out::{B}.str2str.3")).data);

    // ---- the whole block --------------------------------------------------
    let mut ctx = ctx_at(&io, B);
    let blk = IterBlock::load(&p, c);
    let mut st = TrackState {
        msa: msa.clone(),
        pair: pair.clone(),
        xyz: xyz.data.clone(),
        state: state.clone(),
        alpha: Tensor::zeros(&[1, l, 20, 2]),
        quat: vec![0.0; l * 4],
    };
    let inp = BlockInputs {
        seq_unmasked: &seq_unmasked,
        idx: &idx,
        bond_feats: &bond_feats,
        dist_matrix: &dist_matrix,
        same_chain: &same_chain,
        chirals: &chirals,
        atom_frames: &atom_frames,
        is_motif: &is_motif,
        rotation_mask: &rotation_mask,
    };
    blk.forward(&mut st, &inp, &mut ctx);
    let ok = [
        chk("block.msa", &st.msa.data, &io.get(&format!("out::{B}.0")).data),
        chk("block.pair", &st.pair.data, &io.get(&format!("out::{B}.1")).data),
        chk("block.xyz", &st.xyz, &io.get(&format!("out::{B}.2")).data),
        chk("block.state", &st.state.data, &io.get(&format!("out::{B}.3")).data),
        chk("block.alpha", &st.alpha.data, &io.get(&format!("out::{B}.4")).data),
        chk("block.quat", &st.quat, &io.get(&format!("out::{B}.5")).data),
    ];
    assert!(ok.iter().all(|b| *b), "extra_block.0 is not bit-exact");
}
