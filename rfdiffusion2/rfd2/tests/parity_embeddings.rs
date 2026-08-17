//! Rung 6a — the input embeddings, recycling and the template stack.
//!
//! Inputs are the reference's own `rfi.*` (captured by `python/ref_dump.py
//! --pinned`), so a failure here is an embedding bug and not a featurization
//! bug — featurization has its own rung. Targets are the forward-hook captures
//! `out::model.*`, i.e. tensors produced by unmodified upstream code.
//!
//! Tolerance is **exactly 0**: the reference ran pinned, so every op on both
//! sides accumulates in f64 and rounds to f32 once.

use rfd2::model::embeddings::{BondEmb, ExtraEmb, MsaEmb, Recycling, TemplEmb};
use rfd2::nn::{Ctx, Params};
use rfd2::rng::torch::Mt19937;
use rfd2::parity;
use rfd2::tensor::Tensor;
use rfd2::weights::Weights;
use std::path::Path;

fn root() -> &'static str {
    env!("CARGO_MANIFEST_DIR")
}

fn open(rel: &str) -> Option<Weights> {
    let path = format!("{}/../{rel}", root());
    if !Path::new(&path).exists() {
        eprintln!("SKIP: {path} missing");
        return None;
    }
    Some(Weights::open(&path).expect("open"))
}

fn fixtures() -> Option<(Weights, Weights)> {
    let f = open("fixtures/model_pinned/step0.safetensors")?;
    let w = open("fixtures/weights/model_state_dict.safetensors")?;
    Some((f, w))
}

fn report(label: &str, got: &[f32], want: &[f32]) {
    let s = parity::compare(got, want);
    println!("{label:<28} {}", s.summary());
    assert_eq!(got.len(), want.len(), "{label}: length");
    let bad = got
        .iter()
        .zip(want)
        .position(|(g, w)| g.to_bits() != w.to_bits() && !(g.is_nan() && w.is_nan()));
    if let Some(i) = bad {
        panic!(
            "{label}: first mismatch at {i}: got {} ({:#010x}) want {} ({:#010x}) — {} of {} exact",
            got[i],
            got[i].to_bits(),
            want[i],
            want[i].to_bits(),
            s.exact,
            s.n
        );
    }
}

struct Case {
    seq: Vec<i64>,
    idx: Vec<i64>,
    bond_feats: Vec<i64>,
    dist_matrix: Vec<f32>,
    same_chain: Vec<bool>,
    l: usize,
}

fn case(f: &Weights) -> Case {
    let (seq, s) = f.get_i64("rfi.seq");
    let l = s[s.len() - 1];
    Case {
        seq,
        idx: f.get_i64("rfi.idx").0,
        bond_feats: f.get_i64("rfi.bond_feats").0,
        dist_matrix: f.get("rfi.dist_matrix").data,
        same_chain: f.get_i64("rfi.same_chain").0.into_iter().map(|x| x != 0).collect(),
        l,
    }
}

#[test]
fn latent_emb_matches() {
    let Some((f, w)) = fixtures() else { return };
    let c = case(&f);
    let p = Params::root(&w, "model");
    let m = MsaEmb::load(&p.sub("latent_emb"), true);
    let msa = f.get("rfi.msa_latent");
    let (msa_o, pair_o, state_o) = m.forward(
        &msa,
        &c.seq,
        &c.idx,
        &c.bond_feats,
        &c.dist_matrix,
        &c.same_chain,
    );
    report("latent_emb.msa", &msa_o.data, &f.get("out::model.latent_emb.0").data);
    report("latent_emb.pair", &pair_o.data, &f.get("out::model.latent_emb.1").data);
    report("latent_emb.state", &state_o.data, &f.get("out::model.latent_emb.2").data);
    println!("latent_emb: L={} d_pair={}", c.l, pair_o.last());
}

#[test]
fn full_emb_matches() {
    let Some((f, w)) = fixtures() else { return };
    let c = case(&f);
    let p = Params::root(&w, "model");
    let m = ExtraEmb::load(&p.sub("full_emb"));
    let got = m.forward(&f.get("rfi.msa_full"), &c.seq);
    report("full_emb", &got.data, &f.get("out::model.full_emb").data);
}

#[test]
fn bond_emb_matches() {
    let Some((f, w)) = fixtures() else { return };
    let c = case(&f);
    let p = Params::root(&w, "model");
    let m = BondEmb::load(&p.sub("bond_emb"));
    let got = m.forward(&c.bond_feats, c.l);
    report("bond_emb", &got.data, &f.get("out::model.bond_emb").data);
}

/// `RecyclingAllFeatures` on the first (and only) pass: every `*_prev` is zeros,
/// which the model's `forward` substitutes for `None`. The distance feature is
/// still real — it comes from `rfi.xyz`, not from a previous cycle.
#[test]
fn recycle_matches() {
    let Some((f, w)) = fixtures() else { return };
    let c = case(&f);
    let p = Params::root(&w, "model");
    let r = Recycling::load(&p.sub("recycle"));

    let xyz = f.get("rfi.xyz"); // [1,L,36,3]
    let natoms = xyz.shape[2];
    let ca: Vec<f32> = (0..c.l)
        .flat_map(|i| {
            let o = (i * natoms + 1) * 3;
            xyz.data[o..o + 3].to_vec()
        })
        .collect();

    let d_msa = 256;
    let d_pair = 192;
    let d_state = 64;
    let msa_prev = Tensor::zeros(&[1, c.l, d_msa]);
    let pair_prev = Tensor::zeros(&[1, c.l, c.l, d_pair]);
    let state_prev = Tensor::zeros(&[1, c.l, d_state]);
    let sctors = f.get("rfi.sctors");

    let (msa, pair, state) =
        r.forward(&msa_prev, &pair_prev, &ca, &state_prev, &sctors, None);
    report("recycle.msa", &msa.data, &f.get("out::model.recycle.0").data);
    report("recycle.pair", &pair.data, &f.get("out::model.recycle.1").data);
    report("recycle.state", &state.data, &f.get("out::model.recycle.2").data);
}

/// The full embedding chain up to and including `templ_emb`, which is the
/// input the simulator actually receives.
#[test]
fn templ_emb_matches() {
    let Some((f, w)) = fixtures() else { return };
    let c = case(&f);
    let p = Params::root(&w, "model");

    let latent = MsaEmb::load(&p.sub("latent_emb"), true);
    let bond = BondEmb::load(&p.sub("bond_emb"));
    let recycle = Recycling::load(&p.sub("recycle"));

    let (_msa, mut pair, mut state) = latent.forward(
        &f.get("rfi.msa_latent"),
        &c.seq,
        &c.idx,
        &c.bond_feats,
        &c.dist_matrix,
        &c.same_chain,
    );
    let be = bond.forward(&c.bond_feats, c.l);
    for (i, v) in pair.data.iter_mut().enumerate() {
        *v += be.data[i];
    }

    let xyz = f.get("rfi.xyz");
    let natoms = xyz.shape[2];
    let ca: Vec<f32> = (0..c.l)
        .flat_map(|i| xyz.data[(i * natoms + 1) * 3..(i * natoms + 1) * 3 + 3].to_vec())
        .collect();
    let msa_prev = Tensor::zeros(&[1, c.l, 256]);
    let pair_prev = Tensor::zeros(&[1, c.l, c.l, 192]);
    let state_prev = Tensor::zeros(&[1, c.l, 64]);
    let (_mr, pr, sr) = recycle.forward(
        &msa_prev,
        &pair_prev,
        &ca,
        &state_prev,
        &f.get("rfi.sctors"),
        None,
    );
    for (i, v) in pair.data.iter_mut().enumerate() {
        *v += pr.data[i];
    }
    for (i, v) in state.data.iter_mut().enumerate() {
        *v += sr.data[i];
    }

    // templ_emb is the first module in the forward that consumes the RNG
    // (dropout is live at inference), so it needs the generator state the
    // reference was at when it entered the network.
    if !f.has("rng_state_at_model_entry") {
        eprintln!("SKIP templ_emb: fixture predates the RNG capture; re-run python/ref_dump.py --pinned");
        return;
    }
    let bytes: Vec<u8> = f
        .get_i64("rng_state_at_model_entry")
        .0
        .into_iter()
        .map(|v| v as u8)
        .collect();
    let mut ctx = Ctx::new(Mt19937::from_torch_state(&bytes));
    let te = TemplEmb::load(&p.sub("templ_emb"), 4, 64, 2);
    let mask_t: Vec<bool> = f.get_i64("rfi.mask_t").0.into_iter().map(|x| x != 0).collect();
    let (pair_o, state_o) = te.forward(
        &f.get("rfi.t1d"),
        &f.get("rfi.t2d"),
        &f.get("rfi.alpha_t"),
        &f.get("rfi.xyz_t"),
        &mask_t,
        &pair,
        &state,
        &mut ctx,
    );
    report("templ_emb.pair", &pair_o.data, &f.get("out::model.templ_emb.0").data);
    report("templ_emb.state", &state_o.data, &f.get("out::model.templ_emb.1").data);
}
