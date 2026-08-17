//! Rung 6f — `LegacyRoseTTAFoldModule.forward` end to end: embeddings,
//! 36 trunk blocks, 4 refinement blocks and all six auxiliary heads, from the
//! reference's own `rfi.*` and the RNG state captured at model entry.
//!
//! The heads are the last unported piece of the network, and they are also the
//! only place the *pair* track is read out directly, so they are the check that
//! catches a pair-track error the coordinates would hide.

use rfd2::model::rf::{Arch, Rfi, RoseTTAFold};
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

fn rfi_from(f: &Weights) -> Rfi {
    Rfi {
        msa_latent: f.get("rfi.msa_latent"),
        msa_full: f.get("rfi.msa_full"),
        seq: f.get_i64("rfi.seq").0,
        seq_unmasked: f.get_i64("rfi.seq_unmasked").0,
        xyz: f.get("rfi.xyz"),
        sctors: f.get("rfi.sctors"),
        idx: f.get_i64("rfi.idx").0,
        bond_feats: f.get_i64("rfi.bond_feats").0,
        dist_matrix: f.get("rfi.dist_matrix").data,
        chirals: f.get("rfi.chirals").data,
        atom_frames: f.get_i64("rfi.atom_frames").0,
        t1d: f.get("rfi.t1d"),
        t2d: f.get("rfi.t2d"),
        xyz_t: f.get("rfi.xyz_t"),
        alpha_t: f.get("rfi.alpha_t"),
        mask_t: f.get_i64("rfi.mask_t").0.into_iter().map(|v| v != 0).collect(),
        same_chain: f.get_i64("rfi.same_chain").0.into_iter().map(|v| v != 0).collect(),
        is_motif: f.get_i64("rfi.is_motif").0.into_iter().map(|v| v != 0).collect(),
    }
}

#[test]
fn whole_model_forward() {
    let Some(f) = open("fixtures/model_pinned/step0.safetensors") else { return };
    let Some(w) = open("fixtures/weights/model_state_dict.safetensors") else { return };
    let arch = Arch::rfd173();
    let model = RoseTTAFold::load(&Params::root(&w, "model"), arch);
    let rfi = rfi_from(&f);

    let bytes: Vec<u8> =
        f.get_i64("rng_state_at_model_entry").0.into_iter().map(|v| v as u8).collect();
    let mut ctx = Ctx::new(Mt19937::from_torch_state(&bytes));

    let t0 = std::time::Instant::now();
    let out = model.forward(&rfi, &mut ctx);
    println!("full forward in {:.1} s", t0.elapsed().as_secs_f64());

    let mut rows: Vec<(String, parity::Stats)> = Vec::new();
    let mut push = |name: &str, got: &[f32], want: &[f32]| {
        rows.push((name.to_string(), parity::compare(got, want)));
    };
    push("c6d.dist", &out.c6d.dist.data, &f.get("out::model.c6d_pred.0").data);
    push("c6d.omega", &out.c6d.omega.data, &f.get("out::model.c6d_pred.1").data);
    push("c6d.theta", &out.c6d.theta.data, &f.get("out::model.c6d_pred.2").data);
    push("c6d.phi", &out.c6d.phi.data, &f.get("out::model.c6d_pred.3").data);
    push("aa_pred", &out.logits_aa.data, &f.get("out::model.aa_pred").data);
    push("lddt_pred", &out.lddt.data, &f.get("out::model.lddt_pred").data);
    push("pae_pred", &out.logits_pae.data, &f.get("out::model.pae_pred").data);
    push("pde_pred", &out.logits_pde.data, &f.get("out::model.pde_pred").data);
    push("bind_pred", &[out.p_bind], &f.get("out::model.bind_pred").data);
    push("simulator.msa", &out.sim.msa.data, &f.get("out::model.simulator.0").data);
    push("simulator.pair", &out.sim.pair.data, &f.get("out::model.simulator.1").data);
    push("simulator.xyzaa", &out.sim.xyzallatom, &f.get("out::model.simulator.4").data);
    push("simulator.state", &out.sim.state.data, &f.get("out::model.simulator.5").data);

    for (name, s) in &rows {
        println!("{name:<18} {}", s.summary());
    }
    let exact: Vec<&str> =
        rows.iter().filter(|(_, s)| s.exact == s.n).map(|(n, _)| n.as_str()).collect();
    println!("{}/{} outputs bit-identical: {exact:?}", exact.len(), rows.len());

    // WHAT THE HEAD ROWS ACTUALLY PROVE, stated because it is easy to overread:
    // in `RFD_173` the five projection heads are still at their zero
    // initialisation (`AuxiliaryPredictor.reset_parameter` zeros weight *and*
    // bias, and RFdiffusion2 never trains them), so every one of them emits
    // exactly 0.0 and `bind_pred` emits `sigmoid(0) = 0.5`. Measured on the
    // fixture: 0 of 307 501 `c6d.dist` values, 0 of 322 624 `pae` values, etc.
    // are non-zero. So those rows check weight loading, output shape and the
    // channel-first permutes — not arithmetic. They are asserted anyway, at
    // tolerance 0, because a wrong permute or a mis-keyed weight would show.
    for name in ["c6d.dist", "c6d.omega", "c6d.theta", "c6d.phi", "aa_pred",
                 "lddt_pred", "pae_pred", "pde_pred", "bind_pred"] {
        let (_, s) = rows.iter().find(|(n, _)| n == name).unwrap();
        assert_eq!(s.exact, s.n, "{name} not bit-exact: {}", s.summary());
    }

    // The trunk outputs are downstream of `main_block.0`, whose row attention
    // carries a documented <= 1 ULP disagreement with MKL's f64 GEMM
    // (docs/BITEXACT.md), and 40 blocks amplify it. Judge that against the
    // tensor's own scale: a per-element `max_rel` is meaningless where the
    // reference value is ~0, and `pair` has an RMS of 214.
    for (name, want_key, tol) in [
        ("simulator.msa", "out::model.simulator.0", 1e-5),
        ("simulator.pair", "out::model.simulator.1", 1e-5),
        ("simulator.xyzaa", "out::model.simulator.4", 1e-5),
        ("simulator.state", "out::model.simulator.5", 1e-5),
    ] {
        let (_, s) = rows.iter().find(|(n, _)| n == name).unwrap();
        let w = f.get(want_key).data;
        let rms = (w.iter().map(|v| (*v as f64).powi(2)).sum::<f64>() / w.len() as f64).sqrt();
        let scaled = s.max_abs as f64 / rms;
        println!("{name:<18} max|d| / rms = {scaled:.3e}   (rms {rms:.4})");
        assert!(!s.any_nan, "{name} has NaNs");
        assert!(s.cosine > 0.9999999, "{name} diverged: {}", s.summary());
        assert!(scaled < tol, "{name} error is above the fp32 noise floor: {scaled:.3e}");
    }
}
