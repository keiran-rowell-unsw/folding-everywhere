//! Rung 6e — the **whole** `IterativeSimulator`: 4 extra + 32 main blocks and
//! then the 4 refinement blocks, run from the reference's own `rfi.*` and the
//! RNG state at model entry.
//!
//! This is the first test that closes the loop between the network and the
//! gradient sub-ports: each refinement iteration recomputes `calc_lj_grads` and
//! `calc_chiral_grads` from the coordinates the previous one produced, so a
//! single wrong ULP anywhere in either reverse pass compounds through four
//! iterations and moves `xyzallatom`.

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
fn simulator_including_refinement() {
    let Some(f) = open("fixtures/model_pinned/step0.safetensors") else { return };
    let Some(w) = open("fixtures/weights/model_state_dict.safetensors") else { return };
    if !f.has("out::model.simulator.4") {
        eprintln!("SKIP: fixture has no simulator output capture");
        return;
    }
    let arch = Arch::rfd173();
    let model = RoseTTAFold::load(&Params::root(&w, "model"), arch);
    let rfi = rfi_from(&f);
    let l = rfi.idx.len();

    let bytes: Vec<u8> =
        f.get_i64("rng_state_at_model_entry").0.into_iter().map(|v| v as u8).collect();
    let mut ctx = Ctx::new(Mt19937::from_torch_state(&bytes));

    let t0 = std::time::Instant::now();
    let out = model.forward_blocks(&rfi, &mut ctx);
    println!(
        "40 blocks (36 trunk + 4 refinement) in {:.1} s; {} torch draws",
        t0.elapsed().as_secs_f64(),
        ctx.rng.draws()
    );
    assert_eq!(out.xyz_stack.len(), 40, "wrong number of stacked blocks");

    // `out::model.simulator.2` is the stacked xyz `[40, 1, L, 3, 3]`; walking it
    // block by block shows exactly where the run stops agreeing.
    let want_stack = f.get("out::model.simulator.2");
    let per = l * 9;
    println!("per-block xyz agreement along the stack:");
    let mut first_div = None;
    for b in 0..40 {
        let s = parity::compare(&out.xyz_stack[b], &want_stack.data[b * per..(b + 1) * per]);
        let tag = if b < 4 {
            format!("extra_block.{b}")
        } else if b < 36 {
            format!("main_block.{}", b - 4)
        } else {
            format!("ref_block.{}", b - 36)
        };
        if s.exact != s.n && first_div.is_none() {
            first_div = Some(b);
        }
        println!("  {tag:<16} {}", s.summary());
    }

    let sa = parity::compare(&out.xyzallatom, &f.get("out::model.simulator.4").data);
    let ss = parity::compare(&out.state.data, &f.get("out::model.simulator.5").data);
    println!("xyzallatom  {}", sa.summary());
    println!("state       {}", ss.summary());

    // `main_block.0`'s row attention carries a documented <= 1 ULP disagreement
    // with MKL's f64 GEMM (docs/BITEXACT.md), so the tail of the stack inherits
    // it. What this test asserts is that nothing *else* moved: the first four
    // blocks stay bit-identical, and the end-to-end residual stays at round-off.
    for b in 0..4 {
        let s = parity::compare(&out.xyz_stack[b], &want_stack.data[b * per..(b + 1) * per]);
        assert_eq!(s.exact, s.n, "extra_block.{b} regressed: {}", s.summary());
    }
    assert!(
        sa.max_abs < 1e-3 && sa.cosine > 0.9999999,
        "all-atom coordinates diverged: {}",
        sa.summary()
    );
}
