//! The Lennard-Jones energy and its coordinate gradient — the first half of
//! `calc_lj_grads`.
//!
//! Split deliberately: `python/dump_refiner.py` captures `LJLoss.forward`'s own
//! `dljEdx` (the tensor it stashes for its backward), so this half can be
//! checked without the reverse pass through `compute_all_atom` being written
//! yet.

use rfd2::lj::{lj_forward, LjCfg, LjTables};
use rfd2::parity;
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

#[test]
fn lj_energy_and_gradient_match() {
    let Some(io) = open("fixtures/refiner_io/io.safetensors") else { return };
    let Some(step) = open("fixtures/model_pinned/step0.safetensors") else { return };
    if !io.has("ljf.dljEdx") {
        eprintln!("SKIP: fixture predates the LJLoss capture");
        return;
    }
    let seq = io.get_i64("lj0.seq").0;
    let xs = io.get("ljf.xs"); // [1, L, 36, 3]
    let bond_feats = step.get_i64("rfi.bond_feats").0;
    let dist_matrix = step.get("rfi.dist_matrix").data;

    let t = LjTables::new();
    let cfg = LjCfg::default();
    let out = lj_forward(&seq, &xs.data, &bond_feats, &dist_matrix, &t, &cfg);

    let want_e = io.get("ljf.E").data[0];
    println!("lj energy: got {:e} want {:e}", out.energy, want_e);

    let want = io.get("ljf.dljEdx");
    let s = parity::compare(&out.dljedx, &want.data);
    println!("dljEdx: {}", s.summary());
    assert_eq!(out.dljedx.len(), want.data.len());
    assert_eq!(
        out.energy.to_bits(),
        want_e.to_bits(),
        "LJ energy differs: got {} want {}",
        out.energy,
        want_e
    );
    assert_eq!(s.exact, s.n, "dljEdx is not bit-exact");
}
