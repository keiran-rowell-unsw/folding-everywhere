//! `calc_lj_grads` end to end: the LJ gradient back-propagated through
//! `compute_all_atom` to `(dL/dxyz, dL/dalpha)`, the two extra inputs the
//! refinement blocks hand the SE(3) transformer.

use rfd2::lj::{lj_forward, natoms, LjCfg, LjTables};
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
fn lj_grads_match() {
    let Some(io) = open("fixtures/refiner_io/io.safetensors") else { return };
    let Some(step) = open("fixtures/model_pinned/step0.safetensors") else { return };
    let seq = io.get_i64("lj0.seq").0;
    let xyz = io.get("lj0.xyz"); // [1, L, 3, 3]
    let alpha = io.get("lj0.alpha");
    let l = seq.len();
    let bond_feats = step.get_i64("rfi.bond_feats").0;
    let dist_matrix = step.get("rfi.dist_matrix").data;

    let t = LjTables::new();
    let cfg = LjCfg::default();
    let conv = rfd2::model::xyzconv::XyzConverter::new();
    let xyzaa = conv.compute_all_atom(&seq, &xyz.data, 3, &alpha.data);
    let out = lj_forward(&seq, &xyzaa, &bond_feats, &dist_matrix, &t, &cfg);

    // `torch.autograd.grad(natoms * Elj, ...)` -> grad_output = natoms
    let n = natoms(&seq, &t, cfg.use_h);
    let dxyzaa: Vec<f32> = out.dljedx.iter().map(|v| n * v).collect();

    let g = rfd2::xyzconv_bwd::backward(&seq, &xyz.data, 3, &alpha.data, &dxyzaa);

    let want_x = io.get("lj0.dxyz");
    let want_a = io.get("lj0.dalpha");
    let sx = parity::compare(&g.dxyz, &want_x.data);
    let sa = parity::compare(&g.dalpha, &want_a.data);
    println!("dxyz   (L={l}): {}", sx.summary());
    println!("dalpha       : {}", sa.summary());
    assert_eq!(g.dxyz.len(), want_x.data.len());
    assert_eq!(g.dalpha.len(), want_a.data.len());
    // Tolerance is exactly 0. `tests/debug_aa_bwd.rs` checks the 74 stages in
    // between, so a regression here says *where* as well as *that*.
    assert_eq!(sx.exact, sx.n, "dxyz not bit-exact: {}", sx.summary());
    assert_eq!(sa.exact, sa.n, "dalpha not bit-exact: {}", sa.summary());
}
