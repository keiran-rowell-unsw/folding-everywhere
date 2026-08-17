//! The two gradient terms the refinement blocks hand to the SE(3) transformer.
//!
//! `IterativeSimulator` runs `n_ref_block` refinement iterations, and each one
//! recomputes `calc_lj_grads` and `calc_chiral_grads` from the *current*
//! backbone before calling `str_refiner`. Those become
//!
//!   `extra_l0` = `dljdalpha.reshape(1, -1, 2*NTOTALDOFS)`      -> [1, L, 40]
//!   `extra_l1` = `cat([dljdxyz[0], dchiraldxyz[0]], dim=1)`    -> [L, 6, 3]
//!
//! so this asserts them in exactly the shape the refiner receives, against the
//! reference's captured `str_refiner` inputs — which is a stronger check than
//! testing either gradient on its own, because it also pins the concatenation
//! order and the reshape.

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
fn refiner_extra_features_match() {
    let Some(io) = open("fixtures/refiner_io/io.safetensors") else { return };
    let Some(step) = open("fixtures/model_pinned/step0.safetensors") else { return };
    let seq = io.get_i64("lj0.seq").0;
    let xyz = io.get("lj0.xyz");
    let alpha = io.get("lj0.alpha");
    let l = seq.len();
    let bond_feats = step.get_i64("rfi.bond_feats").0;
    let dist_matrix = step.get("rfi.dist_matrix").data;
    let chirals = step.get("rfi.chirals").data;

    // ---- LJ: energy gradient, then back through compute_all_atom ----------
    let t = LjTables::new();
    let cfg = LjCfg::default();
    let conv = rfd2::model::xyzconv::XyzConverter::new();
    let xyzaa = conv.compute_all_atom(&seq, &xyz.data, 3, &alpha.data);
    let out = lj_forward(&seq, &xyzaa, &bond_feats, &dist_matrix, &t, &cfg);
    let n = natoms(&seq, &t, cfg.use_h);
    let dxyzaa: Vec<f32> = out.dljedx.iter().map(|v| n * v).collect();
    let g = rfd2::xyzconv_bwd::backward(&seq, &xyz.data, 3, &alpha.data, &dxyzaa);

    // ---- chiral ----------------------------------------------------------
    let dchiral = rfd2::chiral::chiral_grads(&xyz.data, l, 3, &chirals);

    // ---- assemble exactly as `Track_module` does --------------------------
    let mut extra_l1 = vec![0.0f32; l * 6 * 3];
    for i in 0..l {
        extra_l1[i * 18..i * 18 + 9].copy_from_slice(&g.dxyz[i * 9..i * 9 + 9]);
        extra_l1[i * 18 + 9..i * 18 + 18].copy_from_slice(&dchiral[i * 9..i * 9 + 9]);
    }

    let want_l0 = io.get("in::str_refiner.10");
    let want_l1 = io.get("in::str_refiner.11");
    let s0 = parity::compare(&g.dalpha, &want_l0.data);
    let s1 = parity::compare(&extra_l1, &want_l1.data);
    println!("extra_l0 (dljdalpha, [1,{l},40]): {}", s0.summary());
    println!("extra_l1 (lj|chiral, [{l},6,3]) : {}", s1.summary());

    // The chiral half on its own, so a failure says which of the two moved.
    let mut chir_only = vec![0.0f32; l * 9];
    let mut lj_only = vec![0.0f32; l * 9];
    for i in 0..l {
        lj_only[i * 9..i * 9 + 9].copy_from_slice(&want_l1.data[i * 18..i * 18 + 9]);
        chir_only[i * 9..i * 9 + 9].copy_from_slice(&want_l1.data[i * 18 + 9..i * 18 + 18]);
    }
    let sc = parity::compare(&dchiral, &chir_only);
    let sl = parity::compare(&g.dxyz, &lj_only);
    println!("  lj half     : {}", sl.summary());
    println!("  chiral half : {}", sc.summary());

    assert_eq!(sl.exact, sl.n, "lj dxyz not bit-exact: {}", sl.summary());
    assert_eq!(sc.exact, sc.n, "chiral grad not bit-exact: {}", sc.summary());
    assert_eq!(s0.exact, s0.n, "extra_l0 not bit-exact: {}", s0.summary());
    assert_eq!(s1.exact, s1.n, "extra_l1 not bit-exact: {}", s1.summary());
}
