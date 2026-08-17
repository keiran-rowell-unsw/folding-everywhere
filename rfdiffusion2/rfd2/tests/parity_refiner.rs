//! `str_refiner` — the refinement block, from the reference's own captured
//! inputs and RNG state.
//!
//! It is a `Str2Str` like the ones inside the 36 trunk blocks, but three things
//! differ and each is a place a port can be quietly wrong:
//!
//!   * it takes **top-k = 128** neighbours instead of a full graph, so the
//!     SE(3) graph has a different edge set (and top-k is an ordering op — see
//!     the near-tie audit in SOP §5.5);
//!   * it has **two** SE(3) layers, not one;
//!   * its degree-0 input carries 40 extra channels (`dljdalpha`) and its
//!     degree-1 input 6 (`dljdxyz` and `dchiraldxyz`), so the fibers and every
//!     weight shape downstream are different.

use rfd2::model::rf::Arch;
use rfd2::model::str2str::Str2Str;
use rfd2::nn::{Ctx, Params};
use rfd2::parity;
use rfd2::rng::torch::Mt19937;
use rfd2::weights::Weights;
use rfd2::{chemical_gen, geom};
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
fn str_refiner_matches() {
    let Some(io) = open("fixtures/refiner_io/io.safetensors") else { return };
    let Some(w) = open("fixtures/weights/model_state_dict.safetensors") else { return };
    let arch = Arch::rfd173();
    let p = Params::root(&w, "model").sub("simulator").sub("str_refiner");

    let msa = io.get("in::str_refiner.0");
    let pair = io.get("in::str_refiner.1");
    let xyz = io.get("in::str_refiner.2");
    let state = io.get("in::str_refiner.3");
    let idx = io.get_i64("in::str_refiner.4").0;
    let l = idx.len();
    let rotation_mask: Vec<bool> =
        io.get_i64("in::str_refiner.5").0.into_iter().map(|v| v != 0).collect();
    let bond_feats = io.get_i64("in::str_refiner.6").0;
    let dist_matrix = io.get("in::str_refiner.7").data;
    let atom_frames = io.get_i64("in::str_refiner.8").0;
    let is_motif: Vec<bool> =
        io.get_i64("in::str_refiner.9").0.into_iter().map(|v| v != 0).collect();
    let extra_l0 = io.get("in::str_refiner.10");
    let extra_l1 = io.get("in::str_refiner.11");

    let n_extra_l0 = 2 * chemical_gen::NTOTALDOFS;
    let n_extra_l1 = 6;
    let refiner = Str2Str::load(
        &p,
        arch.se3_ref_layers,
        arch.d_state + n_extra_l0,
        3 + n_extra_l1,
        arch.num_channels,
        arch.num_degrees,
        arch.d_state,
        2,
        arch.n_heads,
        arch.div,
        arch.p_drop,
    );

    let bytes: Vec<u8> =
        io.get_i64("rng::str_refiner").0.into_iter().map(|v| v as u8).collect();
    let mut ctx = Ctx::new(Mt19937::from_torch_state(&bytes));

    let out = refiner.forward(
        &msa,
        &pair,
        &xyz.data,
        3,
        &state,
        &idx,
        &rotation_mask,
        &bond_feats,
        &dist_matrix,
        &atom_frames,
        &is_motif,
        Some(&extra_l0.data),
        &extra_l1.data,
        n_extra_l1,
        arch.refiner_topk,
        &mut ctx,
    );

    let _ = geom::is_atom(0); // keep the import honest if the graph moves
    let mut bad = Vec::new();
    for (name, got, key) in [
        ("xyz", out.xyz.as_slice(), "out::str_refiner.0"),
        ("state", out.state.data.as_slice(), "out::str_refiner.1"),
        ("alpha", out.alpha.data.as_slice(), "out::str_refiner.2"),
        ("quat", out.quat.as_slice(), "out::str_refiner.3"),
    ] {
        let want = io.get(key);
        let s = parity::compare(got, &want.data);
        println!("{name:<6} {}", s.summary());
        if s.exact != s.n {
            bad.push(name);
        }
    }
    println!("torch RNG draws: {}", ctx.rng.draws());
    assert!(bad.is_empty(), "str_refiner outputs not bit-exact: {bad:?}");
}
