//! Rung 7, stage 1 — `RFScore.forward_from_rfi`, the wrapper that turns the
//! trunk's quaternion updates into `px0`.
//!
//! Two things are under test that nothing before this touched: openfold's
//! quaternion algebra (`rot_to_quat` via a 4x4 eigendecomposition,
//! `quat_multiply`, `quat_to_rot`) and the per-forward `psi_pred` draw.
//!
//! The quaternion test is run standalone first, because if `rot_to_quat`
//! disagrees the composed frames will too and the whole-wrapper number would
//! not say where.

use rfd2::model::rf::{Arch, Rfi, RoseTTAFold};
use rfd2::nn::{Ctx, Params};
use rfd2::noiser::rigid_frames_from_atom_14;
use rfd2::rng::torch::Mt19937;
use rfd2::score;
use rfd2::weights::Weights;
use std::path::Path;

fn open(rel: &str) -> Option<Weights> {
    let p = format!("{}/../{rel}", env!("CARGO_MANIFEST_DIR"));
    if !Path::new(&p).exists() {
        eprintln!("SKIP: {p} missing");
        return None;
    }
    Some(Weights::open(&p).expect("open"))
}

fn cmp(got: &[f32], want: &[f32]) -> (usize, usize, f32) {
    assert_eq!(got.len(), want.len(), "len {} vs {}", got.len(), want.len());
    let mut e = 0;
    let mut worst = 0.0f32;
    for (g, w) in got.iter().zip(want) {
        if (g.is_nan() && w.is_nan()) || g.to_bits() == w.to_bits() {
            e += 1;
        } else {
            let d = (g - w).abs();
            if d.is_finite() && d > worst {
                worst = d;
            }
        }
    }
    (e, got.len(), worst)
}

/// Bitwise equality up to an overall sign per quaternion. The eigenvector's
/// sign is arbitrary and every consumer is even in it, so a sign flip is not a
/// disagreement — but a *value* difference is.
fn cmp_signed(got: &[f32], want: &[f32], n_comp: usize) -> (usize, usize, usize) {
    assert_eq!(got.len(), want.len());
    let mut exact = 0;
    let mut flipped = 0;
    for (g, w) in got.chunks(n_comp).zip(want.chunks(n_comp)) {
        if g.iter().zip(w).all(|(a, b)| a.to_bits() == b.to_bits()) {
            exact += 1;
        } else if g.iter().zip(w).all(|(a, b)| (-a).to_bits() == b.to_bits()) {
            flipped += 1;
        }
    }
    (exact, flipped, got.len() / n_comp)
}

/// `rot_to_quat` alone: does a canonical f64 Jacobi land on the same fp32
/// values as the pinned LAPACK `eigh` the reference runs?
#[test]
fn rot_to_quat_matches() {
    let Some(f) = open("fixtures/score/step0.safetensors") else {
        return;
    };
    let mats = f.get("rfr0.rots_t_mats");
    let l = mats.data.len() / 9;
    let want = f.get("rfr0.rots_t_quats");

    let got: Vec<f32> = (0..l)
        .flat_map(|i| {
            let r: [f32; 9] = mats.data[i * 9..i * 9 + 9].try_into().unwrap();
            score::rot_to_quat(&r)
        })
        .collect();
    let (exact, flipped, n) = cmp_signed(&got, &want.data, 4);
    println!("rot_to_quat  {exact} exact + {flipped} sign-flipped / {n} quaternions");

    // and the thing that actually matters: the matrix you get back
    let round: Vec<f32> = got
        .chunks(4)
        .flat_map(|q| score::quat_to_rot(&q.try_into().unwrap()))
        .collect();
    let want_round: Vec<f32> = want
        .data
        .chunks(4)
        .flat_map(|q| score::quat_to_rot(&q.try_into().unwrap()))
        .collect();
    let (e, tot, worst) = cmp(&round, &want_round);
    println!("  quat_to_rot of both  {e} / {tot} bit-identical  max|d| {worst:.3e}");

    assert_eq!(exact + flipped, n, "rot_to_quat disagrees beyond a sign");
    assert_eq!(e, tot, "the round-tripped rotation is not bit-exact");
}

/// The composition over all 40 blocks, from the reference's own `rfo`.
#[test]
fn rigids_from_rfo_matches() {
    let Some(f) = open("fixtures/score/step0.safetensors") else {
        return;
    };
    let quat = f.get("ffr0.rfo_quat"); // [1, I, L, 4]
    let xyz = f.get("ffr0.rfo_xyz"); // [I, 1, L, 3, 3]
    let n_iter = quat.shape[1];
    let l = quat.shape[2];
    let mats = f.get("rfr0.rots_t_mats");

    let quat_stack: Vec<Vec<f32>> =
        (0..n_iter).map(|i| quat.data[i * l * 4..(i + 1) * l * 4].to_vec()).collect();
    let xyz_stack: Vec<Vec<f32>> =
        (0..n_iter).map(|i| xyz.data[i * l * 9..(i + 1) * l * 9].to_vec()).collect();

    let got = score::rigids_from_rfo(&quat_stack, &xyz_stack, &mats.data, l);
    assert_eq!(got.len(), n_iter);

    let want_rots = f.get("ffr0.curr_rots");
    let want_trans = f.get("ffr0.curr_trans");
    let flat_rots: Vec<f32> = got.iter().flat_map(|r| r.rots.clone()).collect();
    let flat_trans: Vec<f32> = got.iter().flat_map(|r| r.trans.clone()).collect();
    let (er, nr, wr) = cmp(&flat_rots, &want_rots.data);
    let (et, nt, wt) = cmp(&flat_trans, &want_trans.data);
    println!("rigids_from_rfo  rots {er} / {nr} (max|d| {wr:.3e})  trans {et} / {nt} (max|d| {wt:.3e})");
    assert_eq!(et, nt, "composed translations not bit-exact");
    assert_eq!(er, nr, "composed rotations not bit-exact");
}

/// `compute_backbone` on the *composed* frames — the call that produces `px0`.
///
/// Driven from the reference's own `curr_rigids` and `psi`, so it isolates the
/// wrapper's geometry from the network's documented end-to-end drift. Unlike
/// the `diffuse`-time call in `parity_backbone.rs`, this Rigid's rotation came
/// from quaternions, so it exercises `quat_to_rot` on the way in.
#[test]
fn compute_backbone_on_composed_frames_matches() {
    let Some(f) = open("fixtures/score/step0.safetensors") else {
        return;
    };
    let rots = f.get("ffr0.curr_rots"); // [1, I, L, 3, 3]
    let trans = f.get("ffr0.curr_trans");
    let psi = f.get("ffr0.psi");
    let want = f.get("ffr0.atom37");
    let n_iter = rots.shape[1];
    let l = rots.shape[2];

    let mut got: Vec<f32> = Vec::with_capacity(want.data.len());
    for it in 0..n_iter {
        let r = rfd2::noiser::Rigids {
            rots: rots.data[it * l * 9..(it + 1) * l * 9].to_vec(),
            trans: trans.data[it * l * 3..(it + 1) * l * 3].to_vec(),
        };
        let p = &psi.data[it * l * 2..(it + 1) * l * 2];
        got.extend(rfd2::openfold::compute_backbone(&r, p).0);
    }
    let (e, n, worst) = cmp(&got, &want.data);
    println!("compute_backbone on composed frames  {e} / {n} bit-identical  max|d| {worst:.3e}");
    assert_eq!(e, n, "atom37 from reference frames is not bit-exact");
}

/// The whole wrapper, network included, from the reference's `rfi` and RNG
/// state. Slow (one full forward), and the only test that proves `px0`.
///
/// This one is **not** asserted at tolerance 0, and the reason is documented
/// rather than assumed: `main_block.0`'s row attention carries a <= 1 ULP
/// disagreement with MKL's f64 GEMM (`docs/BITEXACT.md`) which 40 blocks
/// amplify, so `parity_model.rs` already judges the trunk's own outputs
/// against their RMS instead of bit-for-bit. `px0` is downstream of all of it.
/// What *is* asserted at tolerance 0 here is the generator position, because
/// that is discrete and a drift in it would be a real defect.
#[test]
fn forward_from_rfi_matches() {
    let Some(f) = open("fixtures/score/step0.safetensors") else {
        return;
    };
    let Some(fx) = open("fixtures/model_pinned/step0.safetensors") else {
        return;
    };
    let Some(w) = open("fixtures/weights/model_state_dict.safetensors") else {
        return;
    };

    // the rfi the reference handed the wrapper; only `xyz` differs from the
    // model fixture's copy, and only because `prepro` had already mutated it
    let mut rfi = rfi_from(&fx);
    rfi.xyz = f.get("ffr0.in_xyz");

    let model = RoseTTAFold::load(&Params::root(&w, "model"), Arch::rfd173());
    let bytes: Vec<u8> =
        f.get_i64("ffr0.rng_before").0.into_iter().map(|v| v as u8).collect();
    let mut ctx = Ctx::new(Mt19937::from_torch_state(&bytes));

    let t0 = std::time::Instant::now();
    let out = score::forward_from_rfi(&model, &rfi, &mut ctx);
    println!("forward_from_rfi: {:.1} s, {} draws", t0.elapsed().as_secs_f32(), ctx.rng.draws());

    let want37 = f.get("ffr0.atom37");
    let flat: Vec<f32> = out.atom37.iter().flat_map(|a| a.clone()).collect();
    let (e, n, worst) = cmp(&flat, &want37.data);
    println!("  atom37 (all {} blocks)  {e} / {n} bit-identical  max|d| {worst:.3e}", out.atom37.len());

    let l = rfi.seq.len();
    let want_px0 = f.get("step.px0");
    let (ep, np, wp) = cmp(out.px0(), &want_px0.data);
    println!("  px0                     {ep} / {np} bit-identical  max|d| {wp:.3e}");
    assert_eq!(np, l * 37 * 3);

    // judged against the coordinate scale, as `parity_model.rs` does
    let rms = (want_px0.data.iter().map(|v| (*v as f64).powi(2)).sum::<f64>()
        / want_px0.data.len() as f64)
        .sqrt();
    let scaled = wp as f64 / rms;
    println!("  px0 max|d| / rms        {scaled:.3e}  (rms {rms:.3} A, max|d| {wp:.3e} A)");
    assert!(!out.px0().iter().any(|v| v.is_nan()), "px0 has NaNs");
    assert!(
        scaled < 1e-5,
        "px0 error is above the fp32 noise floor: {scaled:.3e}"
    );

    // the generator must be where the reference left it, or the next step's
    // psi is drawn from the wrong place
    let after: Vec<u8> =
        f.get_i64("ffr0.rng_after").0.into_iter().map(|v| v as u8).collect();
    let mut want_next = Ctx::new(Mt19937::from_torch_state(&after));
    let a: Vec<f32> = (0..32).map(|_| ctx.rng.uniform_f32()).collect();
    let b: Vec<f32> = (0..32).map(|_| want_next.rng.uniform_f32()).collect();
    let (eg, ng, _) = cmp(&a, &b);
    println!("  following RNG draw      {eg} / {ng} bit-identical");

    assert_eq!(eg, ng, "the generator is misplaced after the psi draw");
    let _ = (e, n, ep);
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

/// Silence the unused-import warning when the fixtures are absent.
#[allow(dead_code)]
fn _unused(x: &[f32], l: usize) {
    let _ = rigid_frames_from_atom_14(x, l, 36);
}
