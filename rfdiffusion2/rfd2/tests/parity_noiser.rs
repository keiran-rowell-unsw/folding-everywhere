//! The noiser, function by function, against `fixtures/noiser/stages.safetensors`.

use rfd2::nn::Ctx;
use rfd2::noiser::add_fake_frame_legs;
use rfd2::rng::torch::Mt19937;
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

/// NaN-aware bitwise equality — the NaN pattern in `xyz` is load-bearing.
fn cmp(got: &[f32], want: &[f32]) -> (usize, usize) {
    assert_eq!(got.len(), want.len(), "len {} vs {}", got.len(), want.len());
    let e = got
        .iter()
        .zip(want)
        .filter(|(a, b)| (a.is_nan() && b.is_nan()) || a.to_bits() == b.to_bits())
        .count();
    (e, got.len())
}

#[test]
fn fake_frame_legs_match() {
    let Some(f) = open("fixtures/noiser/stages.safetensors") else {
        return;
    };
    let xin = f.get("legs0.in");
    let l = xin.shape[0];
    let is_sm: Vec<bool> = f
        .get_i64("d0.in_is_sm")
        .0
        .into_iter()
        .map(|v| v != 0)
        .collect();
    let bytes: Vec<u8> = f
        .get_i64("draw0.rng_before")
        .0
        .into_iter()
        .map(|v| v as u8)
        .collect();
    let mut ctx = Ctx::new(Mt19937::from_torch_state(&bytes));

    let got = add_fake_frame_legs(&xin.data, l, &is_sm, &mut ctx);
    let want = f.get("legs0.out");
    let (e, n) = cmp(&got, &want.data);
    println!(
        "add_fake_frame_legs  {e} / {n} bit-identical  ({} ligand rows, {} draws)",
        is_sm.iter().filter(|s| **s).count(),
        ctx.rng.draws()
    );
    assert_eq!(e, n, "add_fake_frame_legs not bit-exact");

    // and the second call, after the noiser, from its own captured state
    let bytes: Vec<u8> = f
        .get_i64("draw6.rng_before")
        .0
        .into_iter()
        .map(|v| v as u8)
        .collect();
    let mut ctx2 = Ctx::new(Mt19937::from_torch_state(&bytes));
    let xin2 = f.get("legs1.in");
    let got2 = add_fake_frame_legs(&xin2.data, l, &is_sm, &mut ctx2);
    let want2 = f.get("legs1.out");
    let (e2, n2) = cmp(&got2, &want2.data);
    println!("  second call        {e2} / {n2} bit-identical");
    assert_eq!(e2, n2, "second add_fake_frame_legs not bit-exact");
}

#[test]
fn rigid_frames_match() {
    let Some(f) = open("fixtures/noiser/stages.safetensors") else {
        return;
    };
    // `rigids_0` is built from the coordinates AFTER the fake legs are added
    let x = f.get("legs0.out");
    let l = x.shape[0];
    let n_atoms = x.shape[1];
    let (rots, trans) = rfd2::noiser::rigid_frames_from_atom_14(&x.data, l, n_atoms);
    let (er, nr) = cmp(&rots, &f.get("d0.rigids_0_rots").data);
    let (et, nt) = cmp(&trans, &f.get("d0.rigids_0_trans").data);
    println!("rigids_0 rots  {er} / {nr} bit-identical");
    println!("rigids_0 trans {et} / {nt} bit-identical");
    assert_eq!(er, nr, "rigid frame rotations not bit-exact");
    assert_eq!(et, nt, "rigid frame translations not bit-exact");
}

#[test]
fn igso3_sampling_matches() {
    let Some(f) = open("fixtures/noiser/stages.safetensors") else {
        return;
    };
    let omega = f.get("igso3.omega_grid").data;
    let cdf_all = f.get("igso3.cdf");
    let n_omega = omega.len();
    // sigma is hard-coded 1.5 upstream, which buckets to the last row
    let row = cdf_all.shape[0] - 1;
    let cdf = cdf_all.data[row * n_omega..(row + 1) * n_omega].to_vec();
    let ig = rfd2::noiser::Igso3::new(omega, cdf);

    // draw 4 is `sample_angle`'s uniform; the angles it produced are captured
    let p = f.get("draw4.out");
    let got = ig.sample_angle(&p.data);
    let want = f.get("igso3.angles");
    let (e, n) = cmp(&got, &want.data);
    println!("igso3 sample_angle  {e} / {n} bit-identical");
    assert_eq!(e, n, "IGSO3 angle sampling not bit-exact");

    // vectors: randn(1,L,3) normalised
    let v = f.get("draw3.out");
    let l = v.data.len() / 3;
    let mut got_v = vec![0.0f32; v.data.len()];
    for i in 0..l {
        let a = &v.data[i * 3..i * 3 + 3];
        // torch.norm(dim=2, keepdim=True) is pinned -> f64 interior, one narrowing
        let nrm =
            ((a[0] as f64 * a[0] as f64 + a[1] as f64 * a[1] as f64 + a[2] as f64 * a[2] as f64)
                .sqrt()) as f32;
        for k in 0..3 {
            got_v[i * 3 + k] = a[k] / nrm;
        }
    }
    let want_v = f.get("igso3.vectors");
    let (ev, nv) = cmp(&got_v, &want_v.data);
    println!("igso3 sample_vector {ev} / {nv} bit-identical");
    assert_eq!(ev, nv, "IGSO3 axis sampling not bit-exact");

    // rotation matrices from axis * angle
    let mut got_r = vec![0.0f32; l * 9];
    for i in 0..l {
        let rv = [
            want_v.data[i * 3] * want.data[i],
            want_v.data[i * 3 + 1] * want.data[i],
            want_v.data[i * 3 + 2] * want.data[i],
        ];
        let m = rfd2::noiser::rotvec_to_rotmat(rv, 1e-7);
        for a in 0..3 {
            for b in 0..3 {
                got_r[i * 9 + a * 3 + b] = m[a][b];
            }
        }
    }
    let want_r = f.get("igso3.sample_out");
    let (er, nr) = cmp(&got_r, &want_r.data);
    println!("igso3 rotmats       {er} / {nr} bit-identical");
    assert_eq!(er, nr, "IGSO3 rotation matrices not bit-exact");
}

#[test]
fn integrated_igso3_sampling_matches_and_advances_rng() {
    let Some(f) = open("fixtures/noiser/stages.safetensors") else {
        return;
    };
    let omega = f.get("igso3.omega_grid").data;
    let cdf_all = f.get("igso3.cdf");
    let n_omega = omega.len();
    let row = cdf_all.shape[0] - 1;
    let cdf = cdf_all.data[row * n_omega..(row + 1) * n_omega].to_vec();
    let ig = rfd2::noiser::Igso3::new(omega, cdf);

    // Start immediately before draw 3.  The integrated call must consume draw
    // 3 (axes) and draw 4 (angles), leaving the stream exactly at draw 5.
    let state3: Vec<u8> = f
        .get_i64("draw3.rng_before")
        .0
        .into_iter()
        .map(|v| v as u8)
        .collect();
    let mut ctx = Ctx::new(Mt19937::from_torch_state(&state3));
    let n = f.get("igso3.angles").data.len();
    let got = ig.sample(n, &mut ctx);
    let want = f.get("igso3.sample_out");
    let (e, total) = cmp(&got, &want.data);
    println!("igso3 integrated     {e} / {total} bit-identical");
    assert_eq!(e, total, "integrated IGSO3 sample not bit-exact");

    let next: Vec<f32> = (0..f.get("draw5.out").data.len())
        .map(|_| ctx.rng.uniform_f32())
        .collect();
    let (erng, nrng) = cmp(&next, &f.get("draw5.out").data);
    println!("igso3 next RNG draw  {erng} / {nrng} bit-identical");
    assert_eq!(erng, nrng, "IGSO3 sampler left RNG at the wrong position");
}

#[test]
fn translation_prior_matches() {
    let Some(f) = open("fixtures/noiser/stages.safetensors") else {
        return;
    };
    let state: Vec<u8> = f
        .get_i64("draw2.rng_before")
        .0
        .into_iter()
        .map(|v| v as u8)
        .collect();
    let mut ctx = Ctx::new(Mt19937::from_torch_state(&state));
    let want = f.get("ot.in_trans_0");
    assert_eq!(want.shape, vec![1, 71, 3]);

    // RFD_173 has center_noise_sample=false.  This comparison also pins the
    // nm-to-Angstrom multiply before Kabsch alignment.
    let got = rfd2::noiser::sample_translation_prior(1, 71, false, &mut ctx);
    let (e, n) = cmp(&got, &want.data);
    println!("translation prior    {e} / {n} bit-identical");
    assert_eq!(e, n, "translation prior not bit-exact");

    // The following operation in the reference is IGSO3's axis draw.
    let next = rfd2::rng::torch::randn(&mut ctx.rng, f.get("draw3.out").data.len());
    let (erng, nrng) = cmp(&next, &f.get("draw3.out").data);
    println!("translation next RNG {erng} / {nrng} bit-identical");
    assert_eq!(
        erng, nrng,
        "translation prior left RNG at the wrong position"
    );
}

#[test]
fn kabsch_alignment_matches() {
    let Some(f) = open("fixtures/noiser/stages.safetensors") else {
        return;
    };
    let p = f.get("ot.in_trans_0");
    let q = f.get("ct.trans_1");
    let got = rfd2::noiser::kabsch_align(&p.data, &q.data);
    let want = f.get("ot.out");
    let s = rfd2::parity::compare(&got, &want.data);
    println!("Kabsch alignment: {}", s.summary());
    assert_eq!(
        s.exact,
        s.n,
        "Kabsch alignment not bit-exact: {}",
        s.summary()
    );
}

#[test]
fn integrated_translation_corruption_matches_and_advances_rng() {
    let Some(f) = open("fixtures/noiser/stages.safetensors") else {
        return;
    };
    let state: Vec<u8> = f
        .get_i64("draw2.rng_before")
        .0
        .into_iter()
        .map(|v| v as u8)
        .collect();
    let mut ctx = Ctx::new(Mt19937::from_torch_state(&state));
    let trans_1 = f.get("ct.trans_1");
    let t = f.get("ct.t").data[0];
    let got = rfd2::noiser::corrupt_trans(&trans_1.data, t, false, &mut ctx);
    let want = f.get("ct.out");
    let (e, n) = cmp(&got, &want.data);
    println!("corrupt_trans        {e} / {n} bit-identical");
    assert_eq!(e, n, "integrated translation corruption not bit-exact");

    let next = rfd2::rng::torch::randn(&mut ctx.rng, f.get("draw3.out").data.len());
    let (erng, nrng) = cmp(&next, &f.get("draw3.out").data);
    println!("corrupt_trans RNG    {erng} / {nrng} bit-identical");
    assert_eq!(
        erng, nrng,
        "translation corruption left RNG at the wrong position"
    );
}

#[test]
fn integrated_rotation_corruption_at_zero_matches_and_advances_rng() {
    let Some(f) = open("fixtures/noiser/stages.safetensors") else {
        return;
    };
    let omega = f.get("igso3.omega_grid").data;
    let cdf_all = f.get("igso3.cdf");
    let n_omega = omega.len();
    let row = cdf_all.shape[0] - 1;
    let ig = rfd2::noiser::Igso3::new(
        omega,
        cdf_all.data[row * n_omega..(row + 1) * n_omega].to_vec(),
    );
    let state: Vec<u8> = f
        .get_i64("draw3.rng_before")
        .0
        .into_iter()
        .map(|v| v as u8)
        .collect();
    let mut ctx = Ctx::new(Mt19937::from_torch_state(&state));
    let r1 = f.get("cr.rotmats_1");
    let t = f.get("geo.t").data[0];
    assert_eq!(t.to_bits(), 0.0f32.to_bits());
    let got = rfd2::noiser::corrupt_rots(&r1.data, t, &ig, &mut ctx);
    let want = f.get("cr.out");
    let (e, n) = cmp(&got, &want.data);
    println!("corrupt_rots(t=0)    {e} / {n} bit-identical");
    assert_eq!(
        e, n,
        "integrated zero-time rotation corruption not bit-exact"
    );

    let next: Vec<f32> = (0..f.get("draw5.out").data.len())
        .map(|_| ctx.rng.uniform_f32())
        .collect();
    let (erng, nrng) = cmp(&next, &f.get("draw5.out").data);
    println!("corrupt_rots RNG     {erng} / {nrng} bit-identical");
    assert_eq!(
        erng, nrng,
        "rotation corruption left RNG at the wrong position"
    );
}

#[test]
fn nonzero_geodesic_matches() {
    let Some(f) = open("fixtures/noiser/stages.safetensors") else {
        return;
    };
    let target = f.get("geo.mat");
    let base = f.get("geo.base_mat");
    let t = f.get("geo.quarter_t").data[0];
    let got = rfd2::noiser::geodesic_t(t, &target.data, &base.data);
    let want = f.get("geo.quarter_out");
    let (e, n) = cmp(&got, &want.data);
    let s = rfd2::parity::compare(&got, &want.data);
    println!(
        "geodesic_t(0.25)     {e} / {n} bit-identical  {}",
        s.summary()
    );
    assert_eq!(e, n, "nonzero SO(3) geodesic not bit-exact");
}

#[test]
fn forward_marginal_matches_and_advances_rng() {
    let Some(f) = open("fixtures/noiser/stages.safetensors") else {
        return;
    };
    let omega = f.get("igso3.omega_grid").data;
    let cdf_all = f.get("igso3.cdf");
    let n_omega = omega.len();
    let row = cdf_all.shape[0] - 1;
    let ig = rfd2::noiser::Igso3::new(
        omega,
        cdf_all.data[row * n_omega..(row + 1) * n_omega].to_vec(),
    );
    let state: Vec<u8> = f
        .get_i64("draw2.rng_before")
        .0
        .into_iter()
        .map(|v| v as u8)
        .collect();
    let mut ctx = Ctx::new(Mt19937::from_torch_state(&state));
    let r0 = rfd2::noiser::Rigids {
        rots: f.get("d0.rigids_0_rots").data,
        trans: f.get("d0.rigids_0_trans").data,
    };
    let mask: Vec<bool> = f
        .get_i64("d0.in_is_diffused")
        .0
        .into_iter()
        .map(|v| v != 0)
        .collect();
    let normalized_t = f.get("d0.in_t").data[0] / 2.0;
    let got = rfd2::noiser::forward_marginal(&r0, normalized_t, &mask, false, &ig, &mut ctx);
    let (et, nt) = cmp(&got.trans, &f.get("d0.rigids_t_trans").data);
    let (er, nr) = cmp(&got.rots, &f.get("d0.rigids_t_rots").data);
    println!("forward_marginal trans {et} / {nt}; rots {er} / {nr} bit-identical");
    assert_eq!(et, nt, "forward_marginal translations not bit-exact");
    assert_eq!(er, nr, "forward_marginal rotations not bit-exact");

    let next: Vec<f32> = (0..f.get("draw5.out").data.len())
        .map(|_| ctx.rng.uniform_f32())
        .collect();
    let (en, nn) = cmp(&next, &f.get("draw5.out").data);
    println!("forward_marginal RNG   {en} / {nn} bit-identical");
    assert_eq!(en, nn, "forward_marginal left RNG at the wrong position");
}
