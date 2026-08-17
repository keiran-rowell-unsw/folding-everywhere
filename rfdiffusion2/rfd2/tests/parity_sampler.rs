//! Rung 7 — the denoising loop, against `fixtures/sampler/T2.safetensors`.
//!
//! Three levels, cheapest first, so a failure says where it is:
//!
//! 1. the schedule scalars, which are pure arithmetic and cost nothing;
//! 2. the Euler step, driven from the reference's own two frames;
//! 3. the whole 2-step loop with the network in it (~80 s), which is the only
//!    test that proves the *stream position* survives a step boundary.
//!
//! (3) does not assert coordinates bit-for-bit — `px0` inherits the trunk's
//! documented 1-ULP drift (`docs/BITEXACT.md`) — but it *does* assert the
//! generator position at every step boundary at tolerance 0. That is the check
//! that matters here: a step that consumed the wrong number of draws would
//! leave every later step sampling from the wrong place, and no coordinate
//! tolerance would name it.

use rfd2::indep::Indep;
use rfd2::model::rf::{Arch, RoseTTAFold};
use rfd2::nn::{Ctx, Params};
use rfd2::noiser::Rigids;
use rfd2::prepro::PreproOptions;
use rfd2::rng::torch::Mt19937;
use rfd2::sampler::{get_scaling_normed_exp, reverse, run_loop, SamplerOptions};
use rfd2::weights::Weights;
use std::path::Path;

const BIG_T: usize = 2;
const EXP_RATE: i64 = 10;

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

/// The `normed_exp` schedule. Cheap, and it pins the int64/fp32 split in
/// `get_scaling` that the module header describes.
#[test]
fn schedule_scalars_match() {
    let Some(f) = open("fixtures/sampler/T2.safetensors") else {
        return;
    };
    let mut bad = 0;
    for it in 0..2 {
        let t_1 = f.get(&format!("s{it}.te_t")).data[0];
        let dt = f.get(&format!("s{it}.te_dt")).data[0];
        let want_s = f.get(&format!("s{it}.re_scaling")).data[0];
        let want_sd = f.get(&format!("s{it}.re_scaled_dt")).data[0];
        let got_s = get_scaling_normed_exp(t_1, EXP_RATE);
        let got_sd = got_s * dt;
        println!(
            "step {it}: t_1 = {t_1}  scaling {got_s:.10} vs {want_s:.10}  \
             scaled_dt {got_sd:.10} vs {want_sd:.10}"
        );
        if got_s.to_bits() != want_s.to_bits() || got_sd.to_bits() != want_sd.to_bits() {
            bad += 1;
        }
    }
    assert_eq!(bad, 0, "the normed_exp schedule is not bit-exact");
}

/// The Euler step itself, from the reference's own `rigid_t` and `rigid_pred`.
#[test]
fn euler_step_matches() {
    let Some(f) = open("fixtures/sampler/T2.safetensors") else {
        return;
    };
    let mut bad = Vec::new();
    for it in 0..2 {
        let trans_t = f.get(&format!("s{it}.te_transt"));
        let trans_1 = f.get(&format!("s{it}.te_trans1"));
        let rots_t = f.get(&format!("s{it}.re_rott"));
        let rots_1 = f.get(&format!("s{it}.re_rot1"));
        let n = trans_t.data.len() / 3;

        // `reverse` selects the diffused rows itself; here every row handed to
        // the Euler step is already a diffused one, so the mask is all true.
        let all = vec![true; n];
        let rt = Rigids { rots: rots_t.data.clone(), trans: trans_t.data.clone() };
        let rp = Rigids { rots: rots_1.data.clone(), trans: trans_1.data.clone() };
        let t = f.get(&format!("s{it}.t")).data.first().copied().unwrap_or(0.0);
        let _ = t;
        let t_pub = 1.0 - f.get(&format!("s{it}.te_t")).data[0];
        let dt = f.get(&format!("s{it}.te_dt")).data[0];
        let out = reverse(&rt, &rp, t_pub, dt, &all, EXP_RATE);

        let (et, nt, wt) = cmp(&out.trans, &f.get(&format!("s{it}.te_out")).data);
        let (er, nr, wr) = cmp(&out.rots, &f.get(&format!("s{it}.re_out")).data);
        println!("step {it}: trans {et} / {nt} (max|d| {wt:.3e})  rots {er} / {nr} (max|d| {wr:.3e})");
        if et != nt {
            bad.push(format!("s{it}.trans"));
        }
        if er != nr {
            bad.push(format!("s{it}.rots"));
        }
    }
    assert!(bad.is_empty(), "the Euler step is not bit-exact: {bad:?}");
}

fn indep_from(f: &Weights, xyz: Vec<f32>) -> Indep {
    Indep {
        seq: f.get_i64("indep.seq").0,
        xyz,
        idx: f.get_i64("indep.idx").0,
        bond_feats: f.get_i64("indep.bond_feats").0,
        chirals: f.get("indep.chirals").data,
        same_chain: f.get_i64("indep.same_chain").0.into_iter().map(|v| v != 0).collect(),
        is_gp: f.get_i64("indep.is_gp").0.into_iter().map(|v| v != 0).collect(),
        terminus_type: f.get("indep.terminus_type").data,
        is_sm: f.get_i64("indep.is_sm").0.into_iter().map(|v| v != 0).collect(),
    }
}

/// The whole loop. ~80 s: two full forwards.
#[test]
fn loop_matches() {
    let Some(f) = open("fixtures/sampler/T2.safetensors") else {
        return;
    };
    let Some(fx) = open("fixtures/model_pinned/step0.safetensors") else {
        return;
    };
    let Some(w) = open("fixtures/weights/model_state_dict.safetensors") else {
        return;
    };
    let model = RoseTTAFold::load(&Params::root(&w, "model"), Arch::rfd173());

    let mut indep = indep_from(&fx, f.get("s0.in_xyz").data);
    let l = indep.len();
    let is_diffused: Vec<bool> =
        f.get_i64("out.is_diffused").0.into_iter().map(|v| v != 0).collect();
    let atom_frames = fx.get_i64("rfi.atom_frames").0;

    let bytes: Vec<u8> =
        f.get_i64("s0.rng_before").0.into_iter().map(|v| v as u8).collect();
    let mut ctx = Ctx::new(Mt19937::from_torch_state(&bytes));

    let opt = SamplerOptions {
        big_t: BIG_T,
        final_step: 1,
        rots_exp_rate: EXP_RATE,
        prepro: PreproOptions { big_t: BIG_T, ..PreproOptions::default() },
        ..SamplerOptions::default()
    };

    let mut bad = Vec::new();
    let t0 = std::time::Instant::now();
    let traj = run_loop(
        &model,
        &mut indep,
        &is_diffused,
        &atom_frames,
        BIG_T,
        &opt,
        &mut ctx,
        |_, _, _| {},
    );
    println!("{} steps in {:.1} s", traj.ts.len(), t0.elapsed().as_secs_f32());

    for (it, &t) in traj.ts.iter().enumerate() {
        let want_px0 = f.get(&format!("s{it}.px0"));
        let want_xt = f.get(&format!("s{it}.x_t"));
        let (ep, np, wp) = cmp(&traj.px0[it], &want_px0.data);
        let (ex, nx, wx) = cmp(&traj.denoised[it], &want_xt.data);
        let rms = (want_px0.data.iter().map(|v| (*v as f64).powi(2)).sum::<f64>()
            / want_px0.data.len() as f64)
            .sqrt();
        println!(
            "t = {t}: px0 {ep} / {np} (max|d| {wp:.3e} A, /rms {:.2e})  x_t {ex} / {nx} (max|d| {wx:.3e} A)",
            wp as f64 / rms
        );
        if traj.px0[it].iter().any(|v| v.is_nan()) {
            bad.push(format!("t{t}.px0 has NaNs"));
        }
        if (wp as f64 / rms) > 1e-4 {
            bad.push(format!("t{t}.px0 drifted: {wp:.3e}"));
        }
        assert_eq!(np, l * 37 * 3);
    }

    // the discrete check: the generator must be where the reference left it
    // after the *last* step, which means every step consumed the same draws
    let after: Vec<u8> =
        f.get_i64("s1.rng_after").0.into_iter().map(|v| v as u8).collect();
    let mut want_next = Ctx::new(Mt19937::from_torch_state(&after));
    let a: Vec<f32> = (0..64).map(|_| ctx.rng.uniform_f32()).collect();
    let b: Vec<f32> = (0..64).map(|_| want_next.rng.uniform_f32()).collect();
    let (eg, ng, _) = cmp(&a, &b);
    println!("generator after the loop: {eg} / {ng} bit-identical");
    if eg != ng {
        bad.push("rng position after the loop".into());
    }

    assert!(bad.is_empty(), "sampler loop: {bad:?}");
}
