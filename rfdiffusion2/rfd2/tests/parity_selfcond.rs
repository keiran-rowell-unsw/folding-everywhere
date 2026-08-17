//! Rung 8 — `inference.str_self_cond`: the model is shown its own previous
//! prediction in template slot 1.
//!
//! Measured to be a real behavioural change before being tested, so the test is
//! not vacuous: with the flag on, the reference's step-1 `px0` moves by up to
//! 0.97 A. Step 0 is untouched, because the guard is `t < T` and the first step
//! of a full trajectory has `t == T`.

use rfd2::indep::Indep;
use rfd2::model::rf::{Arch, RoseTTAFold};
use rfd2::nn::{Ctx, Params};
use rfd2::prepro::PreproOptions;
use rfd2::rng::torch::Mt19937;
use rfd2::sampler::{run_loop, SamplerOptions};
use rfd2::weights::Weights;
use std::path::Path;

const BIG_T: usize = 2;

fn open(rel: &str) -> Option<Weights> {
    let p = format!("{}/../{rel}", env!("CARGO_MANIFEST_DIR"));
    if !Path::new(&p).exists() {
        eprintln!("SKIP: {p} missing");
        return None;
    }
    Some(Weights::open(&p).expect("open"))
}

fn cmp(got: &[f32], want: &[f32]) -> (usize, usize, f32) {
    assert_eq!(got.len(), want.len());
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

#[test]
fn self_conditioned_loop_matches() {
    let Some(f) = open("fixtures/sampler/T2_selfcond.safetensors") else {
        return;
    };
    let Some(fx) = open("fixtures/model_pinned/step0.safetensors") else {
        return;
    };
    let Some(w) = open("fixtures/weights/model_state_dict.safetensors") else {
        return;
    };
    // The fixture must be the self-conditioned capture, or the test is
    // vacuous. `Weights` does not expose safetensors metadata, so this checks
    // the property that actually distinguishes the two runs: with the flag on,
    // step 1's px0 moves by ~1 A, while step 0 (where the `t < T` guard blocks
    // self-conditioning) is untouched.
    if let Some(base) = open("fixtures/sampler/T2.safetensors") {
        let (e0, n0, _) = cmp(&f.get("s0.px0").data, &base.get("s0.px0").data);
        let (e1, n1, d1) = cmp(&f.get("s1.px0").data, &base.get("s1.px0").data);
        println!("fixture vs the non-self-cond run: s0 {e0}/{n0} identical, s1 {e1}/{n1} (max|d| {d1:.3e} A)");
        assert_eq!(e0, n0, "step 0 should be unaffected by self-conditioning");
        assert!(e1 < n1, "step 1 is identical — this fixture is NOT self-conditioned");
    }

    let model = RoseTTAFold::load(&Params::root(&w, "model"), Arch::rfd173());
    let mut indep = Indep {
        seq: fx.get_i64("indep.seq").0,
        xyz: f.get("s0.in_xyz").data,
        idx: fx.get_i64("indep.idx").0,
        bond_feats: fx.get_i64("indep.bond_feats").0,
        chirals: fx.get("indep.chirals").data,
        same_chain: fx.get_i64("indep.same_chain").0.into_iter().map(|v| v != 0).collect(),
        is_gp: fx.get_i64("indep.is_gp").0.into_iter().map(|v| v != 0).collect(),
        terminus_type: fx.get("indep.terminus_type").data,
        is_sm: fx.get_i64("indep.is_sm").0.into_iter().map(|v| v != 0).collect(),
    };
    let is_diffused: Vec<bool> =
        f.get_i64("out.is_diffused").0.into_iter().map(|v| v != 0).collect();
    let atom_frames = fx.get_i64("rfi.atom_frames").0;
    let bytes: Vec<u8> =
        f.get_i64("s0.rng_before").0.into_iter().map(|v| v as u8).collect();
    let mut ctx = Ctx::new(Mt19937::from_torch_state(&bytes));

    let opt = SamplerOptions {
        big_t: BIG_T,
        final_step: 1,
        rots_exp_rate: 10,
        str_self_cond: true,
        partial_t: None,
        prepro: PreproOptions { big_t: BIG_T, ..PreproOptions::default() },
    };
    let t0 = std::time::Instant::now();
    let traj = run_loop(
        &model, &mut indep, &is_diffused, &atom_frames, BIG_T, &opt, &mut ctx, |_, _, _| {},
    );
    println!("{} self-conditioned steps in {:.1} s", traj.ts.len(), t0.elapsed().as_secs_f32());

    let mut bad = Vec::new();
    for (it, &t) in traj.ts.iter().enumerate() {
        let want = f.get(&format!("s{it}.px0"));
        let (e, n, worst) = cmp(&traj.px0[it], &want.data);
        let rms = (want.data.iter().map(|v| (*v as f64).powi(2)).sum::<f64>()
            / want.data.len() as f64)
            .sqrt();
        println!(
            "t = {t}: px0 {e} / {n}  max|d| {worst:.3e} A  (/rms {:.2e})",
            worst as f64 / rms
        );
        if (worst as f64 / rms) > 1e-4 {
            bad.push(format!("t{t} px0 drifted {worst:.3e}"));
        }
    }

    // the discrete check: self-conditioning must not consume any randomness, so
    // the generator has to land exactly where the reference left it
    let after: Vec<u8> =
        f.get_i64("s1.rng_after").0.into_iter().map(|v| v as u8).collect();
    let mut want_next = Ctx::new(Mt19937::from_torch_state(&after));
    let a: Vec<f32> = (0..64).map(|_| ctx.rng.uniform_f32()).collect();
    let b: Vec<f32> = (0..64).map(|_| want_next.rng.uniform_f32()).collect();
    let (eg, ng, _) = cmp(&a, &b);
    println!("generator after the loop: {eg} / {ng} bit-identical");
    if eg != ng {
        bad.push("rng position".into());
    }
    assert!(bad.is_empty(), "self-conditioned loop: {bad:?}");
}
