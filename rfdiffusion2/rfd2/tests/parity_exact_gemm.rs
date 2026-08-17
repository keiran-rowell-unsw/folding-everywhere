//! Stage 0 — is the port's f64 reduction path *correctly rounded*, or is it
//! merely usually right?
//!
//! `docs/BITEXACT.md` argues order-independence from a probability: an f64
//! rounding error is ~9 orders below an fp32 ULP, so the narrowed result is
//! almost always the correctly-rounded one. "Almost always" is measured at
//! ~2e-9 of values (`probe_f64_tie.rs`), and a forward pass produces ~1e9
//! reduction outputs — so a handful of 1-ULP flips per pass is *expected*, and
//! one of them is exactly where the port and the reference disagree.
//!
//! Building with `--features exact` swaps every reduction onto a double-double
//! accumulator (~106 significand bits, see `src/ops/acc.rs`), which makes the
//! port's answer the correctly-rounded one with a margin of ~2^-80 rather than
//! ~2e-9. The experiment is then simply:
//!
//! ```bash
//! RFD2_DUMP=$SCR/fast.bin  cargo test --release --test parity_exact_gemm -- --nocapture
//! RFD2_DUMP=$SCR/exact.bin cargo test --release --features exact \
//!                                     --test parity_exact_gemm -- --nocapture
//! cmp $SCR/fast.bin $SCR/exact.bin
//! ```
//!
//! If the two dumps are identical, the fast path *was* correctly rounded on
//! this input — proven, not argued — and every remaining disagreement with the
//! reference is MKL's f64 GEMM.

use rfd2::model::rf::{Arch, Rfi, RoseTTAFold};
use rfd2::nn::{Ctx, Params};
use rfd2::ops::acc::{exact_mode, Acc};
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

/// The accumulator must actually differ between the two builds, or the
/// experiment proves nothing. This is a sum whose exact value is 1.0 and which
/// plain f64 gets wrong: a large term, many small ones, then the large term
/// cancelled away.
#[test]
fn accumulator_is_what_the_feature_says() {
    let big = 1.0e17f64;
    let mut a = Acc::new();
    a.add(big);
    for _ in 0..1_000_000 {
        a.add(1.0);
    }
    a.add(-big);
    let got = a.get();
    println!(
        "exact_mode = {}   sum(1e17, 1e6 x 1.0, -1e17) = {got}   (exact: 1000000)",
        exact_mode()
    );
    if exact_mode() {
        assert_eq!(got, 1_000_000.0, "the exact accumulator is not compensating");
    } else {
        assert!(
            got != 1_000_000.0,
            "the default accumulator compensated — the two builds are not distinct, \
             so the A/B measurement would be vacuous"
        );
    }
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

/// One full forward, dumped so the two builds can be compared byte for byte,
/// and simultaneously scored against the reference so the run says whether the
/// residual moved.
#[test]
fn whole_network_under_this_accumulator() {
    let Some(f) = open("fixtures/model_pinned/step0.safetensors") else {
        return;
    };
    let Some(w) = open("fixtures/weights/model_state_dict.safetensors") else {
        return;
    };
    let model = RoseTTAFold::load(&Params::root(&w, "model"), Arch::rfd173());
    let rfi = rfi_from(&f);
    let bytes: Vec<u8> = f
        .get_i64("rng_state_at_model_entry")
        .0
        .into_iter()
        .map(|v| v as u8)
        .collect();
    let mut ctx = Ctx::new(Mt19937::from_torch_state(&bytes));

    let t0 = std::time::Instant::now();
    let out = model.forward(&rfi, &mut ctx);
    let secs = t0.elapsed().as_secs_f64();
    println!(
        "exact_mode = {}   full forward in {secs:.1} s",
        exact_mode()
    );

    // the four trunk outputs are what the residual is measured on
    let rows: [(&str, &[f32], &str); 4] = [
        ("simulator.msa", &out.sim.msa.data, "out::model.simulator.0"),
        ("simulator.pair", &out.sim.pair.data, "out::model.simulator.1"),
        ("simulator.xyzaa", &out.sim.xyzallatom, "out::model.simulator.4"),
        ("simulator.state", &out.sim.state.data, "out::model.simulator.5"),
    ];
    for (name, got, key) in rows {
        let want = f.get(key).data;
        let s = parity::compare(got, &want);
        let rms = (want.iter().map(|v| (*v as f64).powi(2)).sum::<f64>() / want.len() as f64)
            .sqrt();
        println!(
            "  {name:<18} {} / {} bit-identical   max|d|/rms {:.3e}",
            s.exact,
            s.n,
            s.max_abs as f64 / rms
        );
    }

    // The dump: every trunk output, concatenated, little-endian f32. Compared
    // across builds with `cmp`, so nothing here has to interpret it.
    if let Ok(path) = std::env::var("RFD2_DUMP") {
        let mut buf: Vec<u8> = Vec::new();
        for (_, got, _) in rows {
            for v in got {
                buf.extend_from_slice(&v.to_le_bytes());
            }
        }
        std::fs::write(&path, &buf).expect("write dump");
        println!("  wrote {path}  ({} bytes)", buf.len());
    } else {
        println!("  (set RFD2_DUMP=<path> to write the comparison dump)");
    }
}
