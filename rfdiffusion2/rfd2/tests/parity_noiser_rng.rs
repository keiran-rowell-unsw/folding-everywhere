//! Rung 4e stage 3, first gate: reproduce every RNG draw `sample_init` makes.
//!
//! Measured by `python/gen_noiser.py`, `sample_init` makes exactly nine draws,
//! and each one is checked here from the reference's own captured generator
//! state. Values cannot agree until the draws do, and a *missed* draw is worse
//! than a wrong one — it shifts every later draw in the stream.
//!
//! ```text
//!  0 normal (50,3)   add_fake_frame_legs        <- diffuse
//!  1 normal (50,3)   add_fake_frame_legs        <- diffuse
//!  2 randn  (1,71,3) sample_gaussian            <- _corrupt_trans
//!  3 randn  (1,71,3) sample_vector              <- igso3.sample
//!  4 rand   (1,71)   sample_angle               <- igso3.sample
//!  5 rand   (71,2)   atom37_from_rigid          <- diffuse        (psi_pred)
//!  6 normal (50,3)   add_fake_frame_legs        <- add_fake_peptide_frame
//!  7 normal (50,3)   add_fake_frame_legs        <- add_fake_peptide_frame
//!  8 rand   (71,2)   atom37_from_rigid          <- idealize_peptide_frames
//! ```
//!
//! Two of those are worth naming. `atom37_from_rigid` looks purely geometric and
//! is not — it draws `psi_pred`. And `add_fake_peptide_frame` runs the whole
//! fake-leg + idealization sequence a *second* time, after the noiser.

use rfd2::rng::torch::{randn, Mt19937};
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

/// The draw kind, in stream order.
const KIND: [&str; 9] = [
    "normal", "normal", "randn", "randn", "rand", "rand", "normal", "normal", "rand",
];

#[test]
fn every_draw_reproduces() {
    let Some(f) = open("fixtures/noiser/stages.safetensors") else { return };

    let mut bad = Vec::new();
    for (i, kind) in KIND.iter().enumerate() {
        let sk = format!("draw{i}.rng_before");
        if !f.has(&sk) {
            eprintln!("SKIP draw {i}: not in fixture");
            continue;
        }
        let bytes: Vec<u8> = f.get_i64(&sk).0.into_iter().map(|v| v as u8).collect();
        let mut g = Mt19937::from_torch_state(&bytes);
        let want = f.get(&format!("draw{i}.out"));
        let n = want.data.len();

        // `torch.normal(zeros_like(x), std=1.0)` and `torch.randn` both go
        // through ATen's normal kernel, so they should consume the stream
        // identically; that equivalence is exactly what this asserts.
        let got: Vec<f32> = match *kind {
            "randn" | "normal" => randn(&mut g, n),
            "rand" => (0..n).map(|_| g.uniform_f32()).collect(),
            k => panic!("unknown draw kind {k}"),
        };

        let exact = got.iter().zip(&want.data).filter(|(a, b)| a.to_bits() == b.to_bits()).count();
        let maxd = got
            .iter()
            .zip(&want.data)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        println!(
            "  draw {i} torch.{kind:<7} n={n:<5} {exact}/{n} bit-identical  max|d| {maxd:.3e}"
        );
        if exact != n {
            bad.push(i);
        }
    }
    assert!(bad.is_empty(), "draws not reproduced: {bad:?}");
}
