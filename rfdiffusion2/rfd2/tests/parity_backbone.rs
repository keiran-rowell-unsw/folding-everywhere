//! `compute_backbone` / `atom37_from_rigid` against the reference's own
//! captured rigids, psi and generator state.
//!
//! Both `sample_init` calls are asserted, not just the first: the second one
//! runs *after* the noiser (from `add_fake_peptide_frame`), so it is the one
//! that proves the stream position is still right at the end of `diffuse`.

use rfd2::nn::Ctx;
use rfd2::noiser::Rigids;
use rfd2::openfold::{atom37_from_rigid, compute_backbone};
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

fn cmp(got: &[f32], want: &[f32]) -> (usize, usize) {
    assert_eq!(got.len(), want.len(), "len {} vs {}", got.len(), want.len());
    let e = got
        .iter()
        .zip(want)
        .filter(|(a, b)| (a.is_nan() && b.is_nan()) || a.to_bits() == b.to_bits())
        .count();
    (e, got.len())
}

fn rigids_from(f: &Weights, tag: &str) -> Rigids {
    Rigids {
        rots: f.get(&format!("{tag}.in_rots")).data,
        trans: f.get(&format!("{tag}.in_trans")).data,
    }
}

/// From the reference's own psi, so this isolates the geometry from the draw.
#[test]
fn compute_backbone_matches() {
    let Some(f) = open("fixtures/noiser/stages.safetensors") else {
        return;
    };
    for n in 0..2 {
        let r = rigids_from(&f, &format!("a37_{n}"));
        let psi = f.get(&format!("cb_{n}.psi"));
        let (atom37, atom14) = compute_backbone(&r, &psi.data);
        let (e37, n37) = cmp(&atom37, &f.get(&format!("cb_{n}.atom37")).data);
        let (e14, n14) = cmp(&atom14, &f.get(&format!("cb_{n}.atom14")).data);
        println!("compute_backbone[{n}]  atom37 {e37} / {n37}  atom14 {e14} / {n14} bit-identical");
        assert_eq!(e37, n37, "compute_backbone[{n}] atom37 not bit-exact");
        assert_eq!(e14, n14, "compute_backbone[{n}] atom14 not bit-exact");
    }
}

/// The whole call including the `psi_pred` draw, started from the generator
/// state the reference had on entry. The post-call generator position is
/// asserted too, because a psi drawn with the wrong element count would still
/// produce a plausible backbone and would shift every later draw.
#[test]
fn atom37_from_rigid_matches_and_advances_rng() {
    let Some(f) = open("fixtures/noiser/stages.safetensors") else {
        return;
    };
    for n in 0..2 {
        let tag = format!("a37_{n}");
        let r = rigids_from(&f, &tag);
        let bytes: Vec<u8> = f
            .get_i64(&format!("{tag}.rng_before"))
            .0
            .into_iter()
            .map(|v| v as u8)
            .collect();
        let mut ctx = Ctx::new(Mt19937::from_torch_state(&bytes));
        let got = atom37_from_rigid(&r, &mut ctx);
        let (e, n_tot) = cmp(&got, &f.get(&format!("{tag}.out")).data);
        println!("atom37_from_rigid[{n}]  {e} / {n_tot} bit-identical, {} draws", ctx.rng.draws());

        // the reference's next state, replayed: same generator, same position
        let after: Vec<u8> = f
            .get_i64(&format!("{tag}.rng_after"))
            .0
            .into_iter()
            .map(|v| v as u8)
            .collect();
        let mut want_next = Ctx::new(Mt19937::from_torch_state(&after));
        let a: Vec<f32> = (0..16).map(|_| ctx.rng.uniform_f32()).collect();
        let b: Vec<f32> = (0..16).map(|_| want_next.rng.uniform_f32()).collect();
        let (eg, ng) = cmp(&a, &b);
        println!("  following RNG draw  {eg} / {ng} bit-identical");

        assert_eq!(e, n_tot, "atom37_from_rigid[{n}] not bit-exact");
        assert_eq!(eg, ng, "atom37_from_rigid[{n}] left the generator misplaced");
    }
}
