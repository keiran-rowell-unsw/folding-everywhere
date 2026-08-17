//! Rung 4b, completed — the whole of `aa_model.Model.prepro`.
//!
//! Every `rfi.*` tensor is compared against `fixtures/model_pinned/step0.safetensors`,
//! which is the same fixture `parity_model.rs` feeds the network from. So a
//! green run here means the port can build the network's input from an `Indep`
//! instead of replaying a captured one — the last thing between rung 4e and the
//! sampler.
//!
//! The two tensors that were never built before are `alpha_t` (60 channels of
//! torsion cos/sin/mask) and `t2d` (68 channels of binned distance and
//! orientation); the rest were green at rung 4b and are re-checked here because
//! the assembler could still lay them out wrong.

use rfd2::indep::Indep;
use rfd2::prepro::{prepro, PreproOptions};
use rfd2::weights::Weights;
use std::path::Path;

/// `diffuser.T` of the captured run, and the step it captured.
const BIG_T: usize = 2;
const T_NOW: usize = 2;

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

fn indep_from(f: &Weights) -> Indep {
    let l = f.get_i64("indep.seq").0.len();
    Indep {
        seq: f.get_i64("indep.seq").0,
        xyz: f.get("indep.xyz").data,
        idx: f.get_i64("indep.idx").0,
        bond_feats: f.get_i64("indep.bond_feats").0,
        chirals: f.get("indep.chirals").data,
        same_chain: f.get_i64("indep.same_chain").0.into_iter().map(|v| v != 0).collect(),
        is_gp: f.get_i64("indep.is_gp").0.into_iter().map(|v| v != 0).collect(),
        terminus_type: f.get("indep.terminus_type").data,
        is_sm: {
            let v: Vec<bool> = f.get_i64("indep.is_sm").0.into_iter().map(|x| x != 0).collect();
            assert_eq!(v.len(), l);
            v
        },
    }
}

#[test]
fn prepro_matches() {
    let Some(f) = open("fixtures/model_pinned/step0.safetensors") else {
        return;
    };
    let mut indep = indep_from(&f);
    let l = indep.len();
    let is_diffused: Vec<bool> =
        f.get_i64("is_diffused").0.into_iter().map(|v| v != 0).collect();
    let atom_frames = f.get_i64("rfi.atom_frames").0;

    let opt = PreproOptions { big_t: BIG_T, ..PreproOptions::default() };
    let rfi = prepro(&mut indep, T_NOW, &is_diffused, &atom_frames, &opt);

    let mut bad = Vec::new();
    let mut chk = |name: &str, got: &[f32]| {
        let want = f.get(&format!("rfi.{name}"));
        if got.len() != want.data.len() {
            println!("  {name:<12} LEN {} vs {}", got.len(), want.data.len());
            bad.push(name.to_string());
            return;
        }
        let (e, n, worst) = cmp(got, &want.data);
        println!("  {name:<12} {e} / {n} bit-identical  max|d| {worst:.3e}");
        if e != n {
            bad.push(name.to_string());
        }
    };
    chk("msa_latent", &rfi.msa_latent.data);
    chk("msa_full", &rfi.msa_full.data);
    chk("xyz", &rfi.xyz.data);
    chk("sctors", &rfi.sctors.data);
    chk("dist_matrix", &rfi.dist_matrix);
    chk("chirals", &rfi.chirals);
    chk("t1d", &rfi.t1d.data);
    chk("t2d", &rfi.t2d.data);
    chk("xyz_t", &rfi.xyz_t.data);
    chk("alpha_t", &rfi.alpha_t.data);

    let mut chk_i64 = |name: &str, got: &[i64]| {
        let want = f.get_i64(&format!("rfi.{name}")).0;
        if got.len() != want.len() {
            println!("  {name:<12} LEN {} vs {}", got.len(), want.len());
            bad.push(name.to_string());
            return;
        }
        let diff = got.iter().zip(&want).filter(|(a, b)| a != b).count();
        println!("  {name:<12} {} / {} exact", got.len() - diff, want.len());
        if diff != 0 {
            bad.push(name.to_string());
        }
    };
    chk_i64("seq", &rfi.seq);
    chk_i64("seq_unmasked", &rfi.seq_unmasked);
    chk_i64("idx", &rfi.idx);
    chk_i64("bond_feats", &rfi.bond_feats);
    chk_i64("mask_t", &rfi.mask_t.iter().map(|b| *b as i64).collect::<Vec<_>>());
    chk_i64(
        "same_chain",
        &rfi.same_chain.iter().map(|b| *b as i64).collect::<Vec<_>>(),
    );
    chk_i64("is_motif", &rfi.is_motif.iter().map(|b| *b as i64).collect::<Vec<_>>());

    // the in-place mutation `prepro` performs on its input: every diffused
    // row's slots 3.. become NaN, and the sampler reads that back
    let nan_rows = (0..l)
        .filter(|&i| indep.xyz[(i * 36 + 3) * 3].is_nan())
        .count();
    println!(
        "in-place mutation: {nan_rows} rows NaN from slot 3 ({} diffused)",
        is_diffused.iter().filter(|d| **d).count()
    );

    assert!(bad.is_empty(), "prepro not bit-exact: {bad:?}");
}
