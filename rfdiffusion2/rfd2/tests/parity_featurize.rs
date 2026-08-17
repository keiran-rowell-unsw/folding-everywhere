//! Rung 4b — `prepro`: Indep -> RFI. **Tolerance: exactly 0.**
//!
//! Inputs come from `fixtures/model_pinned/step0.safetensors`, captured by
//! `python/ref_dump.py` from an unmodified upstream run (L = 71: 21 protein
//! residues + 50 ligand atoms, NAD/OXM, T = 2, seed 0).
//!
//! Feeding the reference's own `indep` in and checking `rfi` out isolates
//! `prepro` from PDB/ligand parsing, so a failure here is a featurization bug
//! and not a parsing bug. Parsing gets its own rung.

use rfd2::chemical_gen::{MASKINDEX, NAATOKENS};
use rfd2::featurize as feat;
use rfd2::weights::Weights;
use std::path::Path;

fn fixture() -> Option<Weights> {
    let root = env!("CARGO_MANIFEST_DIR");
    let path = format!("{root}/../fixtures/model_pinned/step0.safetensors");
    if !Path::new(&path).exists() {
        eprintln!("SKIP: {path} missing (run python/ref_dump.py --pinned)");
        return None;
    }
    Some(Weights::open(&path).expect("open fixture"))
}

/// T and t of the captured run, from the fixture metadata.
const T_BIG: f32 = 2.0;
const T_NOW: f32 = 2.0;

fn indep_seq(f: &Weights) -> Vec<i64> {
    f.get_i64("indep.seq").0
}

fn indep_terminus(f: &Weights) -> Vec<f32> {
    f.get("indep.terminus_type").data
}

fn is_diffused(f: &Weights) -> Vec<bool> {
    f.get_i64("is_diffused").0.into_iter().map(|x| x != 0).collect()
}

fn assert_exact(label: &str, got: &[f32], want: &[f32]) {
    assert_eq!(got.len(), want.len(), "{label}: length");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert_eq!(
            g.to_bits(),
            w.to_bits(),
            "{label}[{i}]: got {g} want {w}"
        );
    }
}

#[test]
fn msa_masked_matches() {
    let Some(f) = fixture() else { return };
    let seq = indep_seq(&f);
    let term = indep_terminus(&f);
    let l = seq.len();

    let got = feat::msa_masked(&seq, &term, true);
    let want = f.get("rfi.msa_latent"); // [1,1,L,164]
    assert_eq!(want.data.len(), l * feat::MSA_MASKED_WIDTH, "msa_latent size");
    assert_eq!(feat::MSA_MASKED_WIDTH, 164);
    assert_exact("msa_masked", &got.data, &want.data);
    println!("msa_masked: [{l}, 164] = {} values exact", got.data.len());
}

#[test]
fn msa_full_matches() {
    let Some(f) = fixture() else { return };
    let seq = indep_seq(&f);
    let term = indep_terminus(&f);
    let l = seq.len();

    let got = feat::msa_full(&seq, &term, true);
    let want = f.get("rfi.msa_full");
    assert_eq!(feat::MSA_FULL_WIDTH, 83);
    assert_exact("msa_full", &got.data, &want.data);
    println!("msa_full: [{l}, 83] = {} values exact", got.data.len());
}

/// `seq` and `seq_unmasked` are handed straight through; asserted so a future
/// change to the masking policy cannot slip past silently.
#[test]
fn seq_passthrough_matches() {
    let Some(f) = fixture() else { return };
    let seq = indep_seq(&f);
    let (rfi_seq, _) = f.get_i64("rfi.seq");
    let (rfi_unmasked, _) = f.get_i64("rfi.seq_unmasked");
    assert_eq!(seq, rfi_seq, "rfi.seq != indep.seq");
    assert_eq!(seq, rfi_unmasked, "rfi.seq_unmasked != indep.seq");
    println!("seq / seq_unmasked: {} tokens exact", seq.len());
}

/// The `t1d` sequence block plus the confidence channel, for both templates.
///
/// Only the first `NAATOKENS` channels are checked here: the remaining 34 come
/// from `extra_tXd` (radius-of-gyration / RASA / timestep embedding), which is
/// a separate sub-port with its own rung.
#[test]
fn t1d_core_and_template_marker_match() {
    let Some(f) = fixture() else { return };
    let seq = indep_seq(&f);
    let diff = is_diffused(&f);
    let l = seq.len();

    let want = f.get("rfi.t1d"); // [1, 2, L, 114]
    let full_w = want.data.len() / (2 * l);
    assert_eq!(full_w, 114, "t1d width");

    let got = feat::t1d_core(&seq, &diff, T_NOW, T_BIG);

    // template 0: sequence one-hot + confidence
    for i in 0..l {
        for c in 0..NAATOKENS {
            let g = got.data[i * NAATOKENS + c];
            let w = want.data[(0 * l + i) * full_w + c];
            assert_eq!(g.to_bits(), w.to_bits(), "t1d[t=0][{i}][{c}]");
        }
    }

    // template 1: identical, except the confidence channel is the -1 marker
    for i in 0..l {
        for c in 0..NAATOKENS {
            let w = want.data[(1 * l + i) * full_w + c];
            let expect = if c == NAATOKENS - 1 {
                feat::SELF_COND_TEMPLATE_MARKER
            } else {
                got.data[i * NAATOKENS + c]
            };
            assert_eq!(w.to_bits(), expect.to_bits(), "t1d[t=1][{i}][{c}]");
        }
    }
    println!("t1d: [2, {l}, {NAATOKENS}] core channels exact (+ self-cond marker -1)");
}

/// The MASK-token collapse. Off by one here silently relabels every token above
/// MASKINDEX, which would look like a mysterious chemistry error much later.
#[test]
fn seq_cat_shift_collapses_mask_token() {
    let Some(f) = fixture() else { return };
    let seq = indep_seq(&f);
    let shifted = feat::seq_cat_shifted(&seq);
    for (i, (&s, &sh)) in seq.iter().zip(&shifted).enumerate() {
        let want = if s as usize >= MASKINDEX { s - 1 } else { s };
        assert_eq!(sh, want, "seq_cat_shifted[{i}]");
        assert!((sh as usize) < NAATOKENS - 1, "shifted[{i}] out of range");
    }
    let n_shifted = seq.iter().filter(|&&s| s as usize >= MASKINDEX).count();
    println!("seq_cat_shifted: {}/{} tokens shifted down past MASKINDEX={MASKINDEX}",
             n_shifted, seq.len());
}

/// `dist_matrix` — BFS over the bond graph, including the infinities.
#[test]
fn bond_distances_match() {
    let Some(f) = fixture() else { return };
    let (bf, bshape) = f.get_i64("indep.bond_feats");
    let l = bshape[0];
    let got = feat::bond_distances(&bf, l);
    let want = f.get("rfi.dist_matrix"); // [1, L, L]

    assert_eq!(got.data.len(), want.data.len(), "dist_matrix size");
    let mut n_inf = 0usize;
    for (i, (g, w)) in got.data.iter().zip(&want.data).enumerate() {
        if w.is_infinite() {
            assert!(
                g.is_infinite() && g.is_sign_positive() == w.is_sign_positive(),
                "dist_matrix[{i}]: got {g} want {w} (unreachable pairs stay +inf)"
            );
            n_inf += 1;
        } else {
            assert_eq!(g.to_bits(), w.to_bits(), "dist_matrix[{i}]: got {g} want {w}");
        }
    }
    println!(
        "dist_matrix: [{l},{l}] = {} values exact ({n_inf} unreachable pairs = +inf)",
        got.data.len()
    );
}

/// Constants that are easy to get wrong precisely because they look trivial.
#[test]
fn constant_features_match() {
    let Some(f) = fixture() else { return };
    let seq = indep_seq(&f);
    let l = seq.len();

    let want_mask = f.get_i64("rfi.mask_t").0;
    let got_mask = feat::mask_t(l);
    assert_eq!(got_mask.len(), want_mask.len(), "mask_t size");
    for (i, (g, w)) in got_mask.iter().zip(&want_mask).enumerate() {
        assert_eq!(*g, *w != 0, "mask_t[{i}]");
    }

    let want_sc = f.get("rfi.sctors");
    let ntotaldofs = want_sc.data.len() / (l * 2);
    let got_sc = feat::sctors(l, ntotaldofs);
    assert_exact("sctors", &got_sc.data, &want_sc.data);

    let want_xyzt = f.get("rfi.xyz_t");
    let xyz_in = f.get("indep.xyz");
    let n_atoms = xyz_in.shape[1];
    let got_xyzt = feat::xyz_t_from_ca(&xyz_in.data, l, n_atoms);
    assert_exact("xyz_t", &got_xyzt.data, &want_xyzt.data);
    let n_zero = want_xyzt.data[l * 3..].iter().filter(|v| **v == 0.0).count();
    assert_eq!(n_zero, l * 3, "template 1 of xyz_t must stay all-zero");

    println!("mask_t [2,{l},{l}] all-true · sctors [{l},{ntotaldofs},2] all-zero \
· xyz_t [2,{l},3] = CA coords + zeros — exact");
}

/// `rfi.xyz` — the template coordinates after `prepro`'s four-rule NaN/zero
/// fill. The rules overlap (a small molecule is also non-diffused), so the
/// order they are applied in decides the result; this is the test that pins it.
#[test]
fn rfi_xyz_fill_rules_match() {
    let Some(f) = fixture() else { return };
    let seq = indep_seq(&f);
    let diff = is_diffused(&f);
    let l = seq.len();
    let is_sm: Vec<bool> = f.get_i64("indep.is_sm").0.into_iter().map(|x| x != 0).collect();

    let xyz_in = f.get("indep.xyz");
    let n_atoms_in = xyz_in.shape[1];
    let want = f.get("rfi.xyz"); // [1, L, 36, 3]
    let ntotal = want.data.len() / (l * 3);
    assert_eq!(ntotal, 36, "rfi.xyz atom slots");

    let got = feat::rfi_xyz(
        &xyz_in.data, &seq, &is_sm, &diff,
        n_atoms_in, /*nheavy*/ 23, /*nheavyprot*/ 14, ntotal,
    );

    let mut n_nan = 0usize;
    let mut n_zero = 0usize;
    let mut n_real = 0usize;
    for (i, (g, w)) in got.data.iter().zip(&want.data).enumerate() {
        if w.is_nan() {
            assert!(g.is_nan(), "rfi.xyz[{i}]: got {g} want NaN");
            n_nan += 1;
        } else {
            assert_eq!(g.to_bits(), w.to_bits(), "rfi.xyz[{i}]: got {g} want {w}");
            if *w == 0.0 { n_zero += 1 } else { n_real += 1 }
        }
    }
    println!(
        "rfi.xyz: [{l},{ntotal},3] = {} values exact ({n_real} real, {n_zero} zeroed, {n_nan} NaN)",
        got.data.len()
    );
}
