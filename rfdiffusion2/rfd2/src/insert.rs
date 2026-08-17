//! `aa_model.Model.insert_contig_pre_atomization` — placing the reference
//! structure into the designed chain.
//!
//! This is where the row count changes: the PDB's residues plus ligand atoms
//! (54 rows for the demo input) become the designed length plus the same ligand
//! atoms (71). Everything is rebuilt against the new layout rather than
//! permuted, so the things to get right are the *rules*, not the arithmetic:
//!
//! * every row starts masked; only rows the contig maps get a real token;
//! * `bond_feats` is rebuilt as a fresh polymer chain over the protein block,
//!   which is then **cut** at each internal chain boundary;
//! * the chirals' atom indices are remapped from reference rows to output rows;
//! * ligand rows are appended after the protein and renumbered with a +200 gap;
//! * missing backbone coordinates are filled from ideal geometry, translated to
//!   the nearest present CA — `kinematics.get_init_xyz`.

use crate::chemical_gen::NTOTAL;
use crate::contig::ContigMap;
use crate::indep::{Indep, C_TERMINUS, N_TERMINUS};

/// `ChemData().num2aa.index('MAS')` — the protein mask token.
const MASKINDEX: i64 = 21;

const BOND_POLYMER: i64 = 5;

/// Gap between the last designed residue index and the first ligand atom index.
const LIGAND_GAP: i64 = 200;

/// `kinematics.get_init_xyz` with `center = False`.
///
/// For every row whose backbone is missing, substitute ideal coordinates
/// translated to the CA of the **nearest row (by index) that does have one**.
/// `argmin` over `|i - j|` breaks ties toward the lower `j`, which matters at
/// every designed residue exactly halfway between two motifs.
fn get_init_xyz(xyz: &mut [f32], l: usize, is_sm: &[bool], init_crds: &[f32]) {
    let present: Vec<bool> = (0..l)
        .map(|i| {
            if is_sm[i] {
                // a ligand row counts as present if its single atom (slot 1) is
                (0..3).all(|k| !xyz[(i * NTOTAL + 1) * 3 + k].is_nan())
            } else {
                (0..3).all(|a| (0..3).all(|k| !xyz[(i * NTOTAL + a) * 3 + k].is_nan()))
            }
        })
        .collect();
    if present.iter().all(|p| !p) {
        // upstream returns pure ideal coordinates in this case
        for i in 0..l {
            for a in 0..NTOTAL {
                for k in 0..3 {
                    let v = init_crds[a * 3 + k];
                    xyz[(i * NTOTAL + a) * 3 + k] =
                        if is_sm[i] && (a == 0 || a == 2) { f32::NAN } else { v };
                }
            }
        }
        return;
    }
    let have: Vec<usize> = (0..l).filter(|i| present[*i]).collect();
    for i in 0..l {
        if present[i] {
            continue;
        }
        // nearest present row; `argmin` keeps the first minimum
        let j = *have
            .iter()
            .min_by_key(|&&j| (j as i64 - i as i64).abs())
            .expect("at least one present row");
        let ca = [
            xyz[(j * NTOTAL + 1) * 3],
            xyz[(j * NTOTAL + 1) * 3 + 1],
            xyz[(j * NTOTAL + 1) * 3 + 2],
        ];
        for a in 0..NTOTAL {
            for k in 0..3 {
                let v = if is_sm[i] && (a == 0 || a == 2) {
                    f32::NAN
                } else {
                    init_crds[a * 3 + k] + ca[k]
                };
                xyz[(i * NTOTAL + a) * 3 + k] = v;
            }
        }
    }
}

/// The masks `insert_contig_pre_atomization` returns alongside the new `Indep`.
pub struct Masks1d {
    /// per row, whether the model is *shown* this structure
    pub is_res_str_shown: Vec<bool>,
    /// per row, whether the model is shown this sequence
    pub is_res_seq_shown: Vec<bool>,
}

/// Insert the contig. `has_termini` is `conf.contigmap.has_termini`, one flag
/// per contig chain.
pub fn insert_contig_pre_atomization(
    indep: &Indep,
    cmap: &ContigMap,
    has_termini: &[bool],
    init_crds: &[f32],
) -> (Indep, Masks1d) {
    let n_sm = indep.is_sm.iter().filter(|s| **s).count();
    let n_prot_ref = indep.len() - n_sm;
    let n_prot = cmap.hal.len();
    let l = n_prot + n_sm;

    // ---- row provenance --------------------------------------------------
    // Contig rows first, then every ligand row in its original order.
    let mut hal_idx0 = cmap.hal_idx0.clone();
    let mut ref_idx0 = cmap.ref_idx0.clone();
    for (k, i) in (0..indep.len()).filter(|i| indep.is_sm[*i]).enumerate() {
        ref_idx0.push(i);
        hal_idx0.push(n_prot + k);
    }

    // ---- ligand chains ----------------------------------------------------
    // Each ligand is its own chain, and that is not cosmetic: chains get
    // separate index runs and a `false` block in `same_chain`. Reading them out
    // of the incoming `same_chain` rather than assuming one ligand chain was
    // worth 528 wrong cells and 6 wrong indices when got wrong.
    let src_chain = indep.chains();
    let sm_chain: Vec<usize> =
        (0..indep.len()).filter(|i| indep.is_sm[*i]).map(|i| src_chain[i]).collect();
    let mut lig_chains: Vec<usize> = sm_chain.clone();
    lig_chains.sort_unstable();
    lig_chains.dedup();

    // ---- idx -------------------------------------------------------------
    // Ligand atoms restart 200 past the largest designed index, and each
    // further ligand chain restarts 200 past the previous one's last index.
    let mut idx: Vec<i64> = cmap.hal.iter().map(|(_, i)| *i).collect();
    let mut max_hal = idx.iter().copied().max().unwrap_or(0);
    let mut lig_idx = vec![0i64; n_sm];
    for ch in &lig_chains {
        let rows: Vec<usize> =
            (0..n_sm).filter(|k| sm_chain[*k] == *ch).collect();
        for (a, k) in rows.iter().enumerate() {
            lig_idx[*k] = a as i64 + LIGAND_GAP + max_hal;
        }
        if let Some(last) = rows.last() {
            max_hal = lig_idx[*last];
        }
    }
    idx.extend_from_slice(&lig_idx);

    // ---- xyz -------------------------------------------------------------
    let mut xyz = vec![f32::NAN; l * NTOTAL * 3];
    for (h, r) in hal_idx0.iter().zip(&ref_idx0) {
        let (ho, ro) = (h * NTOTAL * 3, r * NTOTAL * 3);
        xyz[ho..ho + NTOTAL * 3].copy_from_slice(&indep.xyz[ro..ro + NTOTAL * 3]);
    }

    // ---- seq -------------------------------------------------------------
    let mut seq = vec![MASKINDEX; l];
    for (h, r) in hal_idx0.iter().zip(&ref_idx0) {
        seq[*h] = indep.seq[*r];
    }

    // ---- is_sm / same_chain ---------------------------------------------
    let mut is_sm = vec![false; l];
    for v in is_sm.iter_mut().skip(n_prot) {
        *v = true;
    }
    // one chain id per row: contig chains as given, ligand rows on their own
    let mut chain_of: Vec<usize> = cmap.hal.iter().map(|(c, _)| *c as usize).collect();
    let base = chain_of.iter().copied().max().unwrap_or(0) + 1;
    for k in 0..n_sm {
        let which = lig_chains.iter().position(|c| *c == sm_chain[k]).unwrap();
        chain_of.push(base + which);
    }
    let mut same_chain = vec![false; l * l];
    for i in 0..l {
        for j in 0..l {
            same_chain[i * l + j] = chain_of[i] == chain_of[j];
        }
    }

    get_init_xyz(&mut xyz, l, &is_sm, init_crds);

    // ---- bond_feats ------------------------------------------------------
    let mut bond_feats = vec![0i64; l * l];
    for r in 0..n_prot.saturating_sub(1) {
        bond_feats[r * l + r + 1] = BOND_POLYMER;
        bond_feats[(r + 1) * l + r] = BOND_POLYMER;
    }
    for i in 0..n_sm {
        for j in 0..n_sm {
            bond_feats[(n_prot + i) * l + n_prot + j] =
                indep.bond_feats[(n_prot_ref + i) * indep.len() + n_prot_ref + j];
        }
    }

    // ---- chirals ---------------------------------------------------------
    // The four atom indices point at reference rows; remap them to output rows.
    let mut hal_by_ref = vec![usize::MAX; indep.len()];
    for (h, r) in hal_idx0.iter().zip(&ref_idx0) {
        hal_by_ref[*r] = *h;
    }
    let mut chirals = indep.chirals.clone();
    for row in chirals.chunks_mut(5) {
        for v in row.iter_mut().take(4) {
            let r = *v as usize;
            let h = hal_by_ref[r];
            assert_ne!(h, usize::MAX, "chiral references reference row {r}, which the contig drops");
            *v = h as f32;
        }
    }

    // ---- terminus_type, and the chain cuts in bond_feats -----------------
    let mut terminus_type = vec![0.0f32; l];
    let starts_ends = cmap.chain_start_end();
    assert_eq!(
        has_termini.len(),
        cmap.n_inpaint_chains,
        "contigmap.has_termini must have one entry per contig chain"
    );
    for (use_t, (cs, ce)) in has_termini.iter().zip(&starts_ends) {
        if *ce < l {
            bond_feats[ce * l + ce - 1] = 0;
            bond_feats[(ce - 1) * l + ce] = 0;
        }
        if *use_t {
            terminus_type[*cs] = N_TERMINUS;
            terminus_type[ce - 1] = C_TERMINUS;
        }
    }

    // ---- the shown/diffused masks ---------------------------------------
    let mut is_res_str_shown = cmap.inpaint_str.clone();
    is_res_str_shown.extend(std::iter::repeat(true).take(n_sm));
    let mut is_res_seq_shown = cmap.inpaint_seq.clone();
    is_res_seq_shown.extend(std::iter::repeat(true).take(n_sm));

    (
        Indep {
            seq,
            xyz,
            idx,
            bond_feats,
            chirals,
            same_chain,
            is_gp: vec![false; l],
            terminus_type,
            is_sm,
        },
        Masks1d { is_res_str_shown, is_res_seq_shown },
    )
}
