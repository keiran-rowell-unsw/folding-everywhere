//! Rung 4b — `aa_model.Model.prepro`: `Indep` -> `RFI`, the tensors handed to
//! the network.
//!
//! This module covers the parts of `prepro` that are pure functions of `indep`
//! (sequence, termini, bond graph, diffusion mask). The template features that
//! need the XYZ converter (`alpha_t`, `t2d`) and the `extra_tXd` conditions come
//! in rung 4c/5 — they depend on geometry, which is the next rung down.
//!
//! Everything here is integer/one-hot/graph work, so the tolerance is
//! **exactly 0** (SOP §4: "Anything integer off at all -> a real bug. Always.").
//!
//! Reference: `rf_diffusion/aa_model.py:Model.prepro` (line 1222) and
//! `rf2aa/data/data_loader.py:get_bond_distances` (line 1469).

use crate::chemical_gen::{MASKINDEX, NAATOKENS};
use crate::tensor::Tensor;

/// `aa_model.py:51` — terminus channel encodings.
pub const N_TERMINUS: f32 = 1.0;
/// `aa_model.py:54`
pub const C_TERMINUS: f32 = 2.0;

/// Channel counts `prepro` uses to lay out the MSA features.
pub const NUM_TERMINI: usize = 2;
pub const NUM_INDEL: usize = 1;

/// Width of `msa_masked`: two sequence blocks, two indel blocks, two termini.
pub const MSA_MASKED_WIDTH: usize = 2 * NAATOKENS + 2 * NUM_INDEL + NUM_TERMINI; // 164
/// Width of `msa_full`: one sequence block, one indel, two termini.
pub const MSA_FULL_WIDTH: usize = NAATOKENS + NUM_INDEL + NUM_TERMINI; // 83

/// `msa_masked` — `[1, 1, L, 164]` flattened to `[L, 164]`.
///
/// Note both sequence blocks carry the *same* one-hot: this is a single-sequence
/// model, so the "masked" and "unmasked" halves are identical by construction.
/// The indel channels stay zero.
pub fn msa_masked(seq: &[i64], terminus_type: &[f32], annotate_termini: bool) -> Tensor {
    let l = seq.len();
    let mut out = vec![0.0f32; l * MSA_MASKED_WIDTH];
    for i in 0..l {
        let base = i * MSA_MASKED_WIDTH;
        let s = seq[i] as usize;
        assert!(s < NAATOKENS, "seq[{i}] = {s} >= NAATOKENS");
        out[base + s] = 1.0;
        out[base + NAATOKENS + s] = 1.0;
        if annotate_termini {
            let n_term = 2 * NAATOKENS + 2 * NUM_INDEL;
            out[base + n_term] = (terminus_type[i] == N_TERMINUS) as i32 as f32;
            out[base + n_term + 1] = (terminus_type[i] == C_TERMINUS) as i32 as f32;
        }
    }
    Tensor::new(out, vec![l, MSA_MASKED_WIDTH])
}

/// `msa_full` — `[1, 1, L, 83]` flattened to `[L, 83]`.
pub fn msa_full(seq: &[i64], terminus_type: &[f32], annotate_termini: bool) -> Tensor {
    let l = seq.len();
    let mut out = vec![0.0f32; l * MSA_FULL_WIDTH];
    for i in 0..l {
        let base = i * MSA_FULL_WIDTH;
        out[base + seq[i] as usize] = 1.0;
        if annotate_termini {
            let n_term = NAATOKENS + NUM_INDEL;
            out[base + n_term] = (terminus_type[i] == N_TERMINUS) as i32 as f32;
            out[base + n_term + 1] = (terminus_type[i] == C_TERMINUS) as i32 as f32;
        }
    }
    Tensor::new(out, vec![l, MSA_FULL_WIDTH])
}

/// The sequence index `t1d` one-hots over, after collapsing the MASK token.
///
/// `prepro` does `seq_cat_shifted[seq_cat_shifted >= MASKINDEX] -= 1`, which
/// removes MASK (21) from the alphabet and slides everything above it down by
/// one — so `t1d`'s sequence block is `NAATOKENS - 1 = 79` wide, not 80. Getting
/// this off by one shifts every non-protein token's identity.
pub fn seq_cat_shifted(seq: &[i64]) -> Vec<i64> {
    seq.iter()
        .map(|&s| if s as usize >= MASKINDEX { s - 1 } else { s })
        .collect()
}

/// The structure-confidence channel appended to `t1d`.
///
/// `1.0` for motif (non-diffused) positions, `1 - t/T` for diffused ones.
pub fn strconf(is_diffused: &[bool], t: f32, big_t: f32) -> Vec<f32> {
    is_diffused
        .iter()
        .map(|&d| if d { 1.0 - t / big_t } else { 1.0 })
        .collect()
}

/// The first `NAATOKENS` channels of `t1d` for **one** template:
/// `one_hot(seq_cat_shifted, 79)` followed by the confidence channel.
///
/// The full `t1d` is this tiled over 2 templates with `extra_t1d` concatenated,
/// and with `t1d[0, 1, :, NAATOKENS-1] = -1` marking the self-conditioning
/// template. That marker lives in the confidence channel, which is why it is
/// index `NAATOKENS - 1` and not something derived from the extra width.
pub fn t1d_core(seq: &[i64], is_diffused: &[bool], t: f32, big_t: f32) -> Tensor {
    let l = seq.len();
    let shifted = seq_cat_shifted(seq);
    let conf = strconf(is_diffused, t, big_t);
    let w = NAATOKENS; // 79 one-hot + 1 confidence
    let mut out = vec![0.0f32; l * w];
    for i in 0..l {
        let base = i * w;
        let s = shifted[i] as usize;
        assert!(s < NAATOKENS - 1, "shifted seq[{i}] = {s} out of range");
        out[base + s] = 1.0;
        out[base + NAATOKENS - 1] = conf[i];
    }
    Tensor::new(out, vec![l, w])
}

/// The value written into the confidence channel of template 1 to distinguish
/// the self-conditioning template from the x_t template.
pub const SELF_COND_TEMPLATE_MARKER: f32 = -1.0;

/// `get_bond_distances` — unweighted shortest path over the bond graph.
///
/// Edges are the entries with `0 < bond_feats < 5`, i.e. the real chemical bonds
/// (single/double/triple/aromatic); the higher codes are the residue-residue
/// "virtual" bond types and are deliberately excluded. Unreachable pairs are
/// `inf`, matching `scipy.sparse.csgraph.shortest_path` — the reference keeps
/// the infinities rather than clamping them, and the note in the source says so
/// explicitly ("protein portion is inf and you don't want to mask it out").
///
/// BFS from each node; the graph is undirected and unweighted, so BFS *is* the
/// shortest path and no priority queue is needed.
pub fn bond_distances(bond_feats: &[i64], l: usize) -> Tensor {
    // adjacency
    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); l];
    for i in 0..l {
        for j in 0..l {
            let b = bond_feats[i * l + j];
            if b > 0 && b < 5 {
                adj[i].push(j);
            }
        }
    }

    let mut out = vec![f32::INFINITY; l * l];
    let mut dist = vec![u32::MAX; l];
    let mut queue: Vec<usize> = Vec::with_capacity(l);
    for src in 0..l {
        dist.iter_mut().for_each(|d| *d = u32::MAX);
        queue.clear();
        dist[src] = 0;
        queue.push(src);
        let mut head = 0usize;
        while head < queue.len() {
            let u = queue[head];
            head += 1;
            let du = dist[u];
            for &v in &adj[u] {
                if dist[v] == u32::MAX {
                    dist[v] = du + 1;
                    queue.push(v);
                }
            }
        }
        for j in 0..l {
            out[src * l + j] = if dist[j] == u32::MAX {
                f32::INFINITY
            } else {
                dist[j] as f32
            };
        }
    }
    Tensor::new(out, vec![l, l])
}

/// `mask_t` — `torch.ones(1, 2, L, L).bool()`. Constant, but asserted anyway:
/// a silently wrong mask is invisible until the template stack misbehaves.
pub fn mask_t(l: usize) -> Vec<bool> {
    vec![true; 2 * l * l]
}

/// `sctors` — `torch.zeros((1, L, NTOTALDOFS, 2))`. The sidechain torsions are
/// **not** fed in at inference; they are all zero.
pub fn sctors(l: usize, ntotaldofs: usize) -> Tensor {
    Tensor::zeros(&[l, ntotaldofs, 2])
}

/// Residue-type ranges from `nucleic_compatibility_utils.get_resi_type_mask`.
/// Both bounds are **inclusive** in the reference (`>= lb & <= ub`), which is
/// why `prot` is 0..=20 and not 0..20.
pub fn is_prot_and_mask(tok: i64) -> bool {
    (0..=21).contains(&tok)
}
pub fn is_nucleic(tok: i64) -> bool {
    (22..=31).contains(&tok)
}

/// `xyz_t` — `[2, L, 3]`.
///
/// `prepro` allocates `torch.zeros(1, 2, L, 3)` and then writes
/// `xyz_t[0,0] = xyz[0,:,1]`, i.e. template 0 carries the **CA coordinates**
/// (atom slot 1) and template 1 — the self-conditioning slot — stays zero.
/// Reading only the allocation and stopping at the "NO SELF COND" comment gives
/// an all-zero tensor and a test that fails on the first element.
pub fn xyz_t_from_ca(xyz: &[f32], l: usize, n_atoms: usize) -> Tensor {
    let mut out = vec![0.0f32; 2 * l * 3];
    for i in 0..l {
        for c in 0..3 {
            out[i * 3 + c] = xyz[(i * n_atoms + 1) * 3 + c];
        }
    }
    Tensor::new(out, vec![2, l, 3])
}

/// `rfi.xyz` — `[L, NTOTAL, 3]`, the template coordinates after `prepro`'s
/// nan/zero fill.
///
/// Four rules, applied in this order (they overlap, and the order decides the
/// overlaps):
/// 1. diffused non-small-molecule positions: slots `3..` -> NaN;
/// 2. small molecules: slots `NHEAVYPROT..` -> 0;
/// 3. protein motif: slots `NHEAVYPROT..` -> 0;
/// 4. nucleic motif: slots `NHEAVY..` -> 0.
///
/// Before those, the source tensor is truncated to `NHEAVY` and re-padded to
/// `NTOTAL` with NaN — so hydrogens are dropped, not carried.
pub fn rfi_xyz(
    indep_xyz: &[f32],
    seq: &[i64],
    is_sm: &[bool],
    is_diffused: &[bool],
    n_atoms_in: usize,
    nheavy: usize,
    nheavyprot: usize,
    ntotal: usize,
) -> Tensor {
    let l = seq.len();
    let mut out = vec![f32::NAN; l * ntotal * 3];
    for i in 0..l {
        // keep only the heavy atoms; everything past NHEAVY stays NaN
        for a in 0..nheavy.min(n_atoms_in) {
            for c in 0..3 {
                out[(i * ntotal + a) * 3 + c] = indep_xyz[(i * n_atoms_in + a) * 3 + c];
            }
        }
    }
    for i in 0..l {
        let prot_motif = !is_diffused[i] && !is_sm[i] && is_prot_and_mask(seq[i]);
        let nucl_motif = !is_diffused[i] && !is_sm[i] && is_nucleic(seq[i]);
        if is_diffused[i] && !is_sm[i] {
            for a in 3..ntotal {
                for c in 0..3 {
                    out[(i * ntotal + a) * 3 + c] = f32::NAN;
                }
            }
        }
        if is_sm[i] || prot_motif {
            for a in nheavyprot..ntotal {
                for c in 0..3 {
                    out[(i * ntotal + a) * 3 + c] = 0.0;
                }
            }
        }
        if nucl_motif {
            for a in nheavy..ntotal {
                for c in 0..3 {
                    out[(i * ntotal + a) * 3 + c] = 0.0;
                }
            }
        }
    }
    Tensor::new(out, vec![l, ntotal, 3])
}
