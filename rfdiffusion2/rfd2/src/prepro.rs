//! `rf_diffusion/aa_model.py:Model.prepro` — `Indep` + timestep to the `Rfi`
//! the network consumes.
//!
//! `src/featurize.rs` covers the parts that are pure functions of the sequence
//! and bond graph (rung 4b). This module adds the two template features that
//! need geometry — `alpha_t` (`src/torsions.rs`) and `t2d` (`src/t2d.rs`) —
//! plus the `extra_t1d` conditioning block, and assembles the whole `Rfi`.
//!
//! ## `prepro` mutates its input, and the sampler depends on that
//!
//! The first statement is `xyz_t = indep.xyz` with no clone, and the next line
//! writes `xyz_t[is_diffused, 3:, :] = nan`. So calling `prepro` **NaNs out the
//! sidechain slots of every diffused row of `indep` in place**, and the
//! sampler's `get_x_t_1` then reads that same mutated array. Reproducing the
//! coordinates without reproducing the mutation gives a structure that is
//! right on the first step and wrong on the second, which is why this function
//! takes `&mut Indep`.
//!
//! ## `extra_t1d` is 34 channels wide even though the demo enables nothing
//!
//! The checkpoint's config turns on three featurizers — `radius_of_gyration_v2`,
//! `relative_sasa_v2` and `sinusoidal_timestep_embedding`. The first two are
//! *inactive* (`feature_inference_conf.active` is false), and an inactive
//! featurizer still contributes `zeros(L, n_bins + 1)` = 7 channels each rather
//! than nothing. The remaining 20 are a sinusoidal embedding of `t/T`,
//! identical across rows. Measured against the fixture: channel 14 is
//! `sin(10000)` = -0.3056 and channel 33 is `cos(1)` = 0.5403, which pins
//! `max_positions = 10000` and `embedding_dim = 20`.

use crate::chemical_gen::{NAATOKENS, NHEAVY, NTOTAL, NTOTALDOFS};
use crate::featurize as feat;
use crate::indep::Indep;
use crate::model::rf::Rfi;
use crate::t2d;
use crate::tensor::Tensor;
use crate::torsions;

/// Width of the two inactive conditioning blocks (`n_bins + 1` each).
const N_ROG_CHANNELS: usize = 7;
const N_RASA_CHANNELS: usize = 7;
/// `sinusoidal_timestep_embedding.embedding_dim`
const N_TIMESTEP_CHANNELS: usize = 20;
/// `sinusoidal_timestep_embedding.max_positions`
const MAX_POSITIONS: f32 = 10000.0;
/// Total `extra_t1d` width for the RFD_173 demo configuration.
pub const EXTRA_T1D_WIDTH: usize =
    N_ROG_CHANNELS + N_RASA_CHANNELS + N_TIMESTEP_CHANNELS;

/// Everything about the run that `prepro` reads out of the config.
#[derive(Clone, Debug)]
pub struct PreproOptions {
    /// `diffuser.T`
    pub big_t: usize,
    /// `preprocess.use_cb_to_get_pair_dist`
    pub use_cb: bool,
    /// `preprocess.annotate_termini`
    pub annotate_termini: bool,
}

impl Default for PreproOptions {
    fn default() -> Self {
        PreproOptions { big_t: 100, use_cb: true, annotate_termini: true }
    }
}

/// `features.get_sinusoidal_timestep_embedding(t_cont, 20, 10000)`.
///
/// `emb_i = exp(-i * log(M) / (half - 1))`, scaled by `t * M`, then `sin`
/// followed by `cos` — a *concatenation*, not interleaved pairs. Under pinning
/// `exp`, `sin` and `cos` are each one f64 evaluation with a single narrowing;
/// the multiply between them is fp32.
fn timestep_embedding(t_cont: f32) -> Vec<f32> {
    let half = N_TIMESTEP_CHANNELS / 2;
    let scale = ((MAX_POSITIONS as f64).ln() / (half - 1) as f64) as f32;
    let ts = t_cont * MAX_POSITIONS;
    let mut out = vec![0.0f32; N_TIMESTEP_CHANNELS];
    for i in 0..half {
        let e = ((((i as f32) * -scale) as f64).exp()) as f32;
        let x = ts * e;
        out[i] = ((x as f64).sin()) as f32;
        out[half + i] = ((x as f64).cos()) as f32;
    }
    out
}

/// `indep.extra_t1d` — `[L, 34]`, every row identical.
pub fn extra_t1d(l: usize, t_cont: f32) -> Vec<f32> {
    let emb = timestep_embedding(t_cont);
    let mut out = vec![0.0f32; l * EXTRA_T1D_WIDTH];
    for i in 0..l {
        let base = i * EXTRA_T1D_WIDTH + N_ROG_CHANNELS + N_RASA_CHANNELS;
        out[base..base + N_TIMESTEP_CHANNELS].copy_from_slice(&emb);
    }
    out
}

/// The `xyz_t` that `alpha_t` and `t2d` are measured off — *before* the final
/// hydrogen/motif cleanup, which happens after both are computed.
///
/// Heavy slots 0..NHEAVY come from `indep.xyz`; NHEAVY..NTOTAL are the NaN pad
/// that `torch.cat` appends. Diffused rows have already had slots 3.. NaN'd in
/// `indep` itself by the caller.
fn template_xyz(indep: &Indep, l: usize) -> Vec<f32> {
    let mut out = vec![f32::NAN; l * NTOTAL * 3];
    for i in 0..l {
        for a in 0..NHEAVY {
            for c in 0..3 {
                out[(i * NTOTAL + a) * 3 + c] = indep.xyz[(i * NTOTAL + a) * 3 + c];
            }
        }
    }
    out
}

/// `Model.prepro(indep, t, is_diffused)`.
///
/// `t` is the integer timestep, not `t/T`.
pub fn prepro(
    indep: &mut Indep,
    t: usize,
    is_diffused: &[bool],
    atom_frames: &[i64],
    opt: &PreproOptions,
) -> Rfi {
    let l = indep.len();
    let big_t = opt.big_t as f32;
    let t_f = t as f32;

    // ---- the in-place mutation, first, exactly as upstream orders it ------
    for i in 0..l {
        if is_diffused[i] {
            for a in 3..NTOTAL {
                for c in 0..3 {
                    indep.xyz[(i * NTOTAL + a) * 3 + c] = f32::NAN;
                }
            }
        }
    }

    // The featurize helpers return flat `[L, w]` tensors; the network indexes
    // `msa.shape[0..3]` as `(B, N, L)`, so every tensor is reshaped to the
    // batched layout the model expects before it goes into the `Rfi`. The data
    // order is unchanged — only the declared shape.
    let mut msa_latent =
        feat::msa_masked(&indep.seq, &indep.terminus_type, opt.annotate_termini);
    msa_latent.shape = vec![1, 1, l, feat::MSA_MASKED_WIDTH];
    let mut msa_full = feat::msa_full(&indep.seq, &indep.terminus_type, opt.annotate_termini);
    msa_full.shape = vec![1, 1, l, feat::MSA_FULL_WIDTH];

    // ---- t1d: [2, L, NAATOKENS + EXTRA] -----------------------------------
    let core = feat::t1d_core(&indep.seq, is_diffused, t_f, big_t);
    let extra = extra_t1d(l, t_f / big_t);
    let w = NAATOKENS + EXTRA_T1D_WIDTH;
    let mut t1d = vec![0.0f32; 2 * l * w];
    for tmpl in 0..2 {
        for i in 0..l {
            let o = (tmpl * l + i) * w;
            t1d[o..o + NAATOKENS]
                .copy_from_slice(&core.data[i * NAATOKENS..(i + 1) * NAATOKENS]);
            t1d[o + NAATOKENS..o + w]
                .copy_from_slice(&extra[i * EXTRA_T1D_WIDTH..(i + 1) * EXTRA_T1D_WIDTH]);
        }
    }
    // template 1 is the self-conditioning slot; its confidence channel carries
    // the -1 marker that tells the model which template is which
    for i in 0..l {
        t1d[(l + i) * w + NAATOKENS - 1] = feat::SELF_COND_TEMPLATE_MARKER;
    }

    // ---- alpha_t, measured off the padded template coordinates ------------
    let tmpl_xyz = template_xyz(indep, l);
    let seq_tmp = feat::seq_cat_shifted(&indep.seq);
    let tors = torsions::get_torsions(&seq_tmp, &tmpl_xyz, l);
    let mut alpha_t = vec![0.0f32; 2 * l * 3 * NTOTALDOFS];
    for tmpl in 0..2 {
        for i in 0..l {
            for k in 0..NTOTALDOFS {
                let o = (tmpl * l + i) * 3 * NTOTALDOFS + k * 3;
                alpha_t[o] = tors.alpha[(i * NTOTALDOFS + k) * 2];
                alpha_t[o + 1] = tors.alpha[(i * NTOTALDOFS + k) * 2 + 1];
                alpha_t[o + 2] = if tors.mask[i * NTOTALDOFS + k] { 1.0 } else { 0.0 };
            }
        }
    }

    // ---- t2d and xyz_t: template 0 only, template 1 stays zero ------------
    let mut t2d_full = vec![0.0f32; 2 * l * l * t2d::T2D_WIDTH];
    let t0 = t2d::get_t2d(&tmpl_xyz, l, &indep.is_sm, atom_frames, opt.use_cb);
    t2d_full[..l * l * t2d::T2D_WIDTH].copy_from_slice(&t0);

    let mut xyz_t = vec![0.0f32; 2 * l * 3];
    for i in 0..l {
        for c in 0..3 {
            xyz_t[i * 3 + c] = tmpl_xyz[(i * NTOTAL + 1) * 3 + c];
        }
    }

    // ---- the rest is rung 4b, already green -------------------------------
    let dist_matrix = feat::bond_distances(&indep.bond_feats, l);
    let mut sctors = feat::sctors(l, NTOTALDOFS);
    sctors.shape = vec![1, l, NTOTALDOFS, 2];
    let mask_t = feat::mask_t(l);
    let mut xyz = feat::rfi_xyz(
        &indep.xyz,
        &indep.seq,
        &indep.is_sm,
        is_diffused,
        NTOTAL,
        NHEAVY,
        crate::chemical_gen::NHEAVYPROT,
        NTOTAL,
    );
    xyz.shape = vec![1, l, NTOTAL, 3];

    Rfi {
        msa_latent,
        msa_full,
        seq: indep.seq.clone(),
        seq_unmasked: indep.seq.clone(),
        xyz,
        sctors,
        idx: indep.idx.clone(),
        bond_feats: indep.bond_feats.clone(),
        dist_matrix: dist_matrix.data,
        chirals: indep.chirals.clone(),
        atom_frames: atom_frames.to_vec(),
        t1d: Tensor::new(t1d, vec![1, 2, l, w]),
        t2d: Tensor::new(t2d_full, vec![1, 2, l, l, t2d::T2D_WIDTH]),
        xyz_t: Tensor::new(xyz_t, vec![1, 2, l, 3]),
        alpha_t: Tensor::new(alpha_t, vec![1, 2, l, 3 * NTOTALDOFS]),
        mask_t,
        same_chain: indep.same_chain.clone(),
        is_motif: is_diffused.iter().map(|d| !d).collect(),
    }
}
