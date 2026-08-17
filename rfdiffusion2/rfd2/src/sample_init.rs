//! `Sampler.sample_init` — PDB on disk to the `Indep` the first model step sees.
//!
//! `python/probe_featurize.py` measured the pipeline that actually runs for this
//! configuration (`conf.transforms.names` is empty, so nothing else fires):
//!
//! ```text
//! PDBLoaderDataset.getitem_inner
//!   process_target                 PDB parse                    src/pdb.rs
//!   ContigMap(target_feats, ...)   contig parse                 src/contig.rs
//!   aa_model.make_indep            Indep + ligand               src/indep.rs
//!   extract_centering_origin       no-op for this config
//!   insert_contig_pre_atomization                               src/insert.rs
//! AddConditionalInputs             no-op on `Indep`
//! CenterPostTransform (jitter = 0) no-op on `Indep`
//! update_inference_state           no-op on `Indep`
//! diffuse                          the only randomness           <- this module
//! ```
//!
//! The three transforms above `diffuse` were confirmed to be no-ops *field by
//! field* against the stage captures, not assumed to be — for this
//! configuration only. `Options` records every setting they depend on and
//! [`SampleInit::run`] refuses any value it has not been measured against,
//! rather than quietly producing a structure that looks right.
//!
//! ## The draw order is the specification
//!
//! `diffuse_then_add_conditional` makes nine torch draws, and their *order*
//! matters more than their values because a skipped draw shifts every later one
//! — including the sampler's `psi_pred` on every subsequent step:
//!
//! ```text
//!  0,1  normal (n_sm, 3)  add_fake_frame_legs        <- diffuse
//!  2    randn  (1, L, 3)  sample_gaussian            <- _corrupt_trans
//!  3    randn  (1, L, 3)  sample_vector              <- igso3.sample
//!  4    rand   (1, L)     sample_angle               <- igso3.sample
//!  5    rand   (L, 2)     atom37_from_rigid          <- diffuse       psi_pred
//!  6,7  normal (n_sm, 3)  add_fake_frame_legs        <- add_fake_peptide_frame
//!  8    rand   (L, 2)     atom37_from_rigid          <- idealize_peptide_frames
//! ```
//!
//! Two of those are easy to miss: `atom37_from_rigid` looks purely geometric
//! and draws `psi_pred`, and `add_fake_peptide_frame` repeats the whole
//! fake-leg-plus-idealization sequence a *second* time, after the noiser.

use crate::chemical_gen::{MASKINDEX, NTOTAL};
use crate::contig::{ContigMap, ContigError};
use crate::indep::{make_indep, Indep, IndepError};
use crate::insert::insert_contig_pre_atomization;
use crate::ligand::LigandSet;
use crate::nn::Ctx;
use crate::noiser::{add_fake_frame_legs, forward_marginal, rigid_frames_from_atom_14, Igso3, Rigids};
use crate::openfold::{atom37_from_rigid, N_ATOM37};
use crate::pdb;

/// Every configuration value the measured no-op claims depend on.
///
/// These are checked, not documented: a run with `preserve_motif_sidechains`
/// on takes a different branch of `diffuse` that this port has never been
/// compared against, so it is refused rather than guessed.
#[derive(Clone, Debug)]
pub struct Options {
    /// `diffuser.T`
    pub big_t: usize,
    /// `diffuser.partial_T` — `None` for a full trajectory
    pub partial_t: Option<usize>,
    /// `diffuser.preserve_motif_sidechains`
    pub preserve_motif_sidechains: bool,
    /// `diffuser.independently_center_diffuseds`
    pub independently_center_diffuseds: bool,
    /// the interpolant's `center_noise_sample`
    pub center_noise_sample: bool,
    /// `contigmap.has_termini`, one flag per contig chain
    pub has_termini: Vec<bool>,
    /// `contigmap.length`, already parsed to the half-open `[lo, hi)`
    pub length: Option<(usize, usize)>,
}

impl Default for Options {
    /// The RFD_173 demo configuration (`--config-name=aa`).
    fn default() -> Self {
        Options {
            big_t: 100,
            partial_t: None,
            preserve_motif_sidechains: false,
            independently_center_diffuseds: false,
            center_noise_sample: false,
            has_termini: vec![true],
            length: None,
        }
    }
}

#[derive(Debug)]
pub enum InitError {
    Contig(ContigError),
    Indep(IndepError),
    /// A configuration this port has not been measured against.
    Unsupported(String),
}

impl std::fmt::Display for InitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            InitError::Contig(e) => write!(f, "{e}"),
            InitError::Indep(e) => write!(f, "{e}"),
            InitError::Unsupported(s) => write!(f, "unsupported configuration: {s}"),
        }
    }
}

impl std::error::Error for InitError {}

/// Everything `sample_init` hands back.
pub struct SampleInit {
    /// `indep_cond` — what the sampler iterates on.
    pub indep: Indep,
    /// The pre-diffusion structure; the output writer restores motif
    /// sidechains from it.
    pub indep_orig: Indep,
    /// The fully-diffused structure, kept because the reference keeps it.
    pub indep_uncond: Indep,
    pub is_diffused: Vec<bool>,
    pub is_masked_seq: Vec<bool>,
    pub t_step_input: usize,
    pub cmap: ContigMap,
}

/// `aa_model.mask_indep`.
///
/// `get_full_mask_seq` picks a mask token per molecule class. On this
/// configuration every masked row is already `MAS` by the time
/// `insert_contig_pre_atomization` is done, so this only has to handle protein.
/// A nucleic or ligand row that is genuinely masked would need ` DX`, ` RX` or
/// `ATM`, which no measured run produces — so it is refused rather than
/// silently masked as protein.
fn mask_indep(indep: &mut Indep, is_masked_seq: &[bool]) -> Result<(), InitError> {
    for i in 0..indep.len() {
        if !is_masked_seq[i] {
            continue;
        }
        if indep.is_sm[i] || indep.seq[i] >= 22 {
            return Err(InitError::Unsupported(format!(
                "mask_indep: row {i} (token {}) is not protein, and the per-class \
                 mask tokens have not been measured on any run",
                indep.seq[i]
            )));
        }
        indep.seq[i] = MASKINDEX as i64;
    }
    Ok(())
}

/// `aa_model.diffuse`, for `preserve_motif_sidechains = false`.
///
/// Note what the false branch does: `indep.xyz = xT[:, :NTOTAL]` replaces the
/// *whole* coordinate array with the idealized backbone, so motif sidechains
/// are gone from here on. They come back at output time from `indep_orig`, not
/// from this structure.
fn diffuse(
    indep: &Indep,
    is_diffused: &[bool],
    t: f32,
    opt: &Options,
    igso3: &Igso3,
    ctx: &mut Ctx,
) -> Indep {
    let l = indep.len();
    let xyz = add_fake_frame_legs(&indep.xyz, l, &indep.is_sm, ctx);
    let (rots, trans) = rigid_frames_from_atom_14(&xyz, l, NTOTAL);
    let rigids_0 = Rigids { rots, trans };
    let rigids_t = forward_marginal(
        &rigids_0,
        t,
        is_diffused,
        opt.center_noise_sample,
        igso3,
        ctx,
    );
    let xt = atom37_from_rigid(&rigids_t, ctx);
    let mut out = indep.clone();
    out.xyz = narrow_atom37(&xt, l);
    out
}

/// `aa_model.add_fake_peptide_frame` = fake legs, then
/// `idealize_peptide_frames`. Draws 6, 7 and 8.
fn add_fake_peptide_frame(indep: &Indep, ctx: &mut Ctx) -> Indep {
    let l = indep.len();
    let xyz = add_fake_frame_legs(&indep.xyz, l, &indep.is_sm, ctx);
    let (rots, trans) = rigid_frames_from_atom_14(&xyz, l, NTOTAL);
    let atom37 = atom37_from_rigid(&Rigids { rots, trans }, ctx);
    let mut out = indep.clone();
    out.xyz = narrow_atom37(&atom37, l);
    out
}

/// `xT[:, :ChemData().NTOTAL]` — atom37 down to the 36-slot layout `Indep`
/// uses. Slots 5..36 are zero, not NaN: `compute_backbone` writes zeros there
/// and `diffuse` copies them through.
fn narrow_atom37(atom37: &[f32], l: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; l * NTOTAL * 3];
    for i in 0..l {
        for a in 0..NTOTAL {
            for c in 0..3 {
                out[(i * NTOTAL + a) * 3 + c] = atom37[(i * N_ATOM37 + a) * 3 + c];
            }
        }
    }
    out
}

/// `aa_model.diffuse_then_add_conditional`.
///
/// The unconditional pass diffuses **every** row — `torch.ones_like(is_diffused)`
/// — including the ligand, which is why the noiser sees all L rows and not just
/// the 20 designed ones. The conditional structure is then that same noisy
/// structure with the motif rows overwritten by an *idealized* copy of the
/// original.
fn diffuse_then_add_conditional(
    indep: &Indep,
    is_diffused: &[bool],
    t_step_input: usize,
    opt: &Options,
    igso3: &Igso3,
    ctx: &mut Ctx,
) -> Result<(Indep, Indep), InitError> {
    let l = indep.len();
    let t = t_step_input as f32 / opt.big_t as f32;
    let all = vec![true; l];
    let indep_uncond = diffuse(indep, &all, t, opt, igso3, ctx);

    if opt.independently_center_diffuseds && t_step_input == opt.big_t {
        return Err(InitError::Unsupported(
            "diffuser.independently_center_diffuseds recenters the diffused and \
             motif clouds separately; no measured run exercises it"
                .into(),
        ));
    }

    let idealized = add_fake_peptide_frame(indep, ctx);
    let mut indep_cond = indep_uncond.clone();
    for i in 0..l {
        if !is_diffused[i] {
            let o = i * NTOTAL * 3;
            indep_cond.xyz[o..o + NTOTAL * 3]
                .copy_from_slice(&idealized.xyz[o..o + NTOTAL * 3]);
        }
    }
    Ok((indep_uncond, indep_cond))
}

impl SampleInit {
    /// Build everything the sampler needs from a PDB, a ligand list and a
    /// contig string.
    ///
    /// `ctx` must be seeded exactly as `run_inference.py:seed_all(i_des +
    /// seed_offset)` leaves the torch generator — the nine draws below come
    /// straight off it.
    pub fn run(
        pdb_text: &str,
        ligands: &[String],
        topo: &LigandSet,
        contigs: &str,
        opt: &Options,
        igso3: &Igso3,
        ctx: &mut Ctx,
        py: &mut Option<&mut crate::rng::pyrandom::PyRandom>,
    ) -> Result<Self, InitError> {
        if opt.preserve_motif_sidechains {
            return Err(InitError::Unsupported(
                "diffuser.preserve_motif_sidechains takes the other branch of \
                 `diffuse`, which no measured run exercises"
                    .into(),
            ));
        }
        let feats = pdb::parse_pdb_str(pdb_text, true, true);
        let cmap = ContigMap::parse_with(&feats, contigs, opt.length, py)
            .map_err(InitError::Contig)?;
        let indep_orig = make_indep(&feats, ligands, topo).map_err(InitError::Indep)?;

        let init_crds = crate::chemical::table_f32("INIT_CRDS");
        let (indep, masks) =
            insert_contig_pre_atomization(&indep_orig, &cmap, &opt.has_termini, &init_crds.data);

        let is_diffused: Vec<bool> = masks.is_res_str_shown.iter().map(|s| !s).collect();
        let is_masked_seq: Vec<bool> = masks.is_res_seq_shown.iter().map(|s| !s).collect();

        // `get_t_inference`: the full trajectory starts at T, a partial one at
        // partial_T.
        let t_step_input = opt.partial_t.unwrap_or(opt.big_t);
        if let Some(pt) = opt.partial_t {
            // `validate_partial_diffusion`. The coordinate check is the one
            // that matters: partial diffusion starts from the *input* structure,
            // so every residue the contig implies must actually have coordinates
            // in the PDB, and the contig must be an identity mapping.
            if pt > opt.big_t {
                return Err(InitError::Unsupported(format!(
                    "diffuser.partial_T = {pt} exceeds diffuser.T = {}",
                    opt.big_t
                )));
            }
            let n_sm = indep.is_sm.iter().filter(|s| **s).count();
            if indep.len() != feats.residues.len() + n_sm {
                return Err(InitError::Unsupported(format!(
                    "partial diffusion needs a coordinate in the input PDB for every \
                     residue the contig implies: {} rows vs {} + {n_sm}",
                    indep.len(),
                    feats.residues.len()
                )));
            }
            if (0..indep.len()).any(|i| indep.is_sm[i] && is_diffused[i]) {
                return Err(InitError::Unsupported(
                    "partial diffusion requires every ligand atom to be in the motif".into(),
                ));
            }
            if cmap.hal_idx0 != cmap.ref_idx0 {
                return Err(InitError::Unsupported(
                    "partial diffusion requires the contig to map every position to the \
                     same index it had in the input PDB"
                        .into(),
                ));
            }
        }

        let (mut indep_uncond, mut indep_cond) =
            diffuse_then_add_conditional(&indep, &is_diffused, t_step_input, opt, igso3, ctx)?;
        mask_indep(&mut indep_uncond, &is_masked_seq)?;
        mask_indep(&mut indep_cond, &is_masked_seq)?;

        Ok(SampleInit {
            indep: indep_cond,
            indep_orig: indep,
            indep_uncond,
            is_diffused,
            is_masked_seq,
            t_step_input,
            cmap,
        })
    }
}
