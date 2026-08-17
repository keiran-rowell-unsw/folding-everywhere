//! `run_inference.py:main` — one design, from a PDB path to the output text.
//!
//! This is the top of the port: it seeds the three generators the way
//! `seed_all` does, builds the starting structure (`src/sample_init.rs`), runs
//! the denoising loop (`src/sampler.rs`) and assembles the output files
//! (`src/output.rs`).

use crate::chemical_gen::{NHEAVY, NTOTAL};
use crate::indep::Indep;
use crate::ligand::LigandSet;
use crate::model::rf::RoseTTAFold;
use crate::nn::Ctx;
use crate::noiser::Igso3;
use crate::openfold::N_ATOM37;
use crate::output;
use crate::rng::torch::Mt19937;
use crate::sample_init::{Options as InitOptions, SampleInit};
use crate::sampler::{run_loop, SamplerOptions};

/// Everything the CLI collects.
pub struct DesignConfig {
    pub input_pdb: String,
    pub ligands: Vec<String>,
    pub contigs: String,
    pub big_t: usize,
    pub final_step: usize,
    pub seed_offset: u64,
    pub deterministic: bool,
    pub rots_exp_rate: i64,
    /// `inference.str_self_cond`
    pub str_self_cond: bool,
    /// `diffuser.partial_T` — start the trajectory part-way instead of at T
    pub partial_t: Option<usize>,
    /// `contigmap.length`, verbatim (e.g. "180-180"); parsed by `contig.rs`
    pub length: Option<String>,
}

impl Default for DesignConfig {
    fn default() -> Self {
        DesignConfig {
            input_pdb: String::new(),
            ligands: Vec::new(),
            contigs: String::new(),
            big_t: 100,
            final_step: 1,
            seed_offset: 0,
            deterministic: true,
            rots_exp_rate: 10,
            str_self_cond: false,
            partial_t: None,
            length: None,
        }
    }
}

pub struct DesignOutput {
    /// The `.pdb` text, byte-for-byte what upstream writes.
    pub pdb: String,
    /// `[n_steps][L * 37 * 3]`, newest first (upstream flips both stacks).
    pub px0_stack: Vec<Vec<f32>>,
    pub denoised_stack: Vec<Vec<f32>>,
    pub ts: Vec<usize>,
    pub indep: Indep,
    pub is_diffused: Vec<bool>,
}

/// `run_inference.py:seed_all` — all three generators from one integer.
///
/// Only the torch stream is consumed on this configuration (measured: numpy's
/// and CPython's position counters are unchanged across `sample_init`), but
/// they are seeded anyway so a configuration that does reach them starts from
/// the right place.
pub fn seed_all(seed: u64) -> Ctx {
    Ctx::new(Mt19937::new(seed))
}

/// `save_outputs`, for `contig_as_guidepost = False` and
/// `idealize_sidechain_outputs = False`.
///
/// The stack index is 0 *after* upstream's flip, i.e. the **last** step's
/// prediction — `px0_stack[0]` here is already in that order.
#[allow(clippy::too_many_arguments)]
pub fn save_outputs(
    px0_last: &[f32],
    indep: &Indep,
    indep_orig: &Indep,
    is_diffused: &[bool],
    ligand_names: &[String],
    input_pdb_text: &str,
    topo: &LigandSet,
) -> String {
    let l = indep.len();

    // ---- motif sidechains back, then idealize the backbone ---------------
    // `px0_xyz_stack[..., :NHEAVY, :]` — the 37-atom prediction is truncated
    // before the sidechains are restored, and the restored coordinates come
    // from the *pre-diffusion* structure.
    let mut xyz = vec![0.0f32; l * NHEAVY * 3];
    for i in 0..l {
        for a in 0..NHEAVY {
            for c in 0..3 {
                xyz[(i * NHEAVY + a) * 3 + c] = px0_last[(i * N_ATOM37 + a) * 3 + c];
            }
        }
    }
    let mut orig23 = vec![0.0f32; l * NHEAVY * 3];
    for i in 0..l {
        for a in 0..NHEAVY {
            for c in 0..3 {
                orig23[(i * NHEAVY + a) * 3 + c] = indep_orig.xyz[(i * NTOTAL + a) * 3 + c];
            }
        }
    }
    let act: Vec<bool> = is_diffused.iter().map(|d| !d).collect();
    output::add_implicit_side_chain_atoms(&indep.seq, &act, &mut xyz, &orig23, NHEAVY);

    let protein_rows: Vec<usize> =
        (0..l).filter(|&i| crate::geom::is_protein(indep.seq[i])).collect();
    output::idealize_bb_atoms(&mut xyz, &indep.idx, &protein_rows, NHEAVY);

    // ---- the intermediate stream, then the round-trip --------------------
    let chains = output::chain_letters(indep);
    let seq_design: Vec<i64> = indep
        .seq
        .iter()
        .map(|&s| if s == 20 || s == 21 { 0 } else { s })
        .collect();
    let stream = output::write_traj(
        &xyz,
        NHEAVY,
        &seq_design,
        &indep.idx,
        &chains,
        ligand_names,
        Some(&indep.bond_feats),
        0,
    );
    let stream = output::rewrite(&stream, topo, ligand_names);
    output::rename_ligand_atoms(input_pdb_text, &stream)
}

/// One design, with the CLI's progress reporting.
pub fn run_design(
    model: &RoseTTAFold,
    cfg: &DesignConfig,
    input_pdb_text: &str,
    topo: &LigandSet,
    igso3: &Igso3,
    i_des: usize,
) -> Result<DesignOutput, Box<dyn std::error::Error>> {
    run_design_with(model, cfg, input_pdb_text, topo, igso3, i_des,
                    |it, t, total| eprintln!("  step {it}/{total}  (t = {t})"))
}

/// One design, reporting each denoising step to `on_step(iteration, t, total)`.
///
/// `run_design` is this with the CLI's stderr reporter; a GUI wants the same
/// information on its own progress bar, and the loop is far too long (T is 100
/// by default) to leave a caller with no signal until it finishes.
pub fn run_design_with(
    model: &RoseTTAFold,
    cfg: &DesignConfig,
    input_pdb_text: &str,
    topo: &LigandSet,
    igso3: &Igso3,
    i_des: usize,
    mut on_step: impl FnMut(usize, usize, usize),
) -> Result<DesignOutput, Box<dyn std::error::Error>> {
    // `get_sampler`: `seed_all()` once before the sampler is built, then
    // `seed_all(i_des + seed_offset)` per design. Only the second one matters
    // for the trajectory, because nothing between them draws.
    let seed = i_des as u64 + cfg.seed_offset;
    let mut ctx = seed_all(seed);
    // `seed_all` also seeds CPython's generator; only a variable-length contig
    // draws from it, and only inside `get_sampled_mask`.
    let mut py = crate::rng::pyrandom::PyRandom::new(seed);

    let length = match &cfg.length {
        Some(l) => Some(crate::contig::parse_length(l)?),
        None => None,
    };
    let init_opt = InitOptions {
        big_t: cfg.big_t,
        partial_t: cfg.partial_t,
        length,
        ..InitOptions::default()
    };
    let init = SampleInit::run(
        input_pdb_text,
        &cfg.ligands,
        topo,
        &cfg.contigs,
        &init_opt,
        igso3,
        &mut ctx,
        &mut Some(&mut py),
    )?;

    let mut indep = init.indep;
    let atom_frames = topo.atom_frames();
    let opt = SamplerOptions {
        big_t: cfg.big_t,
        final_step: cfg.final_step,
        rots_exp_rate: cfg.rots_exp_rate,
        str_self_cond: cfg.str_self_cond,
        partial_t: cfg.partial_t,
        prepro: crate::prepro::PreproOptions {
            big_t: cfg.big_t,
            ..crate::prepro::PreproOptions::default()
        },
    };
    let traj = run_loop(
        model,
        &mut indep,
        &init.is_diffused,
        &atom_frames,
        init.t_step_input,
        &opt,
        &mut ctx,
        |it, t, _| on_step(it + 1, t, init.t_step_input),
    );

    // upstream flips both stacks before writing
    let px0_stack: Vec<Vec<f32>> = traj.px0.iter().rev().cloned().collect();
    let denoised_stack: Vec<Vec<f32>> = traj.denoised.iter().rev().cloned().collect();
    let ligand_names = ligand_name_array(&indep, &cfg.ligands, topo);

    let pdb = save_outputs(
        &px0_stack[0],
        &indep,
        &init.indep_orig,
        &init.is_diffused,
        &ligand_names,
        input_pdb_text,
        topo,
    );

    Ok(DesignOutput {
        pdb,
        px0_stack,
        denoised_stack,
        ts: traj.ts,
        indep,
        is_diffused: init.is_diffused,
    })
}

/// `contig_map.ligand_names` — `''` for protein rows, the ligand's residue name
/// for each of its atoms. Set by the `update_inference_state` transform.
fn ligand_name_array(indep: &Indep, _ligands: &[String], topo: &LigandSet) -> Vec<String> {
    let mut out = vec![String::new(); indep.len()];
    let mut sm_rows: Vec<usize> = (0..indep.len()).filter(|&i| indep.is_sm[i]).collect();
    sm_rows.sort_unstable();
    let mut k = 0usize;
    for name in topo.names() {
        let n = topo.get(name).map(|t| t.n_atoms).unwrap_or(0);
        for _ in 0..n {
            if k < sm_rows.len() {
                out[sm_rows[k]] = name.clone();
                k += 1;
            }
        }
    }
    out
}
