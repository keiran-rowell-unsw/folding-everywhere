//! Fully standalone, seed-deterministic ESMFold2 fold in pure Rust — from a bare
//! amino-acid sequence, with no PyTorch/Python and no precomputed fixtures.
//!
//! Features come from [`featurize`], the stochastic diffusion noise from
//! [`rng::TorchRng`] (bit-exact to a PyTorch run pinned at `seed`), drawn in torch's
//! exact order: z-init `trunc_normal` -> per-loop LM `dropout` x4 -> `x_init` ->
//! per-step rotation/translation/churn. The compute path mirrors `bench_fold`.

use crate::atom::{self, AtomInputs};
use crate::featurize::Tables;
use crate::ops::linear_f64;
use crate::rng::TorchRng;
use crate::tensor::Tensor;
use crate::weights::Weights;
use crate::{confidence, diffusion, msa, parcae, pipeline, trunk};

// Diffusion sampler config (checkpoint config.json; validated against fixtures).
const SIGMA_DATA: f32 = 16.0;
const INF_P: f32 = 7.0;
const INF_S_MAX: f32 = 160.0;
const INF_S_MIN: f32 = 0.0004;
const MAX_INF_SIGMA: f32 = 256.0;
const PAIR_DIM: usize = 256;
const LM_DROPOUT: f32 = 0.25;

pub struct FoldOutput {
    pub l: usize,
    pub n_atoms: usize,
    pub coords: Vec<f32>, // [n_atoms*3]  — sample 0 (the written structure)
    pub plddt: Vec<f32>,  // [L]          — sample 0
    pub plddt_mean: f32,  // sample 0
    pub ptm: f32,         // sample 0
    pub iptm: f32,        // sample 0
    pub complex_plddt: f32, // sample 0
    pub pdb: String,      // ready-to-write PDB text (all-atom), sample 0
}

/// Karras power-law noise schedule, then clip to `MAX_INF_SIGMA` and prepend the cap
/// (reproduces `inference_noise_schedule` + the sample() clip).
pub fn karras_schedule(num_steps: usize) -> Vec<f32> {
    let inv_p = 1.0f32 / INF_P;
    let smax_ip = INF_S_MAX.powf(inv_p);
    let smin_ip = INF_S_MIN.powf(inv_p);
    let mut s: Vec<f32> = (0..num_steps)
        .map(|k| {
            let base = smax_ip + (k as f32 / (num_steps as f32 - 1.0)) * (smin_ip - smax_ip);
            SIGMA_DATA * base.powf(INF_P)
        })
        .collect();
    s.push(0.0); // F.pad(..., (0,1), value=0)
    let mut clipped: Vec<f32> = s.into_iter().filter(|&x| x <= MAX_INF_SIGMA).collect();
    clipped.insert(0, MAX_INF_SIGMA);
    clipped
}

/// `_random_rotations(1)`: randn(4) -> sign-fixed unit quaternion -> 3x3 (row-major 9).
fn quat_to_mat(q: [f32; 4]) -> Vec<f32> {
    let scale = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
    let sign = if q[0] < 0.0 { -scale } else { scale };
    let (r, i, j, k) = (q[0] / sign, q[1] / sign, q[2] / sign, q[3] / sign);
    let two_s = 2.0 / (r * r + i * i + j * j + k * k);
    vec![
        1.0 - two_s * (j * j + k * k), two_s * (i * j - k * r), two_s * (i * k + j * r),
        two_s * (i * j + k * r), 1.0 - two_s * (i * i + k * k), two_s * (j * k - i * r),
        two_s * (i * k - j * r), two_s * (j * k + i * r), 1.0 - two_s * (i * i + j * j),
    ]
}

/// Run the complete fold. `num_sampling_steps` defaults to 14 (clips to ~10 actual steps).
/// This produces a **single** diffusion sample (ESMFold2's `num_diffusion_samples` is fixed
/// to 1 here); the released model likewise writes sample 0 of whatever N it generates.
pub fn fold(
    seq: &str,
    seed: u64,
    w_esmc: &Weights,
    w: &Weights,
    num_loops: usize,
    num_sampling_steps: usize,
) -> FoldOutput {
    fold_cb(seq, seed, w_esmc, w, num_loops, num_sampling_steps, &mut |_, _| {})
}

/// As [`fold`], reporting progress via `prog(message, fraction 0..1)` — the same
/// contract as ESMFold1's `fold_cb`. Stage weighting: ESM-C 6B ~55 %, parcae
/// trunk loops ~23 %, diffusion sampling ~12 %, confidence/setup the rest.
/// Numerically identical to `fold`.
#[allow(clippy::too_many_arguments)]
pub fn fold_cb(
    seq: &str,
    seed: u64,
    w_esmc: &Weights,
    w: &Weights,
    num_loops: usize,
    num_sampling_steps: usize,
    prog: &mut dyn FnMut(&str, f32),
) -> FoldOutput {
    use crate::config::{ESMC_N_LAYERS, FOLDING_TRUNK_LAYERS};
    prog("Featurizing sequence…", 0.0);
    let feat = Tables::load().featurize(seq);
    let l = feat.l;
    let n = feat.n_atoms;

    // ---- atom inputs (one-hot + masked) ----
    let ref_pos = feat.ref_pos.clone();
    let ref_space_uid: Vec<f32> = feat.ref_space_uid.iter().map(|&x| x as f32).collect();
    let ref_charge: Vec<f32> = feat.ref_charge.iter().map(|&x| x as f32).collect();
    let ref_element = feat.ref_element_onehot();
    let ref_names = feat.ref_atom_name_onehot();
    let atom_mask_b = feat.atom_attention_mask.clone();
    let atom_to_token = feat.atom_to_token.clone();
    let inp = AtomInputs {
        ref_pos: &ref_pos, ref_space_uid: &ref_space_uid, ref_charge: &ref_charge,
        ref_element: &ref_element, ref_atom_name_chars: &ref_names,
        atom_mask: &atom_mask_b, atom_to_token: &atom_to_token, n_atoms: n, n_tokens: l,
    };

    // ---- token-level features ----
    let aatype = Tensor::new(feat.res_type_onehot(), vec![l, 33]);
    let profile = aatype.clone();
    let msa_oh = Tensor::new(feat.res_type_onehot(), vec![l, 1, 33]);
    let deletion_mean = vec![0.0f32; l];
    let has_deletion = vec![0.0f32; l];
    let deletion_value = vec![0.0f32; l];
    let msa_attn = vec![1.0f32; l];
    let residue_index: Vec<i64> = (0..l as i64).collect();
    let token_index = residue_index.clone();
    let asym_id = vec![0i64; l];
    let sym_id = vec![0i64; l];
    let entity_id = vec![1i64; l];

    // ---- schedule + RNG draws (torch order; computes in between draw nothing) ----
    let schedule = karras_schedule(num_sampling_steps);
    let steps = schedule.len() - 1;
    let mut g = TorchRng::new(seed);

    // (1) z-init trunc_normal [1,L,L,256]
    let std = (2.0f32 / (5.0 * PAIR_DIM as f32)).sqrt();
    let mut z_rand_v = vec![0.0f32; l * l * PAIR_DIM];
    g.fill_trunc_normal(&mut z_rand_v, 0.0, std, -3.0 * std, 3.0 * std);
    let z_rand = Tensor::new(z_rand_v, vec![l, l, PAIR_DIM]);

    // (2) per-loop dropout masks. The trunk runs total_steps = max(1, num_loops+1)
    //     iterations (modeling_esmfold2.py:896), one LM dropout draw each.
    let total_steps = (num_loops + 1).max(1);
    let drop_masks: Vec<Vec<f32>> = (0..total_steps)
        .map(|_| { let mut m = vec![0.0f32; l * l * PAIR_DIM]; g.fill_dropout_scale(&mut m, LM_DROPOUT); m })
        .collect();

    // (3) x_init randn [n,3] * schedule[0]. This fold produces a single diffusion sample
    //     (num_diffusion_samples = 1); torch draws x_init as randn[n,3].
    let mut xinit_raw = vec![0.0f32; n * 3];
    g.fill_randn(&mut xinit_raw);
    let x_init = Tensor::new(xinit_raw.iter().map(|v| v * schedule[0]).collect(), vec![n, 3]);

    // (4) per step: rotation randn(4)->3x3, translation randn(3), churn randn(n*3)
    let mut r_aug = Vec::with_capacity(steps);
    let mut t_aug = Vec::with_capacity(steps);
    let mut churn = Vec::with_capacity(steps);
    for _ in 0..steps {
        let mut q = [0.0f32; 4];
        g.fill_randn(&mut q);
        r_aug.push(quat_to_mat(q));
        let mut t = vec![0.0f32; 3];
        g.fill_randn(&mut t);
        t_aug.push(t);
        let mut c = vec![0.0f32; n * 3];
        g.fill_randn(&mut c);
        churn.push(c);
    }

    // ---- compute (mirrors bench_fold) ----
    // ESM-C 6B language model (80 transformer layers) — the dominant cost: ~55 %.
    prog("Running ESM-C 6B language model…", 0.02);
    let lm_hidden = pipeline::compute_lm_hidden_states_cb(w_esmc, &feat.input_ids, &mut |layer| {
        prog(&format!("ESM-C 6B language model: layer {layer}/{ESMC_N_LAYERS}"),
             0.02 + 0.55 * layer as f32 / ESMC_N_LAYERS as f32);
    });
    prog("Building pair representation…", 0.58);
    let lm_z = trunk::language_model_shim(w, &lm_hidden);
    let x_inputs = atom::inputs_embedder(w, &inp, &aatype, &profile, &deletion_mean);

    let rel_pos = trunk::rel_pos(w, &residue_index, &asym_id, &sym_id, &entity_id, &token_index);
    let tb_feat = Tensor::new(vec![0.0f32; l * l], vec![l, l, 1]);
    let tbe = linear_f64(&tb_feat, &w.get("token_bonds.weight"), None);
    let zi1 = linear_f64(&x_inputs, &w.get("z_init_1.weight"), None);
    let zi2 = linear_f64(&x_inputs, &w.get("z_init_2.weight"), None);
    let z_init = parcae::build_z_init(&zi1, &zi2, &rel_pos, &tbe);
    let msa_pair = msa::encode(w, &z_init, &x_inputs, &msa_oh, &has_deletion, &deletion_value, &msa_attn);

    // per-loop LM dropout: lm_z_i = lm_z .* mask_i
    let lm_loops: Vec<Tensor> = drop_masks.iter().map(|m| {
        Tensor::new(lm_z.data.iter().zip(m).map(|(a, b)| a * b).collect(), lm_z.shape.clone())
    }).collect();
    let mask = vec![1.0f32; l * l];
    // parcae trunk loops (each runs FOLDING_TRUNK_LAYERS pair-update blocks): ~23 %.
    prog("Folding trunk…", 0.60);
    let z = parcae::run_loop_cb(w, &msa_pair, &z_rand, &lm_loops, &mask, &mut |lp, blk, nl| {
        let unit = ((lp - 1) * FOLDING_TRUNK_LAYERS + blk) as f32 / (nl * FOLDING_TRUNK_LAYERS) as f32;
        prog(&format!("Folding trunk — loop {lp}/{nl}: block {blk}/{FOLDING_TRUNK_LAYERS}"),
             0.60 + 0.23 * unit);
    });

    // diffusion structure sampling (~10 denoising steps): ~12 %.
    prog("Diffusion structure sampling…", 0.83);
    let tok_valid = vec![true; l];
    let coords = diffusion::sample_cb(w, &inp, &x_inputs, &z, &rel_pos, &tok_valid,
        &x_init, &schedule, &r_aug, &t_aug, &churn, &mut |step, total| {
            prog(&format!("Diffusion sampling: step {step}/{total}"),
                 0.83 + 0.12 * step as f32 / total as f32);
        });

    prog("Computing confidence (pLDDT / pTM)…", 0.96);
    let token_mask = vec![1.0f32; l];
    let atom_mask_f: Vec<f32> = atom_mask_b.iter().map(|&b| if b { 1.0 } else { 0.0 }).collect();
    let cout = confidence::confidence(w, &x_inputs, &z, &coords, &feat.distogram_atom_idx,
        &token_mask, &atom_to_token, &atom_mask_f, &asym_id, &rel_pos, &tbe);

    let plddt_mean = cout.plddt.iter().sum::<f32>() / l as f32;
    let seq_bytes: Vec<u8> = seq.bytes().collect();
    let pdb = crate::pdb::write_pdb(
        &coords.data, &ref_names, &atom_to_token, &residue_index, &seq_bytes,
        &asym_id, &atom_mask_f, &cout.plddt,
    );
    prog("Done", 1.0);
    FoldOutput {
        l, n_atoms: n, coords: coords.data, plddt: cout.plddt, plddt_mean,
        ptm: cout.ptm, iptm: cout.iptm, complex_plddt: cout.complex_plddt, pdb,
    }
}
