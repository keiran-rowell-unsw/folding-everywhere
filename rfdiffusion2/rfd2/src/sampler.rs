//! `inference/model_runners.py:NRBStyleSelfCond.sample_step` and the loop
//! around it — rung 7.
//!
//! **The demo runs `NRBStyleSelfCond`, not `FlowMatching`.**
//! `inference.model_runner` is `NRBStyleSelfCond` in `base.yaml` and `aa.yaml`
//! does not override it. The two classes take different reverse steps —
//! `FlowMatching` uses `get_dt`/`apply_grads`, this one calls
//! `diffuser.reverse`, an Euler step on the SE(3) interpolant — and they
//! disagree on step size, so porting the wrong one gives a plausible
//! trajectory that is not the reference's.
//!
//! One step, in the order the generator sees it:
//!
//! ```text
//! prepro(indep, t, is_diffused)            src/prepro.rs   MUTATES indep.xyz
//! forward_from_rfi(model, rfi)             src/score.rs    ~2.64 M draws + psi
//! rigids_t   = rigid_frames_from_atom_14(rfi.xyz)
//! rigid_pred = out.rigids_pred()                           the last block
//! reverse(rigids_t, rigid_pred, t/T, 1/T)  <- diffused rows only
//! get_x_t_1(rigids_t2, indep.xyz)                          one more psi draw
//! ```
//!
//! ## Two scalars that are not what they look like
//!
//! `get_scaling` computes `c * exp(-c t) / (exp(-c t) - exp(-c))` where `c` is
//! `torch.tensor(10)` — an **int64** tensor. So `exp(-c*t)` sees an fp32
//! argument and goes through the pinned f64 path, while `exp(-c)` sees int64,
//! is not promoted, and runs the stock fp32 kernel. Measured at T = 2:
//! `scaling(0) = 10.00045394897461` and `scaling(0.5) = 10.06783676147461`;
//! `tests/parity_sampler.rs` asserts both bit-for-bit rather than trusting the
//! derivation.
//!
//! And `_trans_euler_step` divides by `1 - t` unconditionally — the `linear`
//! translation schedule is hard-coded into the Euler step, not read from
//! `_trans_cfg`, which only `FlowMatching.get_dt` consults.

use crate::chemical_gen::{NHEAVY, NTOTAL};
use crate::indep::Indep;
use crate::model::rf::RoseTTAFold;
use crate::nn::Ctx;
use crate::noiser::{geodesic_t, rigid_frames_from_atom_14, Rigids};
use crate::openfold::{atom37_from_rigid, N_ATOM37};
use crate::prepro::{prepro, PreproOptions};
use crate::score::{forward_from_rfi, ScoreOut};

/// `interpolant.Interpolant.get_scaling` for `sample_schedule = normed_exp`.
///
/// See the module header for why the two exponentials are evaluated
/// differently. `exp_rate` is an integer in the config and is kept as one here
/// so the int64/fp32 split is explicit.
pub fn get_scaling_normed_exp(t: f32, exp_rate: i64) -> f32 {
    let c = exp_rate as f32;
    // `torch.exp(-c*t)`: fp32 argument -> pinned, f64 interior, one narrowing
    let ect = (((-c * t) as f64).exp()) as f32;
    // `torch.exp(-c)`: int64 argument -> NOT promoted, stock fp32 kernel
    let ec = (-c).exp();
    c * ect / (ect - ec)
}

/// `interpolant.Interpolant._trans_euler_step`.
fn trans_euler_step(d_t: f32, t: f32, trans_1: &[f32], trans_t: &[f32]) -> Vec<f32> {
    trans_1
        .iter()
        .zip(trans_t)
        .map(|(p, x)| {
            let vf = (p - x) / (1.0 - t);
            x + vf * d_t
        })
        .collect()
}

/// `interpolant.Interpolant._rots_euler_step` — a geodesic step whose length is
/// set by the `normed_exp` schedule.
fn rots_euler_step(
    d_t: f32,
    t: f32,
    rotmats_1: &[f32],
    rotmats_t: &[f32],
    exp_rate: i64,
) -> Vec<f32> {
    geodesic_t(get_scaling_normed_exp(t, exp_rate) * d_t, rotmats_1, rotmats_t)
}

/// `noisers.NormalizingFlow.reverse` — the Euler step, applied to the diffused
/// rows only and copied back into the motif's frame.
///
/// Note the time inversion: the public `t` runs from clean to prior, and
/// `reverse_all` immediately flips it (`t_1 = 1 - t`), so the Euler step's own
/// notion of time is the interpolant's.
pub fn reverse(
    rigid_t: &Rigids,
    rigid_pred: &Rigids,
    t: f32,
    dt: f32,
    is_diffused: &[bool],
    exp_rate: i64,
) -> Rigids {
    let l = rigid_t.len();
    let sel: Vec<usize> = (0..l).filter(|&i| is_diffused[i]).collect();
    let gather = |src: &[f32], w: usize| -> Vec<f32> {
        let mut v = Vec::with_capacity(sel.len() * w);
        for &i in &sel {
            v.extend_from_slice(&src[i * w..i * w + w]);
        }
        v
    };
    let t_1 = 1.0 - t;
    let trans_out = trans_euler_step(
        dt,
        t_1,
        &gather(&rigid_pred.trans, 3),
        &gather(&rigid_t.trans, 3),
    );
    let rots_out = rots_euler_step(
        dt,
        t_1,
        &gather(&rigid_pred.rots, 9),
        &gather(&rigid_t.rots, 9),
        exp_rate,
    );

    let mut out = rigid_t.clone();
    for (j, &i) in sel.iter().enumerate() {
        out.trans[i * 3..i * 3 + 3].copy_from_slice(&trans_out[j * 3..j * 3 + 3]);
        out.rots[i * 9..i * 9 + 9].copy_from_slice(&rots_out[j * 9..j * 9 + 9]);
    }
    out
}

/// `model_runners.get_x_t_1`.
///
/// Draws its own `psi_pred` — that is the second draw of the step, after the
/// one inside `forward_from_rfi`. Motif rows are then restored from `xyz`,
/// but only the **heavy** slots: `[:NHEAVY]`, not the whole row, so the NaN
/// pad above `NHEAVY` survives from the freshly built backbone.
pub fn get_x_t_1(
    rigids_t: &Rigids,
    xyz: &[f32],
    is_diffused: &[bool],
    ctx: &mut Ctx,
) -> Vec<f32> {
    let l = rigids_t.len();
    let a37 = atom37_from_rigid(rigids_t, ctx);
    let mut out = vec![0.0f32; l * NTOTAL * 3];
    for i in 0..l {
        for a in 0..NTOTAL {
            for c in 0..3 {
                out[(i * NTOTAL + a) * 3 + c] = a37[(i * N_ATOM37 + a) * 3 + c];
            }
        }
    }
    for i in 0..l {
        if !is_diffused[i] {
            for a in 0..NHEAVY {
                for c in 0..3 {
                    out[(i * NTOTAL + a) * 3 + c] = xyz[(i * NTOTAL + a) * 3 + c];
                }
            }
        }
    }
    out
}

/// Everything the loop needs that is not in `Indep`.
#[derive(Clone, Debug)]
pub struct SamplerOptions {
    /// `diffuser.T`
    pub big_t: usize,
    /// `inference.final_step`
    pub final_step: usize,
    /// `diffuser.rots.exp_rate`
    pub rots_exp_rate: i64,
    /// `inference.str_self_cond`
    pub str_self_cond: bool,
    /// `diffuser.partial_T`, only used by the self-conditioning guard here
    pub partial_t: Option<usize>,
    pub prepro: PreproOptions,
}

impl Default for SamplerOptions {
    fn default() -> Self {
        SamplerOptions {
            big_t: 100,
            final_step: 1,
            rots_exp_rate: 10,
            str_self_cond: false,
            partial_t: None,
            prepro: PreproOptions::default(),
        }
    }
}

/// `aa_model.self_cond` — show the model its own previous prediction.
///
/// The `Rfi` carries two template slots; slot 0 is the current noisy structure
/// and slot 1 is normally zero. Self-conditioning fills slot 1 with a `t2d`
/// built from the **last block's** coordinates of the *previous* step, plus
/// that structure's CA in `xyz_t`.
///
/// Two details worth naming. The 33 atom slots above the backbone are filled
/// with `torch.zeros`, not NaN, so `generate_Cbeta` sees real numbers; and the
/// write into `t2d` is `[..., :base_d_t2d]`, i.e. only the geometric channels —
/// any `extra_t2d` appended after them is left alone (it is width 0 here, so
/// the distinction does not bite yet, but a configuration with extra_t2d would
/// silently lose those channels if this overwrote the whole row).
pub fn self_cond(
    rfi: &mut crate::model::rf::Rfi,
    xyz_last: &[f32],
    l: usize,
    is_sm: &[bool],
    atom_frames: &[i64],
    use_cb: bool,
) {
    // `torch.cat((xyz_last, zeros), dim=-2)` -> [L, NTOTAL, 3]
    let mut xyz = vec![0.0f32; l * NTOTAL * 3];
    for i in 0..l {
        for a in 0..3 {
            for c in 0..3 {
                xyz[(i * NTOTAL + a) * 3 + c] = xyz_last[(i * 3 + a) * 3 + c];
            }
        }
    }
    let t2d = crate::t2d::get_t2d(&xyz, l, is_sm, atom_frames, use_cb);
    let w = crate::t2d::T2D_WIDTH;
    let full = rfi.t2d.last();
    debug_assert!(full >= w);
    for i in 0..l * l {
        rfi.t2d.data[(l * l + i) * full..(l * l + i) * full + w]
            .copy_from_slice(&t2d[i * w..i * w + w]);
    }
    for i in 0..l {
        for c in 0..3 {
            rfi.xyz_t.data[(l + i) * 3 + c] = xyz[(i * NTOTAL + 1) * 3 + c];
        }
    }
}

/// What one `sample_step` produces.
pub struct StepOut {
    /// `[L, 37, 3]` — the model's prediction of the clean structure
    pub px0: Vec<f32>,
    /// `[L, NTOTAL, 3]` — the coordinates the next step starts from
    pub x_t_1: Vec<f32>,
    pub score: ScoreOut,
}

/// One denoising step.
///
/// `indep` is taken by `&mut` because `prepro` NaNs out the diffused rows'
/// sidechain slots in place and `get_x_t_1` reads that back — see
/// `src/prepro.rs`.
#[allow(clippy::too_many_arguments)]
pub fn sample_step(
    model: &RoseTTAFold,
    indep: &mut Indep,
    t: usize,
    is_diffused: &[bool],
    atom_frames: &[i64],
    opt: &SamplerOptions,
    prev_xyz: Option<&[f32]>,
    ctx: &mut Ctx,
) -> StepOut {
    let l = indep.len();
    let mut rfi = prepro(indep, t, is_diffused, atom_frames, &opt.prepro);

    // `all([t < T, t != partial_T, str_self_cond])` — so the first step of a
    // full trajectory never self-conditions (there is no previous prediction),
    // and neither does the first step of a partial one.
    if opt.str_self_cond && t < opt.big_t && Some(t) != opt.partial_t {
        let prev = prev_xyz.expect(
            "str_self_cond is on and t < T, so the previous step's prediction \
             must have been threaded in",
        );
        self_cond(&mut rfi, prev, l, &indep.is_sm, atom_frames, opt.prepro.use_cb);
    }
    let score = forward_from_rfi(model, &rfi, ctx);

    let (rots_t, trans_t) = rigid_frames_from_atom_14(&rfi.xyz.data, l, NTOTAL);
    let rigids_t = Rigids { rots: rots_t, trans: trans_t };
    let big_t = opt.big_t as f32;
    let rigids_t2 = reverse(
        &rigids_t,
        score.rigids_pred(),
        t as f32 / big_t,
        1.0 / big_t,
        is_diffused,
        opt.rots_exp_rate,
    );
    let x_t_1 = get_x_t_1(&rigids_t2, &indep.xyz, is_diffused, ctx);
    StepOut { px0: score.px0().to_vec(), x_t_1, score }
}

/// The trajectory: `px0` and `x_t` per step, in the order they were produced.
///
/// `run_inference.py` flips both stacks before writing them ("for better
/// visualization in pymol"); that flip belongs to the output writer, not here.
pub struct Trajectory {
    pub px0: Vec<Vec<f32>>,
    pub denoised: Vec<Vec<f32>>,
    /// The timesteps actually taken, `T .. final_step` descending.
    pub ts: Vec<usize>,
}

/// `run_inference.sample`'s loop over `ts`.
pub fn run_loop(
    model: &RoseTTAFold,
    indep: &mut Indep,
    is_diffused: &[bool],
    atom_frames: &[i64],
    t_step_input: usize,
    opt: &SamplerOptions,
    ctx: &mut Ctx,
    mut on_step: impl FnMut(usize, usize, &StepOut),
) -> Trajectory {
    let ts: Vec<usize> = (opt.final_step..=t_step_input).rev().collect();
    let mut traj = Trajectory { px0: Vec::new(), denoised: Vec::new(), ts: ts.clone() };
    // `rfo` in upstream's loop: the previous step's network output, which
    // self-conditioning reads the last block's coordinates out of.
    let mut prev_xyz: Option<Vec<f32>> = None;
    for (it, &t) in ts.iter().enumerate() {
        let out = sample_step(
            model,
            indep,
            t,
            is_diffused,
            atom_frames,
            opt,
            prev_xyz.as_deref(),
            ctx,
        );
        indep.xyz = out.x_t_1.clone();
        prev_xyz = out.score.model.sim.xyz_stack.last().cloned();
        on_step(it, t, &out);
        traj.px0.push(out.px0);
        traj.denoised.push(out.x_t_1);
    }
    traj
}
