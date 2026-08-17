//! The Lennard-Jones term and its coordinate gradient (`rf2aa/loss/loss.py`'s
//! `LJLoss`), which the four refinement blocks feed to the SE(3) transformer as
//! extra degree-0 and degree-1 features.
//!
//! `LJLoss` is a `torch.autograd.Function` whose backward is just
//! `grad_output * dljEdx` — the gradient is computed *in the forward*. So this
//! is not an autodiff problem: it is the same forward, plus the analytic
//! `dE/dr` the reference already writes down.
//!
//! The parameters are not the defaults. `calc_lj_grads` passes `eps=1e-8`,
//! `normNviolations=False` and `useH=True`, and then — because `calc_lj`
//! forwards its `training` argument into `LJLoss.forward`'s
//! `norm_by_atoms_twice` slot — `norm_by_atoms_twice` is **True**. The
//! `norm_by_atoms_twice=False` that `calc_lj_grads` itself takes never reaches
//! the loss. That double normalisation is why the caller multiplies the whole
//! thing by `natoms` again ("#fd a bug in the original implementation meant
//! unnormalized lj grads were returned").

use crate::chemical;
use crate::chemical_gen::NTOTAL;

pub struct LjTables {
    /// `[NAATOKENS, NTOTAL]` — which atom slots exist
    pub aamask: Vec<bool>,
    /// `[NAATOKENS, NTOTAL, 5]`
    pub ljparams: Vec<f32>,
    /// `[NAATOKENS, NTOTAL, 4]` as bools
    pub ljcorr: Vec<bool>,
    /// `[NAATOKENS, NTOTAL, NTOTAL]`
    pub num_bonds: Vec<i64>,
}

impl Default for LjTables {
    fn default() -> Self {
        Self::new()
    }
}

impl LjTables {
    pub fn new() -> Self {
        LjTables {
            aamask: chemical::allatom_mask().0,
            ljparams: chemical::ljlk_parameters().data,
            ljcorr: chemical::lj_correction_parameters().0,
            num_bonds: chemical::num_bonds().0,
        }
    }
}

pub struct LjCfg {
    pub lj_lin: f32,
    pub lj_hb_dis: f32,
    pub lj_ohdon_dis: f32,
    pub lj_hbond_hdis: f32,
    pub eps: f32,
    /// `calc_lj_grads` hard-codes both of these.
    pub norm_n_violations: bool,
    pub use_h: bool,
}

impl Default for LjCfg {
    fn default() -> Self {
        LjCfg {
            lj_lin: 0.75,
            lj_hb_dis: 3.0,
            lj_ohdon_dis: 2.6,
            lj_hbond_hdis: 1.75,
            eps: 1e-8,
            norm_n_violations: false,
            use_h: true,
        }
    }
}

/// Result of the LJ forward: the (normalised) energy and `dE/dx` per atom.
pub struct LjOut {
    pub energy: f32,
    /// `[L, NTOTAL, 3]`
    pub dljedx: Vec<f32>,
}

/// `LJLoss.forward` — the energy and, as a by-product, the coordinate gradient.
///
/// `xs` is `[L, NTOTAL, 3]` (the all-atom coordinates, `xyzaa[..., :3]`).
#[allow(clippy::too_many_arguments)]
pub fn lj_forward(
    seq: &[i64],
    xs: &[f32],
    bond_feats: &[i64],
    dist_matrix: &[f32],
    t: &LjTables,
    cfg: &LjCfg,
) -> LjOut {
    let l = seq.len();
    let a = NTOTAL;
    let nat = if cfg.use_h { NTOTAL } else { 14 };
    let mut dljedx = vec![0.0f32; l * a * 3];

    // `natoms = sum(aamask[seq])` over the same atom subset the pairs use
    let mut natoms = 0i64;
    for &s in seq {
        for k in 0..nat {
            if t.aamask[s as usize * NTOTAL + k] {
                natoms += 1;
            }
        }
    }
    let natoms_f = natoms as f32;

    let get = |i: usize, k: usize, d: usize| xs[(i * a + k) * 3 + d];
    // `ljE.sum(dim=-1)` is a pinned reduction over every scored atom pair.
    let mut energy = 0.0f64;
    // `index_add_(..., alpha=1/natoms)`: the scale is a Python float, narrowed
    // to fp32, and it multiplies the (already natoms-divided) source term.
    let alpha = (1.0f64 / natoms as f64) as f32;
    let mut contribs: Vec<(u32, u32, [f32; 3])> = Vec::new();

    // `torch.triu_indices(L, L, 0)` — the upper triangle, diagonal included, in
    // row-major order.
    for ri in 0..l {
        for rj in ri..l {
            let (si, sj) = (seq[ri] as usize, seq[rj] as usize);
            // Ca-Ca cut-off at 24 A, computed on the DETACHED coordinates
            let mut d2 = 0.0f64;
            for d in 0..3 {
                let v = (get(ri, 1, d) - get(rj, 1, d)) as f64;
                d2 += v * v;
            }
            if d2.sqrt() >= 24.0 {
                continue;
            }
            let intrares = ri == rj;
            for ai in 0..nat {
                if !t.aamask[si * NTOTAL + ai] {
                    continue;
                }
                for aj in 0..nat {
                    if !t.aamask[sj * NTOTAL + aj] {
                        continue;
                    }
                    // upper triangle within a residue
                    if intrares && ai < aj {
                        continue;
                    }
                    // count-pair rules
                    if intrares
                        && t.num_bonds[(si * NTOTAL + ai) * NTOTAL + aj] < 4
                    {
                        continue;
                    }
                    if ri + 1 == rj {
                        let nb = t.num_bonds[(si * NTOTAL + ai) * NTOTAL + 2]
                            + t.num_bonds[(sj * NTOTAL) * NTOTAL + aj]
                            + 1;
                        if nb < 4 {
                            continue;
                        }
                    }
                    if ai == 1 && aj == 1 {
                        // intra-ligand: bonded distance must be >= 4
                        let dm = dist_matrix[ri * l + rj];
                        let dmv = if dm.is_infinite() { 4.0 } else { dm };
                        if dmv < 4.0 {
                            continue;
                        }
                    }
                    if ai < 5 && aj < 5 && bond_feats[ri * l + rj] == 6 {
                        continue;
                    }

                    // pair parameters, with the hydrogen-bond corrections
                    let c = |s: usize, k: usize, i: usize| t.ljcorr[(s * NTOTAL + k) * 4 + i];
                    let p = |s: usize, k: usize, i: usize| t.ljparams[(s * NTOTAL + k) * 5 + i];
                    let mut r = p(si, ai, 0) + p(sj, aj, 0);
                    if (c(si, ai, 0) && c(sj, aj, 1)) || (c(si, ai, 1) && c(sj, aj, 0)) {
                        r = cfg.lj_hb_dis;
                    }
                    if (c(si, ai, 0) && c(si, ai, 1) && c(sj, aj, 0))
                        || (c(si, ai, 0) && c(sj, aj, 0) && c(sj, aj, 1))
                    {
                        r = cfg.lj_ohdon_dis;
                    }
                    if (c(si, ai, 2) && c(sj, aj, 1)) || (c(si, ai, 1) && c(sj, aj, 2)) {
                        r = cfg.lj_hbond_hdis;
                    }
                    // `torch.sqrt(ljparams_i * ljparams_j + eps)`: the product
                    // and the `+eps` are fp32; only the sqrt is pinned.
                    let mut s_eps =
                        (((p(si, ai, 1) * p(sj, aj, 1) + cfg.eps) as f64).sqrt()) as f32;
                    if c(si, ai, 3) && c(sj, aj, 3) {
                        s_eps = 0.0;
                    }

                    // ljVdV
                    let delta = [
                        get(ri, ai, 0) - get(rj, aj, 0),
                        get(ri, ai, 1) - get(rj, aj, 1),
                        get(ri, ai, 2) - get(rj, aj, 2),
                    ];
                    // `torch.sqrt(torch.sum(torch.square(deltas), -1) + eps)`:
                    // square in fp32, the SUM is pinned (f64 -> fp32), then
                    // `+eps` in fp32, then the pinned sqrt.
                    let mut sq = 0.0f64;
                    for d in 0..3 {
                        sq += (delta[d] * delta[d]) as f64;
                    }
                    let sq32 = sq as f32;
                    let dist = (((sq32 + cfg.eps) as f64).sqrt()) as f32;
                    let linpart = dist < cfg.lj_lin * r;
                    let deff = if linpart { cfg.lj_lin * r } else { dist };
                    let sd = r / deff;
                    let sd2 = sd * sd;
                    let sd6 = sd2 * sd2 * sd2;
                    let sd12 = sd6 * sd6;
                    let mut lje = s_eps * (sd12 - 2.0 * sd6);
                    if linpart {
                        lje += s_eps * (-12.0 * sd12 / deff + 12.0 * sd6 / deff)
                            * (dist - deff);
                    }
                    let dljedd_over_r =
                        s_eps * (-12.0 * sd12 / deff + 12.0 * sd6 / deff) / dist;

                    energy += lje as f64;
                    // `normNviolations=False` divides `dljEdd_i` by natoms; the
                    // index_add then applies alpha = 1/natoms on top.
                    //
                    // The two `index_add_` calls are SEPARATE passes over the
                    // whole pair list — every `idxI` contribution lands before
                    // any `idxJ` one. fp32 accumulation is order-dependent, so
                    // interleaving them (the obvious single-loop form) gives a
                    // different answer for every atom that appears on both
                    // sides of some pair.
                    let g = dljedd_over_r / natoms_f;
                    contribs.push((
                        (ri * a + ai) as u32,
                        (rj * a + aj) as u32,
                        [g * delta[0], g * delta[1], g * delta[2]],
                    ));
                }
            }
        }
    }
    // ATen's `index_add_` accumulates `self += alpha * source` through a
    // `cpu_kernel` that GCC contracts into an FMA, so the product is not rounded
    // before the add. Writing it as two separate operations changes ~5 % of the
    // values by a ULP.
    for &(i, _, src) in &contribs {
        for d in 0..3 {
            let a = &mut dljedx[i as usize * 3 + d];
            *a = alpha.mul_add(src[d], *a);
        }
    }
    for &(_, j, src) in &contribs {
        for d in 0..3 {
            let a = &mut dljedx[j as usize * 3 + d];
            *a = (-alpha).mul_add(src[d], *a);
        }
    }
    LjOut { energy: (energy as f32) / natoms_f, dljedx }
}

/// `natoms` as `calc_lj_grads` computes it, for scaling the captured gradient.
pub fn natoms(seq: &[i64], t: &LjTables, use_h: bool) -> f32 {
    let nat = if use_h { NTOTAL } else { 14 };
    let mut n = 0i64;
    for &s in seq {
        for k in 0..nat {
            if t.aamask[s as usize * NTOTAL + k] {
                n += 1;
            }
        }
    }
    n as f32
}
