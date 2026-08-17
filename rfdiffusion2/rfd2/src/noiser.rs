//! The flow-matching noiser: `rf_diffusion/noisers.py:NormalizingFlow` and the
//! `se3_flow_matching` interpolant beneath it.
//!
//! This is the object rung 4e's `diffuse` and rung 7's sampler share, so it is
//! ported once and used twice.
//!
//! ## The draw order is the specification
//!
//! Measured by `python/gen_noiser.py`, `sample_init` makes exactly nine draws
//! from the torch generator, in this order:
//!
//! ```text
//!  0,1  normal (n_sm, 3)   add_fake_frame_legs   <- diffuse
//!  2    randn  (1, L, 3)   sample_gaussian       <- _corrupt_trans
//!  3    randn  (1, L, 3)   sample_vector         <- igso3.sample
//!  4    rand   (1, L)      sample_angle          <- igso3.sample
//!  5    rand   (L, 2)      atom37_from_rigid     <- diffuse   (psi_pred)
//!  6,7  normal (n_sm, 3)   add_fake_frame_legs   <- add_fake_peptide_frame
//!  8    rand   (L, 2)      atom37_from_rigid     <- idealize_peptide_frames
//! ```
//!
//! Two of those are easy to miss and fatal if missed, because a skipped draw
//! shifts every later one: `atom37_from_rigid` looks purely geometric but draws
//! `psi_pred`, and `add_fake_peptide_frame` runs the whole fake-leg plus
//! idealization sequence a *second* time after the noiser has finished.
//! `tests/parity_noiser_rng.rs` asserts all nine against the reference's own
//! captured generator states.

use crate::chemical_gen::NTOTAL;
use crate::nn::Ctx;
use crate::rng::torch::randn;

/// `aa_model.add_fake_frame_legs`.
///
/// The network's `compute_all_atom` needs N and C coordinates even for ligand
/// rows, which have only one real atom. Upstream copies that atom into all three
/// backbone slots and then jitters the N and C copies by unit Gaussian noise —
/// so the legs carry no structural meaning, but they *do* consume the generator,
/// twice, before anything else in `diffuse`.
///
/// The two draws are separate calls over the ligand rows only, N first then C.
pub fn add_fake_frame_legs(xyz: &[f32], l: usize, is_sm: &[bool], ctx: &mut Ctx) -> Vec<f32> {
    let mut out = xyz.to_vec();
    let sm: Vec<usize> = (0..l).filter(|i| is_sm[*i]).collect();
    // xyz[is_atom, :3] = xyz[is_atom, 1][..., None, :]
    for &i in &sm {
        let ca = [
            xyz[(i * NTOTAL + 1) * 3],
            xyz[(i * NTOTAL + 1) * 3 + 1],
            xyz[(i * NTOTAL + 1) * 3 + 2],
        ];
        for a in 0..3 {
            let o = (i * NTOTAL + a) * 3;
            out[o..o + 3].copy_from_slice(&ca);
        }
    }
    // then N (slot 0) and C (slot 2) are jittered, in that order
    for slot in [0usize, 2] {
        let noise = randn(&mut ctx.rng, sm.len() * 3);
        for (k, &i) in sm.iter().enumerate() {
            let o = (i * NTOTAL + slot) * 3;
            for c in 0..3 {
                out[o + c] += noise[k * 3 + c];
            }
        }
    }
    out
}

/// `openfold.rigid_utils.Rigid.from_3_points` — Algorithm 21 Gram-Schmidt.
///
/// **Not** the same function as `geom::rigid_from_3_points`, despite the name.
/// That one is `rf2aa`'s: it uses `torch.norm`, `torch.einsum` and an
/// ideal-angle correction rotation, all of which are pinned to f64. This one is
/// written out per component to dodge AMP downcasting, so only the two `sqrt`
/// calls are pinned and every multiply, add and divide between them is genuinely
/// fp32. Porting one as the other gives a plausible frame and a wrong one.
///
/// Returns `(rot, trans)` with `rot` row-major `[3, 3]` holding `e0, e1, e2` as
/// **columns**, and `trans = origin`.
fn from_3_points(p_neg_x: [f32; 3], origin: [f32; 3], p_xy: [f32; 3]) -> ([[f32; 3]; 3], [f32; 3]) {
    const EPS: f32 = 1e-8;
    let mut e0 = [
        origin[0] - p_neg_x[0],
        origin[1] - p_neg_x[1],
        origin[2] - p_neg_x[2],
    ];
    let mut e1 = [
        p_xy[0] - origin[0],
        p_xy[1] - origin[1],
        p_xy[2] - origin[2],
    ];
    // `sum(c*c for c in e0)` is a Python left-fold over three fp32 tensors
    let denom = sqrt_pinned((e0[0] * e0[0] + e0[1] * e0[1] + e0[2] * e0[2]) + EPS);
    for c in e0.iter_mut() {
        *c /= denom;
    }
    let dot = e0[0] * e1[0] + e0[1] * e1[1] + e0[2] * e1[2];
    for k in 0..3 {
        e1[k] -= e0[k] * dot;
    }
    let denom = sqrt_pinned((e1[0] * e1[0] + e1[1] * e1[1] + e1[2] * e1[2]) + EPS);
    for c in e1.iter_mut() {
        *c /= denom;
    }
    let e2 = [
        e0[1] * e1[2] - e0[2] * e1[1],
        e0[2] * e1[0] - e0[0] * e1[2],
        e0[0] * e1[1] - e0[1] * e1[0],
    ];
    // `torch.stack([c for tup in zip(e0,e1,e2) for c in tup])` then reshape to
    // (3, 3) lays the three vectors out as COLUMNS.
    let rot = [
        [e0[0], e1[0], e2[0]],
        [e0[1], e1[1], e2[1]],
        [e0[2], e1[2], e2[2]],
    ];
    (rot, origin)
}

/// `torch.sqrt` under pinning: f64 interior, one narrowing.
#[inline]
fn sqrt_pinned(x: f32) -> f32 {
    (x as f64).sqrt() as f32
}

/// `openfold.rigid_utils.rot_matmul` — written out by hand upstream to avoid AMP
/// downcasting, so it is a plain fp32 three-term sum and stays one here.
fn rot_matmul(a: &[[f32; 3]; 3], b: &[[f32; 3]; 3]) -> [[f32; 3]; 3] {
    let mut o = [[0.0f32; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            o[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j];
        }
    }
    o
}

/// `frame_diffusion.data.utils.rigid_frames_from_atom_14`.
///
/// Note the deliberate argument swap: the frame is built from **C, CA, N** —
/// `from_3_points(atom[2], atom[1], atom[0])` — and then turned by pi about the
/// y axis, which is the `diag(-1, 1, -1)` composition. Feeding N, CA, C instead
/// produces a frame that looks right and is rotated by pi.
///
/// Returns `(rots, trans)`, `rots` flattened `[L, 9]` row-major and `trans`
/// `[L, 3]`.
pub fn rigid_frames_from_atom_14(xyz: &[f32], l: usize, n_atoms: usize) -> (Vec<f32>, Vec<f32>) {
    // the pi flip about y
    const FLIP: [[f32; 3]; 3] = [[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]];
    let mut rots = vec![0.0f32; l * 9];
    let mut trans = vec![0.0f32; l * 3];
    for i in 0..l {
        let at = |a: usize| -> [f32; 3] {
            let o = (i * n_atoms + a) * 3;
            [xyz[o], xyz[o + 1], xyz[o + 2]]
        };
        let (r, t) = from_3_points(at(2), at(1), at(0));
        let r = rot_matmul(&r, &FLIP);
        for a in 0..3 {
            for b in 0..3 {
                rots[i * 9 + a * 3 + b] = r[a][b];
            }
        }
        trans[i * 3..i * 3 + 3].copy_from_slice(&t);
    }
    (rots, trans)
}

// ---------------------------------------------------------------------------
// IGSO(3) sampling
// ---------------------------------------------------------------------------

/// Inverse-transform sampler for the IGSO(3) angle distribution.
///
/// The distribution has no closed-form inverse CDF, so upstream integrates the
/// series expansion on a `[n_sigma, n_omega]` grid once, caches it to disk, and
/// samples by bucketing a uniform draw into that table and interpolating.
///
/// **Only one row is reachable on the inference path.** `_corrupt_rotmats_multi_t`
/// calls `igso3.sample(torch.tensor([1.5]), ...)` with the sigma written into the
/// source, and `bucketize(1.5, linspace(0.1, 1.5, 1000))` is 999 — the last row.
/// So the port carries 1 000 angles and 1 000 CDF values, not the 10^6-entry
/// matrix, and [`Igso3::new`] asserts the sigma it was built for.
pub struct Igso3 {
    /// `[n_omega]` — the angle grid
    pub omega: Vec<f32>,
    /// `[n_omega]` — the CDF row for this sigma
    pub cdf: Vec<f32>,
    pub tol: f32,
}

impl Igso3 {
    pub fn new(omega: Vec<f32>, cdf: Vec<f32>) -> Self {
        assert_eq!(omega.len(), cdf.len(), "IGSO3 grid and CDF row must match");
        Igso3 {
            omega,
            cdf,
            tol: 1e-7,
        }
    }

    /// `sample_angle` for a single sigma.
    ///
    /// `idx_stop = sum(cdf < p)` — a count, not a `searchsorted`, and the
    /// difference shows at a `p` that lands exactly on a tabulated CDF value.
    /// `idx_start` is `idx_stop - 1` clamped at 0, so the first bin
    /// degenerates to `lerp(omega[0], omega[0], w)` rather than extrapolating.
    pub fn sample_angle(&self, p: &[f32]) -> Vec<f32> {
        p.iter()
            .map(|&pu| {
                let stop = self.cdf.iter().filter(|c| **c < pu).count();
                let stop = stop.min(self.cdf.len() - 1);
                let start = stop.saturating_sub(1);
                let (c0, c1) = (self.cdf[start], self.cdf[stop]);
                let delta = (c1 - c0).max(self.tol);
                let w = ((pu - c0) / delta).clamp(0.0, 1.0);
                // torch.lerp: start + w * (end - start)
                let (o0, o1) = (self.omega[start], self.omega[stop]);
                o0 + w * (o1 - o0)
            })
            .collect()
    }

    /// Draw IGSO(3) rotation matrices from the torch generator.
    ///
    /// This deliberately owns both random draws rather than accepting sampled
    /// axes and angles.  Upstream's `BaseSampleSO3.sample` first calls
    /// `sample_vector` (`torch.randn(n, 3)`) and only then `sample_angle`
    /// (`torch.rand(n)`).  Keeping that order inside one API prevents callers
    /// from accidentally shifting every later draw in the inference stream.
    pub fn sample(&self, n: usize, ctx: &mut Ctx) -> Vec<f32> {
        let raw_axes = randn(&mut ctx.rng, n * 3);
        let uniforms: Vec<f32> = (0..n).map(|_| ctx.rng.uniform_f32()).collect();
        let angles = self.sample_angle(&uniforms);
        let mut out = vec![0.0f32; n * 9];

        for i in 0..n {
            let a = &raw_axes[i * 3..i * 3 + 3];
            // `torch.norm(..., dim=-1)` is f64-pinned in the reference, then
            // each axis component is divided in ordinary fp32.
            let norm = ((a[0] as f64 * a[0] as f64
                + a[1] as f64 * a[1] as f64
                + a[2] as f64 * a[2] as f64)
                .sqrt()) as f32;
            let rotvec = [
                (a[0] / norm) * angles[i],
                (a[1] / norm) * angles[i],
                (a[2] / norm) * angles[i],
            ];
            let rot = rotvec_to_rotmat(rotvec, self.tol);
            for row in 0..3 {
                for col in 0..3 {
                    out[i * 9 + row * 3 + col] = rot[row][col];
                }
            }
        }
        out
    }
}

/// `so3_utils.rotvec_to_rotmat` — Rodrigues, in the form where the skew matrix
/// already carries the angle.
///
/// ```text
/// exp(K) = I + (sin t / t) K + ((1 - cos t) / t^2) K^2
/// ```
///
/// Three things here are easy to get wrong and all three are silent:
///
/// * `K` holds the **unnormalised** rotation vector, so the coefficients divide
///   by `t` and `t^2`. Normalising the axis and using `sin t` / `1 - cos t`
///   directly is the same mathematics and a different rounding.
/// * `tol` is a **branch threshold**, not an epsilon added to the denominator.
///   Below it the coefficients switch to their second-order Taylor forms, which
///   is what keeps `t = 0` finite.
/// * `K^2` is a `torch.einsum`, hence pinned: f64 accumulation, one narrowing.
///   Everything around it — the coefficient divisions, the scaling, the two
///   additions — is genuinely fp32.
pub fn rotvec_to_rotmat(v: [f32; 3], tol: f32) -> [[f32; 3]; 3] {
    // `torch.norm` is pinned
    let theta = ((v[0] as f64 * v[0] as f64
        + v[1] as f64 * v[1] as f64
        + v[2] as f64 * v[2] as f64)
        .sqrt()) as f32;
    let theta_sq = theta * theta;
    let (sin_coeff, cos_coeff) = if theta.abs() < tol {
        (1.0 - theta_sq / 6.0, 0.5 - theta_sq / 24.0)
    } else {
        let s = (theta as f64).sin() as f32;
        let c = (theta as f64).cos() as f32;
        (s / theta, (1.0 - c) / theta_sq)
    };
    let k = [
        [0.0f32, -v[2], v[1]],
        [v[2], 0.0, -v[0]],
        [-v[1], v[0], 0.0],
    ];
    // K @ K, in f64 with a single narrowing, as the pinned einsum does
    let mut kk = [[0.0f32; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            let mut acc = 0.0f64;
            for m in 0..3 {
                acc += k[i][m] as f64 * k[m][j] as f64;
            }
            kk[i][j] = acc as f32;
        }
    }
    let mut o = [[0.0f32; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            let id = if i == j { 1.0f32 } else { 0.0 };
            o[i][j] = (id + sin_coeff * k[i][j]) + cos_coeff * kk[i][j];
        }
    }
    o
}

// ---------------------------------------------------------------------------
// translation corruption
// ---------------------------------------------------------------------------

/// Nanometres to angstroms (`se3_flow_matching.data.utils.NM_TO_ANG_SCALE`).
pub const NM_TO_ANG: f32 = 10.0;

/// `interpolant.sample_gaussian`, including its optional centering convention.
///
/// RFdiffusion2 samples translations in nanometres and converts them to
/// Angstroms before optimal-transport alignment.  The released RFD_173
/// checkpoint has `center_noise_sample = false`, but the centered branch is
/// kept because it is a user-visible diffuser setting needed by rung 8.
pub fn sample_gaussian(num_batch: usize, num_res: usize, center: bool, ctx: &mut Ctx) -> Vec<f32> {
    let mut noise = randn(&mut ctx.rng, num_batch * num_res * 3);
    if center {
        for b in 0..num_batch {
            for c in 0..3 {
                // `torch.mean(dim=-2, keepdims=True)` is pinned: accumulate
                // in f64, narrow once, then subtract in fp32.
                let mut sum = 0.0f64;
                for i in 0..num_res {
                    sum += noise[(b * num_res + i) * 3 + c] as f64;
                }
                let mean = (sum / num_res as f64) as f32;
                for i in 0..num_res {
                    let j = (b * num_res + i) * 3 + c;
                    noise[j] -= mean;
                }
            }
        }
    }
    noise
}

/// Draw the translation prior in the units consumed by the network.
pub fn sample_translation_prior(
    num_batch: usize,
    num_res: usize,
    center: bool,
    ctx: &mut Ctx,
) -> Vec<f32> {
    sample_gaussian(num_batch, num_res, center, ctx)
        .into_iter()
        .map(|x| x * NM_TO_ANG)
        .collect()
}

/// Jacobi eigen-decomposition of a symmetric 3x3 matrix, in f64.
///
/// Used to build the Kabsch rotation without an SVD routine: for
/// `C = U S V^T`, `C^T C = V S^2 V^T`, so `V` is the eigenvector matrix of the
/// symmetric `C^T C` and `U = C V S^-1`. Jacobi is chosen because it converges
/// to full f64 accuracy on 3x3 matrices in a handful of sweeps, which is what
/// makes the fp32 answer independent of the algorithm.
fn jacobi_eigh(a_in: [[f64; 3]; 3]) -> ([f64; 3], [[f64; 3]; 3]) {
    let mut a = a_in;
    let mut v = [[0.0f64; 3]; 3];
    for i in 0..3 {
        v[i][i] = 1.0;
    }
    for _ in 0..64 {
        // largest off-diagonal
        let (mut p, mut q, mut off) = (0usize, 1usize, 0.0f64);
        for i in 0..3 {
            for j in (i + 1)..3 {
                if a[i][j].abs() > off {
                    off = a[i][j].abs();
                    p = i;
                    q = j;
                }
            }
        }
        if off < 1e-300 {
            break;
        }
        let theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q]);
        let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
        let c = 1.0 / (t * t + 1.0).sqrt();
        let s = t * c;
        let mut b = a;
        for k in 0..3 {
            b[k][p] = c * a[k][p] - s * a[k][q];
            b[k][q] = s * a[k][p] + c * a[k][q];
        }
        let mut a2 = b;
        for k in 0..3 {
            a2[p][k] = c * b[p][k] - s * b[q][k];
            a2[q][k] = s * b[p][k] + c * b[q][k];
        }
        a2[p][q] = 0.0;
        a2[q][p] = 0.0;
        a = a2;
        let mut v2 = v;
        for k in 0..3 {
            v2[k][p] = c * v[k][p] - s * v[k][q];
            v2[k][q] = s * v[k][p] + c * v[k][q];
        }
        v = v2;
    }
    ([a[0][0], a[1][1], a[2][2]], v)
}

#[inline]
fn det3(a: &[[f64; 3]; 3]) -> f64 {
    a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
        - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
        + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0])
}

/// Align one `[N,3]` point cloud to another with the same Kabsch convention as
/// `se3_flow_matching.data.utils.batch_align_structures`.
///
/// The pinned reference widens the reductions and `torch.linalg.svd` input to
/// f64, then narrows its returned factors to fp32 before forming the rotation.
/// We compute the tiny SVD through the eigendecomposition of `C^T C`; this is
/// deterministic and avoids introducing a platform LAPACK dependency.
pub fn kabsch_align(points: &[f32], reference: &[f32]) -> Vec<f32> {
    assert_eq!(points.len(), reference.len());
    assert_eq!(points.len() % 3, 0);
    let n = points.len() / 3;
    assert!(n >= 3, "Kabsch alignment needs at least three points");

    let mut pm = [0.0f32; 3];
    let mut qm = [0.0f32; 3];
    for c in 0..3 {
        // `center_zero` calls Tensor.mean, which the pinning shim widens.
        let mut ps = 0.0f64;
        let mut qs = 0.0f64;
        for i in 0..n {
            ps += points[i * 3 + c] as f64;
            qs += reference[i * 3 + c] as f64;
        }
        pm[c] = (ps / n as f64) as f32;
        qm[c] = (qs / n as f64) as f32;
    }

    let mut p = vec![[0.0f32; 3]; n];
    let mut q = vec![[0.0f32; 3]; n];
    for i in 0..n {
        for c in 0..3 {
            p[i][c] = points[i * 3 + c] - pm[c];
            q[i][c] = reference[i * 3 + c] - qm[c];
        }
    }

    // C = Q^T P. `scatter_reduce_` is not reached by the Python pinning shim;
    // at this shape ATen visits rows in order and accumulates in fp32.
    let mut cov_f = [[0.0f32; 3]; 3];
    for a in 0..3 {
        for b in 0..3 {
            for i in 0..n {
                cov_f[a][b] += q[i][a] * p[i][b];
            }
        }
    }
    let mut cov = [[0.0f64; 3]; 3];
    for a in 0..3 {
        for b in 0..3 {
            cov[a][b] = cov_f[a][b] as f64;
        }
    }
    let mut ctc = [[0.0f64; 3]; 3];
    for a in 0..3 {
        for b in 0..3 {
            for k in 0..3 {
                ctc[a][b] += cov[k][a] * cov[k][b];
            }
        }
    }
    let (eval, evec) = jacobi_eigh(ctc);
    let mut order = [0usize, 1, 2];
    order.sort_by(|&a, &b| eval[b].total_cmp(&eval[a]));
    let mut v = [[0.0f64; 3]; 3];
    let mut u = [[0.0f64; 3]; 3];
    for col in 0..3 {
        let src = order[col];
        let s = eval[src].max(0.0).sqrt();
        for row in 0..3 {
            v[row][col] = evec[row][src];
        }
        for row in 0..3 {
            for k in 0..3 {
                u[row][col] += cov[row][k] * v[k][col] / s;
            }
        }
    }
    // Eigenvectors have arbitrary signs, but V U^T is sign-invariant. Apply
    // the reference's right-handed correction to the last singular vector.
    let mut vu_t = [[0.0f64; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            for k in 0..3 {
                vu_t[i][j] += v[i][k] * u[j][k];
            }
        }
    }
    let sign = if det3(&vu_t) < 0.0 { -1.0 } else { 1.0 };
    // The patched `torch.linalg.svd` returns factors narrowed to fp32, and the
    // following bmm widens those stored values again.
    let mut vf = [[0.0f32; 3]; 3];
    let mut uf = [[0.0f32; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            vf[i][j] = v[i][j] as f32;
            uf[i][j] = u[i][j] as f32;
        }
    }
    let mut rot = [[0.0f32; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            let x = vf[i][0] as f64 * uf[j][0] as f64
                + vf[i][1] as f64 * uf[j][1] as f64
                + sign * vf[i][2] as f64 * uf[j][2] as f64;
            rot[i][j] = x as f32;
        }
    }
    let mut out = vec![0.0f32; n * 3];
    for i in 0..n {
        for c in 0..3 {
            let mut acc = 0.0f64;
            for k in 0..3 {
                // `batch_align_structures(..., mask=...)` estimates the
                // rotation from centered masked points, but deliberately
                // applies it to the original, uncentered `pos_1` tensor.
                acc += points[i * 3 + k] as f64 * rot[k][c] as f64;
            }
            out[i * 3 + c] = acc as f32;
        }
    }
    out
}

/// Flow-matching translation corruption for one structure.
///
/// `t=0` is the aligned Gaussian prior and `t=1` is the input structure. The
/// live `forward_marginal` passes `1 - inference_t`, exactly as upstream does.
pub fn corrupt_trans(trans_1: &[f32], t: f32, center_noise: bool, ctx: &mut Ctx) -> Vec<f32> {
    assert_eq!(trans_1.len() % 3, 0);
    let n = trans_1.len() / 3;
    let prior = sample_translation_prior(1, n, center_noise, ctx);
    let aligned = kabsch_align(&prior, trans_1);
    aligned
        .iter()
        .zip(trans_1)
        .map(|(&x0, &x1)| (1.0 - t) * x0 + t * x1)
        .collect()
}

#[inline]
fn matmul3_pinned(a: &[f32], b: &[f32], out: &mut [f32]) {
    for i in 0..3 {
        for j in 0..3 {
            let mut acc = 0.0f64;
            for k in 0..3 {
                acc += a[i * 3 + k] as f64 * b[k * 3 + j] as f64;
            }
            out[i * 3 + j] = acc as f32;
        }
    }
}

/// Stable logarithmic map from one row-major SO(3) matrix to a rotation vector.
pub fn rotmat_to_rotvec(r: &[f32]) -> [f32; 3] {
    assert_eq!(r.len(), 9);
    let skew = [r[7] - r[5], r[2] - r[6], r[3] - r[1]];
    let sin = ((skew[0] as f64 * skew[0] as f64
        + skew[1] as f64 * skew[1] as f64
        + skew[2] as f64 * skew[2] as f64)
        .sqrt() as f32)
        / 2.0;
    let trace = (r[0] as f64 + r[4] as f64 + r[8] as f64) as f32;
    let cos = (trace - 1.0) / 2.0;
    let angle = (sin as f64).atan2(cos as f64) as f32;
    let zero = angle.abs() <= 1e-8;
    let near_pi = (angle - std::f32::consts::PI).abs() <= 1e-2 + 1e-5 * std::f32::consts::PI;

    if near_pi {
        let mut outer = [[0.0f32; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                let id = if i == j { 1.0 } else { 0.0 };
                outer[i][j] = (id + r[i * 3 + j]) / 2.0;
            }
            outer[i][i] = outer[i][i].max(0.0);
        }
        let mut best = 0usize;
        let mut best_norm = f32::NEG_INFINITY;
        for i in 0..3 {
            let norm = (outer[i][0] as f64 * outer[i][0] as f64
                + outer[i][1] as f64 * outer[i][1] as f64
                + outer[i][2] as f64 * outer[i][2] as f64)
                .sqrt() as f32;
            if norm > best_norm {
                best = i;
                best_norm = norm;
            }
        }
        let sign = |x: f32| {
            if x > 0.0 {
                1.0
            } else if x < 0.0 {
                -1.0
            } else {
                0.0
            }
        };
        return [
            ((outer[0][0] as f64).sqrt() as f32) * angle * sign(outer[best][0]),
            ((outer[1][1] as f64).sqrt() as f32) * angle * sign(outer[best][1]),
            ((outer[2][2] as f64).sqrt() as f32) * angle * sign(outer[best][2]),
        ];
    }

    let prefactor = if zero {
        0.5 / (1.0 - angle * angle / 6.0)
    } else {
        angle / (2.0 * sin)
    };
    [
        skew[0] * prefactor,
        skew[1] * prefactor,
        skew[2] * prefactor,
    ]
}

/// `base * Exp(t * Log(base^T * target))` for batches of 3x3 matrices.
pub fn geodesic_t(t: f32, target: &[f32], base: &[f32]) -> Vec<f32> {
    assert_eq!(target.len(), base.len());
    assert_eq!(target.len() % 9, 0);
    let mut out = vec![0.0f32; target.len()];
    for x in 0..target.len() / 9 {
        let a = &target[x * 9..x * 9 + 9];
        let b = &base[x * 9..x * 9 + 9];
        let mut bt = [0.0f32; 9];
        for i in 0..3 {
            for j in 0..3 {
                bt[i * 3 + j] = b[j * 3 + i];
            }
        }
        let mut rel = [0.0f32; 9];
        matmul3_pinned(&bt, a, &mut rel);
        let v = rotmat_to_rotvec(&rel);
        let step = rotvec_to_rotmat([t * v[0], t * v[1], t * v[2]], 1e-7);
        let sf = [
            step[0][0], step[0][1], step[0][2], step[1][0], step[1][1], step[1][2], step[2][0],
            step[2][1], step[2][2],
        ];
        matmul3_pinned(b, &sf, &mut out[x * 9..x * 9 + 9]);
    }
    out
}

/// Flow-matching rotation corruption for one structure.
///
/// The released initialization path calls this at `t=0`, where the geodesic
/// result is exactly its noisy base rotation. Nonzero geodesics are rejected
/// until the canonical SO(3) logarithm is implemented.
pub fn corrupt_rots(rotmats_1: &[f32], t: f32, igso3: &Igso3, ctx: &mut Ctx) -> Vec<f32> {
    assert_eq!(rotmats_1.len() % 9, 0);
    let n = rotmats_1.len() / 9;
    let noise = igso3.sample(n, ctx);
    let mut base = vec![0.0f32; rotmats_1.len()];
    for i in 0..n {
        matmul3_pinned(
            &rotmats_1[i * 9..i * 9 + 9],
            &noise[i * 9..i * 9 + 9],
            &mut base[i * 9..i * 9 + 9],
        );
    }
    geodesic_t(t, rotmats_1, &base)
}

#[derive(Clone, Debug)]
pub struct Rigids {
    /// Row-major `[L,3,3]` rotation matrices.
    pub rots: Vec<f32>,
    /// `[L,3]` translations in Angstroms.
    pub trans: Vec<f32>,
}

impl Rigids {
    pub fn len(&self) -> usize {
        self.trans.len() / 3
    }
}

/// `NormalizingFlow.forward_marginal` for one structure.
///
/// Its public time runs from clean (`t=0`) to prior (`t=1`), while the
/// underlying interpolant runs in the opposite direction, hence `ti = 1-t`.
pub fn forward_marginal(
    rigids_0: &Rigids,
    t: f32,
    diffuse_mask: &[bool],
    center_noise: bool,
    igso3: &Igso3,
    ctx: &mut Ctx,
) -> Rigids {
    let l = rigids_0.len();
    assert_eq!(rigids_0.rots.len(), l * 9);
    assert_eq!(diffuse_mask.len(), l);
    let selected: Vec<usize> = (0..l).filter(|&i| diffuse_mask[i]).collect();
    assert!(
        selected.len() >= 3,
        "forward_marginal needs at least three diffused rows"
    );
    let mut trans_1 = Vec::with_capacity(selected.len() * 3);
    let mut rots_1 = Vec::with_capacity(selected.len() * 9);
    for &i in &selected {
        trans_1.extend_from_slice(&rigids_0.trans[i * 3..i * 3 + 3]);
        rots_1.extend_from_slice(&rigids_0.rots[i * 9..i * 9 + 9]);
    }
    let ti = 1.0 - t;
    let trans_t = corrupt_trans(&trans_1, ti, center_noise, ctx);
    let rots_t = corrupt_rots(&rots_1, ti, igso3, ctx);
    let mut out = rigids_0.clone();
    for (j, &i) in selected.iter().enumerate() {
        out.trans[i * 3..i * 3 + 3].copy_from_slice(&trans_t[j * 3..j * 3 + 3]);
        out.rots[i * 9..i * 9 + 9].copy_from_slice(&rots_t[j * 9..j * 9 + 9]);
    }
    out
}
