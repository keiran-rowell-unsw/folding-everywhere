//! Rung 5 — geometry primitives shared by the embeddings, every IterBlock and
//! the SE(3) refiner.
//!
//! Three of these are places where a "mathematically equivalent" formula is the
//! wrong answer, so each carries the reason it is written the way it is.

use crate::chemical_gen::{NNAPROTAAS, NPROTAAS};
use crate::ops::elem::exp_scalar;
use crate::tensor::Tensor;

// ---------------------------------------------------------------------------
// token predicates (`rf2aa/util.py`)
// ---------------------------------------------------------------------------

#[inline]
pub fn is_atom(tok: i64) -> bool {
    tok > NNAPROTAAS as i64
}

#[inline]
pub fn is_protein(tok: i64) -> bool {
    (tok as usize) < NPROTAAS
}

#[inline]
pub fn is_nucleic(tok: i64) -> bool {
    tok >= NPROTAAS as i64 && tok <= NNAPROTAAS as i64
}

// ---------------------------------------------------------------------------
// cdist
// ---------------------------------------------------------------------------

/// `torch.cdist(x, x)` for 3-D points, **reproducing ATen's matmul expansion**.
///
/// This is not `sqrt(sum((a-b)^2))`, and the difference is visible in fp32.
/// `cdist_impl` picks `_euclidean_dist` whenever `p == 2` and either input has
/// more than 25 rows (L = 71 here, so always), which computes
///
/// ```text
///   d^2 = (-2·x_i)·x_j  +  |x_i|^2 · 1  +  1 · |x_j|^2
/// ```
///
/// as a single dot product over five terms, then `clamp_min(0)` and `sqrt`.
/// On the diagonal the three terms cancel catastrophically, so `d(i,i)` comes
/// out as a few ULP of |x|^2 rather than exactly 0 — and after the sqrt that is
/// ~1e-6, not ~1e-16. Every downstream `rbf` bin sees it. Computing the honest
/// difference formula instead gives exactly 0 there and silently shifts the
/// first RBF channel of every self-pair.
///
/// Under pinning the whole expansion runs in f64 (`torch.cdist` is patched), so
/// the five-term accumulation order does not matter; only the *algebra* does.
pub fn cdist_self(x: &[f32], n: usize) -> Vec<f32> {
    debug_assert_eq!(x.len(), n * 3);
    // x1_norm = x.pow(2).sum(-1), in f64, in k order — as ATen does.
    let mut norm = vec![0.0f64; n];
    for i in 0..n {
        let mut s = 0.0f64;
        for k in 0..3 {
            let v = x[i * 3 + k] as f64;
            s += v * v;
        }
        norm[i] = s;
    }
    let mut out = vec![0.0f32; n * n];
    for i in 0..n {
        for j in 0..n {
            let mut acc = 0.0f64;
            for k in 0..3 {
                acc += (-2.0 * x[i * 3 + k] as f64) * (x[j * 3 + k] as f64);
            }
            acc += norm[i];
            acc += norm[j];
            if acc < 0.0 {
                acc = 0.0;
            }
            out[i * n + j] = acc.sqrt() as f32;
        }
    }
    out
}

/// `torch.cdist(a, b)` for two different 3-D point sets.
pub fn cdist(a: &[f32], na: usize, b: &[f32], nb: usize) -> Vec<f32> {
    let norm = |p: &[f32], i: usize| {
        let mut s = 0.0f64;
        for k in 0..3 {
            let v = p[i * 3 + k] as f64;
            s += v * v;
        }
        s
    };
    let n1: Vec<f64> = (0..na).map(|i| norm(a, i)).collect();
    let n2: Vec<f64> = (0..nb).map(|j| norm(b, j)).collect();
    let mut out = vec![0.0f32; na * nb];
    for i in 0..na {
        for j in 0..nb {
            let mut acc = 0.0f64;
            for k in 0..3 {
                acc += (-2.0 * a[i * 3 + k] as f64) * (b[j * 3 + k] as f64);
            }
            acc += n1[i];
            acc += n2[j];
            if acc < 0.0 {
                acc = 0.0;
            }
            out[i * nb + j] = acc.sqrt() as f32;
        }
    }
    out
}

// ---------------------------------------------------------------------------
// rbf
// ---------------------------------------------------------------------------

pub const D_COUNT: usize = 64;
const D_SIGMA: f32 = 0.5;

/// `util_module.rbf`: 64 Gaussian bins at centres `0, 0.5, … 31.5`.
///
/// The centres come from `torch.linspace(0, 31.5, 64)`, which ATen evaluates as
/// `start + i*step` with `step = (end-start)/(steps-1)` in the output dtype;
/// here that is exactly `0.5`, so every centre is exactly representable and the
/// linspace can be written as `0.5 * i` without drift.
///
/// Only the `exp` is pinned (it is `torch.exp`); the subtraction, division and
/// squaring are fp32 elementwise ops in the reference and stay fp32 here.
pub fn rbf_into(d: &[f32], out: &mut [f32]) {
    debug_assert_eq!(out.len(), d.len() * D_COUNT);
    for (i, &dv) in d.iter().enumerate() {
        for b in 0..D_COUNT {
            let mu = 0.5f32 * b as f32;
            let t = (dv - mu) / D_SIGMA;
            out[i * D_COUNT + b] = exp_scalar(-(t * t));
        }
    }
}

pub fn rbf(d: &[f32], shape_prefix: &[usize]) -> Tensor {
    let mut out = vec![0.0f32; d.len() * D_COUNT];
    rbf_into(d, &mut out);
    let mut shape = shape_prefix.to_vec();
    shape.push(D_COUNT);
    Tensor::new(out, shape)
}

/// `rbf(cdist(CA, CA))` — the pair distance feature used everywhere.
pub fn rbf_ca(ca: &[f32], l: usize) -> Tensor {
    let d = cdist_self(ca, l);
    rbf(&d, &[l, l])
}

// ---------------------------------------------------------------------------
// bond-graph positional features (`util_module.get_res_atom_dist`)
// ---------------------------------------------------------------------------

pub struct ResAtomDist {
    pub res: Vec<i64>,  // [L, L]
    pub atom: Vec<i64>, // [L, L]
}

/// `get_res_atom_dist` — residue-index separation and bond-count separation for
/// a protein / small-molecule complex.
///
/// `dist_matrix` arrives with `+inf` on unreachable pairs (rung 4b pinned that);
/// `nan_to_num(posinf=maxpos_atom)` is what turns those into the saturating
/// token, and it happens **before** the `.long()` cast, so an unreachable pair
/// becomes `maxpos_atom` and not an overflowed integer.
pub fn res_atom_dist(
    idx: &[i64],
    bond_feats: &[i64],
    dist_matrix: &[f32],
    sm_mask: &[bool],
    minpos_res: i64,
    maxpos_res: i64,
    maxpos_atom: i64,
) -> ResAtomDist {
    let l = idx.len();
    let at = |m: &[i64], i: usize, j: usize| m[i * l + j];

    // intra-protein: clamped residue-index difference
    let mut res_dist_prot = vec![0i64; l * l];
    for i in 0..l {
        for j in 0..l {
            let s = idx[j] - idx[i];
            res_dist_prot[i * l + j] = s.clamp(minpos_res, maxpos_res);
        }
    }
    // intra-ligand: bond distance, infinities saturated
    let mut atom_dist_sm = vec![0i64; l * l];
    for i in 0..l * l {
        let v = dist_matrix[i];
        atom_dist_sm[i] = if v.is_infinite() && v.is_sign_positive() {
            maxpos_atom
        } else if v.is_nan() {
            0
        } else {
            v as i64
        };
    }

    // the residue<->atom covalent links (bond type 6)
    let mut i_sm: Vec<usize> = Vec::new();
    let mut i_prot: Vec<usize> = Vec::new();
    for i in 0..l {
        for j in 0..l {
            if at(bond_feats, i, j) == 6 && sm_mask[i] {
                i_sm.push(i);
                i_prot.push(j);
            }
        }
    }

    let mut res_dist_inter = vec![maxpos_res; l * l];
    let mut atom_dist_inter = vec![maxpos_atom; l * l];
    if !i_prot.is_empty() {
        // for each ligand atom, the protein residue reached through the nearest
        // linking atom; `argmin` ties go to the FIRST index, as torch does.
        let mut closest_prot_res = vec![0usize; l];
        for i in 0..l {
            if !sm_mask[i] {
                continue;
            }
            let mut best = i64::MAX;
            let mut arg = 0usize;
            for (t, &s) in i_sm.iter().enumerate() {
                let v = atom_dist_sm[i * l + s];
                if v < best {
                    best = v;
                    arg = t;
                }
            }
            closest_prot_res[i] = i_prot[arg];
        }
        for i in 0..l {
            if sm_mask[i] {
                let r = closest_prot_res[i];
                for j in 0..l {
                    res_dist_inter[i * l + j] = res_dist_prot[r * l + j];
                }
            }
        }
        for j in 0..l {
            if sm_mask[j] {
                let r = closest_prot_res[j];
                for i in 0..l {
                    res_dist_inter[i * l + j] = res_dist_prot[i * l + r];
                }
            }
        }

        let mut closest_atom = vec![0usize; l];
        for i in 0..l {
            if sm_mask[i] {
                continue;
            }
            let mut best = i64::MAX;
            let mut arg = 0usize;
            for (t, &p) in i_prot.iter().enumerate() {
                let v = res_dist_prot[i * l + p].abs();
                if v < best {
                    best = v;
                    arg = t;
                }
            }
            closest_atom[i] = i_sm[arg];
        }
        for i in 0..l {
            if !sm_mask[i] {
                let a = closest_atom[i];
                for j in 0..l {
                    atom_dist_inter[i * l + j] = atom_dist_sm[a * l + j] + 1;
                }
            }
        }
        for j in 0..l {
            if !sm_mask[j] {
                let a = closest_atom[j];
                for i in 0..l {
                    atom_dist_inter[i * l + j] = atom_dist_sm[i * l + a] + 1;
                }
            }
        }
    }

    let mut res = vec![0i64; l * l];
    let mut atom = vec![0i64; l * l];
    for i in 0..l {
        for j in 0..l {
            let k = i * l + j;
            let both_sm = sm_mask[i] && sm_mask[j];
            let both_prot = !sm_mask[i] && !sm_mask[j];
            if both_prot {
                res[k] = res_dist_prot[k];
                atom[k] = maxpos_atom + 1;
            } else if both_sm {
                res[k] = maxpos_res + 1;
                atom[k] = atom_dist_sm[k];
            } else {
                res[k] = res_dist_inter[k];
                atom[k] = atom_dist_inter[k];
            }
        }
    }
    ResAtomDist { res, atom }
}

/// `util_module.get_seqsep_protein_sm` — the single extra edge channel handed to
/// `Str2Str.embed_edge`.
pub fn seqsep_protein_sm(
    idx: &[i64],
    bond_feats: &[i64],
    dist_matrix: &[f32],
    sm_mask: &[bool],
) -> Vec<f32> {
    let l = idx.len();
    let d = res_atom_dist(idx, bond_feats, dist_matrix, sm_mask, -32, 32, 8);
    let mut out = vec![0.0f32; l * l];
    for i in 0..l {
        for j in 0..l {
            let k = i * l + j;
            let mut rd = d.res[k];
            let mut ad = d.atom[k];
            if rd > 1 || rd < -1 {
                rd = 0;
            }
            if ad > 1 {
                ad = 0;
            }
            let both_sm = sm_mask[i] && sm_mask[j];
            let both_prot = !sm_mask[i] && !sm_mask[j];
            out[k] = if both_sm {
                ad as f32
            } else if both_prot {
                rd as f32
            } else {
                // inter: the indicator that this pair is the covalent link
                if bond_feats[k] == 6 {
                    1.0
                } else {
                    0.0
                }
            };
        }
    }
    out
}

/// `torch.bucketize(x, boundaries)` with right=False: the number of boundaries
/// strictly less than `x`… precisely, the index of the first boundary `>= x`.
#[inline]
pub fn bucketize(x: i64, lo: i64, hi: i64) -> usize {
    // boundaries are the contiguous integers lo..=hi
    if x <= lo {
        0
    } else if x > hi {
        (hi - lo + 1) as usize
    } else {
        (x - lo) as usize
    }
}

// ---------------------------------------------------------------------------
// frames
// ---------------------------------------------------------------------------

const COSTGT_PROT: f32 = -0.3616;
const RIGID_EPS: f32 = 1e-4;

/// `torch.norm(v, dim=-1)` under pinning: reduce in f64, **narrow to f32 once**.
///
/// The narrowing is the point. Under `python/pinned.py` each *patched* op takes
/// fp32 in and gives fp32 back — only its interior is double. So a port that
/// carries f64 through a whole formula is not more accurate, it is a different
/// formula: every unpatched elementwise step in between (here the `+eps` and the
/// division) is genuinely fp32 in the reference. This file therefore rounds at
/// exactly the op boundaries the reference rounds at, and nowhere else.
#[inline]
fn norm3(v: [f32; 3]) -> f32 {
    let (a, b, c) = (v[0] as f64, v[1] as f64, v[2] as f64);
    (a * a + b * b + c * c).sqrt() as f32
}

#[inline]
fn dot3_pinned(a: [f32; 3], b: [f32; 3]) -> f32 {
    // `torch.einsum('...li,...li->...l', a, b)` / `torch.sum(a*b, -1)`.
    // NOTE `torch.sum(e1*v2, -1)` multiplies in fp32 first, then reduces in f64;
    // `einsum` does the whole contraction in f64. Both appear in this function
    // and they are not the same — `mul_then_sum` is the `torch.sum` shape.
    let mut s = 0.0f64;
    for k in 0..3 {
        s += a[k] as f64 * b[k] as f64;
    }
    s as f32
}

#[inline]
fn mul_then_sum3(a: [f32; 3], b: [f32; 3]) -> f32 {
    let mut s = 0.0f64;
    for k in 0..3 {
        s += (a[k] * b[k]) as f64; // fp32 product, f64 accumulation
    }
    s as f32
}

#[inline]
fn cross_pinned(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    let f = |x: f32| x as f64;
    [
        (f(a[1]) * f(b[2]) - f(a[2]) * f(b[1])) as f32,
        (f(a[2]) * f(b[0]) - f(a[0]) * f(b[2])) as f32,
        (f(a[0]) * f(b[1]) - f(a[1]) * f(b[0])) as f32,
    ]
}

#[inline]
fn sqrt_pinned(x: f32) -> f32 {
    (x as f64).sqrt() as f32
}

/// The f64 norm the *backward* of a patched `torch.norm` sees.
///
/// The forward narrows to fp32 on the way out, but `norm_backward` runs inside
/// the promoted subgraph and is handed the saved **f64** result, so using the
/// narrowed one is off by up to an fp32 ULP in the denominator.
#[inline]
fn norm3_f64(v: [f32; 3]) -> f64 {
    let (a, b, c) = (v[0] as f64, v[1] as f64, v[2] as f64);
    (a * a + b * b + c * c).sqrt()
}

/// `sqrt`'s derivative under pinning: `grad / (2·result)` evaluated in f64
/// against the **unrounded** f64 root, narrowed once.
#[inline]
fn sqrt_bwd(g: f32, arg: f32) -> f32 {
    ((g as f64) / (2.0 * (arg as f64).sqrt())) as f32
}

/// `rf2aa/util.py:rigid_from_3_points` — Gram-Schmidt frame from N, CA, C with
/// the ideal-angle correction rotation `Rp` applied on the right.
///
/// Returns `R` flattened row-major `[L, 3, 3]` and `T = Ca`.
pub fn rigid_from_3_points(
    n: &[f32],
    ca: &[f32],
    c: &[f32],
    l: usize,
    is_na: &[bool],
    costgt_na: f32,
) -> (Vec<f32>, Vec<f32>) {
    let mut rout = vec![0.0f32; l * 9];
    let tout = ca.to_vec();
    for i in 0..l {
        let g = |p: &[f32], k: usize| p[i * 3 + k];
        let v1 = [g(c, 0) - g(ca, 0), g(c, 1) - g(ca, 1), g(c, 2) - g(ca, 2)];
        let mut v2 = [g(n, 0) - g(ca, 0), g(n, 1) - g(ca, 1), g(n, 2) - g(ca, 2)];

        let d1 = norm3(v1) + RIGID_EPS;
        let e1 = [v1[0] / d1, v1[1] / d1, v1[2] / d1];

        let proj = dot3_pinned(e1, v2);
        let u2 = [v2[0] - proj * e1[0], v2[1] - proj * e1[1], v2[2] - proj * e1[2]];
        let d2 = norm3(u2) + RIGID_EPS;
        let e2 = [u2[0] / d2, u2[1] / d2, u2[2] / d2];
        let e3 = cross_pinned(e1, e2);

        // the reference rebinds `v2` to its normalised self before `cosref`
        let d3 = norm3(v2) + RIGID_EPS;
        v2 = [v2[0] / d3, v2[1] / d3, v2[2] / d3];
        let cosref = mul_then_sum3(e1, v2);

        let costgt = if is_na[i] { costgt_na } else { COSTGT_PROT };
        let inner = (1.0 - cosref * cosref) * (1.0 - costgt * costgt) + RIGID_EPS;
        let cos2del = (cosref * costgt + sqrt_pinned(inner)).clamp(-1.0, 1.0);
        let cosdel = sqrt_pinned(0.5 * (1.0 + cos2del) + RIGID_EPS);
        let sgn = {
            let d = costgt - cosref;
            if d > 0.0 {
                1.0f32
            } else if d < 0.0 {
                -1.0f32
            } else {
                0.0f32
            }
        };
        let sindel = sgn * sqrt_pinned(1.0 - 0.5 * (1.0 + cos2del) + RIGID_EPS);

        // R has e1/e2/e3 as COLUMNS, then R @ Rp (an einsum, so pinned f64)
        let rcol = [[e1[0], e2[0], e3[0]], [e1[1], e2[1], e3[1]], [e1[2], e2[2], e3[2]]];
        let rp = [[cosdel, -sindel, 0.0f32], [sindel, cosdel, 0.0], [0.0, 0.0, 1.0]];
        for r in 0..3 {
            for cc in 0..3 {
                let mut s = 0.0f64;
                for k in 0..3 {
                    s += rcol[r][k] as f64 * rp[k][cc] as f64;
                }
                rout[i * 9 + r * 3 + cc] = s as f32;
            }
        }
    }
    (rout, tout)
}

/// Reverse pass of `rigid_from_3_points` for one residue: given `dL/dR`
/// (3x3, row-major) and the gradient `dt` that `RTF0`'s translation column
/// already put into `CA`'s buffer, return `(dL/dN, dL/dCA, dL/dC)`.
///
/// `dt` is a parameter rather than something the caller adds afterwards because
/// `CA`'s three terms land in a fixed order — the `RTF0[:,:,:3,3] = Ts`
/// assignment is created last so it arrives first, then `-dL/dv2` (`v2 = N-Ca`),
/// then `-dL/dv1` — and each `+=` rounds. Summing the two differences together
/// and adding `dt` to the pair disagreed on 19 of 213 values.
///
/// Written out rather than autodiffed, and mixed-precision for the same reason
/// as `crate::chiral`: the patched ops (`norm`, `cross`, `sum`, `sqrt`, the
/// `einsum` at the end) have their derivatives evaluated inside the promoted
/// subgraph, i.e. in f64 with an fp32 rounding at each boundary, while the
/// divisions and `+eps` between them are fp32.
pub fn rigid_from_3_points_bwd(
    n: &[f32; 3],
    ca: &[f32; 3],
    c: &[f32; 3],
    is_na: bool,
    costgt_na: f32,
    dr: &[[f32; 3]; 3],
    dt: &[f32; 3],
) -> ([f32; 3], [f32; 3], [f32; 3]) {
    let (dn, dca, dc, _) = rigid_from_3_points_bwd_traced(n, ca, c, is_na, costgt_na, dr, dt);
    (dn, dca, dc)
}

/// Every intermediate gradient inside [`rigid_from_3_points_bwd`], named for the
/// reference tensor it corresponds to, so `tests/debug_aa_bwd.rs` can bisect it.
#[derive(Default, Clone, Copy)]
pub struct RigidTrace {
    pub d_rc: [[f32; 3]; 3],
    pub d_rp: [[f32; 3]; 3],
    pub d_cosdel: f32,
    pub d_sindel: f32,
    pub d_cos2del: f32,
    pub d_cosref: f32,
    pub d_e1: [f32; 3],
    pub d_e2: [f32; 3],
    pub d_e3: [f32; 3],
    pub d_v2n: [f32; 3],
    pub d_u2: [f32; 3],
    pub d_proj: f32,
    pub d_v1: [f32; 3],
    pub d_v2: [f32; 3],
}

#[allow(clippy::too_many_arguments)]
pub fn rigid_from_3_points_bwd_traced(
    n: &[f32; 3],
    ca: &[f32; 3],
    c: &[f32; 3],
    is_na: bool,
    costgt_na: f32,
    dr: &[[f32; 3]; 3],
    dt: &[f32; 3],
) -> ([f32; 3], [f32; 3], [f32; 3], RigidTrace) {
    let mut tc = RigidTrace::default();
    // ---- forward, recomputed (cheap, and keeps the two in step) -----------
    let v1 = [c[0] - ca[0], c[1] - ca[1], c[2] - ca[2]];
    let v2 = [n[0] - ca[0], n[1] - ca[1], n[2] - ca[2]];
    let nrm1 = norm3(v1);
    let d1 = nrm1 + RIGID_EPS;
    let e1 = [v1[0] / d1, v1[1] / d1, v1[2] / d1];
    let proj = dot3_pinned(e1, v2);
    let u2 = [v2[0] - proj * e1[0], v2[1] - proj * e1[1], v2[2] - proj * e1[2]];
    let nrm2 = norm3(u2);
    let d2 = nrm2 + RIGID_EPS;
    let e2 = [u2[0] / d2, u2[1] / d2, u2[2] / d2];
    let e3 = cross_pinned(e1, e2);
    let nrm3v = norm3(v2);
    let d3 = nrm3v + RIGID_EPS;
    let v2n = [v2[0] / d3, v2[1] / d3, v2[2] / d3];
    let cosref = mul_then_sum3(e1, v2n);
    let costgt = if is_na { costgt_na } else { COSTGT_PROT };
    let inner = (1.0 - cosref * cosref) * (1.0 - costgt * costgt) + RIGID_EPS;
    let sq_inner = sqrt_pinned(inner);
    let raw = cosref * costgt + sq_inner;
    let cos2del = raw.clamp(-1.0, 1.0);
    let arg_cos = 0.5 * (1.0 + cos2del) + RIGID_EPS;
    let cosdel = sqrt_pinned(arg_cos);
    let sgn = {
        let d = costgt - cosref;
        if d > 0.0 {
            1.0f32
        } else if d < 0.0 {
            -1.0f32
        } else {
            0.0f32
        }
    };
    let arg_sin = 1.0 - 0.5 * (1.0 + cos2del) + RIGID_EPS;
    let q = sqrt_pinned(arg_sin);
    let rcol = [[e1[0], e2[0], e3[0]], [e1[1], e2[1], e3[1]], [e1[2], e2[2], e3[2]]];
    let rp = [[cosdel, -sgn * q, 0.0f32], [sgn * q, cosdel, 0.0], [0.0, 0.0, 1.0]];

    // ---- reverse ----------------------------------------------------------
    // R = Rcol @ Rp  (a pinned einsum): dRcol = dR @ Rp^T, dRp = Rcol^T @ dR
    let mut drcol = [[0.0f32; 3]; 3];
    let mut drp = [[0.0f32; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            let mut a = 0.0f64;
            for k in 0..3 {
                a += dr[i][k] as f64 * rp[j][k] as f64;
            }
            drcol[i][j] = a as f32;
            let mut b = 0.0f64;
            for k in 0..3 {
                b += rcol[k][i] as f64 * dr[k][j] as f64;
            }
            drp[i][j] = b as f32;
        }
    }
    // `Rp` is filled (0,0),(0,1),(1,0),(1,1), so its `CopySlices` chain unwinds
    // in the reverse of that order.
    tc.d_rc = drcol;
    tc.d_rp = drp;
    let dcosdel = drp[1][1] + drp[0][0];
    let dsindel = drp[1][0] + (-drp[0][1]);
    tc.d_cosdel = dcosdel;
    tc.d_sindel = dsindel;

    // `sindel` is written after `cosdel`, so its subgraph runs first and its
    // term reaches `cos2del`'s buffer first.
    //   sindel = sign(costgt-cosref) * sqrt(1 - 0.5*(1+cos2del) + eps)
    let darg_sin = sqrt_bwd(dsindel * sgn, arg_sin);
    let mut dcos2del = -darg_sin * 0.5;
    //   cosdel = sqrt(0.5*(1+cos2del) + eps)
    dcos2del += sqrt_bwd(dcosdel, arg_cos) * 0.5;

    // `clamp_backward` is `where((self >= min) & (self <= max), grad, 0)` — the
    // bounds are INCLUSIVE, so a value sitting exactly on ±1 still passes.
    tc.d_cos2del = dcos2del;
    let draw = if raw >= -1.0 && raw <= 1.0 { dcos2del } else { 0.0 };

    // `torch.sign(costgt - cosref)` has a zero derivative, but it is still a
    // node: it puts a `-0.0` into `cosref`'s buffer before anything else.
    let mut dcosref = -0.0f32;
    // sqrt(inner), inner = (1-cosref^2)*(1-costgt^2) + eps
    let dinner = sqrt_bwd(draw, inner);
    let kk = 1.0 - costgt * costgt;
    // `cosref*cosref` is one node with the SAME tensor on both sides, so it
    // contributes `grad·cosref` twice, separately.
    let dsq = -(dinner * kk);
    dcosref += dsq * cosref;
    dcosref += dsq * cosref;
    dcosref += draw * costgt;

    tc.d_cosref = dcosref;
    // cosref = sum(e1 * v2n). NOTE the assignments: autograd *moves* the first
    // gradient into an input buffer and only `+=`s the later ones, so starting
    // from a zero and adding would turn a `-0.0` into `+0.0` — which is exactly
    // how this stage failed (86 of 213 `d_v2n` values, all of them signed zeros).
    let mut de1 = [0.0f32; 3];
    let mut dv2n = [0.0f32; 3];
    for k in 0..3 {
        de1[k] = dcosref * v2n[k];
        dv2n[k] = dcosref * e1[k];
    }
    // e1, e2, e3 are the COLUMNS of Rcol
    let mut de2 = [0.0f32; 3];
    let mut de3 = [0.0f32; 3];
    for k in 0..3 {
        de1[k] += drcol[k][0];
        de2[k] = drcol[k][1];
        de3[k] = drcol[k][2];
    }
    // e3 = cross(e1, e2): self <- other x grad, other <- grad x self
    let c1 = cross_pinned(e2, de3);
    let c2 = cross_pinned(de3, e1);
    for k in 0..3 {
        de1[k] += c1[k];
        de2[k] += c2[k];
    }
    tc.d_e3 = de3;
    tc.d_e2 = de2;
    tc.d_v2n = dv2n;
    // v2n = v2 / d3, d3 = norm(v2) + eps
    let mut dv2 = [0.0f32; 3];
    let mut dd3 = 0.0f32;
    for k in 0..3 {
        dv2[k] = dv2n[k] / d3;
        dd3 += -dv2n[k] * (v2n[k] / d3);
    }
    let gn3 = dd3 as f64 / norm3_f64(v2);
    for k in 0..3 {
        dv2[k] += (v2[k] as f64 * gn3) as f32;
    }
    // e2 = u2 / d2
    let mut du2 = [0.0f32; 3];
    let mut dd2 = 0.0f32;
    for k in 0..3 {
        du2[k] = de2[k] / d2;
        dd2 += -de2[k] * (e2[k] / d2);
    }
    let gn2 = dd2 as f64 / norm3_f64(u2);
    for k in 0..3 {
        du2[k] += (u2[k] as f64 * gn2) as f32;
    }
    tc.d_u2 = du2;
    // u2 = v2 - proj * e1
    let mut dproj = 0.0f32;
    for k in 0..3 {
        dv2[k] += du2[k];
        dproj += -du2[k] * e1[k];
        de1[k] += -du2[k] * proj;
    }
    // proj = dot(e1, v2)
    for k in 0..3 {
        de1[k] += dproj * v2[k];
        dv2[k] += dproj * e1[k];
    }
    tc.d_proj = dproj;
    tc.d_e1 = de1;
    // e1 = v1 / d1
    let mut dv1 = [0.0f32; 3];
    let mut dd1 = 0.0f32;
    for k in 0..3 {
        dv1[k] = de1[k] / d1;
        dd1 += -de1[k] * (e1[k] / d1);
    }
    let gn1 = dd1 as f64 / norm3_f64(v1);
    for k in 0..3 {
        dv1[k] += (v1[k] as f64 * gn1) as f32;
    }
    // v1 = C - CA ; v2 = N - CA
    let mut dn = [0.0f32; 3];
    let mut dca = [0.0f32; 3];
    let mut dc = [0.0f32; 3];
    for k in 0..3 {
        dc[k] = dv1[k];
        dn[k] = dv2[k];
        dca[k] = (dt[k] + -dv2[k]) + -dv1[k];
    }
    tc.d_v1 = dv1;
    tc.d_v2 = dv2;
    (dn, dca, dc, tc)
}
