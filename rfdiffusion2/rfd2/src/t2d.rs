//! `rf_diffusion/util.py:get_t2d` — the template pair features.
//!
//! `prepro` builds `t2d` as `[1, 2, L, L, 68]` and fills only template 0; the
//! self-conditioning template stays zero unless `inference.str_self_cond` is
//! on, which it is not in the RFD_173 demo configuration. The 68 channels are
//! `61` distance bins + `sin/cos` of `omega, theta, phi` + a mask plane.
//!
//! The chain is
//!
//! ```text
//! util.get_t2d
//!   rf2aa.util.xyz_t_to_frame_xyz_sm_mask   ligand rows borrow a 3-atom frame
//!   rf2aa.kinematics.xyz_to_t2d
//!     xyz_to_c6d       generate_Cbeta, cdist, get_dih, get_ang
//!     dist_to_onehot   two linspace bin edges, bucketize
//! ```
//!
//! ## What is pinned here and what is not
//!
//! `cdist`, `norm`, `sum`, `cross`, `atan2`, `acos`, `sin` and `cos` are all on
//! the pinned list, so each of them is one f64 evaluation with a single
//! narrowing. Everything between them — the subtractions in `get_dih`, the
//! division by `norm + eps`, `generate_Cbeta`'s four-term combination — is
//! plain fp32 elementwise arithmetic in the reference and stays fp32 here.
//! Pinning the whole expression instead would change the answer.
//!
//! ## The two guards that are not decoration
//!
//! `USE_CB` is read from the config (`preprocess.use_cb_to_get_pair_dist`,
//! true here) and picks which atom the distance map is built from; getting it
//! wrong shifts every distance bin. And `c6d[..., 0]` is set to `999.9` on the
//! diagonal *before* the `< DMAX` test, so self-pairs never receive an
//! orientation — they land in the last distance bin with zero orientation and
//! a mask of 1.

use crate::chemical_gen::NTOTAL;

/// `rf2aa.kinematics.PARAMS`.
pub const DMIN: f32 = 1.0;
pub const DMID: f32 = 4.0;
pub const DMAX: f32 = 20.0;
pub const DBINS1: usize = 30;
pub const DBINS2: usize = 30;
/// `DBINS1 + DBINS2 + 1` one-hot distance channels.
pub const N_DIST_BINS: usize = DBINS1 + DBINS2 + 1;
/// distance bins + sin/cos of three angles + the mask plane
pub const T2D_WIDTH: usize = N_DIST_BINS + 6 + 1;

const EPS: f32 = 1e-4;

/// `torch.norm(v, dim=-1)` under pinning: f64 interior, one narrowing.
#[inline]
fn norm3(v: [f32; 3]) -> f32 {
    ((v[0] as f64 * v[0] as f64 + v[1] as f64 * v[1] as f64 + v[2] as f64 * v[2] as f64)
        .sqrt()) as f32
}

/// `torch.sum(a*b, dim=-1)`: the products are fp32, the 3-term sum is f64.
#[inline]
fn dot3(a: [f32; 3], b: [f32; 3]) -> f32 {
    let mut acc = 0.0f64;
    for k in 0..3 {
        acc += (a[k] * b[k]) as f64;
    }
    acc as f32
}

/// `torch.cross(a, b)` under pinning — each component is one f64 expression.
#[inline]
fn cross3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    let f = |x: f32, y: f32, z: f32, w: f32| {
        (x as f64 * y as f64 - z as f64 * w as f64) as f32
    };
    [
        f(a[1], b[2], a[2], b[1]),
        f(a[2], b[0], a[0], b[2]),
        f(a[0], b[1], a[1], b[0]),
    ]
}

#[inline]
fn sub3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

/// `kinematics.generate_Cbeta` — the Rosetta-parameter virtual CB.
#[inline]
fn generate_cbeta(n: [f32; 3], ca: [f32; 3], c: [f32; 3]) -> [f32; 3] {
    let b = sub3(ca, n);
    let cc = sub3(c, ca);
    let a = cross3(b, cc);
    let mut out = [0.0f32; 3];
    for k in 0..3 {
        out[k] = -0.57910144 * a[k] + 0.5689693 * b[k] - 0.5441217 * cc[k] + ca[k];
    }
    out
}

/// `kinematics.get_dih` — `atan2(y + eps, x + eps)`, both offsets included.
fn get_dih(a: [f32; 3], b: [f32; 3], c: [f32; 3], d: [f32; 3]) -> f32 {
    let b0 = sub3(a, b);
    let b1 = sub3(c, b);
    let b2 = sub3(d, c);
    let n1 = norm3(b1) + EPS;
    let b1n = [b1[0] / n1, b1[1] / n1, b1[2] / n1];
    let s0 = dot3(b0, b1n);
    let s2 = dot3(b2, b1n);
    let v = [b0[0] - s0 * b1n[0], b0[1] - s0 * b1n[1], b0[2] - s0 * b1n[2]];
    let w = [b2[0] - s2 * b1n[0], b2[1] - s2 * b1n[1], b2[2] - s2 * b1n[2]];
    let x = dot3(v, w);
    let y = dot3(cross3(b1n, v), w);
    (((y + EPS) as f64).atan2((x + EPS) as f64)) as f32
}

/// `kinematics.get_ang` — `acos(clamp(v·w, -0.999, 0.999))`.
fn get_ang(a: [f32; 3], b: [f32; 3], c: [f32; 3]) -> f32 {
    let v = sub3(a, b);
    let w = sub3(c, b);
    let nv = norm3(v) + EPS;
    let nw = norm3(w) + EPS;
    let vn = [v[0] / nv, v[1] / nv, v[2] / nv];
    let wn = [w[0] / nw, w[1] / nw, w[2] / nw];
    let vw = dot3(vn, wn).clamp(-0.999, 0.999);
    ((vw as f64).acos()) as f32
}

/// `kinematics.dist_to_bins` — `bucketize` against two concatenated linspaces.
///
/// `torch.bucketize` with the default `right=False` returns the number of
/// boundaries strictly less than the value, so a distance landing exactly on a
/// boundary goes to the *lower* bin. NaN was already replaced by 999.9 upstream.
fn dist_bin(d: f32, edges: &[f32]) -> usize {
    edges.iter().filter(|e| **e < d).count()
}

fn bin_edges() -> Vec<f32> {
    let dstep1 = (DMID - DMIN) / DBINS1 as f32;
    let dstep2 = (DMAX - DMID) / DBINS2 as f32;
    let mut e = Vec::with_capacity(DBINS1 + DBINS2);
    // torch.linspace(start, end, n) = start + i*(end-start)/(n-1)
    let lin = |start: f32, end: f32, n: usize, out: &mut Vec<f32>| {
        let step = (end - start) / (n - 1) as f32;
        for i in 0..n {
            out.push(start + i as f32 * step);
        }
    };
    lin(DMIN + dstep1, DMID, DBINS1, &mut e);
    lin(DMID + dstep2, DMAX, DBINS2, &mut e);
    e
}

/// `rf2aa.util.xyz_t_to_frame_xyz_sm_mask`.
///
/// A ligand row is a single atom, so it has no N/CA/C of its own. Upstream
/// substitutes three *neighbouring* atoms named by `atom_frames`, an
/// `[n_sm, 3, 2]` table of `(row offset, atom slot)` pairs relative to that
/// ligand row. Those three coordinates go into slots 0..3 and the rest of the
/// row is left alone.
///
/// `atom_frames` is itself an input, not a computation — `get_atom_frames`
/// breaks priority ties by CPython set iteration order, and 20 of the 50 atoms
/// here tie (see `results/README.md`, rung 4d).
pub fn frame_xyz(xyz: &[f32], l: usize, is_sm: &[bool], atom_frames: &[i64]) -> Vec<f32> {
    let mut out = xyz.to_vec();
    let sm: Vec<usize> = (0..l).filter(|i| is_sm[*i]).collect();
    if sm.is_empty() {
        return out;
    }
    assert_eq!(
        atom_frames.len(),
        sm.len() * 6,
        "atom_frames must be [n_sm, 3, 2]"
    );
    // Upstream flattens the ligand block to [atom_L * natoms, 3] and indexes it
    // with `(i + offset) * natoms + slot`, so the offset is relative to the
    // ligand row's position *within the ligand block*, not within `xyz`.
    for (i, &row) in sm.iter().enumerate() {
        for a in 0..3 {
            let off = atom_frames[(i * 3 + a) * 2] as i64;
            let slot = atom_frames[(i * 3 + a) * 2 + 1] as usize;
            let src_block = (i as i64 + off) as usize;
            let src = sm[src_block];
            for c in 0..3 {
                out[(row * NTOTAL + a) * 3 + c] = xyz[(src * NTOTAL + slot) * 3 + c];
            }
        }
    }
    out
}

/// `kinematics.xyz_to_t2d` for a single template, with the mask plane all true.
///
/// `xyz` is `[L, NTOTAL, 3]`; only slots 0..3 (N, CA, C) are read. Returns
/// `[L, L, 68]`.
pub fn xyz_to_t2d(xyz: &[f32], l: usize, use_cb: bool) -> Vec<f32> {
    let at = |i: usize, a: usize| -> [f32; 3] {
        let o = (i * NTOTAL + a) * 3;
        [xyz[o], xyz[o + 1], xyz[o + 2]]
    };
    let n: Vec<[f32; 3]> = (0..l).map(|i| at(i, 0)).collect();
    let ca: Vec<[f32; 3]> = (0..l).map(|i| at(i, 1)).collect();
    let c: Vec<[f32; 3]> = (0..l).map(|i| at(i, 2)).collect();
    let cb: Vec<[f32; 3]> = (0..l).map(|i| generate_cbeta(n[i], ca[i], c[i])).collect();

    // `get_pair_dist` is `torch.cdist`, which is pinned — same expansion as
    // `geom::cdist_self`, including the catastrophic diagonal cancellation.
    let pts: Vec<f32> = if use_cb {
        cb.iter().flatten().copied().collect()
    } else {
        ca.iter().flatten().copied().collect()
    };
    let mut dist = crate::geom::cdist_self(&pts, l);

    // `dist[isnan] = 999.9`, then `+ 999.9 * eye`
    for d in dist.iter_mut() {
        if d.is_nan() {
            *d = 999.9;
        }
    }
    for i in 0..l {
        dist[i * l + i] += 999.9;
    }

    let edges = bin_edges();
    let mut out = vec![0.0f32; l * l * T2D_WIDTH];
    for i in 0..l {
        for j in 0..l {
            let mut d = dist[i * l + j];
            // the `< DMAX` test happens before the long-range fixup
            let (om, th, ph) = if d < DMAX {
                (
                    get_dih(ca[i], cb[i], cb[j], ca[j]),
                    get_dih(n[i], ca[i], cb[i], cb[j]),
                    get_ang(ca[i], cb[i], cb[j]),
                )
            } else {
                (0.0, 0.0, 0.0)
            };
            if d >= DMAX {
                d = 999.9;
            }
            // `nan_to_num` after the fixup: an all-NaN row would otherwise
            // poison the bins.
            let fix = |v: f32| if v.is_nan() { 0.0 } else { v };
            let (d, om, th, ph) = (fix(d), fix(om), fix(th), fix(ph));

            let o = (i * l + j) * T2D_WIDTH;
            out[o + dist_bin(d, &edges)] = 1.0;
            for (k, a) in [om, th, ph].iter().enumerate() {
                out[o + N_DIST_BINS + k] = ((*a as f64).sin()) as f32;
                out[o + N_DIST_BINS + 3 + k] = ((*a as f64).cos()) as f32;
            }
            out[o + T2D_WIDTH - 1] = 1.0;
        }
    }
    out
}

/// `util.get_t2d(xyz, is_sm, atom_frames, use_cb)`.
pub fn get_t2d(
    xyz: &[f32],
    l: usize,
    is_sm: &[bool],
    atom_frames: &[i64],
    use_cb: bool,
) -> Vec<f32> {
    let framed = frame_xyz(xyz, l, is_sm, atom_frames);
    xyz_to_t2d(&framed, l, use_cb)
}
