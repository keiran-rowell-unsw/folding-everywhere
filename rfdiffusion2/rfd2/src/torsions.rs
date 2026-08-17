//! `rf2aa/util_module.py:XYZConverter.get_torsions` — the `alpha_t` template
//! feature.
//!
//! Twenty degrees of freedom per token: seven protein dihedrals (omega, phi,
//! psi, chi1..chi4), three *angle* features built against per-residue reference
//! angles (CB bend, CB twist, CG bend), and ten nucleic-acid dihedrals that are
//! all masked off for a protein + ligand input. Each is stored as a
//! `(cos, sin)` pair, and `prepro` appends the mask as a third channel to get
//! the 60-wide `alpha_t`.
//!
//! ## Two things that are easy to get wrong
//!
//! * **The coordinates are idealized first.** `get_torsions` does not measure
//!   the dihedrals off the input backbone — it rebuilds N and C from the
//!   `rigid_from_3_points` frame (`idealize_reference_frame`) and measures off
//!   *that*. Skipping it gives angles that are close and never exact.
//! * **`torsion_indices` can reach into the previous residue.** A negative
//!   entry means "the same atom slot, one row back"; the sign is the row
//!   offset and the magnitude is the slot. Row 0's omega/phi therefore index
//!   row -1, which wraps in torch and lands on the *last* row — reproduced
//!   here deliberately rather than clamped, because the reference's value is
//!   what the network was trained against, and the mask hides it anyway.
//!
//! Pinning follows the reference op by op: `sum`, `sqrt`, `cross` and `norm`
//! are f64-with-one-narrowing; the subtractions, divisions and `+eps` between
//! them are fp32.

use crate::chemical::table_f32;
use crate::chemical_gen::{COSTGTNA, NTOTAL, NTOTALDOFS};
use crate::geom;

const EPS: f32 = 1e-4;

/// `x.square().sum(-1, keepdim=True).add(eps).sqrt()` — the sum and the sqrt
/// are pinned, the `+eps` between them is fp32.
#[inline]
fn th_norm(x: [f32; 3]) -> f32 {
    let mut acc = 0.0f64;
    for k in 0..3 {
        acc += (x[k] * x[k]) as f64;
    }
    let s = (acc as f32) + EPS;
    ((s as f64).sqrt()) as f32
}

/// `th_N(x)` — normalise by `th_norm`, with `alpha = 0`.
#[inline]
fn th_n(x: [f32; 3]) -> [f32; 3] {
    let d = th_norm(x);
    [x[0] / d, x[1] / d, x[2] / d]
}

/// `(a*b).sum(-1)` — products in fp32, the 3-term sum in f64.
#[inline]
fn mul_sum(a: [f32; 3], b: [f32; 3]) -> f32 {
    let mut acc = 0.0f64;
    for k in 0..3 {
        acc += (a[k] * b[k]) as f64;
    }
    acc as f32
}

/// `torch.cross` under pinning.
#[inline]
fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    let f = |x: f32, y: f32, z: f32, w: f32| (x as f64 * y as f64 - z as f64 * w as f64) as f32;
    [
        f(a[1], b[2], a[2], b[1]),
        f(a[2], b[0], a[0], b[2]),
        f(a[0], b[1], a[1], b[0]),
    ]
}

#[inline]
fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

/// `chemical.th_dih_v` — returns `(cos, sin)`.
fn th_dih_v(ab: [f32; 3], bc: [f32; 3], cd: [f32; 3]) -> [f32; 2] {
    let ab = th_n(ab);
    let bc = th_n(bc);
    let cd = th_n(cd);
    let n1 = th_n(cross(ab, bc));
    let n2 = th_n(cross(bc, cd));
    let sin_angle = mul_sum(cross(n1, bc), n2);
    let cos_angle = mul_sum(n1, n2);
    [cos_angle, sin_angle]
}

/// `chemical.th_dih(a, b, c, d)`.
#[inline]
fn th_dih(a: [f32; 3], b: [f32; 3], c: [f32; 3], d: [f32; 3]) -> [f32; 2] {
    th_dih_v(sub(a, b), sub(b, c), sub(c, d))
}

/// `chemical.th_ang_v` — returns `(cos, sin)` with `sin >= 0`.
fn th_ang_v(ab: [f32; 3], bc: [f32; 3]) -> [f32; 2] {
    let ab = th_n(ab);
    let bc = th_n(bc);
    let cos_angle = mul_sum(ab, bc).clamp(-1.0, 1.0);
    let inner = 1.0 - cos_angle * cos_angle + EPS;
    let sin_angle = ((inner as f64).sqrt()) as f32;
    [cos_angle, sin_angle]
}

/// `rf2aa/util.py:idealize_reference_frame`.
///
/// N and C are replaced by the ideal positions implied by the residue frame,
/// so the torsions are measured off a canonical backbone rather than off
/// whatever the model or the noiser produced.
pub fn idealize_reference_frame(seq: &[i64], xyz: &[f32], l: usize) -> Vec<f32> {
    let mut out = xyz.to_vec();
    let is_na: Vec<bool> = seq.iter().map(|t| geom::is_nucleic(*t)).collect();
    let pick = |a: usize| -> Vec<f32> {
        let mut v = Vec::with_capacity(l * 3);
        for i in 0..l {
            let o = (i * NTOTAL + a) * 3;
            v.extend_from_slice(&xyz[o..o + 3]);
        }
        v
    };
    let (n, ca, c) = (pick(0), pick(1), pick(2));
    let (rs, ts) = geom::rigid_from_3_points(&n, &ca, &c, l, &is_na, COSTGTNA);

    let init_n = table_f32("init_N").data;
    let init_c = table_f32("init_C").data;
    let init_o1 = table_f32("init_O1").data;
    let init_o2 = table_f32("init_O2").data;

    // `torch.einsum('...ij,j->...i', R, v) + T` — pinned, so f64 interior.
    let apply = |r: &[f32], t: &[f32], i: usize, v: &[f32]| -> [f32; 3] {
        let mut o = [0.0f32; 3];
        for a in 0..3 {
            let mut acc = 0.0f64;
            for b in 0..3 {
                acc += r[i * 9 + a * 3 + b] as f64 * v[b] as f64;
            }
            o[a] = (acc as f32) + t[i * 3 + a];
        }
        o
    };
    for i in 0..l {
        let (slot0, slot2) = if is_na[i] {
            (apply(&rs, &ts, i, &init_o1), apply(&rs, &ts, i, &init_o2))
        } else {
            (apply(&rs, &ts, i, &init_n), apply(&rs, &ts, i, &init_c))
        };
        for k in 0..3 {
            out[(i * NTOTAL) * 3 + k] = slot0[k];
            out[(i * NTOTAL + 2) * 3 + k] = slot2[k];
        }
    }
    out
}

/// What `get_torsions` returns, restricted to the two things `prepro` reads.
pub struct Torsions {
    /// `[L, NTOTALDOFS, 2]` — `(cos, sin)` per degree of freedom
    pub alpha: Vec<f32>,
    /// `[L, NTOTALDOFS]`
    pub mask: Vec<bool>,
}

/// `XYZConverter.get_torsions(xyz, seq)` with `mask_in = None`.
pub fn get_torsions(seq: &[i64], xyz: &[f32], l: usize) -> Torsions {
    let (ti, ti_shape) = crate::chemical::table_i64("torsion_indices");
    assert_eq!(ti_shape, vec![80, NTOTALDOFS, 4]);
    let ref_angles = table_f32("reference_angles").data; // [80, 3, 2]

    let xyz = idealize_reference_frame(seq, xyz, l);
    let at = |i: usize, a: usize| -> [f32; 3] {
        let o = (i * NTOTAL + a) * 3;
        [xyz[o], xyz[o + 1], xyz[o + 2]]
    };

    let mut alpha = vec![0.0f32; l * NTOTALDOFS * 2];
    let mut mask = vec![false; l * NTOTALDOFS];

    for i in 0..l {
        let s = seq[i] as usize;
        // `tors_mask = torsion_indices[seq][..., -1] > 0`
        for t in 0..NTOTALDOFS {
            mask[i * NTOTALDOFS + t] = ti[(s * NTOTALDOFS + t) * 4 + 3] > 0;
        }

        // `xs = arange(L) - (ts < 0)`, `ys = abs(ts)`. A negative index wraps
        // in torch, so row 0 reaching back lands on row L-1.
        let gather = |t: usize, k: usize| -> [f32; 3] {
            let v = ti[(s * NTOTALDOFS + t) * 4 + k];
            let row = if v < 0 { (i + l - 1) % l } else { i };
            at(row, v.unsigned_abs() as usize)
        };

        // protein dihedrals (omega, phi, psi, chi1..chi4), then the NA block
        for t in (0..7).chain(10..NTOTALDOFS) {
            let d = th_dih(gather(t, 0), gather(t, 1), gather(t, 2), gather(t, 3));
            alpha[(i * NTOTALDOFS + t) * 2] = d[0];
            alpha[(i * NTOTALDOFS + t) * 2 + 1] = d[1];
        }
        // psi is shifted by pi
        alpha[(i * NTOTALDOFS + 2) * 2] *= -1.0;
        alpha[(i * NTOTALDOFS + 2) * 2 + 1] *= -1.0;

        // the three angle features, expressed against the residue's reference
        // angle as a rotation: (cos, sin) = (t . t0, t_x t0_y - t_y t0_x)
        let nc = {
            let (n, c) = (at(i, 0), at(i, 2));
            [
                0.5 * (n[0] + c[0]),
                0.5 * (n[1] + c[1]),
                0.5 * (n[2] + c[2]),
            ]
        };
        let ca = at(i, 1);
        let cb = at(i, 4);
        let cg = at(i, 5);
        let against = |t: [f32; 2], slot: usize, out: &mut [f32]| {
            let t0 = [
                ref_angles[(s * 3 + slot) * 2],
                ref_angles[(s * 3 + slot) * 2 + 1],
            ];
            // `torch.sum(t*t0, -1)` is pinned; the cross term is elementwise.
            let mut acc = 0.0f64;
            for k in 0..2 {
                acc += (t[k] * t0[k]) as f64;
            }
            out[0] = acc as f32;
            out[1] = t[0] * t0[1] - t[1] * t0[0];
        };

        let mut buf = [0.0f32; 2];
        // CB bend
        against(th_ang_v(sub(cb, ca), sub(nc, ca)), 0, &mut buf);
        alpha[(i * NTOTALDOFS + 7) * 2] = buf[0];
        alpha[(i * NTOTALDOFS + 7) * 2 + 1] = buf[1];

        // CB twist — NC' projected off the NC-CA axis
        let nc_ca = sub(nc, ca);
        let ncp = sub(at(i, 2), at(i, 0));
        let num = mul_sum(ncp, nc_ca);
        let den = mul_sum(nc_ca, nc_ca);
        let f = num / den;
        let ncpp = [
            ncp[0] - f * nc_ca[0],
            ncp[1] - f * nc_ca[1],
            ncp[2] - f * nc_ca[2],
        ];
        against(th_ang_v(sub(cb, ca), ncpp), 1, &mut buf);
        alpha[(i * NTOTALDOFS + 8) * 2] = buf[0];
        alpha[(i * NTOTALDOFS + 8) * 2 + 1] = buf[1];

        // CG bend
        against(th_ang_v(sub(cg, cb), sub(ca, cb)), 2, &mut buf);
        alpha[(i * NTOTALDOFS + 9) * 2] = buf[0];
        alpha[(i * NTOTALDOFS + 9) * 2 + 1] = buf[1];
    }

    // NaN -> (1, 0), per channel independently, exactly as upstream does it
    for v in alpha.chunks_mut(2) {
        if v[0].is_nan() {
            v[0] = 1.0;
        }
        if v[1].is_nan() {
            v[1] = 0.0;
        }
    }

    Torsions { alpha, mask }
}
