//! `util_module.XYZConverter.compute_all_atom` — backbone frames plus torsions
//! to all-atom coordinates.
//!
//! Every table it needs (`RTs_by_torsion`, `xyzs_in_base_frame`, `base_indices`)
//! is already in the embedded chemical export, so this is frame algebra only.
//!
//! Numerics follow the pinned convention op by op: `torch.einsum`,
//! `torch.linalg.norm`, `Tensor.cross` and `torch.sum` compute in f64 and round
//! once; the divisions, `+eps` and subtractions between them are fp32.

use crate::chemical;
use crate::chemical_gen::{NPROTAAS, NTOTAL, NNAPROTAAS, COSTGTNA};
use crate::geom;
use crate::tensor::Tensor;

const EPS_ROT: f32 = 1e-6;
const EPS_AXIS: f32 = 1e-4;

/// 4x4 homogeneous transform, row-major.
type M4 = [[f32; 4]; 4];

const I4: M4 = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
];

/// `torch.einsum('...ij,...jk->...ik')` — pinned, so f64 inside, f32 out.
fn mm4(a: &M4, b: &M4) -> M4 {
    let mut o = [[0.0f32; 4]; 4];
    for i in 0..4 {
        for k in 0..4 {
            let mut acc = 0.0f64;
            for j in 0..4 {
                acc += a[i][j] as f64 * b[j][k] as f64;
            }
            o[i][k] = acc as f32;
        }
    }
    o
}

/// f64 matrix product with **no** narrowing — the intermediate of a multi-operand
/// einsum.
fn mm4_f64(a: &[[f64; 4]; 4], b: &[[f64; 4]; 4]) -> [[f64; 4]; 4] {
    let mut o = [[0.0f64; 4]; 4];
    for i in 0..4 {
        for k in 0..4 {
            let mut acc = 0.0f64;
            for j in 0..4 {
                acc += a[i][j] * b[j][k];
            }
            o[i][k] = acc;
        }
    }
    o
}

#[inline]
fn to64(m: &M4) -> [[f64; 4]; 4] {
    let mut o = [[0.0f64; 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            o[i][j] = m[i][j] as f64;
        }
    }
    o
}

#[inline]
fn to32(m: &[[f64; 4]; 4]) -> M4 {
    let mut o = [[0.0f32; 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            o[i][j] = m[i][j] as f32;
        }
    }
    o
}

/// `torch.einsum('brij,brjk,brkl->bril', a, b, c)`.
///
/// Multi-operand einsum is **not** one big sum: torch contracts the operands
/// pairwise, left to right, and each intermediate is a real f64 tensor with its
/// own rounding. Fusing the three into a single triple loop is more accurate and
/// therefore does not match — it changes 37 of 7 668 all-atom coordinates.
fn mm4_3(a: &M4, b: &M4, c: &M4) -> M4 {
    let ab = mm4_f64(&to64(a), &to64(b));
    to32(&mm4_f64(&ab, &to64(c)))
}

/// `torch.einsum('brij,brjk,brkl,brlm->brim', a, b, c, d)`.
///
/// `torch.einsum` hands multi-operand contractions to **opt_einsum** when it is
/// installed (it is, `rf2aa` depends on it), so the association is the path
/// opt_einsum picks, not left-to-right. With every dimension equal to 4 the
/// costs are symmetric and it pairs them up: `(a@b) @ (c@d)`. Getting this wrong
/// moves 37 of 7 668 all-atom coordinates — all of them in the chi2..chi4 chain
/// of the one long sidechain in the test case, because `RTF4` feeds `RTF5..7`.
fn mm4_4(a: &M4, b: &M4, c: &M4, d: &M4) -> M4 {
    to32(&mm4_4_assoc(a, b, c, d, ASSOC))
}

/// Which pairwise association `torch.einsum` uses for the 4-operand contraction.
/// Selected by measurement, not by assumption — see `mm4_4`.
pub static ASSOC: usize = 0;

pub fn mm4_4_assoc(a: &M4, b: &M4, c: &M4, d: &M4, which: usize) -> [[f64; 4]; 4] {
    let (a, b, c, d) = (to64(a), to64(b), to64(c), to64(d));
    match which {
        0 => mm4_f64(&mm4_f64(&mm4_f64(&a, &b), &c), &d), // ((ab)c)d
        1 => mm4_f64(&mm4_f64(&a, &b), &mm4_f64(&c, &d)), // (ab)(cd)
        2 => mm4_f64(&a, &mm4_f64(&b, &mm4_f64(&c, &d))), // a(b(cd))
        3 => mm4_f64(&a, &mm4_f64(&mm4_f64(&b, &c), &d)), // a((bc)d)
        _ => mm4_f64(&mm4_f64(&a, &mm4_f64(&b, &c)), &d), // (a(bc))d
    }
}

#[inline]
fn norm2_pinned(a: f32, b: f32) -> f32 {
    ((a as f64) * (a as f64) + (b as f64) * (b as f64)).sqrt() as f32
}

/// `make_rotX`: a rotation about x built from a (cos, sin) pair that is
/// normalised by its own 2-norm plus `eps` — not assumed to be a unit vector.
fn make_rot_x(c: f32, s: f32) -> M4 {
    let n = norm2_pinned(c, s) + EPS_ROT;
    let mut r = I4;
    r[1][1] = c / n;
    r[1][2] = -s / n;
    r[2][1] = s / n;
    r[2][2] = c / n;
    r
}

pub fn make_rot_x_pub(c: f32, s: f32) -> M4 {
    make_rot_x(c, s)
}

pub fn make_rot_z_pub(c: f32, s: f32) -> M4 {
    make_rot_z(c, s)
}

fn make_rot_z(c: f32, s: f32) -> M4 {
    let n = norm2_pinned(c, s) + EPS_ROT;
    let mut r = I4;
    r[0][0] = c / n;
    r[0][1] = -s / n;
    r[1][0] = s / n;
    r[1][1] = c / n;
    r
}

fn make_rot_axis(c: f32, s: f32, u: [f32; 3]) -> M4 {
    let n = norm2_pinned(c, s) + EPS_ROT;
    let ct = c / n;
    let st = s / n;
    let (u0, u1, u2) = (u[0], u[1], u[2]);
    let mut r = I4;
    r[0][0] = ct + u0 * u0 * (1.0 - ct);
    r[0][1] = u0 * u1 * (1.0 - ct) - u2 * st;
    r[0][2] = u0 * u2 * (1.0 - ct) + u1 * st;
    r[1][0] = u0 * u1 * (1.0 - ct) + u2 * st;
    r[1][1] = ct + u1 * u1 * (1.0 - ct);
    r[1][2] = u1 * u2 * (1.0 - ct) - u0 * st;
    r[2][0] = u0 * u2 * (1.0 - ct) - u1 * st;
    r[2][1] = u1 * u2 * (1.0 - ct) + u0 * st;
    r[2][2] = ct + u2 * u2 * (1.0 - ct);
    r
}

pub struct XyzConverter {
    /// `[NAATOKENS, 17, 4, 4]`
    rts: Tensor,
    /// `[NAATOKENS, NTOTAL, 4]`
    basexyz: Tensor,
    /// `[NAATOKENS, NTOTAL]`
    base_indices: Vec<i64>,
}

impl Default for XyzConverter {
    fn default() -> Self {
        Self::new()
    }
}

impl XyzConverter {
    pub fn new() -> Self {
        XyzConverter {
            rts: chemical::table_f32("RTs_by_torsion"),
            basexyz: chemical::table_f32("xyzs_in_base_frame"),
            base_indices: chemical::table_i64("base_indices").0,
        }
    }

    /// Exposed for the association probe in `tests/parity_xyzconv.rs`.
    pub fn rt_pub(&self, tok: usize, t: usize) -> M4 {
        self.rt(tok, t)
    }

    fn rt(&self, tok: usize, t: usize) -> M4 {
        let mut m = [[0.0f32; 4]; 4];
        let o = (tok * 17 + t) * 16;
        for i in 0..4 {
            for j in 0..4 {
                m[i][j] = self.rts.data[o + i * 4 + j];
            }
        }
        m
    }

    /// `(RTframes, xyz)` for one batch element.
    ///
    /// `xyz` is `[L, >=3, 3]` (only N/CA/C are read), `alphas` is
    /// `[L, NTOTALDOFS, 2]`. Returns all-atom coordinates `[L, NTOTAL, 3]`.
    pub fn compute_all_atom(&self, seq: &[i64], xyz: &[f32], n_in: usize, alphas: &[f32]) -> Vec<f32> {
        self.compute_all_atom_with_frames(seq, xyz, n_in, alphas).1
    }

    /// `(RTframes [L,17,4,4], xyz [L,NTOTAL,3])`.
    pub fn compute_all_atom_with_frames(
        &self,
        seq: &[i64],
        xyz: &[f32],
        n_in: usize,
        alphas: &[f32],
    ) -> (Vec<f32>, Vec<f32>) {
        let l = seq.len();
        let ndof = alphas.len() / (l * 2);
        let is_na: Vec<bool> = seq.iter().map(|&t| geom::is_nucleic(t)).collect();
        let g = |i: usize, a: usize, k: usize| xyz[(i * n_in + a) * 3 + k];
        let n: Vec<f32> = (0..l).flat_map(|i| (0..3).map(move |k| g(i, 0, k))).collect();
        let ca: Vec<f32> = (0..l).flat_map(|i| (0..3).map(move |k| g(i, 1, k))).collect();
        let c: Vec<f32> = (0..l).flat_map(|i| (0..3).map(move |k| g(i, 2, k))).collect();
        let (rs, ts) = geom::rigid_from_3_points(&n, &ca, &c, l, &is_na, COSTGTNA);

        let alpha = |i: usize, t: usize| (alphas[(i * ndof + t) * 2], alphas[(i * ndof + t) * 2 + 1]);
        let mut out = vec![0.0f32; l * NTOTAL * 3];
        let mut frames_out = vec![0.0f32; l * 17 * 16];

        for i in 0..l {
            let tok = seq[i] as usize;
            let mut f0 = I4;
            for r in 0..3 {
                for cc in 0..3 {
                    f0[r][cc] = rs[i * 9 + r * 3 + cc];
                }
                f0[r][3] = ts[i * 3 + r];
            }

            let mk = |t: usize| {
                let (cx, sx) = alpha(i, t);
                make_rot_x(cx, sx)
            };
            let f1 = mm4_3(&f0, &self.rt(tok, 0), &mk(0));
            let f2 = mm4_3(&f0, &self.rt(tok, 1), &mk(1));
            let f3 = mm4_3(&f0, &self.rt(tok, 2), &mk(2));

            // CB bend / twist axes, from the residue's ideal geometry
            let bx = |a: usize, k: usize| self.basexyz.data[(tok * NTOTAL + a) * 4 + k];
            let ncr = [
                0.5 * (bx(2, 0) + bx(0, 0)),
                0.5 * (bx(2, 1) + bx(0, 1)),
                0.5 * (bx(2, 2) + bx(0, 2)),
            ];
            let car = [bx(1, 0), bx(1, 1), bx(1, 2)];
            let cbr = [bx(4, 0), bx(4, 1), bx(4, 2)];
            let d1 = [cbr[0] - car[0], cbr[1] - car[1], cbr[2] - car[2]];
            let d2 = [ncr[0] - car[0], ncr[1] - car[1], ncr[2] - car[2]];
            let mut ax1 = cross_pinned(d1, d2);
            let n1 = norm3_pinned(ax1) + EPS_AXIS;
            for v in ax1.iter_mut() {
                *v /= n1;
            }
            let ncp = [bx(2, 0) - bx(0, 0), bx(2, 1) - bx(0, 1), bx(2, 2) - bx(0, 2)];
            let num = sum_mul3_pinned(ncp, ncr);
            let den = sum_mul3_pinned(ncr, ncr);
            let ncpp = [
                ncp[0] - num / den * ncr[0],
                ncp[1] - num / den * ncr[1],
                ncp[2] - num / den * ncr[2],
            ];
            let mut ax2 = cross_pinned(d1, ncpp);
            let n2 = norm3_pinned(ax2) + EPS_AXIS;
            for v in ax2.iter_mut() {
                *v /= n2;
            }
            let (c7, s7) = alpha(i, 7);
            let (c8, s8) = alpha(i, 8);
            let f8 = mm4_3(&f0, &make_rot_axis(c7, s7, ax1), &make_rot_axis(c8, s8, ax2));

            let (c9, s9) = alpha(i, 9);
            let f4 = mm4_4(&f8, &self.rt(tok, 3), &mk(3), &make_rot_z(c9, s9));
            let f5 = mm4_3(&f4, &self.rt(tok, 4), &mk(4));
            let f6 = mm4_3(&f5, &self.rt(tok, 5), &mk(5));
            let f7 = mm4_3(&f6, &self.rt(tok, 6), &mk(6));

            // Nucleic-acid frames. `ChemicalData` is built with
            // `use_phospate_frames_for_NA = True` (that is `rf_diffusion.chemical`'s
            // default, and it is the one the inference path initialises), so this
            // is the phosphate-frame chain: alpha off the base frame, then
            // beta -> gamma -> delta, nu2 off *gamma*, nu1/nu0 off nu2, chi off
            // nu1. The other branch chains in the opposite direction.
            //
            // No protein or ligand token's `base_indices` ever points at frames
            // 9..16, so this does not affect the present test case — it is here so
            // that a nucleic-acid input is not silently wrong.
            let f9 = mm4_3(&f0, &self.rt(tok, 9), &mk(12));
            let f10 = mm4_3(&f9, &self.rt(tok, 10), &mk(13));
            let f11 = mm4_3(&f10, &self.rt(tok, 11), &mk(14));
            let f12 = mm4_3(&f11, &self.rt(tok, 12), &mk(15));
            let f13 = mm4_3(&f11, &self.rt(tok, 13), &mk(16));
            let f14 = mm4_3(&f13, &self.rt(tok, 14), &mk(17));
            let f15 = mm4_3(&f14, &self.rt(tok, 15), &mk(18));
            let f16 = mm4_3(&f14, &self.rt(tok, 16), &mk(19));

            let frames = [f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15, f16];
            for (t, m) in frames.iter().enumerate() {
                for r in 0..4 {
                    for cc in 0..4 {
                        frames_out[((i * 17 + t) * 4 + r) * 4 + cc] = m[r][cc];
                    }
                }
            }
            for a in 0..NTOTAL {
                let fi = self.base_indices[tok * NTOTAL + a] as usize;
                let m = &frames[fi];
                for r in 0..3 {
                    let mut acc = 0.0f64;
                    for k in 0..4 {
                        acc += m[r][k] as f64 * bx(a, k) as f64;
                    }
                    out[(i * NTOTAL + a) * 3 + r] = acc as f32;
                }
            }
        }
        (frames_out, out)
    }
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
fn norm3_pinned(v: [f32; 3]) -> f32 {
    let (a, b, c) = (v[0] as f64, v[1] as f64, v[2] as f64);
    (a * a + b * b + c * c).sqrt() as f32
}

#[inline]
fn sum_mul3_pinned(a: [f32; 3], b: [f32; 3]) -> f32 {
    let mut s = 0.0f64;
    for k in 0..3 {
        s += (a[k] * b[k]) as f64;
    }
    s as f32
}

#[allow(dead_code)]
fn _unused() {
    let _ = (NPROTAAS, NNAPROTAAS);
}
