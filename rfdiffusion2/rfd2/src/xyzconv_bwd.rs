//! Reverse pass through `XYZConverter.compute_all_atom`.
//!
//! `calc_lj_grads` gets this from autograd; a port has to write it out. The
//! forward is a tree of 4x4 frame products, so the reverse is the usual
//! matrix-chain rule — for `Y = A·B·C` with `B` constant,
//! `dA = dY·(BC)^T` and `dC = (AB)^T·dY` — plus the derivatives of the three
//! rotation constructors and of `rigid_from_3_points`.
//!
//! Precision follows the same rule as `crate::chiral`: `python/pinned.py` builds
//! the autograd graph on *promoted* tensors, so each patched op's derivative is
//! evaluated in f64 with an fp32 rounding at the promotion and narrowing nodes,
//! while the unpatched elementwise steps between them stay fp32.

use crate::chemical;
use crate::chemical_gen::{COSTGTNA, NTOTAL};
use crate::geom;

type M4 = [[f32; 4]; 4];
type M4d = [[f64; 4]; 4];

const EPS_ROT: f32 = 1e-6;
const EPS_AXIS: f32 = 1e-4;

#[inline]
fn to64(m: &M4) -> M4d {
    let mut o = [[0.0f64; 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            o[i][j] = m[i][j] as f64;
        }
    }
    o
}

#[inline]
fn to32(m: &M4d) -> M4 {
    let mut o = [[0.0f32; 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            o[i][j] = m[i][j] as f32;
        }
    }
    o
}

fn mm(a: &M4d, b: &M4d) -> M4d {
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

fn tr(a: &M4d) -> M4d {
    let mut o = [[0.0f64; 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            o[i][j] = a[j][i];
        }
    }
    o
}

fn addm(a: &mut M4, b: &M4) {
    for i in 0..4 {
        for j in 0..4 {
            a[i][j] += b[i][j];
        }
    }
}

/// `Y = A·B·C` with `B` constant: returns `(dA, dC)`.
fn bwd3(dy: &M4, a: &M4, b: &M4, c: &M4) -> (M4, M4) {
    let (dy, a64, b64, c64) = (to64(dy), to64(a), to64(b), to64(c));
    let bc = mm(&b64, &c64);
    let ab = mm(&a64, &b64);
    (to32(&mm(&dy, &tr(&bc))), to32(&mm(&tr(&ab), &dy)))
}

/// `Y = A·B·C` with every operand live: returns `(dA, dB, dC)`.
fn bwd3_all(dy: &M4, a: &M4, b: &M4, c: &M4) -> (M4, M4, M4) {
    let (dy, a64, b64, c64) = (to64(dy), to64(a), to64(b), to64(c));
    let bc = mm(&b64, &c64);
    let ab = mm(&a64, &b64);
    (
        to32(&mm(&dy, &tr(&bc))),
        to32(&mm(&mm(&tr(&a64), &dy), &tr(&c64))),
        to32(&mm(&tr(&ab), &dy)),
    )
}

/// `Y = A·B·C·D` with `B` constant: returns `(dA, dC, dD)`.
fn bwd4(dy: &M4, a: &M4, b: &M4, c: &M4, d: &M4) -> (M4, M4, M4) {
    let (dy, a64, b64, c64, d64) = (to64(dy), to64(a), to64(b), to64(c), to64(d));
    let bcd = mm(&mm(&b64, &c64), &d64);
    let ab = mm(&a64, &b64);
    let abc = mm(&ab, &c64);
    let da = mm(&dy, &tr(&bcd));
    let dc = mm(&mm(&tr(&ab), &dy), &tr(&d64));
    let dd = mm(&tr(&abc), &dy);
    (to32(&da), to32(&dc), to32(&dd))
}

#[inline]
fn norm2(c: f32, s: f32) -> f32 {
    ((c as f64) * (c as f64) + (s as f64) * (s as f64)).sqrt() as f32
}

/// Backward of `NORM = torch.linalg.norm(angs, dim=-1) + eps`.
///
/// `linalg.norm` is one of the pinned entry points, so its subgraph is
/// `angs -> .double() -> norm_f64 -> .float()`: the derivative
/// `self * (grad / norm)` runs entirely in f64 against the **unrounded** f64
/// norm, and rounds to fp32 once, at the `.double()` node's backward.
#[inline]
fn norm_bwd(dn: f32, c: f32, s: f32) -> (f32, f32) {
    let nrm64 = ((c as f64) * (c as f64) + (s as f64) * (s as f64)).sqrt();
    let t = (dn as f64) / nrm64;
    (((c as f64) * t) as f32, ((s as f64) * t) as f32)
}

/// Gradient of `make_rotX` / `make_rotZ` w.r.t. its `(cos, sin)` pair, and of
/// `NORM`.
///
/// The four live entries are `c/n`, `-s/n`, `s/n`, `c/n` at positions that
/// differ between X and Z. Each is a *separate* `div` node in the reference, so
/// each contributes its own fp32-rounded term, and ATen's
/// `div_tensor_other_backward` is `-grad * ((self/other)/other)` — two chained
/// divisions, not one multiply by `1/n²`.
///
/// Accumulation order is the autograd engine's, i.e. reverse of creation: the
/// four assignments run (1,1)-last, so `NORM` sums its terms in the order
/// `(2,2), (2,1), (1,2), (1,1)` and `angs` receives the `div` terms before the
/// `norm` term.
fn rot_bwd(dm: &M4, c: f32, s: f32, z: bool) -> (f32, f32, f32) {
    let n = norm2(c, s) + EPS_ROT;
    // (row, col, numerator) in source order
    let ent: [(usize, usize, f32); 4] = if z {
        [(0, 0, c), (0, 1, -s), (1, 0, s), (1, 1, c)]
    } else {
        [(1, 1, c), (1, 2, -s), (2, 1, s), (2, 2, c)]
    };
    let mut dn = 0.0f32;
    let mut dc = 0.0f32;
    let mut ds = 0.0f32;
    // reverse creation order
    for k in (0..4).rev() {
        let (r, col, num) = ent[k];
        let g = dm[r][col];
        dn += -g * ((num / n) / n);
        let t = g / n;
        // entries 1 and 2 carry the sin; entry 1 went through a `neg`
        match k {
            0 | 3 => dc += t,
            1 => ds += -t,
            _ => ds += t,
        }
    }
    let (nc, ns) = norm_bwd(dn, c, s);
    (dc + nc, ds + ns, dn)
}

/// Gradient of `make_rot_axis` w.r.t. its `(cos, sin)` pair; the axis `u` comes
/// from `xyzs_in_base_frame`, a non-differentiable buffer, so it needs none.
///
/// Unlike `make_rotX`, `ct` and `st` are each computed **once** and shared by
/// the nine entries, so their gradients are accumulated over the entries first
/// and divided by `NORM` once. Each entry contributes through a separate
/// `rsub(1, ct)` node, hence a term `-(g · u_a u_b)` rounded on its own; the
/// three diagonal entries contribute `+g` as well, from the `add`, and that
/// term lands **before** the `rsub` one.
fn rot_axis_bwd(dm: &M4, c: f32, s: f32, u: [f32; 3]) -> (f32, f32, f32) {
    let n = norm2(c, s) + EPS_ROT;
    let (u0, u1, u2) = (u[0], u[1], u[2]);
    let (w0, w1, w2) = (u0 * u0, u1 * u1, u2 * u2);
    let (u01, u02, u12) = (u0 * u1, u0 * u2, u1 * u2);
    // (row, col, coefficient of the shared `1-ct`, `ct` appears directly,
    //  coefficient of `st`, `st` appears) — in source order.
    let ent: [(usize, usize, f32, bool, f32, bool); 9] = [
        (0, 0, w0, true, 0.0, false),
        (0, 1, u01, false, -u2, true),
        (0, 2, u02, false, u1, true),
        (1, 0, u01, false, u2, true),
        (1, 1, w1, true, 0.0, false),
        (1, 2, u12, false, -u0, true),
        (2, 0, u02, false, -u1, true),
        (2, 1, u12, false, u0, true),
        (2, 2, w2, true, 0.0, false),
    ];
    let mut dct = 0.0f32;
    let mut dst = 0.0f32;
    for k in (0..9).rev() {
        let (r, col, wco, direct, sco, has_s) = ent[k];
        let g = dm[r][col];
        if direct {
            dct += g;
        }
        dct += -(g * wco);
        if has_s {
            dst += g * sco;
        }
    }
    // `st`'s div node was created after `ct`'s, so it runs first.
    let mut dn = -dst * ((s / n) / n);
    dn += -dct * ((c / n) / n);
    let (nc, ns) = norm_bwd(dn, c, s);
    (dct / n + nc, dst / n + ns, dn)
}

pub struct AaGrads {
    /// `[L, 3, 3]` — gradient w.r.t. the N/CA/C coordinates
    pub dxyz: Vec<f32>,
    /// `[L, NTOTALDOFS, 2]`
    pub dalpha: Vec<f32>,
}

/// Every intermediate gradient of the reverse pass, in the reference's own
/// layout, so `tests/debug_aa_bwd.rs` can bisect against
/// `fixtures/refiner_io/aa_bwd.safetensors` instead of only checking the two
/// leaves.
#[derive(Default)]
pub struct Trace {
    /// `[L, 17, 4, 4]` — dL/dRTF{t}, fully accumulated
    pub dframe: Vec<f32>,
    /// `[L, 20, 4, 4]` — dL/d(rotation matrix) per alpha slot
    pub drot: Vec<f32>,
    /// `[L, 20]` — dL/dNORM per alpha slot
    pub dnorm: Vec<f32>,
    /// `[L, 3, 3]` — dL/dR out of the frame constructor
    pub drigid: Vec<f32>,
    /// per residue, the inside of `rigid_from_3_points_bwd`
    pub rigid: Vec<crate::geom::RigidTrace>,
}

/// Back-propagate `dL/dxyzaa` (`[L, NTOTAL, 3]`) through `compute_all_atom`.
pub fn backward(
    seq: &[i64],
    xyz: &[f32],
    n_in: usize,
    alphas: &[f32],
    dxyzaa: &[f32],
) -> AaGrads {
    backward_traced(seq, xyz, n_in, alphas, dxyzaa, None)
}

/// As [`backward`], optionally recording every intermediate into `trace`.
pub fn backward_traced(
    seq: &[i64],
    xyz: &[f32],
    n_in: usize,
    alphas: &[f32],
    dxyzaa: &[f32],
    mut trace: Option<&mut Trace>,
) -> AaGrads {
    let l = seq.len();
    let ndof = alphas.len() / (l * 2);
    let rts = chemical::table_f32("RTs_by_torsion");
    let basexyz = chemical::table_f32("xyzs_in_base_frame");
    let base_indices = chemical::table_i64("base_indices").0;

    let conv = crate::model::xyzconv::XyzConverter::new();
    let (frames, _) = conv.compute_all_atom_with_frames(seq, xyz, n_in, alphas);

    let mut dxyz = vec![0.0f32; l * 3 * 3];
    let mut dalpha = vec![0.0f32; l * ndof * 2];
    if let Some(t) = trace.as_deref_mut() {
        t.dframe = vec![0.0; l * 17 * 16];
        t.drot = vec![0.0; l * 20 * 16];
        t.dnorm = vec![0.0; l * 20];
        t.drigid = vec![0.0; l * 9];
    }

    for i in 0..l {
        let tok = seq[i] as usize;
        let f = |t: usize| -> M4 {
            let mut m = [[0.0f32; 4]; 4];
            for r in 0..4 {
                for c in 0..4 {
                    m[r][c] = frames[((i * 17 + t) * 4 + r) * 4 + c];
                }
            }
            m
        };
        let rt = |t: usize| -> M4 {
            let mut m = [[0.0f32; 4]; 4];
            let o = (tok * 17 + t) * 16;
            for r in 0..4 {
                for c in 0..4 {
                    m[r][c] = rts.data[o + r * 4 + c];
                }
            }
            m
        };
        let alpha = |t: usize| (alphas[(i * ndof + t) * 2], alphas[(i * ndof + t) * 2 + 1]);
        let mkx = |t: usize| {
            let (c, s) = alpha(t);
            crate::model::xyzconv::make_rot_x_pub(c, s)
        };

        // dL/dF[f] from the atom placement. `RTframes.gather(...)` is outside
        // the pinned einsum, so its backward is an fp32 `scatter_add` over the
        // atoms in ascending order — and the row-3 terms are `0.0 * basexyz`,
        // which is a *signed* zero, so they are computed rather than skipped.
        let mut df = [[[0.0f32; 4]; 4]; 17];
        for a in 0..NTOTAL {
            let fi = base_indices[tok * NTOTAL + a] as usize;
            for r in 0..4 {
                let g = if r < 3 { dxyzaa[(i * NTOTAL + a) * 3 + r] } else { 0.0 };
                for k in 0..4 {
                    df[fi][r][k] += g * basexyz.data[(tok * NTOTAL + a) * 4 + k];
                }
            }
        }

        let mut d_alpha_local = vec![0.0f32; ndof * 2];
        let mut drot_local = [[[0.0f32; 4]; 4]; 20];
        let mut dnorm_local = [0.0f32; 20];

        // protein sidechain chain, in reverse
        for (fi, prev, bi, ai) in [(7usize, 6usize, 6usize, 6usize), (6, 5, 5, 5), (5, 4, 4, 4)] {
            let (dprev, dr) = bwd3(&df[fi], &f(prev), &rt(bi), &mkx(ai));
            addm(&mut df[prev], &dprev);
            let (c, s) = alpha(ai);
            let (dc, ds, dn) = rot_bwd(&dr, c, s, false);
            d_alpha_local[ai * 2] += dc;
            d_alpha_local[ai * 2 + 1] += ds;
            drot_local[ai] = dr;
            dnorm_local[ai] = dn;
        }
        // F4 = F8 · B3 · RX(a3) · RZ(a9)
        {
            let (c9, s9) = alpha(9);
            let rz = crate::model::xyzconv::make_rot_z_pub(c9, s9);
            let (d8, drx, drz) = bwd4(&df[4], &f(8), &rt(3), &mkx(3), &rz);
            addm(&mut df[8], &d8);
            let (c3, s3) = alpha(3);
            let (dc, ds, dn) = rot_bwd(&drx, c3, s3, false);
            d_alpha_local[3 * 2] += dc;
            d_alpha_local[3 * 2 + 1] += ds;
            drot_local[3] = drx;
            dnorm_local[3] = dn;
            let (dc, ds, dn) = rot_bwd(&drz, c9, s9, true);
            d_alpha_local[9 * 2] += dc;
            d_alpha_local[9 * 2 + 1] += ds;
            drot_local[9] = drz;
            dnorm_local[9] = dn;
        }
        // nucleic-acid chain (phosphate-frame order), in reverse
        for (fi, prev, bi, ai) in [
            (16usize, 14usize, 16usize, 19usize),
            (15, 14, 15, 18),
            (14, 13, 14, 17),
            (13, 11, 13, 16),
            (12, 11, 12, 15),
            (11, 10, 11, 14),
            (10, 9, 10, 13),
            (9, 0, 9, 12),
        ] {
            let (dprev, dr) = bwd3(&df[fi], &f(prev), &rt(bi), &mkx(ai));
            addm(&mut df[prev], &dprev);
            let (c, s) = alpha(ai);
            let (dc, ds, dn) = rot_bwd(&dr, c, s, false);
            d_alpha_local[ai * 2] += dc;
            d_alpha_local[ai * 2 + 1] += ds;
            drot_local[ai] = dr;
            dnorm_local[ai] = dn;
        }
        // F8 = F0 · CBrot1(a7) · CBrot2(a8) — all three operands carry gradient
        {
            let (ax1, ax2) = cb_axes(&basexyz.data, tok);
            let (c7, s7) = alpha(7);
            let (c8, s8) = alpha(8);
            let r1 = make_rot_axis(c7, s7, ax1);
            let r2 = make_rot_axis(c8, s8, ax2);
            let (d0, dr1, dr2) = bwd3_all(&df[8], &f(0), &r1, &r2);
            addm(&mut df[0], &d0);
            let (dc, ds, dn) = rot_axis_bwd(&dr1, c7, s7, ax1);
            d_alpha_local[7 * 2] += dc;
            d_alpha_local[7 * 2 + 1] += ds;
            drot_local[7] = dr1;
            dnorm_local[7] = dn;
            let (dc, ds, dn) = rot_axis_bwd(&dr2, c8, s8, ax2);
            d_alpha_local[8 * 2] += dc;
            d_alpha_local[8 * 2 + 1] += ds;
            drot_local[8] = dr2;
            dnorm_local[8] = dn;
        }
        // F1/F2/F3 = F0 · B{0,1,2} · RX(a{0,1,2})
        for (fi, bi, ai) in [(3usize, 2usize, 2usize), (2, 1, 1), (1, 0, 0)] {
            let (d0, dr) = bwd3(&df[fi], &f(0), &rt(bi), &mkx(ai));
            addm(&mut df[0], &d0);
            let (c, s) = alpha(ai);
            let (dc, ds, dn) = rot_bwd(&dr, c, s, false);
            d_alpha_local[ai * 2] += dc;
            d_alpha_local[ai * 2 + 1] += ds;
            drot_local[ai] = dr;
            dnorm_local[ai] = dn;
        }

        for t in 0..ndof * 2 {
            dalpha[i * ndof * 2 + t] = d_alpha_local[t];
        }
        if let Some(tr) = trace.as_deref_mut() {
            for t in 0..17 {
                for r in 0..4 {
                    for c in 0..4 {
                        tr.dframe[((i * 17 + t) * 4 + r) * 4 + c] = df[t][r][c];
                    }
                }
            }
            for t in 0..20 {
                tr.dnorm[i * 20 + t] = dnorm_local[t];
                for r in 0..4 {
                    for c in 0..4 {
                        tr.drot[((i * 20 + t) * 4 + r) * 4 + c] = drot_local[t][r][c];
                    }
                }
            }
        }

        // F0 = [R | T]: R from rigid_from_3_points, T = CA
        let dr3 = {
            let mut m = [[0.0f32; 3]; 3];
            for r in 0..3 {
                for c in 0..3 {
                    m[r][c] = df[0][r][c];
                }
            }
            m
        };
        let dt = [df[0][0][3], df[0][1][3], df[0][2][3]];
        if let Some(tr) = trace.as_deref_mut() {
            for r in 0..3 {
                for c in 0..3 {
                    tr.drigid[(i * 3 + r) * 3 + c] = dr3[r][c];
                }
            }
        }
        let g = |a: usize, k: usize| xyz[(i * n_in + a) * 3 + k];
        let n3 = [g(0, 0), g(0, 1), g(0, 2)];
        let ca3 = [g(1, 0), g(1, 1), g(1, 2)];
        let c3 = [g(2, 0), g(2, 1), g(2, 2)];
        let (dn, dca, dc, rtc) = geom::rigid_from_3_points_bwd_traced(
            &n3,
            &ca3,
            &c3,
            geom::is_nucleic(seq[i]),
            COSTGTNA,
            &dr3,
            &dt,
        );
        if let Some(tr) = trace.as_deref_mut() {
            tr.rigid.push(rtc);
        }
        for k in 0..3 {
            dxyz[(i * 3) * 3 + k] += dn[k];
            dxyz[(i * 3 + 1) * 3 + k] += dca[k];
            dxyz[(i * 3 + 2) * 3 + k] += dc[k];
        }
    }

    AaGrads { dxyz, dalpha }
}

fn make_rot_axis(c: f32, s: f32, u: [f32; 3]) -> M4 {
    let n = norm2(c, s) + EPS_ROT;
    let ct = c / n;
    let st = s / n;
    let (u0, u1, u2) = (u[0], u[1], u[2]);
    let mut r = [[0.0f32; 4]; 4];
    for i in 0..4 {
        r[i][i] = 1.0;
    }
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

/// The two CB rotation axes, from the residue's ideal internal geometry. They
/// are constants of the token, so they carry no gradient.
fn cb_axes(basexyz: &[f32], tok: usize) -> ([f32; 3], [f32; 3]) {
    let bx = |a: usize, k: usize| basexyz[(tok * NTOTAL + a) * 4 + k];
    let ncr = [
        0.5 * (bx(2, 0) + bx(0, 0)),
        0.5 * (bx(2, 1) + bx(0, 1)),
        0.5 * (bx(2, 2) + bx(0, 2)),
    ];
    let car = [bx(1, 0), bx(1, 1), bx(1, 2)];
    let cbr = [bx(4, 0), bx(4, 1), bx(4, 2)];
    let d1 = [cbr[0] - car[0], cbr[1] - car[1], cbr[2] - car[2]];
    let d2 = [ncr[0] - car[0], ncr[1] - car[1], ncr[2] - car[2]];
    let cross = |a: [f32; 3], b: [f32; 3]| {
        let f = |x: f32| x as f64;
        [
            (f(a[1]) * f(b[2]) - f(a[2]) * f(b[1])) as f32,
            (f(a[2]) * f(b[0]) - f(a[0]) * f(b[2])) as f32,
            (f(a[0]) * f(b[1]) - f(a[1]) * f(b[0])) as f32,
        ]
    };
    let n3 = |v: [f32; 3]| {
        let (a, b, c) = (v[0] as f64, v[1] as f64, v[2] as f64);
        (a * a + b * b + c * c).sqrt() as f32
    };
    let mut ax1 = cross(d1, d2);
    let n1 = n3(ax1) + EPS_AXIS;
    for v in ax1.iter_mut() {
        *v /= n1;
    }
    let ncp = [bx(2, 0) - bx(0, 0), bx(2, 1) - bx(0, 1), bx(2, 2) - bx(0, 2)];
    let sm = |a: [f32; 3], b: [f32; 3]| {
        let mut s = 0.0f64;
        for k in 0..3 {
            s += (a[k] * b[k]) as f64;
        }
        s as f32
    };
    let num = sm(ncp, ncr);
    let den = sm(ncr, ncr);
    let ncpp = [
        ncp[0] - num / den * ncr[0],
        ncp[1] - num / den * ncr[1],
        ncp[2] - num / den * ncr[2],
    ];
    let mut ax2 = cross(d1, ncpp);
    let n2 = n3(ax2) + EPS_AXIS;
    for v in ax2.iter_mut() {
        *v /= n2;
    }
    (ax1, ax2)
}
