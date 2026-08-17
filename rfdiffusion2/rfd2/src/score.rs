//! `frame_diffusion/rf_score/model.py:RFScore.forward_from_rfi` — the layer
//! between the network and the sampler.
//!
//! The trunk returns per-block quaternion *updates*; nobody downstream uses
//! them directly. This wrapper composes them onto the input frame, draws a psi
//! torsion, and builds the idealized backbone — and **that** is `px0`, the
//! structure the sampler writes out. No document described this layer; it was
//! found by reading the sampler's call chain.
//!
//! ```text
//! rigids_t    = rigid_frames_from_atom_14(rfi.xyz)     src/noiser.rs
//! rfo         = RoseTTAFold::forward(rfi)              src/model/rf.rs
//! curr_rigids = rigids_from_rfo(rfo, rigids_t.rots)    <- 40 quaternion composes
//! psi_pred    = rand(1, I, L, 2)                       <- one draw per forward
//! atom37      = compute_backbone(curr_rigids, psi)     src/openfold.rs
//! px0         = atom37[0, -1]
//! ```
//!
//! ## The quaternion path is where the precision risk lives
//!
//! `Rotation.get_quats()` on a matrix-backed rotation calls openfold's
//! `rot_to_quat`, which assembles a symmetric 4x4 and takes the **last
//! eigenvector of `torch.linalg.eigh`**. Under pinning that is LAPACK in f64
//! with a single narrowing to fp32, so the port needs an f64 symmetric
//! eigensolver whose answer agrees to well under an fp32 ULP — the same
//! argument that makes the pinned GEMM reproducible, applied to an
//! eigendecomposition. A canonical cyclic Jacobi is used here, mirroring what
//! `python/pinned.py` already does for the 3x3 Kabsch SVD.
//!
//! The eigenvector's **sign is arbitrary** and does not need to match. Every
//! consumer is even in the quaternion: `quat_to_rot` is a product of two
//! components, `quat_multiply` is bilinear and IEEE addition is
//! sign-symmetric, so negating `q` reproduces the identical bit pattern
//! downstream. `tests/parity_score.rs` asserts that empirically rather than
//! taking it on faith.
//!
//! ## `normalize_quats` fires more than once, and is not idempotent
//!
//! `Rotation.__init__` divides by `torch.linalg.norm` whenever it is handed
//! quaternions, and the composition path constructs `Rotation` four times per
//! block. The norm of an already-normalised quaternion is 1 only to within an
//! ULP, so each re-normalisation changes bits. Skipping the redundant ones
//! gives a plausible frame that is off in the last place.

use crate::model::rf::{ModelOut, Rfi, RoseTTAFold};
use crate::nn::Ctx;
use crate::noiser::{rigid_frames_from_atom_14, Rigids};
use crate::openfold::{compute_backbone, tables, N_ATOM37};
use crate::chemical_gen::NTOTAL;

/// `openfold.rigid_utils.Rotation.__init__(normalize_quats=True)` — divide by
/// the pinned `torch.linalg.norm`.
#[inline]
fn normalize_quat(q: [f32; 4]) -> [f32; 4] {
    let mut acc = 0.0f64;
    for v in q {
        acc += v as f64 * v as f64;
    }
    let n = (acc.sqrt()) as f32;
    [q[0] / n, q[1] / n, q[2] / n, q[3] / n]
}

/// Canonical cyclic Jacobi eigendecomposition of a symmetric 4x4, in f64.
///
/// Returns eigenvalues and eigenvectors as **columns**, sorted ascending, which
/// is LAPACK's `eigh` convention — `rot_to_quat` takes the last column.
///
/// Written out rather than delegated because the whole point is that both
/// sides run the *same* algorithm in f64: two different f64 eigensolvers agree
/// to ~1e-16 relative, which is nine orders below an fp32 ULP, so the narrowed
/// fp32 answer is the correctly-rounded one either way.
fn jacobi_eigh4(a_in: [[f64; 4]; 4]) -> ([f64; 4], [[f64; 4]; 4]) {
    let mut a = a_in;
    let mut v = [[0.0f64; 4]; 4];
    for (i, row) in v.iter_mut().enumerate() {
        row[i] = 1.0;
    }
    for _ in 0..128 {
        // largest off-diagonal magnitude
        let (mut p, mut q, mut off) = (0usize, 1usize, 0.0f64);
        for i in 0..4 {
            for j in (i + 1)..4 {
                if a[i][j].abs() > off {
                    p = i;
                    q = j;
                    off = a[i][j].abs();
                }
            }
        }
        if off < 1e-300 {
            break;
        }
        let theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q]);
        let sign = if theta < 0.0 {
            -1.0
        } else if theta > 0.0 {
            1.0
        } else {
            0.0
        };
        let t = sign / (theta.abs() + (theta * theta + 1.0).sqrt());
        let c = 1.0 / (t * t + 1.0).sqrt();
        let s = t * c;

        let b = {
            let mut b = a;
            for k in 0..4 {
                b[k][p] = c * a[k][p] - s * a[k][q];
                b[k][q] = s * a[k][p] + c * a[k][q];
            }
            b
        };
        let mut a2 = b;
        for k in 0..4 {
            a2[p][k] = c * b[p][k] - s * b[q][k];
            a2[q][k] = s * b[p][k] + c * b[q][k];
        }
        a2[p][q] = 0.0;
        a2[q][p] = 0.0;
        a = a2;

        let mut v2 = v;
        for k in 0..4 {
            v2[k][p] = c * v[k][p] - s * v[k][q];
            v2[k][q] = s * v[k][p] + c * v[k][q];
        }
        v = v2;
    }
    let mut eig = [0.0f64; 4];
    for i in 0..4 {
        eig[i] = a[i][i];
    }
    // ascending, as LAPACK returns them
    let mut order = [0usize, 1, 2, 3];
    order.sort_by(|x, y| eig[*x].partial_cmp(&eig[*y]).unwrap());
    let mut vals = [0.0f64; 4];
    let mut vecs = [[0.0f64; 4]; 4];
    for (col, &src) in order.iter().enumerate() {
        vals[col] = eig[src];
        for row in 0..4 {
            vecs[row][col] = v[row][src];
        }
    }
    (vals, vecs)
}

/// `openfold.rigid_utils.rot_to_quat`.
///
/// The 4x4 is assembled in fp32 (plain adds and one scalar multiply), then
/// `eigh` runs f64-pinned and the last eigenvector is narrowed once.
pub fn rot_to_quat(r: &[f32; 9]) -> [f32; 4] {
    let (xx, xy, xz) = (r[0], r[1], r[2]);
    let (yx, yy, yz) = (r[3], r[4], r[5]);
    let (zx, zy, zz) = (r[6], r[7], r[8]);
    let k32 = [
        [xx + yy + zz, zy - yz, xz - zx, yx - xy],
        [zy - yz, xx - yy - zz, xy + yx, xz + zx],
        [xz - zx, xy + yx, yy - xx - zz, yz + zy],
        [yx - xy, xz + zx, yz + zy, zz - xx - yy],
    ];
    let third = 1.0f32 / 3.0;
    let mut k = [[0.0f64; 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            k[i][j] = (third * k32[i][j]) as f64;
        }
    }
    let (_, vecs) = jacobi_eigh4(k);
    [
        vecs[0][3] as f32,
        vecs[1][3] as f32,
        vecs[2][3] as f32,
        vecs[3][3] as f32,
    ]
}

/// `openfold.rigid_utils.quat_to_rot` — a masked 16-term sum against `_QTR_MAT`,
/// pinned, so the accumulation runs in f64 with one narrowing.
pub fn quat_to_rot(q: &[f32; 4]) -> [f32; 9] {
    let m = &tables().qtr_mat; // [4, 4, 3, 3]
    let mut out = [0.0f32; 9];
    for r in 0..3 {
        for c in 0..3 {
            let mut acc = 0.0f64;
            for i in 0..4 {
                for j in 0..4 {
                    let outer = q[i] * q[j];
                    acc += (outer * m[((i * 4 + j) * 3 + r) * 3 + c]) as f64;
                }
            }
            out[r * 3 + c] = acc as f32;
        }
    }
    out
}

/// `openfold.rigid_utils.quat_multiply` — likewise a pinned 16-term sum.
pub fn quat_multiply(q1: &[f32; 4], q2: &[f32; 4]) -> [f32; 4] {
    let m = &tables().quat_multiply; // [4, 4, 4]
    let mut out = [0.0f32; 4];
    for k in 0..4 {
        let mut acc = 0.0f64;
        for i in 0..4 {
            for j in 0..4 {
                acc += (m[(i * 4 + j) * 4 + k] * q1[i] * q2[j]) as f64;
            }
        }
        out[k] = acc as f32;
    }
    out
}

/// `rf_score/model.py:rigids_from_rfo`.
///
/// Composes the per-block quaternion updates onto the input frame, left-
/// multiplying: `curr = q_block ⊗ curr`. The translation is taken straight
/// from the block's CA, not composed.
///
/// `quat_stack` is `[I][L * 4]` and `xyz_stack` is `[I][L * 3 * 3]`.
pub fn rigids_from_rfo(
    quat_stack: &[Vec<f32>],
    xyz_stack: &[Vec<f32>],
    rots_t: &[f32],
    l: usize,
) -> Vec<Rigids> {
    let n_iter = quat_stack.len();
    assert_eq!(xyz_stack.len(), n_iter);
    // `Rotation(quats=rots_t.get_quats())` — convert, then normalise
    let mut curr: Vec<[f32; 4]> = (0..l)
        .map(|i| {
            let r: [f32; 9] = rots_t[i * 9..i * 9 + 9].try_into().unwrap();
            normalize_quat(rot_to_quat(&r))
        })
        .collect();

    let mut out = Vec::with_capacity(n_iter);
    for it in 0..n_iter {
        let mut rots = vec![0.0f32; l * 9];
        let mut trans = vec![0.0f32; l * 3];
        for i in 0..l {
            let qb: [f32; 4] = quat_stack[it][i * 4..i * 4 + 4].try_into().unwrap();
            // `Rotation(quats=rfo.quat[:, i]).compose_q(curr_rots)`: both sides
            // are normalised on construction, and the product is normalised
            // again by the Rotation `compose_q` returns.
            let composed = normalize_quat(quat_multiply(&normalize_quat(qb), &curr[i]));
            curr[i] = composed;
            // `Rotation(quats=rot_blocks)` normalises the whole stack once more
            // before anything reads a matrix out of it.
            let r = quat_to_rot(&normalize_quat(composed));
            rots[i * 9..i * 9 + 9].copy_from_slice(&r);
            // c_alpha = xyz_stack[..., 1, :]
            let o = (i * 3 + 1) * 3;
            trans[i * 3..i * 3 + 3].copy_from_slice(&xyz_stack[it][o..o + 3]);
        }
        out.push(Rigids { rots, trans });
    }
    out
}

/// What `forward_from_rfi` hands the sampler.
pub struct ScoreOut {
    /// `[I][L * 37 * 3]` — the idealized backbone per refinement block
    pub atom37: Vec<Vec<f32>>,
    /// `[I]` composed frames
    pub rigids: Vec<Rigids>,
    /// The network's own output, kept because the sampler returns `rfo`.
    pub model: ModelOut,
}

impl ScoreOut {
    /// `model_out['atom37'][0, -1]` — the last block's backbone, i.e. `px0`.
    pub fn px0(&self) -> &[f32] {
        self.atom37.last().expect("no refinement blocks")
    }

    /// `model_out['rigids_raw'][:, -1]` — the frame the sampler steps from.
    pub fn rigids_pred(&self) -> &Rigids {
        self.rigids.last().expect("no refinement blocks")
    }
}

/// `RFScore.forward_from_rfi(rfi, t)`.
///
/// The psi draw is `rand(1, I, L, 2)` — **I × L × 2 values in one call**, not
/// one draw per block. Drawing per block would consume the same count and land
/// the generator in the same place while producing a different assignment of
/// values to blocks.
pub fn forward_from_rfi(model: &RoseTTAFold, rfi: &Rfi, ctx: &mut Ctx) -> ScoreOut {
    let l = rfi.seq.len();
    let (rots_t, _trans_t) = rigid_frames_from_atom_14(&rfi.xyz.data, l, NTOTAL);
    let out = model.forward(rfi, ctx);
    let rigids = rigids_from_rfo(&out.sim.quat_stack, &out.sim.xyz_stack, &rots_t, l);
    let n_iter = rigids.len();

    let psi: Vec<f32> = (0..n_iter * l * 2).map(|_| ctx.rng.uniform_f32()).collect();
    let mut atom37 = Vec::with_capacity(n_iter);
    for (it, r) in rigids.iter().enumerate() {
        let slice = &psi[it * l * 2..(it + 1) * l * 2];
        atom37.push(compute_backbone(r, slice).0);
    }
    debug_assert_eq!(atom37[0].len(), l * N_ATOM37 * 3);
    ScoreOut { atom37, rigids, model: out }
}
