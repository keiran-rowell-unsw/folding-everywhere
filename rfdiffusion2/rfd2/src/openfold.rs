//! `compute_backbone` — openfold's idealized backbone from a rigid frame.
//!
//! `frame_diffusion/data/all_atom.py:compute_backbone` turns one rigid per
//! token plus a psi torsion into atom14/atom37 coordinates, by way of
//! `openfold.utils.feats.torsion_angles_to_frames` and
//! `all_atom.frames_to_atom14_pos`. It is the last geometric step of `diffuse`
//! and it is also how the sampler turns the network's predicted frames into
//! `px0`, so it is ported once and used from both.
//!
//! ## Three things that decide whether this is right
//!
//! 1. **`aatype` is always zero.** `compute_backbone` writes
//!    `aatype = torch.zeros(bb_rigids.shape).long()`, so every token — protein
//!    or ligand — gets *alanine's* frames. For ALA, `group_idx14` is
//!    `[0,0,0,3,0,...]` and `atom_mask14` is `[1,1,1,1,1,0,...]`: N, CA, C and
//!    CB ride the backbone frame, O rides the psi frame, and the rest are
//!    masked to zero. The port keeps all 21 rows of every table so a future
//!    configuration that passes a real sequence cannot silently reuse row 0.
//!
//! 2. **`torch.sum` is f64-pinned; the surrounding arithmetic is not.**
//!    `frames_to_atom14_pos` selects a frame per atom by multiplying by a
//!    one-hot mask and *summing* the 8 products. That sum goes through the
//!    pinned f64 path, while `rot_matmul` and `rot_vec_mul` are written out by
//!    hand upstream (to dodge AMP downcasting) and stay genuinely fp32.
//!    Getting that split backwards is invisible at any tolerance and wrong.
//!
//! 3. **The masked-out frames are not free.** `0.0 * x` is `-0.0` when `x` is
//!    negative, and a sum that begins `-0.0 + -0.0` and reaches a `+0.0`
//!    selected value returns `+0.0`. The port therefore performs the multiply
//!    and the 8-term sum literally rather than indexing the selected frame —
//!    the same class of defect as the autograd `+0.0`/`-0.0` finding in
//!    `results/README.md`.
//!
//! Validated by `tests/parity_backbone.rs` against the reference's own captured
//! rigids and psi, at tolerance **exactly 0**.

use crate::nn::Ctx;
use crate::noiser::Rigids;
use crate::weights::Weights;
use std::sync::OnceLock;

/// The four AF2 residue-constant tables, exported by `python/gen_openfold.py`.
static BLOB: &[u8] = include_bytes!("../data/openfold.safetensors");

static STORE: OnceLock<Tables> = OnceLock::new();

pub struct Tables {
    /// `[21, 8, 4, 4]` — `restype_rigid_group_default_frame`
    pub default_frames: Vec<f32>,
    /// `[21, 14, 3]` — `restype_atom14_rigid_group_positions`
    pub idealized_pos14: Vec<f32>,
    /// `[21, 14]` — `restype_atom14_mask`
    pub atom_mask14: Vec<f32>,
    /// `[21, 14]` — `restype_atom14_to_rigid_group`
    pub group_idx14: Vec<i64>,
    /// `[4, 4, 3, 3]` — `rigid_utils._QTR_MAT`, the quaternion-to-matrix table
    pub qtr_mat: Vec<f32>,
    /// `[4, 4, 4]` — `rigid_utils._QUAT_MULTIPLY`
    pub quat_multiply: Vec<f32>,
}

/// Number of rigid groups per residue (backbone, omega, phi, psi, chi1..chi4).
pub const N_GROUPS: usize = 8;
/// Atoms in the atom14 representation.
pub const N_ATOM14: usize = 14;
/// Atoms in the atom37 representation.
pub const N_ATOM37: usize = 37;

pub fn tables() -> &'static Tables {
    STORE.get_or_init(|| {
        let w = Weights::from_static(BLOB).expect("embedded openfold.safetensors is corrupt");
        Tables {
            default_frames: w.get("default_frames").data,
            idealized_pos14: w.get("idealized_pos14").data,
            atom_mask14: w.get("atom_mask14").data,
            group_idx14: w.get_i64("group_idx14").0,
            qtr_mat: w.get("qtr_mat").data,
            quat_multiply: w.get("quat_multiply").data,
        }
    })
}

/// `openfold.rigid_utils.rot_matmul` — hand-written upstream, plain fp32.
#[inline]
fn rot_matmul(a: &[f32; 9], b: &[f32; 9]) -> [f32; 9] {
    let mut o = [0.0f32; 9];
    for i in 0..3 {
        for j in 0..3 {
            o[i * 3 + j] =
                a[i * 3] * b[j] + a[i * 3 + 1] * b[3 + j] + a[i * 3 + 2] * b[6 + j];
        }
    }
    o
}

/// `openfold.rigid_utils.rot_vec_mul` — hand-written upstream, plain fp32.
#[inline]
fn rot_vec_mul(r: &[f32; 9], t: &[f32; 3]) -> [f32; 3] {
    [
        r[0] * t[0] + r[1] * t[1] + r[2] * t[2],
        r[3] * t[0] + r[4] * t[1] + r[5] * t[2],
        r[6] * t[0] + r[7] * t[1] + r[8] * t[2],
    ]
}

/// One rigid transform: row-major `[3,3]` rotation and a translation.
#[derive(Clone, Copy, Debug)]
struct Rt {
    r: [f32; 9],
    t: [f32; 3],
}

impl Rt {
    /// `Rigid.compose` — `rot = self.r @ o.r`, `trans = self.r @ o.t + self.t`.
    #[inline]
    fn compose(&self, o: &Rt) -> Rt {
        let rv = rot_vec_mul(&self.r, &o.t);
        Rt {
            r: rot_matmul(&self.r, &o.r),
            t: [rv[0] + self.t[0], rv[1] + self.t[1], rv[2] + self.t[2]],
        }
    }

    /// `Rigid.apply` — rotate then translate a point.
    #[inline]
    fn apply(&self, p: &[f32; 3]) -> [f32; 3] {
        let rv = rot_vec_mul(&self.r, p);
        [rv[0] + self.t[0], rv[1] + self.t[1], rv[2] + self.t[2]]
    }
}

/// `openfold.utils.feats.torsion_angles_to_frames`, specialised to the tiled
/// psi that `compute_backbone` builds.
///
/// `alpha[0]` is the fixed `[0, 1]` backbone rotation (identity) and
/// `alpha[1..8]` are all the same psi pair, because `compute_backbone` tiles it
/// across all 7 torsion slots. The chi2..chi4 frames are still composed down
/// the chain exactly as upstream does; for `aatype = 0` nothing reads them, but
/// composing them is free and keeps this function correct for a real sequence.
fn torsion_angles_to_frames(bb: &Rt, psi: [f32; 2], aatype: usize) -> [Rt; N_GROUPS] {
    let tb = tables();
    let base = aatype * N_GROUPS * 16;

    // `default_r = Rigid.from_tensor_4x4(DEFAULT_FRAMES[aatype])`
    let mut all_frames = [Rt { r: [0.0; 9], t: [0.0; 3] }; N_GROUPS];
    for k in 0..N_GROUPS {
        let m = &tb.default_frames[base + k * 16..base + k * 16 + 16];
        let default = Rt {
            r: [m[0], m[1], m[2], m[4], m[5], m[6], m[8], m[9], m[10]],
            t: [m[3], m[7], m[11]],
        };
        // `alpha[..., 0]` is the backbone slot's fixed [0, 1]; every torsion
        // slot carries the same psi.
        let (a1, a2) = if k == 0 { (0.0f32, 1.0f32) } else { (psi[0], psi[1]) };
        // [[1, 0, 0], [0, a2, -a1], [0, a1, a2]]
        let upd = Rt {
            r: [1.0, 0.0, 0.0, 0.0, a2, -a1, 0.0, a1, a2],
            t: [0.0, 0.0, 0.0],
        };
        all_frames[k] = default.compose(&upd);
    }

    // chi2/3/4 are expressed relative to the previous chi frame upstream, so
    // they are composed down the chain before anything uses them.
    let mut out = all_frames;
    out[5] = all_frames[4].compose(&all_frames[5]);
    out[6] = out[5].compose(&all_frames[6]);
    out[7] = out[6].compose(&all_frames[7]);

    // `all_frames_to_global = r[..., None].compose(all_frames_to_bb)`
    for f in out.iter_mut() {
        *f = bb.compose(f);
    }
    out
}

/// `all_atom.frames_to_atom14_pos`.
///
/// The frame selection is a one-hot multiply and an 8-term sum, and the sum is
/// f64-pinned — see the module header for why that is not interchangeable with
/// indexing the selected frame.
fn frames_to_atom14_pos(frames: &[Rt; N_GROUPS], aatype: usize) -> [[f32; 3]; N_ATOM14] {
    let tb = tables();
    let mut out = [[0.0f32; 3]; N_ATOM14];
    for a in 0..N_ATOM14 {
        let g = tb.group_idx14[aatype * N_ATOM14 + a] as usize;

        // `r[..., None, :] * group_mask` then `sum(dim=-1)`: the products are
        // formed in fp32 and accumulated in f64 with one narrowing.
        let mut r = [0.0f32; 9];
        let mut t = [0.0f32; 3];
        for (c, rc) in r.iter_mut().enumerate() {
            let mut acc = 0.0f64;
            for (k, f) in frames.iter().enumerate() {
                let mask = if k == g { 1.0f32 } else { 0.0f32 };
                acc += (f.r[c] * mask) as f64;
            }
            *rc = acc as f32;
        }
        for (c, tc) in t.iter_mut().enumerate() {
            let mut acc = 0.0f64;
            for (k, f) in frames.iter().enumerate() {
                let mask = if k == g { 1.0f32 } else { 0.0f32 };
                acc += (f.t[c] * mask) as f64;
            }
            *tc = acc as f32;
        }

        let lit = &tb.idealized_pos14[(aatype * N_ATOM14 + a) * 3..][..3];
        let p = Rt { r, t }.apply(&[lit[0], lit[1], lit[2]]);
        let m = tb.atom_mask14[aatype * N_ATOM14 + a];
        out[a] = [p[0] * m, p[1] * m, p[2] * m];
    }
    out
}

/// `all_atom.compute_backbone` — returns `(atom37, atom14)`, both flattened.
///
/// `atom37` carries only the five backbone atoms; slots 5..37 stay zero, which
/// is what `diffuse` then copies into `indep.xyz`.
pub fn compute_backbone(rigids: &Rigids, psi: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let l = rigids.len();
    assert_eq!(psi.len(), l * 2, "compute_backbone: psi must be [L, 2]");
    let mut atom37 = vec![0.0f32; l * N_ATOM37 * 3];
    let mut atom14 = vec![0.0f32; l * N_ATOM14 * 3];
    for i in 0..l {
        let bb = Rt {
            r: rigids.rots[i * 9..i * 9 + 9].try_into().unwrap(),
            t: rigids.trans[i * 3..i * 3 + 3].try_into().unwrap(),
        };
        let frames = torsion_angles_to_frames(&bb, [psi[i * 2], psi[i * 2 + 1]], 0);
        let pos = frames_to_atom14_pos(&frames, 0);
        for a in 0..N_ATOM14 {
            for c in 0..3 {
                atom14[(i * N_ATOM14 + a) * 3 + c] = pos[a][c];
            }
        }
        for a in 0..5 {
            for c in 0..3 {
                atom37[(i * N_ATOM37 + a) * 3 + c] = pos[a][c];
            }
        }
    }
    (atom37, atom14)
}

/// `all_atom.atom37_from_rigid` — **draws `psi_pred` first**, then builds.
///
/// The draw is the reason this cannot be treated as a pure function of the
/// rigids: it is draw 5 and draw 8 of the nine `sample_init` makes, and the
/// sampler takes one per step. Skipping it shifts every later draw in the
/// stream, which is a failure no coordinate comparison can localise.
pub fn atom37_from_rigid(rigids: &Rigids, ctx: &mut Ctx) -> Vec<f32> {
    let l = rigids.len();
    let psi: Vec<f32> = (0..l * 2).map(|_| ctx.rng.uniform_f32()).collect();
    compute_backbone(rigids, &psi).0
}
