//! Rigid-frame math for the structure module. Quaternion-based backbone frames
//! (no eigendecomposition on the inference path) + 3x3 rotation-matrix frames for
//! the all-atom reconstruction. All formulas mirror openfold rigid_utils exactly.

/// Quaternion (a,b,c,d) -> 3x3 rotation matrix (row-major [9]). Assumes unit quat.
pub fn quat_to_rot(q: &[f32; 4]) -> [f32; 9] {
    let (a, b, c, d) = (q[0], q[1], q[2], q[3]);
    [
        a * a + b * b - c * c - d * d,
        2.0 * b * c - 2.0 * a * d,
        2.0 * b * d + 2.0 * a * c,
        2.0 * b * c + 2.0 * a * d,
        a * a - b * b + c * c - d * d,
        2.0 * c * d - 2.0 * a * b,
        2.0 * b * d - 2.0 * a * c,
        2.0 * c * d + 2.0 * a * b,
        a * a - b * b - c * c + d * d,
    ]
}

#[inline]
pub fn rot_vec_mul(r: &[f32; 9], v: [f32; 3]) -> [f32; 3] {
    [
        r[0] * v[0] + r[1] * v[1] + r[2] * v[2],
        r[3] * v[0] + r[4] * v[1] + r[5] * v[2],
        r[6] * v[0] + r[7] * v[1] + r[8] * v[2],
    ]
}

pub fn rot_matmul(a: &[f32; 9], b: &[f32; 9]) -> [f32; 9] {
    let mut o = [0.0f32; 9];
    for i in 0..3 {
        for j in 0..3 {
            o[i * 3 + j] = a[i * 3] * b[j] + a[i * 3 + 1] * b[3 + j] + a[i * 3 + 2] * b[6 + j];
        }
    }
    o
}

pub fn rot_transpose(r: &[f32; 9]) -> [f32; 9] {
    [r[0], r[3], r[6], r[1], r[4], r[7], r[2], r[5], r[8]]
}

/// quaternion * pure-vector quaternion (0,x,y,z), openfold convention.
fn quat_mul_by_vec(q: &[f32; 4], v: [f32; 3]) -> [f32; 4] {
    let (a, b, c, d) = (q[0], q[1], q[2], q[3]);
    let (x, y, z) = (v[0], v[1], v[2]);
    [
        -(b * x + c * y + d * z),
        a * x + c * z - d * y,
        a * y - b * z + d * x,
        a * z + b * y - c * x,
    ]
}

/// Rigid.compose_q_update_vec: update quats (normalized) and trans in place.
pub fn compose_q_update(quat: &mut [f32; 4], trans: &mut [f32; 3], update: &[f32; 6]) {
    let q_vec = [update[0], update[1], update[2]];
    let t_vec = [update[3], update[4], update[5]];
    // translation uses the OLD rotation
    let old_rot = quat_to_rot(quat);
    let tu = rot_vec_mul(&old_rot, t_vec);
    trans[0] += tu[0];
    trans[1] += tu[1];
    trans[2] += tu[2];
    // rotation update
    let upd = quat_mul_by_vec(quat, q_vec);
    let mut nq = [quat[0] + upd[0], quat[1] + upd[1], quat[2] + upd[2], quat[3] + upd[3]];
    let norm = (nq[0] * nq[0] + nq[1] * nq[1] + nq[2] * nq[2] + nq[3] * nq[3]).sqrt();
    for x in nq.iter_mut() {
        *x /= norm;
    }
    *quat = nq;
}

/// A rotation-matrix frame (rot [9] row-major, trans [3]).
#[derive(Clone, Copy)]
pub struct Frame {
    pub rot: [f32; 9],
    pub trans: [f32; 3],
}

impl Frame {
    pub fn identity() -> Self {
        Frame { rot: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], trans: [0.0; 3] }
    }

    /// From a 4x4 homogeneous matrix (row-major [16]).
    pub fn from_4x4(m: &[f32]) -> Self {
        Frame {
            rot: [m[0], m[1], m[2], m[4], m[5], m[6], m[8], m[9], m[10]],
            trans: [m[3], m[7], m[11]],
        }
    }

    /// self ∘ other (compose): rot = self.rot @ other.rot; trans = self.rot @ other.trans + self.trans.
    pub fn compose(&self, other: &Frame) -> Frame {
        let rot = rot_matmul(&self.rot, &other.rot);
        let t = rot_vec_mul(&self.rot, other.trans);
        Frame { rot, trans: [t[0] + self.trans[0], t[1] + self.trans[1], t[2] + self.trans[2]] }
    }

    #[inline]
    pub fn apply(&self, p: [f32; 3]) -> [f32; 3] {
        let r = rot_vec_mul(&self.rot, p);
        [r[0] + self.trans[0], r[1] + self.trans[1], r[2] + self.trans[2]]
    }
}
