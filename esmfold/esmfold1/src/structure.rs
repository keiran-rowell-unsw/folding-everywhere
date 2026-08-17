//! Structure module: IPA + backbone-frame update + angle resnet + all-atom
//! reconstruction (torsion->frames, frames+literature->atom14). 8 shared
//! iterations. Mirrors EsmFoldStructureModule exactly.

use crate::constants::Constants;
use crate::ops;
use crate::rigid::{self, Frame};
use crate::tensor::Tensor;
use crate::weights::Weights;
use rayon::prelude::*;

const C_S: usize = 384;
const C_Z: usize = 128;
const IPA_DIM: usize = 16;
const H: usize = 12;
const QK_PTS: usize = 4;
const V_PTS: usize = 8;
const NUM_ITERS: usize = 8;
const EPS: f32 = 1e-8;
const TRANS_SCALE: f32 = 10.0;
const SM: &str = "trunk.structure_module";

fn ln(x: &Tensor, w: &Weights, p: &str) -> Tensor {
    ops::layer_norm(x, &w.get(&format!("{p}.weight")), &w.get(&format!("{p}.bias")), 1e-5)
}
fn lin(x: &Tensor, w: &Weights, p: &str) -> Tensor {
    ops::linear(x, &w.get(&format!("{p}.weight")), Some(&w.get(&format!("{p}.bias"))))
}

pub struct StructIter {
    pub frames7: Vec<f32>,   // [N,7] quat(4)+trans*scale(3)
    pub positions: Vec<f32>, // [N,14,3]
    pub states: Vec<f32>,    // [N,384]
}

/// Invariant Point Attention. quat/trans are the current backbone frames (unscaled).
fn ipa(s: &Tensor, z: &Tensor, quat: &[[f32; 4]], trans: &[[f32; 3]], w: &Weights, n: usize) -> Tensor {
    let q = lin(s, w, &format!("{SM}.ipa.linear_q")).data; // [N,192]
    let kv = lin(s, w, &format!("{SM}.ipa.linear_kv")).data; // [N,384]
    let qp = lin(s, w, &format!("{SM}.ipa.linear_q_points")).data; // [N,144]
    let kvp = lin(s, w, &format!("{SM}.ipa.linear_kv_points")).data; // [N,432]
    let b = lin(z, w, &format!("{SM}.ipa.linear_b")).data; // [N,N,12]
    let hw_raw = w.get(&format!("{SM}.ipa.head_weights")).data; // [12]

    let rots: Vec<[f32; 9]> = quat.iter().map(rigid::quat_to_rot).collect();
    // global points
    let mut qg = vec![[0.0f32; 3]; n * H * QK_PTS];
    let mut kg = vec![[0.0f32; 3]; n * H * QK_PTS];
    let mut vg = vec![[0.0f32; 3]; n * H * V_PTS];
    for i in 0..n {
        let (r, t) = (&rots[i], &trans[i]);
        for h in 0..H {
            for p in 0..QK_PTS {
                let idx = h * QK_PTS + p;
                let loc = [qp[i * 144 + idx], qp[i * 144 + 48 + idx], qp[i * 144 + 96 + idx]];
                let g = rigid::rot_vec_mul(r, loc);
                qg[(i * H + h) * QK_PTS + p] = [g[0] + t[0], g[1] + t[1], g[2] + t[2]];
            }
            for m in 0..(QK_PTS + V_PTS) {
                let idx = h * (QK_PTS + V_PTS) + m;
                let loc = [kvp[i * 432 + idx], kvp[i * 432 + 144 + idx], kvp[i * 432 + 288 + idx]];
                let g = rigid::rot_vec_mul(r, loc);
                let gp = [g[0] + t[0], g[1] + t[1], g[2] + t[2]];
                if m < QK_PTS {
                    kg[(i * H + h) * QK_PTS + m] = gp;
                } else {
                    vg[(i * H + h) * V_PTS + (m - QK_PTS)] = gp;
                }
            }
        }
    }
    let hw: Vec<f32> = hw_raw.iter().map(|&x| ops::softplus_scalar(x) * (1.0f32 / 54.0).sqrt()).collect();
    let cq = (1.0f32 / 48.0).sqrt();
    let cb = (1.0f32 / 3.0).sqrt();

    // output buffer [N, 2112]
    let out_dim = H * (C_Z + IPA_DIM + V_PTS * 4); // 2112
    let mut concat = vec![0.0f32; n * out_dim];
    concat.par_chunks_mut(out_dim).enumerate().for_each(|(i, crow)| {
        for h in 0..H {
            // logits over j
            let mut a = vec![0.0f32; n];
            for j in 0..n {
                let mut sc = 0.0f32;
                for c in 0..IPA_DIM {
                    sc += q[i * 192 + h * IPA_DIM + c] * kv[j * 384 + h * 32 + c];
                }
                let mut ptsum = 0.0f32;
                for p in 0..QK_PTS {
                    let a0 = qg[(i * H + h) * QK_PTS + p];
                    let b0 = kg[(j * H + h) * QK_PTS + p];
                    let dx = a0[0] - b0[0];
                    let dy = a0[1] - b0[1];
                    let dz = a0[2] - b0[2];
                    ptsum += dx * dx + dy * dy + dz * dz;
                }
                let pt = -0.5 * hw[h] * ptsum;
                a[j] = cq * sc + cb * b[(i * n + j) * H + h] + pt;
            }
            // softmax over j
            let m = a.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            for x in a.iter_mut() {
                *x = libm::expf(*x - m);
                sum += *x;
            }
            let inv = 1.0 / sum;
            for x in a.iter_mut() {
                *x *= inv;
            }
            // o (scalar) [16] -> concat[h*16 .. ]
            for c in 0..IPA_DIM {
                let mut acc = 0.0f32;
                for j in 0..n {
                    acc += a[j] * kv[j * 384 + h * 32 + IPA_DIM + c];
                }
                crow[h * IPA_DIM + c] = acc;
            }
            // o_pt (points), invert_apply by frame i, + norm
            let rt = rigid::rot_transpose(&rots[i]);
            let base_pt = H * IPA_DIM; // 192
            let base_normsec = base_pt + H * V_PTS * 3; // after x,y,z blocks (192+288=480)
            for p in 0..V_PTS {
                let mut g = [0.0f32; 3];
                for j in 0..n {
                    let v = vg[(j * H + h) * V_PTS + p];
                    g[0] += a[j] * v[0];
                    g[1] += a[j] * v[1];
                    g[2] += a[j] * v[2];
                }
                // invert_apply: R_i^T (g - t_i)
                let d = [g[0] - trans[i][0], g[1] - trans[i][1], g[2] - trans[i][2]];
                let loc = rigid::rot_vec_mul(&rt, d);
                let hp = h * V_PTS + p; // 0..96
                crow[base_pt + hp] = loc[0]; // x block
                crow[base_pt + H * V_PTS + hp] = loc[1]; // y block
                crow[base_pt + 2 * H * V_PTS + hp] = loc[2]; // z block
                crow[base_normsec + hp] = (loc[0] * loc[0] + loc[1] * loc[1] + loc[2] * loc[2] + EPS).sqrt();
            }
            // o_pair [128] -> after norm block (480+96=576)
            let base_pair = base_normsec + H * V_PTS; // 576
            for c in 0..C_Z {
                let mut acc = 0.0f32;
                for j in 0..n {
                    acc += a[j] * z.data[(i * n + j) * C_Z + c];
                }
                crow[base_pair + h * C_Z + c] = acc;
            }
        }
    });
    let concat_t = Tensor::new(concat, vec![n, out_dim]);
    lin(&concat_t, w, &format!("{SM}.ipa.linear_out"))
}

fn angle_resnet(s: &Tensor, s_initial: &Tensor, w: &Weights, n: usize) -> Tensor {
    let p = format!("{SM}.angle_resnet");
    let si = lin(&ops::relu(s_initial), w, &format!("{p}.linear_initial"));
    let mut h = lin(&ops::relu(s), w, &format!("{p}.linear_in"));
    h = Tensor::new(h.data.iter().zip(&si.data).map(|(a, b)| a + b).collect(), h.shape.clone());
    for layer in 0..2 {
        let res = h.clone();
        let t = lin(&ops::relu(&h), w, &format!("{p}.layers.{layer}.linear_1"));
        let t = lin(&ops::relu(&t), w, &format!("{p}.layers.{layer}.linear_2"));
        h = Tensor::new(res.data.iter().zip(&t.data).map(|(a, b)| a + b).collect(), h.shape.clone());
    }
    let h = lin(&ops::relu(&h), w, &format!("{p}.linear_out")); // [N,14]
    // normalize (s,c) pairs
    let mut ang = vec![0.0f32; n * 14];
    for nn in 0..n {
        for k in 0..7 {
            let s0 = h.data[nn * 14 + k * 2];
            let s1 = h.data[nn * 14 + k * 2 + 1];
            let denom = (s0 * s0 + s1 * s1).max(EPS).sqrt();
            ang[nn * 14 + k * 2] = s0 / denom;
            ang[nn * 14 + k * 2 + 1] = s1 / denom;
        }
    }
    Tensor::new(ang, vec![n, 7, 2])
}

fn transition(s: &Tensor, w: &Weights) -> Tensor {
    let p = format!("{SM}.transition");
    let t = lin(s, w, &format!("{p}.layers.0.linear_1"));
    let t = ops::relu(&t);
    let t = lin(&t, w, &format!("{p}.layers.0.linear_2"));
    let t = ops::relu(&t);
    let t = lin(&t, w, &format!("{p}.layers.0.linear_3"));
    let s2 = Tensor::new(s.data.iter().zip(&t.data).map(|(a, b)| a + b).collect(), s.shape.clone());
    ln(&s2, w, &format!("{p}.layer_norm"))
}

/// torsion_angles_to_frames -> per-residue 8 global frames.
fn torsion_to_frames(backb: &[Frame], angles: &Tensor, aatype: &[usize], c: &Constants, n: usize) -> Vec<[Frame; 8]> {
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let a = aatype[i];
        // default frames [8]
        let mut default = [Frame::identity(); 8];
        for g in 0..8 {
            let base = ((a * 8 + g) * 16) as usize;
            default[g] = Frame::from_4x4(&c.default_frames[base..base + 16]);
        }
        // alpha: [bb (0,1)] ++ angles[i] (7) = 8
        let mut all_frames = [Frame::identity(); 8];
        for g in 0..8 {
            let (a0, a1) = if g == 0 {
                (0.0f32, 1.0f32)
            } else {
                (angles.data[(i * 7 + (g - 1)) * 2], angles.data[(i * 7 + (g - 1)) * 2 + 1])
            };
            let rot = [1.0, 0.0, 0.0, 0.0, a1, -a0, 0.0, a0, a1];
            all_frames[g] = default[g].compose(&Frame { rot, trans: [0.0; 3] });
        }
        // chi chaining
        let mut to_bb = all_frames;
        to_bb[5] = all_frames[4].compose(&all_frames[5]);
        to_bb[6] = to_bb[5].compose(&all_frames[6]);
        to_bb[7] = to_bb[6].compose(&all_frames[7]);
        // global
        let mut glob = [Frame::identity(); 8];
        for g in 0..8 {
            glob[g] = backb[i].compose(&to_bb[g]);
        }
        out.push(glob);
    }
    out
}

fn frames_to_atom14(frames: &[[Frame; 8]], aatype: &[usize], c: &Constants, n: usize) -> Vec<f32> {
    let mut pos = vec![0.0f32; n * 14 * 3];
    for i in 0..n {
        let a = aatype[i];
        for at in 0..14 {
            let group = c.atom14_to_rigid_group[a * 14 + at];
            let mask = c.atom14_mask[a * 14 + at];
            let lp = &c.atom14_rigid_group_positions[(a * 14 + at) * 3..(a * 14 + at) * 3 + 3];
            let g = frames[i][group].apply([lp[0], lp[1], lp[2]]);
            let o = (i * 14 + at) * 3;
            pos[o] = g[0] * mask;
            pos[o + 1] = g[1] * mask;
            pos[o + 2] = g[2] * mask;
        }
    }
    pos
}

/// Run the structure module; returns per-iteration outputs.
pub fn structure_module(single: &Tensor, pair: &Tensor, aatype: &[usize], w: &Weights, c: &Constants, n: usize) -> Vec<StructIter> {
    let s_norm = ln(single, w, &format!("{SM}.layer_norm_s"));
    let z = ln(pair, w, &format!("{SM}.layer_norm_z"));
    let s_initial = s_norm.clone();
    let mut s = lin(&s_norm, w, &format!("{SM}.linear_in"));

    let mut quat = vec![[1.0f32, 0.0, 0.0, 0.0]; n];
    let mut trans = vec![[0.0f32; 3]; n];
    let mut outputs = Vec::with_capacity(NUM_ITERS);

    for _ in 0..NUM_ITERS {
        let upd = ipa(&s, &z, &quat, &trans, w, n);
        s = Tensor::new(s.data.iter().zip(&upd.data).map(|(a, b)| a + b).collect(), s.shape.clone());
        s = ln(&s, w, &format!("{SM}.layer_norm_ipa"));
        s = transition(&s, w);
        let update = lin(&s, w, &format!("{SM}.bb_update.linear")); // [N,6]
        for i in 0..n {
            let u: [f32; 6] = std::array::from_fn(|k| update.data[i * 6 + k]);
            rigid::compose_q_update(&mut quat[i], &mut trans[i], &u);
        }
        // backbone global frames (scaled translation)
        let backb: Vec<Frame> = (0..n)
            .map(|i| Frame { rot: rigid::quat_to_rot(&quat[i]), trans: [trans[i][0] * TRANS_SCALE, trans[i][1] * TRANS_SCALE, trans[i][2] * TRANS_SCALE] })
            .collect();
        let angles = angle_resnet(&s, &s_initial, w, n);
        let glob = torsion_to_frames(&backb, &angles, aatype, c, n);
        let positions = frames_to_atom14(&glob, aatype, c, n);
        // frames7 = quat ++ trans*scale
        let mut frames7 = vec![0.0f32; n * 7];
        for i in 0..n {
            frames7[i * 7..i * 7 + 4].copy_from_slice(&quat[i]);
            frames7[i * 7 + 4] = trans[i][0] * TRANS_SCALE;
            frames7[i * 7 + 5] = trans[i][1] * TRANS_SCALE;
            frames7[i * 7 + 6] = trans[i][2] * TRANS_SCALE;
        }
        outputs.push(StructIter { frames7, positions, states: s.data.clone() });
    }
    outputs
}
