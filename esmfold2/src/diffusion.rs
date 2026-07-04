//! ESMFold2 diffusion structure head — one denoising step (DiffusionModule).
//! Conditioning (s,z) -> EDM input scale -> atom encoder (coords) -> s_to_token
//! inject -> token transformer (12x AttentionPairBias + ConditionedTransition) ->
//! token_norm -> atom decoder -> EDM output preconditioning.

use crate::atom::{self, AtomInputs};
use crate::ops::*;
use crate::tensor::Tensor;
use crate::weights::Weights;

const EPS: f32 = 1e-5;
const SIGMA_DATA: f32 = 16.0;
const C_TOKEN: usize = 768;
const C_Z: usize = 256;
const TOK_HEADS: usize = 16;
const TOK_HEAD_DIM: usize = 48; // 768/16
const TOK_BLOCKS: usize = 12;

fn lin(x: &Tensor, w: &Weights, name: &str) -> Tensor { linear_f64(x, &w.get(name), None) }
fn lin_b(x: &Tensor, w: &Weights, wn: &str, bn: &str) -> Tensor {
    let b = w.get(bn); linear_f64(x, &w.get(wn), Some(&b))
}
fn ln(x: &Tensor, w: &Weights, wn: &str, bn: &str) -> Tensor {
    layernorm(x, &w.get_vec(wn), Some(&w.get_vec(bn)), EPS)
}

/// TransitionLayer: out_proj(silu(a_proj(norm(x))) * b_proj(norm(x))). No residual.
fn transition_layer(w: &Weights, p: &str, x: &Tensor) -> Tensor {
    let xn = ln(x, w, &format!("{p}.norm.weight"), &format!("{p}.norm.bias"));
    let a = lin(&xn, w, &format!("{p}.a_proj.weight"));
    let b = lin(&xn, w, &format!("{p}.b_proj.weight"));
    let mut h = vec![0.0f32; a.numel()];
    for i in 0..h.len() { h[i] = silu_scalar(a.data[i]) * b.data[i]; }
    let h = Tensor::new(h, a.shape.clone());
    lin(&h, w, &format!("{p}.out_proj.weight"))
}

/// AdaptiveLayerNorm: sigmoid(s_gate(LN(s)*s_scale)) * LN(a) + s_shift(LN(s)).
fn adaln(w: &Weights, p: &str, a: &Tensor, s: &Tensor) -> Tensor {
    let dm = a.last();
    let a_norm = layernorm(a, &vec![1.0f32; dm], None, EPS); // no affine
    let s_scale = w.get_vec(&format!("{p}.s_scale"));
    let s_norm = layernorm(s, &s_scale, None, EPS);
    let gate = lin_b(&s_norm, w, &format!("{p}.s_gate.weight"), &format!("{p}.s_gate.bias"));
    let shift = lin(&s_norm, w, &format!("{p}.s_shift.weight"));
    let mut out = vec![0.0f32; a.numel()];
    for i in 0..out.len() {
        out[i] = sigmoid_scalar(gate.data[i]) * a_norm.data[i] + shift.data[i];
    }
    Tensor::new(out, a.shape.clone())
}

/// Conditioning: returns (s[L,768], z[L,L,256]).
pub fn conditioning(w: &Weights, t_hat: f32, s_inputs: &Tensor, z_trunk: &Tensor, rel_pos: &Tensor) -> (Tensor, Tensor) {
    let cp = "structure_head.diffusion_module.conditioning";
    let l = z_trunk.shape[0];
    // z = z_proj(z_input_norm(cat[z_trunk, rel])) ; 2x residual transition
    let mut zc = vec![0.0f32; l * l * 2 * C_Z];
    for i in 0..l * l {
        zc[i * 2 * C_Z..i * 2 * C_Z + C_Z].copy_from_slice(&z_trunk.data[i * C_Z..i * C_Z + C_Z]);
        zc[i * 2 * C_Z + C_Z..i * 2 * C_Z + 2 * C_Z].copy_from_slice(&rel_pos.data[i * C_Z..i * C_Z + C_Z]);
    }
    let zc = Tensor::new(zc, vec![l, l, 2 * C_Z]);
    let zn = ln(&zc, w, &format!("{cp}.z_input_norm.weight"), &format!("{cp}.z_input_norm.bias"));
    let mut z = lin(&zn, w, &format!("{cp}.z_proj.weight")); // [L,L,256]
    for blk in 0..2 {
        let d = transition_layer(w, &format!("{cp}.z_transitions.{blk}"), &z);
        z = z.add(&d);
    }
    // s = s_proj(s_input_norm(s_inputs))
    let sn = ln(s_inputs, w, &format!("{cp}.s_input_norm.weight"), &format!("{cp}.s_input_norm.bias"));
    let mut s = lin(&sn, w, &format!("{cp}.s_proj.weight")); // [L,768]
    // noise embedding
    let t_noise = 0.25 * (t_hat / SIGMA_DATA).max(1e-20).ln();
    let fw = w.get_vec(&format!("{cp}.fourier.w"));
    let fb = w.get_vec(&format!("{cp}.fourier.b"));
    let two_pi = 2.0 * std::f64::consts::PI;
    let mut nfeat = vec![0.0f32; fw.len()];
    for k in 0..fw.len() {
        let ang = two_pi * ((t_noise * fw[k] + fb[k]) as f64);
        nfeat[k] = ang.cos() as f32;
    }
    let nfeat = Tensor::new(nfeat, vec![1, fw.len()]);
    let nn = ln(&nfeat, w, &format!("{cp}.noise_norm.weight"), &format!("{cp}.noise_norm.bias"));
    let nproj = lin(&nn, w, &format!("{cp}.noise_proj.weight")); // [1,768]
    for i in 0..l {
        for d in 0..C_TOKEN { s.data[i * C_TOKEN + d] += nproj.data[d]; }
    }
    for blk in 0..2 {
        let d = transition_layer(w, &format!("{cp}.s_transitions.{blk}"), &s);
        s = s.add(&d);
    }
    (s, z)
}

/// AttentionPairBias. a,s:[L,768]; z:[L,L,256]; valid:[L]. -> [L,768].
fn attention_pair_bias(w: &Weights, p: &str, a: &Tensor, s: &Tensor, z: &Tensor, valid: &[bool]) -> Tensor {
    let l = a.shape[0];
    let x = adaln(w, &format!("{p}.adaln"), a, s);
    let q = lin_b(&x, w, &format!("{p}.q_proj.weight"), &format!("{p}.q_proj.bias")); // [L,768]
    let kv = lin(&x, w, &format!("{p}.kv_proj.weight")); // [L,1536]
    let g = lin(&x, w, &format!("{p}.g_proj.weight")); // [L,768]
    let scale = (TOK_HEAD_DIM as f32).powf(-0.5);
    // pair bias [L,L,16]
    let zn = ln(z, w, &format!("{p}.pair_norm.weight"), &format!("{p}.pair_norm.bias"));
    let pbias = lin(&zn, w, &format!("{p}.pair_bias_proj.weight")); // [L,L,16]
    // per-head attention
    let mut out = vec![0.0f32; l * C_TOKEN];
    for h in 0..TOK_HEADS {
        for i in 0..l {
            let qbase = i * C_TOKEN + h * TOK_HEAD_DIM;
            let qi = &q.data[qbase..qbase + TOK_HEAD_DIM];
            let mut logits = vec![0.0f32; l];
            let mut mx = f32::NEG_INFINITY;
            for j in 0..l {
                let kbase = j * 2 * C_TOKEN + h * TOK_HEAD_DIM; // k is first half of kv
                let kj = &kv.data[kbase..kbase + TOK_HEAD_DIM];
                let mut sdot = 0.0f32;
                for d in 0..TOK_HEAD_DIM { sdot += qi[d] * kj[d]; }
                let mut lg = sdot * scale + pbias.data[(i * l + j) * TOK_HEADS + h];
                if !valid[j] { lg += f32::MIN; }
                logits[j] = lg;
                if lg > mx { mx = lg; }
            }
            let mut sum = 0.0f32;
            for j in 0..l { logits[j] = (logits[j] - mx).exp(); sum += logits[j]; }
            let inv = 1.0 / sum;
            let obase = i * C_TOKEN + h * TOK_HEAD_DIM;
            for j in 0..l {
                let aw = logits[j] * inv;
                let vbase = j * 2 * C_TOKEN + C_TOKEN + h * TOK_HEAD_DIM; // v = second half
                let vj = &kv.data[vbase..vbase + TOK_HEAD_DIM];
                for d in 0..TOK_HEAD_DIM { out[obase + d] += aw * vj[d]; }
            }
            // gate
            for d in 0..TOK_HEAD_DIM { out[obase + d] *= sigmoid_scalar(g.data[obase + d]); }
        }
    }
    let ctx = Tensor::new(out, vec![l, C_TOKEN]);
    let mut o = lin(&ctx, w, &format!("{p}.out_proj.weight")).data;
    let og = lin_b(s, w, &format!("{p}.out_gate.weight"), &format!("{p}.out_gate.bias"));
    for i in 0..o.len() { o[i] *= sigmoid_scalar(og.data[i]); }
    Tensor::new(o, vec![l, C_TOKEN])
}

/// ConditionedTransitionBlock. a,s:[L,768] -> [L,768].
fn conditioned_transition(w: &Weights, p: &str, a: &Tensor, s: &Tensor) -> Tensor {
    let x = adaln(w, &format!("{p}.adaln"), a, s);
    let sw = lin(&x, w, &format!("{p}.lin_swish.weight")); // [L,3072]
    let hidden = sw.last() / 2; // 1536
    let l = x.shape[0];
    let mut h = vec![0.0f32; l * hidden];
    for i in 0..l {
        let row = &sw.data[i * 2 * hidden..i * 2 * hidden + 2 * hidden];
        for d in 0..hidden { h[i * hidden + d] = silu_scalar(row[d]) * row[hidden + d]; }
    }
    let h = Tensor::new(h, vec![l, hidden]);
    let mut o = lin(&h, w, &format!("{p}.lin_out.weight")).data;
    let og = lin_b(s, w, &format!("{p}.output_gate.weight"), &format!("{p}.output_gate.bias"));
    for i in 0..o.len() { o[i] *= sigmoid_scalar(og.data[i]); }
    Tensor::new(o, vec![l, C_TOKEN])
}

/// DiffusionTransformer (token): 12 blocks of attn + transition residuals.
pub fn token_transformer(w: &Weights, a: &Tensor, s: &Tensor, z: &Tensor, valid: &[bool]) -> Tensor {
    let tp = "structure_head.diffusion_module.token_transformer";
    let mut x = a.clone();
    for blk in 0..TOK_BLOCKS {
        let attn = attention_pair_bias(w, &format!("{tp}.attn_blocks.{blk}"), &x, s, z, valid);
        x = x.add(&attn);
        let trans = conditioned_transition(w, &format!("{tp}.transition_blocks.{blk}"), &x, s);
        x = x.add(&trans);
    }
    x
}

/// Structure-prediction atom encoder for diffusion. Returns (a[L,768], q_skip[N,128],
/// c_base[N,128], cos, sin).
pub fn atom_encoder_sp(
    w: &Weights, inp: &AtomInputs, r_noisy: &Tensor,
) -> (Tensor, Tensor, Tensor, Vec<f32>, Vec<f32>) {
    let ap = "structure_head.diffusion_module.atom_encoder";
    let n = inp.n_atoms;
    let c_base = atom::compute_c_base(w, ap, inp);
    let (cos, sin) = atom::rope3d(inp.ref_pos, inp.ref_space_uid, n);
    // coords inject: r_input = cat[r_noisy, zeros] [N,6]; q = c_base + coords_linear(r_input)
    let mut rin = vec![0.0f32; n * 6];
    for i in 0..n {
        rin[i * 6..i * 6 + 3].copy_from_slice(&r_noisy.data[i * 3..i * 3 + 3]);
        // pred_r1 = zeros
    }
    let rin = Tensor::new(rin, vec![n, 6]);
    let r_to_q = atom::lin_f64(&rin, w, &format!("{ap}.coords_linear.weight"));
    let q0 = c_base.add(&r_to_q);
    let q = atom::run_atom_transformer(w, &format!("{ap}.atom_transformer"), &q0, &c_base, &cos, &sin, inp.atom_mask);
    let mut q_to_a = atom::lin_f64(&q, w, &format!("{ap}.atom_to_token_linear.weight")).data; // [N,768]
    for x in q_to_a.iter_mut() { if *x < 0.0 { *x = 0.0; } }
    let q_to_a = Tensor::new(q_to_a, vec![n, C_TOKEN]);
    let a = atom::scatter_atoms_to_tokens(&q_to_a, inp.atom_to_token, inp.n_tokens, inp.atom_mask);
    (a, q, c_base, cos, sin)
}

/// Atom decoder -> coordinate update [N,3].
pub fn atom_decoder(
    w: &Weights, a_i: &Tensor, q_skip: &Tensor, c_skip: &Tensor, cos: &[f32], sin: &[f32], inp: &AtomInputs,
) -> Tensor {
    let ap = "structure_head.diffusion_module.atom_decoder";
    let n = inp.n_atoms;
    let a_to_q = atom::lin_f64(a_i, w, &format!("{ap}.token_to_atom_linear.weight")); // [L,128]
    let a_to_q = atom::gather_tokens_to_atoms(&a_to_q, inp.atom_to_token, n); // [N,128]
    let q_l = q_skip.add(&a_to_q);
    let q_l = atom::run_atom_transformer(w, &format!("{ap}.atom_transformer"), &q_l, c_skip, cos, sin, inp.atom_mask);
    let qn = ln(&q_l, w, &format!("{ap}.norm.weight"), &format!("{ap}.norm.bias"));
    atom::lin_f64(&qn, w, &format!("{ap}.output_linear.weight")) // [N,3]
}

/// Full DiffusionModule denoising step -> x_denoised [N,3].
pub fn diffusion_module_step(
    w: &Weights, x_noisy: &Tensor, t_hat: f32, inp: &AtomInputs,
    s_inputs: &Tensor, z_trunk: &Tensor, rel_pos: &Tensor, valid: &[bool],
) -> Tensor {
    let n = inp.n_atoms;
    let (s, z) = conditioning(w, t_hat, s_inputs, z_trunk, rel_pos);
    let denom = (t_hat * t_hat + SIGMA_DATA * SIGMA_DATA).sqrt();
    let mut r_noisy = vec![0.0f32; n * 3];
    for i in 0..n * 3 { r_noisy[i] = x_noisy.data[i] / denom; }
    let r_noisy = Tensor::new(r_noisy, vec![n, 3]);
    let (mut a, q_skip, c_skip, cos, sin) = atom_encoder_sp(w, inp, &r_noisy);
    // a = a + s_to_token(s_step_norm(s))
    let dp = "structure_head.diffusion_module";
    let ssn = ln(&s, w, &format!("{dp}.s_step_norm.weight"), &format!("{dp}.s_step_norm.bias"));
    let stt = lin(&ssn, w, &format!("{dp}.s_to_token.weight"));
    a = a.add(&stt);
    a = token_transformer(w, &a, &s, &z, valid);
    a = ln(&a, w, &format!("{dp}.token_norm.weight"), &format!("{dp}.token_norm.bias"));
    let r_update = atom_decoder(w, &a, &q_skip, &c_skip, &cos, &sin, inp);
    // EDM output
    let sigma2 = SIGMA_DATA * SIGMA_DATA;
    let t2 = t_hat * t_hat;
    let c_skip_s = sigma2 / (sigma2 + t2);
    let c_out = (SIGMA_DATA * t_hat) / (sigma2 + t2).sqrt();
    let mut out = vec![0.0f32; n * 3];
    for i in 0..n * 3 { out[i] = c_skip_s * x_noisy.data[i] + c_out * r_update.data[i]; }
    Tensor::new(out, vec![n, 3])
}

// ===========================================================================
// EDM / ODE sampler (Algorithm 18). RNG (x_init, per-step rotation+translation)
// is injected from the reference for validation; reproduce torch RNG later.
// ===========================================================================

// Checkpoint values (override the dataclass defaults): SDE sampler with churn.
const GAMMA_0: f32 = 0.8;
const GAMMA_MIN: f32 = 1.0;
const NOISE_SCALE: f32 = 1.003; // lam (churn); >0 => stochastic SDE
const STEP_SCALE: f32 = 1.5; // eta

/// Jacobi eigendecomposition of a symmetric 3x3 (f64). Returns (eigvecs columns, eigvals).
fn sym_eig3(mut a: [[f64; 3]; 3]) -> ([[f64; 3]; 3], [f64; 3]) {
    let mut v = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    for _sweep in 0..50 {
        // off-diagonal magnitude
        let off = a[0][1].abs() + a[0][2].abs() + a[1][2].abs();
        if off < 1e-300 { break; }
        for &(p, q) in &[(0usize, 1usize), (0, 2), (1, 2)] {
            let apq = a[p][q];
            if apq.abs() < 1e-300 { continue; }
            let theta = (a[q][q] - a[p][p]) / (2.0 * apq);
            let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
            let c = 1.0 / (t * t + 1.0).sqrt();
            let s = t * c;
            // rotate
            for k in 0..3 {
                let akp = a[k][p]; let akq = a[k][q];
                a[k][p] = c * akp - s * akq;
                a[k][q] = s * akp + c * akq;
            }
            for k in 0..3 {
                let apk = a[p][k]; let aqk = a[q][k];
                a[p][k] = c * apk - s * aqk;
                a[q][k] = s * apk + c * aqk;
            }
            for k in 0..3 {
                let vkp = v[k][p]; let vkq = v[k][q];
                v[k][p] = c * vkp - s * vkq;
                v[k][q] = s * vkp + c * vkq;
            }
        }
    }
    let eig = [a[0][0], a[1][1], a[2][2]];
    (v, eig)
}

/// 3x3 SVD via eigendecomposition of H^T H. Returns (U, S, Vt) with H = U diag(S) Vt.
fn svd3(h: [[f64; 3]; 3]) -> ([[f64; 3]; 3], [f64; 3], [[f64; 3]; 3]) {
    // A = H^T H
    let mut ata = [[0.0; 3]; 3];
    for i in 0..3 { for j in 0..3 { let mut s = 0.0; for k in 0..3 { s += h[k][i] * h[k][j]; } ata[i][j] = s; } }
    let (vv, eig) = sym_eig3(ata);
    // sort descending
    let mut idx = [0usize, 1, 2];
    idx.sort_by(|&a, &b| eig[b].partial_cmp(&eig[a]).unwrap());
    let mut vmat = [[0.0; 3]; 3]; // columns = eigenvectors
    let mut s = [0.0; 3];
    for (col, &i) in idx.iter().enumerate() {
        for r in 0..3 { vmat[r][col] = vv[r][i]; }
        s[col] = eig[i].max(0.0).sqrt();
    }
    // U columns = H v_col / s_col
    let mut umat = [[0.0; 3]; 3];
    for col in 0..3 {
        let mut u = [0.0; 3];
        for r in 0..3 { let mut acc = 0.0; for k in 0..3 { acc += h[r][k] * vmat[k][col]; } u[r] = acc; }
        if s[col] > 1e-12 {
            for r in 0..3 { umat[r][col] = u[r] / s[col]; }
        } else {
            for r in 0..3 { umat[r][col] = 0.0; }
        }
    }
    // fix any degenerate U columns via cross product to keep orthonormal
    let norm = |a: [f64; 3]| (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt();
    let col = |m: &[[f64; 3]; 3], c: usize| [m[0][c], m[1][c], m[2][c]];
    if norm(col(&umat, 2)) < 1e-9 {
        let u0 = col(&umat, 0); let u1 = col(&umat, 1);
        let cr = [u0[1] * u1[2] - u0[2] * u1[1], u0[2] * u1[0] - u0[0] * u1[2], u0[0] * u1[1] - u0[1] * u1[0]];
        for r in 0..3 { umat[r][2] = cr[r]; }
    }
    // Vt = V^T
    let mut vt = [[0.0; 3]; 3];
    for i in 0..3 { for j in 0..3 { vt[i][j] = vmat[j][i]; } }
    (umat, s, vt)
}

fn det3(m: &[[f64; 3]; 3]) -> f64 {
    m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
}
fn matmul3(a: &[[f64; 3]; 3], b: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let mut o = [[0.0; 3]; 3];
    for i in 0..3 { for j in 0..3 { let mut s = 0.0; for k in 0..3 { s += a[i][k] * b[k][j]; } o[i][j] = s; } }
    o
}

/// Weighted rigid (Kabsch) align of x to x_gt with weights w, then translate to mu_gt.
fn weighted_rigid_align(x: &[f32], x_gt: &[f32], w: &[f32], n: usize) -> Vec<f32> {
    let mut wsum = 0.0f64;
    for &wi in w.iter() { wsum += wi as f64; }
    if wsum < 1e-8 { wsum = 1e-8; }
    let mut mu = [0.0f64; 3];
    let mut mu_gt = [0.0f64; 3];
    for nn in 0..n {
        let wi = w[nn] as f64;
        for d in 0..3 { mu[d] += wi * x[nn * 3 + d] as f64; mu_gt[d] += wi * x_gt[nn * 3 + d] as f64; }
    }
    for d in 0..3 { mu[d] /= wsum; mu_gt[d] /= wsum; }
    // H[i,j] = sum_n w[n]*xgt_c[n,i]*x_c[n,j]
    let mut hmat = [[0.0f64; 3]; 3];
    for nn in 0..n {
        let wi = w[nn] as f64;
        let xc = [x[nn * 3] as f64 - mu[0], x[nn * 3 + 1] as f64 - mu[1], x[nn * 3 + 2] as f64 - mu[2]];
        let gc = [x_gt[nn * 3] as f64 - mu_gt[0], x_gt[nn * 3 + 1] as f64 - mu_gt[1], x_gt[nn * 3 + 2] as f64 - mu_gt[2]];
        for i in 0..3 { for j in 0..3 { hmat[i][j] += wi * gc[i] * xc[j]; } }
    }
    let (u, _s, vt) = svd3(hmat);
    let uvt = matmul3(&u, &vt);
    let d = det3(&uvt);
    let diag = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, d.signum().max(-1.0).min(1.0) * if d == 0.0 { 1.0 } else { d.signum() }]];
    // R = U @ diag(1,1,sign(d)) @ Vt   (det correction: third entry = det(U Vt))
    let mut dd = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, d]];
    let _ = diag;
    dd[2][2] = d; // exactly det(U@Vt)
    let r = matmul3(&matmul3(&u, &dd), &vt);
    // aligned[n,i] = sum_j x_c[n,j]*R[i,j] + mu_gt[i]
    let mut out = vec![0.0f32; n * 3];
    for nn in 0..n {
        let xc = [x[nn * 3] as f64 - mu[0], x[nn * 3 + 1] as f64 - mu[1], x[nn * 3 + 2] as f64 - mu[2]];
        for i in 0..3 {
            let mut acc = mu_gt[i];
            for j in 0..3 { acc += xc[j] * r[i][j]; }
            out[nn * 3 + i] = acc as f32;
        }
    }
    out
}

/// Center + apply injected rotation R[3x3] + injected translation t[3] to x and second.
fn center_random_augmentation(x: &mut [f32], second: &mut [f32], mask: &[f32], r: &[f32], t: &[f32], n: usize) {
    let mut denom = 0.0f32;
    for &m in mask.iter() { denom += m; }
    if denom < 1.0 { denom = 1.0; }
    let mut mean = [0.0f32; 3];
    for nn in 0..n { let m = mask[nn]; for d in 0..3 { mean[d] += m * x[nn * 3 + d]; } }
    for d in 0..3 { mean[d] /= denom; }
    // x = (x-mean) @ R + t ; einsum "md,ds->ms"
    for nn in 0..n {
        let xc = [x[nn * 3] - mean[0], x[nn * 3 + 1] - mean[1], x[nn * 3 + 2] - mean[2]];
        let sc = [second[nn * 3] - mean[0], second[nn * 3 + 1] - mean[1], second[nn * 3 + 2] - mean[2]];
        for s in 0..3 {
            let mut ax = 0.0f32; let mut asd = 0.0f32;
            for d in 0..3 { ax += xc[d] * r[d * 3 + s]; asd += sc[d] * r[d * 3 + s]; }
            x[nn * 3 + s] = ax + t[s];
            second[nn * 3 + s] = asd + t[s];
        }
    }
}

/// Full EDM sampler with injected RNG. schedule[steps+1], r_aug/t_aug per step.
pub fn sample(
    w: &Weights, inp: &AtomInputs, s_inputs: &Tensor, z_trunk: &Tensor, rel_pos: &Tensor,
    tok_valid: &[bool], x_init: &Tensor, schedule: &[f32], r_aug: &[Vec<f32>], t_aug: &[Vec<f32>],
    churn: &[Vec<f32>],
) -> Tensor {
    sample_cb(w, inp, s_inputs, z_trunk, rel_pos, tok_valid, x_init, schedule, r_aug, t_aug, churn, &mut |_, _| {})
}

/// As [`sample`], invoking `prog(step, n_steps)` after each denoising step
/// (step counted 1..=n_steps). Numerically identical to `sample`.
#[allow(clippy::too_many_arguments)]
pub fn sample_cb(
    w: &Weights, inp: &AtomInputs, s_inputs: &Tensor, z_trunk: &Tensor, rel_pos: &Tensor,
    tok_valid: &[bool], x_init: &Tensor, schedule: &[f32], r_aug: &[Vec<f32>], t_aug: &[Vec<f32>],
    churn: &[Vec<f32>], prog: &mut dyn FnMut(usize, usize),
) -> Tensor {
    let n = inp.n_atoms;
    let atom_mask_f: Vec<f32> = inp.atom_mask.iter().map(|&b| if b { 1.0 } else { 0.0 }).collect();
    // gammas[i] = gamma_0 if schedule[i] > gamma_min else 0
    let gammas: Vec<f32> = schedule.iter().map(|&s| if s > GAMMA_MIN { GAMMA_0 } else { 0.0 }).collect();
    let n_steps = schedule.len() - 1;
    let mut x = x_init.data.clone();
    let mut x_denoised_prev = vec![0.0f32; n * 3];
    for step in 0..n_steps {
        let sigma_tm = schedule[step];
        let sigma_t = schedule[step + 1];
        let gamma = gammas[step + 1];
        center_random_augmentation(&mut x, &mut x_denoised_prev, &atom_mask_f, &r_aug[step], &t_aug[step], n);
        let t_hat = sigma_tm * (1.0 + gamma);
        // SDE churn: x_noisy = x + eps_std * churn,  eps_std = lam*sqrt(max(t_hat^2 - sigma_tm^2, 0))
        let eps_std = NOISE_SCALE * (t_hat * t_hat - sigma_tm * sigma_tm).max(0.0).sqrt();
        let mut xn = x.clone();
        for i in 0..n * 3 { xn[i] += eps_std * churn[step][i]; }
        let x_noisy = Tensor::new(xn, vec![n, 3]);
        let x_denoised = diffusion_module_step(w, &x_noisy, t_hat, inp, s_inputs, z_trunk, rel_pos, tok_valid);
        let aligned = weighted_rigid_align(&x_noisy.data, &x_denoised.data, &atom_mask_f, n);
        // x = aligned + eta*(sigma_t - t_hat)*(aligned - x_denoised)/t_hat
        let coef = STEP_SCALE * (sigma_t - t_hat) / t_hat;
        for i in 0..n * 3 {
            x[i] = aligned[i] + coef * (aligned[i] - x_denoised.data[i]);
        }
        x_denoised_prev = x_denoised.data;
        prog(step + 1, n_steps);
    }
    Tensor::new(x, vec![n, 3])
}
