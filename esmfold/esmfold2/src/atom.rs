//! ESMFold2 atom encoder (SWA attention + 3D RoPE) and the InputsEmbedder.
//!
//! Bit-exactness: the reference rounds the 3D-RoPE cos/sin and the attention
//! q/k/v to **bf16**, then runs SDPA in fp32 opmath (torch upcasts bf16->fp32
//! internally) and rounds the attention output back to bf16. We replicate those
//! exact rounding points with `bf16_round` and do the SDPA math in fp32.

use crate::ops::*;
use crate::tensor::{bf16_round, Tensor};
use crate::weights::Weights;

/// When true, emulate the reference's hard-coded bf16 casts in the SWA atom
/// attention (matches the released bf16 inference path). When false, run the
/// atom attention in pure fp32 — use this with the fp32-both-sides reference
/// (bf16 casts patched out) for sub-mÅ reproducibility.
const EMULATE_BF16: bool = false;
#[inline]
fn maybe_bf16(x: f32) -> f32 { if EMULATE_BF16 { bf16_round(x) } else { x } }

const D_ATOM: usize = 128;
const N_HEADS: usize = 4;
const HEAD_DIM: usize = 32; // d_atom / n_heads
const HALF: usize = 16; // head_dim / 2
pub const N_BLOCKS: usize = 3;
const HALF_WINDOW: i64 = 64; // swa_window_size(128) / 2
const LN_EPS: f32 = 1e-5;

fn lin(x: &Tensor, w: &Weights, name: &str) -> Tensor {
    linear_f64(x, &w.get(name), None)
}

/// Public 3D RoPE builder for reuse by the diffusion atom enc/dec.
pub fn rope3d(ref_pos: &[f32], ref_space_uid: &[f32], n: usize) -> (Vec<f32>, Vec<f32>) {
    build_3d_rope(ref_pos, ref_space_uid, n)
}

/// Run the SWA atom transformer (N_BLOCKS blocks) given q,c and rope/valid.
/// `tprefix` is the atom_transformer prefix (e.g. "<enc>.atom_transformer").
pub fn run_atom_transformer(
    w: &Weights, tprefix: &str, q: &Tensor, c: &Tensor, cos: &[f32], sin: &[f32], valid: &[bool],
) -> Tensor {
    let mut q = q.clone();
    for blk in 0..N_BLOCKS {
        q = atom_block(w, &format!("{tprefix}.blocks.{blk}"), &q, c, cos, sin, valid);
    }
    q
}

/// atom_linear + atom_norm over assembled atom features -> c_base [N,128].
pub fn compute_c_base(w: &Weights, ap: &str, inp: &AtomInputs) -> Tensor {
    let n = inp.n_atoms;
    let mut feats = vec![0.0f32; n * ATOM_FEAT_DIM];
    for i in 0..n {
        let b = i * ATOM_FEAT_DIM;
        feats[b] = inp.ref_pos[i * 3];
        feats[b + 1] = inp.ref_pos[i * 3 + 1];
        feats[b + 2] = inp.ref_pos[i * 3 + 2];
        feats[b + 3] = inp.ref_charge[i];
        feats[b + 4] = if inp.atom_mask[i] { 1.0 } else { 0.0 };
        feats[b + 5..b + 5 + 128].copy_from_slice(&inp.ref_element[i * 128..i * 128 + 128]);
        feats[b + 133..b + 133 + 256].copy_from_slice(&inp.ref_atom_name_chars[i * 256..i * 256 + 256]);
    }
    let feats = Tensor::new(feats, vec![n, ATOM_FEAT_DIM]);
    let lin_out = lin(&feats, w, &format!("{ap}.atom_linear.weight"));
    layernorm(&lin_out, &w.get_vec(&format!("{ap}.atom_norm.weight")),
              Some(&w.get_vec(&format!("{ap}.atom_norm.bias"))), LN_EPS)
}

// ---- 3D RoPE (atom config: n_spatial=2 base 20, n_uid=10 base 10000) ----
/// Returns (cos[N,16], sin[N,16]), each bf16-rounded.
fn build_3d_rope(ref_pos: &[f32], ref_space_uid: &[f32], n: usize) -> (Vec<f32>, Vec<f32>) {
    const NSP: usize = 2;
    const NUID: usize = 10;
    let sp_base = 20.0f32;
    let uid_base = 10000.0f32;
    let mut sp_inv = [0.0f32; NSP];
    for k in 0..NSP { sp_inv[k] = 1.0 / sp_base.powf(k as f32 / NSP as f32); }
    let mut uid_inv = [0.0f32; NUID];
    for k in 0..NUID { uid_inv[k] = 1.0 / uid_base.powf(k as f32 / NUID as f32); }
    let mut cos = vec![0.0f32; n * HALF];
    let mut sin = vec![0.0f32; n * HALF];
    for i in 0..n {
        let mut freqs = [0.0f32; HALF];
        // spatial: axis-major, freq-minor -> indices 0..6
        for a in 0..3 {
            let p = ref_pos[i * 3 + a];
            for k in 0..NSP { freqs[a * NSP + k] = p * sp_inv[k]; }
        }
        // uid: indices 6..16
        let u = ref_space_uid[i];
        for k in 0..NUID { freqs[3 * NSP + k] = u * uid_inv[k]; }
        for j in 0..HALF {
            cos[i * HALF + j] = maybe_bf16(freqs[j].cos());
            sin[i * HALF + j] = maybe_bf16(freqs[j].sin());
        }
    }
    (cos, sin)
}

/// Apply 3D RoPE to x[N,H,32] with cos/sin[N,16] (tiled), result fp32.
fn apply_rope(x: &[f32], cos: &[f32], sin: &[f32], n: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; n * N_HEADS * HEAD_DIM];
    for i in 0..n {
        let cs = &cos[i * HALF..i * HALF + HALF];
        let sn = &sin[i * HALF..i * HALF + HALF];
        for h in 0..N_HEADS {
            let base = (i * N_HEADS + h) * HEAD_DIM;
            let xr = &x[base..base + HEAD_DIM];
            let orow = &mut out[base..base + HEAD_DIM];
            for j in 0..HALF {
                orow[j] = xr[j] * cs[j] - xr[j + HALF] * sn[j];
                orow[j + HALF] = xr[j + HALF] * cs[j] + xr[j] * sn[j];
            }
        }
    }
    out
}

/// Per-head RMSNorm over head_dim (eps = 2^-23, no weight), x[N,H,32].
fn qk_norm(x: &mut [f32], n: usize) {
    let eps = norm::RMS_DEFAULT_EPS;
    for i in 0..n {
        for h in 0..N_HEADS {
            let base = (i * N_HEADS + h) * HEAD_DIM;
            let row = &mut x[base..base + HEAD_DIM];
            let mut ms = 0.0f32;
            for &v in row.iter() { ms += v * v; }
            ms /= HEAD_DIM as f32;
            let inv = 1.0 / (ms + eps).sqrt();
            for v in row.iter_mut() { *v *= inv; }
        }
    }
}

/// SWA attention. x_input[N,128] is the adaLN'd input (also used for the gate).
fn swa_attention(w: &Weights, p: &str, x_input: &Tensor, cos: &[f32], sin: &[f32], valid: &[bool]) -> Tensor {
    let n = x_input.shape[0];
    let qkv = lin(x_input, w, &format!("{p}.Wqkv.weight")); // [N, 384]
    // split into q,k,v [N,H,32] (C-contiguous: sel//128, head, dim)
    let mut q = vec![0.0f32; n * N_HEADS * HEAD_DIM];
    let mut k = vec![0.0f32; n * N_HEADS * HEAD_DIM];
    let mut v = vec![0.0f32; n * N_HEADS * HEAD_DIM];
    for i in 0..n {
        let row = &qkv.data[i * 3 * D_ATOM..i * 3 * D_ATOM + 3 * D_ATOM];
        let dst = i * N_HEADS * HEAD_DIM;
        q[dst..dst + D_ATOM].copy_from_slice(&row[0..D_ATOM]);
        k[dst..dst + D_ATOM].copy_from_slice(&row[D_ATOM..2 * D_ATOM]);
        v[dst..dst + D_ATOM].copy_from_slice(&row[2 * D_ATOM..3 * D_ATOM]);
    }
    qk_norm(&mut q, n);
    qk_norm(&mut k, n);
    let mut q = apply_rope(&q, cos, sin, n);
    let mut k = apply_rope(&k, cos, sin, n);
    // bf16 round q,k,v before SDPA (only when emulating the bf16 reference path)
    if EMULATE_BF16 {
        for x in q.iter_mut() { *x = bf16_round(*x); }
        for x in k.iter_mut() { *x = bf16_round(*x); }
        for x in v.iter_mut() { *x = bf16_round(*x); }
    }

    let scale = (HEAD_DIM as f32).powf(-0.5);
    // rank = cumsum(valid)-1
    let mut rank = vec![0i64; n];
    let mut c = 0i64;
    for i in 0..n { if valid[i] { c += 1; } rank[i] = c - 1; }

    let mut out = vec![0.0f32; n * N_HEADS * HEAD_DIM];
    for h in 0..N_HEADS {
        for i in 0..n {
            // scores over allowed keys
            let qi = &q[(i * N_HEADS + h) * HEAD_DIM..(i * N_HEADS + h) * HEAD_DIM + HEAD_DIM];
            let mut scores = vec![f32::NEG_INFINITY; n];
            let mut m = f32::NEG_INFINITY;
            for j in 0..n {
                let allowed = (i == j)
                    || ((rank[i] - rank[j]).abs() <= HALF_WINDOW && valid[i] && valid[j]);
                if !allowed { continue; }
                let kj = &k[(j * N_HEADS + h) * HEAD_DIM..(j * N_HEADS + h) * HEAD_DIM + HEAD_DIM];
                let mut s = 0.0f32;
                for d in 0..HEAD_DIM { s += qi[d] * kj[d]; }
                s *= scale;
                scores[j] = s;
                if s > m { m = s; }
            }
            // softmax over allowed
            let mut sum = 0.0f32;
            for j in 0..n {
                if scores[j] == f32::NEG_INFINITY { scores[j] = 0.0; }
                else { scores[j] = (scores[j] - m).exp(); sum += scores[j]; }
            }
            let inv = 1.0 / sum;
            let obase = (i * N_HEADS + h) * HEAD_DIM;
            for j in 0..n {
                if scores[j] == 0.0 { continue; }
                let wgt = scores[j] * inv;
                let vj = &v[(j * N_HEADS + h) * HEAD_DIM..(j * N_HEADS + h) * HEAD_DIM + HEAD_DIM];
                for d in 0..HEAD_DIM { out[obase + d] += wgt * vj[d]; }
            }
        }
    }
    // SDPA returns bf16 -> round (only in bf16-emulation mode); zero invalid rows
    if EMULATE_BF16 { for x in out.iter_mut() { *x = bf16_round(*x); } }
    for i in 0..n {
        if !valid[i] {
            for x in out[i * D_ATOM..i * D_ATOM + D_ATOM].iter_mut() { *x = 0.0; }
        }
    }
    let out = Tensor::new(out, vec![n, D_ATOM]);
    // gate from x_input, then out_proj
    let gate = lin(x_input, w, &format!("{p}.gate_proj.weight"));
    let mut gated = out.data.clone();
    for (o, g) in gated.iter_mut().zip(&gate.data) { *o *= sigmoid_scalar(*g); }
    let gated = Tensor::new(gated, vec![n, D_ATOM]);
    lin(&gated, w, &format!("{p}.out_proj.weight"))
}

/// SwiGLUFFN: w_up[512,128] -> split(256) -> silu(x1)*x2 -> w_down[128,256].
fn swiglu_ffn(w: &Weights, p: &str, x: &Tensor) -> Tensor {
    let up = lin(x, w, &format!("{p}.w_up.weight"));
    let g = swiglu_split(&up);
    lin(&g, w, &format!("{p}.w_down.weight"))
}

/// rms_norm(x) * (1+scale) + shift; eps = 2^-23. x,scale,shift all [N,128].
fn rms_adaln(x: &Tensor, scale: &[f32], shift: &[f32]) -> Tensor {
    let normed = rmsnorm(x, None, norm::RMS_DEFAULT_EPS);
    let mut out = normed.data;
    for idx in 0..out.len() {
        out[idx] = out[idx] * (1.0 + scale[idx]) + shift[idx];
    }
    Tensor::new(out, x.shape.clone())
}

/// One SWAAtomBlock. x,c are [N,128].
fn atom_block(w: &Weights, p: &str, x: &Tensor, c: &Tensor, cos: &[f32], sin: &[f32], valid: &[bool]) -> Tensor {
    let n = x.shape[0];
    // mod = silu(c) @ adaln_modulation.1.weight -> [N, 768]; chunk6
    let c_silu = silu(c);
    let m = lin(&c_silu, w, &format!("{p}.adaln_modulation.1.weight")); // [N,768]
    let chunk = |idx: usize| -> Vec<f32> {
        let mut v = vec![0.0f32; n * D_ATOM];
        for i in 0..n {
            v[i * D_ATOM..i * D_ATOM + D_ATOM]
                .copy_from_slice(&m.data[i * 6 * D_ATOM + idx * D_ATOM..i * 6 * D_ATOM + (idx + 1) * D_ATOM]);
        }
        v
    };
    let (shift_a, scale_a, gate_a) = (chunk(0), chunk(1), chunk(2));
    let (shift_f, scale_f, gate_f) = (chunk(3), chunk(4), chunk(5));

    let attn_input = rms_adaln(x, &scale_a, &shift_a);
    let attn_out = swa_attention(w, &format!("{p}.attn"), &attn_input, cos, sin, valid);
    let mut x2 = x.data.clone();
    for idx in 0..x2.len() { x2[idx] += gate_a[idx] * attn_out.data[idx]; }
    let x2 = Tensor::new(x2, x.shape.clone());

    let ffn_input = rms_adaln(&x2, &scale_f, &shift_f);
    let ffn_out = swiglu_ffn(w, &format!("{p}.ffn"), &ffn_input);
    let mut x3 = x2.data.clone();
    for idx in 0..x3.len() { x3[idx] += gate_f[idx] * ffn_out.data[idx]; }
    Tensor::new(x3, x.shape.clone())
}

/// scatter_atom_to_token mean over valid atoms. feats[N,d] -> [n_tokens,d].
fn scatter_mean(feats: &Tensor, atom_to_token: &[i64], n_tokens: usize, valid: &[bool]) -> Tensor {
    let n = feats.shape[0];
    let d = feats.shape[1];
    let mut out = vec![0.0f32; n_tokens * d];
    let mut count = vec![0.0f32; n_tokens];
    for a in 0..n {
        if !valid[a] { continue; }
        let t = atom_to_token[a] as usize;
        count[t] += 1.0;
        let src = &feats.data[a * d..a * d + d];
        let dst = &mut out[t * d..t * d + d];
        for c in 0..d { dst[c] += src[c]; }
    }
    for t in 0..n_tokens {
        if count[t] > 0.0 {
            let inv = 1.0 / count[t];
            for c in 0..d { out[t * d + c] *= inv; }
        }
    }
    Tensor::new(out, vec![n_tokens, d])
}

/// Inputs to the atom encoder / inputs embedder (B=1, single sequence).
pub struct AtomInputs<'a> {
    pub ref_pos: &'a [f32],          // [N*3]
    pub ref_space_uid: &'a [f32],    // [N]
    pub ref_charge: &'a [f32],       // [N]
    pub ref_element: &'a [f32],      // [N*128] one-hot, masked
    pub ref_atom_name_chars: &'a [f32], // [N*256] one-hot, masked (reshaped 4*64)
    pub atom_mask: &'a [bool],       // [N]
    pub atom_to_token: &'a [i64],    // [N]
    pub n_atoms: usize,
    pub n_tokens: usize,
}

const ATOM_FEAT_DIM: usize = 389; // 3+1+1+128+256

/// ESMFold2AtomEncoder (structure_prediction=False): -> token features [L,384].
pub fn atom_encoder(w: &Weights, prefix: &str, inp: &AtomInputs) -> Tensor {
    let ap = format!("{prefix}.atom_attention_encoder");
    let c_base = compute_c_base(w, &ap, inp);
    let (cos, sin) = build_3d_rope(inp.ref_pos, inp.ref_space_uid, inp.n_atoms);
    let q = run_atom_transformer(w, &format!("{ap}.atom_transformer"), &c_base, &c_base, &cos, &sin, inp.atom_mask);
    let q_to_a = lin(&q, w, &format!("{ap}.atom_to_token_linear.weight")); // [N,384]
    let mut q_to_a = q_to_a.data;
    for x in q_to_a.iter_mut() { if *x < 0.0 { *x = 0.0; } }
    let q_to_a = Tensor::new(q_to_a, vec![inp.n_atoms, 384]);
    scatter_mean(&q_to_a, inp.atom_to_token, inp.n_tokens, inp.atom_mask)
}

/// Public scatter-mean (atoms -> tokens) for reuse by the diffusion encoder.
pub fn scatter_atoms_to_tokens(feats: &Tensor, atom_to_token: &[i64], n_tokens: usize, valid: &[bool]) -> Tensor {
    scatter_mean(feats, atom_to_token, n_tokens, valid)
}
/// Public gather (tokens -> atoms) for the diffusion decoder.
pub fn gather_tokens_to_atoms(token_feats: &Tensor, atom_to_token: &[i64], n_atoms: usize) -> Tensor {
    let d = token_feats.shape[1];
    let mut out = vec![0.0f32; n_atoms * d];
    for a in 0..n_atoms {
        let t = atom_to_token[a] as usize;
        out[a * d..a * d + d].copy_from_slice(&token_feats.data[t * d..t * d + d]);
    }
    Tensor::new(out, vec![n_atoms, d])
}
/// Public coords linear helper: linear_f64 wrapper.
pub fn lin_f64(x: &Tensor, w: &Weights, name: &str) -> Tensor { lin(x, w, name) }

/// InputsEmbedder: -> [L, 451] = cat[atom_enc(384), aatype(33), profile(33), deletion_mean(1)].
pub fn inputs_embedder(
    w: &Weights,
    inp: &AtomInputs,
    aatype: &Tensor,      // [L,33]
    profile: &Tensor,     // [L,33]
    deletion_mean: &[f32], // [L]
) -> Tensor {
    let a = atom_encoder(w, "inputs_embedder", inp); // [L,384]
    let l = a.shape[0];
    let da = a.shape[1];
    let na = aatype.shape[1];
    let np = profile.shape[1];
    let out_dim = da + na + np + 1;
    let mut out = vec![0.0f32; l * out_dim];
    for i in 0..l {
        let b = i * out_dim;
        out[b..b + da].copy_from_slice(&a.data[i * da..i * da + da]);
        out[b + da..b + da + na].copy_from_slice(&aatype.data[i * na..i * na + na]);
        out[b + da + na..b + da + na + np].copy_from_slice(&profile.data[i * np..i * np + np]);
        out[b + da + na + np] = deletion_mean[i];
    }
    Tensor::new(out, vec![l, out_dim])
}
