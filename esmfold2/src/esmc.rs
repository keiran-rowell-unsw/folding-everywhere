//! ESM-C 6B encoder (frozen PLM backbone), B=1.
//!
//! Architecture (per biohub/ESMC-6B): Embedding(64,2560) -> 80 ×
//! UnifiedTransformerBlock(Pre-LN attention with QK-LayerNorm + RoPE, SwiGLU
//! FFN, residual scaling sqrt(80/36), all bias-free) -> final LayerNorm.
//! Weights are streamed per-layer from the mmap'd sharded checkpoint to keep
//! peak RAM low (~one layer + activations).

use crate::config::*;
use crate::ops::*;
use crate::tensor::Tensor;
use crate::weights::Weights;

const LN_EPS: f32 = 1e-5;

/// Multi-head self-attention (B=1). x:[T,2560] -> [T,2560].
fn attention(x: &Tensor, w: &Weights, p: &str, cos: &[f32], sin: &[f32], seq_id: &[i64]) -> Tensor {
    let t = x.shape[0];
    let d = ESMC_D_MODEL;
    let h = ESMC_N_HEADS;
    let hd = ESMC_HEAD_DIM;

    let ln_w = w.get_vec(&format!("{p}.layernorm_qkv.layer_norm_weight"));
    let ln_b = w.get_vec(&format!("{p}.layernorm_qkv.layer_norm_bias"));
    let qkv_w = w.get(&format!("{p}.layernorm_qkv.weight"));
    let normed = layernorm(x, &ln_w, Some(&ln_b), LN_EPS);
    let qkv = linear(&normed, &qkv_w, None); // [T, 7680]

    // split q,k,v
    let mut q = Tensor::zeros(&[t, d]);
    let mut k = Tensor::zeros(&[t, d]);
    let mut v = Tensor::zeros(&[t, d]);
    for ti in 0..t {
        let row = &qkv.data[ti * 3 * d..ti * 3 * d + 3 * d];
        q.data[ti * d..ti * d + d].copy_from_slice(&row[0..d]);
        k.data[ti * d..ti * d + d].copy_from_slice(&row[d..2 * d]);
        v.data[ti * d..ti * d + d].copy_from_slice(&row[2 * d..3 * d]);
    }
    let q_ln = w.get_vec(&format!("{p}.q_ln.weight"));
    let k_ln = w.get_vec(&format!("{p}.k_ln.weight"));
    let q = layernorm(&q, &q_ln, None, LN_EPS);
    let k = layernorm(&k, &k_ln, None, LN_EPS);

    // RoPE on [1,T,H,HD]
    let q = apply_rope_bshd(&q.clone().reshape(&[1, t, h, hd]), cos, sin);
    let k = apply_rope_bshd(&k.clone().reshape(&[1, t, h, hd]), cos, sin);
    // shapes back to [T,H,HD] views via data layout (contiguous already)
    let scale = (hd as f32).powf(-0.5); // 0.125

    // context [T, H, HD]
    let mut ctx = vec![0.0f32; t * d];
    for hi in 0..h {
        // scores [T,T]
        let mut scores = vec![0.0f32; t * t];
        for i in 0..t {
            let qi = &q.data[(i * h + hi) * hd..(i * h + hi) * hd + hd];
            for jx in 0..t {
                let kj = &k.data[(jx * h + hi) * hd..(jx * h + hi) * hd + hd];
                let mut s = 0.0f32;
                for dd in 0..hd { s += qi[dd] * kj[dd]; }
                s *= scale;
                if seq_id[i] != seq_id[jx] { s = f32::NEG_INFINITY; }
                scores[i * t + jx] = s;
            }
        }
        // softmax over keys (j) per row i
        for i in 0..t {
            let row = &mut scores[i * t..i * t + t];
            let mut m = f32::NEG_INFINITY;
            for &s in row.iter() { if s > m { m = s; } }
            let mut sum = 0.0f32;
            for s in row.iter_mut() { *s = (*s - m).exp(); sum += *s; }
            let inv = 1.0 / sum;
            for s in row.iter_mut() { *s *= inv; }
        }
        // ctx[i,h] = sum_j scores[i,j] * v[j,h]
        for i in 0..t {
            let out = &mut ctx[(i * h + hi) * hd..(i * h + hi) * hd + hd];
            for jx in 0..t {
                let wgt = scores[i * t + jx];
                let vj = &v.data[(jx * h + hi) * hd..(jx * h + hi) * hd + hd];
                for dd in 0..hd { out[dd] += wgt * vj[dd]; }
            }
        }
    }
    let ctx = Tensor::new(ctx, vec![t, d]);
    let out_w = w.get(&format!("{p}.out_proj.weight"));
    linear(&ctx, &out_w, None)
}

/// SwiGLU FFN (B-collapsed). x:[T,2560] -> [T,2560].
fn ffn(x: &Tensor, w: &Weights, p: &str) -> Tensor {
    let ln_w = w.get_vec(&format!("{p}.layer_norm_weight"));
    let ln_b = w.get_vec(&format!("{p}.layer_norm_bias"));
    let fc1 = w.get(&format!("{p}.fc1_weight")); // [13824, 2560]
    let fc2 = w.get(&format!("{p}.fc2_weight")); // [2560, 6912]
    let normed = layernorm(x, &ln_w, Some(&ln_b), LN_EPS);
    let h = linear(&normed, &fc1, None); // [T, 13824]
    let g = swiglu_split(&h); // [T, 6912]
    linear(&g, &fc2, None) // [T, 2560]
}

/// Run ESM-C over a single sequence. Returns either all 81 collected hidden
/// states (`collect_all=true`) or just the final post-norm state.
pub fn forward(w: &Weights, input_ids: &[i64], seq_id: &[i64], collect_all: bool) -> Vec<Tensor> {
    forward_cb(w, input_ids, seq_id, collect_all, &mut |_| {})
}

/// As [`forward`], invoking `prog(layer)` after each of the `ESMC_N_LAYERS`
/// transformer blocks completes (layer counted 1..=ESMC_N_LAYERS). Numerically
/// identical to `forward`; the callback only observes the loop index.
pub fn forward_cb(
    w: &Weights,
    input_ids: &[i64],
    seq_id: &[i64],
    collect_all: bool,
    prog: &mut dyn FnMut(usize),
) -> Vec<Tensor> {
    let t = input_ids.len();
    let d = ESMC_D_MODEL;
    let embed = w.get("esmc.embed.weight"); // [64, 2560]
    let mut x = Tensor::zeros(&[t, d]);
    for (ti, &id) in input_ids.iter().enumerate() {
        let id = id as usize;
        x.data[ti * d..ti * d + d].copy_from_slice(&embed.data[id * d..id * d + d]);
    }
    let (cos, sin) = build_rope_cos_sin(t, ESMC_HEAD_DIM, ESMC_ROPE_BASE);
    let scale = (ESMC_RESIDUE_SCALE.sqrt()) as f32;

    let mut collected: Vec<Tensor> = Vec::new();
    for layer in 0..ESMC_N_LAYERS {
        if collect_all { collected.push(x.clone()); }
        let ap = format!("esmc.transformer.blocks.{layer}.attn");
        let fp = format!("esmc.transformer.blocks.{layer}.ffn");
        let attn_out = attention(&x, w, &ap, &cos, &sin, seq_id);
        for (xi, a) in x.data.iter_mut().zip(&attn_out.data) { *xi += a / scale; }
        let ffn_out = ffn(&x, w, &fp);
        for (xi, fo) in x.data.iter_mut().zip(&ffn_out.data) { *xi += fo / scale; }
        prog(layer + 1);
    }
    let norm_w = w.get_vec("esmc.transformer.norm.weight");
    let normed = layernorm(&x, &norm_w, None, LN_EPS);
    if collect_all {
        collected.push(normed);
        collected
    } else {
        vec![normed]
    }
}
