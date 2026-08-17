//! 1-D rotary position embeddings (RoPE), GPT-NeoX / rotate-half convention,
//! matching ESM-C's `RotaryEmbedding` (interleaved=False, pos_idx_in_fp32=True).

use crate::tensor::Tensor;

/// Build cos/sin caches of shape [seq, head_dim/2], computed in fp32.
/// inv_freq[i] = 1 / base^(2i/head_dim); freqs[p,i] = p * inv_freq[i].
pub fn build_rope_cos_sin(seq: usize, head_dim: usize, base: f32) -> (Vec<f32>, Vec<f32>) {
    let half = head_dim / 2;
    let mut inv_freq = vec![0.0f32; half];
    for i in 0..half {
        let exp = (2 * i) as f32 / head_dim as f32;
        inv_freq[i] = 1.0 / base.powf(exp);
    }
    let mut cos = vec![0.0f32; seq * half];
    let mut sin = vec![0.0f32; seq * half];
    for p in 0..seq {
        for i in 0..half {
            let f = p as f32 * inv_freq[i];
            cos[p * half + i] = f.cos();
            sin[p * half + i] = f.sin();
        }
    }
    (cos, sin)
}

/// Apply RoPE to a tensor shaped [B, S, H, D] (D = head_dim, even).
/// cos/sin are [S, D/2].
pub fn apply_rope_bshd(x: &Tensor, cos: &[f32], sin: &[f32]) -> Tensor {
    assert_eq!(x.ndim(), 4);
    let (b, s, h, d) = (x.shape[0], x.shape[1], x.shape[2], x.shape[3]);
    let half = d / 2;
    let mut out = vec![0.0f32; x.numel()];
    for bi in 0..b {
        for si in 0..s {
            let cs = &cos[si * half..si * half + half];
            let sn = &sin[si * half..si * half + half];
            for hi in 0..h {
                let base = (((bi * s + si) * h) + hi) * d;
                let xr = &x.data[base..base + d];
                let orow = &mut out[base..base + d];
                for j in 0..half {
                    orow[j] = xr[j] * cs[j] - xr[j + half] * sn[j];
                    orow[j + half] = xr[j + half] * cs[j] + xr[j] * sn[j];
                }
            }
        }
    }
    Tensor::new(out, x.shape.clone())
}
