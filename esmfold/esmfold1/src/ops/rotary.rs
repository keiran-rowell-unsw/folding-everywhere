//! Rotary position embeddings, ESM-2 flavour.
//!
//! inv_freq[i] = 1 / 10000^(2i/dim),  freqs = outer(positions, inv_freq),
//! emb = cat(freqs, freqs); cos/sin of that. apply: x*cos + rotate_half(x)*sin
//! with rotate_half(x) = cat(-x[H:], x[:H]).

/// Returns (cos, sin), each laid out [seq_len, dim] row-major.
///
/// `inv_freq` is computed in fp32 to match torch's `1/(10000**(arange/dim))`
/// exactly; the `t*inv_freq` product and cos/sin are evaluated in f64 then
/// rounded to f32 (correctly-rounded), which minimizes transcendental error vs
/// the PyTorch reference. Residual diff is libm-vs-torch last-bit (~1e-6).
pub fn build_cos_sin(seq_len: usize, dim: usize) -> (Vec<f32>, Vec<f32>) {
    let half = dim / 2;
    let mut inv_freq = vec![0.0f32; half];
    for i in 0..half {
        let expo = (2 * i) as f32 / dim as f32;
        inv_freq[i] = 1.0f32 / libm::powf(10000.0, expo);
    }
    let mut cos = vec![0.0f32; seq_len * dim];
    let mut sin = vec![0.0f32; seq_len * dim];
    for t in 0..seq_len {
        for i in 0..half {
            // torch forms freqs in fp32 (outer(t_fp32, inv_freq_fp32)); replicate
            // that fp32 product, then take cos/sin in f64 for a correctly-rounded f32.
            let f = (t as f32 * inv_freq[i]) as f64;
            let c = f.cos() as f32;
            let s = f.sin() as f32;
            // emb = cat(freqs, freqs): positions i and i+half share the same freq
            cos[t * dim + i] = c;
            cos[t * dim + i + half] = c;
            sin[t * dim + i] = s;
            sin[t * dim + i + half] = s;
        }
    }
    (cos, sin)
}

/// Apply rotary in place to `x` laid out as [n_mat, seq_len, dim] (n_mat = heads,
/// optionally times batch). cos/sin are [seq_len, dim].
pub fn apply_rotary_inplace(x: &mut [f32], n_mat: usize, seq_len: usize, dim: usize, cos: &[f32], sin: &[f32]) {
    let half = dim / 2;
    for m in 0..n_mat {
        for t in 0..seq_len {
            let base = (m * seq_len + t) * dim;
            let crow = &cos[t * dim..t * dim + dim];
            let srow = &sin[t * dim..t * dim + dim];
            // snapshot row so rotate_half reads original values
            let row: Vec<f32> = x[base..base + dim].to_vec();
            for d in 0..dim {
                let rh = if d < half { -row[d + half] } else { row[d - half] };
                x[base + d] = row[d] * crow[d] + rh * srow[d];
            }
        }
    }
}
