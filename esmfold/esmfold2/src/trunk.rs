//! ESMFold2 folding trunk + LM shim + relative-position encoding (deterministic).
//! All Linear weights are torch `[out,in]`; LayerNorm eps = 1e-5 throughout.

use crate::ops::*;
use crate::tensor::Tensor;
use crate::weights::Weights;

const EPS: f32 = 1e-5;

// The folding trunk is a deep, expansive iterative map (input pair amax ~750,
// output ~4000+), so it is highly sensitive to the fp32 accumulation ORDER of
// its GEMMs. We accumulate matmuls in f64 (≈ the ideal real-valued result,
// which torch's blocked CPU sgemm approximates closely) so the Rust trunk
// tracks the reference instead of diverging chaotically.
fn lin(x: &Tensor, w: &Weights, name: &str) -> Tensor {
    linear_f64(x, &w.get(name), None)
}
fn lin_b(x: &Tensor, w: &Weights, wn: &str, bn: &str) -> Tensor {
    let b = w.get(bn);
    linear_f64(x, &w.get(wn), Some(&b))
}
fn ln(x: &Tensor, w: &Weights, wn: &str, bn: &str) -> Tensor {
    layernorm(x, &w.get_vec(wn), Some(&w.get_vec(bn)), EPS)
}

// ---------------------------------------------------------------------------
// Relative-position pair encoding (ResIdxAsymIdSymIdEntityIdEncoding)
// ---------------------------------------------------------------------------
/// inputs each length L (i64). Returns [L,L,256].
pub fn rel_pos(
    w: &Weights,
    residue_index: &[i64],
    asym_id: &[i64],
    sym_id: &[i64],
    entity_id: &[i64],
    token_index: &[i64],
) -> Tensor {
    let l = residue_index.len();
    const NF: usize = 139; // 66 + 66 + 1 + 6
    let mut feats = vec![0.0f32; l * l * NF];
    let clip = |x: i64, lo: i64, hi: i64| -> usize { x.max(lo).min(hi) as usize };
    for i in 0..l {
        for j in 0..l {
            let base = (i * l + j) * NF;
            let same_chain = asym_id[i] == asym_id[j];
            let same_res = residue_index[i] == residue_index[j];
            let same_entity = entity_id[i] == entity_id[j];
            // residue relpos -> one_hot(66) at offset 0
            let dij_res = if same_chain { clip(residue_index[i] - residue_index[j] + 32, 0, 64) } else { 65 };
            feats[base + dij_res] = 1.0;
            // token relpos -> one_hot(66) at offset 66
            let dij_tok = if same_chain && same_res { clip(token_index[i] - token_index[j] + 32, 0, 64) } else { 65 };
            feats[base + 66 + dij_tok] = 1.0;
            // same entity -> 1 scalar at offset 132
            feats[base + 132] = if same_entity { 1.0 } else { 0.0 };
            // chain relpos -> one_hot(6) at offset 133; same_chain -> bin 5
            let dij_chain = if same_chain { 5 } else { clip(sym_id[i] - sym_id[j] + 2, 0, 4) };
            feats[base + 133 + dij_chain] = 1.0;
        }
    }
    let feats = Tensor::new(feats, vec![l, l, NF]);
    lin(&feats, w, "rel_pos.embed.weight")
}

// ---------------------------------------------------------------------------
// SingleToPair: x[L,Din] -> [L,L,out]
// ---------------------------------------------------------------------------
fn single_to_pair(x: &Tensor, w: &Weights, prefix: &str) -> Tensor {
    let xd = lin_b(x, w, &format!("{prefix}.downproject.weight"), &format!("{prefix}.downproject.bias"));
    let l = xd.shape[0];
    let dp = xd.shape[1];
    // cat[ product(i,j), difference(i,j) ] -> [L,L,2*dp]
    let mut cat = vec![0.0f32; l * l * 2 * dp];
    for i in 0..l {
        let xi = &xd.data[i * dp..i * dp + dp];
        for j in 0..l {
            let xj = &xd.data[j * dp..j * dp + dp];
            let base = (i * l + j) * 2 * dp;
            for d in 0..dp {
                cat[base + d] = xi[d] * xj[d];
                cat[base + dp + d] = xi[d] - xj[d];
            }
        }
    }
    let cat = Tensor::new(cat, vec![l, l, 2 * dp]);
    let h = lin_b(&cat, w, &format!("{prefix}.output_mlp.0.weight"), &format!("{prefix}.output_mlp.0.bias"));
    let h = gelu(&h);
    lin_b(&h, w, &format!("{prefix}.output_mlp.2.weight"), &format!("{prefix}.output_mlp.2.bias"))
}

// ---------------------------------------------------------------------------
// LanguageModelShim: hidden_states[L,81,2560] -> [L,L,256]
// ---------------------------------------------------------------------------
pub fn language_model_shim(w: &Weights, hidden_states: &Tensor) -> Tensor {
    let l = hidden_states.shape[0];
    let nl = hidden_states.shape[1]; // 81
    let d = hidden_states.shape[2]; // 2560
    // base_z_linear: LayerNorm(2560) then Linear(2560->256)
    let hs2 = hidden_states.clone().reshape(&[l * nl, d]);
    let normed = ln(&hs2, w, "language_model.base_z_linear.0.weight", "language_model.base_z_linear.0.bias");
    let projected = lin(&normed, w, "language_model.base_z_linear.1.weight"); // [L*81, 256]
    let dz = projected.last();
    // softmax over the 81 layers
    let combine = w.get_vec("language_model.base_z_combine");
    let mut m = f32::NEG_INFINITY;
    for &c in &combine { if c > m { m = c; } }
    let mut sum = 0.0f32;
    let mut wsoft = vec![0.0f32; nl];
    for n in 0..nl { wsoft[n] = (combine[n] - m).exp(); sum += wsoft[n]; }
    for n in 0..nl { wsoft[n] /= sum; }
    // weighted sum over layers -> [L, 256]
    let mut lm_z = vec![0.0f32; l * dz];
    for li in 0..l {
        for n in 0..nl {
            let wn = wsoft[n];
            let src = &projected.data[(li * nl + n) * dz..(li * nl + n) * dz + dz];
            let dst = &mut lm_z[li * dz..li * dz + dz];
            for c in 0..dz { dst[c] += wn * src[c]; }
        }
    }
    let lm_z = Tensor::new(lm_z, vec![l, dz]);
    // base_z_mlp: SingleToPair(256,256,256) then LayerNorm(256)
    let pair = single_to_pair(&lm_z, w, "language_model.base_z_mlp.0");
    ln(&pair, w, "language_model.base_z_mlp.1.weight", "language_model.base_z_mlp.1.bias")
}

// ---------------------------------------------------------------------------
// TriangleMultiplicativeUpdate engine
// ---------------------------------------------------------------------------
fn triangle_contract(left: &[f32], right: &[f32], l: usize, c: usize, outgoing: bool) -> Vec<f32> {
    #[cfg(feature = "native")]
    use rayon::prelude::*;
    let mut out = vec![0.0f32; l * l * c];
    let process = |i: usize, orow: &mut [f32]| {
        let mut acc64 = vec![0.0f64; c];
        for j in 0..l {
            for a in acc64.iter_mut() { *a = 0.0; }
            for k in 0..l {
                let (lo, ro) = if outgoing {
                    ((i * l + k) * c, (j * l + k) * c)
                } else {
                    ((k * l + i) * c, (k * l + j) * c)
                };
                let la = &left[lo..lo + c];
                let rb = &right[ro..ro + c];
                for d in 0..c { acc64[d] += la[d] as f64 * rb[d] as f64; }
            }
            let dst = &mut orow[j * c..j * c + c];
            for d in 0..c { dst[d] = acc64[d] as f32; }
        }
    };
    #[cfg(feature = "native")]
    out.par_chunks_mut(l * c).enumerate().for_each(|(i, orow)| process(i, orow));
    #[cfg(not(feature = "native"))]
    out.chunks_mut(l * c).enumerate().for_each(|(i, orow)| process(i, orow));
    out
}

/// z:[L,L,C], mask:[L,L]. `outgoing` selects the einsum.
pub fn tri_mul(w: &Weights, engine_prefix: &str, z: &Tensor, mask: &[f32], outgoing: bool) -> Tensor {
    let l = z.shape[0];
    let c = z.shape[2];
    let normed = ln(z, w, &format!("{engine_prefix}.norm_start.weight"), &format!("{engine_prefix}.norm_start.bias"));
    let bundled = lin(&normed, w, &format!("{engine_prefix}.proj_bundle.weight")); // [L,L,4*latent]
    let two_lat = bundled.last() / 2; // 2*latent = 512
    let lat = two_lat / 2; // 256
    // routed = signal * sigmoid(gate_logits), masked; then split into left/right
    let mut left = vec![0.0f32; l * l * lat];
    let mut right = vec![0.0f32; l * l * lat];
    for i in 0..l {
        for j in 0..l {
            let b = (i * l + j) * bundled.last();
            let m = mask[i * l + j];
            let row = &bundled.data[b..b + bundled.last()];
            for d in 0..lat {
                let sig = row[d]; // signal first half
                let gate = row[two_lat + d]; // gate_logits = second half (offset 2*latent)
                let routed = sig * sigmoid_scalar(gate) * m;
                left[(i * l + j) * lat + d] = routed;
                let sig2 = row[lat + d];
                let gate2 = row[two_lat + lat + d];
                right[(i * l + j) * lat + d] = sig2 * sigmoid_scalar(gate2) * m;
            }
        }
    }
    let contracted = triangle_contract(&left, &right, l, lat, outgoing);
    let contracted = Tensor::new(contracted, vec![l, l, lat]);
    let mixed = lin(&ln(&contracted, w, &format!("{engine_prefix}.norm_mix.weight"), &format!("{engine_prefix}.norm_mix.bias")),
                    w, &format!("{engine_prefix}.proj_emit.weight"));
    let out_gate = lin(&normed, w, &format!("{engine_prefix}.proj_gate.weight"));
    // return mixed * sigmoid(out_gate)
    let mut out = mixed.data.clone();
    for (o, g) in out.iter_mut().zip(&out_gate.data) { *o *= sigmoid_scalar(*g); }
    Tensor::new(out, vec![l, l, c])
}

// ---------------------------------------------------------------------------
// Transition (SwiGLU MLP with internal residual)
// ---------------------------------------------------------------------------
pub fn transition(w: &Weights, prefix: &str, x: &Tensor) -> Tensor {
    let xn = ln(x, w, &format!("{prefix}.norm.weight"), &format!("{prefix}.norm.bias"));
    let x12 = lin(&xn, w, &format!("{prefix}.ffn.w12.weight"));
    let g = swiglu_split(&x12);
    let out = lin(&g, w, &format!("{prefix}.ffn.w3.weight"));
    x.add(&out)
}

// ---------------------------------------------------------------------------
// PairUpdateBlock / FoldingTrunk
// ---------------------------------------------------------------------------
pub fn pair_update_block(w: &Weights, prefix: &str, pair: &Tensor, mask: &[f32]) -> Tensor {
    let d = tri_mul(w, &format!("{prefix}.tri_mul_out._engine"), pair, mask, true);
    let mut pair = pair.add(&d);
    let d = tri_mul(w, &format!("{prefix}.tri_mul_in._engine"), &pair, mask, false);
    pair = pair.add(&d);
    transition(w, &format!("{prefix}.pair_transition"), &pair)
}

/// Run `n_layers` PairUpdateBlocks under `trunk_prefix` (e.g. "folding_trunk").
pub fn folding_trunk(w: &Weights, trunk_prefix: &str, pair: &Tensor, mask: &[f32], n_layers: usize) -> Tensor {
    folding_trunk_cb(w, trunk_prefix, pair, mask, n_layers, &mut |_| {})
}

/// As [`folding_trunk`], invoking `prog(block)` after each PairUpdateBlock
/// (block counted 1..=n_layers). Numerically identical to `folding_trunk`.
pub fn folding_trunk_cb(
    w: &Weights,
    trunk_prefix: &str,
    pair: &Tensor,
    mask: &[f32],
    n_layers: usize,
    prog: &mut dyn FnMut(usize),
) -> Tensor {
    let mut pair = pair.clone();
    for i in 0..n_layers {
        pair = pair_update_block(w, &format!("{trunk_prefix}.blocks.{i}"), &pair, mask);
        prog(i + 1);
    }
    pair
}
