//! ESMFold2 MSA encoder (runs each parcae loop on the 1-row query MSA and
//! overwrites z_inject). 4 blocks: OuterProductMean -> [MSAPairWeightedAveraging
//! + msa_transition (non-final)] -> tri_mul_out/in -> pair_transition.

use crate::ops::*;
use crate::tensor::Tensor;
use crate::trunk;
use crate::weights::Weights;

const D_MSA: usize = 128;
const D_PAIR: usize = 256;
const D_HIDDEN: usize = 32; // OPM
const N_HEADS: usize = 8; // MSAPWA
const HEAD_W: usize = 16;
const EPS: f32 = 1e-5;

fn lin(x: &Tensor, w: &Weights, name: &str) -> Tensor { linear_f64(x, &w.get(name), None) }
fn lin_b(x: &Tensor, w: &Weights, wn: &str, bn: &str) -> Tensor {
    let b = w.get(bn); linear_f64(x, &w.get(wn), Some(&b))
}
fn ln(x: &Tensor, w: &Weights, wn: &str, bn: &str) -> Tensor {
    layernorm(x, &w.get_vec(wn), Some(&w.get_vec(bn)), EPS)
}

/// OuterProductMean: m[L,M,128], mask[L,M] -> pair[L,L,256].
/// default: Wout(outer)/n_valid (bias divided too).
fn outer_product_mean(w: &Weights, p: &str, m: &Tensor, mask: &[f32], l: usize, mm: usize) -> Tensor {
    let m_norm = ln(m, w, &format!("{p}.norm.weight"), &format!("{p}.norm.bias"));
    let x = lin(&m_norm, w, &format!("{p}.W.weight")); // [L,M,64]
    // a,b = chunk; masked
    let mut a = vec![0.0f32; l * mm * D_HIDDEN];
    let mut b = vec![0.0f32; l * mm * D_HIDDEN];
    for i in 0..l {
        for s in 0..mm {
            let mk = mask[i * mm + s];
            let row = &x.data[(i * mm + s) * 2 * D_HIDDEN..(i * mm + s) * 2 * D_HIDDEN + 2 * D_HIDDEN];
            for d in 0..D_HIDDEN {
                a[(i * mm + s) * D_HIDDEN + d] = row[d] * mk;
                b[(i * mm + s) * D_HIDDEN + d] = row[D_HIDDEN + d] * mk;
            }
        }
    }
    // outer[i,j, c*32+d] = sum_m a[i,m,c]*b[j,m,d]  (f64 accum)
    let hh = D_HIDDEN * D_HIDDEN;
    let mut outer = vec![0.0f32; l * l * hh];
    for i in 0..l {
        for j in 0..l {
            let mut acc = vec![0.0f64; hh];
            for s in 0..mm {
                let ai = &a[(i * mm + s) * D_HIDDEN..(i * mm + s) * D_HIDDEN + D_HIDDEN];
                let bj = &b[(j * mm + s) * D_HIDDEN..(j * mm + s) * D_HIDDEN + D_HIDDEN];
                for c in 0..D_HIDDEN {
                    let av = ai[c] as f64;
                    for d in 0..D_HIDDEN { acc[c * D_HIDDEN + d] += av * bj[d] as f64; }
                }
            }
            let base = (i * l + j) * hh;
            for k in 0..hh { outer[base + k] = acc[k] as f32; }
        }
    }
    let outer = Tensor::new(outer, vec![l, l, hh]);
    let proj = lin_b(&outer, w, &format!("{p}.Wout.weight"), &format!("{p}.Wout.bias")); // [L,L,256]
    // n_valid[i,j] = sum_m mask[i,m]*mask[j,m], clamp min 1
    let mut out = proj.data;
    for i in 0..l {
        for j in 0..l {
            let mut nv = 0.0f32;
            for s in 0..mm { nv += mask[i * mm + s] * mask[j * mm + s]; }
            if nv < 1.0 { nv = 1.0; }
            let inv = 1.0 / nv;
            let base = (i * l + j) * D_PAIR;
            for d in 0..D_PAIR { out[base + d] *= inv; }
        }
    }
    Tensor::new(out, vec![l, l, D_PAIR])
}

/// MSAPairWeightedAveraging: m[L,M,128], pair[L,L,256], pair_mask[L,L] -> [L,M,128].
fn msa_pwa(w: &Weights, p: &str, m: &Tensor, pair: &Tensor, pair_mask: &[f32], l: usize, mm: usize) -> Tensor {
    let m_norm = ln(m, w, &format!("{p}.norm_single.weight"), &format!("{p}.norm_single.bias"));
    // bias = compute_bias(pair): LN(pair) then Linear(256->8)
    let pn = ln(pair, w, &format!("{p}.compute_bias.0.weight"), &format!("{p}.compute_bias.0.bias"));
    let bias = lin(&pn, w, &format!("{p}.compute_bias.1.weight")); // [L,L,8]
    // attn[i,j,h] = softmax over j of (bias masked with -1e5)
    let mut attn = vec![0.0f32; l * l * N_HEADS];
    for i in 0..l {
        for h in 0..N_HEADS {
            // gather over j
            let mut mx = f32::NEG_INFINITY;
            let mut tmp = vec![0.0f32; l];
            for j in 0..l {
                let mut bv = bias.data[(i * l + j) * N_HEADS + h];
                if pair_mask[i * l + j] == 0.0 { bv = -1e5; }
                tmp[j] = bv;
                if bv > mx { mx = bv; }
            }
            let mut sum = 0.0f32;
            for j in 0..l { tmp[j] = (tmp[j] - mx).exp(); sum += tmp[j]; }
            let inv = 1.0 / sum;
            for j in 0..l { attn[(i * l + j) * N_HEADS + h] = tmp[j] * inv; }
        }
    }
    let v = lin(&m_norm, w, &format!("{p}.Wv.weight")); // [L,M,128] = [L,M,8*16]
    let gate_lin = lin(&m_norm, w, &format!("{p}.Wgate.weight"));
    // output[i,m,h,d] = sigmoid(gate[i,m,h,d]) * sum_j attn[i,j,h]*v[j,m,h,d]
    let mut out = vec![0.0f32; l * mm * N_HEADS * HEAD_W];
    for i in 0..l {
        for s in 0..mm {
            for h in 0..N_HEADS {
                for d in 0..HEAD_W {
                    let mut acc = 0.0f32;
                    for j in 0..l {
                        let aw = attn[(i * l + j) * N_HEADS + h];
                        let vv = v.data[((j * mm + s) * N_HEADS + h) * HEAD_W + d];
                        acc += aw * vv;
                    }
                    let g = sigmoid_scalar(gate_lin.data[((i * mm + s) * N_HEADS + h) * HEAD_W + d]);
                    out[((i * mm + s) * N_HEADS + h) * HEAD_W + d] = g * acc;
                }
            }
        }
    }
    let out = Tensor::new(out, vec![l * mm, N_HEADS * HEAD_W]);
    let res = lin(&out, w, &format!("{p}.Wout.weight"));
    res.reshape(&[l, mm, D_MSA])
}

/// Full MSA encoder. Returns the updated pair [L,L,256].
pub fn encode(
    w: &Weights,
    x_pair: &Tensor,   // [L,L,256] = z_init
    x_inputs: &Tensor, // [L,451]
    msa_oh: &Tensor,   // [L,M,33]
    has_deletion: &[f32], // [L,M]
    deletion_value: &[f32], // [L,M]
    msa_attn: &[f32],  // [L,M]
) -> Tensor {
    let l = x_pair.shape[0];
    let mm = msa_oh.shape[1];
    let oh = msa_oh.shape[2]; // 33
    // m_feat = cat([msa_oh, has_deletion, deletion_value], -1) [L,M,35]
    let mf = oh + 2;
    let mut m_feat = vec![0.0f32; l * mm * mf];
    for i in 0..l {
        for s in 0..mm {
            let b = (i * mm + s) * mf;
            m_feat[b..b + oh].copy_from_slice(&msa_oh.data[(i * mm + s) * oh..(i * mm + s) * oh + oh]);
            m_feat[b + oh] = has_deletion[i * mm + s];
            m_feat[b + oh + 1] = deletion_value[i * mm + s];
        }
    }
    let m_feat = Tensor::new(m_feat, vec![l * mm, mf]);
    let m_embed = lin(&m_feat, w, "msa_encoder.embed.weight").reshape(&[l, mm, D_MSA]);
    let proj = lin(x_inputs, w, "msa_encoder.project_inputs.weight"); // [L,128]
    // m = m_embed + proj.unsqueeze(2)
    let mut m = m_embed.data;
    for i in 0..l {
        for s in 0..mm {
            let pb = i * D_MSA;
            let mb = (i * mm + s) * D_MSA;
            for d in 0..D_MSA { m[mb + d] += proj.data[pb + d]; }
        }
    }
    let mut m = Tensor::new(m, vec![l, mm, D_MSA]);
    let mut pair = x_pair.clone();
    // pair_attention_mask = tok_mask[i] & tok_mask[j], tok_mask = msa_attn[:,0]
    let mut pair_mask = vec![0.0f32; l * l];
    for i in 0..l {
        for j in 0..l {
            let ti = msa_attn[i * mm] != 0.0;
            let tj = msa_attn[j * mm] != 0.0;
            pair_mask[i * l + j] = if ti && tj { 1.0 } else { 0.0 };
        }
    }
    for blk in 0..4 {
        let p = format!("msa_encoder.blocks.{blk}");
        let opm = outer_product_mean(w, &format!("{p}.outer_product_mean"), &m, msa_attn, l, mm);
        pair = pair.add(&opm);
        if blk != 3 {
            let pwa = msa_pwa(w, &format!("{p}.msa_pair_weighted_averaging"), &m, &pair, &pair_mask, l, mm);
            m = m.add(&pwa);
            // msa_transition (PairTransition on d_msa) — reuse trunk::transition
            let mt = trunk::transition(w, &format!("{p}.msa_transition"), &m);
            m = mt; // transition has internal residual already (x + ffn(norm(x)))
        }
        let d = trunk::tri_mul(w, &format!("{p}.tri_mul_out._engine"), &pair, &pair_mask, true);
        pair = pair.add(&d);
        let d = trunk::tri_mul(w, &format!("{p}.tri_mul_in._engine"), &pair, &pair_mask, false);
        pair = pair.add(&d);
        pair = trunk::transition(w, &format!("{p}.pair_transition"), &pair);
    }
    pair
}
