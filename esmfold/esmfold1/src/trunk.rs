//! Folding trunk: 48 EsmFoldTriangularSelfAttentionBlock + relative position +
//! 4-recycle driver. Single-chain only (mask = all ones, so masking is a no-op).
//!
//! Shapes: s [L, C_S=1024], z [L, L, C_Z=128].

use crate::ops;
use crate::tensor::Tensor;
use crate::weights::Weights;
use rayon::prelude::*;

pub const C_S: usize = 1024;
pub const C_Z: usize = 128;
const SEQ_HEADS: usize = 32;
const SEQ_HW: usize = 32; // sequence_head_width
const PAIR_HEADS: usize = 4;
const PAIR_HW: usize = 32; // pairwise_head_width
const EPS: f32 = 1e-5;
pub const NUM_BLOCKS: usize = 48;
const POS_BINS: i64 = 32;

fn ln(x: &Tensor, w: &Weights, p: &str) -> Tensor {
    ops::layer_norm(x, &w.get(&format!("{p}.weight")), &w.get(&format!("{p}.bias")), EPS)
}
fn lin(x: &Tensor, w: &Weights, p: &str) -> Tensor {
    ops::linear(x, &w.get(&format!("{p}.weight")), Some(&w.get(&format!("{p}.bias"))))
}
fn lin_nb(x: &Tensor, w: &Weights, p: &str) -> Tensor {
    ops::linear(x, &w.get(&format!("{p}.weight")), None)
}
fn add(a: &Tensor, b: &Tensor) -> Tensor {
    Tensor::new(a.data.iter().zip(&b.data).map(|(x, y)| x + y).collect(), a.shape.clone())
}

/// Residue MLP: x + Linear(ReLU(Linear(LN(x)))).
fn residue_mlp(x: &Tensor, w: &Weights, p: &str) -> Tensor {
    let h = ln(x, w, &format!("{p}.mlp.0"));
    let h = lin(&h, w, &format!("{p}.mlp.1"));
    let h = ops::relu(&h);
    let h = lin(&h, w, &format!("{p}.mlp.3"));
    add(x, &h)
}

/// Relative position embedding -> z bias [L,L,C_Z]. diff[i,j] = j - i (clamped).
pub fn relative_position(l: usize, w: &Weights) -> Tensor {
    let emb = w.get("trunk.pairwise_positional_embedding.embedding.weight"); // [2*bins+2, C_Z]
    let mut out = vec![0.0f32; l * l * C_Z];
    for i in 0..l {
        for j in 0..l {
            let mut d = (j as i64) - (i as i64);
            d = d.clamp(-POS_BINS, POS_BINS) + POS_BINS + 1;
            let row = d as usize * C_Z;
            let o = (i * l + j) * C_Z;
            out[o..o + C_Z].copy_from_slice(&emb.data[row..row + C_Z]);
        }
    }
    Tensor::new(out, vec![l, l, C_Z])
}

/// pair_to_sequence: LN(z) -> Linear(no bias) -> bias [L,L,SEQ_HEADS].
fn pair_to_sequence(z: &Tensor, w: &Weights, bp: &str) -> Tensor {
    let zl = ln(z, w, &format!("{bp}.pair_to_sequence.layernorm"));
    lin_nb(&zl, w, &format!("{bp}.pair_to_sequence.linear"))
}

/// Gated sequence self-attention with pair bias. `y` is the LN'd sequence [L,C_S].
fn seq_attention(y: &Tensor, bias: &Tensor, w: &Weights, bp: &str, l: usize) -> Tensor {
    let p = format!("{bp}.seq_attention");
    let proj = lin_nb(y, w, &format!("{p}.proj")); // [L, 3*C_S] = [L, heads*96]
    let scale = (SEQ_HW as f32).powf(-0.5);
    let per_head = 3 * SEQ_HW; // 96
    // output context [L, C_S]
    let mut ctx = vec![0.0f32; l * C_S];
    let pd = &proj.data;
    let bd = &bias.data; // [L,L,SEQ_HEADS]
    ctx.par_chunks_mut(C_S).enumerate().for_each(|(i, crow)| {
        for h in 0..SEQ_HEADS {
            // scores over j
            let mut scores = vec![0.0f32; l];
            let qbase = i * (SEQ_HEADS * per_head) + h * per_head;
            for j in 0..l {
                let kbase = j * (SEQ_HEADS * per_head) + h * per_head + SEQ_HW;
                let mut s = 0.0f32;
                for c in 0..SEQ_HW {
                    s += pd[qbase + c] * scale * pd[kbase + c];
                }
                scores[j] = s + bd[(i * l + j) * SEQ_HEADS + h];
            }
            // softmax
            let m = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            for s in scores.iter_mut() {
                *s = libm::expf(*s - m);
                sum += *s;
            }
            let inv = 1.0 / sum;
            // ctx[i, h*HW + c] = sum_j p_ij * v[j,h,c]
            for c in 0..SEQ_HW {
                let mut acc = 0.0f32;
                for j in 0..l {
                    let vbase = j * (SEQ_HEADS * per_head) + h * per_head + 2 * SEQ_HW;
                    acc += scores[j] * inv * pd[vbase + c];
                }
                crow[h * SEQ_HW + c] = acc;
            }
        }
    });
    let ctx_t = Tensor::new(ctx, vec![l, C_S]);
    // gating: sigmoid(g_proj(y)) * ctx ; then o_proj
    let g = ops::sigmoid(&lin(y, w, &format!("{p}.g_proj")));
    let gated = Tensor::new(ctx_t.data.iter().zip(&g.data).map(|(a, b)| a * b).collect(), vec![l, C_S]);
    lin(&gated, w, &format!("{p}.o_proj"))
}

/// sequence_to_pair: outer product/diff -> pair update [L,L,C_Z].
fn sequence_to_pair(s: &Tensor, w: &Weights, bp: &str, l: usize) -> Tensor {
    let p = format!("{bp}.sequence_to_pair");
    let sl = ln(s, w, &format!("{p}.layernorm"));
    let proj = lin(&sl, w, &format!("{p}.proj")); // [L, C_Z] (inner_dim*2 = 128)
    let inner = C_Z / 2; // 64
    let pd = &proj.data;
    // x[i,j] = cat(prod, diff); prod[i,j,c]=q[j,c]*k[i,c]; diff[i,j,c]=q[j,c]-k[i,c]
    // q = proj[:, :inner], k = proj[:, inner:2*inner]
    let mut x = vec![0.0f32; l * l * (2 * inner)];
    x.par_chunks_mut(l * 2 * inner).enumerate().for_each(|(i, xrow)| {
        for j in 0..l {
            let qj = &pd[j * C_Z..j * C_Z + inner];
            let ki = &pd[i * C_Z + inner..i * C_Z + 2 * inner];
            let base = j * (2 * inner);
            for c in 0..inner {
                xrow[base + c] = qj[c] * ki[c];
                xrow[base + inner + c] = qj[c] - ki[c];
            }
        }
    });
    let xt = Tensor::new(x, vec![l, l, 2 * inner]);
    lin(&xt, w, &format!("{p}.o_proj"))
}

/// Triangle multiplicative update. outgoing: out[i,j,c]=sum_k a[i,k,c]*b[j,k,c];
/// incoming: out[i,j,c]=sum_k a[k,i,c]*b[k,j,c].
fn triangle_mul(z: &Tensor, w: &Weights, bp: &str, outgoing: bool, l: usize) -> Tensor {
    let p = format!("{bp}.{}", if outgoing { "tri_mul_out" } else { "tri_mul_in" });
    let zl = ln(z, w, &format!("{p}.layer_norm_in"));
    let ag = ops::sigmoid(&lin(&zl, w, &format!("{p}.linear_a_g")));
    let ap = lin(&zl, w, &format!("{p}.linear_a_p"));
    let bg = ops::sigmoid(&lin(&zl, w, &format!("{p}.linear_b_g")));
    let bp_ = lin(&zl, w, &format!("{p}.linear_b_p"));
    let a: Vec<f32> = ag.data.iter().zip(&ap.data).map(|(g, v)| g * v).collect();
    let b: Vec<f32> = bg.data.iter().zip(&bp_.data).map(|(g, v)| g * v).collect();
    // contraction
    let mut out = vec![0.0f32; l * l * C_Z];
    out.par_chunks_mut(l * C_Z).enumerate().for_each(|(i, orow)| {
        for j in 0..l {
            let ob = j * C_Z;
            for k in 0..l {
                let (ai, bj) = if outgoing {
                    (&a[(i * l + k) * C_Z..(i * l + k) * C_Z + C_Z], &b[(j * l + k) * C_Z..(j * l + k) * C_Z + C_Z])
                } else {
                    (&a[(k * l + i) * C_Z..(k * l + i) * C_Z + C_Z], &b[(k * l + j) * C_Z..(k * l + j) * C_Z + C_Z])
                };
                for c in 0..C_Z {
                    orow[ob + c] += ai[c] * bj[c];
                }
            }
        }
    });
    let outt = Tensor::new(out, vec![l, l, C_Z]);
    let outt = ln(&outt, w, &format!("{p}.layer_norm_out"));
    let outt = lin(&outt, w, &format!("{p}.linear_z"));
    let g = ops::sigmoid(&lin(&zl, w, &format!("{p}.linear_g")));
    Tensor::new(outt.data.iter().zip(&g.data).map(|(x, y)| x * y).collect(), vec![l, l, C_Z])
}

/// Transpose first two axes of a [L,L,C] tensor.
fn transpose_ij(z: &Tensor, l: usize, c: usize) -> Tensor {
    let mut out = vec![0.0f32; l * l * c];
    for i in 0..l {
        for j in 0..l {
            let src = (j * l + i) * c;
            let dst = (i * l + j) * c;
            out[dst..dst + c].copy_from_slice(&z.data[src..src + c]);
        }
    }
    Tensor::new(out, vec![l, l, c])
}

/// Triangle attention (starting node). For `ending`, caller transposes z first.
fn triangle_attention_start(z: &Tensor, w: &Weights, p: &str, l: usize) -> Tensor {
    let zl = ln(z, w, &format!("{p}.layer_norm"));
    let tb = lin_nb(&zl, w, &format!("{p}.linear")); // [L,L,PAIR_HEADS]
    let q = lin_nb(&zl, w, &format!("{p}.mha.linear_q")); // [L,L,PAIR_HEADS*PAIR_HW]
    let k = lin_nb(&zl, w, &format!("{p}.mha.linear_k"));
    let v = lin_nb(&zl, w, &format!("{p}.mha.linear_v"));
    let g = ops::sigmoid(&lin(&zl, w, &format!("{p}.mha.linear_g"))); // [L,L,H*HW]
    let hd = PAIR_HEADS * PAIR_HW; // 128
    let scale = 1.0f32 / (PAIR_HW as f32).sqrt();
    let (qd, kd, vd, tbd, gd) = (&q.data, &k.data, &v.data, &tb.data, &g.data);
    // output gathered (pre o_proj) [L,L,H*HW]
    let mut o = vec![0.0f32; l * l * hd];
    o.par_chunks_mut(l * hd).enumerate().for_each(|(r, orow)| {
        for h in 0..PAIR_HEADS {
            for a in 0..l {
                // scores over b
                let mut scores = vec![0.0f32; l];
                let qb = (r * l + a) * hd + h * PAIR_HW;
                for b in 0..l {
                    let kb = (r * l + b) * hd + h * PAIR_HW;
                    let mut s = 0.0f32;
                    for c in 0..PAIR_HW {
                        s += qd[qb + c] * scale * kd[kb + c];
                    }
                    // triangle bias tb[a,b,h]
                    scores[b] = s + tbd[(a * l + b) * PAIR_HEADS + h];
                }
                let m = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0.0f32;
                for s in scores.iter_mut() {
                    *s = libm::expf(*s - m);
                    sum += *s;
                }
                let inv = 1.0 / sum;
                for c in 0..PAIR_HW {
                    let mut acc = 0.0f32;
                    for b in 0..l {
                        let vb = (r * l + b) * hd + h * PAIR_HW;
                        acc += scores[b] * inv * vd[vb + c];
                    }
                    let oi = (r * l + a) * hd + h * PAIR_HW + c;
                    orow[(a) * hd + h * PAIR_HW + c] = acc * gd[oi];
                }
            }
        }
    });
    let ot = Tensor::new(o, vec![l, l, hd]);
    lin(&ot, w, &format!("{p}.mha.linear_o"))
}

fn triangle_attention(z: &Tensor, w: &Weights, bp: &str, starting: bool, l: usize) -> Tensor {
    if starting {
        triangle_attention_start(z, w, &format!("{bp}.tri_att_start"), l)
    } else {
        let zt = transpose_ij(z, l, C_Z);
        let out = triangle_attention_start(&zt, w, &format!("{bp}.tri_att_end"), l);
        transpose_ij(&out, l, C_Z)
    }
}

/// One trunk block. Returns (s, z).
pub fn block(s: &Tensor, z: &Tensor, w: &Weights, idx: usize, l: usize) -> (Tensor, Tensor) {
    let bp = format!("trunk.blocks.{idx}");
    // sequence update
    let bias = pair_to_sequence(z, w, &bp);
    let y = ln(s, w, &format!("{bp}.layernorm_1"));
    let y = seq_attention(&y, &bias, w, &bp, l);
    let mut s = add(s, &y);
    s = residue_mlp(&s, w, &format!("{bp}.mlp_seq"));
    // pair update
    let mut z = add(z, &sequence_to_pair(&s, w, &bp, l));
    z = add(&z, &triangle_mul(&z, w, &bp, true, l));
    z = add(&z, &triangle_mul(&z, w, &bp, false, l));
    z = add(&z, &triangle_attention(&z, w, &bp, true, l));
    z = add(&z, &triangle_attention(&z, w, &bp, false, l));
    z = residue_mlp(&z, w, &format!("{bp}.mlp_pair"));
    (s, z)
}

/// trunk_iter: add relpos to z, run 48 blocks. Returns (s, z) and optionally all
/// per-block outputs (for parity).
pub fn trunk_iter(s0: &Tensor, z0: &Tensor, w: &Weights, l: usize, capture: bool) -> (Tensor, Tensor, Vec<(Tensor, Tensor)>) {
    trunk_iter_cb(s0, z0, w, l, capture, &mut |_| {})
}

/// As `trunk_iter`, calling `prog(block_index)` after each of the 48 blocks.
pub fn trunk_iter_cb(s0: &Tensor, z0: &Tensor, w: &Weights, l: usize, capture: bool, prog: &mut dyn FnMut(usize)) -> (Tensor, Tensor, Vec<(Tensor, Tensor)>) {
    let relpos = relative_position(l, w);
    let mut s = s0.clone();
    let mut z = add(z0, &relpos);
    let mut caps = Vec::new();
    for idx in 0..NUM_BLOCKS {
        let (ns, nz) = block(&s, &z, w, idx, l);
        s = ns;
        z = nz;
        if capture {
            caps.push((s.clone(), z.clone()));
        }
        prog(idx + 1);
    }
    (s, z, caps)
}
