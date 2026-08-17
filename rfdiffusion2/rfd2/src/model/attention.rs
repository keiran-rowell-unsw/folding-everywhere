//! `rf2aa/model/layers/Attention_module.py` — every attention flavour the
//! inference path touches.
//!
//! Which flavours those are is decided by the checkpoint, not by the defaults:
//! `use_flash_attention=False` is hard-wired where `RoseTTAFoldModel` builds the
//! simulator, so the MSA column attentions are the **`Old*` einsum versions**,
//! not the `scaled_dot_product_attention` ones. Porting the SDPA variants would
//! have been porting dead code — and worse, they are not numerically identical
//! to the einsum versions, so it would have been *wrong* dead code.
//!
//! Numerics: `F.linear`, `F.softmax` and every `einsum` are patched in
//! `python/pinned.py`, so each accumulates in f64 and rounds to f32 once. The
//! elementwise steps between them (`attn + bias`, `gate * out`, the `* scaling`)
//! are genuinely fp32 in the reference and are fp32 here.

use crate::ops::acc::Acc;
use crate::dropout::nn_dropout;
use crate::nn::{Ctx, LayerNorm, Linear, Params};
use crate::ops::dot_f64;
use crate::ops::elem::{sigmoid_scalar, softmax_dim};
use crate::tensor::Tensor;
use rayon::prelude::*;

/// `1/sqrt(d_hidden)`, computed the way Python does: `1/math.sqrt(d)` in f64,
/// then used as a Python float (f64) multiplying an f32 tensor — torch treats a
/// Python scalar as a wrapped number, so the multiply happens in f32 with the
/// scalar rounded to f32 first.
#[inline]
pub fn scaling(d_hidden: usize) -> f32 {
    (1.0f64 / (d_hidden as f64).sqrt()) as f32
}

// ---------------------------------------------------------------------------
// Attention (used by Templ_emb for pointwise template attention)
// ---------------------------------------------------------------------------

pub struct Attention {
    pub to_q: Linear,
    pub to_k: Linear,
    pub to_v: Linear,
    pub to_out: Linear,
    pub h: usize,
    pub dim: usize,
}

impl Attention {
    pub fn load(p: &Params, n_head: usize, d_hidden: usize) -> Self {
        Attention {
            to_q: Linear::load_nobias(&p.sub("to_q")),
            to_k: Linear::load_nobias(&p.sub("to_k")),
            to_v: Linear::load_nobias(&p.sub("to_v")),
            to_out: Linear::load(&p.sub("to_out")),
            h: n_head,
            dim: d_hidden,
        }
    }

    /// query `[B, Q, d_query]`, key/value `[B, K, d_key]` -> `[B, Q, d_out]`.
    pub fn forward(&self, query: &Tensor, key: &Tensor, value: &Tensor) -> Tensor {
        let b = query.shape[0];
        let q_n = query.shape[1];
        let k_n = key.shape[1];
        let (h, dim) = (self.h, self.dim);
        let s = scaling(dim);

        let mut qp = self.to_q.forward(query); // [B,Q,h*dim]
        let kp = self.to_k.forward(key);
        let vp = self.to_v.forward(value);
        for v in qp.data.iter_mut() {
            *v *= s;
        }

        // attn[b,h,q,k] = sum_d q[b,q,h,d] * k[b,k,h,d]
        let mut attn = vec![0.0f32; b * h * q_n * k_n];
        for bi in 0..b {
            for hh in 0..h {
                for qi in 0..q_n {
                    let qo = ((bi * q_n + qi) * h + hh) * dim;
                    for ki in 0..k_n {
                        let ko = ((bi * k_n + ki) * h + hh) * dim;
                        let mut acc = Acc::new();
                        for d in 0..dim {
                            acc.add(qp.data[qo + d] as f64 * kp.data[ko + d] as f64);
                        }
                        attn[((bi * h + hh) * q_n + qi) * k_n + ki] = acc.get() as f32;
                    }
                }
            }
        }
        let attn = softmax_dim(&Tensor::new(attn, vec![b, h, q_n, k_n]), 3);

        // out[b,q,h,d] = sum_k attn[b,h,q,k] * v[b,k,h,d]
        let mut out = vec![0.0f32; b * q_n * h * dim];
        for bi in 0..b {
            for qi in 0..q_n {
                for hh in 0..h {
                    for d in 0..dim {
                        let mut acc = Acc::new();
                        for ki in 0..k_n {
                            let a = attn.data[((bi * h + hh) * q_n + qi) * k_n + ki] as f64;
                            let v = vp.data[((bi * k_n + ki) * h + hh) * dim + d] as f64;
                            acc.add(a * v);
                        }
                        out[((bi * q_n + qi) * h + hh) * dim + d] = acc.get() as f32;
                    }
                }
            }
        }
        let out = Tensor::new(out, vec![b, q_n, h * dim]);
        self.to_out.forward(&out)
    }
}

// ---------------------------------------------------------------------------
// SequenceWeight + MSARowAttentionWithBias
// ---------------------------------------------------------------------------

pub struct SequenceWeight {
    pub to_query: Linear,
    pub to_key: Linear,
    pub h: usize,
    pub dim: usize,
}

impl SequenceWeight {
    /// Hard-wired in `MSARowAttentionWithBias.__init__`
    /// (`SequenceWeight(..., p_drop=0.1)`), independent of the block's p_drop.
    const P_DROP: f64 = 0.1;

    pub fn load(p: &Params, n_head: usize, d_hidden: usize) -> Self {
        SequenceWeight {
            to_query: Linear::load(&p.sub("to_query")),
            to_key: Linear::load(&p.sub("to_key")),
            h: n_head,
            dim: d_hidden,
        }
    }

    /// msa `[B, N, L, d_msa]` -> attn `[B, N, L, h, 1]`, softmax over N,
    /// then `nn.Dropout(0.1)` — which is applied to the *attention weights*,
    /// not to the values, and is live because the network runs in training mode.
    ///
    /// The `q` index of the einsum output (`bqihd,bkihd->bkihq`) has extent 1
    /// because the query is only the target sequence, so the trailing axis of
    /// the result is a singleton and the softmax is over the *sequence* axis.
    pub fn forward(&self, msa: &Tensor, ctx: &mut Ctx) -> Vec<f32> {
        let (b, n, l) = (msa.shape[0], msa.shape[1], msa.shape[2]);
        let d = msa.last();
        let (h, dim) = (self.h, self.dim);
        let scale = scaling(dim);

        // tar_seq = msa[:, 0]
        let mut tar = vec![0.0f32; b * l * d];
        for bi in 0..b {
            tar[bi * l * d..(bi + 1) * l * d]
                .copy_from_slice(&msa.data[bi * n * l * d..bi * n * l * d + l * d]);
        }
        let mut q = self.to_query.forward(&Tensor::new(tar, vec![b, l, d]));
        for v in q.data.iter_mut() {
            *v *= scale;
        }
        let k = self.to_key.forward(msa);

        let mut attn = vec![0.0f32; b * n * l * h];
        for bi in 0..b {
            for ni in 0..n {
                for li in 0..l {
                    for hh in 0..h {
                        let qo = ((bi * l + li) * h + hh) * dim;
                        let ko = (((bi * n + ni) * l + li) * h + hh) * dim;
                        let mut acc = Acc::new();
                        for dd in 0..dim {
                            acc.add(q.data[qo + dd] as f64 * k.data[ko + dd] as f64);
                        }
                        attn[((bi * n + ni) * l + li) * h + hh] = acc.get() as f32;
                    }
                }
            }
        }
        // softmax over the sequence axis (dim=1 of [B,N,L,h,1])
        let t = Tensor::new(attn, vec![b, n, l, h]);
        let t = softmax_dim(&t, 1);
        nn_dropout(&mut ctx.rng, &t, Self::P_DROP).data
    }
}

pub struct MSARowAttentionWithBias {
    pub norm_msa: LayerNorm,
    pub norm_pair: LayerNorm,
    pub seq_weight: SequenceWeight,
    pub to_q: Linear,
    pub to_k: Linear,
    pub to_v: Linear,
    pub to_b: Linear,
    pub to_g: Linear,
    pub to_out: Linear,
    pub h: usize,
    pub dim: usize,
}

impl MSARowAttentionWithBias {
    pub fn load(p: &Params, n_head: usize, d_hidden: usize) -> Self {
        MSARowAttentionWithBias {
            norm_msa: LayerNorm::load(&p.sub("norm_msa")),
            norm_pair: LayerNorm::load(&p.sub("norm_pair")),
            seq_weight: SequenceWeight::load(&p.sub("seq_weight"), n_head, d_hidden),
            to_q: Linear::load_nobias(&p.sub("to_q")),
            to_k: Linear::load_nobias(&p.sub("to_k")),
            to_v: Linear::load_nobias(&p.sub("to_v")),
            to_b: Linear::load_nobias(&p.sub("to_b")),
            to_g: Linear::load(&p.sub("to_g")),
            to_out: Linear::load(&p.sub("to_out")),
            h: n_head,
            dim: d_hidden,
        }
    }

    /// msa `[B,N,L,d_msa]`, pair `[B,L,L,d_pair]` -> `[B,N,L,d_msa]`.
    pub fn forward(&self, msa: &Tensor, pair: &Tensor, ctx: &mut Ctx) -> Tensor {
        let (b, n, l) = (msa.shape[0], msa.shape[1], msa.shape[2]);
        let (h, dim) = (self.h, self.dim);
        let msa = self.norm_msa.forward(msa);
        let pair = self.norm_pair.forward(pair);

        let sw = self.seq_weight.forward(&msa, ctx); // [B,N,L,h]
        let mut q = self.to_q.forward(&msa);
        let mut k = self.to_k.forward(&msa);
        let v = self.to_v.forward(&msa);
        let bias = self.to_b.forward(&pair); // [B,L,L,h]
        let gate = self.to_g.forward(&msa);

        // query = query * seq_weight (broadcast over the d axis)
        let s = scaling(dim);
        for bi in 0..b {
            for ni in 0..n {
                for li in 0..l {
                    let w = &sw[((bi * n + ni) * l + li) * h..][..h];
                    let off = ((bi * n + ni) * l + li) * h * dim;
                    for hh in 0..h {
                        for dd in 0..dim {
                            q.data[off + hh * dim + dd] *= w[hh];
                            // key scaling is a separate elementwise pass in the
                            // reference; nseq_normalization is False here
                        }
                    }
                    for x in k.data[off..off + h * dim].iter_mut() {
                        *x *= s;
                    }
                }
            }
        }

        // attn[b,q,k,h] = sum_{s,d} query[b,s,q,h,d] * key[b,s,k,h,d]
        let mut attn = vec![0.0f32; b * l * l * h];
        for bi in 0..b {
            for qi in 0..l {
                for ki in 0..l {
                    for hh in 0..h {
                        let mut acc = Acc::new();
                        for si in 0..n {
                            let qo = (((bi * n + si) * l + qi) * h + hh) * dim;
                            let ko = (((bi * n + si) * l + ki) * h + hh) * dim;
                            for dd in 0..dim {
                                acc.add(q.data[qo + dd] as f64 * k.data[ko + dd] as f64);
                            }
                        }
                        attn[((bi * l + qi) * l + ki) * h + hh] = acc.get() as f32;
                    }
                }
            }
        }
        for (i, a) in attn.iter_mut().enumerate() {
            *a += bias.data[i];
        }
        // softmax over the KEY axis, which is dim=-2 of [B,Q,K,h]
        let attn = softmax_dim(&Tensor::new(attn, vec![b, l, l, h]), 2);

        // out[b,s,q,h,d] = sum_k attn[b,q,k,h] * value[b,s,k,h,d]
        let mut out = vec![0.0f32; b * n * l * h * dim];
        for bi in 0..b {
            for si in 0..n {
                for qi in 0..l {
                    for hh in 0..h {
                        for dd in 0..dim {
                            let mut acc = Acc::new();
                            for ki in 0..l {
                                let a = attn.data[((bi * l + qi) * l + ki) * h + hh] as f64;
                                let vv =
                                    v.data[(((bi * n + si) * l + ki) * h + hh) * dim + dd] as f64;
                                acc.add(a * vv);
                            }
                            out[(((bi * n + si) * l + qi) * h + hh) * dim + dd] = acc.get() as f32;
                        }
                    }
                }
            }
        }
        for (i, o) in out.iter_mut().enumerate() {
            *o *= sigmoid_scalar(gate.data[i]);
        }
        let out = Tensor::new(out, vec![b, n, l, h * dim]);
        self.to_out.forward(&out)
    }
}

// ---------------------------------------------------------------------------
// OldMSAColAttention / OldMSAColGlobalAttention
// ---------------------------------------------------------------------------

pub struct MSAColAttention {
    pub norm_msa: LayerNorm,
    pub to_q: Linear,
    pub to_k: Linear,
    pub to_v: Linear,
    pub to_g: Linear,
    pub to_out: Linear,
    pub h: usize,
    pub dim: usize,
    pub global: bool,
}

impl MSAColAttention {
    pub fn load(p: &Params, n_head: usize, d_hidden: usize, global: bool) -> Self {
        MSAColAttention {
            norm_msa: LayerNorm::load(&p.sub("norm_msa")),
            to_q: Linear::load_nobias(&p.sub("to_q")),
            to_k: Linear::load_nobias(&p.sub("to_k")),
            to_v: Linear::load_nobias(&p.sub("to_v")),
            to_g: Linear::load(&p.sub("to_g")),
            to_out: Linear::load(&p.sub("to_out")),
            h: n_head,
            dim: d_hidden,
            global,
        }
    }

    pub fn forward(&self, msa: &Tensor) -> Tensor {
        if self.global {
            self.forward_global(msa)
        } else {
            self.forward_plain(msa)
        }
    }

    fn forward_plain(&self, msa: &Tensor) -> Tensor {
        let (b, n, l) = (msa.shape[0], msa.shape[1], msa.shape[2]);
        let (h, dim) = (self.h, self.dim);
        let msa = self.norm_msa.forward(msa);
        let mut q = self.to_q.forward(&msa);
        let k = self.to_k.forward(&msa);
        let v = self.to_v.forward(&msa);
        let gate = self.to_g.forward(&msa);
        let s = scaling(dim);
        for x in q.data.iter_mut() {
            *x *= s;
        }
        // attn[b,i,h,q,k] = sum_d query[b,q,i,h,d]*key[b,k,i,h,d]
        let mut attn = vec![0.0f32; b * l * h * n * n];
        for bi in 0..b {
            for li in 0..l {
                for hh in 0..h {
                    for qi in 0..n {
                        for ki in 0..n {
                            let qo = (((bi * n + qi) * l + li) * h + hh) * dim;
                            let ko = (((bi * n + ki) * l + li) * h + hh) * dim;
                            let mut acc = Acc::new();
                            for dd in 0..dim {
                                acc.add(q.data[qo + dd] as f64 * k.data[ko + dd] as f64);
                            }
                            attn[(((bi * l + li) * h + hh) * n + qi) * n + ki] = acc.get() as f32;
                        }
                    }
                }
            }
        }
        let attn = softmax_dim(&Tensor::new(attn, vec![b, l, h, n, n]), 4);
        let mut out = vec![0.0f32; b * n * l * h * dim];
        for bi in 0..b {
            for qi in 0..n {
                for li in 0..l {
                    for hh in 0..h {
                        for dd in 0..dim {
                            let mut acc = Acc::new();
                            for ki in 0..n {
                                let a =
                                    attn.data[(((bi * l + li) * h + hh) * n + qi) * n + ki] as f64;
                                let vv =
                                    v.data[(((bi * n + ki) * l + li) * h + hh) * dim + dd] as f64;
                                acc.add(a * vv);
                            }
                            out[(((bi * n + qi) * l + li) * h + hh) * dim + dd] = acc.get() as f32;
                        }
                    }
                }
            }
        }
        for (i, o) in out.iter_mut().enumerate() {
            *o *= sigmoid_scalar(gate.data[i]);
        }
        self.to_out.forward(&Tensor::new(out, vec![b, n, l, h * dim]))
    }

    /// `OldMSAColGlobalAttention`: the query is the **mean over sequences**, and
    /// key/value are single-headed (`to_k`/`to_v` project to `d_hidden`, not
    /// `h*d_hidden`), so the output is broadcast back over N by the gate.
    fn forward_global(&self, msa: &Tensor) -> Tensor {
        let (b, n, l) = (msa.shape[0], msa.shape[1], msa.shape[2]);
        let (h, dim) = (self.h, self.dim);
        let msa = self.norm_msa.forward(msa);
        let qfull = self.to_q.forward(&msa); // [B,N,L,h*dim]
        let k = self.to_k.forward(&msa); // [B,N,L,dim]
        let v = self.to_v.forward(&msa);
        let gate = self.to_g.forward(&msa); // [B,N,L,h*dim]

        // query.mean(dim=1) — `Tensor.mean` is patched, so f64 accumulate + f32
        let mut q = vec![0.0f32; b * l * h * dim];
        for bi in 0..b {
            for li in 0..l {
                for c in 0..h * dim {
                    let mut acc = Acc::new();
                    for ni in 0..n {
                        acc.add(qfull.data[((bi * n + ni) * l + li) * h * dim + c] as f64);
                    }
                    q[(bi * l + li) * h * dim + c] = (acc.get() / n as f64) as f32;
                }
            }
        }
        let s = scaling(dim);
        for x in q.iter_mut() {
            *x *= s;
        }
        // attn[b,i,h,k] = sum_d q[b,i,h,d]*key[b,k,i,d]
        let mut attn = vec![0.0f32; b * l * h * n];
        for bi in 0..b {
            for li in 0..l {
                for hh in 0..h {
                    for ki in 0..n {
                        let mut acc = Acc::new();
                        for dd in 0..dim {
                            let qq = q[(bi * l + li) * h * dim + hh * dim + dd] as f64;
                            let kk = k.data[((bi * n + ki) * l + li) * dim + dd] as f64;
                            acc.add(qq * kk);
                        }
                        attn[((bi * l + li) * h + hh) * n + ki] = acc.get() as f32;
                    }
                }
            }
        }
        let attn = softmax_dim(&Tensor::new(attn, vec![b, l, h, n]), 3);
        // out[b,i,h,d] = sum_k attn[b,i,h,k]*value[b,k,i,d]
        let mut base = vec![0.0f32; b * l * h * dim];
        for bi in 0..b {
            for li in 0..l {
                for hh in 0..h {
                    for dd in 0..dim {
                        let mut acc = Acc::new();
                        for ki in 0..n {
                            let a = attn.data[((bi * l + li) * h + hh) * n + ki] as f64;
                            let vv = v.data[((bi * n + ki) * l + li) * dim + dd] as f64;
                            acc.add(a * vv);
                        }
                        base[((bi * l + li) * h + hh) * dim + dd] = acc.get() as f32;
                    }
                }
            }
        }
        let mut out = vec![0.0f32; b * n * l * h * dim];
        for bi in 0..b {
            for ni in 0..n {
                for li in 0..l {
                    for c in 0..h * dim {
                        let i = ((bi * n + ni) * l + li) * h * dim + c;
                        out[i] = sigmoid_scalar(gate.data[i]) * base[(bi * l + li) * h * dim + c];
                    }
                }
            }
        }
        self.to_out.forward(&Tensor::new(out, vec![b, n, l, h * dim]))
    }
}

// ---------------------------------------------------------------------------
// TriangleMultiplication
// ---------------------------------------------------------------------------

pub struct TriangleMultiplication {
    pub norm: LayerNorm,
    pub left_proj: Linear,
    pub right_proj: Linear,
    pub left_gate: Linear,
    pub right_gate: Linear,
    pub gate: Linear,
    pub norm_out: LayerNorm,
    pub out_proj: Linear,
    pub outgoing: bool,
}

impl TriangleMultiplication {
    pub fn load(p: &Params, outgoing: bool) -> Self {
        TriangleMultiplication {
            norm: LayerNorm::load(&p.sub("norm")),
            left_proj: Linear::load(&p.sub("left_proj")),
            right_proj: Linear::load(&p.sub("right_proj")),
            left_gate: Linear::load(&p.sub("left_gate")),
            right_gate: Linear::load(&p.sub("right_gate")),
            gate: Linear::load(&p.sub("gate")),
            norm_out: LayerNorm::load(&p.sub("norm_out")),
            out_proj: Linear::load(&p.sub("out_proj")),
            outgoing,
        }
    }

    pub fn forward(&self, pair: &Tensor) -> Tensor {
        let (b, l) = (pair.shape[0], pair.shape[1]);
        let pair = self.norm.forward(pair);
        let mut left = self.left_proj.forward(&pair);
        let lg = self.left_gate.forward(&pair);
        for (i, x) in left.data.iter_mut().enumerate() {
            *x *= sigmoid_scalar(lg.data[i]);
        }
        let mut right = self.right_proj.forward(&pair);
        let rg = self.right_gate.forward(&pair);
        for (i, x) in right.data.iter_mut().enumerate() {
            *x *= sigmoid_scalar(rg.data[i]);
        }
        // `right / float(L)` happens BEFORE the contraction, in fp32
        let dh = left.last();
        for x in right.data.iter_mut() {
            *x /= l as f32;
        }
        let mut out = vec![0.0f32; b * l * l * dh];
        for bi in 0..b {
            for i in 0..l {
                for j in 0..l {
                    for d in 0..dh {
                        let mut acc = Acc::new();
                        for k in 0..l {
                            let (li, rj) = if self.outgoing {
                                // 'bikd,bjkd->bijd'
                                (((bi * l + i) * l + k) * dh + d, ((bi * l + j) * l + k) * dh + d)
                            } else {
                                // 'bkid,bkjd->bijd'
                                (((bi * l + k) * l + i) * dh + d, ((bi * l + k) * l + j) * dh + d)
                            };
                            acc.add(left.data[li] as f64 * right.data[rj] as f64);
                        }
                        out[((bi * l + i) * l + j) * dh + d] = acc.get() as f32;
                    }
                }
            }
        }
        let out = Tensor::new(out, vec![b, l, l, dh]);
        let out = self.norm_out.forward(&out);
        let mut out = self.out_proj.forward(&out);
        let g = self.gate.forward(&pair);
        for (i, x) in out.data.iter_mut().enumerate() {
            *x *= sigmoid_scalar(g.data[i]);
        }
        out
    }
}

// ---------------------------------------------------------------------------
// BiasedAxialAttention
// ---------------------------------------------------------------------------

pub struct BiasedAxialAttention {
    pub norm_pair: LayerNorm,
    pub norm_bias: LayerNorm,
    pub to_q: Linear,
    pub to_k: Linear,
    pub to_v: Linear,
    pub to_b: Linear,
    pub to_g: Linear,
    pub to_out: Linear,
    pub h: usize,
    pub dim: usize,
    pub is_row: bool,
}

impl BiasedAxialAttention {
    pub fn load(p: &Params, n_head: usize, d_hidden: usize, is_row: bool) -> Self {
        BiasedAxialAttention {
            norm_pair: LayerNorm::load(&p.sub("norm_pair")),
            norm_bias: LayerNorm::load(&p.sub("norm_bias")),
            to_q: Linear::load_nobias(&p.sub("to_q")),
            to_k: Linear::load_nobias(&p.sub("to_k")),
            to_v: Linear::load_nobias(&p.sub("to_v")),
            to_b: Linear::load_nobias(&p.sub("to_b")),
            to_g: Linear::load(&p.sub("to_g")),
            to_out: Linear::load(&p.sub("to_out")),
            h: n_head,
            dim: d_hidden,
            is_row,
        }
    }

    /// pair/bias `[B,L,L,C]` -> `[B,L,L,C]`.
    ///
    /// The row variant transposes both inputs on the way in and the output on
    /// the way out, so a single implementation covers both; `key /= L` is the
    /// tied-attention normalisation and happens in fp32 before the contraction.
    pub fn forward(&self, pair_in: &Tensor, bias_in: &Tensor) -> Tensor {
        let (b, l) = (pair_in.shape[0], pair_in.shape[1]);
        let (h, dim) = (self.h, self.dim);
        let pair = if self.is_row { pair_in.permute(&[0, 2, 1, 3]) } else { pair_in.clone() };
        let bias = if self.is_row { bias_in.permute(&[0, 2, 1, 3]) } else { bias_in.clone() };
        let pair = self.norm_pair.forward(&pair);
        let bias = self.norm_bias.forward(&bias);

        let mut q = self.to_q.forward(&pair);
        let mut k = self.to_k.forward(&pair);
        let v = self.to_v.forward(&pair);
        let bb = self.to_b.forward(&bias); // [B,L,L,h]
        let gate = self.to_g.forward(&pair);

        let s = scaling(dim);
        for x in q.data.iter_mut() {
            *x *= s;
        }
        for x in k.data.iter_mut() {
            *x /= l as f32;
        }

        // Both contractions below are written against a **regrouped** copy of
        // `q`/`k`/`v`. The natural index order `[b, n, i, h, d]` puts the tied
        // axis `n` on a stride of `l*h*dim` — 54 kB here — so the obvious loop
        // walks memory in 128-byte hops with a single f64 accumulator, which is
        // both cache-hostile and latency-bound. Measured, the two loops were
        // 53 % of a whole trunk block.
        //
        // Regrouping to `[h][i][n*dim + d]` makes each contraction a dot product
        // over a contiguous run, which is what `ops::dot_f64` is for: `LANES`
        // independent chains, so the adds pipeline. The transpose itself is one
        // pass over ~1 M elements.
        let stride = l * dim;
        // `q`/`k` are contracted over the tied axis `n`, so they are grouped as
        // `[h][i][n*dim + d]`; `v` is contracted over `j`, so it is grouped as
        // `[h][n][j*dim + d]`. Same source layout, different major axis.
        let regroup = |src: &Tensor, tied_major: bool| -> Vec<f32> {
            let mut out = vec![0.0f32; b * h * l * stride];
            for bi in 0..b {
                for ni in 0..l {
                    for i in 0..l {
                        for hh in 0..h {
                            let so = (((bi * l + ni) * l + i) * h + hh) * dim;
                            let (major, minor) = if tied_major { (ni, i) } else { (i, ni) };
                            let dof = ((bi * h + hh) * l + major) * stride + minor * dim;
                            out[dof..dof + dim].copy_from_slice(&src.data[so..so + dim]);
                        }
                    }
                }
            }
            out
        };
        let qt = regroup(&q, false);
        let kt = regroup(&k, false);
        let vt = regroup(&v, true);

        // attn[b,i,j,h] = sum_{n,d} q[b,n,i,h,d]*k[b,n,j,h,d]   (tied over n)
        let mut attn = vec![0.0f32; b * l * l * h];
        attn.par_chunks_mut(l * h).enumerate().for_each(|(bi_i, arow)| {
            let (bi, i) = (bi_i / l, bi_i % l);
            for j in 0..l {
                for hh in 0..h {
                    let qo = ((bi * h + hh) * l + i) * stride;
                    let ko = ((bi * h + hh) * l + j) * stride;
                    arow[j * h + hh] =
                        dot_f64(&qt[qo..qo + stride], &kt[ko..ko + stride], stride) as f32;
                }
            }
        });
        for (i, a) in attn.iter_mut().enumerate() {
            *a += bb.data[i];
        }
        let attn = softmax_dim(&Tensor::new(attn, vec![b, l, l, h]), 2);

        // out[b,n,i,h,d] = sum_j attn[b,i,j,h]*v[b,n,j,h,d]
        //
        // `d` is the contiguous axis of the regrouped `v`, so accumulating a
        // whole `dim`-wide row at a time gives `dim` independent chains for
        // free and reads `v` linearly.
        let mut out = vec![0.0f32; b * l * l * h * dim];
        out.par_chunks_mut(l * h * dim).enumerate().for_each(|(bi_n, orow)| {
            let (bi, ni) = (bi_n / l, bi_n % l);
            let mut acc = vec![Acc::new(); dim];
            for i in 0..l {
                for hh in 0..h {
                    acc.iter_mut().for_each(|a| *a = Acc::new());
                    let vo = ((bi * h + hh) * l + ni) * stride;
                    for j in 0..l {
                        let a = attn.data[((bi * l + i) * l + j) * h + hh] as f64;
                        let vr = &vt[vo + j * dim..vo + j * dim + dim];
                        debug_assert_eq!(vr.len(), dim);
                        for d in 0..dim {
                            acc[d].add(a * vr[d] as f64);
                        }
                    }
                    let o = (i * h + hh) * dim;
                    for d in 0..dim {
                        orow[o + d] = acc[d].get() as f32;
                    }
                }
            }
        });
        for (i, o) in out.iter_mut().enumerate() {
            *o *= sigmoid_scalar(gate.data[i]);
        }
        let out = self.to_out.forward(&Tensor::new(out, vec![b, l, l, h * dim]));
        if self.is_row {
            out.permute(&[0, 2, 1, 3])
        } else {
            out
        }
    }
}
