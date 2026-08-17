//! `rf2aa/model/Track_module.py` — the three-track update blocks.
//!
//! One `IterBlock` is: MSA <- (pair, structure) ; pair <- MSA ; pair <- pair ;
//! structure <- everything. The simulator runs 4 `extra_block`s over the
//! reduced-width MSA, then 32 `main_block`s, then 4 passes of the standalone
//! `str_refiner`. That last group is 93 % of the parameters.

use crate::ops::acc::Acc;
use crate::model::attention::{MSAColAttention, MSARowAttentionWithBias};
use crate::model::embeddings::{PairStr2Pair, PositionalEncoding2D};
use crate::dropout::rf_dropout;
use crate::nn::{Ctx, FeedForward, LayerNorm, Linear, Params};
use crate::ops::relu_;
use crate::tensor::Tensor;

// ---------------------------------------------------------------------------
// MSAPairStr2MSA
// ---------------------------------------------------------------------------

pub struct MSAPairStr2MSA {
    pub norm_pair: LayerNorm,
    pub emb_rbf: Linear,
    pub norm_state: LayerNorm,
    pub proj_state: Linear,
    pub row_attn: MSARowAttentionWithBias,
    pub col_attn: MSAColAttention,
    pub ff: FeedForward,
    pub p_drop: f64,
}

impl MSAPairStr2MSA {
    pub fn load(
        p: &Params,
        n_head: usize,
        d_hidden: usize,
        use_global_attn: bool,
        p_drop: f64,
    ) -> Self {
        MSAPairStr2MSA {
            norm_pair: LayerNorm::load(&p.sub("norm_pair")),
            emb_rbf: Linear::load(&p.sub("emb_rbf")),
            norm_state: LayerNorm::load(&p.sub("norm_state")),
            proj_state: Linear::load(&p.sub("proj_state")),
            row_attn: MSARowAttentionWithBias::load(&p.sub("row_attn"), n_head, d_hidden),
            col_attn: MSAColAttention::load(&p.sub("col_attn"), n_head, d_hidden, use_global_attn),
            // `FeedForwardLayer(d_msa, 4, p_drop=p_drop)` — here the block's
            // p_drop *is* forwarded, unlike PairStr2Pair's.
            ff: FeedForward::load(&p.sub("ff"), p_drop),
            p_drop,
        }
    }

    pub fn forward(
        &self,
        msa: &Tensor,      // [B,N,L,d_msa]
        pair: &Tensor,     // [B,L,L,d_pair]
        rbf_feat: &Tensor, // [B,L,L,64]
        state: &Tensor,    // [B,L,d_state]
        ctx: &mut Ctx,
    ) -> Tensor {
        let (b, n, l) = (msa.shape[0], msa.shape[1], msa.shape[2]);
        let dm = msa.last();
        let mut pair = self.norm_pair.forward(pair);
        let rbf = self.emb_rbf.forward(rbf_feat);
        for (i, v) in pair.data.iter_mut().enumerate() {
            *v += rbf.data[i];
        }

        let st = self.norm_state.forward(state);
        let st = self.proj_state.forward(&st); // [B,L,d_msa]

        // `msa.index_add(1, [0], state)` — only sequence 0 gets the state.
        let mut msa = msa.clone();
        for bi in 0..b {
            for li in 0..l {
                let o = (bi * n * l + li) * dm;
                for c in 0..dm {
                    msa.data[o + c] += st.data[(bi * l + li) * dm + c];
                }
            }
        }

        let d = self.row_attn.forward(&msa, &pair, ctx);
        let d = rf_dropout(&mut ctx.rng, &d, Some(1), self.p_drop);
        for (i, v) in msa.data.iter_mut().enumerate() {
            *v += d.data[i];
        }
        let d = self.col_attn.forward(&msa);
        for (i, v) in msa.data.iter_mut().enumerate() {
            *v += d.data[i];
        }
        let d = self.ff.forward(&msa, ctx);
        for (i, v) in msa.data.iter_mut().enumerate() {
            *v += d.data[i];
        }
        msa
    }
}

// ---------------------------------------------------------------------------
// MSA2Pair
// ---------------------------------------------------------------------------

pub struct MSA2Pair {
    pub norm: LayerNorm,
    pub proj_left: Linear,
    pub proj_right: Linear,
    pub proj_out: Linear,
}

impl MSA2Pair {
    pub fn load(p: &Params) -> Self {
        MSA2Pair {
            norm: LayerNorm::load(&p.sub("norm")),
            proj_left: Linear::load(&p.sub("proj_left")),
            proj_right: Linear::load(&p.sub("proj_right")),
            proj_out: Linear::load(&p.sub("proj_out")),
        }
    }

    pub fn forward(&self, msa: &Tensor, pair: &Tensor) -> Tensor {
        let (b, n, l) = (msa.shape[0], msa.shape[1], msa.shape[2]);
        let msa = self.norm.forward(msa);
        let left = self.proj_left.forward(&msa);
        let mut right = self.proj_right.forward(&msa);
        let dh = left.last();
        // `right / float(N)` in fp32 before the contraction
        for v in right.data.iter_mut() {
            *v /= n as f32;
        }
        let mut out = vec![0.0f32; b * l * l * dh * dh];
        for bi in 0..b {
            for li in 0..l {
                for mi in 0..l {
                    let o = ((bi * l + li) * l + mi) * dh * dh;
                    for i in 0..dh {
                        for j in 0..dh {
                            let mut acc = Acc::new();
                            for si in 0..n {
                                let lv =
                                    left.data[((bi * n + si) * l + li) * dh + i] as f64;
                                let rv =
                                    right.data[((bi * n + si) * l + mi) * dh + j] as f64;
                                acc.add(lv * rv);
                            }
                            out[o + i * dh + j] = acc.get() as f32;
                        }
                    }
                }
            }
        }
        let out = self
            .proj_out
            .forward(&Tensor::new(out, vec![b, l, l, dh * dh]));
        let mut pair = pair.clone();
        for (i, v) in pair.data.iter_mut().enumerate() {
            *v += out.data[i];
        }
        pair
    }
}

// ---------------------------------------------------------------------------
// SCPred
// ---------------------------------------------------------------------------

pub struct SCPred {
    pub norm_s0: LayerNorm,
    pub norm_si: LayerNorm,
    pub linear_s0: Linear,
    pub linear_si: Linear,
    pub linear_1: Linear,
    pub linear_2: Linear,
    pub linear_3: Linear,
    pub linear_4: Linear,
    pub linear_out: Linear,
}

impl SCPred {
    pub fn load(p: &Params) -> Self {
        SCPred {
            norm_s0: LayerNorm::load(&p.sub("norm_s0")),
            norm_si: LayerNorm::load(&p.sub("norm_si")),
            linear_s0: Linear::load(&p.sub("linear_s0")),
            linear_si: Linear::load(&p.sub("linear_si")),
            linear_1: Linear::load(&p.sub("linear_1")),
            linear_2: Linear::load(&p.sub("linear_2")),
            linear_3: Linear::load(&p.sub("linear_3")),
            linear_4: Linear::load(&p.sub("linear_4")),
            linear_out: Linear::load(&p.sub("linear_out")),
        }
    }

    /// seq `[B,L,d_msa]` (i.e. `msa[:,0]`, **unnormalised** on the way in),
    /// state `[B,L,d_state]` -> `[B,L,NTOTALDOFS,2]`.
    pub fn forward(&self, seq: &Tensor, state: &Tensor) -> Tensor {
        let (b, l) = (seq.shape[0], seq.shape[1]);
        let s = self.norm_s0.forward(seq);
        let st = self.norm_si.forward(state);
        let a = self.linear_s0.forward(&s);
        let bb = self.linear_si.forward(&st);
        let mut si = Tensor::new(
            a.data.iter().zip(&bb.data).map(|(x, y)| x + y).collect(),
            a.shape.clone(),
        );

        // `si = si + linear_2(relu_(linear_1(relu_(si))))` — and `F.relu_` is
        // IN-PLACE, so the `si` on the left of the `+` is the *rectified* one.
        // Reading it as `si + f(relu(si))` (i.e. treating relu_ as pure) gives a
        // different residual, and one that still trains-looking-fine.
        for (l1, l2) in [(&self.linear_1, &self.linear_2), (&self.linear_3, &self.linear_4)] {
            relu_(&mut si);
            let mut h = l1.forward(&si);
            relu_(&mut h);
            let h = l2.forward(&h);
            for (i, v) in si.data.iter_mut().enumerate() {
                *v += h.data[i];
            }
        }
        relu_(&mut si);
        let out = self.linear_out.forward(&si);
        let n = out.last() / 2;
        Tensor::new(out.data, vec![b, l, n, 2])
    }
}

// ---------------------------------------------------------------------------
// IterBlock plumbing that is not the SE(3) transformer
// ---------------------------------------------------------------------------

pub struct IterBlockTrunk {
    pub pos: PositionalEncoding2D,
    pub msa2msa: MSAPairStr2MSA,
    pub msa2pair: MSA2Pair,
    pub pair2pair: PairStr2Pair,
}

impl IterBlockTrunk {
    pub fn load(
        p: &Params,
        n_head_msa: usize,
        d_hidden_msa: usize,
        n_head_pair: usize,
        d_hidden: usize,
        use_global_attn: bool,
        enable_same_chain: bool,
        p_drop: f64,
    ) -> Self {
        IterBlockTrunk {
            pos: PositionalEncoding2D::load(&p.sub("pos"), enable_same_chain),
            msa2msa: MSAPairStr2MSA::load(
                &p.sub("msa2msa"),
                n_head_msa,
                d_hidden_msa,
                use_global_attn,
                p_drop,
            ),
            msa2pair: MSA2Pair::load(&p.sub("msa2pair")),
            pair2pair: PairStr2Pair::load(&p.sub("pair2pair"), n_head_pair, d_hidden, p_drop),
        }
    }
}
