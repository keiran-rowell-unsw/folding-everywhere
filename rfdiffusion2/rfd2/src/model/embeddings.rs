//! `rf2aa/model/layers/Embeddings.py` plus `Track_module.PositionalEncoding2D`.
//!
//! Which classes are live is decided by the checkpoint config, not the module
//! defaults:
//!
//! * `recycling_type = "all"` -> `RecyclingAllFeatures`, **not** `Recycling`.
//!   The two share a name prefix and differ in what they project, so picking the
//!   wrong one loads with no missing keys and silently drops the sidechain-
//!   torsion and state contributions.
//! * `enable_same_chain = True` -> `PositionalEncoding2D` really does add
//!   `emb_chain(same_chain)`. In the `use_same_chain && !enable_same_chain`
//!   configuration upstream adds `emb_c * 0` instead ("cursed but exists for
//!   backwards compatibility"), which is why the weight exists in checkpoints
//!   that never use it.

use crate::chemical_gen::NBTYPES;
use crate::geom;
use crate::model::attention::{Attention, BiasedAxialAttention, TriangleMultiplication};
use crate::dropout::rf_dropout;
use crate::nn::{Ctx, FeedForward, LayerNorm, Linear, Params};
use crate::ops::elem::sigmoid_scalar;
use crate::ops::relu_;
use crate::tensor::Tensor;

// ---------------------------------------------------------------------------
// PositionalEncoding2D
// ---------------------------------------------------------------------------

pub struct PositionalEncoding2D {
    pub emb_res: Tensor,   // [66, C]
    pub emb_atom: Tensor,  // [10, C]
    pub emb_chain: Tensor, // [2, C]
    pub minpos: i64,
    pub maxpos: i64,
    pub maxpos_atom: i64,
    pub enable_same_chain: bool,
}

impl PositionalEncoding2D {
    pub fn load(p: &Params, enable_same_chain: bool) -> Self {
        PositionalEncoding2D {
            emb_res: p.sub("emb_res").get("weight"),
            emb_atom: p.sub("emb_atom").get("weight"),
            emb_chain: p.sub("emb_chain").get("weight"),
            minpos: -32,
            maxpos: 32,
            maxpos_atom: 8,
            enable_same_chain,
        }
    }

    pub fn forward(
        &self,
        seq: &[i64],
        idx: &[i64],
        bond_feats: &[i64],
        dist_matrix: &[f32],
        same_chain: &[bool],
    ) -> Tensor {
        let l = seq.len();
        let c = self.emb_res.shape[1];
        let sm_mask: Vec<bool> = seq.iter().map(|&t| geom::is_atom(t)).collect();
        let d = geom::res_atom_dist(
            idx,
            bond_feats,
            dist_matrix,
            &sm_mask,
            self.minpos,
            self.maxpos,
            self.maxpos_atom,
        );
        let mut out = vec![0.0f32; l * l * c];
        for k in 0..l * l {
            let ir = geom::bucketize(d.res[k], self.minpos, self.maxpos);
            let ia = geom::bucketize(d.atom[k], 0, self.maxpos_atom);
            let o = k * c;
            for ci in 0..c {
                out[o + ci] = self.emb_res.data[ir * c + ci] + self.emb_atom.data[ia * c + ci];
            }
            if self.enable_same_chain {
                let ic = same_chain[k] as usize;
                for ci in 0..c {
                    out[o + ci] += self.emb_chain.data[ic * c + ci];
                }
            }
        }
        Tensor::new(out, vec![1, l, l, c])
    }
}

// ---------------------------------------------------------------------------
// MSA_emb / Extra_emb / Bond_emb
// ---------------------------------------------------------------------------

pub struct MsaEmb {
    pub emb: Linear,
    pub emb_q: Tensor,
    pub emb_left: Tensor,
    pub emb_right: Tensor,
    pub emb_state: Tensor,
    pub pos: PositionalEncoding2D,
}

impl MsaEmb {
    pub fn load(p: &Params, enable_same_chain: bool) -> Self {
        MsaEmb {
            emb: Linear::load(&p.sub("emb")),
            emb_q: p.sub("emb_q").get("weight"),
            emb_left: p.sub("emb_left").get("weight"),
            emb_right: p.sub("emb_right").get("weight"),
            emb_state: p.sub("emb_state").get("weight"),
            pos: PositionalEncoding2D::load(&p.sub("pos"), enable_same_chain),
        }
    }

    /// Returns `(msa, pair, state)`.
    pub fn forward(
        &self,
        msa: &Tensor, // [B,N,L,d_init]
        seq: &[i64],
        idx: &[i64],
        bond_feats: &[i64],
        dist_matrix: &[f32],
        same_chain: &[bool],
    ) -> (Tensor, Tensor, Tensor) {
        let (b, n, l) = (msa.shape[0], msa.shape[1], msa.shape[2]);
        let mut m = self.emb.forward(msa); // [B,N,L,d_msa]
        let dm = m.last();
        for bi in 0..b {
            for ni in 0..n {
                for li in 0..l {
                    let t = seq[bi * l + li] as usize;
                    let o = ((bi * n + ni) * l + li) * dm;
                    for c in 0..dm {
                        m.data[o + c] += self.emb_q.data[t * dm + c];
                    }
                }
            }
        }

        let dp = self.emb_left.shape[1];
        let mut pair = vec![0.0f32; l * l * dp];
        for i in 0..l {
            let ti = seq[i] as usize;
            for j in 0..l {
                let tj = seq[j] as usize;
                let o = (i * l + j) * dp;
                for c in 0..dp {
                    // left is indexed by the COLUMN, right by the ROW
                    pair[o + c] = self.emb_left.data[tj * dp + c] + self.emb_right.data[ti * dp + c];
                }
            }
        }
        let pos = self.pos.forward(seq, idx, bond_feats, dist_matrix, same_chain);
        for (i, v) in pair.iter_mut().enumerate() {
            *v += pos.data[i];
        }

        let ds = self.emb_state.shape[1];
        let mut state = vec![0.0f32; l * ds];
        for i in 0..l {
            let t = seq[i] as usize;
            state[i * ds..i * ds + ds].copy_from_slice(&self.emb_state.data[t * ds..t * ds + ds]);
        }

        (m, Tensor::new(pair, vec![1, l, l, dp]), Tensor::new(state, vec![1, l, ds]))
    }
}

pub struct ExtraEmb {
    pub emb: Linear,
    pub emb_q: Tensor,
}

impl ExtraEmb {
    pub fn load(p: &Params) -> Self {
        ExtraEmb { emb: Linear::load(&p.sub("emb")), emb_q: p.sub("emb_q").get("weight") }
    }

    pub fn forward(&self, msa: &Tensor, seq: &[i64]) -> Tensor {
        let (b, n, l) = (msa.shape[0], msa.shape[1], msa.shape[2]);
        let mut m = self.emb.forward(msa);
        let dm = m.last();
        for bi in 0..b {
            for ni in 0..n {
                for li in 0..l {
                    let t = seq[bi * l + li] as usize;
                    let o = ((bi * n + ni) * l + li) * dm;
                    for c in 0..dm {
                        m.data[o + c] += self.emb_q.data[t * dm + c];
                    }
                }
            }
        }
        m
    }
}

pub struct BondEmb {
    pub emb: Linear,
}

impl BondEmb {
    pub fn load(p: &Params) -> Self {
        BondEmb { emb: Linear::load(&p.sub("emb")) }
    }

    /// One-hot the bond type to `NBTYPES` then project. `F.one_hot` would throw
    /// on an out-of-range class, so the assert reproduces that rather than
    /// wrapping silently.
    pub fn forward(&self, bond_feats: &[i64], l: usize) -> Tensor {
        let mut oh = vec![0.0f32; l * l * NBTYPES];
        for (k, &b) in bond_feats.iter().enumerate() {
            assert!(
                (0..NBTYPES as i64).contains(&b),
                "bond_feats[{k}] = {b} outside 0..{NBTYPES}"
            );
            oh[k * NBTYPES + b as usize] = 1.0;
        }
        self.emb.forward(&Tensor::new(oh, vec![1, l, l, NBTYPES]))
    }
}

// ---------------------------------------------------------------------------
// RecyclingAllFeatures
// ---------------------------------------------------------------------------

pub struct Recycling {
    pub proj_dist: Linear,
    pub norm_pair: LayerNorm,
    pub proj_sctors: Linear,
    pub norm_msa: LayerNorm,
    pub norm_state: LayerNorm,
}

impl Recycling {
    pub fn load(p: &Params) -> Self {
        Recycling {
            proj_dist: Linear::load(&p.sub("proj_dist")),
            norm_pair: LayerNorm::load(&p.sub("norm_pair")),
            proj_sctors: Linear::load(&p.sub("proj_sctors")),
            norm_msa: LayerNorm::load(&p.sub("norm_msa")),
            norm_state: LayerNorm::load(&p.sub("norm_state")),
        }
    }

    /// `(msa_prev, pair_prev, xyz, state_prev, sctors)` -> `(msa, pair, state)`.
    ///
    /// `mask_recycle` is `None` on the inference path (the sampler never passes
    /// one), so the distance features are unmasked; the parameter is kept so a
    /// future recycling loop cannot forget it exists.
    pub fn forward(
        &self,
        msa: &Tensor,   // [B,L,d_msa]
        pair: &Tensor,  // [B,L,L,d_pair]
        ca: &[f32],     // [L,3]
        state: &Tensor, // [B,L,d_state]
        sctors: &Tensor,
        mask_recycle: Option<&[f32]>,
    ) -> (Tensor, Tensor, Tensor) {
        let l = pair.shape[1];
        let state = self.norm_state.forward(state);
        let ds = state.last();
        let mut dist = geom::rbf_ca(ca, l); // [L,L,64]
        if let Some(m) = mask_recycle {
            for k in 0..l * l {
                for c in 0..geom::D_COUNT {
                    dist.data[k * geom::D_COUNT + c] *= m[k];
                }
            }
        }
        let w = geom::D_COUNT + 2 * ds;
        let mut cat = vec![0.0f32; l * l * w];
        for i in 0..l {
            for j in 0..l {
                let o = (i * l + j) * w;
                cat[o..o + geom::D_COUNT]
                    .copy_from_slice(&dist.data[(i * l + j) * geom::D_COUNT..][..geom::D_COUNT]);
                // left = state[i] broadcast over j, right = state[j]
                cat[o + geom::D_COUNT..o + geom::D_COUNT + ds]
                    .copy_from_slice(&state.data[i * ds..i * ds + ds]);
                cat[o + geom::D_COUNT + ds..o + w]
                    .copy_from_slice(&state.data[j * ds..j * ds + ds]);
            }
        }
        let mut projected = self.proj_dist.forward(&Tensor::new(cat, vec![1, l, l, w]));
        let normed_pair = self.norm_pair.forward(pair);
        for (i, v) in projected.data.iter_mut().enumerate() {
            *v += normed_pair.data[i];
        }

        let nt2 = sctors.numel() / l;
        let sc = Tensor::new(sctors.data.clone(), vec![1, l, nt2]);
        let mut msa_out = self.proj_sctors.forward(&sc);
        let normed_msa = self.norm_msa.forward(msa);
        for (i, v) in msa_out.data.iter_mut().enumerate() {
            *v += normed_msa.data[i];
        }

        (msa_out, projected, state)
    }
}

// ---------------------------------------------------------------------------
// PairStr2Pair — shared by TemplatePairStack and every IterBlock
// ---------------------------------------------------------------------------

pub struct PairStr2Pair {
    pub norm_state: LayerNorm,
    pub proj_left: Linear,
    pub proj_right: Linear,
    pub to_gate: Linear,
    pub emb_rbf: Linear,
    pub tri_mul_out: TriangleMultiplication,
    pub tri_mul_in: TriangleMultiplication,
    pub row_attn: BiasedAxialAttention,
    pub col_attn: BiasedAxialAttention,
    pub ff: FeedForward,
    /// Feeds `drop_row` / `drop_col`. 0.15 in the trunk blocks, **0.25** in the
    /// template stack (`Templ_emb` passes `p_drop=0.25` down). The two are
    /// different modules with the same name, and using the trunk value inside
    /// the template stack changes both the mask and its RNG consumption.
    pub p_drop: f64,
}

impl PairStr2Pair {
    pub fn load(p: &Params, n_head: usize, d_hidden: usize, p_drop: f64) -> Self {
        PairStr2Pair {
            norm_state: LayerNorm::load(&p.sub("norm_state")),
            proj_left: Linear::load(&p.sub("proj_left")),
            proj_right: Linear::load(&p.sub("proj_right")),
            to_gate: Linear::load(&p.sub("to_gate")),
            emb_rbf: Linear::load(&p.sub("emb_rbf")),
            tri_mul_out: TriangleMultiplication::load(&p.sub("tri_mul_out"), true),
            tri_mul_in: TriangleMultiplication::load(&p.sub("tri_mul_in"), false),
            row_attn: BiasedAxialAttention::load(&p.sub("row_attn"), n_head, d_hidden, true),
            col_attn: BiasedAxialAttention::load(&p.sub("col_attn"), n_head, d_hidden, false),
            // `PairStr2Pair.ff = FeedForwardLayer(d_pair, 2)` — no p_drop is
            // passed, so this one keeps the 0.1 default even when the block's
            // own dropout is 0.15 or 0.25.
            ff: FeedForward::load(&p.sub("ff"), 0.1),
            p_drop,
        }
    }

    /// `crop` is -1 on the inference path, so the striped `subblock` route is
    /// never taken; the dense branch below is the one that runs.
    pub fn forward(
        &self,
        pair: &Tensor,
        rbf_feat: &Tensor,
        state: &Tensor,
        ctx: &mut Ctx,
    ) -> Tensor {
        let (b, l) = (pair.shape[0], pair.shape[1]);
        let mut rbf = self.emb_rbf.forward(rbf_feat); // [B,L,L,d_pair]

        let st = self.norm_state.forward(state);
        let left = self.proj_left.forward(&st); // [B,L,dh]
        let right = self.proj_right.forward(&st);
        let dh = left.last();
        let mut gate_in = vec![0.0f32; b * l * l * dh * dh];
        for bi in 0..b {
            for li in 0..l {
                for mi in 0..l {
                    let o = ((bi * l + li) * l + mi) * dh * dh;
                    for i in 0..dh {
                        let lv = left.data[(bi * l + li) * dh + i] as f64;
                        for j in 0..dh {
                            let rv = right.data[(bi * l + mi) * dh + j] as f64;
                            gate_in[o + i * dh + j] = (lv * rv) as f32;
                        }
                    }
                }
            }
        }
        let gate = self
            .to_gate
            .forward(&Tensor::new(gate_in, vec![b, l, l, dh * dh]));
        for (i, v) in rbf.data.iter_mut().enumerate() {
            *v *= sigmoid_scalar(gate.data[i]);
        }

        let mut pair = pair.clone();
        // Order matters twice over: each residual feeds the next, and each
        // `drop_row`/`drop_col` advances the shared RNG.
        let d = self.tri_mul_out.forward(&pair);
        let d = rf_dropout(&mut ctx.rng, &d, Some(1), self.p_drop);
        for (i, v) in pair.data.iter_mut().enumerate() {
            *v += d.data[i];
        }
        let d = self.tri_mul_in.forward(&pair);
        let d = rf_dropout(&mut ctx.rng, &d, Some(1), self.p_drop);
        for (i, v) in pair.data.iter_mut().enumerate() {
            *v += d.data[i];
        }
        let d = self.row_attn.forward(&pair, &rbf);
        let d = rf_dropout(&mut ctx.rng, &d, Some(1), self.p_drop);
        for (i, v) in pair.data.iter_mut().enumerate() {
            *v += d.data[i];
        }
        let d = self.col_attn.forward(&pair, &rbf);
        let d = rf_dropout(&mut ctx.rng, &d, Some(2), self.p_drop);
        for (i, v) in pair.data.iter_mut().enumerate() {
            *v += d.data[i];
        }
        let d = self.ff.forward(&pair, ctx);
        for (i, v) in pair.data.iter_mut().enumerate() {
            *v += d.data[i];
        }
        pair
    }
}

// ---------------------------------------------------------------------------
// Templ_emb
// ---------------------------------------------------------------------------

pub struct TemplatePairStack {
    pub proj_t1d: Linear,
    pub block: Vec<PairStr2Pair>,
    pub norm: LayerNorm,
}

impl TemplatePairStack {
    pub fn load(p: &Params, n_block: usize, n_head: usize, d_hidden: usize, p_drop: f64) -> Self {
        TemplatePairStack {
            proj_t1d: Linear::load(&p.sub("proj_t1d")),
            block: (0..n_block)
                .map(|i| PairStr2Pair::load(&p.sub("block").idx(i), n_head, d_hidden, p_drop))
                .collect(),
            norm: LayerNorm::load(&p.sub("norm")),
        }
    }

    /// templ `[B*T,L,L,d_templ]`, rbf_feat `[B*T,L,L,64]`, t1d `[B*T,L,d_t1d]`.
    pub fn forward(
        &self,
        templ: &Tensor,
        rbf_feat: &Tensor,
        t1d: &Tensor,
        ctx: &mut Ctx,
    ) -> Tensor {
        let state = self.proj_t1d.forward(t1d);
        let mut x = templ.clone();
        for blk in &self.block {
            x = blk.forward(&x, rbf_feat, &state, ctx);
        }
        self.norm.forward(&x)
    }
}

pub struct TemplEmb {
    pub emb: Linear,
    pub templ_stack: TemplatePairStack,
    pub attn: Attention,
    pub emb_t1d: Linear,
    pub proj_t1d: Linear,
    pub attn_tor: Attention,
}

impl TemplEmb {
    pub fn load(p: &Params, n_head: usize, d_hidden: usize, n_block: usize) -> Self {
        TemplEmb {
            emb: Linear::load(&p.sub("emb")),
            // the stack's PairStr2Pair inherits `d_hidden` from Templ_emb, i.e.
            // `d_hidden_templ` (64), not the trunk's `d_hidden` (32)
            // `Templ_emb(..., p_drop=0.25)` — the template stack's dropout is
            // 0.25, not the trunk's 0.15.
            templ_stack: TemplatePairStack::load(
                &p.sub("templ_stack"), n_block, n_head, d_hidden, 0.25),
            attn: Attention::load(&p.sub("attn"), n_head, d_hidden),
            emb_t1d: Linear::load(&p.sub("emb_t1d")),
            proj_t1d: Linear::load(&p.sub("proj_t1d")),
            attn_tor: Attention::load(&p.sub("attn_tor"), n_head, d_hidden),
        }
    }

    /// Returns the updated `(pair, state)`.
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        t1d: &Tensor,    // [B,T,L,d_t1d]
        t2d: &Tensor,    // [B,T,L,L,d_t2d]
        alpha_t: &Tensor, // [B,T,L,d_tor]
        xyz_t: &Tensor,  // [B,T,L,3]
        mask_t: &[bool], // [B,T,L,L]
        pair: &Tensor,   // [B,L,L,d_pair]
        state: &Tensor,  // [B,L,d_state]
        ctx: &mut Ctx,
    ) -> (Tensor, Tensor) {
        let (b, t, l) = (t1d.shape[0], t1d.shape[1], t1d.shape[2]);
        let d1 = t1d.last();
        let d2 = t2d.last();
        let bt = b * t;

        // templ = emb(cat(t2d, left=t1d[..,i,:], right=t1d[..,j,:]))
        let w = d2 + 2 * d1;
        let mut cat = vec![0.0f32; bt * l * l * w];
        for x in 0..bt {
            for i in 0..l {
                for j in 0..l {
                    let o = ((x * l + i) * l + j) * w;
                    cat[o..o + d2]
                        .copy_from_slice(&t2d.data[((x * l + i) * l + j) * d2..][..d2]);
                    // `left = t1d.unsqueeze(3)` broadcasts over the LAST index,
                    // so `left` varies with i and `right` with j.
                    cat[o + d2..o + d2 + d1]
                        .copy_from_slice(&t1d.data[(x * l + i) * d1..][..d1]);
                    cat[o + d2 + d1..o + w]
                        .copy_from_slice(&t1d.data[(x * l + j) * d1..][..d1]);
                }
            }
        }
        let templ = self.emb.forward(&Tensor::new(cat, vec![bt, l, l, w]));

        // rbf_feat = rbf(cdist(xyz_t, xyz_t)) * mask_t[..., None]
        let mut rbf_all = vec![0.0f32; bt * l * l * geom::D_COUNT];
        for x in 0..bt {
            let pts = &xyz_t.data[x * l * 3..(x + 1) * l * 3];
            let d = geom::cdist_self(pts, l);
            let dst = &mut rbf_all[x * l * l * geom::D_COUNT..(x + 1) * l * l * geom::D_COUNT];
            geom::rbf_into(&d, dst);
            for k in 0..l * l {
                if !mask_t[x * l * l + k] {
                    for c in 0..geom::D_COUNT {
                        dst[k * geom::D_COUNT + c] = 0.0;
                    }
                }
            }
        }
        let rbf_feat = Tensor::new(rbf_all, vec![bt, l, l, geom::D_COUNT]);

        let t1d_flat = Tensor::new(t1d.data.clone(), vec![bt, l, d1]);
        let templ = self.templ_stack.forward(&templ, &rbf_feat, &t1d_flat, ctx);
        let dt = templ.last();

        // torsion branch: t1d = proj_t1d(relu_(emb_t1d(cat(t1d, alpha_t))))
        let da = alpha_t.last();
        let mut tcat = vec![0.0f32; bt * l * (d1 + da)];
        for x in 0..bt {
            for i in 0..l {
                let o = (x * l + i) * (d1 + da);
                tcat[o..o + d1].copy_from_slice(&t1d.data[(x * l + i) * d1..][..d1]);
                tcat[o + d1..o + d1 + da]
                    .copy_from_slice(&alpha_t.data[(x * l + i) * da..][..da]);
            }
        }
        let mut h = self.emb_t1d.forward(&Tensor::new(tcat, vec![bt, l, d1 + da]));
        relu_(&mut h);
        let tor = self.proj_t1d.forward(&h); // [B*T, L, d_templ]

        // state attention: query = state[b*l, 1, d_state], key/value = tor over T
        let ds = state.last();
        let q = Tensor::new(state.data.clone(), vec![b * l, 1, ds]);
        let mut kv = vec![0.0f32; b * l * t * dt];
        for bi in 0..b {
            for li in 0..l {
                for ti in 0..t {
                    let src = ((bi * t + ti) * l + li) * dt;
                    let dstv = ((bi * l + li) * t + ti) * dt;
                    kv[dstv..dstv + dt].copy_from_slice(&tor.data[src..src + dt]);
                }
            }
        }
        let kv = Tensor::new(kv, vec![b * l, t, dt]);
        let out = self.attn_tor.forward(&q, &kv, &kv);
        let mut state_out = state.data.clone();
        for (i, v) in state_out.iter_mut().enumerate() {
            *v += out.data[i];
        }

        // pair attention: query = pair[b*l*l, 1, d_pair], key/value = templ over T
        let dp = pair.last();
        let q = Tensor::new(pair.data.clone(), vec![b * l * l, 1, dp]);
        let mut kv = vec![0.0f32; b * l * l * t * dt];
        for bi in 0..b {
            for i in 0..l {
                for j in 0..l {
                    for ti in 0..t {
                        let src = (((bi * t + ti) * l + i) * l + j) * dt;
                        let dstv = (((bi * l + i) * l + j) * t + ti) * dt;
                        kv[dstv..dstv + dt].copy_from_slice(&templ.data[src..src + dt]);
                    }
                }
            }
        }
        let kv = Tensor::new(kv, vec![b * l * l, t, dt]);
        let out = self.attn.forward(&q, &kv, &kv);
        let mut pair_out = pair.data.clone();
        for (i, v) in pair_out.iter_mut().enumerate() {
            *v += out.data[i];
        }

        (
            Tensor::new(pair_out, pair.shape.clone()),
            Tensor::new(state_out, state.shape.clone()),
        )
    }
}
