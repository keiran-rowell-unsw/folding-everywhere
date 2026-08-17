//! `IterBlock` — one three-track update, and the simulator loop over them.

use crate::geom;
use crate::model::embeddings::PairStr2Pair;
use crate::model::str2str::{Str2Str, StrOut};
use crate::model::track::{MSA2Pair, MSAPairStr2MSA};
use crate::model::embeddings::PositionalEncoding2D;
use crate::nn::{Ctx, Params};
use crate::tensor::Tensor;

/// Everything about the block that is fixed by the checkpoint config.
#[derive(Clone, Copy)]
pub struct BlockCfg {
    pub n_head_msa: usize,
    pub d_hidden_msa: usize,
    pub n_head_pair: usize,
    pub d_hidden: usize,
    pub use_global_attn: bool,
    pub enable_same_chain: bool,
    pub p_drop: f64,
    pub se3_num_layers: usize,
    pub l0_in: usize,
    pub l1_in: usize,
    pub num_channels: usize,
    pub num_degrees: usize,
    pub l0_out: usize,
    pub l1_out: usize,
    pub n_heads: usize,
    pub div: usize,
    /// -1 on this path, i.e. a full graph; only `str_refiner` passes 128.
    pub top_k: i64,
    pub n_extra_l1: usize,
}

pub struct IterBlock {
    pub pos: PositionalEncoding2D,
    pub msa2msa: MSAPairStr2MSA,
    pub msa2pair: MSA2Pair,
    pub pair2pair: PairStr2Pair,
    pub str2str: Str2Str,
    pub cfg: BlockCfg,
}

/// The mutable state the simulator threads from block to block.
pub struct TrackState {
    pub msa: Tensor,
    pub pair: Tensor,
    pub xyz: Vec<f32>, // [L, 3, 3] — only N/CA/C reach the simulator
    pub state: Tensor,
    pub alpha: Tensor,
    pub quat: Vec<f32>,
}

/// The invariants every block reads but none of them change.
pub struct BlockInputs<'a> {
    pub seq_unmasked: &'a [i64],
    pub idx: &'a [i64],
    pub bond_feats: &'a [i64],
    pub dist_matrix: &'a [f32],
    pub same_chain: &'a [bool],
    pub chirals: &'a [f32],
    pub atom_frames: &'a [i64],
    pub is_motif: &'a [bool],
    pub rotation_mask: &'a [bool],
}

impl IterBlock {
    pub fn load(p: &Params, cfg: BlockCfg) -> Self {
        IterBlock {
            pos: PositionalEncoding2D::load(&p.sub("pos"), cfg.enable_same_chain),
            msa2msa: MSAPairStr2MSA::load(
                &p.sub("msa2msa"),
                cfg.n_head_msa,
                cfg.d_hidden_msa,
                cfg.use_global_attn,
                cfg.p_drop,
            ),
            msa2pair: MSA2Pair::load(&p.sub("msa2pair")),
            pair2pair: PairStr2Pair::load(
                &p.sub("pair2pair"),
                cfg.n_head_pair,
                cfg.d_hidden,
                cfg.p_drop,
            ),
            str2str: Str2Str::load(
                &p.sub("str2str"),
                cfg.se3_num_layers,
                cfg.l0_in,
                cfg.l1_in,
                cfg.num_channels,
                cfg.num_degrees,
                cfg.l0_out,
                cfg.l1_out,
                cfg.n_heads,
                cfg.div,
                cfg.p_drop,
            ),
            cfg,
        }
    }

    pub fn forward(&self, st: &mut TrackState, inp: &BlockInputs, ctx: &mut Ctx) {
        let l = inp.idx.len();
        // rbf_feat = rbf(cdist(CA,CA)) + pos(...)  — computed BEFORE the block's
        // own updates, from the coordinates it was handed.
        let ca: Vec<f32> = (0..l).flat_map(|i| st.xyz[i * 9 + 3..i * 9 + 6].to_vec()).collect();
        let mut rbf = geom::rbf_ca(&ca, l).reshape(&[1, l, l, geom::D_COUNT]);
        let pos = self.pos.forward(
            inp.seq_unmasked,
            inp.idx,
            inp.bond_feats,
            inp.dist_matrix,
            inp.same_chain,
        );
        for (i, v) in rbf.data.iter_mut().enumerate() {
            *v += pos.data[i];
        }

        st.msa = self.msa2msa.forward(&st.msa, &st.pair, &rbf, &st.state, ctx);
        st.pair = self.msa2pair.forward(&st.msa, &st.pair);
        st.pair = self.pair2pair.forward(&st.pair, &rbf, &st.state, ctx);

        // the chiral gradient is recomputed from the CURRENT coordinates every
        // block, before str2str runs
        let extra_l1 = crate::chiral::chiral_grads(&st.xyz, l, 3, inp.chirals);

        let out: StrOut = self.str2str.forward(
            &st.msa,
            &st.pair,
            &st.xyz,
            3,
            &st.state,
            inp.idx,
            inp.rotation_mask,
            inp.bond_feats,
            inp.dist_matrix,
            inp.atom_frames,
            inp.is_motif,
            None,
            &extra_l1,
            self.cfg.n_extra_l1,
            self.cfg.top_k,
            ctx,
        );
        st.xyz = out.xyz;
        st.state = out.state;
        st.alpha = out.alpha;
        st.quat = out.quat;
    }
}
