//! `LegacyRoseTTAFoldModule` — the whole network, and the `IterativeSimulator`
//! loop inside it.
//!
//! Shapes and branch selection come from the **checkpoint's** config
//! (`fixtures/weights/ckpt_conf.json`), not from the module defaults, which
//! differ in several places that matter (`d_pair` 192 vs 128, `n_head_pair` 6 vs
//! 4, `recycling_type = "all"`, `enable_same_chain = True`).

use crate::lj::{lj_forward, natoms, LjCfg, LjTables};
use crate::model::aux::{self, BinderNetwork, DistanceNetwork, Proj};
use crate::model::embeddings::{BondEmb, ExtraEmb, MsaEmb, Recycling, TemplEmb};
use crate::model::iterblock::{BlockCfg, BlockInputs, IterBlock, TrackState};
use crate::model::str2str::Str2Str;
use crate::model::xyzconv::XyzConverter;
use crate::nn::{Ctx, Params};
use crate::tensor::Tensor;

/// The architecture, as read out of the checkpoint.
#[derive(Clone, Copy, Debug)]
pub struct Arch {
    pub d_msa: usize,
    pub d_msa_full: usize,
    pub d_pair: usize,
    pub d_state: usize,
    pub d_hidden: usize,
    pub d_hidden_msa_extra: usize,
    pub n_head_msa: usize,
    pub n_head_pair: usize,
    pub n_head_templ: usize,
    pub d_hidden_templ: usize,
    pub n_extra_block: usize,
    pub n_main_block: usize,
    pub n_ref_block: usize,
    pub p_drop: f64,
    pub enable_same_chain: bool,
    pub use_chiral_l1: bool,
    pub use_lj_l1: bool,
    pub refiner_topk: i64,
    pub se3_layers: usize,
    pub se3_ref_layers: usize,
    pub num_channels: usize,
    pub num_degrees: usize,
    pub n_heads: usize,
    pub div: usize,
}

impl Arch {
    /// `RFD_173.pt`'s `conf.rf.model`.
    pub fn rfd173() -> Self {
        Arch {
            d_msa: 256,
            d_msa_full: 64,
            d_pair: 192,
            d_state: 64,
            d_hidden: 32,
            d_hidden_msa_extra: 8,
            n_head_msa: 8,
            n_head_pair: 6,
            n_head_templ: 4,
            d_hidden_templ: 64,
            n_extra_block: 4,
            n_main_block: 32,
            n_ref_block: 4,
            p_drop: 0.15,
            enable_same_chain: true,
            use_chiral_l1: true,
            use_lj_l1: true,
            refiner_topk: 128,
            se3_layers: 1,
            se3_ref_layers: 2,
            num_channels: 32,
            num_degrees: 2,
            n_heads: 4,
            div: 4,
        }
    }

    fn block_cfg(&self, extra: bool) -> BlockCfg {
        BlockCfg {
            n_head_msa: self.n_head_msa,
            // `IterBlock(d_hidden_msa=8)` for the extra blocks only; the main
            // blocks leave it None, which falls back to `d_hidden`.
            d_hidden_msa: if extra { self.d_hidden_msa_extra } else { self.d_hidden },
            n_head_pair: self.n_head_pair,
            d_hidden: self.d_hidden,
            use_global_attn: extra,
            enable_same_chain: self.enable_same_chain,
            p_drop: self.p_drop,
            se3_num_layers: self.se3_layers,
            l0_in: self.d_state,
            l1_in: 3 + if self.use_chiral_l1 { 3 } else { 0 },
            num_channels: self.num_channels,
            num_degrees: self.num_degrees,
            l0_out: self.d_state,
            l1_out: 2,
            n_heads: self.n_heads,
            div: self.div,
            // `topk_crop = -1` on the sampler's path, so the extra and main
            // blocks build a FULL graph; only `str_refiner` uses top-k.
            top_k: -1,
            n_extra_l1: if self.use_chiral_l1 { 3 } else { 0 },
        }
    }
}

pub struct Simulator {
    pub extra_block: Vec<IterBlock>,
    pub main_block: Vec<IterBlock>,
    pub str_refiner: Str2Str,
    pub arch: Arch,
}

/// What one simulator pass produces, in the order `IterativeSimulator` returns.
pub struct SimOut {
    pub msa: Tensor,
    pub pair: Tensor,
    /// Stacked coordinates, one entry per block: `[n_blocks, L, 3, 3]`.
    pub xyz_stack: Vec<Vec<f32>>,
    pub alpha_stack: Vec<Tensor>,
    pub quat_stack: Vec<Vec<f32>>,
    /// `compute_all_atom` of the final backbone: `[L, NTOTAL, 3]`.
    pub xyzallatom: Vec<f32>,
    pub state: Tensor,
}

impl Simulator {
    pub fn load(p: &Params, arch: Arch) -> Self {
        let n_extra_l0 = if arch.use_lj_l1 { 2 * crate::chemical_gen::NTOTALDOFS } else { 0 };
        let n_extra_l1 = if arch.use_chiral_l1 { 3 } else { 0 }
            + if arch.use_lj_l1 { 3 } else { 0 };
        Simulator {
            extra_block: (0..arch.n_extra_block)
                .map(|i| IterBlock::load(&p.sub("extra_block").idx(i), arch.block_cfg(true)))
                .collect(),
            main_block: (0..arch.n_main_block)
                .map(|i| IterBlock::load(&p.sub("main_block").idx(i), arch.block_cfg(false)))
                .collect(),
            str_refiner: Str2Str::load(
                &p.sub("str_refiner"),
                arch.se3_ref_layers,
                arch.d_state + n_extra_l0,
                3 + n_extra_l1,
                arch.num_channels,
                arch.num_degrees,
                arch.d_state,
                2,
                arch.n_heads,
                arch.div,
                arch.p_drop,
            ),
            arch,
        }
    }

    /// The extra + main block loops. The refinement loop is separate because it
    /// needs the Lennard-Jones gradients, which are their own sub-port.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_blocks(
        &self,
        msa: &Tensor,
        msa_full: &Tensor,
        pair: &Tensor,
        xyz: &[f32],
        state: &Tensor,
        inp: &BlockInputs,
        ctx: &mut Ctx,
    ) -> SimOut {
        let l = inp.idx.len();
        let mut xyz_stack = Vec::new();
        let mut alpha_stack = Vec::new();
        let mut quat_stack = Vec::new();

        // the extra blocks update `msa_full`, the main blocks update `msa`;
        // pair / xyz / state are threaded through both.
        let mut st = TrackState {
            msa: msa_full.clone(),
            pair: pair.clone(),
            xyz: xyz.to_vec(),
            state: state.clone(),
            alpha: Tensor::zeros(&[1, l, crate::chemical_gen::NTOTALDOFS, 2]),
            quat: vec![0.0; l * 4],
        };
        for blk in &self.extra_block {
            blk.forward(&mut st, inp, ctx);
            xyz_stack.push(st.xyz.clone());
            alpha_stack.push(st.alpha.clone());
            quat_stack.push(st.quat.clone());
        }
        let msa_full_out = std::mem::replace(&mut st.msa, msa.clone());
        for blk in &self.main_block {
            blk.forward(&mut st, inp, ctx);
            xyz_stack.push(st.xyz.clone());
            alpha_stack.push(st.alpha.clone());
            quat_stack.push(st.quat.clone());
        }
        let _ = msa_full_out;

        // ---- the refinement loop -----------------------------------------
        // Each iteration re-derives the two gradient terms from the *current*
        // backbone, so they cannot be hoisted: `xyz` and `alpha` change every
        // pass. Upstream also calls `compute_all_atom` once before the loop and
        // throws the result away; it draws no randomness, so it is omitted.
        let conv = XyzConverter::new();
        let lj_tables = LjTables::new();
        let lj_cfg = LjCfg::default();
        for _ in 0..self.arch.n_ref_block {
            let mut extra_l1 = vec![0.0f32; l * (3 + 3) * 3];
            let mut extra_l0: Option<Vec<f32>> = None;
            if self.arch.use_lj_l1 {
                let xyzaa = conv.compute_all_atom(inp.seq_unmasked, &st.xyz, 3, &st.alpha.data);
                let out = lj_forward(
                    inp.seq_unmasked,
                    &xyzaa,
                    inp.bond_feats,
                    inp.dist_matrix,
                    &lj_tables,
                    &lj_cfg,
                );
                // `torch.autograd.grad(natoms * Elj, ...)`, so the incoming
                // gradient on `xyzaa` is the atom count, not 1.
                let n = natoms(inp.seq_unmasked, &lj_tables, lj_cfg.use_h);
                let dxyzaa: Vec<f32> = out.dljedx.iter().map(|v| n * v).collect();
                let g = crate::xyzconv_bwd::backward(
                    inp.seq_unmasked,
                    &st.xyz,
                    3,
                    &st.alpha.data,
                    &dxyzaa,
                );
                for i in 0..l {
                    extra_l1[i * 18..i * 18 + 9].copy_from_slice(&g.dxyz[i * 9..i * 9 + 9]);
                }
                extra_l0 = Some(g.dalpha);
            }
            if self.arch.use_chiral_l1 {
                let dch = crate::chiral::chiral_grads(&st.xyz, l, 3, inp.chirals);
                for i in 0..l {
                    extra_l1[i * 18 + 9..i * 18 + 18].copy_from_slice(&dch[i * 9..i * 9 + 9]);
                }
            }
            let out = self.str_refiner.forward(
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
                extra_l0.as_deref(),
                &extra_l1,
                6,
                self.arch.refiner_topk,
                ctx,
            );
            st.xyz = out.xyz;
            st.state = out.state;
            st.alpha = out.alpha;
            st.quat = out.quat;
            xyz_stack.push(st.xyz.clone());
            alpha_stack.push(st.alpha.clone());
            quat_stack.push(st.quat.clone());
        }
        let xyzallatom =
            conv.compute_all_atom(inp.seq_unmasked, &st.xyz, 3, &st.alpha.data);

        SimOut {
            msa: st.msa,
            pair: st.pair,
            xyz_stack,
            alpha_stack,
            quat_stack,
            xyzallatom,
            state: st.state,
        }
    }
}

pub struct RoseTTAFold {
    pub latent_emb: MsaEmb,
    pub full_emb: ExtraEmb,
    pub bond_emb: BondEmb,
    pub templ_emb: TemplEmb,
    pub recycle: Recycling,
    pub simulator: Simulator,
    pub c6d_pred: DistanceNetwork,
    pub aa_pred: Proj,
    pub lddt_pred: Proj,
    pub pae_pred: Proj,
    pub pde_pred: Proj,
    pub bind_pred: BinderNetwork,
    pub arch: Arch,
}

/// Everything the network reads out of `RFI`.
pub struct Rfi {
    pub msa_latent: Tensor,
    pub msa_full: Tensor,
    pub seq: Vec<i64>,
    pub seq_unmasked: Vec<i64>,
    pub xyz: Tensor, // [1, L, NTOTAL, 3]
    pub sctors: Tensor,
    pub idx: Vec<i64>,
    pub bond_feats: Vec<i64>,
    pub dist_matrix: Vec<f32>,
    pub chirals: Vec<f32>,
    pub atom_frames: Vec<i64>,
    pub t1d: Tensor,
    pub t2d: Tensor,
    pub xyz_t: Tensor,
    pub alpha_t: Tensor,
    pub mask_t: Vec<bool>,
    pub same_chain: Vec<bool>,
    pub is_motif: Vec<bool>,
}

impl RoseTTAFold {
    pub fn load(p: &Params, arch: Arch) -> Self {
        RoseTTAFold {
            latent_emb: MsaEmb::load(&p.sub("latent_emb"), arch.enable_same_chain),
            full_emb: ExtraEmb::load(&p.sub("full_emb")),
            bond_emb: BondEmb::load(&p.sub("bond_emb")),
            templ_emb: TemplEmb::load(
                &p.sub("templ_emb"),
                arch.n_head_templ,
                arch.d_hidden_templ,
                2,
            ),
            recycle: Recycling::load(&p.sub("recycle")),
            simulator: Simulator::load(&p.sub("simulator"), arch),
            c6d_pred: DistanceNetwork::load(&p.sub("c6d_pred")),
            aa_pred: Proj::load(&p.sub("aa_pred")),
            lddt_pred: Proj::load(&p.sub("lddt_pred")),
            pae_pred: Proj::load(&p.sub("pae_pred")),
            pde_pred: Proj::load(&p.sub("pde_pred")),
            bind_pred: BinderNetwork::load(&p.sub("bind_pred")),
            arch,
        }
    }

    /// The embedding half of `forward`, up to the point the simulator starts.
    pub fn embed(&self, rfi: &Rfi, ctx: &mut Ctx) -> (Tensor, Tensor, Tensor, Tensor) {
        let l = rfi.seq.len();
        let a = &self.arch;
        let (mut msa, mut pair, mut state) = self.latent_emb.forward(
            &rfi.msa_latent,
            &rfi.seq,
            &rfi.idx,
            &rfi.bond_feats,
            &rfi.dist_matrix,
            &rfi.same_chain,
        );
        let msa_full = self.full_emb.forward(&rfi.msa_full, &rfi.seq);
        let be = self.bond_emb.forward(&rfi.bond_feats, l);
        for (i, v) in pair.data.iter_mut().enumerate() {
            *v += be.data[i];
        }

        // recycling: every `*_prev` is None on this path, i.e. zeros
        let natoms = rfi.xyz.shape[2];
        let ca: Vec<f32> = (0..l)
            .flat_map(|i| rfi.xyz.data[(i * natoms + 1) * 3..(i * natoms + 1) * 3 + 3].to_vec())
            .collect();
        let (mr, pr, sr) = self.recycle.forward(
            &Tensor::zeros(&[1, l, a.d_msa]),
            &Tensor::zeros(&[1, l, l, a.d_pair]),
            &ca,
            &Tensor::zeros(&[1, l, a.d_state]),
            &rfi.sctors,
            None,
        );
        // msa_latent[:, 0] += msa_recycle
        for i in 0..l * a.d_msa {
            msa.data[i] += mr.data[i];
        }
        for (i, v) in pair.data.iter_mut().enumerate() {
            *v += pr.data[i];
        }
        for (i, v) in state.data.iter_mut().enumerate() {
            *v += sr.data[i];
        }

        let (pair, state) = self.templ_emb.forward(
            &rfi.t1d,
            &rfi.t2d,
            &rfi.alpha_t,
            &rfi.xyz_t,
            &rfi.mask_t,
            &pair,
            &state,
            ctx,
        );
        (msa, msa_full, pair, state)
    }

    /// Embedding + the extra/main block loops. Returns the simulator output.
    pub fn forward_blocks(&self, rfi: &Rfi, ctx: &mut Ctx) -> SimOut {
        let l = rfi.seq.len();
        let (msa, msa_full, pair, state) = self.embed(rfi, ctx);
        let natoms = rfi.xyz.shape[2];
        // the simulator only ever sees N/CA/C
        let xyz3: Vec<f32> = (0..l)
            .flat_map(|i| rfi.xyz.data[i * natoms * 3..i * natoms * 3 + 9].to_vec())
            .collect();
        let rotation_mask: Vec<bool> =
            rfi.seq_unmasked.iter().map(|&t| crate::geom::is_atom(t)).collect();
        let inp = BlockInputs {
            seq_unmasked: &rfi.seq_unmasked,
            idx: &rfi.idx,
            bond_feats: &rfi.bond_feats,
            dist_matrix: &rfi.dist_matrix,
            same_chain: &rfi.same_chain,
            chirals: &rfi.chirals,
            atom_frames: &rfi.atom_frames,
            is_motif: &rfi.is_motif,
            rotation_mask: &rotation_mask,
        };
        self.simulator
            .forward_blocks(&msa, &msa_full, &pair, &xyz3, &state, &inp, ctx)
    }

    /// The whole `LegacyRoseTTAFoldModule.forward`: embeddings, simulator and
    /// the six auxiliary heads, in the order upstream returns them.
    pub fn forward(&self, rfi: &Rfi, ctx: &mut Ctx) -> ModelOut {
        let sim = self.forward_blocks(rfi, ctx);
        let l = rfi.seq.len();

        // `msa` reaches the heads as [1, N, L, d]; the simulator hands back the
        // same tensor, and `aa_pred` folds N into the output's last axis.
        let logits_aa = aux::masked_token(&self.aa_pred, &sim.msa);
        let c6d = self.c6d_pred.forward(&sim.pair);
        let lddt = aux::permute_last_to_front(&self.lddt_pred.proj.forward(&sim.state));
        let logits_pae = aux::permute_last_to_front(&self.pae_pred.proj.forward(&sim.pair));

        // `pde_pred(pair + pair.permute(0,2,1,3))` — the symmetrisation is on
        // the *input*, unlike `c6d_pred`, which symmetrises its own projection.
        let dp = sim.pair.last();
        let mut sym = sim.pair.data.clone();
        for i in 0..l {
            for j in 0..l {
                for k in 0..dp {
                    sym[(i * l + j) * dp + k] += sim.pair.data[(j * l + i) * dp + k];
                }
            }
        }
        let sym = Tensor::new(sym, sim.pair.shape.clone());
        let logits_pde = aux::permute_last_to_front(&self.pde_pred.proj.forward(&sym));

        let p_bind = self.bind_pred.forward(&logits_pae, &rfi.same_chain);

        ModelOut {
            c6d,
            logits_aa,
            logits_pae,
            logits_pde,
            p_bind,
            lddt,
            sim,
        }
    }
}

/// Everything `LegacyRoseTTAFoldModule.forward` returns.
pub struct ModelOut {
    pub c6d: aux::C6d,
    /// `[1, NAATOKENS, N*L]`
    pub logits_aa: Tensor,
    /// `[1, 64, L, L]`
    pub logits_pae: Tensor,
    /// `[1, 64, L, L]`
    pub logits_pde: Tensor,
    pub p_bind: f32,
    /// `[1, 50, L]`
    pub lddt: Tensor,
    pub sim: SimOut,
}
