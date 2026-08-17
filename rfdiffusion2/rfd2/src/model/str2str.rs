//! `Str2Str` and the two loops around it (`IterBlock`, `IterativeSimulator`).
//!
//! Measured on the real sampler (`python/probe_forward.py`, raw output in
//! `results/forward_probe.txt`), so the branch selection below is observed and
//! not inferred from defaults:
//!
//! ```text
//! IterativeSimulator: p2p_crop=-1 topk_crop=-1 use_checkpoint=False use_atom_frames=True
//!   n_extra_block=4 n_main_block=32 n_ref_block=4 use_lj_l1=True use_chiral_l1=True refiner_topk=128
//! ```
//!
//! Three consequences:
//!
//! * `p2p_crop = -1` -> `PairStr2Pair` takes its **dense** branch; the striped
//!   `subblock` path never runs.
//! * `topk_crop = -1` -> the extra and main blocks build a **full** graph.
//!   Only `str_refiner` uses the top-k graph, at `refiner_topk = 128`.
//! * `use_checkpoint = False` on this path — the `checkpoint.checkpoint(...)`
//!   branch in `IterBlock.forward` is dead here (it belongs to `get_rfo`, a
//!   different entry point).

use crate::ops::acc::Acc;
use crate::chiral::chiral_grads;
use crate::geom;
use crate::model::se3::{self, Fiber, Se3Transformer};
use crate::model::track::SCPred;
use crate::nn::{Ctx, FeedForward, LayerNorm, Linear, Params};
use crate::tensor::Tensor;

pub struct Str2Str {
    pub norm_msa: LayerNorm,
    pub norm_pair: LayerNorm,
    pub norm_state: LayerNorm,
    pub embed_node: Linear,
    pub ff_node: FeedForward,
    pub norm_node: LayerNorm,
    pub embed_edge: Linear,
    pub ff_edge: FeedForward,
    pub norm_edge: LayerNorm,
    pub se3: Se3Transformer,
    pub sc_predictor: SCPred,
}

pub struct StrOut {
    pub xyz: Vec<f32>,   // [L, n_atoms, 3]
    pub state: Tensor,   // [1, L, d_state]
    pub alpha: Tensor,   // [1, L, NTOTALDOFS, 2]
    pub quat: Vec<f32>,  // [L, 4]
}

impl Str2Str {
    #[allow(clippy::too_many_arguments)]
    pub fn load(
        p: &Params,
        se3_num_layers: usize,
        l0_in: usize,
        l1_in: usize,
        num_channels: usize,
        num_degrees: usize,
        l0_out: usize,
        l1_out: usize,
        n_heads: usize,
        div: usize,
        p_drop: f64,
    ) -> Self {
        let fiber_in = Fiber::new(&[(0, l0_in), (1, l1_in)]);
        let fiber_hidden =
            Fiber::new(&(0..num_degrees).map(|d| (d, num_channels)).collect::<Vec<_>>());
        let fiber_out = Fiber::new(&[(0, l0_out), (1, l1_out)]);
        Str2Str {
            norm_msa: LayerNorm::load(&p.sub("norm_msa")),
            norm_pair: LayerNorm::load(&p.sub("norm_pair")),
            norm_state: LayerNorm::load(&p.sub("norm_state")),
            embed_node: Linear::load(&p.sub("embed_node")),
            ff_node: FeedForward::load(&p.sub("ff_node"), p_drop),
            norm_node: LayerNorm::load(&p.sub("norm_node")),
            embed_edge: Linear::load(&p.sub("embed_edge")),
            ff_edge: FeedForward::load(&p.sub("ff_edge"), p_drop),
            norm_edge: LayerNorm::load(&p.sub("norm_edge")),
            se3: Se3Transformer::load(
                &p.sub("se3").sub("se3"),
                se3_num_layers,
                &fiber_in,
                &fiber_hidden,
                &fiber_out,
                n_heads,
                div,
            ),
            sc_predictor: SCPred::load(&p.sub("sc_predictor")),
        }
    }

    /// `xyz_frame_from_rotation_mask`: for every ligand atom, replace its three
    /// backbone slots with the coordinates of its frame atoms.
    ///
    /// The frame entries are `(residue offset, atom slot)` *relative to the
    /// ligand-local index*, and the flattening is over the ligand block only —
    /// `(i + offset) * n_atoms + slot` indexes `atom_crds.reshape(-1, 3)`, not
    /// the full-length coordinate array.
    pub fn xyz_frame(
        xyz: &[f32],
        n_res: usize,
        n_atoms: usize,
        rotation_mask: &[bool],
        atom_frames: &[i64],
    ) -> Vec<f32> {
        let mut out = xyz.to_vec();
        let atom_idx: Vec<usize> = (0..n_res).filter(|&i| rotation_mask[i]).collect();
        if atom_idx.is_empty() {
            return out;
        }
        // atom_crds[i] = xyz[atom_idx[i]]
        let flat: Vec<f32> = atom_idx
            .iter()
            .flat_map(|&i| xyz[i * n_atoms * 3..(i + 1) * n_atoms * 3].to_vec())
            .collect();
        for (i, &res) in atom_idx.iter().enumerate() {
            for f in 0..3 {
                let off = atom_frames[(i * 3 + f) * 2];
                let slot = atom_frames[(i * 3 + f) * 2 + 1];
                let src = ((i as i64 + off) * n_atoms as i64 + slot) as usize;
                for k in 0..3 {
                    out[(res * n_atoms + f) * 3 + k] = flat[src * 3 + k];
                }
            }
        }
        out
    }

    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        msa: &Tensor,   // [1,N,L,d_msa]
        pair: &Tensor,  // [1,L,L,d_pair]
        xyz: &[f32],    // [L, n_atoms, 3]
        n_atoms: usize,
        state: &Tensor, // [1,L,d_state]
        idx: &[i64],
        rotation_mask: &[bool],
        bond_feats: &[i64],
        dist_matrix: &[f32],
        atom_frames: &[i64],
        is_motif: &[bool],
        extra_l0: Option<&[f32]>,
        extra_l1: &[f32], // [L, n_extra, 3]
        n_extra_l1: usize,
        top_k: i64,
        ctx: &mut Ctx,
    ) -> StrOut {
        let l = idx.len();
        let d_msa = msa.last();

        // node features
        let seq = Tensor::new(msa.data[..l * d_msa].to_vec(), vec![1, l, d_msa]);
        let seq = self.norm_msa.forward(&seq);
        let pair_n = self.norm_pair.forward(pair);
        let st = self.norm_state.forward(state);
        let ds = st.last();
        let mut cat = vec![0.0f32; l * (d_msa + ds)];
        for i in 0..l {
            cat[i * (d_msa + ds)..i * (d_msa + ds) + d_msa]
                .copy_from_slice(&seq.data[i * d_msa..(i + 1) * d_msa]);
            cat[i * (d_msa + ds) + d_msa..(i + 1) * (d_msa + ds)]
                .copy_from_slice(&st.data[i * ds..(i + 1) * ds]);
        }
        let mut node = self.embed_node.forward(&Tensor::new(cat, vec![1, l, d_msa + ds]));
        let d = self.ff_node.forward(&node, ctx);
        for (i, v) in node.data.iter_mut().enumerate() {
            *v += d.data[i];
        }
        let node = self.norm_node.forward(&node);

        // edge features
        let neighbor = geom::seqsep_protein_sm(idx, bond_feats, dist_matrix, rotation_mask);
        let ca: Vec<f32> = (0..l)
            .flat_map(|i| xyz[(i * n_atoms + 1) * 3..(i * n_atoms + 1) * 3 + 3].to_vec())
            .collect();
        let rbf = geom::rbf_ca(&ca, l);
        let dp = pair_n.last();
        let w = dp + geom::D_COUNT + 1;
        let mut edge = vec![0.0f32; l * l * w];
        for k in 0..l * l {
            let o = k * w;
            edge[o..o + dp].copy_from_slice(&pair_n.data[k * dp..(k + 1) * dp]);
            edge[o + dp..o + dp + geom::D_COUNT]
                .copy_from_slice(&rbf.data[k * geom::D_COUNT..(k + 1) * geom::D_COUNT]);
            edge[o + w - 1] = neighbor[k];
        }
        let mut edge = self.embed_edge.forward(&Tensor::new(edge, vec![1, l, l, w]));
        let d = self.ff_edge.forward(&edge, ctx);
        for (i, v) in edge.data.iter_mut().enumerate() {
            *v += d.data[i];
        }
        let edge = self.norm_edge.forward(&edge);
        let de = edge.last();

        let graph = if top_k > 0 {
            se3::make_topk_graph(&ca, idx, top_k as usize)
        } else {
            se3::make_full_graph(&ca, idx)
        };
        let n_e = graph.n_edges();
        let mut edge_feats = vec![0.0f32; n_e * de];
        for e in 0..n_e {
            let (i, j) = (graph.src[e] as usize, graph.dst[e] as usize);
            edge_feats[e * de..(e + 1) * de]
                .copy_from_slice(&edge.data[(i * l + j) * de..(i * l + j + 1) * de]);
        }
        let edge_feats = Tensor::new(edge_feats, vec![n_e, de]);

        // degree-1 node features: frame offsets, then the extra channels
        let frames = Self::xyz_frame(xyz, l, n_atoms, rotation_mask, atom_frames);
        let n_l1 = n_atoms + n_extra_l1;
        let mut l1 = vec![0.0f32; l * n_l1 * 3];
        for i in 0..l {
            let camid = [
                frames[(i * n_atoms + 1) * 3],
                frames[(i * n_atoms + 1) * 3 + 1],
                frames[(i * n_atoms + 1) * 3 + 2],
            ];
            for a in 0..n_atoms {
                for k in 0..3 {
                    l1[(i * n_l1 + a) * 3 + k] = frames[(i * n_atoms + a) * 3 + k] - camid[k];
                }
            }
            for a in 0..n_extra_l1 {
                for k in 0..3 {
                    l1[(i * n_l1 + n_atoms + a) * 3 + k] = extra_l1[(i * n_extra_l1 + a) * 3 + k];
                }
            }
        }

        let dn = node.last();
        let n_l0 = dn + extra_l0.map(|e| e.len() / l).unwrap_or(0);
        let mut l0 = vec![0.0f32; l * n_l0];
        for i in 0..l {
            l0[i * n_l0..i * n_l0 + dn].copy_from_slice(&node.data[i * dn..(i + 1) * dn]);
            if let Some(e) = extra_l0 {
                let ne = e.len() / l;
                l0[i * n_l0 + dn..(i + 1) * n_l0].copy_from_slice(&e[i * ne..(i + 1) * ne]);
            }
        }

        let shift = self.se3.forward(&graph, &[l0, l1], &edge_feats);
        let state_out = Tensor::new(shift[0].clone(), vec![1, l, shift[0].len() / l]);

        // offset -> quaternion -> rotation, then apply to the local frame
        let off = &shift[1]; // [L, 2, 3]
        let mut quat = vec![0.0f32; l * 4];
        let mut xyz_out = vec![0.0f32; l * n_atoms * 3];
        for i in 0..l {
            let motif = is_motif[i];
            let t = |k: usize| if motif { 0.0f32 } else { off[(i * 2) * 3 + k] / 10.0 };
            let r = |k: usize| if motif { 0.0f32 } else { off[(i * 2 + 1) * 3 + k] / 100.0 };
            let (r0, r1, r2) = (r(0), r(1), r(2));
            // Qnorm = sqrt(1 + sum(R*R)); the sum is pinned, the sqrt is pinned
            let ssum = {
                let mut s = Acc::new();
                for v in [r0, r1, r2] {
                    s.add((v * v) as f64);
                }
                s.get() as f32
            };
            let qn = ((1.0f32 + ssum) as f64).sqrt() as f32;
            let (qa, qb, qc, qd) = (1.0 / qn, r0 / qn, r1 / qn, r2 / qn);
            quat[i * 4] = qa;
            quat[i * 4 + 1] = qb;
            quat[i * 4 + 2] = qc;
            quat[i * 4 + 3] = qd;
            let rot = if rotation_mask[i] {
                [[1.0f32, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
            } else {
                [
                    [
                        qa * qa + qb * qb - qc * qc - qd * qd,
                        2.0 * qb * qc - 2.0 * qa * qd,
                        2.0 * qb * qd + 2.0 * qa * qc,
                    ],
                    [
                        2.0 * qb * qc + 2.0 * qa * qd,
                        qa * qa - qb * qb + qc * qc - qd * qd,
                        2.0 * qc * qd - 2.0 * qa * qb,
                    ],
                    [
                        2.0 * qb * qd - 2.0 * qa * qc,
                        2.0 * qc * qd + 2.0 * qa * qb,
                        qa * qa - qb * qb - qc * qc + qd * qd,
                    ],
                ]
            };
            let cai = [
                xyz[(i * n_atoms + 1) * 3],
                xyz[(i * n_atoms + 1) * 3 + 1],
                xyz[(i * n_atoms + 1) * 3 + 2],
            ];
            for a in 0..n_atoms {
                let v = [
                    xyz[(i * n_atoms + a) * 3] - cai[0],
                    xyz[(i * n_atoms + a) * 3 + 1] - cai[1],
                    xyz[(i * n_atoms + a) * 3 + 2] - cai[2],
                ];
                for k in 0..3 {
                    // einsum('blij,blaj->blai') is pinned
                    let mut acc = Acc::new();
                    for j in 0..3 {
                        acc.add(rot[k][j] as f64 * v[j] as f64);
                    }
                    xyz_out[(i * n_atoms + a) * 3 + k] = acc.get() as f32 + cai[k] + t(k);
                }
            }
        }

        let seq0 = Tensor::new(msa.data[..l * d_msa].to_vec(), vec![1, l, d_msa]);
        let alpha = self.sc_predictor.forward(&seq0, &state_out);
        StrOut { xyz: xyz_out, state: state_out, alpha, quat }
    }
}

/// The per-block chiral degree-1 features: `calc_chiral_grads(xyz.detach(),
/// chirals)`, recomputed **every block** from the current coordinates.
pub fn chiral_extra_l1(xyz: &[f32], l: usize, n_atoms: usize, chirals: &[f32]) -> Vec<f32> {
    chiral_grads(xyz, l, n_atoms, chirals)
}

/// The graph an `IterBlock` builds around `Str2Str`. Kept as a free function
/// because `IterBlockTrunk` (the MSA/pair half) already lives in `track`.
pub fn block_rbf_and_pos(
    ca: &[f32],
    l: usize,
    pos: &crate::model::embeddings::PositionalEncoding2D,
    seq_unmasked: &[i64],
    idx: &[i64],
    bond_feats: &[i64],
    dist_matrix: &[f32],
    same_chain: &[bool],
) -> Tensor {
    let mut rbf = geom::rbf_ca(ca, l);
    let p = pos.forward(seq_unmasked, idx, bond_feats, dist_matrix, same_chain);
    for (i, v) in rbf.data.iter_mut().enumerate() {
        *v += p.data[i];
    }
    rbf
}
