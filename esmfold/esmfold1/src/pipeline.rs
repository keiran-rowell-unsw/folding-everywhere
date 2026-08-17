//! End-to-end ESMFold v1 forward: tokenize -> ESM-2 -> LM/trunk glue ->
//! 4-recycle trunk+structure -> heads -> atom37.

use crate::constants::{seq_to_aatype, Constants};
use crate::esm2::esm2_states_cb;
use crate::heads;
use crate::ops;
use crate::structure::{structure_module, StructIter};
use crate::tensor::Tensor;
use crate::tokenizer::tokenize;
use crate::trunk;
use crate::weights::Weights;

const NUM_RECYCLES: usize = 4;
const RECYCLE_BINS: usize = 15;

/// LM 37-state stack (each [L+2,2560]) -> s_s_0 [L,1024].
pub fn lm_to_trunk(states: &[Tensor], aatype: &[usize], w: &Weights) -> Tensor {
    let lp2 = states[0].shape[0];
    let l = lp2 - 2;
    let d = states[0].shape[1]; // 2560
    let n_layers = states.len(); // 37
    // softmax(esm_s_combine)
    let comb = w.get("esm_s_combine").data;
    let m = comb.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sm = vec![0.0f32; n_layers];
    let mut sum = 0.0f32;
    for k in 0..n_layers {
        sm[k] = libm::expf(comb[k] - m);
        sum += sm[k];
    }
    for x in sm.iter_mut() {
        *x /= sum;
    }
    // combined[l,c] = sum_layer sm[layer] * state[layer][(l+1), c]
    let mut combined = vec![0.0f32; l * d];
    for layer in 0..n_layers {
        let wgt = sm[layer];
        let sd = &states[layer].data;
        for li in 0..l {
            let src = (li + 1) * d;
            let dst = li * d;
            for c in 0..d {
                combined[dst + c] += wgt * sd[src + c];
            }
        }
    }
    let combined = Tensor::new(combined, vec![l, d]);
    // esm_s_mlp: LN -> Linear -> ReLU -> Linear
    let h = ops::layer_norm(&combined, &w.get("esm_s_mlp.0.weight"), &w.get("esm_s_mlp.0.bias"), 1e-5);
    let h = ops::linear(&h, &w.get("esm_s_mlp.1.weight"), Some(&w.get("esm_s_mlp.1.bias")));
    let h = ops::relu(&h);
    let mut s_s0 = ops::linear(&h, &w.get("esm_s_mlp.3.weight"), Some(&w.get("esm_s_mlp.3.bias")));
    // + embedding(aatype)
    let emb = w.get("embedding.weight"); // [23,1024]
    let cs = s_s0.shape[1];
    for li in 0..l {
        let row = aatype[li] * cs;
        for c in 0..cs {
            s_s0.data[li * cs + c] += emb.data[row + c];
        }
    }
    s_s0
}

/// CB distogram bins for recycling. positions [L,14,3] (atoms N,CA,C at 0,1,2).
fn distogram_bins(positions: &[f32], l: usize) -> Vec<usize> {
    // boundaries = linspace(3.375, 21.375, 14)^2
    let (minb, maxb, nb) = (3.375f32, 21.375f32, RECYCLE_BINS - 1); // 14
    let step = (maxb - minb) / (nb as f32 - 1.0);
    let boundaries: Vec<f32> = (0..nb).map(|k| {
        let v = minb + k as f32 * step;
        v * v
    }).collect();
    // CB per residue
    let mut cb = vec![[0.0f32; 3]; l];
    for i in 0..l {
        let n = [positions[(i * 14) * 3], positions[(i * 14) * 3 + 1], positions[(i * 14) * 3 + 2]];
        let ca = [positions[(i * 14 + 1) * 3], positions[(i * 14 + 1) * 3 + 1], positions[(i * 14 + 1) * 3 + 2]];
        let c = [positions[(i * 14 + 2) * 3], positions[(i * 14 + 2) * 3 + 1], positions[(i * 14 + 2) * 3 + 2]];
        let b = [ca[0] - n[0], ca[1] - n[1], ca[2] - n[2]];
        let cc = [c[0] - ca[0], c[1] - ca[1], c[2] - ca[2]];
        // a = b x cc
        let a = [b[1] * cc[2] - b[2] * cc[1], b[2] * cc[0] - b[0] * cc[2], b[0] * cc[1] - b[1] * cc[0]];
        for k in 0..3 {
            cb[i][k] = -0.58273431 * a[k] + 0.56802827 * b[k] - 0.54067466 * cc[k] + ca[k];
        }
    }
    let mut bins = vec![0usize; l * l];
    for i in 0..l {
        for j in 0..l {
            let dx = cb[i][0] - cb[j][0];
            let dy = cb[i][1] - cb[j][1];
            let dz = cb[i][2] - cb[j][2];
            let dist = dx * dx + dy * dy + dz * dz;
            bins[i * l + j] = boundaries.iter().filter(|&&bd| dist > bd).count();
        }
    }
    bins
}

pub struct FoldOutput {
    pub l: usize,
    pub aatype: Vec<usize>,
    pub atom37: Tensor,    // [L,37,3]
    pub plddt: Tensor,     // [L,37]
    pub ptm: f32,
    pub pae: Tensor,       // [L,L]
    pub distogram: Tensor, // [L,L,64]
    pub s_z: Tensor,
    pub structure: Vec<StructIter>,
}

pub fn fold(w: &Weights, consts: &Constants, seq: &str) -> FoldOutput {
    fold_cb(w, consts, seq, &mut |_, _| {})
}

/// As `fold`, reporting progress via `prog(message, fraction 0..1)`.
pub fn fold_cb(w: &Weights, consts: &Constants, seq: &str, prog: &mut dyn FnMut(&str, f32)) -> FoldOutput {
    let ids = tokenize(seq);
    let aatype = seq_to_aatype(seq);
    let l = seq.chars().count();
    // progress weighting: ESM ~30%, each recycle ~16%, heads ~2%
    prog("Running ESM-2 language model…", 0.0);
    let states = esm2_states_cb(w, &ids, &mut |layer| {
        prog(&format!("ESM-2 transformer: layer {layer}/36"), 0.30 * layer as f32 / 36.0);
    });
    prog("Combining language-model features…", 0.30);
    let s_s0 = lm_to_trunk(&states, &aatype, w);
    let s_z0 = Tensor::zeros(&[l, l, trunk::C_Z]);

    let mut recycle_s = Tensor::zeros(&[l, trunk::C_S]);
    let mut recycle_z = Tensor::zeros(&[l, l, trunk::C_Z]);
    let mut recycle_bins = vec![0usize; l * l];

    let disto_w = w.get("trunk.recycle_disto.weight"); // [15,128]
    let mut s_z = s_z0.clone();
    let mut structure: Vec<StructIter> = Vec::new();

    for r in 0..NUM_RECYCLES {
        let base = 0.30 + 0.165 * r as f32; // each recycle ~16.5%
        let rs = ops::layer_norm(&recycle_s, &w.get("trunk.recycle_s_norm.weight"), &w.get("trunk.recycle_s_norm.bias"), 1e-5);
        let mut rz = ops::layer_norm(&recycle_z, &w.get("trunk.recycle_z_norm.weight"), &w.get("trunk.recycle_z_norm.bias"), 1e-5);
        for ij in 0..l * l {
            let b = recycle_bins[ij] * trunk::C_Z;
            for c in 0..trunk::C_Z {
                rz.data[ij * trunk::C_Z + c] += disto_w.data[b + c];
            }
        }
        let s_in = Tensor::new(s_s0.data.iter().zip(&rs.data).map(|(a, b)| a + b).collect(), s_s0.shape.clone());
        let z_in = Tensor::new(s_z0.data.iter().zip(&rz.data).map(|(a, b)| a + b).collect(), s_z0.shape.clone());
        let (ss, sz, _) = trunk::trunk_iter_cb(&s_in, &z_in, w, l, false, &mut |blk| {
            prog(&format!("Folding trunk — recycle {}/{}: block {blk}/48", r + 1, NUM_RECYCLES), base + 0.15 * blk as f32 / 48.0);
        });
        prog(&format!("Structure module — recycle {}/{}", r + 1, NUM_RECYCLES), base + 0.15);
        let sm_single = ops::linear(&ss, &w.get("trunk.trunk2sm_s.weight"), Some(&w.get("trunk.trunk2sm_s.bias")));
        let sm_pair = ops::linear(&sz, &w.get("trunk.trunk2sm_z.weight"), Some(&w.get("trunk.trunk2sm_z.bias")));
        structure = structure_module(&sm_single, &sm_pair, &aatype, w, consts, l);
        recycle_bins = distogram_bins(&structure.last().unwrap().positions, l);
        recycle_s = ss;
        recycle_z = sz.clone();
        s_z = sz;
    }
    prog("Computing confidence (pLDDT / pTM / PAE)…", 0.97);

    let states_final = Tensor::new(structure.last().unwrap().states.clone(), vec![l, 384]);
    let distogram = heads::distogram(&s_z, w);
    let plddt = heads::plddt(&states_final, w);
    let ptm_logits = heads::ptm_logits(&s_z, w);
    let ptm = heads::compute_ptm(&ptm_logits, l);
    let pae = heads::compute_pae(&ptm_logits, l);
    let atom37 = heads::atom14_to_atom37(&structure.last().unwrap().positions, &aatype, consts, l);
    prog("Done", 1.0);

    FoldOutput { l, aatype, atom37, plddt, ptm, pae, distogram, s_z, structure }
}
