//! ESMFold2 confidence head: pLDDT, pTM, ipTM, PAE, PDE.
//! Builds a pair rep from z + s_inputs + distogram(x_pred), runs a 4-block trunk,
//! row-attention-pools to a single rep, then per-atom pLDDT and pair pTM/PAE.

use crate::config::*;
use crate::ops::*;
use crate::tensor::Tensor;
use crate::trunk;
use crate::weights::Weights;

const EPS: f32 = 1e-5;
const CP: &str = "confidence_head";

fn lin(x: &Tensor, w: &Weights, name: &str) -> Tensor { linear_f64(x, &w.get(name), None) }
fn ln(x: &Tensor, w: &Weights, p: &str) -> Tensor {
    layernorm(x, &w.get_vec(&format!("{p}.weight")), Some(&w.get_vec(&format!("{p}.bias"))), EPS)
}

/// softmax(logits) @ bin_centers, bins evenly spaced in [start,end].
fn categorical_mean(logits: &Tensor, start: f32, end: f32) -> Tensor {
    let nb = logits.last();
    let rows = logits.rows();
    let mut centers = vec![0.0f32; nb];
    for b in 0..nb {
        let e0 = start + (end - start) * (b as f32) / (nb as f32);
        let e1 = start + (end - start) * ((b + 1) as f32) / (nb as f32);
        centers[b] = (e0 + e1) / 2.0;
    }
    let sm = softmax_last(logits);
    let mut out = vec![0.0f32; rows];
    for r in 0..rows {
        let row = &sm.data[r * nb..r * nb + nb];
        let mut s = 0.0f32;
        for b in 0..nb { s += row[b] * centers[b]; }
        out[r] = s;
    }
    let mut shape = logits.shape.clone(); shape.pop();
    Tensor::new(out, shape)
}

pub struct ConfOut {
    pub plddt: Vec<f32>,       // [L]
    pub complex_plddt: f32,
    pub ptm: f32,
    pub iptm: f32,
    pub pae: Tensor,           // [L,L]
}

#[allow(clippy::too_many_arguments)]
pub fn confidence(
    w: &Weights,
    s_inputs: &Tensor,    // [L,451]
    z: &Tensor,           // [L,L,256]
    x_pred: &Tensor,      // [N,3]
    distogram_atom_idx: &[i64], // [L]
    token_mask: &[f32],   // [L]
    atom_to_token: &[i64],// [N]
    atom_mask: &[f32],    // [N]
    asym_id: &[i64],      // [L]
    rel_pos: &Tensor,     // [L,L,256]
    token_bonds_enc: &Tensor, // [L,L,256]
) -> ConfOut {
    let l = s_inputs.shape[0];
    let n = atom_to_token.len();
    let sin = ln(s_inputs, w, &format!("{CP}.s_inputs_norm")); // [L,451]
    // z_base
    let mut zb = ln(z, w, &format!("{CP}.z_norm")).data;
    for i in 0..zb.len() { zb[i] += rel_pos.data[i] + token_bonds_enc.data[i]; }
    let sz = lin(&sin, w, &format!("{CP}.s_to_z.weight")); // [L,256]
    let szt = lin(&sin, w, &format!("{CP}.s_to_z_transpose.weight"));
    let p1 = lin(&sin, w, &format!("{CP}.s_to_z_prod_in1.weight"));
    let p2 = lin(&sin, w, &format!("{CP}.s_to_z_prod_in2.weight"));
    let mut prod = vec![0.0f32; l * l * D_PAIR];
    for i in 0..l { for j in 0..l { for c in 0..D_PAIR {
        prod[(i * l + j) * D_PAIR + c] = p1.data[i * D_PAIR + c] * p2.data[j * D_PAIR + c];
    }}}
    let prod = lin(&Tensor::new(prod, vec![l, l, D_PAIR]), w, &format!("{CP}.s_to_z_prod_out.weight"));
    for i in 0..l { for j in 0..l { for c in 0..D_PAIR {
        let idx = (i * l + j) * D_PAIR + c;
        zb[idx] += sz.data[i * D_PAIR + c] + szt.data[j * D_PAIR + c] + prod.data[idx];
    }}}
    // distogram bins from predicted rep-atom coords
    let boundaries = w.get_vec(&format!("{CP}.boundaries")); // [38]
    let mut rep = vec![0.0f32; l * 3];
    for i in 0..l {
        let a = distogram_atom_idx[i] as usize;
        rep[i * 3..i * 3 + 3].copy_from_slice(&x_pred.data[a * 3..a * 3 + 3]);
    }
    let embed = w.get(&format!("{CP}.dist_bin_pairwise_embed.weight")); // [39,256]
    for i in 0..l {
        for j in 0..l {
            let dx = rep[i * 3] - rep[j * 3];
            let dy = rep[i * 3 + 1] - rep[j * 3 + 1];
            let dz = rep[i * 3 + 2] - rep[j * 3 + 2];
            let dist = (dx * dx + dy * dy + dz * dz).sqrt();
            let mut bin = 0usize;
            for &b in boundaries.iter() { if dist > b { bin += 1; } }
            let base = (i * l + j) * D_PAIR;
            for c in 0..D_PAIR { zb[base + c] += embed.data[bin * D_PAIR + c]; }
        }
    }
    let mut pair = Tensor::new(zb, vec![l, l, D_PAIR]);
    let pmask: Vec<f32> = {
        let mut m = vec![0.0f32; l * l];
        for i in 0..l { for j in 0..l { m[i * l + j] = token_mask[i] * token_mask[j]; } }
        m
    };
    let delta = trunk::folding_trunk(w, &format!("{CP}.folding_trunk"), &pair, &pmask, CONFIDENCE_TRUNK_LAYERS);
    pair = pair.add(&delta);
    // row attention pooling -> single [L,384]
    let single = row_attention_pooling(w, &pair, token_mask, l);

    // pLDDT (per atom)
    let mut s_at = vec![0.0f32; n * D_SINGLE];
    for a in 0..n {
        let t = atom_to_token[a] as usize;
        s_at[a * D_SINGLE..a * D_SINGLE + D_SINGLE].copy_from_slice(&single.data[t * D_SINGLE..t * D_SINGLE + D_SINGLE]);
    }
    let s_at = Tensor::new(s_at, vec![n, D_SINGLE]);
    let s_at_ln = ln(&s_at, w, &format!("{CP}.plddt_ln"));
    let intra = compute_intra_token_idx(atom_to_token);
    let plddt_weight = w.get(&format!("{CP}.plddt_weight")); // [23,384,50]
    let max_intra = plddt_weight.shape[0] - 1;
    let nbins = NUM_PLDDT_BINS;
    let mut plddt_logits = vec![0.0f32; n * nbins];
    for a in 0..n {
        let ii = (intra[a] as usize).min(max_intra);
        let wbase = ii * D_SINGLE * nbins;
        for b in 0..nbins {
            let mut acc = 0.0f64;
            for c in 0..D_SINGLE { acc += s_at_ln.data[a * D_SINGLE + c] as f64 * plddt_weight.data[wbase + c * nbins + b] as f64; }
            plddt_logits[a * nbins + b] = acc as f32;
        }
    }
    let plddt_per_atom = categorical_mean(&Tensor::new(plddt_logits, vec![n, nbins]), 0.0, 1.0);
    // per-token mean (masked)
    let mut plddt = vec![0.0f32; l];
    let mut cnt = vec![0.0f32; l];
    for a in 0..n {
        let t = atom_to_token[a] as usize;
        plddt[t] += plddt_per_atom.data[a] * atom_mask[a];
        cnt[t] += atom_mask[a];
    }
    for t in 0..l { plddt[t] /= cnt[t].max(1e-6); }
    let mut cp_num = 0.0f32; let mut cp_den = 0.0f32;
    for a in 0..n { cp_num += plddt_per_atom.data[a] * atom_mask[a]; cp_den += atom_mask[a]; }
    let complex_plddt = cp_num / (cp_den + 1e-6);

    // PAE + pTM/ipTM
    let pae_logits = lin(&ln(&pair, w, &format!("{CP}.pae_ln")), w, &format!("{CP}.pae_head.weight")); // [L,L,64]
    let pae = categorical_mean(&pae_logits, 0.0, 32.0).reshape(&[l, l]);
    let (ptm, iptm) = compute_ptm(&pae_logits, token_mask, asym_id, l);

    ConfOut { plddt, complex_plddt, ptm, iptm, pae }
}

fn row_attention_pooling(w: &Weights, z: &Tensor, mask: &[f32], l: usize) -> Tensor {
    let scores = lin(z, w, &format!("{CP}.row_attention_pooling.attn_proj.weight")); // [L,L,1]
    let mut pooled = vec![0.0f32; l * D_PAIR];
    for i in 0..l {
        // softmax over j with mask bias
        let mut sc = vec![0.0f32; l];
        let mut mx = f32::NEG_INFINITY;
        for j in 0..l {
            let s = scores.data[i * l + j] + if mask[j] != 0.0 { 0.0 } else { -1e9 };
            sc[j] = s; if s > mx { mx = s; }
        }
        let mut sum = 0.0f32;
        for j in 0..l { sc[j] = (sc[j] - mx).exp(); sum += sc[j]; }
        let inv = 1.0 / sum;
        for j in 0..l {
            let wgt = sc[j] * inv;
            let zb = (i * l + j) * D_PAIR;
            for c in 0..D_PAIR { pooled[i * D_PAIR + c] += wgt * z.data[zb + c]; }
        }
    }
    lin(&Tensor::new(pooled, vec![l, D_PAIR]), w, &format!("{CP}.row_attention_pooling.out_proj.weight"))
}

fn compute_intra_token_idx(atom_to_token: &[i64]) -> Vec<i64> {
    let n = atom_to_token.len();
    let mut out = vec![0i64; n];
    let mut local = 0i64;
    for a in 0..n {
        if a > 0 && atom_to_token[a] == atom_to_token[a - 1] { local += 1; } else { local = 0; }
        out[a] = local;
    }
    out
}

fn compute_ptm(pae_logits: &Tensor, mask: &[f32], asym_id: &[i64], l: usize) -> (f32, f32) {
    let nb = pae_logits.last(); // 64
    let bin_width = 32.0f32 / nb as f32;
    let mut centers = vec![0.0f32; nb];
    for b in 0..nb { centers[b] = 0.5 * bin_width + b as f32 * bin_width; }
    let n_res: f32 = mask.iter().sum();
    let d0 = 1.24 * (n_res.max(19.0) - 15.0).powf(1.0 / 3.0) - 1.8;
    let mut tm_per_bin = vec![0.0f32; nb];
    for b in 0..nb { let r = centers[b] / d0; tm_per_bin[b] = 1.0 / (1.0 + r * r); }
    let probs = softmax_last(pae_logits); // [L,L,64]
    // tm_expected[i,j]
    let mut tm = vec![0.0f32; l * l];
    for i in 0..l { for j in 0..l {
        let base = (i * l + j) * nb;
        let mut s = 0.0f32;
        for b in 0..nb { s += probs.data[base + b] * tm_per_bin[b]; }
        tm[i * l + j] = s;
    }}
    let mut ptm = f32::NEG_INFINITY;
    let mut iptm = f32::NEG_INFINITY;
    for i in 0..l {
        let (mut num, mut den) = (0.0f32, 0.0f32);
        let (mut inum, mut iden) = (0.0f32, 0.0f32);
        for j in 0..l {
            let pm = mask[i] * mask[j];
            num += tm[i * l + j] * pm; den += pm;
            let inter = if asym_id[i] != asym_id[j] { pm } else { 0.0 };
            inum += tm[i * l + j] * inter; iden += inter;
        }
        let row = num / (den + 1e-8);
        if row > ptm { ptm = row; }
        let irow = inum / (iden + 1e-8);
        if irow > iptm { iptm = irow; }
    }
    (ptm, iptm)
}
