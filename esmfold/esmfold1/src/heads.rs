//! Output heads: distogram, pLDDT, pTM, PAE, and atom14->atom37.

use crate::constants::Constants;
use crate::ops;
use crate::tensor::Tensor;
use crate::weights::Weights;

fn lin(x: &Tensor, w: &Weights, p: &str) -> Tensor {
    ops::linear(x, &w.get(&format!("{p}.weight")), Some(&w.get(&format!("{p}.bias"))))
}

/// distogram_head(s_z) then symmetrize (d + d^T)/2. s_z [L,L,128] -> [L,L,64].
pub fn distogram(s_z: &Tensor, w: &Weights) -> Tensor {
    let d = lin(s_z, w, "distogram_head");
    let l = s_z.shape[0];
    let bins = d.shape[2];
    let mut out = vec![0.0f32; l * l * bins];
    for i in 0..l {
        for j in 0..l {
            for b in 0..bins {
                out[(i * l + j) * bins + b] = 0.5 * (d.data[(i * l + j) * bins + b] + d.data[(j * l + i) * bins + b]);
            }
        }
    }
    Tensor::new(out, vec![l, l, bins])
}

/// pLDDT [L,37] in [0,1] from the final structure-module states [L,384].
pub fn plddt(states_final: &Tensor, w: &Weights) -> Tensor {
    let h = ops::layer_norm(states_final, &w.get("lddt_head.0.weight"), &w.get("lddt_head.0.bias"), 1e-5);
    let h = lin(&h, w, "lddt_head.1");
    let h = lin(&h, w, "lddt_head.2");
    let h = lin(&h, w, "lddt_head.3"); // [L, 37*50]
    let l = states_final.shape[0];
    let bins = 50;
    // bin centers = midpoints of linspace(0,1,51)
    let centers: Vec<f32> = (0..bins).map(|k| ((k as f32 / bins as f32) + ((k + 1) as f32 / bins as f32)) * 0.5).collect();
    let mut out = vec![0.0f32; l * 37];
    for li in 0..l {
        for a in 0..37 {
            let base = li * (37 * bins) + a * bins;
            let logits = &h.data[base..base + bins];
            let m = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            let mut probs = [0.0f32; 50];
            for k in 0..bins {
                probs[k] = libm::expf(logits[k] - m);
                sum += probs[k];
            }
            let mut v = 0.0f32;
            for k in 0..bins {
                v += (probs[k] / sum) * centers[k];
            }
            out[li * 37 + a] = v;
        }
    }
    Tensor::new(out, vec![l, 37])
}

pub fn ptm_logits(s_z: &Tensor, w: &Weights) -> Tensor {
    lin(s_z, w, "ptm_head")
}

/// 64 PAE/pTM bin centers for max_bin=31.
fn bin_centers() -> Vec<f32> {
    let no_bins = 64;
    let max_bin = 31.0f32;
    let nb = no_bins - 1; // 63 boundaries
    let step = max_bin / (nb as f32 - 1.0); // linspace(0,31,63)
    let mut c: Vec<f32> = (0..nb).map(|k| k as f32 * step + step / 2.0).collect();
    c.push(c[nb - 1] + step);
    c
}

/// pTM scalar = max_i mean_j sum_bin softmax(logits)[i,j]*tm_per_bin.
pub fn compute_ptm(ptm_logits: &Tensor, l: usize) -> f32 {
    let bins = 64;
    let centers = bin_centers();
    let clipped = (l.max(19)) as f32;
    let d0 = 1.24 * (clipped - 15.0).powf(1.0 / 3.0) - 1.8;
    let tm_per_bin: Vec<f32> = centers.iter().map(|&c| 1.0 / (1.0 + (c * c) / (d0 * d0))).collect();
    let probs = softmax_last_bins(ptm_logits, l, bins);
    let mut best = f32::NEG_INFINITY;
    for i in 0..l {
        let mut acc = 0.0f32;
        for j in 0..l {
            let mut t = 0.0f32;
            for b in 0..bins {
                t += probs[(i * l + j) * bins + b] * tm_per_bin[b];
            }
            acc += t;
        }
        let per = acc / l as f32;
        if per > best {
            best = per;
        }
    }
    best
}

/// PAE [L,L] = sum_bin softmax(logits)*bin_centers.
pub fn compute_pae(ptm_logits: &Tensor, l: usize) -> Tensor {
    let bins = 64;
    let centers = bin_centers();
    let probs = softmax_last_bins(ptm_logits, l, bins);
    let mut out = vec![0.0f32; l * l];
    for i in 0..l {
        for j in 0..l {
            let mut v = 0.0f32;
            for b in 0..bins {
                v += probs[(i * l + j) * bins + b] * centers[b];
            }
            out[i * l + j] = v;
        }
    }
    Tensor::new(out, vec![l, l])
}

fn softmax_last_bins(t: &Tensor, l: usize, bins: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; l * l * bins];
    for r in 0..l * l {
        let row = &t.data[r * bins..r * bins + bins];
        let m = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for b in 0..bins {
            out[r * bins + b] = libm::expf(row[b] - m);
            sum += out[r * bins + b];
        }
        for b in 0..bins {
            out[r * bins + b] /= sum;
        }
    }
    out
}

/// atom14 [L,14,3] -> atom37 [L,37,3], masked.
pub fn atom14_to_atom37(positions: &[f32], aatype: &[usize], c: &Constants, l: usize) -> Tensor {
    let mut out = vec![0.0f32; l * 37 * 3];
    for li in 0..l {
        let a = aatype[li];
        for a37 in 0..37 {
            let idx14 = c.atom37_to_atom14[a * 37 + a37];
            let mask = c.atom37_mask[a * 37 + a37];
            for xyz in 0..3 {
                out[(li * 37 + a37) * 3 + xyz] = positions[(li * 14 + idx14) * 3 + xyz] * mask;
            }
        }
    }
    Tensor::new(out, vec![l, 37, 3])
}
