//! `ProteinFeatures`: builds the k-nearest-neighbour graph and its edge features.
//!
//! ```text
//!   X [L,4,3] ──► virtual Cb ──► Ca-Ca distance matrix ──► top-k (k=48, smallest)
//!                                                             │  E_idx [L,K]
//!            ┌────────────────────────────────────────────────┘
//!            ├─ 25 ordered atom-pair RBF blocks × 16 bins        = 400 features
//!            └─ relative-position one-hot (66) ──► Linear(66,16) =  16 features
//!                                                     concat ──► 416
//!                              Linear(416,128, no bias) ──► LayerNorm ──► E
//! ```
//!
//! The 25 pair blocks must stay in the upstream order — they are concatenated
//! into one 400-wide vector consumed by a single dense layer, so any permutation
//! silently produces wrong (but plausible) edges.

use crate::ops;
use crate::tensor::Tensor;
use crate::weights::Weights;

pub const NUM_RBF: usize = 16;
pub const MAX_RELATIVE_FEATURE: i64 = 32;

/// `torch.linspace(2., 22., 16)` in fp32.
///
/// ATen's linspace does not compute `start + i*step` throughout: past the
/// halfway point it switches to `end - (steps-1-i)*step` so the last element is
/// exactly `end`. Reproduced here because these centres feed every RBF.
fn rbf_centres() -> [f32; NUM_RBF] {
    let (start, end, steps) = (2.0f32, 22.0f32, NUM_RBF);
    let step = (end - start) / (steps - 1) as f32;
    let halfway = steps / 2;
    let mut out = [0.0f32; NUM_RBF];
    for (i, o) in out.iter_mut().enumerate() {
        *o = if i < halfway {
            start + step * i as f32
        } else {
            end - step * (steps - i - 1) as f32
        };
    }
    out
}

/// `_rbf`: `exp(-((D - mu)/sigma)^2)` over the 16 centres, appended to `out`.
#[inline]
fn rbf_into(d: f32, centres: &[f32; NUM_RBF], out: &mut Vec<f32>) {
    const SIGMA: f32 = (22.0 - 2.0) / NUM_RBF as f32; // 1.25
    for &mu in centres.iter() {
        let t = (d - mu) / SIGMA;
        out.push(libm::expf(-(t * t)));
    }
}

/// Squared distance accumulated in the same order as `torch.sum(dX**2, -1)`.
#[inline]
fn dist(a: &[f32; 3], b: &[f32; 3]) -> f32 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dz = a[2] - b[2];
    let s = (dx * dx + dy * dy) + dz * dz;
    (s + 1e-6).sqrt()
}

/// Ca-Ca kNN graph with the masked neighbour distances `_dist` returns.
/// Exposed so parity tests can separate the geometry from the RBF transform.
pub fn ca_knn(x: &[f32], mask: &[f32], top_k: usize) -> (usize, Vec<i64>, Vec<f32>) {
    let l = mask.len();
    let ca: Vec<[f32; 3]> = (0..l)
        .map(|i| [x[(i * 4 + 1) * 3], x[(i * 4 + 1) * 3 + 1], x[(i * 4 + 1) * 3 + 2]])
        .collect();
    let (k, e_idx) = knn(&ca, mask, top_k);
    let mut d = vec![0.0f32; l * k];
    for i in 0..l {
        for t in 0..k {
            let j = e_idx[i * k + t] as usize;
            d[i * k + t] = mask[i] * mask[j] * dist(&ca[i], &ca[j]);
        }
    }
    (k, e_idx, d)
}

/// `_rbf` for a single distance — exposed for parity tests.
pub fn rbf(d: f32) -> [f32; NUM_RBF] {
    let centres = rbf_centres();
    let mut v = Vec::with_capacity(NUM_RBF);
    rbf_into(d, &centres, &mut v);
    let mut out = [0.0f32; NUM_RBF];
    out.copy_from_slice(&v);
    out
}

/// Everything the encoder and decoder need about the graph.
pub struct Graph {
    pub l: usize,
    pub k: usize,
    /// `[L,K]` neighbour indices.
    pub e_idx: Vec<i64>,
    /// `[L,K,128]` edge embeddings.
    pub e: Tensor,
}

pub struct FeatureWeights {
    pos_w: Tensor,
    pos_b: Tensor,
    edge_w: Tensor,
    norm_w: Tensor,
    norm_b: Tensor,
}

impl FeatureWeights {
    pub fn load(w: &Weights) -> Self {
        FeatureWeights {
            pos_w: w.get("features.embeddings.linear.weight"),
            pos_b: w.get("features.embeddings.linear.bias"),
            edge_w: w.get("features.edge_embedding.weight"),
            norm_w: w.get("features.norm_edges.weight"),
            norm_b: w.get("features.norm_edges.bias"),
        }
    }
}

/// Virtual Cb from the backbone, in the upstream expression order.
///
/// `torch.cross`'s CPU kernel is a plain scalar loop, which PyTorch compiles
/// with FMA contraction (GCC's default `-ffp-contract=fast`), so `a1*b2 - a2*b1`
/// becomes `fma(a1, b2, -(a2*b1))`. Without that, ~1 in 150 Cb coordinates comes
/// out 1 ULP off, which then perturbs every Cb-derived RBF feature.
pub fn virtual_cb(x: &[f32], l: usize) -> Vec<[f32; 3]> {
    let at = |i: usize, a: usize, k: usize| x[(i * 4 + a) * 3 + k];
    let mut cb = Vec::with_capacity(l);
    for i in 0..l {
        let b = [
            at(i, 1, 0) - at(i, 0, 0),
            at(i, 1, 1) - at(i, 0, 1),
            at(i, 1, 2) - at(i, 0, 2),
        ];
        let c = [
            at(i, 2, 0) - at(i, 1, 0),
            at(i, 2, 1) - at(i, 1, 1),
            at(i, 2, 2) - at(i, 1, 2),
        ];
        // torch.cross(b, c, dim=-1), FMA-contracted as above.
        let a = [
            b[1].mul_add(c[2], -(b[2] * c[1])),
            b[2].mul_add(c[0], -(b[0] * c[2])),
            b[0].mul_add(c[1], -(b[1] * c[0])),
        ];
        const CA_: f32 = -0.58273431;
        const CB_: f32 = 0.56802827;
        const CC_: f32 = -0.54067466;
        let mut v = [0.0f32; 3];
        for k in 0..3 {
            // -0.58273431*a + 0.56802827*b - 0.54067466*c + Ca
            v[k] = ((CA_ * a[k] + CB_ * b[k]) + CC_ * c[k]) + at(i, 1, k);
        }
        cb.push(v);
    }
    cb
}

/// `_dist`: masked Ca-Ca distances, then `topk(..., largest=False)`.
///
/// Masked-out rows/columns are pushed to the back by adding the row max, so a
/// residue with missing atoms is only ever chosen as a neighbour when there are
/// fewer than K real candidates.
fn knn(ca: &[[f32; 3]], mask: &[f32], top_k: usize) -> (usize, Vec<i64>) {
    let l = ca.len();
    let k = top_k.min(l);
    let mut e_idx = vec![0i64; l * k];
    let mut row = vec![0.0f32; l];
    for i in 0..l {
        let mut dmax = f32::NEG_INFINITY;
        for j in 0..l {
            let m2 = mask[i] * mask[j];
            let d = m2 * dist(&ca[i], &ca[j]);
            row[j] = d;
            if d > dmax {
                dmax = d;
            }
        }
        for j in 0..l {
            let m2 = mask[i] * mask[j];
            row[j] += (1.0 - m2) * dmax;
        }
        // torch.topk(..., largest=False, sorted=True); ties resolved by index so
        // the result is deterministic (real distances are distinct in practice).
        let mut ord: Vec<u32> = (0..l as u32).collect();
        ord.sort_by(|&a, &b| {
            row[a as usize]
                .partial_cmp(&row[b as usize])
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.cmp(&b))
        });
        for t in 0..k {
            e_idx[i * k + t] = ord[t] as i64;
        }
    }
    (k, e_idx)
}

/// Build the graph and its 128-d edge embeddings.
pub fn protein_features(
    fw: &FeatureWeights,
    x: &[f32],
    mask: &[f32],
    residue_idx: &[i64],
    chain_labels: &[i64],
    top_k: usize,
) -> Graph {
    let (l, k, e_idx, raw) = edge_input(fw, x, mask, residue_idx, chain_labels, top_k);
    let e = ops::linear(&raw, &fw.edge_w, None);
    let e = ops::layer_norm(&e, &fw.norm_w, &fw.norm_b, 1e-5);
    Graph { l, k, e_idx, e }
}

/// The pre-projection edge tensor: `[L, K, 416]` = 16 positional + 400 RBF.
///
/// Split out so tests can check the purely geometric part (no matmul, so it is
/// bit-exact against PyTorch) separately from the 416-wide GEMM that follows.
pub fn edge_input(
    fw: &FeatureWeights,
    x: &[f32],
    mask: &[f32],
    residue_idx: &[i64],
    chain_labels: &[i64],
    top_k: usize,
) -> (usize, usize, Vec<i64>, Tensor) {
    let l = mask.len();
    let at = |i: usize, a: usize| -> [f32; 3] {
        [x[(i * 4 + a) * 3], x[(i * 4 + a) * 3 + 1], x[(i * 4 + a) * 3 + 2]]
    };
    let n: Vec<[f32; 3]> = (0..l).map(|i| at(i, 0)).collect();
    let ca: Vec<[f32; 3]> = (0..l).map(|i| at(i, 1)).collect();
    let c: Vec<[f32; 3]> = (0..l).map(|i| at(i, 2)).collect();
    let o: Vec<[f32; 3]> = (0..l).map(|i| at(i, 3)).collect();
    let cb = virtual_cb(x, l);

    let (k, e_idx) = knn(&ca, mask, top_k);
    let centres = rbf_centres();

    // The 25 (A, B) atom pairs, in upstream order. Note Ca-Ca comes from the
    // masked `D_neighbors` produced by `_dist`, not from `_get_rbf`, so it is
    // handled separately below.
    let pairs: [(&Vec<[f32; 3]>, &Vec<[f32; 3]>); 24] = [
        (&n, &n),
        (&c, &c),
        (&o, &o),
        (&cb, &cb),
        (&ca, &n),
        (&ca, &c),
        (&ca, &o),
        (&ca, &cb),
        (&n, &c),
        (&n, &o),
        (&n, &cb),
        (&cb, &c),
        (&cb, &o),
        (&o, &c),
        (&n, &ca),
        (&c, &ca),
        (&o, &ca),
        (&cb, &ca),
        (&c, &n),
        (&o, &n),
        (&cb, &n),
        (&c, &cb),
        (&o, &cb),
        (&c, &o),
    ];

    // RBF_all: [L, K, 400]
    let mut rbf_all = Vec::with_capacity(l * k * 25 * NUM_RBF);
    for i in 0..l {
        for t in 0..k {
            let j = e_idx[i * k + t] as usize;
            // Ca-Ca uses the masked distance from `_dist`.
            let m2 = mask[i] * mask[j];
            rbf_into(m2 * dist(&ca[i], &ca[j]), &centres, &mut rbf_all);
            for (a, b) in pairs.iter() {
                rbf_into(dist(&a[i], &b[j]), &centres, &mut rbf_all);
            }
        }
    }
    let rbf_all = Tensor::new(rbf_all, vec![l, k, 25 * NUM_RBF]);

    // Relative positional encoding -> one-hot(66) -> Linear(66, 16).
    let mut onehot = vec![0.0f32; l * k * 66];
    for i in 0..l {
        for t in 0..k {
            let j = e_idx[i * k + t] as usize;
            let offset = residue_idx[i] - residue_idx[j];
            let same_chain = (chain_labels[i] - chain_labels[j] == 0) as i64;
            let clipped = (offset + MAX_RELATIVE_FEATURE).clamp(0, 2 * MAX_RELATIVE_FEATURE);
            let d = clipped * same_chain + (1 - same_chain) * (2 * MAX_RELATIVE_FEATURE + 1);
            onehot[(i * k + t) * 66 + d as usize] = 1.0;
        }
    }
    let onehot = Tensor::new(onehot, vec![l, k, 66]);
    let e_positional = ops::linear(&onehot, &fw.pos_w, Some(&fw.pos_b));

    (l, k, e_idx, Tensor::cat_last(&[&e_positional, &rbf_all]))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Pinned against `torch.linspace(2., 22., 16)`.
    #[test]
    fn linspace_matches_torch() {
        let c = rbf_centres();
        assert_eq!(c[0], 2.0);
        assert_eq!(c[NUM_RBF - 1], 22.0);
        for (i, v) in c.iter().enumerate() {
            let ideal = 2.0 + 20.0 * i as f64 / 15.0;
            assert!((*v as f64 - ideal).abs() < 1e-5, "centre {i}: {v} vs {ideal}");
        }
    }
}
