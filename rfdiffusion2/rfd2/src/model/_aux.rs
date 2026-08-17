//! `rf2aa/model/layers/AuxiliaryPredictor.py` — the heads.
//!
//! None of these feed the coordinates: `px0` comes from the simulator's rigids,
//! not from `c6d_pred`/`aa_pred`/`lddt_pred`/`pae_pred`. They are ported because
//! the sampler writes some of them into the `.trb` and because SOP §5.6 counts
//! the output format as part of the port.

use crate::ops::acc::Acc;
use crate::nn::{Linear, Params};
use crate::ops::elem::sigmoid_scalar;
use crate::tensor::Tensor;

/// `DistanceNetwork`: dist/omega from a **symmetrised** projection, theta/phi
/// from an asymmetric one. The symmetrisation `logits + logits.permute(0,2,1,3)`
/// happens *before* the split, so `dist` and `omega` are both symmetric and
/// `theta`/`phi` are not.
pub struct DistanceNetwork {
    pub proj_symm: Linear,
    pub proj_asymm: Linear,
}

pub struct C6d {
    pub dist: Tensor,  // [1, 61, L, L]
    pub omega: Tensor, // [1, 37, L, L]
    pub theta: Tensor, // [1, 37, L, L]
    pub phi: Tensor,   // [1, 19, L, L]
}

fn to_channels_first(x: &[f32], l: usize, w: usize, lo: usize, hi: usize) -> Tensor {
    let c = hi - lo;
    let mut out = vec![0.0f32; c * l * l];
    for i in 0..l {
        for j in 0..l {
            for k in lo..hi {
                out[((k - lo) * l + i) * l + j] = x[(i * l + j) * w + k];
            }
        }
    }
    Tensor::new(out, vec![1, c, l, l])
}

impl DistanceNetwork {
    pub fn load(p: &Params) -> Self {
        DistanceNetwork {
            proj_symm: Linear::load(&p.sub("proj_symm")),
            proj_asymm: Linear::load(&p.sub("proj_asymm")),
        }
    }

    pub fn forward(&self, pair: &Tensor) -> C6d {
        let l = pair.shape[1];
        let asym = self.proj_asymm.forward(pair);
        let wa = asym.last();
        let mut symm = self.proj_symm.forward(pair);
        let ws = symm.last();
        let orig = symm.data.clone();
        for i in 0..l {
            for j in 0..l {
                for k in 0..ws {
                    symm.data[(i * l + j) * ws + k] += orig[(j * l + i) * ws + k];
                }
            }
        }
        C6d {
            dist: to_channels_first(&symm.data, l, ws, 0, 61),
            omega: to_channels_first(&symm.data, l, ws, 61, ws),
            theta: to_channels_first(&asym.data, l, wa, 0, 37),
            phi: to_channels_first(&asym.data, l, wa, 37, wa),
        }
    }
}

pub struct Proj {
    pub proj: Linear,
}

impl Proj {
    pub fn load(p: &Params) -> Self {
        Proj { proj: Linear::load(&p.sub("proj")) }
    }
}

/// `MaskedTokenNetwork`: `[B,N,L,d] -> [B, NAATOKENS, N*L]`.
pub fn masked_token(head: &Proj, msa: &Tensor) -> Tensor {
    let (b, n, l) = (msa.shape[0], msa.shape[1], msa.shape[2]);
    let x = head.proj.forward(msa);
    let c = x.last();
    let mut out = vec![0.0f32; b * c * n * l];
    for bi in 0..b {
        for ni in 0..n {
            for li in 0..l {
                for k in 0..c {
                    out[(bi * c + k) * n * l + ni * l + li] =
                        x.data[((bi * n + ni) * l + li) * c + k];
                }
            }
        }
    }
    Tensor::new(out, vec![b, c, n * l])
}

/// `LDDTNetwork` / `PAENetwork`: a projection plus a channel-first permute.
pub fn permute_last_to_front(x: &Tensor) -> Tensor {
    let nd = x.shape.len();
    let mut axes: Vec<usize> = vec![0, nd - 1];
    axes.extend(1..nd - 1);
    x.permute(&axes)
}

/// `BinderNetwork`: mean of the PAE logits over **inter-chain** pairs, then a
/// 1-unit linear and a sigmoid. Single-chain inputs have no such pairs, so the
/// mean is over an empty set — `nan_to_num()` turns the resulting NaN into 0,
/// and the head returns `sigmoid(bias)`.
pub struct BinderNetwork {
    pub classify: Linear,
}

impl BinderNetwork {
    pub fn load(p: &Params) -> Self {
        BinderNetwork { classify: Linear::load(&p.sub("classify")) }
    }

    /// `pae` is channel-first `[1, C, L, L]`; `same_chain` is `[L*L]`.
    pub fn forward(&self, pae: &Tensor, same_chain: &[bool]) -> f32 {
        let c = pae.shape[1];
        let l = pae.shape[2];
        let n_inter = same_chain.iter().filter(|s| !**s).count();
        let mut mean = vec![0.0f32; c];
        for (k, m) in mean.iter_mut().enumerate() {
            let mut acc = Acc::new();
            for (idx, &s) in same_chain.iter().enumerate() {
                if !s {
                    acc.add(pae.data[(k * l + idx / l) * l + idx % l] as f64);
                }
            }
            *m = if n_inter == 0 { 0.0 } else { (acc.get() / n_inter as f64) as f32 };
        }
        let logits = Tensor::new(mean, vec![1, c]);
        let out = self.classify.forward(&logits);
        sigmoid_scalar(out.data[0])
    }
}
