//! Message-passing layers: `EncLayer`, `DecLayer` and `PositionWiseFeedForward`.
//!
//! Both layers follow the same shape: build a `[L,K,*]` message tensor by
//! concatenating the node with its edge/neighbour context, push it through a
//! 3-layer GELU MLP, mask it, sum over the K neighbours, divide by `scale`
//! (30), and add-and-norm. Dropout is inference-time identity and omitted.

use crate::ops;
use crate::tensor::Tensor;
use crate::weights::Weights;

pub const SCALE: f32 = 30.0;

/// `gather_nodes(nodes[L,C], E_idx[L,K]) -> [L,K,C]`.
pub fn gather_nodes(nodes: &Tensor, e_idx: &[i64], k: usize) -> Tensor {
    let l = nodes.shape[0];
    let c = nodes.shape[1];
    let mut out: Vec<f32> = Vec::with_capacity(l * k * c);
    for i in 0..l {
        for t in 0..k {
            let j = e_idx[i * k + t] as usize;
            out.extend_from_slice(&nodes.data[j * c..j * c + c]);
        }
    }
    Tensor::new(out, vec![l, k, c])
}

/// `cat_neighbors_nodes(h_nodes, h_neighbors, E_idx)` =
/// `cat([h_neighbors, gather_nodes(h_nodes, E_idx)], -1)`.
///
/// Fused: the gather is written straight into the concatenated buffer instead of
/// materialising an intermediate `[L,K,C]` tensor and copying it.
pub fn cat_neighbors_nodes(
    h_nodes: &Tensor,
    h_neighbors: &Tensor,
    e_idx: &[i64],
    k: usize,
) -> Tensor {
    let l = h_neighbors.shape[0];
    let c1 = h_neighbors.last();
    let c2 = h_nodes.shape[1];
    let mut out: Vec<f32> = Vec::with_capacity(l * k * (c1 + c2));
    for i in 0..l {
        for t in 0..k {
            let base = (i * k + t) * c1;
            out.extend_from_slice(&h_neighbors.data[base..base + c1]);
            let j = e_idx[i * k + t] as usize;
            out.extend_from_slice(&h_nodes.data[j * c2..j * c2 + c2]);
        }
    }
    Tensor::new(out, vec![l, k, c1 + c2])
}

/// `cat([h_V expanded, h_E, gather(h_V)], -1)` in one pass — the message input
/// both `EncLayer` steps build. Equivalent to
/// `cat_last([expand_nodes(h_v), cat_neighbors_nodes(h_v, h_e, ...)])`.
pub fn cat_self_edge_neighbor(h_v: &Tensor, h_e: &Tensor, e_idx: &[i64], k: usize) -> Tensor {
    let l = h_e.shape[0];
    let cv = h_v.shape[1];
    let ce = h_e.last();
    let mut out: Vec<f32> = Vec::with_capacity(l * k * (cv + ce + cv));
    for i in 0..l {
        let self_row = &h_v.data[i * cv..i * cv + cv];
        for t in 0..k {
            out.extend_from_slice(self_row);
            let base = (i * k + t) * ce;
            out.extend_from_slice(&h_e.data[base..base + ce]);
            let j = e_idx[i * k + t] as usize;
            out.extend_from_slice(&h_v.data[j * cv..j * cv + cv]);
        }
    }
    Tensor::new(out, vec![l, k, cv + ce + cv])
}

/// Broadcast `h_V[L,C]` to `[L,K,C]` (the `unsqueeze(-2).expand(...)` idiom).
pub fn expand_nodes(h_v: &Tensor, k: usize) -> Tensor {
    let l = h_v.shape[0];
    let c = h_v.shape[1];
    let mut out: Vec<f32> = Vec::with_capacity(l * k * c);
    for i in 0..l {
        let row = &h_v.data[i * c..i * c + c];
        for _ in 0..k {
            out.extend_from_slice(row);
        }
    }
    Tensor::new(out, vec![l, k, c])
}

/// `sum(x[L,K,C], dim=-2) / SCALE`, accumulating over k in ascending order.
fn sum_neighbors_scaled(x: &Tensor) -> Tensor {
    let (l, k, c) = (x.shape[0], x.shape[1], x.shape[2]);
    let mut out = vec![0.0f32; l * c];
    for i in 0..l {
        let o = &mut out[i * c..i * c + c];
        for t in 0..k {
            let row = &x.data[(i * k + t) * c..(i * k + t) * c + c];
            for ci in 0..c {
                o[ci] += row[ci];
            }
        }
        for v in o.iter_mut() {
            *v /= SCALE;
        }
    }
    Tensor::new(out, vec![l, c])
}

fn mask_rows(x: &mut Tensor, mask: &[f32]) {
    let c = x.last();
    for (i, &m) in mask.iter().enumerate() {
        for v in x.data[i * c..i * c + c].iter_mut() {
            *v *= m;
        }
    }
}

struct Linear {
    w: Tensor,
    b: Tensor,
}

impl Linear {
    fn load(w: &Weights, p: &str) -> Self {
        Linear { w: w.get(&format!("{p}.weight")), b: w.get(&format!("{p}.bias")) }
    }
    fn apply(&self, x: &Tensor) -> Tensor {
        ops::linear(x, &self.w, Some(&self.b))
    }
}

struct Norm {
    w: Tensor,
    b: Tensor,
}

impl Norm {
    fn load(w: &Weights, p: &str) -> Self {
        Norm { w: w.get(&format!("{p}.weight")), b: w.get(&format!("{p}.bias")) }
    }
    fn apply(&self, x: &Tensor) -> Tensor {
        ops::layer_norm(x, &self.w, &self.b, 1e-5)
    }
}

/// `PositionWiseFeedForward`: Linear -> GELU -> Linear.
struct Ffn {
    w_in: Linear,
    w_out: Linear,
}

impl Ffn {
    fn load(w: &Weights, p: &str) -> Self {
        Ffn { w_in: Linear::load(w, &format!("{p}.W_in")), w_out: Linear::load(w, &format!("{p}.W_out")) }
    }
    fn apply(&self, x: &Tensor) -> Tensor {
        let mut h = self.w_in.apply(x);
        ops::gelu_(&mut h);
        self.w_out.apply(&h)
    }
}

pub struct EncLayer {
    w1: Linear,
    w2: Linear,
    w3: Linear,
    w11: Linear,
    w12: Linear,
    w13: Linear,
    norm1: Norm,
    norm2: Norm,
    norm3: Norm,
    dense: Ffn,
}

impl EncLayer {
    pub fn load(w: &Weights, i: usize) -> Self {
        let p = format!("encoder_layers.{i}");
        EncLayer {
            w1: Linear::load(w, &format!("{p}.W1")),
            w2: Linear::load(w, &format!("{p}.W2")),
            w3: Linear::load(w, &format!("{p}.W3")),
            w11: Linear::load(w, &format!("{p}.W11")),
            w12: Linear::load(w, &format!("{p}.W12")),
            w13: Linear::load(w, &format!("{p}.W13")),
            norm1: Norm::load(w, &format!("{p}.norm1")),
            norm2: Norm::load(w, &format!("{p}.norm2")),
            norm3: Norm::load(w, &format!("{p}.norm3")),
            dense: Ffn::load(w, &format!("{p}.dense")),
        }
    }

    /// Returns the updated `(h_V, h_E)`.
    pub fn forward(
        &self,
        h_v: &Tensor,
        h_e: &Tensor,
        e_idx: &[i64],
        k: usize,
        mask_v: &[f32],
        mask_attend: &[f32],
    ) -> (Tensor, Tensor) {
        // --- node update -----------------------------------------------------
        let h_ev = cat_self_edge_neighbor(h_v, h_e, e_idx, k);

        let mut h_message = self.w1.apply(&h_ev);
        ops::gelu_(&mut h_message);
        let mut h_message = self.w2.apply(&h_message);
        ops::gelu_(&mut h_message);
        let mut h_message = self.w3.apply(&h_message);
        // mask_attend [L,K] broadcast over the channel axis
        {
            let c = h_message.last();
            for (idx, &m) in mask_attend.iter().enumerate() {
                for v in h_message.data[idx * c..idx * c + c].iter_mut() {
                    *v *= m;
                }
            }
        }
        let dh = sum_neighbors_scaled(&h_message);
        let h_v = self.norm1.apply(&h_v.add(&dh));

        let dh = self.dense.apply(&h_v);
        let mut h_v = self.norm2.apply(&h_v.add(&dh));
        mask_rows(&mut h_v, mask_v);

        // --- edge update -----------------------------------------------------
        let h_ev = cat_self_edge_neighbor(&h_v, h_e, e_idx, k);

        let mut h_message = self.w11.apply(&h_ev);
        ops::gelu_(&mut h_message);
        let mut h_message = self.w12.apply(&h_message);
        ops::gelu_(&mut h_message);
        let h_message = self.w13.apply(&h_message);
        let h_e = self.norm3.apply(&h_e.add(&h_message));

        (h_v, h_e)
    }
}

pub struct DecLayer {
    w1: Linear,
    w2: Linear,
    w3: Linear,
    norm1: Norm,
    norm2: Norm,
    dense: Ffn,
}

impl DecLayer {
    pub fn load(w: &Weights, i: usize) -> Self {
        let p = format!("decoder_layers.{i}");
        DecLayer {
            w1: Linear::load(w, &format!("{p}.W1")),
            w2: Linear::load(w, &format!("{p}.W2")),
            w3: Linear::load(w, &format!("{p}.W3")),
            norm1: Norm::load(w, &format!("{p}.norm1")),
            norm2: Norm::load(w, &format!("{p}.norm2")),
            dense: Ffn::load(w, &format!("{p}.dense")),
        }
    }

    /// `h_V [L,C]`, `h_E [L,K,C3]` -> `[L,C]`.
    pub fn forward(&self, h_v: &Tensor, h_e: &Tensor, mask_v: &[f32]) -> Tensor {
        let k = h_e.shape[1];
        // cat([h_V expanded over K, h_E], -1), built in one pass.
        let h_ev = {
            let (l, cv, ce) = (h_e.shape[0], h_v.shape[1], h_e.last());
            let mut out: Vec<f32> = Vec::with_capacity(l * k * (cv + ce));
            for i in 0..l {
                let self_row = &h_v.data[i * cv..i * cv + cv];
                for t in 0..k {
                    out.extend_from_slice(self_row);
                    let base = (i * k + t) * ce;
                    out.extend_from_slice(&h_e.data[base..base + ce]);
                }
            }
            Tensor::new(out, vec![l, k, cv + ce])
        };

        let mut h_message = self.w1.apply(&h_ev);
        ops::gelu_(&mut h_message);
        let mut h_message = self.w2.apply(&h_message);
        ops::gelu_(&mut h_message);
        let h_message = self.w3.apply(&h_message);

        let dh = sum_neighbors_scaled(&h_message);
        let h_v2 = self.norm1.apply(&h_v.add(&dh));
        let dh = self.dense.apply(&h_v2);
        let mut out = self.norm2.apply(&h_v2.add(&dh));
        mask_rows(&mut out, mask_v);
        out
    }
}
