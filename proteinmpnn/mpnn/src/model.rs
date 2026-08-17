//! The `ProteinMPNN` encoder/decoder, plus the two entry points the CLI uses:
//! `forward` (score a given sequence) and `sample` (design a new one).

use crate::features::{protein_features, FeatureWeights, Graph};
use crate::featurize::Batch;
use crate::layers::{cat_neighbors_nodes, DecLayer, EncLayer};
use crate::ops;
use crate::rng::Mt19937;
use crate::tensor::Tensor;
use crate::weights::Weights;

pub struct ProteinMpnn {
    fw: FeatureWeights,
    w_e: (Tensor, Tensor),
    w_s: Tensor,
    w_out: (Tensor, Tensor),
    encoder: Vec<EncLayer>,
    decoder: Vec<DecLayer>,
    pub k_neighbors: usize,
}

impl ProteinMpnn {
    pub fn load(w: &Weights, k_neighbors: usize, n_enc: usize, n_dec: usize) -> Self {
        ProteinMpnn {
            fw: FeatureWeights::load(w),
            w_e: (w.get("W_e.weight"), w.get("W_e.bias")),
            w_s: w.get("W_s.weight"),
            w_out: (w.get("W_out.weight"), w.get("W_out.bias")),
            encoder: (0..n_enc).map(|i| EncLayer::load(w, i)).collect(),
            decoder: (0..n_dec).map(|i| DecLayer::load(w, i)).collect(),
            k_neighbors,
        }
    }

    /// Shared prefix of `forward` and `sample`: features -> encoder stack.
    ///
    /// This depends only on the backbone, never on the sequence, so it is
    /// computed once per protein and reused for every sampled sequence and
    /// every scoring pass. (The reference implementation recomputes it inside
    /// both `sample()` and `forward()`, which is ~2/3 of its total work when
    /// designing many sequences for one structure.)
    pub fn encode(&self, b: &Batch) -> Encoded {
        let (g, h_v, h_e) = self.encode_inner(b);
        Encoded { g, h_v, h_e }
    }

    fn encode_inner(&self, b: &Batch) -> (Graph, Tensor, Tensor) {
        let g = protein_features(
            &self.fw,
            &b.x,
            &b.mask,
            &b.residue_idx,
            &b.chain_encoding,
            self.k_neighbors,
        );
        let (l, k) = (g.l, g.k);
        let mut h_v = Tensor::zeros(&[l, g.e.last()]);
        let mut h_e = ops::linear(&g.e, &self.w_e.0, Some(&self.w_e.1));

        // mask_attend[i,t] = mask[i] * mask[E_idx[i,t]]
        let mut mask_attend = vec![0.0f32; l * k];
        for i in 0..l {
            for t in 0..k {
                let j = g.e_idx[i * k + t] as usize;
                mask_attend[i * k + t] = b.mask[i] * b.mask[j];
            }
        }
        for layer in &self.encoder {
            let (v, e) = layer.forward(&h_v, &h_e, &g.e_idx, k, &b.mask, &mask_attend);
            h_v = v;
            h_e = e;
        }
        (g, h_v, h_e)
    }

    /// `order_mask_backward[q, p] = 1` iff position `q` is decoded *after* `p`,
    /// gathered along the neighbour axis -> `[L,K]`.
    ///
    /// Upstream expresses this as an einsum over one-hot permutation matrices;
    /// the rank comparison below is the same function in O(L*K).
    pub fn attend_mask(decoding_order: &[i64], e_idx: &[i64], l: usize, k: usize) -> Vec<f32> {
        let mut rank = vec![0usize; l];
        for (step, &pos) in decoding_order.iter().enumerate() {
            rank[pos as usize] = step;
        }
        let mut out = vec![0.0f32; l * k];
        for i in 0..l {
            for t in 0..k {
                let j = e_idx[i * k + t] as usize;
                out[i * k + t] = if rank[i] > rank[j] { 1.0 } else { 0.0 };
            }
        }
        out
    }

    /// Decoding order from `argsort((chain_mask + 1e-4) * |randn|)`.
    pub fn decoding_order(randn: &[f32], chain_mask: &[f32]) -> Vec<i64> {
        let l = randn.len();
        let keys: Vec<f32> = (0..l).map(|i| (chain_mask[i] + 0.0001) * randn[i].abs()).collect();
        let mut order: Vec<i64> = (0..l as i64).collect();
        order.sort_by(|&a, &b| {
            keys[a as usize]
                .partial_cmp(&keys[b as usize])
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.cmp(&b))
        });
        order
    }

    /// Teacher-forced pass: log-probabilities of every residue given the full
    /// backbone and the (already known) sequence `s`, under `decoding_order`.
    /// This is what produces the reported score.
    pub fn forward(&self, b: &Batch, s: &[i64], decoding_order: &[i64]) -> Tensor {
        self.forward_with(&self.encode(b), b, s, decoding_order)
    }

    /// `forward` against a pre-computed encoder state.
    pub fn forward_with(
        &self,
        enc: &Encoded,
        b: &Batch,
        s: &[i64],
        decoding_order: &[i64],
    ) -> Tensor {
        let (g, h_v, h_e) = (&enc.g, enc.h_v.clone(), &enc.h_e);
        let (l, k) = (g.l, g.k);

        let h_s = ops::embedding(s, &self.w_s, &[l, self.w_s.shape[1]]);
        let h_es = cat_neighbors_nodes(&h_s, h_e, &g.e_idx, k);
        let zeros = Tensor::zeros(&h_s.shape);
        let h_ex_encoder = cat_neighbors_nodes(&zeros, h_e, &g.e_idx, k);
        let h_exv_encoder = cat_neighbors_nodes(&h_v, &h_ex_encoder, &g.e_idx, k);

        let mask_attend = Self::attend_mask(decoding_order, &g.e_idx, l, k);
        let w = h_exv_encoder.last();
        let mut h_exv_encoder_fw = h_exv_encoder;
        for i in 0..l {
            for t in 0..k {
                // mask_fw = mask[i] * (1 - attend)
                let f = b.mask[i] * (1.0 - mask_attend[i * k + t]);
                for v in h_exv_encoder_fw.data[(i * k + t) * w..(i * k + t) * w + w].iter_mut() {
                    *v *= f;
                }
            }
        }

        let mut h_v = h_v;
        for layer in &self.decoder {
            let h_esv = cat_neighbors_nodes(&h_v, &h_es, &g.e_idx, k);
            let mut h_esv = h_esv;
            for i in 0..l {
                for t in 0..k {
                    let bw = b.mask[i] * mask_attend[i * k + t];
                    let base = (i * k + t) * w;
                    for c in 0..w {
                        h_esv.data[base + c] =
                            h_esv.data[base + c] * bw + h_exv_encoder_fw.data[base + c];
                    }
                }
            }
            h_v = layer.forward(&h_v, &h_esv, &b.mask);
        }

        let logits = ops::linear(&h_v, &self.w_out.0, Some(&self.w_out.1));
        ops::log_softmax_last(&logits)
    }

    /// Autoregressive sampling. Returns the designed sequence and the
    /// per-position probability rows the reference reports.
    ///
    /// `omit` is a 21-vector of 1.0 for amino acids to forbid (the CLI defaults
    /// to forbidding X, like `protein_mpnn_run.py`).
    #[allow(clippy::too_many_arguments)]
    pub fn sample(
        &self,
        b: &Batch,
        gen: &mut Mt19937,
        decoding_order: &[i64],
        temperature: f64,
        omit: &[f64; 21],
        bias: &[f64; 21],
    ) -> SampleOut {
        self.sample_with(&self.encode(b), b, gen, decoding_order, temperature, omit, bias)
    }

    /// `sample` against a pre-computed encoder state.
    #[allow(clippy::too_many_arguments)]
    pub fn sample_with(
        &self,
        enc: &Encoded,
        b: &Batch,
        gen: &mut Mt19937,
        decoding_order: &[i64],
        temperature: f64,
        omit: &[f64; 21],
        bias: &[f64; 21],
    ) -> SampleOut {
        let (g, h_v0, h_e) = (&enc.g, enc.h_v.clone(), &enc.h_e);
        let (l, k) = (g.l, g.k);
        let hidden = h_v0.last();
        let chain_mask = b.design_mask();

        let mask_attend = Self::attend_mask(decoding_order, &g.e_idx, l, k);

        let zeros = Tensor::zeros(&[l, hidden]);
        let h_ex_encoder = cat_neighbors_nodes(&zeros, h_e, &g.e_idx, k);
        let h_exv_encoder = cat_neighbors_nodes(&h_v0, &h_ex_encoder, &g.e_idx, k);
        let w = h_exv_encoder.last();
        let mut h_exv_encoder_fw = h_exv_encoder;
        for i in 0..l {
            for t in 0..k {
                let f = b.mask[i] * (1.0 - mask_attend[i * k + t]);
                for v in h_exv_encoder_fw.data[(i * k + t) * w..(i * k + t) * w + w].iter_mut() {
                    *v *= f;
                }
            }
        }

        let n_dec = self.decoder.len();
        // h_V_stack[0] is the encoder output; each decoder layer writes into the
        // next slot, one position at a time, as decoding proceeds.
        let mut stack: Vec<Tensor> = std::iter::once(h_v0)
            .chain((0..n_dec).map(|_| Tensor::zeros(&[l, hidden])))
            .collect();
        let mut h_s = Tensor::zeros(&[l, hidden]);
        let mut s_out = vec![0i64; l];
        let mut probs_out = vec![0.0f32; l * 21];

        for &t_pos in decoding_order.iter() {
            let i = t_pos as usize;
            if b.mask[i] == 0.0 {
                // Padded or missing region: copy the native residue through.
                let s_t = b.s[i];
                s_out[i] = s_t;
                let row = ops::embedding(&[s_t], &self.w_s, &[1, hidden]);
                h_s.data[i * hidden..i * hidden + hidden].copy_from_slice(&row.data);
                continue;
            }

            let e_idx_t = &g.e_idx[i * k..i * k + k];
            // This position's slice of the edge embeddings, [K, C].
            let ce = h_e.last();
            let h_e_t = Tensor::new(
                h_e.data[i * k * ce..(i + 1) * k * ce].to_vec(),
                vec![k, ce],
            );
            // h_ES_t = cat([h_E_t, h_S[E_idx_t]], -1)
            let h_es_t = Tensor::cat_last(&[&h_e_t, &gather_nodes_at(&h_s, e_idx_t)]);
            let h_exv_encoder_t = Tensor::new(
                h_exv_encoder_fw.data[i * k * w..(i + 1) * k * w].to_vec(),
                vec![k, w],
            );

            for l_i in 0..n_dec {
                let g_v = gather_nodes_at(&stack[l_i], e_idx_t);
                let h_esv_dec_t = Tensor::cat_last(&[&h_es_t, &g_v]);
                let h_v_t = Tensor::new(
                    stack[l_i].data[i * hidden..i * hidden + hidden].to_vec(),
                    vec![1, hidden],
                );
                let mut h_esv_t = h_esv_dec_t;
                for t in 0..k {
                    let bw = b.mask[i] * mask_attend[i * k + t];
                    for c in 0..w {
                        h_esv_t.data[t * w + c] =
                            h_esv_t.data[t * w + c] * bw + h_exv_encoder_t.data[t * w + c];
                    }
                }
                let h_esv_t = h_esv_t.reshape(&[1, k, w]);
                let out = self.decoder[l_i].forward(&h_v_t, &h_esv_t, &[b.mask[i]]);
                stack[l_i + 1].data[i * hidden..i * hidden + hidden].copy_from_slice(&out.data);
            }

            let h_v_t = Tensor::new(
                stack[n_dec].data[i * hidden..i * hidden + hidden].to_vec(),
                vec![1, hidden],
            );
            let logits = ops::linear(&h_v_t, &self.w_out.0, Some(&self.w_out.1));

            // The reference performs this step in float64: `bias_AAs_np` is a
            // float64 numpy array, which promotes the whole expression (and
            // therefore the softmax *and* the multinomial draw) to double.
            let mut z = [0.0f64; 21];
            for a in 0..21 {
                let scaled = (logits.data[a] as f64 / temperature) as f32;
                let f32_part = scaled - (omit[a] as f32) * 1e8;
                z[a] = f32_part as f64 + bias[a] / temperature;
            }
            let probs = softmax_f64(&z);
            let pick = multinomial_f64(gen, &probs);

            // Fixed positions keep the native residue.
            let cm = chain_mask[i];
            let s_t = if cm > 0.0 { pick as i64 } else { b.s[i] };
            s_out[i] = s_t;
            for a in 0..21 {
                probs_out[i * 21 + a] = (cm as f64 * probs[a]) as f32;
            }

            let row = ops::embedding(&[s_t], &self.w_s, &[1, hidden]);
            h_s.data[i * hidden..i * hidden + hidden].copy_from_slice(&row.data);
        }

        SampleOut { s: s_out, probs: probs_out }
    }
}

pub struct SampleOut {
    pub s: Vec<i64>,
    pub probs: Vec<f32>,
}

/// Backbone-only encoder state: the kNN graph plus the encoder's final node and
/// edge embeddings. Reusable across every sequence designed for one structure.
pub struct Encoded {
    pub g: Graph,
    pub h_v: Tensor,
    pub h_e: Tensor,
}

/// Number of 32-bit RNG draws `protein_mpnn_run.py` consumes between
/// `torch.manual_seed(seed)` and its first `torch.randn`.
///
/// This is not incidental: the script seeds the global generator, *then*
/// constructs `ProteinMPNN`, and construction initialises every parameter —
/// `nn.Linear.reset_parameters` (kaiming_uniform on the weight + uniform on the
/// bias), `nn.Embedding.reset_parameters` (normal_), and finally the explicit
/// `xavier_uniform_` loop in `ProteinMPNN.__init__` over every parameter with
/// `dim() > 1`. Only then are the loaded weights copied in, discarding all of
/// it. So the numbers are thrown away but the generator has moved.
///
/// Reproducing `--seed N` therefore means advancing the stream by the same
/// amount. Since MT19937 is a single stream, only the *total* matters, not the
/// order in which the modules drew. For v_48_020 this comes to 3,305,317.
///
/// Derived from the checkpoint's own tensor list so it stays correct across the
/// vanilla / soluble / CA model variants rather than being a magic constant.
pub fn torch_init_draws(w: &Weights) -> u64 {
    let mut linear_w = 0u64; // kaiming_uniform_ over Linear weights
    let mut linear_b = 0u64; // uniform_ over Linear biases
    let mut embedding = 0u64; // normal_ over nn.Embedding weights
    let mut xavier = 0u64; // the explicit xavier_uniform_ loop

    for name in w.names() {
        let shape = w.shape(&name).unwrap();
        let numel: u64 = shape.iter().product::<usize>() as u64;
        if shape.len() > 1 {
            xavier += numel;
            if name == "W_s.weight" {
                // nn.Embedding: normal_ takes the `normal_fill` path, which
                // redraws the last 16 values when numel is not a multiple of 16.
                embedding += numel + if numel % 16 != 0 { 16 } else { 0 };
            } else {
                linear_w += numel;
            }
        } else if !name.contains("norm") {
            // LayerNorm's weight/bias are filled with ones/zeros: no RNG.
            linear_b += numel;
        }
    }
    linear_w + linear_b + embedding + xavier
}

/// Gather the rows of `nodes[L,C]` named by `idx` -> `[K,C]`.
fn gather_nodes_at(nodes: &Tensor, idx: &[i64]) -> Tensor {
    let c = nodes.shape[1];
    let mut out = vec![0.0f32; idx.len() * c];
    for (t, &j) in idx.iter().enumerate() {
        let j = j as usize;
        out[t * c..t * c + c].copy_from_slice(&nodes.data[j * c..j * c + c]);
    }
    Tensor::new(out, vec![idx.len(), c])
}

fn softmax_f64(z: &[f64; 21]) -> [f64; 21] {
    let mut m = f64::NEG_INFINITY;
    for &v in z.iter() {
        if v > m {
            m = v;
        }
    }
    let mut out = [0.0f64; 21];
    let mut s = 0.0f64;
    for a in 0..21 {
        let e = (z[a] - m).exp();
        out[a] = e;
        s += e;
    }
    for v in out.iter_mut() {
        *v /= s;
    }
    out
}

/// `torch.multinomial(probs, 1)` on a float64 tensor: `argmax(probs / Exp(1))`
/// with the exponentials drawn (and kept) in double precision.
fn multinomial_f64(gen: &mut Mt19937, probs: &[f64; 21]) -> usize {
    let mut best = 0usize;
    let mut best_v = f64::NEG_INFINITY;
    for a in 0..21 {
        let u = gen.uniform_f64();
        let q = -(-u).ln_1p();
        let v = probs[a] / q;
        if v > best_v {
            best_v = v;
            best = a;
        }
    }
    best
}

/// `_scores`: mean negative log-likelihood over masked positions.
pub fn score(s: &[i64], log_probs: &Tensor, mask: &[f32]) -> f32 {
    let c = log_probs.last();
    let mut num = 0.0f32;
    let mut den = 0.0f32;
    for i in 0..s.len() {
        let lp = log_probs.data[i * c + s[i] as usize];
        num += -lp * mask[i];
        den += mask[i];
    }
    num / den
}
