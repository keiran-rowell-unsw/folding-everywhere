//! ESM-2 3B backbone, producing the 37 hidden states ESMFold stacks.
//!
//! Pre-LN transformer, 36 layers, hidden 2560, 40 heads (head 64), erf-GELU FFN,
//! rotary on q/k (query pre-scaled by 64^-0.5 BEFORE rotary), token_dropout x0.88
//! at inference. The 37-stack is [emb, L1..L35, LN_after(L36)] (verified against HF).

use crate::ops;
use crate::tensor::Tensor;
use crate::weights::Weights;

const D: usize = 2560;
const N_LAYERS: usize = 36;
const N_HEADS: usize = 40;
const HEAD: usize = D / N_HEADS; // 64
const EPS: f32 = 1e-5;
const TOKEN_DROPOUT_SCALE: f32 = 0.88; // (1 - 0.12) / (1 - 0)

fn ln(x: &Tensor, w: &Weights, prefix: &str) -> Tensor {
    ops::layer_norm(x, &w.get(&format!("{prefix}.weight")), &w.get(&format!("{prefix}.bias")), EPS)
}

fn lin(x: &Tensor, w: &Weights, prefix: &str) -> Tensor {
    ops::linear(x, &w.get(&format!("{prefix}.weight")), Some(&w.get(&format!("{prefix}.bias"))))
}

/// Multi-head self-attention with rotary; `x` is [L, D]. Returns [L, D].
fn attention(x_ln: &Tensor, w: &Weights, layer: usize, cos: &[f32], sin: &[f32], l: usize) -> Tensor {
    let p = format!("esm.encoder.layer.{layer}.attention.self");
    let q = lin(x_ln, w, &format!("{p}.query")); // [L, D]
    let k = lin(x_ln, w, &format!("{p}.key"));
    let v = lin(x_ln, w, &format!("{p}.value"));

    // [L, D] -> [H, L, HEAD]
    let to_heads = |t: &Tensor| -> Vec<f32> {
        // t[L, H*HEAD] -> out[H, L, HEAD]
        let mut out = vec![0.0f32; N_HEADS * l * HEAD];
        for li in 0..l {
            for h in 0..N_HEADS {
                for d in 0..HEAD {
                    out[(h * l + li) * HEAD + d] = t.data[li * D + h * HEAD + d];
                }
            }
        }
        out
    };
    let mut qh = to_heads(&q);
    let mut kh = to_heads(&k);
    let vh = to_heads(&v);

    // query pre-scale by HEAD^-0.5 BEFORE rotary
    let scale = (HEAD as f32).powf(-0.5);
    for x in qh.iter_mut() {
        *x *= scale;
    }
    ops::apply_rotary_inplace(&mut qh, N_HEADS, l, HEAD, cos, sin);
    ops::apply_rotary_inplace(&mut kh, N_HEADS, l, HEAD, cos, sin);

    // per-head: scores = q @ k^T [L,L]; softmax; ctx = scores @ v [L,HEAD]
    let mut ctx = vec![0.0f32; l * D]; // [L, H*HEAD]
    for h in 0..N_HEADS {
        let qbh = &qh[h * l * HEAD..(h + 1) * l * HEAD];
        let kbh = &kh[h * l * HEAD..(h + 1) * l * HEAD];
        let vbh = &vh[h * l * HEAD..(h + 1) * l * HEAD];
        // scores[i,j] = sum_d q[i,d]*k[j,d]
        let mut scores = vec![0.0f32; l * l];
        for i in 0..l {
            for j in 0..l {
                let mut s = 0.0f32;
                for d in 0..HEAD {
                    s += qbh[i * HEAD + d] * kbh[j * HEAD + d];
                }
                scores[i * l + j] = s;
            }
        }
        let probs = ops::softmax_last(&Tensor::new(scores, vec![l, l]));
        // ctx[i,d] = sum_j probs[i,j]*v[j,d]
        for i in 0..l {
            for d in 0..HEAD {
                let mut s = 0.0f32;
                for j in 0..l {
                    s += probs.data[i * l + j] * vbh[j * HEAD + d];
                }
                ctx[i * D + h * HEAD + d] = s;
            }
        }
    }
    let ctx_t = Tensor::new(ctx, vec![l, D]);
    lin(&ctx_t, w, &format!("esm.encoder.layer.{layer}.attention.output.dense"))
}

fn add(a: &Tensor, b: &Tensor) -> Tensor {
    let data = a.data.iter().zip(&b.data).map(|(x, y)| x + y).collect();
    Tensor::new(data, a.shape.clone())
}

/// Run ESM-2 and return the 37 stacked hidden states, each [L, D] (L = ids.len()).
pub fn esm2_states(w: &Weights, ids: &[i64]) -> Vec<Tensor> {
    esm2_states_cb(w, ids, &mut |_| {})
}

/// As `esm2_states`, calling `prog(layer_index)` after each of the 36 layers.
pub fn esm2_states_cb(w: &Weights, ids: &[i64], prog: &mut dyn FnMut(usize)) -> Vec<Tensor> {
    let l = ids.len();
    let (cos, sin) = ops::build_cos_sin(l, HEAD);

    // embeddings: word lookup * token_dropout scale
    let we = w.get("esm.embeddings.word_embeddings.weight"); // [33, D]
    let mut emb = vec![0.0f32; l * D];
    for (i, &id) in ids.iter().enumerate() {
        let row = id as usize * D;
        for d in 0..D {
            emb[i * D + d] = we.data[row + d] * TOKEN_DROPOUT_SCALE;
        }
    }
    let mut x = Tensor::new(emb, vec![l, D]);

    let mut states: Vec<Tensor> = Vec::with_capacity(N_LAYERS + 1);
    states.push(x.clone()); // state_0 = embeddings

    for layer in 0..N_LAYERS {
        let lp = format!("esm.encoder.layer.{layer}");
        // attention sub-block (pre-LN)
        let x_ln = ln(&x, w, &format!("{lp}.attention.LayerNorm"));
        let attn = attention(&x_ln, w, layer, &cos, &sin, l);
        x = add(&x, &attn);
        // ffn sub-block (pre-LN)
        let y_ln = ln(&x, w, &format!("{lp}.LayerNorm"));
        let up = lin(&y_ln, w, &format!("{lp}.intermediate.dense"));
        let act = ops::gelu_erf(&up);
        let down = lin(&act, w, &format!("{lp}.output.dense"));
        x = add(&x, &down);

        if layer < N_LAYERS - 1 {
            states.push(x.clone()); // state_{layer+1}
        } else {
            // last layer: store LN_after(output) as state_36 (== last_hidden_state)
            let last = ln(&x, w, "esm.encoder.emb_layer_norm_after");
            states.push(last);
        }
        prog(layer + 1);
    }
    states
}
