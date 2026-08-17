//! ESMFold2 "parcae" linear-recurrent trunk loop + readout/coda + distogram head.
//!
//! Per loop: refined_lm = lm_encoder(lm_z_i); z_inject = z_init + refined_lm;
//! injected = parcae_input_norm(z_inject); z = a*z + linear(injected, b);
//! z = folding_trunk(z). After loops: z = parcae_readout(z); z = parcae_coda(z).
//! lm_z_i (post per-loop dropout) is injected from the reference for validation.

use crate::config::*;
use crate::ops::*;
use crate::tensor::Tensor;
use crate::trunk;
use crate::weights::Weights;

const LN_EPS: f32 = 1e-5;

#[inline]
fn softplus(x: f32) -> f32 {
    if x > 20.0 { x } else { (1.0 + x.exp()).ln() }
}

/// Discretized SSM dynamics: returns (a[256], b[256,256]).
/// delta = softplus(log_delta); a = exp(-delta*exp(log_a)); b[o,i] = delta[o]*b_cont[o,i].
pub fn discretized_dynamics(w: &Weights) -> (Vec<f32>, Tensor) {
    let log_a = w.get_vec("parcae_log_a");
    let log_delta = w.get_vec("parcae_log_delta");
    let b_cont = w.get("parcae_b_cont"); // [256,256]
    let c = log_a.len();
    let mut a = vec![0.0f32; c];
    let mut delta = vec![0.0f32; c];
    for i in 0..c {
        delta[i] = softplus(log_delta[i]);
        a[i] = (-delta[i] * log_a[i].exp()).exp();
    }
    let mut b = b_cont.data.clone();
    for o in 0..c {
        for i in 0..c { b[o * c + i] *= delta[o]; }
    }
    (a, Tensor::new(b, vec![c, c]))
}

/// z_init[i,j,c] = zinit1[i,c] + zinit2[j,c] + relpos[i,j,c] + token_bonds[i,j,c].
pub fn build_z_init(zinit1: &Tensor, zinit2: &Tensor, relpos: &Tensor, token_bonds: &Tensor) -> Tensor {
    let l = zinit1.shape[0];
    let c = zinit1.shape[1];
    let mut z = vec![0.0f32; l * l * c];
    for i in 0..l {
        for j in 0..l {
            let base = (i * l + j) * c;
            for d in 0..c {
                z[base + d] = zinit1.data[i * c + d] + zinit2.data[j * c + d]
                    + relpos.data[base + d] + token_bonds.data[base + d];
            }
        }
    }
    Tensor::new(z, vec![l, l, c])
}

/// Run the parcae loop. `z_inject_base` is the constant per-loop injection base
/// (= msa_encoder(z_init, ...), which is constant across loops for a single-row
/// MSA). `z_rand` is the trunc_normal pair init; `lm_z_loops` are the per-loop
/// post-dropout LM pair tensors. Returns the final pair `z` (post parcae_coda).
pub fn run_loop(
    w: &Weights,
    z_inject_base: &Tensor,
    z_rand: &Tensor,
    lm_z_loops: &[Tensor],
    mask: &[f32],
) -> Tensor {
    run_loop_cb(w, z_inject_base, z_rand, lm_z_loops, mask, &mut |_, _, _| {})
}

/// As [`run_loop`], invoking `prog(loop_index, block, n_loops)` after each
/// folding-trunk block within each loop (both counted 1-based). Numerically
/// identical to `run_loop`; the callback only observes loop/block indices.
pub fn run_loop_cb(
    w: &Weights,
    z_inject_base: &Tensor,
    z_rand: &Tensor,
    lm_z_loops: &[Tensor],
    mask: &[f32],
    prog: &mut dyn FnMut(usize, usize, usize),
) -> Tensor {
    let (a, b) = discretized_dynamics(w);
    let n_loops = lm_z_loops.len();
    let l = z_inject_base.shape[0];
    let c = z_inject_base.shape[2];
    let pin_w = w.get_vec("parcae_input_norm.weight");
    let pin_b = w.get_vec("parcae_input_norm.bias");

    let mut z = z_rand.clone();
    for (li, lm_z_i) in lm_z_loops.iter().enumerate() {
        let refined = trunk::folding_trunk(w, "lm_encoder", lm_z_i, mask, LM_ENCODER_LAYERS);
        // z_inject = msa_pair + refined_lm
        let mut zinj = z_inject_base.data.clone();
        for (a_, b_) in zinj.iter_mut().zip(&refined.data) { *a_ += *b_; }
        let zinj = Tensor::new(zinj, vec![l, l, c]);
        let injected = layernorm(&zinj, &pin_w, Some(&pin_b), LN_EPS);
        // z = a*z + linear(injected, b)
        let upd = linear_f64(&injected, &b, None); // out[o]=sum_d injected[d]*b[o,d]
        for i in 0..l {
            for j in 0..l {
                let base = (i * l + j) * c;
                for d in 0..c {
                    z.data[base + d] = a[d] * z.data[base + d] + upd.data[base + d];
                }
            }
        }
        z = trunk::folding_trunk_cb(w, "folding_trunk", &z, mask, FOLDING_TRUNK_LAYERS, &mut |blk| {
            prog(li + 1, blk, n_loops)
        });
    }
    // readout (Linear no bias) then coda (FoldingTrunk 2)
    z = linear_f64(&z, &w.get("parcae_readout.weight"), None);
    trunk::folding_trunk(w, "parcae_coda", &z, mask, PARCAE_CODA_LAYERS)
}

/// distogram_head(z + z^T_over_LL): Linear(d_pair -> bins) with bias.
pub fn distogram(w: &Weights, z: &Tensor) -> Tensor {
    let l = z.shape[0];
    let c = z.shape[2];
    let mut sym = vec![0.0f32; l * l * c];
    for i in 0..l {
        for j in 0..l {
            let b1 = (i * l + j) * c;
            let b2 = (j * l + i) * c;
            for d in 0..c { sym[b1 + d] = z.data[b1 + d] + z.data[b2 + d]; }
        }
    }
    let sym = Tensor::new(sym, vec![l, l, c]);
    let bias = w.get("distogram_head.bias");
    linear_f64(&sym, &w.get("distogram_head.weight"), Some(&bias))
}
