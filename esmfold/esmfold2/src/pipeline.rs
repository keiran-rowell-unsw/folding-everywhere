//! Full-fold helpers that chain ESM-C into the ESMFold2 head.

use crate::config::ESMC_D_MODEL;
use crate::esmc;
use crate::tensor::Tensor;
use crate::weights::Weights;

/// Run ESM-C over [BOS, res_ids..., EOS] (single chain) and scatter the 81
/// collected hidden states back to per-residue layout -> [L, 81, 2560].
pub fn compute_lm_hidden_states(w_esmc: &Weights, res_ids: &[i64]) -> Tensor {
    compute_lm_hidden_states_cb(w_esmc, res_ids, &mut |_| {})
}

/// As [`compute_lm_hidden_states`], forwarding `prog(layer)` from the ESM-C
/// transformer (layer counted 1..=ESMC_N_LAYERS) so callers can show progress.
pub fn compute_lm_hidden_states_cb(
    w_esmc: &Weights,
    res_ids: &[i64],
    prog: &mut dyn FnMut(usize),
) -> Tensor {
    let l = res_ids.len();
    let d = ESMC_D_MODEL;
    // [BOS=0] ids [EOS=2]
    let mut ids = Vec::with_capacity(l + 2);
    ids.push(0i64);
    ids.extend_from_slice(res_ids);
    ids.push(2i64);
    let t = ids.len();
    let seq_id = vec![0i64; t]; // single chain
    let states = esmc::forward_cb(w_esmc, &ids, &seq_id, true, prog); // 81 × [T, 2560]
    let nl = states.len();
    let mut out = vec![0.0f32; l * nl * d];
    for layer in 0..nl {
        let s = &states[layer].data;
        for li in 0..l {
            // protein token li -> LM position li+1 (after the leading BOS)
            let src = (li + 1) * d;
            let dst = (li * nl + layer) * d;
            out[dst..dst + d].copy_from_slice(&s[src..src + d]);
        }
    }
    Tensor::new(out, vec![l, nl, d])
}
