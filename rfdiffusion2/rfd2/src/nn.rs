//! Parameter plumbing and the three primitive layers (`Linear`, `LayerNorm`,
//! `Embedding`).
//!
//! Every layer here holds its tensors by value, pulled once out of the
//! checkpoint by dotted name. The name is the *reference's own* module path
//! (`model.simulator.main_block.0.msa2msa.row_attn.to_q.weight`), so a typo is a
//! missing-key panic at load time rather than a silent zero — the checkpoint has
//! 7 208 tensors and several near-identical name families, which is precisely
//! where a port loses a layer without noticing.
//!
//! Arithmetic follows the pinned convention: `F.linear` and `F.layer_norm` are
//! patched in the reference (`python/pinned.py`), so both sides accumulate in
//! f64 and round to f32 exactly once.

use crate::ops::{layer_norm_f64, linear_f64};
use crate::tensor::Tensor;
use crate::weights::Weights;

/// A cursor into the checkpoint's dotted namespace.
#[derive(Clone)]
pub struct Params<'a> {
    pub w: &'a Weights,
    pub prefix: String,
}

impl<'a> Params<'a> {
    pub fn root(w: &'a Weights, prefix: &str) -> Self {
        Params { w, prefix: prefix.to_string() }
    }

    pub fn sub(&self, name: &str) -> Params<'a> {
        Params {
            w: self.w,
            prefix: if self.prefix.is_empty() {
                name.to_string()
            } else {
                format!("{}.{}", self.prefix, name)
            },
        }
    }

    /// `sub` for a numeric index, e.g. `main_block.7`.
    pub fn idx(&self, i: usize) -> Params<'a> {
        self.sub(&i.to_string())
    }

    pub fn name(&self, leaf: &str) -> String {
        if self.prefix.is_empty() {
            leaf.to_string()
        } else {
            format!("{}.{}", self.prefix, leaf)
        }
    }

    pub fn get(&self, leaf: &str) -> Tensor {
        self.w.get(&self.name(leaf))
    }

    pub fn has(&self, leaf: &str) -> bool {
        self.w.has(&self.name(leaf))
    }
}

// ---------------------------------------------------------------------------

pub struct Linear {
    pub weight: Tensor, // [out, in]
    pub bias: Option<Tensor>,
    /// The same weight and bias widened to f64 once at load time — see
    /// `ops::WeightsF64` for why that is worth the memory.
    pre: crate::ops::WeightsF64,
}

fn mk(weight: Tensor, bias: Option<Tensor>) -> Linear {
    let pre = crate::ops::WeightsF64::new(&weight, bias.as_ref());
    Linear { weight, bias, pre }
}

impl Linear {
    pub fn load(p: &Params) -> Self {
        mk(p.get("weight"), if p.has("bias") { Some(p.get("bias")) } else { None })
    }

    /// Load a layer that is known to be bias-free (`nn.Linear(..., bias=False)`).
    /// Asserting rather than tolerating a missing bias keeps a renamed layer from
    /// silently degrading into "no bias".
    pub fn load_nobias(p: &Params) -> Self {
        assert!(!p.has("bias"), "{} unexpectedly has a bias", p.prefix);
        mk(p.get("weight"), None)
    }

    pub fn out_dim(&self) -> usize {
        self.weight.shape[0]
    }

    pub fn in_dim(&self) -> usize {
        self.weight.shape[1]
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        crate::ops::linear_pre(x, &self.pre)
    }
}

// ---------------------------------------------------------------------------

pub struct LayerNorm {
    pub weight: Tensor,
    pub bias: Option<Tensor>,
    pub eps: f64,
}

impl LayerNorm {
    /// `nn.LayerNorm` default eps is 1e-5, and the reference never overrides it.
    pub const DEFAULT_EPS: f64 = 1e-5;

    pub fn load(p: &Params) -> Self {
        LayerNorm {
            weight: p.get("weight"),
            bias: if p.has("bias") { Some(p.get("bias")) } else { None },
            eps: Self::DEFAULT_EPS,
        }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        match &self.bias {
            Some(b) => layer_norm_f64(x, &self.weight, b, self.eps),
            None => {
                let zeros = Tensor::zeros(&[self.weight.numel()]);
                layer_norm_f64(x, &self.weight, &zeros, self.eps)
            }
        }
    }
}

// ---------------------------------------------------------------------------

pub struct Embedding {
    pub weight: Tensor, // [num_embeddings, dim]
}

impl Embedding {
    pub fn load(p: &Params) -> Self {
        Embedding { weight: p.get("weight") }
    }

    pub fn dim(&self) -> usize {
        self.weight.shape[1]
    }

    /// Gather rows by index; `out_shape` must end in `dim`.
    pub fn forward(&self, ids: &[i64], out_shape: &[usize]) -> Tensor {
        crate::ops::embedding(ids, &self.weight, out_shape)
    }
}

// ---------------------------------------------------------------------------

/// Everything a forward pass needs beyond its own weights.
///
/// Right now that is just the torch RNG, and it is not optional: RFdiffusion2
/// runs its network **in training mode at inference** (measured: 5 758/5 758
/// modules with `training == True`), so dropout fires inside the forward pass
/// and consumes the generator. Threading the RNG explicitly makes it impossible
/// to add a layer that silently forgets to draw — which would not just change
/// that layer, it would shift every later draw in the stream, including
/// `psi_pred`.
pub struct Ctx {
    pub rng: crate::rng::torch::Mt19937,
}

impl Ctx {
    pub fn new(rng: crate::rng::torch::Mt19937) -> Self {
        Ctx { rng }
    }
}

/// `Attention_module.FeedForwardLayer`: LayerNorm -> Linear -> ReLU ->
/// **Dropout** -> Linear.
///
/// The dropout is `nn.Dropout`, i.e. the MKL-seeded path in `crate::dropout`,
/// and it costs exactly one `u32` from the torch stream per call regardless of
/// tensor size.
pub struct FeedForward {
    pub norm: LayerNorm,
    pub linear1: Linear,
    pub linear2: Linear,
    pub p_drop: f64,
}

impl FeedForward {
    pub fn load(p: &Params, p_drop: f64) -> Self {
        FeedForward {
            norm: LayerNorm::load(&p.sub("norm")),
            linear1: Linear::load(&p.sub("linear1")),
            linear2: Linear::load(&p.sub("linear2")),
            p_drop,
        }
    }

    pub fn forward(&self, x: &Tensor, ctx: &mut Ctx) -> Tensor {
        let h = self.norm.forward(x);
        let mut h = self.linear1.forward(&h);
        crate::ops::relu_(&mut h);
        let h = crate::dropout::nn_dropout(&mut ctx.rng, &h, self.p_drop);
        self.linear2.forward(&h)
    }
}
