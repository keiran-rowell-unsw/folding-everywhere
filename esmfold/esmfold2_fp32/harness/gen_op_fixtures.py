"""Generate op-level fixtures (inputs + torch fp32 outputs) as .npy files.

Ground truth is torch on CPU fp32. RoPE uses the fork's exact RotaryEmbedding.
"""
import os
import numpy as np
import torch
import torch.nn.functional as F

torch.manual_seed(0)
FIX = os.path.join(os.path.dirname(__file__), "..", "fixtures")
os.makedirs(FIX, exist_ok=True)

def save(name, arr):
    np.save(os.path.join(FIX, name + ".npy"), np.ascontiguousarray(arr))

def t(x): return x.detach().to(torch.float32).numpy()

# ---- linear (K not a multiple of 8 to exercise the tail) ----
M, K, O = 7, 2570, 19
x = torch.randn(M, K); w = torch.randn(O, K); b = torch.randn(O)
save("op_linear_x", t(x)); save("op_linear_w", t(w)); save("op_linear_b", t(b))
save("op_linear_y", t(F.linear(x, w)))
save("op_linear_yb", t(F.linear(x, w, b)))

# ---- layernorm (eps 1e-5, with and without bias) ----
N, C = 11, 2560
x = torch.randn(N, C); g = torch.randn(C); be = torch.randn(C)
save("op_ln_x", t(x)); save("op_ln_w", t(g)); save("op_ln_b", t(be))
save("op_ln_y", t(F.layer_norm(x, (C,), g, be, 1e-5)))
save("op_ln_y_nobias", t(F.layer_norm(x, (C,), g, None, 1e-5)))

# ---- rmsnorm (default eps = finfo(f32).eps, with and without weight) ----
x = torch.randn(N, C); g = torch.randn(C)
save("op_rms_x", t(x)); save("op_rms_w", t(g))
save("op_rms_y", t(F.rms_norm(x, (C,), g)))
save("op_rms_y_noweight", t(F.rms_norm(x, (C,), None)))

# ---- activations ----
x = torch.randn(257) * 3.0
save("op_act_x", t(x))
save("op_silu_y", t(F.silu(x)))
save("op_gelu_y", t(F.gelu(x)))            # approximate='none' (erf)
save("op_sigmoid_y", t(torch.sigmoid(x)))
# swiglu split: silu(x1)*x2 over last dim halves
xs = torch.randn(5, 512)
x1, x2 = xs.chunk(2, dim=-1)
save("op_swiglu_x", t(xs)); save("op_swiglu_y", t(F.silu(x1) * x2))
# softmax over last dim
xm = torch.randn(6, 41)
save("op_softmax_x", t(xm)); save("op_softmax_y", t(F.softmax(xm, dim=-1)))

# ---- RoPE (ESM-C RotaryEmbedding, head_dim=64, base 10000) ----
from transformers.models.esmc.modeling_esmc import RotaryEmbedding
T, H, D = 13, 40, 64
rot = RotaryEmbedding(D)   # base 10000, interleaved False, pos_idx_in_fp32
q = torch.randn(1, T, H, D); k = torch.randn(1, T, H, D)
qr, kr = rot(q, k)
save("op_rope_q", t(q)); save("op_rope_qr", t(qr)); save("op_rope_kr", t(kr))

print("op fixtures written to", os.path.abspath(FIX))
