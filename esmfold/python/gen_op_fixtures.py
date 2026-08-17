"""Per-op fixtures for P1, using the ACTUAL reference functions where possible
so formulas (erf-GELU, rotary, etc.) match exactly."""
import math
import os

import torch
import torch.nn.functional as F

from common import FIX, Manifest, save_fixture
from transformers.models.esm.modeling_esm import RotaryEmbedding, gelu, rotate_half

g = torch.Generator().manual_seed(1234)


def rnd(*shape, scale=1.0):
    return (torch.randn(*shape, generator=g) * scale).float()


def main():
    man = Manifest(os.path.join(FIX, "ops", "manifest.json"))

    # 1. linear (small K and large K)
    for tag, (M, K, O) in {"small": (7, 16, 5), "bigK": (4, 2560, 8)}.items():
        x = rnd(M, K)
        w = rnd(O, K, scale=0.05)
        b = rnd(O)
        y = F.linear(x, w, b)
        save_fixture("ops", f"linear_{tag}", {"x": x, "w": w, "b": b, "y": y})
        man.add(op="linear", name=f"linear_{tag}", shape=list(y.shape))

    # 2. matmul2d
    a = rnd(6, 2560)
    bm = rnd(2560, 9)
    save_fixture("ops", "matmul_bigK", {"a": a, "b": bm, "y": a @ bm})
    man.add(op="matmul2d", name="matmul_bigK", shape=[6, 9])

    # 3. layer_norm (eps 1e-5)
    x = rnd(7, 2560, scale=3.0)
    w = rnd(2560, scale=0.5) + 1.0
    b = rnd(2560, scale=0.1)
    y = F.layer_norm(x, (2560,), w, b, eps=1e-5)
    save_fixture("ops", "layernorm", {"x": x, "w": w, "b": b, "y": y})
    man.add(op="layer_norm", name="layernorm", shape=[7, 2560], eps=1e-5)

    # 4. erf-GELU (esm's explicit gelu)
    x = rnd(1000, scale=4.0)
    save_fixture("ops", "gelu_erf", {"x": x, "y": gelu(x)})
    man.add(op="gelu_erf", name="gelu_erf", shape=[1000])

    # 5. sigmoid, softplus, relu
    x = rnd(1000, scale=6.0)
    save_fixture("ops", "sigmoid", {"x": x, "y": torch.sigmoid(x)})
    man.add(op="sigmoid", name="sigmoid", shape=[1000])
    save_fixture("ops", "softplus", {"x": x, "y": F.softplus(x)})
    man.add(op="softplus", name="softplus", shape=[1000])
    save_fixture("ops", "relu", {"x": x, "y": F.relu(x)})
    man.add(op="relu", name="relu", shape=[1000])

    # 6. softmax over last dim
    x = rnd(7, 131, scale=5.0)
    save_fixture("ops", "softmax_last", {"x": x, "y": F.softmax(x, dim=-1)})
    man.add(op="softmax_last", name="softmax_last", shape=[7, 131])

    # 7. rotary: inv_freq + cos/sin tables + applied q/k
    dim, L, H = 64, 20, 2
    rot = RotaryEmbedding(dim)
    q = rnd(1, H, L, dim)
    k = rnd(1, H, L, dim)
    # RotaryEmbedding.forward(q,k) returns rotated (q,k); also expose tables
    qr, kr = rot(q, k)
    # rebuild cos/sin the way the module caches them (seq_len = L)
    t = torch.arange(L).type_as(rot.inv_freq)
    freqs = torch.outer(t, rot.inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos()
    sin = emb.sin()
    save_fixture(
        "ops",
        "rotary",
        {
            "inv_freq": rot.inv_freq,
            "cos": cos,
            "sin": sin,
            "q": q[0],
            "k": k[0],
            "q_rot": qr[0],
            "k_rot": kr[0],
        },
    )
    man.add(op="rotary", name="rotary", dim=dim, L=L, H=H)

    # also a standalone rotate_half check
    x = rnd(3, dim)
    save_fixture("ops", "rotate_half", {"x": x, "y": rotate_half(x)})
    man.add(op="rotate_half", name="rotate_half", shape=[3, dim])

    man.write()


if __name__ == "__main__":
    main()
