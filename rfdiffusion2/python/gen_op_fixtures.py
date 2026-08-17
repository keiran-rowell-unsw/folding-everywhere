#!/usr/bin/env python3
"""SOP §2 / rung 1 — per-op fixtures **at the widths RFdiffusion2 really uses**.

The widths are not invented: they come from the checkpoint's own config
(`fixtures/weights/ckpt_conf.json`) and from the shapes in
`fixtures/weights/inventory_model_state_dict.json`:

    d_msa 256   d_msa_full 64   d_pair 192   d_templ 64   d_state 64
    d_hidden 32 d_hidden_templ 64   d_t1d 114
    n_head_msa 8   n_head_pair 6   n_head_templ 4
    SE3: l0 64, num_channels 32, num_edge_features 64, num_degrees 2

Op inventory (grep over rf2aa/model/ and the SE3 transformer):
    nn.Linear x124   nn.LayerNorm x47   F.relu x13   F.softmax x8
    nn.Embedding x8  nn.ReLU x4         nn.Dropout x4  nn.ELU x3
No GELU anywhere in this model.

    .venv/bin/python python/gen_op_fixtures.py
"""
import common
import torch
import torch.nn.functional as F

# (in, out) pairs actually present in the checkpoint, plus the odd SE3 ones
LINEAR_WIDTHS = [
    (256, 256),   # msa track
    (192, 192),   # pair track
    (256, 32 * 8),   # msa attention qkv (d_hidden * n_head_msa)
    (192, 32 * 6),   # pair attention qkv (d_hidden * n_head_pair)
    (320, 64),    # str_refiner.embed_node  (d_msa + d_state)
    (80, 64),     # latent_emb.emb_state
    (114, 64),    # t1d projection
    (64, 64),     # state / templ
    (64, 32),     # SE3 radial_func first layer
    (32, 32),     # SE3 radial_func hidden
    (32, 576),    # SE3 radial_func out (l1)
    (32, 3328),   # SE3 radial_func out (large)
    (65, 32),     # SE3 radial_func in (num_edge_features + 1)
    (192, 37),    # c6d distance head
    (64, 1),      # scalar heads
]

NORM_WIDTHS = [256, 192, 128, 64, 32]
SOFTMAX_WIDTHS = [37, 64, 128, 150, 192, 256]
ROWS = [1, 7, 150]


def main():
    g = torch.Generator().manual_seed(20260809)
    out = {}

    # ---- linear -----------------------------------------------------------
    for (din, dout) in LINEAR_WIDTHS:
        for rows in ROWS:
            x = torch.randn(rows, din, generator=g)
            w = torch.randn(dout, din, generator=g) * (din ** -0.5)
            b = torch.randn(dout, generator=g)
            tag = f"lin_{din}x{dout}_r{rows}"
            out[f"{tag}_x"] = x
            out[f"{tag}_w"] = w
            out[f"{tag}_b"] = b
            out[f"{tag}_y"] = F.linear(x, w, b)
            out[f"{tag}_y_nobias"] = F.linear(x, w)
            # BIT-EXACT mode (docs/BITEXACT.md): accumulate the dot product and
            # the bias in f64, round to f32 exactly once. Measured to be
            # independent of summation order (python/probe_f64_pinning.py:
            # 299 200 values, 4 different orders, 0 disagreements), so Rust and
            # PyTorch agree bit-for-bit without matching MKL's blocking.
            out[f"{tag}_y_pinned"] = (
                x.double() @ w.double().T + b.double()).float()
            out[f"{tag}_y_pinned_nobias"] = (x.double() @ w.double().T).float()

    # ---- layernorm --------------------------------------------------------
    for c in NORM_WIDTHS:
        for rows in ROWS:
            x = torch.randn(rows, c, generator=g)
            w = torch.randn(c, generator=g)
            b = torch.randn(c, generator=g)
            tag = f"ln_{c}_r{rows}"
            out[f"{tag}_x"] = x
            out[f"{tag}_w"] = w
            out[f"{tag}_b"] = b
            out[f"{tag}_y"] = F.layer_norm(x, (c,), w, b, 1e-5)
            out[f"{tag}_y_pinned"] = F.layer_norm(
                x.double(), (c,), w.double(), b.double(), 1e-5).float()

    # ---- softmax ----------------------------------------------------------
    for c in SOFTMAX_WIDTHS:
        for rows in ROWS:
            x = torch.randn(rows, c, generator=g) * 3.0
            tag = f"sm_{c}_r{rows}"
            out[f"{tag}_x"] = x
            out[f"{tag}_y"] = F.softmax(x, dim=-1)
            out[f"{tag}_y_pinned"] = F.softmax(x.double(), dim=-1).float()

    # ---- activations ------------------------------------------------------
    # span the interesting regions: negatives (ELU's expm1 branch), zero,
    # positives, and magnitudes where expm1 vs exp-1 visibly differ
    x = torch.cat([
        torch.linspace(-8, 8, 4001),
        torch.tensor([0.0, -0.0, 1e-8, -1e-8, 1e-4, -1e-4, 30.0, -30.0]),
        torch.randn(2000, generator=g) * 5,
    ])
    out["act_x"] = x
    out["act_relu"] = F.relu(x)
    out["act_elu_1.0"] = F.elu(x, alpha=1.0)

    # ---- embedding --------------------------------------------------------
    # NAATOKENS-ish vocabularies used by the msa/seq embeddings
    for (vocab, dim) in [(80, 256), (83, 64), (164, 256)]:
        table = torch.randn(vocab, dim, generator=g)
        idx = torch.randint(0, vocab, (150,), generator=g)
        tag = f"emb_{vocab}x{dim}"
        out[f"{tag}_w"] = table
        out[f"{tag}_idx"] = idx
        out[f"{tag}_y"] = F.embedding(idx, table)

    common.write_fixture("ops", "ops", out, {"torch_version": torch.__version__})


if __name__ == "__main__":
    main()
