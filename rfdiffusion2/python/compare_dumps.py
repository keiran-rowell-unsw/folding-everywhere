#!/usr/bin/env python3
"""Compare two ref_dump captures tensor by tensor.

Two uses:
  1. stock vs pinned  -- how much does the bit-exact convention move the model?
  2. pinned vs pinned -- is pinned mode reproducible run to run? (it must be,
     or the whole strategy is void)

    .venv/bin/python python/compare_dumps.py model_stock model_pinned
"""
import sys

import common
import numpy as np
import torch
from safetensors.torch import load_file


def load(sub):
    import os
    return load_file(os.path.join(common.FIXTURES, sub, "step0.safetensors"))


def main(a_name, b_name):
    a, b = load(a_name), load(b_name)
    keys = sorted(set(a) & set(b))
    print(f"{a_name} vs {b_name}: {len(keys)} common tensors "
          f"({len(set(a) ^ set(b))} unique)\n")
    print(f"{'tensor':52s} {'n':>9s} {'bitexact%':>10s} {'max|Δ|':>11s} {'cos':>12s}")

    n_identical = 0
    worst = []
    for k in keys:
        x, y = a[k], b[k]
        if x.shape != y.shape:
            print(f"{k:52s} SHAPE {list(x.shape)} vs {list(y.shape)}")
            continue
        xf = x.float().flatten()
        yf = y.float().flatten()
        finite = torch.isfinite(xf) & torch.isfinite(yf)
        if finite.sum() == 0:
            continue
        xf, yf = xf[finite], yf[finite]
        same = (xf.view(torch.int32) == yf.view(torch.int32)).float().mean().item()
        d = (xf - yf).abs().max().item()
        denom = (xf.norm() * yf.norm()).item()
        cos = (xf @ yf).item() / denom if denom > 0 else 1.0
        if same == 1.0:
            n_identical += 1
        worst.append((d, k, same, cos, xf.numel()))

    worst.sort(reverse=True)
    for d, k, same, cos, n in worst[:25]:
        print(f"{k:52s} {n:9d} {100*same:9.4f}% {d:11.3e} {cos:12.9f}")

    print(f"\n{n_identical}/{len(worst)} tensors bit-identical")
    if n_identical == len(worst):
        print("ALL TENSORS BIT-IDENTICAL")
    else:
        mx = max(w[0] for w in worst)
        print(f"max |Δ| over all tensors: {mx:.3e}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "model_stock",
         sys.argv[2] if len(sys.argv) > 2 else "model_pinned")
