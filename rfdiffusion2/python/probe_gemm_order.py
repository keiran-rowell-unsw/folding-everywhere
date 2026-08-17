#!/usr/bin/env python3
"""Can a reproducible reduction order match PyTorch's fp32 linear bit-for-bit?

This is the load-bearing experiment for an end-to-end bit-exact port. The SOP
§0 says fp32 GEMM accumulation order is MKL's business and not reproducible --
but that is a general claim, and RFdiffusion2's K dimensions are small
(32/64/114/192/256/320). Small K is exactly where a BLAS may not block at all.
So: measure, at the shapes the model really uses.

For each candidate order we compute y = x @ w.T in fp32 and compare BIT PATTERNS
against torch.nn.functional.linear.

    .venv/bin/python python/probe_gemm_order.py
"""
import common
import numpy as np
import torch
import torch.nn.functional as F

# (M, K, N) with M = rows, K = contraction, N = outputs -- the real shapes
SHAPES = [
    (1, 256, 256),
    (7, 256, 256),
    (150, 256, 256),
    (150, 192, 192),
    (150, 192, 192 * 2),
    (150, 320, 64),
    (150, 114, 64),
    (150, 64, 32),
    (150, 32, 576),
    (1, 64, 1),
    (150, 65, 32),
]


def bits(a):
    return a.view(np.uint32) if a.dtype == np.float32 else a


def frac_equal(a, b):
    return float((bits(np.ascontiguousarray(a)) == bits(np.ascontiguousarray(b))).mean())


def cand_sequential(prod):
    """acc = p0; acc += p1; ... strict left-to-right fp32."""
    acc = prod[..., 0].copy()
    for k in range(1, prod.shape[-1]):
        acc = acc + prod[..., k]
    return acc


def cand_pairwise(prod):
    """numpy's own pairwise summation."""
    return prod.sum(axis=-1, dtype=np.float32)


def cand_lanes(prod, L):
    """L independent accumulators (lane k%L), then lanes combined left-to-right.
    This is the shape of every SIMD dot product: L = 8 for AVX2, 16 for AVX-512."""
    K = prod.shape[-1]
    if K % L:
        return None
    p = prod.reshape(prod.shape[:-1] + (K // L, L))
    acc = p[..., 0, :].copy()
    for k in range(1, K // L):
        acc = acc + p[..., k, :]
    out = acc[..., 0].copy()
    for l in range(1, L):
        out = out + acc[..., l]
    return out


def cand_lanes_tree(prod, L):
    """L accumulators, combined by a balanced tree instead of left-to-right."""
    K = prod.shape[-1]
    if K % L:
        return None
    p = prod.reshape(prod.shape[:-1] + (K // L, L))
    acc = p[..., 0, :].copy()
    for k in range(1, K // L):
        acc = acc + p[..., k, :]
    cur = [acc[..., l].copy() for l in range(L)]
    while len(cur) > 1:
        cur = [cur[i] + cur[i + 1] for i in range(0, len(cur), 2)]
    return cur[0]


def cand_f64(x, w):
    """f64 accumulation, rounded once at the end (correctly-rounded-ish)."""
    return (x.astype(np.float64) @ w.astype(np.float64).T).astype(np.float32)


def main():
    torch.set_num_threads(1)
    g = torch.Generator().manual_seed(11)
    print(f"torch {torch.__version__}  numpy {np.__version__}\n")
    print(f"{'shape (M,K,N)':>20s} " + " ".join(
        f"{n:>10s}" for n in
        ["seq", "pairwise", "lane8", "lane16", "lane8tree", "f64"]))

    totals = {}
    for (M, K, N) in SHAPES:
        x = torch.randn(M, K, generator=g)
        w = torch.randn(N, K, generator=g) * (K ** -0.5)
        want = F.linear(x, w).numpy()

        xn, wn = x.numpy(), w.numpy()
        # (M, N, K) elementwise products in fp32 -- the multiplies are exact-
        # rounded identically in every candidate, so only the ADD order varies
        prod = (xn[:, None, :] * wn[None, :, :]).astype(np.float32)

        row = {}
        row["seq"] = cand_sequential(prod)
        row["pairwise"] = cand_pairwise(prod)
        row["lane8"] = cand_lanes(prod, 8)
        row["lane16"] = cand_lanes(prod, 16)
        row["lane8tree"] = cand_lanes_tree(prod, 8)
        row["f64"] = cand_f64(xn, wn)

        cells = []
        for name in ["seq", "pairwise", "lane8", "lane16", "lane8tree", "f64"]:
            v = row[name]
            if v is None:
                cells.append(f"{'-':>10s}")
                continue
            fe = frac_equal(v, want)
            totals.setdefault(name, []).append(fe)
            cells.append(f"{100*fe:9.4f}%")
        print(f"{str((M,K,N)):>20s} " + " ".join(cells))

    print("\nmean bit-identical fraction across shapes:")
    for name, vals in totals.items():
        print(f"  {name:10s} {100*np.mean(vals):8.4f}%")

    print("\nInterpretation: 100% for some candidate at every shape means an "
          "end-to-end\nbit-exact port is reachable by pinning that order. "
          "Anything less means the\nreference's GEMM must itself be pinned "
          "(see docs/BITEXACT.md).")


if __name__ == "__main__":
    main()
