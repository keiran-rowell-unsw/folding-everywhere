#!/usr/bin/env python3
"""Probe v2 — the same question, but with FMA.

v1 found that a strict left-to-right fp32 accumulation matches torch on ~99% of
outputs when K <= 65, and ~10-15% when K >= 192. A 99% match is not "nearly the
right order" -- it is the right order with one detail wrong. The obvious
candidate is **FMA**: MKL's kernels issue `vfmadd`, so `x[k]*w[k]` is never
rounded to fp32 before being added. v1 rounded every product.

An FMA on fp32 inputs is exactly representable in f64 (24+24 = 48 <= 53 bits),
so it can be emulated exactly:

    fma(a, b, c) = float32( float64(c) + float64(a) * float64(b) )

This script tests sequential and L-lane accumulation, each with and without FMA.

    .venv/bin/python python/probe_gemm_order2.py
"""
import common
import numpy as np
import torch
import torch.nn.functional as F

SHAPES = [
    (1, 32, 32), (7, 32, 32), (150, 32, 32),
    (150, 64, 32), (150, 65, 32), (150, 114, 64),
    (150, 128, 128), (150, 192, 192), (150, 256, 256),
    (150, 320, 64), (150, 32, 576),
]

F32 = np.float32
F64 = np.float64


def frac_equal(a, b):
    a = np.ascontiguousarray(a.astype(F32))
    b = np.ascontiguousarray(b.astype(F32))
    return float((a.view(np.uint32) == b.view(np.uint32)).mean())


def seq_fma(x, w):
    """acc = fma(x[k], w[k], acc), strict k order, fp32 accumulator."""
    M, K = x.shape
    N = w.shape[0]
    acc = np.zeros((M, N), dtype=F64)
    xd, wd = x.astype(F64), w.astype(F64)
    for k in range(K):
        acc = (acc + xd[:, k][:, None] * wd[:, k][None, :]).astype(F32).astype(F64)
    return acc.astype(F32)


def lanes_fma(x, w, L, tree=False):
    """L independent fp32 accumulators, lane = k % L, each updated by FMA;
    lanes then combined by plain fp32 adds (left-to-right, or balanced tree)."""
    M, K = x.shape
    N = w.shape[0]
    xd, wd = x.astype(F64), w.astype(F64)
    acc = np.zeros((M, N, L), dtype=F64)
    for k in range(K):
        l = k % L
        acc[:, :, l] = (acc[:, :, l]
                        + xd[:, k][:, None] * wd[:, k][None, :]).astype(F32).astype(F64)
    parts = [acc[:, :, l].astype(F32) for l in range(L)]
    if tree:
        while len(parts) > 1:
            parts = [(parts[i] + parts[i + 1]).astype(F32) if i + 1 < len(parts)
                     else parts[i] for i in range(0, len(parts), 2)]
        return parts[0]
    out = parts[0]
    for l in range(1, L):
        out = (out + parts[l]).astype(F32)
    return out


def main():
    torch.set_num_threads(1)
    g = torch.Generator().manual_seed(11)
    print(f"torch {torch.__version__}\n")

    names = ["seq+fma", "l2+fma", "l4+fma", "l8+fma", "l16+fma", "l8tree+fma"]
    print(f"{'shape (M,K,N)':>18s} " + " ".join(f"{n:>11s}" for n in names))

    totals = {n: [] for n in names}
    for (M, K, N) in SHAPES:
        x = torch.randn(M, K, generator=g)
        w = torch.randn(N, K, generator=g) * (K ** -0.5)
        want = F.linear(x, w).numpy()
        xn, wn = x.numpy(), w.numpy()

        got = {
            "seq+fma": seq_fma(xn, wn),
            "l2+fma": lanes_fma(xn, wn, 2),
            "l4+fma": lanes_fma(xn, wn, 4),
            "l8+fma": lanes_fma(xn, wn, 8),
            "l16+fma": lanes_fma(xn, wn, 16),
            "l8tree+fma": lanes_fma(xn, wn, 8, tree=True),
        }
        cells = []
        for n in names:
            fe = frac_equal(got[n], want)
            totals[n].append(fe)
            cells.append(f"{100*fe:10.4f}%")
        print(f"{str((M,K,N)):>18s} " + " ".join(cells))

    print("\nmean bit-identical fraction:")
    for n in names:
        print(f"  {n:12s} {100*np.mean(totals[n]):8.4f}%")


if __name__ == "__main__":
    main()
