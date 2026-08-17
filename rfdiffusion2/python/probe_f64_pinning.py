#!/usr/bin/env python3
"""Probe v3 — is an f64-accumulated GEMM order-INDEPENDENT after rounding to f32?

v1/v2 established that no fp32 reduction order reproduces PyTorch's GEMM: the
best candidate reaches 99.1% at small K and ~10% at K>=192. So bit-exactness
against *stock* PyTorch is not reachable by picking an order (SOP §0 is right).

There is a different route. If both sides accumulate the dot product in f64 and
round to f32 exactly once, the result is the correctly-rounded f32 dot product
-- which does not depend on summation order at all. Two implementations that
block, vectorise and thread completely differently then still agree bit-for-bit.

That is only true if the f64 accumulation error (~1e-16 relative) never straddles
an f32 rounding boundary (~1e-7 relative). It is not a theorem -- a value landing
within 1e-16 of a tie would break it -- so this script MEASURES the disagreement
rate between deliberately different f64 orders, over millions of values.

    .venv/bin/python python/probe_f64_pinning.py
"""
import common
import numpy as np
import torch

SHAPES = [
    (150, 32, 576), (150, 64, 32), (150, 114, 64), (150, 128, 128),
    (150, 192, 192), (150, 256, 256), (150, 320, 64), (400, 256, 256),
]

F32, F64 = np.float32, np.float64


def bits_equal(a, b):
    a = np.ascontiguousarray(a.astype(F32)).view(np.uint32)
    b = np.ascontiguousarray(b.astype(F32)).view(np.uint32)
    return a == b


def f64_blas(x, w):
    """f64 GEMM via BLAS (blocked, vectorised, whatever order MKL likes)."""
    return (x.astype(F64) @ w.astype(F64).T).astype(F32)


def f64_sequential(x, w):
    """f64 accumulation in strict k order -- a deliberately different order."""
    M, K = x.shape
    N = w.shape[0]
    xd, wd = x.astype(F64), w.astype(F64)
    acc = np.zeros((M, N), dtype=F64)
    for k in range(K):
        acc += xd[:, k][:, None] * wd[:, k][None, :]
    return acc.astype(F32)


def f64_reversed(x, w):
    """f64 accumulation in reverse k order -- another different order."""
    M, K = x.shape
    N = w.shape[0]
    xd, wd = x.astype(F64), w.astype(F64)
    acc = np.zeros((M, N), dtype=F64)
    for k in range(K - 1, -1, -1):
        acc += xd[:, k][:, None] * wd[:, k][None, :]
    return acc.astype(F32)


def f64_lane8(x, w):
    """8 independent f64 accumulators, combined at the end."""
    M, K = x.shape
    N = w.shape[0]
    xd, wd = x.astype(F64), w.astype(F64)
    acc = np.zeros((M, N, 8), dtype=F64)
    for k in range(K):
        acc[:, :, k % 8] += xd[:, k][:, None] * wd[:, k][None, :]
    return acc.sum(axis=-1).astype(F32)


def main():
    torch.set_num_threads(1)
    g = torch.Generator().manual_seed(2026)
    print(f"torch {torch.__version__}  numpy {np.__version__}\n")
    print("Comparing four DIFFERENT f64 summation orders, each rounded once to f32.")
    print("If f64-accumulate is order-independent at f32 output precision, all")
    print("four agree bit-for-bit everywhere.\n")

    print(f"{'shape (M,K,N)':>18s} {'values':>10s} {'blas=seq':>11s} "
          f"{'blas=rev':>11s} {'blas=lane8':>11s}")

    tot = 0
    dis = {"seq": 0, "rev": 0, "lane8": 0}
    for (M, K, N) in SHAPES:
        x = torch.randn(M, K, generator=g)
        w = torch.randn(N, K, generator=g) * (K ** -0.5)
        xn, wn = x.numpy(), w.numpy()

        a = f64_blas(xn, wn)
        b = f64_sequential(xn, wn)
        c = f64_reversed(xn, wn)
        d = f64_lane8(xn, wn)

        n = a.size
        eq_b = int(bits_equal(a, b).sum())
        eq_c = int(bits_equal(a, c).sum())
        eq_d = int(bits_equal(a, d).sum())
        tot += n
        dis["seq"] += n - eq_b
        dis["rev"] += n - eq_c
        dis["lane8"] += n - eq_d
        print(f"{str((M,K,N)):>18s} {n:10d} {100*eq_b/n:10.6f}% "
              f"{100*eq_c/n:10.6f}% {100*eq_d/n:10.6f}%")

    print(f"\ntotal values compared: {tot}")
    for k, v in dis.items():
        print(f"  disagreements (blas vs {k:5s}): {v}  "
              f"({'ALL IDENTICAL' if v == 0 else f'rate {v/tot:.3e}'})")

    print("\nFor reference, how much does f64-pinning change the VALUE vs stock "
          "fp32 torch:")
    import torch.nn.functional as F
    for (M, K, N) in SHAPES[:4]:
        x = torch.randn(M, K, generator=g)
        w = torch.randn(N, K, generator=g) * (K ** -0.5)
        stock = F.linear(x, w).numpy()
        pinned = f64_blas(x.numpy(), w.numpy())
        d = np.abs(stock - pinned)
        rel = d / np.maximum(np.abs(stock), 1e-12)
        print(f"  {str((M,K,N)):>18s} max|Δ| {d.max():.3e}  max rel {rel.max():.3e}")


if __name__ == "__main__":
    main()
