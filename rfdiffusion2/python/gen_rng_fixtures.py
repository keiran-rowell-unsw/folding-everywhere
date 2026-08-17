#!/usr/bin/env python3
"""SOP §2 / rung 2 — fixtures for every distribution RFdiffusion2 draws from.

Three generators, because `run_inference.py:seed_all` seeds three
(docs/RECON.md §1.3). Rung 2 tolerance is **exactly 0** for all of them.

Run with the pinned venv:
    .venv/bin/python python/gen_rng_fixtures.py
"""
import random

import common
import numpy as np
import torch
from scipy.spatial.transform import Rotation

# (seed, size) pairs: sizes chosen to straddle every branch —
#   <16 and >=16 (torch's normal_fill threshold)
#   multiples of 16 and not (the redraw-the-last-16 rule)
#   >=256 (where the AVX2 FMA contraction becomes observable)
#   odd counts (numpy's polar-method cache leaves a value behind)
SEEDS = [0, 1, 37, 43, 1234, 2**31 - 1]
SIZES = [1, 2, 3, 7, 15, 16, 17, 31, 32, 33, 64, 100, 255, 256, 257, 1000, 4096]


def gen_torch():
    """at::mt19937 + ATen distributions."""
    out, meta = {}, {}

    for seed in SEEDS:
        for n in SIZES:
            torch.manual_seed(seed)
            out[f"randn_s{seed}_n{n}"] = torch.randn(n)
            torch.manual_seed(seed)
            out[f"rand_s{seed}_n{n}"] = torch.rand(n)

    # torch.rand with the exact shape RFScore.forward_from_rfi uses for psi_pred:
    # (B, I, L, 2) with B=1, I = n_ref_block+1 layers of output, L = length.
    for seed in (0, 43):
        for (B, I, L) in [(1, 1, 10), (1, 5, 10), (1, 5, 150), (1, 5, 37)]:
            torch.manual_seed(seed)
            out[f"psi_s{seed}_{B}x{I}x{L}"] = torch.rand((B, I, L, 2))

    # torch.randint — inference/centering.py:38
    for seed in (0, 1, 43):
        for hi in (2, 10, 150, 1000):
            torch.manual_seed(seed)
            out[f"randint_s{seed}_hi{hi}"] = torch.randint(
                low=0, high=hi, size=(64,))

    # sequential draws from ONE generator (SOP §4 rung 8: "one long run that
    # draws many samples from one generator")
    torch.manual_seed(43)
    seq = [torch.randn(17), torch.rand(5), torch.randn(256),
           torch.randint(0, 150, (8,)).float(), torch.rand((1, 5, 37, 2)).reshape(-1)]
    out["sequential_s43"] = torch.cat(seq)

    meta["torch_version"] = torch.__version__
    common.write_fixture("rng", "torch", out, meta)


def gen_numpy():
    """numpy legacy RandomState — the stream SciPy's Rotation.random uses."""
    out, meta = {}, {}

    for seed in SEEDS:
        for n in SIZES:
            np.random.seed(seed)
            out[f"normal_s{seed}_n{n}"] = torch.from_numpy(
                np.random.normal(size=n))            # float64 on purpose
            np.random.seed(seed)
            out[f"random_s{seed}_n{n}"] = torch.from_numpy(
                np.random.random_sample(size=n))

    # the polar-method cache: an odd-length draw leaves a value for the NEXT
    # call. Two calls of 3 must not equal one call of 6 unless the cache is
    # modelled correctly -- fixture both so the test can tell them apart.
    for seed in (0, 43):
        np.random.seed(seed)
        a = np.concatenate([np.random.normal(size=3), np.random.normal(size=3)])
        out[f"normal_cache_s{seed}_3then3"] = torch.from_numpy(a)
        np.random.seed(seed)
        out[f"normal_cache_s{seed}_6"] = torch.from_numpy(
            np.random.normal(size=6))

    # shuffle / permutation (contigs.py:64, inference/utils.py:200)
    for seed in (0, 1, 43):
        for n in (2, 5, 10, 64, 150):
            np.random.seed(seed)
            v = np.arange(n)
            np.random.shuffle(v)
            out[f"shuffle_s{seed}_n{n}"] = torch.from_numpy(v.astype(np.int64))
            np.random.seed(seed)
            out[f"permutation_s{seed}_n{n}"] = torch.from_numpy(
                np.random.permutation(n).astype(np.int64))

    # choice (inference/utils.py:254)
    for seed in (0, 43):
        for n in (5, 150):
            np.random.seed(seed)
            picks = [int(np.random.choice(np.arange(n))) for _ in range(32)]
            out[f"choice_s{seed}_n{n}"] = torch.tensor(picks, dtype=torch.int64)

    meta["numpy_version"] = np.__version__
    common.write_fixture("rng", "numpy", out, meta)


def gen_scipy_rotations():
    """_uniform_so3 -> Rotation.random -> the initial rotation noise of x_T."""
    out, meta = {}, {}
    import scipy

    for seed in SEEDS:
        for n in (1, 2, 5, 17, 150):
            np.random.seed(seed)
            m = Rotation.random(n).as_matrix()           # float64
            out[f"rotmat_s{seed}_n{n}"] = torch.from_numpy(
                np.ascontiguousarray(m))
            # and the fp32 narrowing _uniform_so3 actually applies
            out[f"rotmat32_s{seed}_n{n}"] = torch.from_numpy(
                np.ascontiguousarray(m)).to(torch.float32)

    # the raw quaternion draw, so a failure can be localised to the generator
    # rather than to the quaternion->matrix conversion
    for seed in (0, 43):
        for n in (2, 17):
            np.random.seed(seed)
            out[f"quat_raw_s{seed}_n{n}"] = torch.from_numpy(
                np.random.normal(size=(n, 4)))

    meta["scipy_version"] = scipy.__version__
    common.write_fixture("rng", "scipy_rot", out, meta)


def gen_pyrandom():
    """CPython's random module (contigs.py:184, aa_model.py:2655)."""
    out, meta = {}, {}
    import sys

    for seed in SEEDS:
        random.seed(seed)
        out[f"random_s{seed}_n64"] = torch.tensor(
            [random.random() for _ in range(64)], dtype=torch.float64)

    for seed in (0, 1, 43):
        for (lo, hi) in [(0, 1), (0, 9), (3, 7), (0, 149), (10, 1000)]:
            random.seed(seed)
            out[f"randint_s{seed}_{lo}_{hi}"] = torch.tensor(
                [random.randint(lo, hi) for _ in range(64)], dtype=torch.int64)

    for seed in (0, 43):
        for n in (2, 5, 150):
            random.seed(seed)
            out[f"choice_s{seed}_n{n}"] = torch.tensor(
                [random.choice(range(n)) for _ in range(64)], dtype=torch.int64)
            random.seed(seed)
            v = list(range(n))
            random.shuffle(v)
            out[f"shuffle_s{seed}_n{n}"] = torch.tensor(v, dtype=torch.int64)

    for seed in (0, 43):
        for k in (1, 8, 16, 31, 32, 33, 53, 64):
            random.seed(seed)
            out[f"getrandbits_s{seed}_k{k}"] = torch.tensor(
                [random.getrandbits(k) for _ in range(32)], dtype=torch.uint64
                if hasattr(torch, "uint64") else torch.int64)

    meta["python_version"] = sys.version.split()[0]
    common.write_fixture("rng", "pyrandom", out, meta)


def main():
    print("== torch ==")
    gen_torch()
    print("== numpy ==")
    gen_numpy()
    print("== scipy rotations ==")
    gen_scipy_rotations()
    print("== python random ==")
    gen_pyrandom()


if __name__ == "__main__":
    main()
