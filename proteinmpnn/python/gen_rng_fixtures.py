"""Fixtures pinning PyTorch's CPU RNG: randn, exponential_, multinomial.

These are the two randomness sources ProteinMPNN uses (decoding order + amino
acid sampling), so they have to be bit-exact for `--seed N` to mean the same
thing in Rust as it does in PyTorch.
"""
import numpy as np
import torch

from common import Manifest, save_fixture, FIX
import os

SIZES = [16, 17, 20, 37, 64, 100, 128, 129, 200, 256, 257, 301, 512, 1000]
SEEDS = [0, 1, 37, 12345, 999999]


def main():
    man = Manifest(os.path.join(FIX, "rng", "manifest.json"))

    # ---- torch.randn: every (seed, size) pair -------------------------------
    for seed in SEEDS:
        for size in SIZES:
            torch.manual_seed(seed)
            x = torch.randn(size)
            name = f"randn_s{seed}_n{size}"
            save_fixture("rng", name, {"y": x})
            man.add(kind="randn", name=name, seed=seed, size=size)

    # ---- Tensor.exponential_(1) --------------------------------------------
    for seed in (0, 7, 4242):
        for size in (21, 105, 512):
            torch.manual_seed(seed)
            x = torch.empty(size, dtype=torch.float32).exponential_(1)
            name = f"exp_s{seed}_n{size}"
            save_fixture("rng", name, {"y": x})
            man.add(kind="exponential", name=name, seed=seed, size=size)

    # ---- torch.multinomial(probs, 1) ---------------------------------------
    # A long chain of draws from the same generator, mimicking the decoder loop:
    # each step consumes exactly 21 uniforms, so any drift shows up immediately.
    rs = np.random.RandomState(0)
    for seed in (0, 5, 2024):
        probs = rs.rand(400, 21).astype(np.float32)
        probs = probs / probs.sum(-1, keepdims=True)
        pt = torch.from_numpy(probs)
        torch.manual_seed(seed)
        picks = np.array([torch.multinomial(pt[i], 1).item() for i in range(400)], dtype=np.int64)
        name = f"multinomial_s{seed}"
        save_fixture("rng", name, {"probs": probs, "picks": picks})
        man.add(kind="multinomial", name=name, seed=seed, n=400)

    # ---- argsort on |randn|: the decoding-order primitive -------------------
    for seed in (0, 37):
        for L in (50, 137, 256):
            torch.manual_seed(seed)
            randn = torch.randn(1, L)
            chain_mask = torch.ones(1, L)
            order = torch.argsort((chain_mask + 0.0001) * torch.abs(randn))
            name = f"decorder_s{seed}_L{L}"
            save_fixture("rng", name, {"randn": randn, "order": order})
            man.add(kind="decoding_order", name=name, seed=seed, L=L)

    man.write()


if __name__ == "__main__":
    main()
