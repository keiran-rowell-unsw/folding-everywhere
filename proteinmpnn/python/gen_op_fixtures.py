"""Per-op fixtures: the primitive kernels the port is built from.

Shapes are chosen to match the ones ProteinMPNN actually uses (K = 384/512 for
the message MLPs, 416 for the edge projection, 66 for the positional one-hot),
so the reduction widths under test are the real ones.
"""
import os

import numpy as np
import torch

from common import FIX, Manifest, save_fixture

rs = np.random.RandomState(0)


def rnd(*shape, scale=1.0):
    return (rs.randn(*shape) * scale).astype(np.float32)


def main():
    man = Manifest(os.path.join(FIX, "ops", "manifest.json"))

    # ---- Linear at every width the model uses ------------------------------
    for name, (m, k, o) in {
        "linear_66x16": (512, 66, 16),
        "linear_416x128": (512, 416, 128),
        "linear_384x128": (512, 384, 128),
        "linear_512x128": (512, 512, 128),
        "linear_128x512": (512, 128, 512),
        "linear_128x21": (256, 128, 21),
    }.items():
        x, w, b = rnd(m, k), rnd(o, k, scale=0.1), rnd(o, scale=0.1)
        y = torch.nn.functional.linear(torch.from_numpy(x), torch.from_numpy(w), torch.from_numpy(b))
        save_fixture("ops", name, {"x": x, "w": w, "b": b, "y": y})
        man.add(kind="linear", name=name, m=m, k=k, o=o)

    # ---- LayerNorm ---------------------------------------------------------
    for name, (rows, c) in {"layernorm_128": (1024, 128), "layernorm_512": (256, 512)}.items():
        x, w, b = rnd(rows, c), rnd(c, scale=0.5) + 1.0, rnd(c, scale=0.1)
        y = torch.nn.functional.layer_norm(
            torch.from_numpy(x), (c,), torch.from_numpy(w), torch.from_numpy(b), 1e-5
        )
        save_fixture("ops", name, {"x": x, "w": w, "b": b, "y": y})
        man.add(kind="layer_norm", name=name, rows=rows, c=c)

    # ---- Activations & normalisations --------------------------------------
    x = np.concatenate([rnd(4096, scale=4.0), np.array([0, -0, 20, -20, 1e-8], np.float32)])
    save_fixture("ops", "gelu", {"x": x, "y": torch.nn.functional.gelu(torch.from_numpy(x))})
    man.add(kind="gelu", name="gelu", n=int(x.size))

    z = rnd(512, 21, scale=3.0)
    save_fixture("ops", "softmax_last", {"x": z, "y": torch.softmax(torch.from_numpy(z), -1)})
    man.add(kind="softmax", name="softmax_last")
    save_fixture(
        "ops", "log_softmax_last",
        {"x": z, "y": torch.log_softmax(torch.from_numpy(z), -1)},
    )
    man.add(kind="log_softmax", name="log_softmax_last")

    # ---- Embedding (W_s lookup) --------------------------------------------
    w = rnd(21, 128, scale=0.5)
    ids = rs.randint(0, 21, size=256).astype(np.int64)
    y = torch.nn.functional.embedding(torch.from_numpy(ids), torch.from_numpy(w))
    save_fixture("ops", "embedding", {"w": w, "ids": ids, "y": y})
    man.add(kind="embedding", name="embedding")

    # ---- sum over the neighbour axis, /30 (EncLayer & DecLayer message pool)
    x = rnd(64, 48, 128)
    y = torch.sum(torch.from_numpy(x), -2) / 30
    save_fixture("ops", "sum_neighbors", {"x": x, "y": y})
    man.add(kind="sum_neighbors", name="sum_neighbors")

    man.write()


if __name__ == "__main__":
    main()
