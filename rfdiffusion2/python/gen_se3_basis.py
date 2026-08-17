#!/usr/bin/env python3
"""Export the Clebsch–Gordan (Wigner 3-j) tensors the SE(3) transformer needs.

`se3_transformer/model/basis.py` builds its pairwise bases from
`o3.wigner_3j(J, d_in, d_out, dtype=float64).permute(2, 1, 0)`. Those are
*constants* — they depend only on the degrees, never on the data — so they are
exported once here rather than re-derived in Rust.

That is the same decision as the chemical tables (`gen_chemical.py`): closed-form
recurrences for the 3-j symbols are easy to write and easy to get subtly wrong in
the sign convention, and a wrong sign in a basis tensor produces a network that
runs, produces plausible coordinates, and is wrong. Exporting removes the
question entirely — and `tests/parity_se3.rs` still checks the tensors it uses
against a reference-produced basis.

Degrees are exported up to `--max-degree` (default 2, one above what the shipped
checkpoint's `num_degrees=2` needs) so a different config does not silently fall
off the end of the table.

    PYTHONPATH=<ref> .venv/bin/python python/gen_se3_basis.py
"""
import argparse
import os

import common
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-degree", type=int, default=2)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import e3nn.o3 as o3

    tensors = {}
    for d_in in range(args.max_degree + 1):
        for d_out in range(args.max_degree + 1):
            for j in range(abs(d_in - d_out), d_in + d_out + 1):
                q = o3.wigner_3j(j, d_in, d_out, dtype=torch.float64).permute(2, 1, 0)
                # shape (2*d_out+1, 2*d_in+1, 2*J+1)
                assert q.shape == (2 * d_out + 1, 2 * d_in + 1, 2 * j + 1), q.shape
                tensors[f"cg_{d_in}_{d_out}_{j}"] = q.contiguous()

    out = args.out or os.path.join(
        common.PORT_ROOT, "rfd2", "data", "se3_cg.safetensors")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    from safetensors.torch import save_file
    save_file(tensors, out, metadata={"max_degree": str(args.max_degree),
                                      "e3nn": __import__("e3nn").__version__})
    n = sum(v.numel() for v in tensors.values())
    print(f"wrote {out}: {len(tensors)} tensors, {n} values")


if __name__ == "__main__":
    main()
