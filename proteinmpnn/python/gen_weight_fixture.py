"""Dump the checkpoint's state_dict to safetensors so the Rust .pt reader can be
cross-checked value-by-value against torch.load."""
import os
import sys

import torch

from common import FIX, weights_path


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "v_48_020"
    kind = sys.argv[2] if len(sys.argv) > 2 else "vanilla"
    ckpt = torch.load(weights_path(name, kind), map_location="cpu", weights_only=False)
    sd = ckpt["model_state_dict"]
    from safetensors.torch import save_file

    out_dir = os.path.join(FIX, "weights")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{name}.safetensors")
    save_file({k: v.contiguous() for k, v in sd.items()}, path)
    print(f"wrote {path}  ({len(sd)} tensors, num_edges={ckpt['num_edges']}, "
          f"noise_level={ckpt['noise_level']})")


if __name__ == "__main__":
    main()
