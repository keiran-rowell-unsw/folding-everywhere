"""End-to-end fp32 reference fold — confirms the CPU pipeline + dumps outputs.

Usage: python ref_e2e.py <protein_name> [num_samples] [num_loops] [num_steps]
"""
from __future__ import annotations
import sys, time
import numpy as np
import torch
import common

def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "crambin46"
    n_samples = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    n_loops   = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    n_steps   = int(sys.argv[4]) if len(sys.argv) > 4 else 14
    seq = common.PROTEINS[name]
    print(f"=== {name} L={len(seq)} samples={n_samples} loops={n_loops} steps={n_steps} ===", flush=True)

    torch.set_num_threads(8)
    model = common.load_model(fp32=True)
    feats = common.features(seq)

    torch.manual_seed(0)
    t0 = time.time()
    with torch.inference_mode():
        out = model(**feats, num_loops=n_loops, num_diffusion_samples=n_samples,
                    num_sampling_steps=n_steps)
    dt = time.time() - t0
    print(f"[fold] {dt:.1f}s", flush=True)

    coords = out["sample_atom_coords"]
    print("sample_atom_coords:", tuple(coords.shape), coords.dtype)
    for k in ("plddt", "ptm", "iptm", "complex_plddt"):
        if k in out:
            v = out[k]
            print(f"  {k}: shape={tuple(v.shape)} mean={float(v.float().mean()):.6f}")
    common.savez(f"e2e_{name}",
                 sample_atom_coords=coords,
                 plddt=out["plddt"], ptm=out["ptm"],
                 complex_plddt=out["complex_plddt"],
                 distogram_logits=out["distogram_logits"],
                 atom_pad_mask=out["atom_pad_mask"])
    print("DONE", flush=True)

if __name__ == "__main__":
    main()
