"""Config sweep — PyTorch fp32 reference for the Rust-vs-PyTorch bit-exactness check.

Runs one protein through 6 configurations that vary `num_loops` and `num_sampling_steps`
(a single diffusion sample throughout — `num_diffusion_samples=1`), and saves each run's
all-atom coordinates plus pLDDT/pTM. The pure-Rust `fold_standalone <seq> 0 out.npy
<loops> <steps>` is run at the same configs and compared; agreement to fp32 round-off
(~1e-4 Å) is the bit-exactness result.

Usage: python sweep_configs.py <protein_name> <out_dir>
"""
from __future__ import annotations
import sys, os, numpy as np, torch
import common

# (num_loops, num_sampling_steps) — num_diffusion_samples is fixed to 1.
CONFIGS = [
    (3, 14),    # baseline (the benchmark setting)
    (6, 14),    # more trunk loops
    (10, 14),   # more trunk loops
    (3, 28),    # more sampling steps
    (3, 42),    # more sampling steps
    (6, 28),    # both
]

def single_sample(out):
    """Extract the (single) sample exactly as the reference output_to_pdb does."""
    c = out["sample_atom_coords"]
    if c.dim() == 4:
        c = c[:, 0]
    c = c.detach().cpu().numpy()[0]           # [n_atoms, 3]
    plddt = float(out["plddt"].detach().float()[0].mean())
    ptm = float(out["ptm"].detach().float().reshape(-1)[0])
    return c, plddt, ptm

def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "crambin46"
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "/tmp/sweep"
    os.makedirs(out_dir, exist_ok=True)
    seq = common.PROTEINS[name]
    torch.set_num_threads(8)
    model = common.load_model(fp32=True)
    feats = common.features(seq)
    print("loops,steps,plddt,ptm", flush=True)
    for (loops, steps) in CONFIGS:
        torch.manual_seed(0)
        with torch.inference_mode():
            out = model(**feats, num_loops=loops,
                        num_diffusion_samples=1, num_sampling_steps=steps)
        coords, plddt, ptm = single_sample(out)
        np.save(os.path.join(out_dir, f"pt_{name}_l{loops}_s{steps}.npy"), coords)
        print(f"{loops},{steps},{plddt:.6f},{ptm:.6f}", flush=True)
    print("DONE", flush=True)

if __name__ == "__main__":
    main()
