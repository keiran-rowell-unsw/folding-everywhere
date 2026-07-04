"""PyTorch fp32 ESMFold2: fold one sequence at several num_loops (fixed sampling,
seed 0), dumping sample_atom_coords npy per loop count. Model loaded once.
Usage: python pt_loop_sweep.py <SEQ> <steps> <out_prefix> <loops...>
"""
import os, sys, json, numpy as np, torch
import common

seq, steps, prefix = sys.argv[1], int(sys.argv[2]), sys.argv[3]
loops_list = [int(x) for x in sys.argv[4:]]
torch.set_num_threads(int(os.environ.get("REF_THREADS", "8")))

model = common.load_model(fp32=True)
feats = common.features(seq)
for loops in loops_list:
    torch.manual_seed(0)                    # reseed before each fold (sweep order)
    with torch.inference_mode():
        out = model(**feats, num_loops=loops, num_diffusion_samples=1, num_sampling_steps=steps)
    coords = out["sample_atom_coords"].detach().cpu().numpy()[0]
    np.save(f"{prefix}_l{loops}_s{steps}.npy", coords)
    print(json.dumps({"loops": loops, "steps": steps, "n_atoms": int(coords.shape[0]),
        "plddt_mean": round(float(out["plddt"].float().mean()), 6),
        "ptm": round(float(out["ptm"]), 6)}), flush=True)
