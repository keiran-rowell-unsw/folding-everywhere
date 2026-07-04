"""Generalized ESMFold2 fp32 reference fold. Matches the sweep/dump_fold order:
prepare features, then manual_seed(0), then forward. Dumps sample_atom_coords
(flat AF3 atom order, [n_atoms,3]) as npy and a PDB. Thread count via REF_THREADS.

Usage: python ref_fold_ef2.py <name> <SEQ> <loops> <steps> <out_prefix>
"""
import os, sys, json, time, numpy as np, torch
import common

name, seq, loops, steps, prefix = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
torch.set_num_threads(int(os.environ.get("REF_THREADS", "8")))

model = common.load_model(fp32=True)
feats = common.features(seq)
torch.manual_seed(0)                      # seed AFTER featurization, like the sweep
t0 = time.time()
with torch.inference_mode():
    out = model(**feats, num_loops=loops, num_diffusion_samples=1, num_sampling_steps=steps)
fold_s = time.time() - t0

coords = out["sample_atom_coords"].detach().cpu().numpy()[0]   # [n_atoms,3]
np.save(prefix + "_coords.npy", coords)

# also write a PDB via the released output_to_pdb path
from transformers.models.esmfold2.protein_utils import OUTPUT_TO_PDB_FEATURE_KEYS
for k in OUTPUT_TO_PDB_FEATURE_KEYS:
    out[k] = feats[k]
open(prefix + ".pdb", "w").write(model.output_to_pdb(out))

print(json.dumps({"name": name, "loops": loops, "steps": steps, "threads": torch.get_num_threads(),
    "n_atoms": int(coords.shape[0]), "fold_s": round(fold_s, 2),
    "plddt_mean": round(float(out["plddt"].float().mean()), 6),
    "ptm": round(float(out["ptm"]), 6)}))
