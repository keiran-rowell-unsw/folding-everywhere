"""PyTorch fold benchmark for one protein + precision variant.
  fp32 : esmc fp32, fp32 head, no autocast (== Rust target)
  bf16 : esmc bf16 + CPU bf16 autocast (== released GPU path's CPU equivalent)
Times the forward, saves coords, prints JSON metrics.
Usage: python bench_pytorch.py <protein> <fp32|bf16>
"""
import os, sys, time, json, numpy as np, torch, common
from proteins10 import PROTEINS10
name, prec = sys.argv[1], sys.argv[2]
seq = PROTEINS10[name]; FIX = common.FIX
torch.set_num_threads(8)

if prec == "fp32":
    model = common.load_model(fp32=True)
    ctx = torch.inference_mode()
    feats = common.features(seq)
    torch.manual_seed(0); t0 = time.time()
    with ctx:
        out = model(**feats, num_loops=3, num_diffusion_samples=1, num_sampling_steps=14)
    fold_s = time.time() - t0
else:
    # released bf16: fp32 head + bf16 ESM-C, run under CPU bf16 autocast
    from transformers.models.esmfold2.modeling_esmfold2 import ESMFold2Model
    model = ESMFold2Model.from_pretrained("biohub/ESMFold2", dtype=torch.float32,
                                          low_cpu_mem_usage=True, esmc_precision="bf16").eval()
    model.set_kernel_backend(None); model.set_chunk_size(None)
    feats = common.features(seq)
    torch.manual_seed(0); t0 = time.time()
    with torch.inference_mode(), torch.autocast("cpu", dtype=torch.bfloat16):
        out = model(**feats, num_loops=3, num_diffusion_samples=1, num_sampling_steps=14)
    fold_s = time.time() - t0

np.save(os.path.join(FIX, f"fold_{name}_pt_{prec}_coords.npy"),
        out["sample_atom_coords"][0].detach().cpu().float().numpy())
print(json.dumps({"protein": name, "variant": f"pt_{prec}", "L": len(seq), "fold_s": round(fold_s, 2),
    "plddt_mean": round(float(out["plddt"].float().mean()), 5), "ptm": round(float(out["ptm"]), 5),
    "complex_plddt": round(float(out["complex_plddt"]), 5)}))
