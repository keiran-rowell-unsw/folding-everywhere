"""Capture the MSA encoder's inputs+output (first loop call) for crambin."""
import os, numpy as np, torch, common
import sys as _s; name = _s.argv[1] if len(_s.argv)>1 else "crambin46"; seq = common.PROTEINS[name]; FIX = common.FIX
torch.set_num_threads(8)
model = common.load_model(fp32=True); feats = common.features(seq)
saved = {}
def save(tag, t):
    t = t.detach().cpu().to(torch.float32).numpy()
    np.save(os.path.join(FIX, f"msa_{name}_{tag}.npy"), t); saved[tag] = t.shape
cnt = {"i": 0}
def hook(m, args, kwargs, out):
    if cnt["i"] == 0:
        for k, v in kwargs.items():
            if torch.is_tensor(v): save(k, v)
        save("out", out)
    cnt["i"] += 1
h = model.msa_encoder.register_forward_hook(hook, with_kwargs=True)
torch.manual_seed(0)
with torch.inference_mode():
    model(**feats, num_loops=3, num_diffusion_samples=1, num_sampling_steps=14)
h.remove()
print(f"msa_encoder called {cnt['i']}x; saved:")
for k in sorted(saved): print(f"  {k}: {saved[k]}")
