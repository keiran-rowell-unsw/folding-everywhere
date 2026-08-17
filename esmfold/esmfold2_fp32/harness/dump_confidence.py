"""Capture confidence_head inputs + outputs (crambin, seeded)."""
import os, numpy as np, torch, common
import sys as _s; name = _s.argv[1] if len(_s.argv)>1 else "crambin46"; seq = common.PROTEINS[name]; FIX = common.FIX
torch.set_num_threads(8)
model = common.load_model(fp32=True); feats = common.features(seq)
saved = {}
def save(tag, t):
    if not torch.is_tensor(t): return
    t = t.detach().cpu()
    a = t.to(torch.float32).numpy() if t.is_floating_point() else t.numpy()
    np.save(os.path.join(FIX, f"cnf_{name}_{tag}.npy"), a); saved[tag] = a.shape
def hook(m, args, kwargs, out):
    for k, v in kwargs.items(): save(f"in_{k}", v)
    for k in ("plddt", "ptm", "iptm", "complex_plddt", "pae", "plddt_logits", "pae_logits"):
        if k in out: save(k, out[k])
h = model.confidence_head.register_forward_hook(hook, with_kwargs=True)
torch.manual_seed(0)
with torch.inference_mode():
    model(**feats, num_loops=3, num_diffusion_samples=1, num_sampling_steps=14)
h.remove()
print(f"saved {len(saved)}:")
for k in sorted(saved): print(f"  {k}: {saved[k]}")
