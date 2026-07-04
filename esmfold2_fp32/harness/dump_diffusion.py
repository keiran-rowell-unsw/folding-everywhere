"""Capture diffusion module step-0 I/O + sub-module outputs (1 sample, seeded)."""
import os, numpy as np, torch, common
import sys as _s; name = _s.argv[1] if len(_s.argv)>1 else "crambin46"; seq = common.PROTEINS[name]; FIX = common.FIX
torch.set_num_threads(8)
model = common.load_model(fp32=True); feats = common.features(seq)
saved = {}
def save(tag, t):
    if not torch.is_tensor(t): return
    t = t.detach().cpu()
    arr = t.to(torch.float32).numpy() if t.is_floating_point() else t.numpy()
    np.save(os.path.join(FIX, f"dif_{name}_{tag}.npy"), arr); saved[tag] = arr.shape

dm = model.structure_head.diffusion_module
cnt = {"dm":0,"cond":0,"tt":0,"ae":0,"ad":0}
def dm_hook(m, args, kwargs, out):
    if cnt["dm"]==0:
        for k,v in kwargs.items(): save(f"in_{k}", v)
        save("x_denoised", out["x_denoised"])
        if out.get("token_repr") is not None: save("token_repr", out["token_repr"])
    cnt["dm"]+=1
def cond_hook(m, args, kwargs, out):
    if cnt["cond"]==0: save("cond_s", out[0]); save("cond_z", out[1])
    cnt["cond"]+=1
def tt_hook(m, args, kwargs, out):
    if cnt["tt"]==0:
        save("tt_in_a", args[0] if args else kwargs.get("a"))
        save("tt_out", out[0] if isinstance(out, tuple) else out)
    cnt["tt"]+=1
def ae_hook(m, args, kwargs, out):
    if cnt["ae"]==0: save("ae_a", out[0])
    cnt["ae"]+=1
def ad_hook(m, args, kwargs, out):
    if cnt["ad"]==0: save("ad_rupdate", out[0])
    cnt["ad"]+=1
hooks=[dm.register_forward_hook(dm_hook, with_kwargs=True),
       dm.conditioning.register_forward_hook(cond_hook, with_kwargs=True),
       dm.token_transformer.register_forward_hook(tt_hook, with_kwargs=True),
       dm.atom_encoder.register_forward_hook(ae_hook, with_kwargs=True),
       dm.atom_decoder.register_forward_hook(ad_hook, with_kwargs=True)]
torch.manual_seed(0)
with torch.inference_mode():
    model(**feats, num_loops=3, num_diffusion_samples=1, num_sampling_steps=14)
for h in hooks: h.remove()
print(f"diffusion_module called {cnt['dm']}x; saved {len(saved)} fixtures:")
for k in sorted(saved): print(f"  {k}: {saved[k]}")
