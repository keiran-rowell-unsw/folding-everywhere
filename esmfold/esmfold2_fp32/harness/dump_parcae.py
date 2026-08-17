"""Capture parcae-loop tensors + per-loop intermediates (one seeded fold)."""
import os, numpy as np, torch, torch.nn as nn, common

import sys as _s; name = _s.argv[1] if len(_s.argv)>1 else "crambin46"; seq = common.PROTEINS[name]; FIX = common.FIX
torch.set_num_threads(8)
model = common.load_model(fp32=True)
feats = common.features(seq)

saved = {}
def save(tag, t):
    t = t.detach().cpu().to(torch.float32).numpy()
    np.save(os.path.join(FIX, f"par_{name}_{tag}.npy"), t); saved[tag] = t.shape

_orig_tn = nn.init.trunc_normal_
def _tn(tensor, *a, **k):
    out = _orig_tn(tensor, *a, **k); save("z_rand", out); return out
nn.init.trunc_normal_ = _tn

hooks = []
def hook_out(tag):
    def fn(m, args, kwargs, out): save(tag, out)
    return fn
cnt = {"lm": 0, "ft": 0, "pin": 0}
def lm_hook(m, args, kwargs, out):
    i = cnt["lm"]; save(f"lm_z_loop{i}", args[0]); save(f"refined_lm{i}", out); cnt["lm"] += 1
def ft_hook(m, args, kwargs, out):
    i = cnt["ft"]; save(f"ft_in{i}", args[0]); save(f"ft_out{i}", out); cnt["ft"] += 1
def pin_hook(m, args, kwargs, out):
    i = cnt["pin"]; save(f"zinject{i}", args[0]); save(f"injected{i}", out); cnt["pin"] += 1

hooks.append(model.z_init_1.register_forward_hook(hook_out("zinit1"), with_kwargs=True))
hooks.append(model.z_init_2.register_forward_hook(hook_out("zinit2"), with_kwargs=True))
hooks.append(model.rel_pos.register_forward_hook(hook_out("relpos"), with_kwargs=True))
hooks.append(model.token_bonds.register_forward_hook(hook_out("token_bonds"), with_kwargs=True))
hooks.append(model.lm_encoder.register_forward_hook(lm_hook, with_kwargs=True))
hooks.append(model.folding_trunk.register_forward_hook(ft_hook, with_kwargs=True))
hooks.append(model.parcae_input_norm.register_forward_hook(pin_hook, with_kwargs=True))
hooks.append(model.parcae_readout.register_forward_hook(hook_out("readout_out"), with_kwargs=True))
hooks.append(model.parcae_coda.register_forward_hook(hook_out("final_z"), with_kwargs=True))
hooks.append(model.distogram_head.register_forward_hook(hook_out("distogram"), with_kwargs=True))

torch.manual_seed(0)
with torch.inference_mode():
    model(**feats, num_loops=3, num_diffusion_samples=1, num_sampling_steps=14)
for h in hooks: h.remove()
nn.init.trunc_normal_ = _orig_tn
print(f"saved {len(saved)} fixtures; loops lm={cnt['lm']} ft={cnt['ft']} pin={cnt['pin']}")
