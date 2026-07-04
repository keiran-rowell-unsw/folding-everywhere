"""Capture EDM sampler RNG (x_init, per-step rotation R + translation t),
the noise schedule, and the final sample_atom_coords (1 sample, seeded)."""
import os, numpy as np, torch
import torch.nn.functional as F
import common
import sys as _s; name = _s.argv[1] if len(_s.argv)>1 else "crambin46"; seq = common.PROTEINS[name]; FIX = common.FIX
torch.set_num_threads(8)
model = common.load_model(fp32=True); feats = common.features(seq)
head = model.structure_head
caps = {"R": [], "t": [], "xinit_raw": None}

orig_rr = head._random_rotations
def rr(n, dtype, device):
    R = orig_rr(n, dtype, device)
    caps["R"].append(R.detach().cpu().float().numpy())
    return R
head._random_rotations = rr

orig_randn = torch.randn
def randn(*a, **k):
    out = orig_randn(*a, **k)
    if caps["xinit_raw"] is None and out.dim() == 3 and out.shape[-1] == 3 and out.shape[1] > 1:
        caps["xinit_raw"] = out.detach().cpu().float().numpy()
    return out
caps["churn"] = []
orig_rl = torch.randn_like
def randn_like(x, *a, **k):
    out = orig_rl(x, *a, **k)
    if out.dim() == 3 and out.shape[1] == 1:
        caps["t"].append(out.detach().cpu().float().numpy())
    elif out.dim() == 3 and out.shape[1] > 1 and out.shape[-1] == 3:
        caps["churn"].append(out.detach().cpu().float().numpy())
    return out
torch.randn = randn; torch.randn_like = randn_like

# noise schedule (clipped) — compute the same way sample() does
sched = head.inference_noise_schedule(14)
sched = sched[sched <= 256.0]
sched = F.pad(sched, (1, 0), value=256.0)

# per-step x_noisy (diffusion_module input) + x_denoised (output)
dm = model.structure_head.diffusion_module
step_xn, step_xd = [], []
def dm_hook(m, args, kwargs, o):
    step_xn.append(kwargs["x_noisy"].detach().cpu().float().numpy()[0])
    step_xd.append(o["x_denoised"].detach().cpu().float().numpy()[0])
hh = dm.register_forward_hook(dm_hook, with_kwargs=True)

torch.manual_seed(0)
with torch.inference_mode():
    out = model(**feats, num_loops=3, num_diffusion_samples=1, num_sampling_steps=14)
torch.randn = orig_randn; torch.randn_like = orig_rl
hh.remove()
np.save(os.path.join(FIX, f"smp_{name}_step_xnoisy.npy"), np.stack(step_xn, 0).astype(np.float32))
np.save(os.path.join(FIX, f"smp_{name}_step_xdenoised.npy"), np.stack(step_xd, 0).astype(np.float32))

def save(tag, a): np.save(os.path.join(FIX, f"smp_{name}_{tag}.npy"), np.asarray(a, dtype=np.float32))
save("schedule", sched.detach().cpu().numpy())
save("xinit_raw", caps["xinit_raw"][0])           # [N,3]
save("R", np.stack([r[0] for r in caps["R"]], 0))  # [steps,3,3]
save("t", np.stack([t[0,0] for t in caps["t"]], 0))# [steps,3]
save("churn", np.stack([c[0] for c in caps["churn"]], 0))  # [steps,N,3]
save("coords", out["sample_atom_coords"][0].detach().cpu().numpy())  # [N,3]
print("schedule:", sched.detach().cpu().numpy())
print("n_rot=%d n_trans=%d xinit=%s coords amax=%.3f" % (
    len(caps["R"]), len(caps["t"]), caps["xinit_raw"].shape, float(out["sample_atom_coords"].abs().max())))
