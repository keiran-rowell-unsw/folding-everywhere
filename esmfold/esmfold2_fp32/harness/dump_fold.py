"""pt_fp32 benchmark run + Rust fixtures (fp32 reference). Captures ESM-C input_ids,
all features, RNG, confidence metadata, and reference coords/pLDDT/pTM. Disk writes
are deferred until AFTER the timed forward. Naming: fold_<prot>_<key>.npy
Usage: python dump_fold.py <protein_name>
"""
import os, sys, time, json, numpy as np, torch, torch.nn as nn
import torch.nn.functional as F
import common
from proteins10 import PROTEINS10

name = sys.argv[1]; seq = PROTEINS10[name]; FIX = common.FIX
torch.set_num_threads(8)
model = common.load_model(fp32=True)
from transformers.models.esmfold2 import protein_utils as pu
feats = pu.prepare_protein_features(seq)

store = {}  # tag -> np array (written after the timed forward)
def put(tag, t):
    if not torch.is_tensor(t): return
    t = t.detach().cpu()
    store[tag] = t.to(torch.float32).numpy() if t.is_floating_point() else t.numpy()

put("res_ids", feats["input_ids"][0])

_otn = nn.init.trunc_normal_
def _tn(t, *a, **k):
    o = _otn(t, *a, **k); put("z_rand", o); return o
nn.init.trunc_normal_ = _tn
head = model.structure_head
caps = {"R": [], "t": [], "churn": [], "xinit": None}
_orr = head._random_rotations
def _rr(n, dt, dev):
    R = _orr(n, dt, dev); caps["R"].append(R.detach().cpu().float().numpy()); return R
head._random_rotations = _rr
_ornd = torch.randn
def _rnd(*a, **k):
    o = _ornd(*a, **k)
    if caps["xinit"] is None and o.dim() == 3 and o.shape[-1] == 3 and o.shape[1] > 1:
        caps["xinit"] = o.detach().cpu().float().numpy()
    return o
_orl = torch.randn_like
def _rl(x, *a, **k):
    o = _orl(x, *a, **k)
    if o.dim() == 3 and o.shape[1] == 1: caps["t"].append(o.detach().cpu().float().numpy())
    elif o.dim() == 3 and o.shape[1] > 1 and o.shape[-1] == 3: caps["churn"].append(o.detach().cpu().float().numpy())
    return o
torch.randn = _rnd; torch.randn_like = _rl

cnt = {"lm": 0, "msa": 0}; hooks = []
def reg(m, fn): hooks.append(m.register_forward_hook(fn, with_kwargs=True))
def ie_hook(m, a, kw, o):
    for k, v in kw.items(): put(f"ie_{k}", v)
reg(model.inputs_embedder, ie_hook)
def rp_hook(m, a, kw, o):
    for k, v in kw.items(): put(f"rp_{k}", v)
reg(model.rel_pos, rp_hook)
def tb_hook(m, a, kw, o):
    put("token_bonds_feat", a[0] if a else kw.get("input"))
reg(model.token_bonds, tb_hook)
def msa_hook(m, a, kw, o):
    if cnt["msa"] == 0:
        for k, v in kw.items(): put(f"msa_{k}", v)
    cnt["msa"] += 1
reg(model.msa_encoder, msa_hook)
def lm_hook(m, a, kw, o):
    put(f"lm_z_loop{cnt['lm']}", a[0]); cnt["lm"] += 1
reg(model.lm_encoder, lm_hook)
def cnf_hook(m, a, kw, o):
    for k in ("distogram_atom_idx", "mol_type", "token_attention_mask"):
        if k in kw: put(f"cnf_{k}", kw[k])
    for k in ("plddt", "ptm", "iptm", "complex_plddt"):
        if k in o: put(f"ref_{k}", o[k])
reg(model.confidence_head, cnf_hook)

torch.manual_seed(0)
t0 = time.time()
with torch.inference_mode():
    out = model(**feats, num_loops=3, num_diffusion_samples=1, num_sampling_steps=14)
fold_s = time.time() - t0
for h in hooks: h.remove()
nn.init.trunc_normal_ = _otn; torch.randn = _ornd; torch.randn_like = _orl; head._random_rotations = _orr

put("ref_coords", out["sample_atom_coords"][0])
sched = head.inference_noise_schedule(14); sched = sched[sched <= 256.0]; sched = F.pad(sched, (1, 0), value=256.0)
store["schedule"] = sched.cpu().numpy().astype(np.float32)
store["xinit_raw"] = caps["xinit"][0].astype(np.float32)
store["R"] = np.stack([r[0] for r in caps["R"]], 0).astype(np.float32)
store["t"] = np.stack([t[0, 0] for t in caps["t"]], 0).astype(np.float32)
store["churn"] = np.stack([c[0] for c in caps["churn"]], 0).astype(np.float32)
for tag, arr in store.items():
    np.save(os.path.join(FIX, f"fold_{name}_{tag}.npy"), np.ascontiguousarray(arr))

L = len(seq)
print(json.dumps({"protein": name, "variant": "pt_fp32", "L": L, "fold_s": round(fold_s, 2),
    "plddt_mean": round(float(out["plddt"].float().mean()), 5), "ptm": round(float(out["ptm"]), 5),
    "complex_plddt": round(float(out["complex_plddt"]), 5)}))
