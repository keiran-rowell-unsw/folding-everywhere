"""Dump the main folding_trunk's per-block outputs (last loop call) for crambin."""
import os, numpy as np, torch, common
name = "crambin46"; seq = common.PROTEINS[name]; FIX = common.FIX
torch.set_num_threads(8)
model = common.load_model(fp32=True)
feats = common.features(seq)
store = {}
def mk(i):
    def fn(mod, args, kwargs, out):
        store[i] = out.detach().cpu().to(torch.float32).numpy()
    return fn
hooks = [model.folding_trunk.blocks[i].register_forward_hook(mk(i), with_kwargs=True)
         for i in range(len(model.folding_trunk.blocks))]
torch.manual_seed(0)
with torch.inference_mode():
    model(**feats, num_loops=3, num_diffusion_samples=1, num_sampling_steps=14)
for h in hooks: h.remove()
arr = np.stack([store[i][0] for i in range(len(store))], axis=0)  # [24, L, L, 256]
np.save(os.path.join(FIX, f"trunkblocks_{name}.npy"), arr)
print("saved per-block outputs", arr.shape, "amax per block:",
      [round(float(np.abs(arr[i]).max()),1) for i in range(arr.shape[0])])
