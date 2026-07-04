"""Run a real fold and dump selected submodule inputs/outputs as .npy fixtures.

Uses forward hooks (with_kwargs) to capture the exact tensors each module sees,
so the Rust port can be validated module-by-module against ground truth even for
modules whose inputs depend on upstream RNG (we feed the captured input).
"""
import os, sys
import numpy as np
import torch
import common

def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "crambin46"
    seq = common.PROTEINS[name]
    FIX = common.FIX
    torch.set_num_threads(8)
    model = common.load_model(fp32=True)
    feats = common.features(seq)

    saved = {}
    def save(tag, arr):
        if torch.is_tensor(arr):
            arr = arr.detach().cpu()
            np.save(os.path.join(FIX, f"mod_{name}_{tag}.npy"),
                    arr.to(torch.float32).numpy() if arr.is_floating_point() else arr.numpy())
            saved[tag] = tuple(arr.shape)

    hooks = []
    def hook(tag, *, in_names=None, out_names=None):
        def fn(mod, args, kwargs, output):
            if in_names:
                allin = list(args) + [kwargs.get(k) for k in kwargs]
                # save positional inputs by index name
                for i, a in enumerate(args):
                    if torch.is_tensor(a):
                        save(f"{tag}_in{i}", a)
                for k, v in kwargs.items():
                    if torch.is_tensor(v):
                        save(f"{tag}_kw_{k}", v)
            if isinstance(output, tuple):
                for i, o in enumerate(output):
                    if torch.is_tensor(o):
                        save(f"{tag}_out{i}", o)
            elif torch.is_tensor(output):
                save(f"{tag}_out", output)
        return fn

    def reg(module, tag):
        hooks.append(module.register_forward_hook(hook(tag, in_names=True), with_kwargs=True))

    # Deterministic trunk modules (inputs captured -> feed same to Rust).
    reg(model.rel_pos, "relpos")
    reg(model.language_model, "lmshim")
    reg(model.folding_trunk, "trunk")
    reg(model.folding_trunk.blocks[0], "trunk_block0")
    reg(model.folding_trunk.blocks[0].tri_mul_out, "trimul_out0")
    reg(model.folding_trunk.blocks[0].tri_mul_in, "trimul_in0")
    reg(model.folding_trunk.blocks[0].pair_transition, "pair_trans0")
    reg(model.inputs_embedder, "inputs_embedder")
    reg(model.z_init_1, "zinit1")
    reg(model.z_init_2, "zinit2")
    reg(model.token_bonds, "token_bonds")

    torch.manual_seed(0)
    with torch.inference_mode():
        model(**feats, num_loops=3, num_diffusion_samples=1, num_sampling_steps=14)
    for h in hooks:
        h.remove()
    print(f"saved {len(saved)} module fixtures for {name}:", flush=True)
    for k in sorted(saved):
        print(f"  {k}: {saved[k]}", flush=True)

if __name__ == "__main__":
    main()
