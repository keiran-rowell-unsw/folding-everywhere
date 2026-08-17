#!/usr/bin/env python3
"""SOP §1.5 — inventory the checkpoint.

Prints every tensor name, shape and dtype, plus the total parameter count.
That count becomes a test on the Rust side.

Also dumps the config embedded in the checkpoint, because RFdiffusion2's
Sampler.load_model() builds the network from `weights_pkl['conf']` merged over
`config/training/base.yaml` — i.e. the architecture is decided by the file, not
by the inference yaml. The Rust port must read the same source.

    python3 inventory_checkpoint.py [path/to/RFD_173.pt]
"""
import sys
import json
import collections

import common
import torch


def main(path=None):
    path = path or common.CKPT_173
    print(f"=== {path} ===")
    # The pickle references upstream classes. Try the real ones first (so the
    # config comes back as real OmegaConf objects); fall back to stubs, which
    # is enough for the tensor inventory and does not need dgl/openbabel/rdkit.
    common.add_ref_to_path()
    try:
        ck = torch.load(path, map_location="cpu", weights_only=False)
        print("(loaded with real upstream classes)")
    except ModuleNotFoundError as e:
        import stub_pickle
        print(f"(missing dep {e.name!r}; loading with stub unpickler)")
        ck = torch.load(path, map_location="cpu", weights_only=False,
                        pickle_module=stub_pickle)
        if stub_pickle.stubbed():
            print("stubbed classes: " + ", ".join(stub_pickle.stubbed()))

    print(f"top-level keys: {list(ck.keys())}")
    for k, v in ck.items():
        if isinstance(v, dict):
            print(f"  {k:24s} dict[{len(v)}]")
        elif torch.is_tensor(v):
            print(f"  {k:24s} tensor {list(v.shape)} {v.dtype}")
        else:
            print(f"  {k:24s} {type(v).__name__}")

    # ---- the config that decides the architecture -------------------------
    if "conf" in ck:
        try:
            from omegaconf import OmegaConf
            conf = OmegaConf.to_container(ck["conf"], resolve=False)
        except Exception:
            conf = ck["conf"]
        out = common.write_json("weights", "ckpt_conf",
                                json.loads(json.dumps(conf, default=str)))
        print(f"config -> {out}")

    # ---- every state dict in the file -------------------------------------
    for sd_name in ("model_state_dict", "final_state_dict", "model"):
        if sd_name not in ck or not isinstance(ck[sd_name], dict):
            continue
        sd = ck[sd_name]
        n_tensors = sum(1 for v in sd.values() if torch.is_tensor(v))
        n_params = sum(v.numel() for v in sd.values() if torch.is_tensor(v))
        dtypes = collections.Counter(
            str(v.dtype) for v in sd.values() if torch.is_tensor(v))
        print(f"\n--- {sd_name}: {n_tensors} tensors, {n_params} parameters, "
              f"dtypes {dict(dtypes)} ---")

        # group by top-level module so the port can be planned module by module
        by_prefix = collections.Counter()
        params_by_prefix = collections.Counter()
        for k, v in sd.items():
            if not torch.is_tensor(v):
                continue
            pre = k.split(".")[0] if "." in k else k
            by_prefix[pre] += 1
            params_by_prefix[pre] += v.numel()
        print(f"{'module':32s} {'tensors':>8s} {'params':>14s}")
        for pre, cnt in sorted(params_by_prefix.items(),
                               key=lambda kv: -kv[1]):
            print(f"{pre:32s} {by_prefix[pre]:8d} {params_by_prefix[pre]:14d}")

        # the full listing, to a file (it is thousands of lines)
        listing = [
            {"name": k, "shape": list(v.shape), "dtype": str(v.dtype),
             "numel": v.numel()}
            for k, v in sd.items() if torch.is_tensor(v)
        ]
        common.write_json("weights", f"inventory_{sd_name}", {
            "checkpoint": path,
            "n_tensors": n_tensors,
            "n_params": n_params,
            "dtypes": dict(dtypes),
            "tensors": listing,
        })

    # ---- do the two state dicts differ? (ema vs final) --------------------
    if ("model_state_dict" in ck and "final_state_dict" in ck
            and isinstance(ck["model_state_dict"], dict)):
        a, b = ck["model_state_dict"], ck["final_state_dict"]
        same_keys = set(a) == set(b)
        n_equal = sum(1 for k in a
                      if torch.is_tensor(a[k]) and k in b
                      and torch.equal(a[k], b[k]))
        print(f"\nmodel_state_dict vs final_state_dict: same keys={same_keys}, "
              f"bit-identical tensors={n_equal}/{len(a)}")
        print("(inference/base.yaml uses state_dict_to_load=model_state_dict, "
              "i.e. the EMA weights)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
