#!/usr/bin/env python3
"""SOP §2 / rung 3 — dump the checkpoint's state dict to safetensors.

The Rust side reads `RFD_173.pt` **directly** (ZIP + pickle, `src/pth.rs`); this
fixture exists purely so the loader can be checked value-for-value against what
`torch.load` produces. Tolerance on rung 3 is exactly 0.

Note both state dicts are dumped: `model_state_dict` (EMA — what
`inference.state_dict_to_load` selects by default) and `final_state_dict`. Only
570 of 7 208 tensors are identical between them, so a loader that silently picks
the wrong one would otherwise look fine on a spot check.

    .venv/bin/python python/gen_weight_fixture.py            # EMA only (332 MB)
    .venv/bin/python python/gen_weight_fixture.py --both     # both (664 MB)
"""
import os
import sys

import common
import torch


def main(argv):
    both = "--both" in argv
    path = common.CKPT_173

    common.add_ref_to_path()
    try:
        ck = torch.load(path, map_location="cpu", weights_only=False)
    except ModuleNotFoundError as e:
        import stub_pickle
        print(f"(missing dep {e.name!r}; loading with stub unpickler)")
        ck = torch.load(path, map_location="cpu", weights_only=False,
                        pickle_module=stub_pickle)

    names = ["model_state_dict"] + (["final_state_dict"] if both else [])
    for sd_name in names:
        sd = {k: v for k, v in ck[sd_name].items() if torch.is_tensor(v)}
        n_params = sum(v.numel() for v in sd.values())
        print(f"{sd_name}: {len(sd)} tensors, {n_params} parameters")
        common.write_fixture("weights", sd_name, sd, {
            "checkpoint": os.path.basename(path),
            "n_tensors": len(sd),
            "n_params": n_params,
            "torch_version": torch.__version__,
        })


if __name__ == "__main__":
    main(sys.argv[1:])
