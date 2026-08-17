"""Determinism pinning, paths, and the fixture writer.

SOP §2: this module pins determinism BEFORE torch is imported anywhere else, so
every other script in python/ must `import common` first.
"""
import os

# ---- must happen before `import torch` (SOP §2) ----------------------------
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("MKL_CBWR", "COMPATIBLE")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import torch  # noqa: E402
import numpy as np  # noqa: E402

torch.set_num_threads(1)
torch.set_default_dtype(torch.float32)

# ---- paths -----------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
PORT_ROOT = os.path.dirname(HERE)                       # rfdiffusion2-rs/
DESIGN_ROOT = os.path.dirname(PORT_ROOT)                # protein_design/
REF_ROOT = os.path.join(DESIGN_ROOT, "ref_RFdiffusion2")
WEIGHTS_DIR = os.path.join(REF_ROOT, "rf_diffusion", "model_weights")
CKPT_173 = os.path.join(WEIGHTS_DIR, "RFD_173.pt")
CKPT_140 = os.path.join(WEIGHTS_DIR, "RFD_140.pt")
FIXTURES = os.path.join(PORT_ROOT, "fixtures")
RESULTS = os.path.join(PORT_ROOT, "results")


def add_ref_to_path():
    """Put the pinned upstream repo on sys.path so we import it UNMODIFIED."""
    import sys
    for p in (REF_ROOT, os.path.join(REF_ROOT, "lib", "se3_flow_matching")):
        if os.path.isdir(p) and p not in sys.path:
            sys.path.insert(0, p)
    return REF_ROOT


def seed_all(seed=0):
    """Exactly what rf_diffusion/run_inference.py:seed_all does."""
    import random
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


# ---- fixture writer --------------------------------------------------------
def write_fixture(subdir, name, tensors, meta=None):
    """Write a dict of tensors to fixtures/<subdir>/<name>.safetensors.

    Integer tensors are written as int64 so indices survive the round-trip
    (SOP §2). Floating tensors are written as-is; assert fp32 unless the
    reference genuinely computes in fp64 (see docs/RECON.md §dtypes).
    """
    from safetensors.torch import save_file
    out_dir = os.path.join(FIXTURES, subdir)
    os.makedirs(out_dir, exist_ok=True)
    clean = {}
    for k, v in tensors.items():
        if not torch.is_tensor(v):
            v = torch.as_tensor(v)
        v = v.detach().cpu().contiguous()
        if v.dtype in (torch.int8, torch.int16, torch.int32, torch.bool,
                       torch.uint8):
            v = v.to(torch.int64)
        clean[k] = v
    path = os.path.join(out_dir, f"{name}.safetensors")
    save_file(clean, path, metadata={k: str(x) for k, x in (meta or {}).items()})
    total = sum(v.numel() for v in clean.values())
    print(f"  wrote {path}  ({len(clean)} tensors, {total} values)")
    return path


def write_json(subdir, name, obj):
    import json
    out_dir = os.path.join(FIXTURES, subdir)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{name}.json")
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2, sort_keys=True)
    print(f"  wrote {path}")
    return path
