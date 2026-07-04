"""Shared helpers for the PyTorch fp32 reference harness.

Everything is pinned to deterministic, single-thread fp32 so the reference is
reproducible run-to-run and forms a stable bit-target for the Rust port.
"""
import glob
import json
import os

import numpy as np
import torch

# ---- determinism / fp32 pinning -------------------------------------------------
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("MKL_CBWR", "COMPATIBLE")
torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)
torch.manual_seed(0)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIX = os.path.join(REPO, "fixtures")


def weights_path():
    cands = glob.glob(
        os.path.expanduser(
            "~/.cache/huggingface/hub/models--facebook--esmfold_v1/snapshots/*/model.safetensors"
        )
    )
    if not cands:
        raise FileNotFoundError("esmfold_v1 model.safetensors not found in HF cache")
    return cands[0]


def save_fixture(subdir, name, tensors: dict):
    """Save a dict of (name -> array/tensor) as an fp32 safetensors file."""
    from safetensors.numpy import save_file

    out_dir = os.path.join(FIX, subdir)
    os.makedirs(out_dir, exist_ok=True)
    np_tensors = {}
    for k, v in tensors.items():
        if isinstance(v, torch.Tensor):
            v = v.detach().contiguous().cpu().float().numpy()
        np_tensors[k] = np.ascontiguousarray(v, dtype=np.float32)
    path = os.path.join(out_dir, name + ".safetensors")
    save_file(np_tensors, path)
    return path


class Manifest:
    def __init__(self, path):
        self.path = path
        self.items = []

    def add(self, **kw):
        self.items.append(kw)

    def write(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "w") as f:
            json.dump(self.items, f, indent=1)
        print(f"wrote manifest {self.path} ({len(self.items)} entries)")
