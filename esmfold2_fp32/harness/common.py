"""Shared helpers for the ESMFold2 fp32 reference harness.

The reference is the Biohub transformers fork, loaded in **fp32 on CPU** with the
pure-PyTorch (no TE / no flash / no Triton) path so it is deterministic and
matchable by the Rust port. Triton kernels are disabled in this venv (see the
neutered kernels/__init__.py), so the model uses kernel_backend=None.
"""
from __future__ import annotations
import os, sys, time, contextlib
import numpy as np
import torch

REPO = "biohub/ESMFold2"
FIX = os.path.join(os.path.dirname(__file__), "..", "fixtures")

# Four test proteins (incl flgM, UniProt P26477). Short → long.
PROTEINS = {
    "crambin46":  "TTCCPSIVARSNFNVCRLPGTPEALCATYTGCIIIPGATCPGDYAN",
    "ubiquitin76":"MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG",
    "flgM97":     "MSIDRTSPLKPVSTVQTRETSDTPVQKTRQEKTSAATSASVTLSDAQAKLMQPGVSDINLERVEALKTAIRNGELKMDTGKIADSLIKEAESYLQGK".replace(" ", ""),
    "trxa109":    "SDKIIHLTDDSFDTDVLKADGAILVDFWAEWCGPCKMIAPILDEIADEYQGKLTVAKLNIDQNPGTAPKYGIRGIPTLLLFKNGEVAATKVGALSKGQLKEFLDANLA",
}

def load_model(fp32=True, device="cpu"):
    from transformers.models.esmfold2.modeling_esmfold2 import ESMFold2Model
    t0 = time.time()
    dtype = torch.float32 if fp32 else torch.bfloat16
    model = ESMFold2Model.from_pretrained(
        REPO, dtype=dtype, low_cpu_mem_usage=True,
        esmc_precision="fp32" if fp32 else "bf16",
    )
    model = model.to(device).eval()
    model.set_kernel_backend(None)   # reference path, pure PyTorch
    model.set_chunk_size(None)       # no chunking for small L (faster, identical)
    sys.stderr.write(f"[load] {time.time()-t0:.1f}s\n"); sys.stderr.flush()
    return model

def features(seq, device="cpu"):
    from transformers.models.esmfold2 import protein_utils as pu
    feats = pu.prepare_protein_features(seq)
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in feats.items()}

def savez(name, **arrays):
    os.makedirs(FIX, exist_ok=True)
    path = os.path.join(FIX, name)
    np.savez(path, **{k: (v.detach().cpu().numpy() if torch.is_tensor(v) else np.asarray(v))
                      for k, v in arrays.items()})
    sys.stderr.write(f"[saved] {path}.npz ({len(arrays)} arrays)\n"); sys.stderr.flush()
