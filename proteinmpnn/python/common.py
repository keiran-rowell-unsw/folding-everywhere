"""Shared helpers for the PyTorch fp32 reference harness.

Everything is pinned to deterministic, single-thread fp32 so the reference is
reproducible run-to-run and forms a stable bit-target for the Rust port.

`REF` points at a clone of https://github.com/dauparas/ProteinMPNN — the harness
imports the *unmodified* upstream `protein_mpnn_utils` so the reference can never
silently drift from the published model.
"""
import glob
import json
import os
import sys

import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("MKL_CBWR", "COMPATIBLE")

import torch  # noqa: E402  (must follow the thread-pinning env vars)

torch.set_num_threads(1)
torch.manual_seed(0)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIX = os.path.join(REPO, "fixtures")
RESULTS = os.path.join(REPO, "results")
REF = os.environ.get("PROTEINMPNN_REF", os.path.join(os.path.dirname(REPO), "ref_ProteinMPNN"))

if REF not in sys.path:
    sys.path.insert(0, REF)


def weights_path(model_name="v_48_020", kind="vanilla"):
    p = os.path.join(REF, f"{kind}_model_weights", f"{model_name}.pt")
    if not os.path.exists(p):
        raise FileNotFoundError(f"ProteinMPNN weights not found: {p}")
    return p


def load_model(model_name="v_48_020", kind="vanilla", backbone_noise=0.0, ca_only=False):
    """Instantiate upstream ProteinMPNN exactly as protein_mpnn_run.py does."""
    from protein_mpnn_utils import ProteinMPNN

    ckpt = torch.load(weights_path(model_name, kind), map_location="cpu", weights_only=False)
    hidden_dim, num_layers = 128, 3
    model = ProteinMPNN(
        ca_only=ca_only,
        num_letters=21,
        node_features=hidden_dim,
        edge_features=hidden_dim,
        hidden_dim=hidden_dim,
        num_encoder_layers=num_layers,
        num_decoder_layers=num_layers,
        augment_eps=backbone_noise,
        k_neighbors=ckpt["num_edges"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, ckpt


def save_fixture(subdir, name, tensors: dict):
    """Save a dict of (name -> array/tensor) as a safetensors file.

    fp32 stays fp32; integer arrays are written as int64 so the Rust side can read
    indices (E_idx, decoding_order, S) back exactly rather than through a float.
    """
    from safetensors.numpy import save_file

    out_dir = os.path.join(FIX, subdir)
    os.makedirs(out_dir, exist_ok=True)
    np_tensors = {}
    for k, v in tensors.items():
        if isinstance(v, torch.Tensor):
            v = v.detach().contiguous().cpu().numpy()
        v = np.asarray(v)
        if v.dtype.kind in "iub":
            np_tensors[k] = np.ascontiguousarray(v, dtype=np.int64)
        else:
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


def pdb_inputs():
    """All PDBs shipped with the upstream repo (used to pick benchmark targets)."""
    pats = [
        os.path.join(REF, "inputs", "**", "*.pdb"),
        os.path.join(REPO, "results", "pdb", "*.pdb"),
    ]
    out = []
    for p in pats:
        out.extend(glob.glob(p, recursive=True))
    return sorted(out)
