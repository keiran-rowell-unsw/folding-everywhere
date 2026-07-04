"""Export residue_constants tables + atom14/atom37 maps needed by the structure
module / all-atom reconstruction, as an fp32+int safetensors the Rust binary reads.
"""
import os

import numpy as np
import torch

from common import FIX
from transformers.models.esm.openfold_utils import residue_constants as rc


def main():
    out = {}
    # rigid group default frames [21,8,4,4], literature atom positions [21,14,3]
    out["restype_rigid_group_default_frame"] = np.asarray(
        rc.restype_rigid_group_default_frame, dtype=np.float32
    )
    out["restype_atom14_to_rigid_group"] = np.asarray(
        rc.restype_atom14_to_rigid_group, dtype=np.float32
    )
    out["restype_atom14_mask"] = np.asarray(rc.restype_atom14_mask, dtype=np.float32)
    out["restype_atom14_rigid_group_positions"] = np.asarray(
        rc.restype_atom14_rigid_group_positions, dtype=np.float32
    )

    # atom14<->atom37 maps via make_atom14_masks-equivalent (build directly from rc)
    restype_atom14_to_atom37 = []
    restype_atom37_to_atom14 = []
    restype_atom14_mask = []
    for rt in rc.restypes:  # 20, in rc order
        atom_names = rc.restype_name_to_atom14_names[rc.restype_1to3[rt]]
        restype_atom14_to_atom37.append([rc.atom_order[n] if n else 0 for n in atom_names])
        atom_name_to_idx14 = {n: i for i, n in enumerate(atom_names)}
        restype_atom37_to_atom14.append(
            [atom_name_to_idx14.get(n, 0) for n in rc.atom_types]
        )
        restype_atom14_mask.append([1.0 if n else 0.0 for n in atom_names])
    # unknown restype (index 20) -> zeros
    restype_atom14_to_atom37.append([0] * 14)
    restype_atom37_to_atom14.append([0] * 37)
    restype_atom14_mask.append([0.0] * 14)

    out["restype_atom14_to_atom37"] = np.asarray(restype_atom14_to_atom37, dtype=np.float32)
    out["restype_atom37_to_atom14"] = np.asarray(restype_atom37_to_atom14, dtype=np.float32)
    out["restype_atom14_exists"] = np.asarray(restype_atom14_mask, dtype=np.float32)
    restype_atom37_mask = np.zeros((21, 37), dtype=np.float32)
    for i, rt in enumerate(rc.restypes):
        for n in rc.residue_atoms[rc.restype_1to3[rt]]:
            restype_atom37_mask[i, rc.atom_order[n]] = 1.0
    out["restype_atom37_mask"] = restype_atom37_mask

    os.makedirs(os.path.join(FIX, "constants"), exist_ok=True)
    from safetensors.numpy import save_file

    path = os.path.join(FIX, "constants", "residue_constants.safetensors")
    save_file({k: np.ascontiguousarray(v) for k, v in out.items()}, path)
    print("wrote", path)
    for k, v in out.items():
        print(f"  {k:42s} {v.shape} {v.dtype}")
    print("atom_types(37):", rc.atom_types)
    print("restypes(20):", "".join(rc.restypes), " restype_order_with_x X->", rc.restype_order_with_x.get("X"))


if __name__ == "__main__":
    main()
