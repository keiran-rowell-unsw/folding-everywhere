#!/usr/bin/env python3
"""Export the openfold/AlphaFold2 residue constants that `compute_backbone` uses.

`frame_diffusion/data/all_atom.py` builds four module-level tables out of
`residue_constants` at import time:

    DEFAULT_FRAMES  restype_rigid_group_default_frame      [21, 8, 4, 4]
    IDEALIZED_POS   restype_atom14_rigid_group_positions   [21, 14, 3]
    ATOM_MASK       restype_atom14_mask                    [21, 14]
    GROUP_IDX       restype_atom14_to_rigid_group          [21, 14]

`compute_backbone` indexes all four with `aatype`, and on the inference path
`aatype` is *always* zero (`torch.zeros(bb_rigids.shape).long()`) — the backbone
it builds is alanine's. The port still carries all 21 rows so a future
configuration that passes a real sequence does not silently index row 0.

These are ordinary AF2 constants, not something the reference computes from the
checkpoint, but they are exported rather than retyped for the same reason the
chemical tables are: 2 688 hand-copied floats is 2 688 chances to be one digit
wrong, and the error would surface as a plausible-looking backbone.

    PYTHONPATH=<ref> .venv/bin/python python/gen_openfold.py
"""
import os
import sys

os.environ.setdefault("PYTORCH_JIT", "0")

import common  # noqa: E402
import torch  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)


def main():
    import pinned
    print(f"PINNED: {len(pinned.enable())} entry points")
    common.add_ref_to_path()

    from rf_diffusion.frame_diffusion.data import all_atom
    from openfold.utils import rigid_utils as ru
    import torch as _t

    tensors = {
        "default_frames": all_atom.DEFAULT_FRAMES,
        "idealized_pos14": all_atom.IDEALIZED_POS,
        "atom_mask14": all_atom.ATOM_MASK,
        "group_idx14": all_atom.GROUP_IDX,
        # The two quaternion constant tables. `quat_to_rot` and `quat_multiply`
        # are written as a masked sum against these rather than as algebra, and
        # the sum is f64-pinned — so the *table* has to be the reference's, not
        # an equivalent one written out by hand.
        "qtr_mat": _t.as_tensor(ru._QTR_MAT, dtype=_t.float32),
        "quat_multiply": _t.as_tensor(ru._QUAT_MULTIPLY, dtype=_t.float32),
    }
    clean = {}
    for k, v in tensors.items():
        v = v.detach().cpu().contiguous()
        if v.dtype in (torch.int8, torch.int16, torch.int32, torch.bool,
                       torch.uint8):
            v = v.to(torch.int64)
        clean[k] = v
        print(f"  {k:<16} {tuple(v.shape)} {v.dtype}")

    from safetensors.torch import save_file
    data_dir = os.path.join(common.PORT_ROOT, "rfd2", "data")
    os.makedirs(data_dir, exist_ok=True)
    blob = os.path.join(data_dir, "openfold.safetensors")
    save_file(clean, blob)
    n = sum(t.numel() for t in clean.values())
    print(f"  wrote {blob}  ({len(clean)} tables, {n} values, "
          f"{os.path.getsize(blob)} B)")

    # The two rows the inference path actually reads, printed so a reader can
    # sanity-check the Rust side without opening the blob.
    print("\nALA (aatype 0):")
    print(f"  group_idx14  {all_atom.GROUP_IDX[0].tolist()}")
    print(f"  atom_mask14  {all_atom.ATOM_MASK[0].tolist()}")


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
