#!/usr/bin/env python3
"""Rung 4c — fixtures for `parsers.parse_pdb_lines_target`.

Parsed on the real demo inputs, including the ligand-bearing one, so the fixture
exercises HETATM handling, insertion-free numbering, and the duplicate-residue
path rather than a toy file.

    PYTHONPATH=<ref> .venv/bin/python python/gen_parse_fixtures.py
"""
import json
import os

import common
import numpy as np
import torch

CASES = [
    ("mcsa41", "rf_diffusion/benchmark/input/mcsa_41/M0584_1ldm.pdb"),
    ("1qys", "rf_diffusion/test_data/1qys.pdb"),
    ("ec1", "rf_diffusion/benchmark/input/mcsa_41/M0151_1q0n.pdb"),
]


def main():
    ref = common.add_ref_to_path()
    from rf_diffusion.parsers import parse_pdb_lines_target

    out = {}
    meta = {}
    for tag, rel in CASES:
        path = os.path.join(ref, rel)
        if not os.path.isfile(path):
            print(f"  (skip {tag}: {path} missing)")
            continue
        with open(path) as fh:
            lines = fh.readlines()
        tf = parse_pdb_lines_target(lines, parse_hetatom=True)

        out[f"{tag}.xyz"] = torch.from_numpy(tf["xyz"])
        out[f"{tag}.mask"] = torch.from_numpy(tf["mask"].astype(np.int64))
        out[f"{tag}.idx"] = torch.from_numpy(tf["idx"].astype(np.int64))
        out[f"{tag}.seq"] = torch.from_numpy(tf["seq"].astype(np.int64))
        if len(tf.get("xyz_het", [])) > 0:
            out[f"{tag}.xyz_het"] = torch.from_numpy(
                np.asarray(tf["xyz_het"], dtype=np.float32))
        meta[tag] = {
            "path": rel,
            "n_res": int(tf["xyz"].shape[0]),
            "n_het": int(len(tf.get("info_het", []))),
            "pdb_idx": [[c, int(i)] for c, i in tf["pdb_idx"]],
            "info_het": [
                {k: (int(v) if isinstance(v, (int, np.integer)) else str(v))
                 for k, v in d.items()}
                for d in tf.get("info_het", [])
            ],
        }
        print(f"  {tag}: {tf['xyz'].shape[0]} residues, "
              f"{len(tf.get('info_het', []))} hetatoms")

    common.write_fixture("parse", "parse", out, {"cases": len(meta)})
    common.write_json("parse", "parse_meta", meta)


if __name__ == "__main__":
    main()
