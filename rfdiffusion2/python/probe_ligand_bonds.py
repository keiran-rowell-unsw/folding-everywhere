#!/usr/bin/env python3
"""How much does OpenBabel actually contribute to ligand bond features?

`load_ligand_from_pdb` hands the HETATM+CONECT block to OpenBabel, which does
two separable jobs:

  1. **connectivity** — which atoms are bonded. CONECT records may already say.
  2. **bond order perception** — single / double / triple / aromatic, inferred
     from geometry and valence when the file does not say.

Porting all of OpenBabel is out of the question; porting (2) for a bounded set
of elements might be fine. So: measure which job it is actually doing on the
demo ligands, before deciding what to port.

    PYTHONPATH=<ref> .venv/bin/python python/probe_ligand_bonds.py
"""
import collections
import os

import common
import torch

CASES = [
    ("mcsa_41/M0584_1ldm.pdb", ["NAD", "OXM"]),
    ("mcsa_41/M0151_1q0n.pdb", None),
    ("trimmed_ec2_M0151_NO_ORI_zero_com0.pdb", ["PH2"]),
]


def conect_graph(path, lig_names, remove_h=True):
    """Connectivity from CONECT records alone, over the ligand's HETATM serials."""
    serial_to_elem = {}
    keep = []
    with open(path) as fh:
        lines = fh.readlines()
    for l in lines:
        if not l.startswith("HETATM"):
            continue
        if lig_names is not None and l[17:20].strip() not in lig_names:
            continue
        elem = l[76:78].strip()
        if remove_h and elem == "H":
            continue
        serial = int(l[6:11])
        serial_to_elem[serial] = elem
        keep.append(serial)

    edges = set()
    n_conect = 0
    for l in lines:
        if not l.startswith("CONECT"):
            continue
        n_conect += 1
        try:
            a = int(l[6:11])
        except ValueError:
            continue
        for st in (11, 16, 21, 26):
            frag = l[st:st + 5].strip()
            if not frag:
                continue
            b = int(frag)
            if a in serial_to_elem and b in serial_to_elem:
                edges.add((min(a, b), max(a, b)))
    return keep, edges, n_conect


def main():
    ref = common.add_ref_to_path()
    from rf_diffusion.parsers import load_ligand_from_pdb

    base = os.path.join(ref, "rf_diffusion", "benchmark", "input")
    for rel, lig_names in CASES:
        path = os.path.join(base, rel)
        if not os.path.isfile(path):
            print(f"(skip {rel}: missing)")
            continue
        print(f"\n=== {rel}  ligands={lig_names} ===")

        names = lig_names or [None]
        for lig in names:
            try:
                mol, xyz, mask, msa, bf, atom_names = load_ligand_from_pdb(
                    path, lig, remove_H=True)
            except SystemExit as e:
                print(f"  {lig}: {e}")
                continue
            n = bf.shape[0]
            ob_edges = set()
            orders = collections.Counter()
            for i in range(n):
                for j in range(i + 1, n):
                    o = int(bf[i, j])
                    if o > 0:
                        ob_edges.add((i, j))
                        orders[o] += 1
            keep, cedges, n_conect = conect_graph(path, [lig] if lig else None)

            print(f"  {lig or '(all)'}: {n} atoms, "
                  f"{len(ob_edges)} bonds from OpenBabel, "
                  f"{len(cedges)} CONECT edges over those atoms "
                  f"({n_conect} CONECT lines in file)")
            print(f"    bond order histogram (1=single 2=double 3=triple 4=aromatic): "
                  f"{dict(sorted(orders.items()))}")

            # do CONECT records alone reproduce the connectivity?
            # map serial -> index in the kept order
            if len(cedges) > 0:
                idx = {s: i for i, s in enumerate(keep)}
                cedge_idx = {(idx[a], idx[b]) if idx[a] < idx[b] else (idx[b], idx[a])
                             for a, b in cedges if a in idx and b in idx}
                same = cedge_idx == ob_edges
                print(f"    CONECT connectivity == OpenBabel connectivity: {same}")
                if not same:
                    only_ob = ob_edges - cedge_idx
                    only_ce = cedge_idx - ob_edges
                    print(f"      only OpenBabel: {len(only_ob)}  only CONECT: {len(only_ce)}")
            else:
                print("    no CONECT records cover this ligand -> connectivity is "
                      "PERCEIVED from geometry")

            elems = collections.Counter(
                common_elem(int(m)) for m in msa)
            print(f"    elements: {dict(elems)}")


def common_elem(tok):
    from rf_diffusion.chemical import ChemicalData as ChemData
    return ChemData().num2aa[tok]


if __name__ == "__main__":
    main()
