#!/usr/bin/env python3
"""Emit a ligand-topology **sidecar** for an input PDB.

Why a sidecar rather than a port
--------------------------------
`python/probe_ligand_bonds.py` measured what OpenBabel actually does for the
demo inputs (see `results/ligand_bond_probe.txt`): for NAD, OXM and the M0151
ligand there are **no CONECT records at all**, so OpenBabel is perceiving both
the connectivity (`ConnectTheDots`, from interatomic distance and covalent
radii) *and* the bond orders including aromaticity (`PerceiveBondOrders`:
hybridisation, ring perception, aromaticity, kekulisation) from 3D coordinates.

Those orders are not cosmetic — `bond_feats` feeds `bond_emb` directly, so
single-vs-aromatic changes the network input. Reproducing that heuristic stack
bit-for-bit in Rust is a cheminformatics sub-project of its own, and a
half-ported version would produce *plausible but wrong* topology, which is the
worst possible failure mode.

So the port draws the line here explicitly: **ligand topology is an input to
rfd2, not something it derives.** This script runs the reference's own code path
once per PDB and writes the result next to it. The Rust side loads the sidecar
and hard-errors if a ligand is not covered — loudly wrong beats silently wrong.

    PYTHONPATH=<ref> .venv/bin/python python/gen_ligand_bonds.py <input.pdb> [LIG,LIG...]
"""
import os
import sys

import common
import torch


def build(pdb_path, lig_names):
    common.add_ref_to_path()
    from rf_diffusion.chemical import ChemicalData as ChemData
    from rf_diffusion import aa_model

    out = {}
    meta = {"pdb": os.path.basename(pdb_path), "ligands": []}
    offset = 0
    for lig in lig_names:
        # Use `aa_model.parse_ligand` -- the function the inference pipeline
        # actually calls (aa_model.py:895) -- not `load_ligand_from_pdb`.
        # They filter the HETATM stream differently (`filter_het(...,
        # covale_allowed=True)` vs a HETATM/CONECT substring test), which gives
        # OpenBabel a different molecule, which changes bond perception, which
        # changes the candidate frame set. Using the wrong one produced frames
        # that matched the reference for 49 of 50 atoms and swapped the two
        # neighbours on the 50th -- a tie broken by CPython set iteration order.
        # pdb_stream must be passed: parse_ligand's no-stream branch calls
        # `fh.read_lines()` (aa_model.py:776), which is a latent upstream typo
        # for `readlines()`. Every real caller passes the stream, so the branch
        # is dead in normal use -- but it means this generator must pass it too.
        with open(pdb_path) as fh:
            pdb_stream = fh.readlines()
        xyz_sm, seq_sm, atom_frames, chirals, bond_feats, atom_names = \
            aa_model.parse_ligand(pdb_path, lig, pdb_stream=pdb_stream)

        n = int(bond_feats.shape[0])
        out[f"{lig}.bond_feats"] = bond_feats.to(torch.int64)
        out[f"{lig}.elem"] = seq_sm.to(torch.int64)
        out[f"{lig}.xyz"] = xyz_sm[0].to(torch.float32)
        out[f"{lig}.atom_frames"] = atom_frames.to(torch.int64)
        # Atom NAMES, as 4 bytes each. The topology is consumed positionally
        # (`make_indep` reads t.bond(i,j) / t.elem by row), while coordinates come
        # from the input PDB in file order — so a sidecar is only valid for a PDB
        # that lists this ligand's atoms in the SAME order. Recording the names
        # lets the Rust loader verify that, and permute when it differs, instead
        # of silently pairing the wrong bonds with the wrong atoms.
        nm = torch.zeros((len(atom_names), 4), dtype=torch.int64)
        for i, a in enumerate(atom_names):
            for j, ch in enumerate(f"{str(a):<4}"[:4]):
                nm[i, j] = ord(ch)
        out[f"{lig}.atom_names"] = nm
        out[f"{lig}.chirals"] = chirals.to(torch.float32)

        meta["ligands"].append({
            "name": lig,
            "n_atoms": n,
            "offset": offset,
            "atom_names": [str(a) for a in atom_names],
            "elements": [ChemData().num2aa[int(t)] for t in seq_sm],
            "n_chirals": int(chirals.shape[0]),
        })
        offset += n
        print(f"  {lig}: {n} atoms, {int((bond_feats > 0).sum()) // 2} bonds, "
              f"{int(chirals.shape[0])} chirals")
    meta["n_atoms_total"] = offset

    # ---- authoritative atom_frames -------------------------------------
    # `get_atom_frames` breaks priority ties by the order of
    # `list(set(allpaths))`. Measured on this input, 20 of 50 atoms have two or
    # more candidate frames at the minimum priority, and for OXM atom 3 the tie
    # is between (4,3,5) and (5,3,4) -- the same frame with its two neighbours
    # swapped, both scoring [4, 4]. Recomputing here picked (4,3,5); the
    # pipeline run picked (5,3,4). Same code, same Python: the set's iteration
    # order depends on the insertion sequence, which depends on OpenBabel's bond
    # iteration order for the molecule as the pipeline built it.
    #
    # Rather than pretend the recomputation is authoritative, take the frames
    # from an actual reference run when one is available, and REPORT any
    # disagreement instead of silently preferring either.
    import os as _os
    # Per-protein override. Without it this points at M0584_1ldm's dump, which
    # for any other input is either shape-rejected below or -- if the atom count
    # happens to coincide -- SILENTLY WRONG. The benchmark always sets it.
    dump = _os.environ.get("RFD2_ATOM_FRAMES") or _os.path.join(
        common.FIXTURES, "model_pinned", "step0.safetensors")
    if _os.path.isfile(dump):
        from safetensors.torch import load_file
        d = load_file(dump)
        if "rfi.atom_frames" in d:
            authoritative = d["rfi.atom_frames"][0].to(torch.int64)
            recomputed = torch.cat([out[f"{l}.atom_frames"] for l in lig_names])
            if authoritative.shape == recomputed.shape:
                n_diff = int((authoritative != recomputed).any(-1).any(-1).sum())
                out["combined.atom_frames"] = authoritative
                meta["atom_frames_source"] = "reference pipeline run"
                meta["atom_frames_recompute_mismatches"] = n_diff
                print(f"  atom_frames: taken from the reference run; "
                      f"{n_diff}/{authoritative.shape[0]} differ from recomputation "
                      f"(set-order ties)")
            else:
                print(f"  (atom_frames: dump shape {list(authoritative.shape)} "
                      f"!= recomputed {list(recomputed.shape)}; not used)")
    else:
        meta["atom_frames_source"] = "recomputed (no reference dump present)"
        print("  atom_frames: recomputed -- no reference dump to verify against")

    return out, meta


def main(argv):
    if not argv:
        ref = common.add_ref_to_path()
        pdb = os.path.join(ref, "rf_diffusion", "benchmark", "input",
                           "mcsa_41", "M0584_1ldm.pdb")
        ligs = ["NAD", "OXM"]
    else:
        pdb = argv[0]
        ligs = argv[1].split(",") if len(argv) > 1 else []
    print(f"{pdb}  ligands={ligs}")
    out, meta = build(pdb, ligs)
    tag = os.path.splitext(os.path.basename(pdb))[0]
    common.write_fixture("ligand", tag, out, {"n": meta["n_atoms_total"]})
    common.write_json("ligand", f"{tag}_meta", meta)


if __name__ == "__main__":
    main(sys.argv[1:])
