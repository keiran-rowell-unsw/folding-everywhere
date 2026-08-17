"""Randomly select benchmark structures from the PDB and download them.

Selection is a uniform draw over *all* entries matching the filter, not over the
first page of results (which the RCSB API returns in ID order and would bias the
sample toward early identifiers). Reproducible: the draw is seeded.

Filter: single protein entity, single deposited chain, X-ray, < 2.0 A, 60-250
residues — i.e. ordinary well-resolved monomers, the regime ProteinMPNN is
normally used in.
"""
import json
import os
import random
import sys
import urllib.request

from common import RESULTS

SEARCH = "https://search.rcsb.org/rcsbsearch/v2/query"
DOWNLOAD = "https://files.rcsb.org/download/{}.pdb"
N_TARGET = 20
SEED = 20240804

QUERY = {
    "type": "group",
    "logical_operator": "and",
    "nodes": [
        {"type": "terminal", "service": "text", "parameters": {
            "attribute": "rcsb_entry_info.polymer_entity_count_protein",
            "operator": "equals", "value": 1}},
        {"type": "terminal", "service": "text", "parameters": {
            "attribute": "rcsb_assembly_info.polymer_monomer_count",
            "operator": "range", "value": {"from": 60, "to": 250}}},
        {"type": "terminal", "service": "text", "parameters": {
            "attribute": "exptl.method",
            "operator": "exact_match", "value": "X-RAY DIFFRACTION"}},
        {"type": "terminal", "service": "text", "parameters": {
            "attribute": "rcsb_entry_info.resolution_combined",
            "operator": "less", "value": 2.0}},
        {"type": "terminal", "service": "text", "parameters": {
            "attribute": "rcsb_entry_info.deposited_polymer_entity_instance_count",
            "operator": "equals", "value": 1}},
    ],
}


def search(start, rows):
    body = {
        "query": QUERY,
        "return_type": "entry",
        "request_options": {
            "paginate": {"start": start, "rows": rows},
            "results_content_type": ["experimental"],
        },
    }
    req = urllib.request.Request(
        SEARCH, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=90) as r:
        return json.load(r)


def main():
    out_dir = os.path.join(RESULTS, "pdb")
    os.makedirs(out_dir, exist_ok=True)

    total = search(0, 1)["total_count"]
    print(f"matching entries in the PDB: {total}")

    rng = random.Random(SEED)
    offsets = rng.sample(range(total), min(4 * N_TARGET, total))  # extras for failures

    chosen, meta = [], []
    for off in offsets:
        if len(chosen) >= N_TARGET:
            break
        pdb_id = search(off, 1)["result_set"][0]["identifier"]
        if pdb_id in chosen:
            continue
        path = os.path.join(out_dir, f"{pdb_id}.pdb")
        if not os.path.exists(path):
            try:
                with urllib.request.urlopen(DOWNLOAD.format(pdb_id), timeout=90) as r:
                    data = r.read()
            except Exception as e:  # entry may be PDB-format-unavailable (large/obsolete)
                print(f"  skip {pdb_id}: {e}")
                continue
            with open(path, "wb") as f:
                f.write(data)
        n_ca = sum(
            1 for line in open(path, "rb")
            if line[:4] == b"ATOM" and line[12:16].strip() == b"CA"
        )
        if n_ca < 40:
            print(f"  skip {pdb_id}: only {n_ca} CA atoms")
            os.remove(path)
            continue
        chosen.append(pdb_id)
        meta.append({"pdb_id": pdb_id, "offset": off, "n_ca": n_ca})
        print(f"  [{len(chosen):2d}/{N_TARGET}] {pdb_id}  CA={n_ca}")

    if len(chosen) < N_TARGET:
        print(f"WARNING: only got {len(chosen)}/{N_TARGET}", file=sys.stderr)

    with open(os.path.join(RESULTS, "proteins.json"), "w") as f:
        json.dump({"seed": SEED, "total_matching": total, "proteins": meta}, f, indent=1)
    print(f"\nselected {len(chosen)} structures -> {out_dir}")


if __name__ == "__main__":
    main()
