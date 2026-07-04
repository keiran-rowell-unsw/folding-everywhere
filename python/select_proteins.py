"""Fetch a diverse candidate set, keep those with length in [40,130], pick 15."""
import os, sys
import urllib.request
from common import REPO

# (name, UniProt accession) — diverse folds/organisms; full canonical sequences.
CANDIDATES = [
    ("crambin", "P01542"), ("flgM", "P26477"), ("thioredoxin", "P0AA25"),
    ("cytochrome_c", "P00004"), ("sumo1", "P63165"), ("histone_h4", "P62805"),
    ("acylphosphatase", "P14621"), ("b2_microglobulin", "P61769"), ("bpti", "P00974"),
    ("insulin", "P01308"), ("ubiquitin", "P0CG48"), ("calmodulin", "P0DP23"),
    ("hemoglobin_a", "P69905"), ("hemoglobin_b", "P68871"), ("myoglobin", "P02144"),
    ("lysozyme_hew", "P00698"), ("lysozyme_human", "P61626"), ("ribonuclease_a", "P61823"),
    ("protein_g", "P06654"), ("cold_shock", "P0A9Y6"), ("fkbp12", "P62942"),
    ("cyclophilin_a", "P62937"), ("ww_domain", "P46937"), ("villin_hp", "P14923"),
]

def fetch(acc):
    url = f"https://rest.uniprot.org/uniprotkb/{acc}.fasta"
    try:
        with urllib.request.urlopen(url, timeout=20) as r:
            lines = r.read().decode().strip().splitlines()
        return "".join(lines[1:]).replace(" ", "")
    except Exception as e:
        return None

def main():
    chosen = []
    for name, acc in CANDIDATES:
        seq = fetch(acc)
        if seq is None:
            print(f"  skip {name} {acc}: fetch failed"); continue
        L = len(seq)
        ok = 40 <= L <= 130
        print(f"  {'KEEP' if ok else 'drop'} {name:18s} {acc} L={L}")
        if ok:
            chosen.append((name, acc, seq))
        if len(chosen) >= 15:
            break
    out = os.path.join(REPO, "results", "proteins.fasta")
    with open(out, "w") as f:
        for name, acc, seq in chosen:
            f.write(f">{name} {acc} L={len(seq)}\n{seq}\n")
    print(f"\nselected {len(chosen)} proteins -> {out}")

if __name__ == "__main__":
    main()
