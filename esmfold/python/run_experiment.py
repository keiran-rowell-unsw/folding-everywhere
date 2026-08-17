"""Full benchmark experiment: 15 proteins, PyTorch fp32 vs pure-Rust fp32.
Records wall time + peak RSS for each version, writes both PDBs, computes RMSD,
and saves results/metrics.csv. Resumable (skips proteins already in metrics.csv).
"""
import csv
import json
import os
import re
import subprocess
import sys

import numpy as np
from safetensors.numpy import load_file

from common import FIX, REPO

RESULTS = os.path.join(REPO, "results")
PDB = os.path.join(RESULTS, "pdb")
DUMPS = os.path.join(RESULTS, "dumps")
CSV = os.path.join(RESULTS, "metrics.csv")
RUST_BIN = os.path.join(REPO, "target", "release", "fold")
CONSTS = os.path.join(FIX, "constants", "residue_constants.safetensors")
FIELDS = ["name", "L", "torch_time", "torch_rss_gb", "rust_time", "rust_rss_gb",
          "torch_plddt", "rust_plddt", "torch_ptm", "rust_ptm", "rmsd", "max_dev", "natoms"]


def parse_fasta(path):
    out = []
    name, seq = None, ""
    for line in open(path):
        line = line.strip()
        if line.startswith(">"):
            if name:
                out.append((name, seq))
            name = line[1:].split()[0]
            seq = ""
        else:
            seq += line
    if name:
        out.append((name, seq))
    return out


def timed(cmd):
    p = subprocess.run(["/usr/bin/time", "-v"] + cmd, capture_output=True, text=True)
    err = p.stderr
    rss = re.search(r"Maximum resident set size \(kbytes\):\s*(\d+)", err)
    wall = re.search(r"Elapsed \(wall clock\).*?:\s*([\d:.]+)", err)
    rss_gb = int(rss.group(1)) / 1e6 if rss else float("nan")
    w = float("nan")
    if wall:
        parts = [float(x) for x in wall.group(1).split(":")]
        w = parts[-1] + (parts[-2] * 60 if len(parts) > 1 else 0) + (parts[-3] * 3600 if len(parts) > 2 else 0)
    return w, rss_gb, p.returncode, err


def rmsd(name):
    ref = load_file(os.path.join(FIX, f"bench/{name}", "ref.safetensors"))
    ra = ref["atom37"].reshape(-1, 3)
    ex = ref["atom37_atom_exists"].reshape(-1) > 0.5
    rust = np.fromfile(os.path.join(DUMPS, f"{name}_rust.atom37.f32"), dtype=np.float32).reshape(-1, 3)
    d = ra[ex] - rust[ex]
    per = np.sqrt((d * d).sum(-1))
    return float(np.sqrt((per ** 2).mean())), float(per.max()), int(ex.sum())


def main():
    os.makedirs(PDB, exist_ok=True)
    os.makedirs(DUMPS, exist_ok=True)
    proteins = parse_fasta(os.path.join(RESULTS, "proteins.fasta"))
    done = set()
    rows = []
    if os.path.exists(CSV):
        rows = list(csv.DictReader(open(CSV)))
        done = {r["name"] for r in rows}

    for name, seq in proteins:
        if name in done:
            print(f"skip {name} (done)")
            continue
        L = len(seq)
        print(f"=== {name} L={L} ===", flush=True)

        print("  pytorch...", flush=True)
        tw, trss, rc1, err1 = timed([sys.executable, os.path.join(REPO, "python", "ref_fold.py"),
                                     name, seq, os.path.join(PDB, f"{name}_pytorch.pdb")])
        if rc1 != 0:
            print(f"  PYTORCH FAILED rc={rc1}\n{err1[-600:]}")
            continue
        ref_meta = json.load(open(os.path.join(FIX, f"bench/{name}", "ref_meta.json")))

        print("  rust...", flush=True)
        rw, rrss, rc2, err2 = timed([RUST_BIN, "--seq", seq, "--name", name,
                                     "-o", os.path.join(PDB, f"{name}_rust.pdb"),
                                     "--dump", os.path.join(DUMPS, f"{name}_rust"),
                                     "--constants", CONSTS])
        if rc2 != 0:
            print(f"  RUST FAILED rc={rc2}\n{err2[-600:]}")
            continue
        rust_meta = json.load(open(os.path.join(DUMPS, f"{name}_rust.meta.json")))

        bb, mx, nat = rmsd(name)
        row = dict(name=name, L=L, torch_time=f"{tw:.1f}", torch_rss_gb=f"{trss:.2f}",
                   rust_time=f"{rw:.1f}", rust_rss_gb=f"{rrss:.2f}",
                   torch_plddt=f"{ref_meta['plddt_mean']:.2f}", rust_plddt=f"{rust_meta['plddt_mean']:.2f}",  # 0..100
                   torch_ptm=f"{ref_meta['ptm']:.4f}", rust_ptm=f"{rust_meta['ptm']:.4f}",
                   rmsd=f"{bb:.5f}", max_dev=f"{mx:.5f}", natoms=nat)
        rows.append(row)
        # write incrementally (resumable)
        with open(CSV, "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=FIELDS)
            wr.writeheader()
            wr.writerows(rows)
        print(f"  -> torch {tw:.0f}s/{trss:.1f}GB  rust {rw:.0f}s/{rrss:.1f}GB  RMSD {bb:.4f}A  "
              f"pLDDT {ref_meta['plddt_mean']:.2f}/{rust_meta['plddt_mean']:.2f}  "
              f"pTM {ref_meta['ptm']:.3f}/{rust_meta['ptm']:.3f}", flush=True)

    print(f"\nDONE. {len(rows)} proteins -> {CSV}")


if __name__ == "__main__":
    main()
