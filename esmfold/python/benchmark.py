"""Benchmark: PyTorch fp32 reference vs pure-Rust fp32, on 5 varying-length
proteins (incl flgM). Records wall time + peak RSS for both, and atom37 RMSD.

Run:  python benchmark.py   (after building: cargo build --release --bin fold)
"""
import json
import os
import re
import subprocess
import sys

import numpy as np
from safetensors.numpy import load_file

from common import FIX, REPO

PROTEINS = [
    ("crambin", "TTCCPSIVARSNFNVCRLPGTPEALCATYTGCIIIPGATCPGDYAN"),
    ("ubiquitin", "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"),
    ("flgM", "MSIDRTSPLKPVSTVQTRETSDTPVQKTRQEKTSAATSASVTLSDAQAKLMQPGVSDINMERVEALKTAIRNGELKMDTGKIADSLIREAQSYLQSK"),
    ("trxa", "MSDKIIHLTDDSFDTDVLKADGAILVDFWAEWCGPCKMIAPILDEIADEYQGKLTVAKLNIDQNPGTAPKYGIRGIPTLLLFKNGEVAATKVGALSKGQLKEFLDANLA"),
    ("lysozyme", "KVFGRCELAAAMKRHGLDNYRGYSLGNWVCAAKFESNFNTQATNRNTDGSTDYGILQINSRWWCNDGRTPGSRNLCNIPCSALLSSDITASVNCAKKIVSDGNGMNAWVAWRNRCKGTDVQAWIRGCRL"),
]
RUST_BIN = os.path.join(REPO, "target", "release", "fold")
CONSTS = os.path.join(FIX, "constants", "residue_constants.safetensors")


def timed(cmd, env=None):
    """Run under /usr/bin/time -v; return (wall_s, peak_rss_gb, stdout)."""
    full = ["/usr/bin/time", "-v"] + cmd
    p = subprocess.run(full, capture_output=True, text=True, env={**os.environ, **(env or {})})
    err = p.stderr
    rss = re.search(r"Maximum resident set size \(kbytes\):\s*(\d+)", err)
    wall = re.search(r"Elapsed \(wall clock\).*?:\s*([\d:.]+)", err)
    rss_gb = int(rss.group(1)) / 1e6 if rss else float("nan")
    w = float("nan")
    if wall:
        parts = wall.group(1).split(":")
        w = float(parts[-1]) + (float(parts[-2]) * 60 if len(parts) > 1 else 0) + (float(parts[-3]) * 3600 if len(parts) > 2 else 0)
    if p.returncode != 0:
        print(f"  !! exit {p.returncode}\n{err[-800:]}")
    return w, rss_gb, p.stdout, err


def rmsd(name):
    ref = load_file(os.path.join(FIX, f"bench/{name}", "ref.safetensors"))
    ra = ref["atom37"].reshape(-1, 3)
    ex = ref["atom37_atom_exists"].reshape(-1)
    rust = np.fromfile(os.path.join(FIX, f"bench/{name}", "rust.atom37.f32"), dtype=np.float32).reshape(-1, 3)
    m = ex > 0.5
    d = ra[m] - rust[m]
    return float(np.sqrt((d * d).sum(-1).mean())), float(np.sqrt((d * d).sum(-1)).max()), int(m.sum())


def main():
    rows = []
    for name, seq in PROTEINS:
        L = len(seq)
        print(f"=== {name} L={L} ===", flush=True)
        outdir = os.path.join(FIX, f"bench/{name}")
        os.makedirs(outdir, exist_ok=True)

        print("  pytorch ref...", flush=True)
        tw, trss, tout, _ = timed([sys.executable, os.path.join(REPO, "python", "ref_fold.py"), name, seq])
        ref_meta = json.load(open(os.path.join(outdir, "ref_meta.json")))

        print("  rust...", flush=True)
        rw, rrss, _, _ = timed(
            [RUST_BIN, "--seq", seq, "--name", name, "-o", os.path.join(outdir, "rust.pdb"),
             "--dump", os.path.join(outdir, "rust"), "--constants", CONSTS],
        )
        rust_meta = json.load(open(os.path.join(outdir, "rust.meta.json")))

        bb_rmsd, max_dev, natoms = rmsd(name)
        row = dict(name=name, L=L, torch_t=tw, torch_rss=trss, rust_t=rw, rust_rss=rrss,
                   torch_plddt=ref_meta["plddt_mean"], rust_plddt=rust_meta["plddt_mean"],
                   torch_ptm=ref_meta["ptm"], rust_ptm=rust_meta["ptm"],
                   rmsd=bb_rmsd, max_dev=max_dev, natoms=natoms)
        rows.append(row)
        print(f"  -> torch {tw:.0f}s/{trss:.1f}GB  rust {rw:.0f}s/{rrss:.1f}GB  RMSD {bb_rmsd:.4f}A  pLDDT {ref_meta['plddt_mean']:.2f}/{rust_meta['plddt_mean']:.2f}  pTM {ref_meta['ptm']:.3f}/{rust_meta['ptm']:.3f}", flush=True)

    # markdown table
    md = ["# ESMFold v1 — PyTorch fp32 vs pure-Rust fp32 benchmark\n",
          "| protein | L | PyTorch time | PyTorch peak RSS | Rust time | Rust peak RSS | atom RMSD | max dev | pLDDT (torch/rust) | pTM (torch/rust) |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['name']} | {r['L']} | {r['torch_t']:.0f}s | {r['torch_rss']:.1f} GB | {r['rust_t']:.0f}s | {r['rust_rss']:.1f} GB | {r['rmsd']:.4f} Å | {r['max_dev']:.3f} Å | {r['torch_plddt']:.2f}/{r['rust_plddt']:.2f} | {r['torch_ptm']:.3f}/{r['rust_ptm']:.3f} |")
    # Printed, not written to a file: the benchmark writeup lives in
    # results/esmfold1/README.md and the numbers in results/esmfold1/metrics.csv,
    # so a second copy on disk would only be a third thing to keep in sync.
    out = "\n".join(md) + "\n"
    print("\n" + out)


if __name__ == "__main__":
    main()
