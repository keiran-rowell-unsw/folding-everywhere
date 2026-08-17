"""Peak resident memory of both implementations, measured per process.

Kept separate from run_benchmark.py because `resource.getrusage(RUSAGE_CHILDREN)`
reports the maximum over *all* children the process has ever reaped, so once one
large child has run the per-child delta reads as zero. `/usr/bin/time -f %M`
measures each process independently and is what this uses.

Two sequences per structure is enough: peak RSS is set by model + activation
buffers, not by how many sequences are sampled.
"""
import csv
import os
import subprocess
import sys

from common import REF, REPO, RESULTS

RUST_BIN = os.path.join(REPO, "target", "release", "mpnn")
WEIGHTS = os.path.join(REF, "vanilla_model_weights", "v_48_020.pt")


def peak_rss_mb(cmd, env=None):
    p = subprocess.run(
        ["/usr/bin/time", "-f", "%M", *cmd],
        capture_output=True, text=True, env=env or os.environ,
    )
    if p.returncode != 0:
        raise RuntimeError(f"{cmd[:2]} failed: {p.stderr[-800:]}")
    return int(p.stderr.strip().splitlines()[-1]) / 1024.0


def main():
    pdb_dir = os.path.join(RESULTS, "pdb")
    pdbs = sorted(os.path.join(pdb_dir, f) for f in os.listdir(pdb_dir) if f.endswith(".pdb"))
    rows = []
    for i, pdb in enumerate(pdbs, 1):
        name = os.path.splitext(os.path.basename(pdb))[0]
        out = os.path.join(RESULTS, "_runs", "mem", name)
        os.makedirs(out, exist_ok=True)
        t = peak_rss_mb([
            sys.executable, os.path.join(REF, "protein_mpnn_run.py"),
            "--pdb_path", pdb, "--out_folder", out,
            "--num_seq_per_target", "2", "--sampling_temp", "0.1",
            "--seed", "37", "--batch_size", "1",
        ])
        r = peak_rss_mb([
            RUST_BIN, "--pdb", pdb, "--weights", WEIGHTS, "--out", os.devnull,
            "--num_seq_per_target", "2", "--seed", "37", "--quiet",
        ])
        n_ca = sum(1 for line in open(pdb, "rb")
                   if line[:4] == b"ATOM" and line[12:16].strip() == b"CA")
        rows.append(dict(name=name, L=n_ca, torch_rss_mb=round(t, 1), rust_rss_mb=round(r, 1)))
        print(f"[{i}/{len(pdbs)}] {name:6s} pytorch={t:6.0f} MB   rust={r:6.0f} MB")

    path = os.path.join(RESULTS, "memory.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    tm = sorted(r["torch_rss_mb"] for r in rows)[len(rows) // 2]
    rm = sorted(r["rust_rss_mb"] for r in rows)[len(rows) // 2]
    print(f"\nmedian peak RSS: PyTorch {tm:.0f} MB, Rust {rm:.0f} MB ({tm/rm:.1f}x)")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
