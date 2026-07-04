"""Compare pure-Rust fp32 vs PyTorch fp32 coordinates across the config sweep.

Reads pt_<name>_l<L>_s<S>.npy and rust_<name>_l<L>_s<S>.npy from a directory, computes the
all-atom deviation per config, and writes sweep.csv. Both sides are single-sample folds at
seed 0, so they share a frame — a direct (unaligned) comparison is the bit-exactness metric
(agreement to ~1e-4 A = fp32 round-off).

Usage: python compare_sweep.py <name> <npy_dir> <out_csv>
"""
from __future__ import annotations
import sys, os, csv, numpy as np

CONFIGS = [(3, 14), (6, 14), (10, 14), (3, 28), (3, 42), (6, 28)]  # (num_loops, num_sampling_steps)

def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "crambin46"
    d = sys.argv[2] if len(sys.argv) > 2 else "/tmp/sweep"
    out_csv = sys.argv[3] if len(sys.argv) > 3 else os.path.join(d, "sweep.csv")
    rows = []
    print(f"{'loops':>5} {'steps':>5}  {'RMSD (A)':>12} {'max dev (A)':>12}  verdict")
    for (l, s) in CONFIGS:
        tag = f"{name}_l{l}_s{s}"
        pt = np.load(os.path.join(d, f"pt_{tag}.npy"))
        ru = np.load(os.path.join(d, f"rust_{tag}.npy"))
        m = min(len(pt), len(ru))
        dev = np.linalg.norm(pt[:m] - ru[:m], axis=1)
        rmsd, mx = float(np.sqrt((dev ** 2).mean())), float(dev.max())
        ok = mx < 1e-3                      # fp32 round-off floor
        rows.append([l, s, m, f"{rmsd:.3e}", f"{mx:.3e}", "bit-exact(fp32)" if ok else "MISMATCH"])
        print(f"{l:>5} {s:>5}  {rmsd:>12.3e} {mx:>12.3e}  {'OK' if ok else 'FAIL'}")
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["num_loops", "num_sampling_steps", "n_atoms",
                    "rust_vs_pt_rmsd_A", "rust_vs_pt_max_A", "verdict"])
        w.writerows(rows)
    print(f"\nwrote {out_csv}")

if __name__ == "__main__":
    main()
