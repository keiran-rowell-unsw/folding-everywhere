"""Run PyTorch ProteinMPNN and the Rust port on the same structures and compare.

Both are driven through their *public CLIs* (`protein_mpnn_run.py` and `mpnn`) at
the same seed, so this measures exactly what a user would get — not a
hand-wired comparison that could quietly share state.

Recorded per protein: length, wall time, the native score, and for every sampled
sequence whether the two implementations agree residue-for-residue and how far
apart their reported scores are.

Peak memory is *not* measured here — `getrusage(RUSAGE_CHILDREN)` reports the
maximum over all children ever reaped, so per-child deltas read as zero once a
larger child has run. See measure_memory.py, which uses `/usr/bin/time -f %M`.
"""
import argparse
import json
import os
import re
import resource
import shutil
import subprocess
import sys
import time

import numpy as np

from common import REF, REPO, RESULTS

RUST_BIN = os.path.join(REPO, "target", "release", "mpnn")
WEIGHTS = os.path.join(REF, "vanilla_model_weights", "v_48_020.pt")

HDR = re.compile(r"score=([0-9.eE+-]+)")


def parse_fasta(path):
    """-> (native_seq, native_score, [(seq, score), ...])"""
    heads, seqs = [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            (heads if line.startswith(">") else seqs).append(line)
    native_score = float(HDR.search(heads[0]).group(1))
    samples = [(seqs[i], float(HDR.search(heads[i]).group(1))) for i in range(1, len(heads))]
    return seqs[0], native_score, samples


def run(cmd, cwd=None):
    """Run a subprocess, returning (wall seconds, peak child RSS in MB)."""
    before = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    t0 = time.perf_counter()
    p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    dt = time.perf_counter() - t0
    after = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    if p.returncode != 0:
        raise RuntimeError(f"{cmd[:3]} failed:\n{p.stdout[-2000:]}\n{p.stderr[-2000:]}")
    # ru_maxrss is the max over all children so far; the delta is a lower bound
    # on this child's peak, which is what we want when it is the largest so far.
    return dt, max(after - before, 0) / 1024.0, p.stderr


def bench_one(pdb, n_seq, temp, seed, workdir):
    name = os.path.splitext(os.path.basename(pdb))[0]

    # ---- PyTorch reference -------------------------------------------------
    # Timed twice, because the two numbers answer different questions:
    #   * 1 thread  — the apples-to-apples per-core comparison, and the setting
    #     the reference harness pins for determinism;
    #   * default threads — what a user actually gets when they run the script.
    # Reporting only the first would flatter the Rust port.
    out_t = os.path.join(workdir, "torch", name)
    os.makedirs(out_t, exist_ok=True)
    t_cmd = [
        sys.executable, os.path.join(REF, "protein_mpnn_run.py"),
        "--pdb_path", pdb, "--out_folder", out_t,
        "--num_seq_per_target", str(n_seq), "--sampling_temp", str(temp),
        "--seed", str(seed), "--batch_size", "1",
    ]
    env1 = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    t0 = time.perf_counter()
    p = subprocess.run(t_cmd, capture_output=True, text=True, env=env1)
    t_time = time.perf_counter() - t0
    if p.returncode != 0:
        raise RuntimeError(f"pytorch failed on {name}:\n{p.stdout[-1500:]}\n{p.stderr[-1500:]}")
    t_fa = os.path.join(out_t, "seqs", f"{name}.fa")

    envN = {k: v for k, v in os.environ.items()
            if k not in ("OMP_NUM_THREADS", "MKL_NUM_THREADS")}
    t0 = time.perf_counter()
    pN = subprocess.run(t_cmd, capture_output=True, text=True, env=envN)
    t_time_mt = time.perf_counter() - t0
    if pN.returncode != 0:
        t_time_mt = float("nan")

    # ---- Rust port ---------------------------------------------------------
    r_fa = os.path.join(workdir, "rust", f"{name}.fa")
    os.makedirs(os.path.dirname(r_fa), exist_ok=True)
    r_cmd = [
        RUST_BIN, "--pdb", pdb, "--weights", WEIGHTS, "--out", r_fa,
        "--num_seq_per_target", str(n_seq), "--sampling_temp", str(temp),
        "--seed", str(seed),
    ]
    r_time, _r_rss_unused, _r_err = run(r_cmd)

    tn_seq, t_native, t_samples = parse_fasta(t_fa)
    rn_seq, r_native, r_samples = parse_fasta(r_fa)

    assert tn_seq == rn_seq, f"{name}: native sequences differ"
    assert len(t_samples) == len(r_samples)
    L = len(tn_seq)

    identical = sum(1 for (a, _), (b, _) in zip(t_samples, r_samples) if a == b)
    per_res = [
        sum(1 for x, y in zip(a, b) if x == y) / max(len(a), 1)
        for (a, _), (b, _) in zip(t_samples, r_samples)
    ]
    dscore = [abs(sa - sb) for (_, sa), (_, sb) in zip(t_samples, r_samples)]
    recov = [
        sum(1 for x, y in zip(tn_seq, a) if x == y) / max(L, 1) for a, _ in t_samples
    ]

    return dict(
        name=name, L=L, n_seq=len(t_samples), temp=temp, seed=seed,
        torch_time=round(t_time, 3), torch_time_mt=round(t_time_mt, 3),
        rust_time=round(r_time, 3),
        torch_native_score=t_native, rust_native_score=r_native,
        native_score_absdiff=abs(t_native - r_native),
        seqs_identical=identical,
        mean_seq_identity=float(np.mean(per_res)),
        min_seq_identity=float(np.min(per_res)),
        max_score_absdiff=float(np.max(dscore)) if dscore else 0.0,
        mean_score_absdiff=float(np.mean(dscore)) if dscore else 0.0,
        mean_recovery=float(np.mean(recov)),
        torch_ms_per_seq=round(1000 * t_time / max(len(t_samples), 1), 1),
        torch_mt_ms_per_seq=round(1000 * t_time_mt / max(len(t_samples), 1), 1),
        rust_ms_per_seq=round(1000 * r_time / max(len(r_samples), 1), 1),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num_seq", type=int, default=8)
    ap.add_argument("--temp", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=37)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    if not os.path.exists(RUST_BIN):
        sys.exit(f"build the Rust binary first: cargo build --release ({RUST_BIN} missing)")

    pdbs = sorted(
        os.path.join(RESULTS, "pdb", f)
        for f in os.listdir(os.path.join(RESULTS, "pdb"))
        if f.endswith(".pdb")
    )
    if args.limit:
        pdbs = pdbs[: args.limit]

    workdir = os.path.join(RESULTS, "_runs")
    shutil.rmtree(workdir, ignore_errors=True)
    os.makedirs(workdir, exist_ok=True)

    rows = []
    for i, pdb in enumerate(pdbs, 1):
        name = os.path.splitext(os.path.basename(pdb))[0]
        print(f"[{i}/{len(pdbs)}] {name} ...", end=" ", flush=True)
        try:
            r = bench_one(pdb, args.num_seq, args.temp, args.seed, workdir)
        except Exception as e:
            print(f"FAILED: {e}")
            continue
        rows.append(r)
        print(
            f"L={r['L']:4d} identical={r['seqs_identical']}/{r['n_seq']} "
            f"dscore={r['max_score_absdiff']:.2e} "
            f"torch1={r['torch_time']:.1f}s torchMT={r['torch_time_mt']:.1f}s "
            f"rust={r['rust_time']:.1f}s"
        )

    import csv
    out_csv = os.path.join(RESULTS, "metrics.csv")
    with open(out_csv, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    with open(os.path.join(RESULTS, "metrics.json"), "w") as f:
        json.dump(rows, f, indent=1)

    tot_seqs = sum(r["n_seq"] for r in rows)
    tot_ident = sum(r["seqs_identical"] for r in rows)
    print(f"\n{len(rows)} proteins, {tot_seqs} sequences")
    print(f"  sequences identical to PyTorch : {tot_ident}/{tot_seqs} "
          f"({100 * tot_ident / max(tot_seqs, 1):.2f}%)")
    print(f"  max |score difference|         : {max(r['max_score_absdiff'] for r in rows):.3e}")
    import statistics as _st
    print(f"  median speedup vs 1-thread torch: "
          f"{_st.median(r['torch_time'] / r['rust_time'] for r in rows):.2f}x")
    print(f"  median speedup vs default torch : "
          f"{_st.median(r['torch_time_mt'] / r['rust_time'] for r in rows):.2f}x")
    print(f"  wrote {out_csv}")


if __name__ == "__main__":
    main()
