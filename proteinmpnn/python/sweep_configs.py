"""Does the Rust port give identical output for identical input, across settings?

The main benchmark pins one configuration (v_48_020, T=0.1, seed 37, monomers).
That is not enough to answer the question in general, so this sweeps the axes a
user would actually vary and diffs the two CLIs' FASTA output at each point:

  * all four vanilla checkpoints
  * a range of sampling temperatures
  * several seeds
  * multi-chain complexes and homo-oligomers
  * designing only a subset of chains (the rest held fixed)
  * several sequences per run, so the RNG stream is exercised at length

Reports, per configuration, how many sampled sequences are identical.
"""
import argparse
import csv
import os
import re
import subprocess
import sys

from common import REF, REPO, RESULTS

RUST_BIN = os.path.join(REPO, "target", "release", "mpnn")
HDR = re.compile(r"score=([0-9.eE+-]+)")

# Two of the 20 benchmark structures, chosen to bracket the length range:
# the shortest (62 residues) and the longest (249). Running the whole matrix on
# all 20 would add nothing — the axes under test here are the model settings,
# not the structure.
BENCH = os.path.join(RESULTS, "pdb")
SHORT = os.path.join(BENCH, "6EKB.pdb")   # L = 62
LONG = os.path.join(BENCH, "7NL3.pdb")    # L = 249

# Multi-chain cases have to come from the reference repo's inputs: every
# structure in the benchmark set is a single-chain monomer by construction.
COMPLEX = os.path.join(REF, "inputs", "PDB_complexes", "pdbs", "3HTN.pdb")
COMPLEX2 = os.path.join(REF, "inputs", "PDB_complexes", "pdbs", "4YOW.pdb")
HOMO = os.path.join(REF, "inputs", "PDB_homooligomers", "pdbs", "4GYT.pdb")


def parse_fasta(path):
    heads, seqs = [], []
    for line in open(path):
        line = line.strip()
        if line:
            (heads if line.startswith(">") else seqs).append(line)
    return seqs[0], [(seqs[i], float(HDR.search(heads[i]).group(1))) for i in range(1, len(heads))]


def run_case(tag, pdb, model, temp, seed, n_seq, chains, workdir):
    name = os.path.splitext(os.path.basename(pdb))[0]
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")

    out_t = os.path.join(workdir, "torch", tag)
    os.makedirs(out_t, exist_ok=True)
    cmd = [
        sys.executable, os.path.join(REF, "protein_mpnn_run.py"),
        "--pdb_path", pdb, "--out_folder", out_t, "--model_name", model,
        "--num_seq_per_target", str(n_seq), "--sampling_temp", str(temp),
        "--seed", str(seed), "--batch_size", "1",
    ]
    if chains:
        cmd += ["--pdb_path_chains", chains]
    p = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if p.returncode != 0:
        return dict(tag=tag, error=(p.stderr or p.stdout)[-300:])

    r_fa = os.path.join(workdir, "rust", f"{tag}.fa")
    os.makedirs(os.path.dirname(r_fa), exist_ok=True)
    rcmd = [
        RUST_BIN, "--pdb", pdb, "--out", r_fa, "--model_name", model,
        "--num_seq_per_target", str(n_seq), "--sampling_temp", str(temp),
        "--seed", str(seed), "--quiet",
    ]
    if chains:
        rcmd += ["--pdb_path_chains", chains]
    r = subprocess.run(rcmd, capture_output=True, text=True)
    if r.returncode != 0:
        return dict(tag=tag, error=r.stderr[-300:])

    try:
        tn, ts = parse_fasta(os.path.join(out_t, "seqs", f"{name}.fa"))
        rn, rs = parse_fasta(r_fa)
    except (IndexError, IOError) as e:
        return dict(tag=tag, error=f"could not parse output: {e}")
    if len(ts) != len(rs):
        return dict(tag=tag, error=f"sequence-count mismatch {len(ts)} vs {len(rs)}")
    ident = sum(1 for (a, _), (b, _) in zip(ts, rs) if a == b)
    dscore = max((abs(x - y) for (_, x), (_, y) in zip(ts, rs)), default=0.0)
    mismatched_res = sum(
        sum(1 for x, y in zip(a, b) if x != y) for (a, _), (b, _) in zip(ts, rs)
    )
    return dict(
        tag=tag, pdb=name, model=model, temp=temp, seed=seed, n_seq=n_seq,
        chains=chains or "all", L=len(tn), native_match=int(tn == rn),
        identical=ident, total=len(ts), mismatched_residues=mismatched_res,
        max_score_absdiff=dscore,
    )


def cases():
    """The full model-setting matrix on two benchmark structures, plus the
    multi-chain paths the monomer benchmark cannot reach."""
    out = []
    for pdb, tag in ((SHORT, "6EKB"), (LONG, "7NL3")):
        # all four published checkpoints
        for m in ["v_48_002", "v_48_010", "v_48_020", "v_48_030"]:
            out.append((f"{tag}_model_{m}", pdb, m, 0.1, 37, 4, None))
        # temperature: higher T flattens the distribution, so the multinomial
        # draw gets more sensitive to any perturbation of the probabilities
        for t in [0.05, 0.2, 0.3, 0.5, 1.0]:
            out.append((f"{tag}_temp_{t}", pdb, "v_48_020", t, 37, 4, None))
        # seeds. 0 is deliberately absent: `protein_mpnn_run.py` does
        # `if args.seed:`, and 0 is falsy in Python, so `--seed 0` means "pick a
        # random seed" — non-reproducible on both sides, so it cannot be an
        # identity test. The Rust CLI matches that semantic.
        for s in [1, 2, 12345, 999999]:
            out.append((f"{tag}_seed_{s}", pdb, "v_48_020", 0.1, s, 4, None))
        # one long RNG stream: 16 sequences drawn from a single generator
        out.append((f"{tag}_many_seqs", pdb, "v_48_020", 0.2, 7, 16, None))

    # multi-chain: complexes, a homo-oligomer, and designing only a subset of
    # chains while the rest are held fixed (exercises the chain ordering,
    # residue_idx offsets and chain_M masking)
    out.append(("complex_3HTN_all", COMPLEX, "v_48_020", 0.1, 37, 4, None))
    out.append(("complex_4YOW_all", COMPLEX2, "v_48_020", 0.2, 5, 4, None))
    out.append(("homooligomer_4GYT", HOMO, "v_48_020", 0.1, 37, 2, None))
    out.append(("complex_3HTN_chainA", COMPLEX, "v_48_020", 0.1, 37, 4, "A"))
    out.append(("complex_3HTN_chainsAB", COMPLEX, "v_48_020", 0.1, 37, 4, "A B"))
    out.append(("complex_4YOW_chainB", COMPLEX2, "v_48_020", 0.1, 11, 4, "B"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--only", default="", help="substring filter on the config tag")
    ap.add_argument("--merge", action="store_true",
                    help="update matching rows in an existing config_sweep.csv "
                         "instead of overwriting the file")
    args = ap.parse_args()

    workdir = os.path.join(RESULTS, "_sweep")
    os.makedirs(workdir, exist_ok=True)
    cs = cases()
    if args.only:
        cs = [c for c in cs if args.only in c[0]]
    if args.limit:
        cs = cs[: args.limit]

    rows, bad = [], 0
    for i, (tag, pdb, model, temp, seed, n, chains) in enumerate(cs, 1):
        r = run_case(tag, pdb, model, temp, seed, n, chains, workdir)
        if "error" in r:
            print(f"[{i}/{len(cs)}] {tag:24s} ERROR {r['error'][:120]}")
            bad += 1
            continue
        ok = r["identical"] == r["total"] and r["native_match"]
        bad += 0 if ok else 1
        rows.append(r)
        print(f"[{i}/{len(cs)}] {tag:24s} {r['pdb']:6s} L={r['L']:4d} "
              f"{'OK ' if ok else 'DIFF'} {r['identical']}/{r['total']} identical  "
              f"max|dscore|={r['max_score_absdiff']:.1e}")

    path = os.path.join(RESULTS, "config_sweep.csv")
    if args.merge and os.path.exists(path):
        # Replace the rows we just re-ran, drop any tag no longer in cases()
        # (e.g. the seed-0 entries), and keep the rest in canonical order.
        prev = {r["tag"]: r for r in csv.DictReader(open(path))}
        prev.update({r["tag"]: {k: str(v) for k, v in r.items()} for r in rows})
        order = [c[0] for c in cases()]
        rows = [prev[t] for t in order if t in prev]
        for r in rows:
            for k in ("identical", "total", "L", "n_seq", "native_match",
                      "mismatched_residues"):
                r[k] = int(float(r[k]))
            r["max_score_absdiff"] = float(r["max_score_absdiff"])
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    tot = sum(int(r["total"]) for r in rows)
    ident = sum(int(r["identical"]) for r in rows)
    print(f"\n{len(rows)} configurations, {tot} sequences")
    print(f"  identical to PyTorch : {ident}/{tot} ({100*ident/max(tot,1):.2f}%)")
    print(f"  configurations with any difference : {bad}")
    print(f"  wrote {path}")


if __name__ == "__main__":
    main()
