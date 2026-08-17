"""Re-run the 10-protein benchmark with the STANDALONE Rust fold (fold_standalone:
bare sequence + seed 0, no fixtures/PyTorch). 3 variants: pt_fp32, rust (standalone),
pt_bf16 -- each under /usr/bin/time -v. Records time, peak RSS, pLDDT/pTM, and Kabsch
RMSD vs pt_fp32. Writes results/metrics.csv + results/accuracy.csv.
Usage: python run_benchmark_standalone.py [prot1 ...]
"""
import os, sys, re, json, subprocess, csv, numpy as np
from proteins10 import PROTEINS10

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
FIX = os.path.join(ROOT, "fixtures")
RES = os.path.join(ROOT, "results"); os.makedirs(RES, exist_ok=True)
VENV_PY = os.path.join(ROOT, "..", "esmfold2_venv", "bin", "python")
FOLD_SA = os.path.join(ROOT, "target", "release", "fold_standalone")

def timed(cmd, cwd):
    p = subprocess.run(["/usr/bin/time", "-v"] + cmd, cwd=cwd, capture_output=True, text=True)
    rss_kb = 0
    m = re.search(r"Maximum resident set size \(kbytes\): (\d+)", p.stderr)
    if m: rss_kb = int(m.group(1))
    js = None
    for line in p.stdout.splitlines():
        s = line.strip()
        if s.startswith("{"):
            try: js = json.loads(s)
            except Exception: pass
    return js, rss_kb / 1024.0, p.returncode, p.stderr

def kabsch_rmsd(P, Q, mask):
    P = P[mask]; Q = Q[mask]
    Pc = P - P.mean(0); Qc = Q - Q.mean(0); H = Pc.T @ Qc
    U, S, Vt = np.linalg.svd(H); d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T; diff = Pc @ R.T - Qc
    return float(np.sqrt((diff ** 2).sum() / len(P))), float(np.abs(diff).max())

def main():
    prots = sys.argv[1:] if len(sys.argv) > 1 else list(PROTEINS10.keys())
    mcsv = os.path.join(RES, "metrics.csv"); acsv = os.path.join(RES, "accuracy.csv")
    with open(mcsv, "w", newline="") as mf, open(acsv, "w", newline="") as af:
        mw = csv.writer(mf); aw = csv.writer(af)
        mw.writerow(["protein", "L", "variant", "fold_s", "peak_rss_mb", "plddt_mean", "ptm", "complex_plddt"])
        aw.writerow(["protein", "L", "rust_vs_fp32_rmsd_A", "rust_vs_fp32_max_A", "bf16_vs_fp32_rmsd_A", "bf16_vs_fp32_max_A"])
        for p in prots:
            print(f"### {p} ###", flush=True)
            seq = PROTEINS10[p]
            # 1. pt_fp32 (generates ref coords + mask fixtures)
            js, rss, rc, err = timed([VENV_PY, "dump_fold.py", p], HERE)
            if js: mw.writerow([p, js["L"], "pt_fp32", js["fold_s"], round(rss, 1), js["plddt_mean"], js["ptm"], js["complex_plddt"]])
            print(f"  pt_fp32: {js['fold_s'] if js else '?'}s {rss:.0f}MB", flush=True)
            # 2. rust STANDALONE (bare sequence + seed 0)
            out = f"{FIX}/fold_{p}_standalone_coords.npy"
            js, rss, rc, err = timed([FOLD_SA, seq, "0", out], ROOT)
            if js: mw.writerow([p, js["L"], "rust_fp32", js["fold_s"], round(rss, 1), js["plddt_mean"], js["ptm"], js["complex_plddt"]])
            print(f"  rust_standalone: {js['fold_s'] if js else '?'}s {rss:.0f}MB", flush=True)
            if not js: print("  RUST ERR:", err[-400:], flush=True)
            # 3. pt_bf16
            js, rss, rc, err = timed([VENV_PY, "bench_pytorch.py", p, "bf16"], HERE)
            if js: mw.writerow([p, js["L"], "pt_bf16", js["fold_s"], round(rss, 1), js["plddt_mean"], js["ptm"], js["complex_plddt"]])
            print(f"  pt_bf16: {js['fold_s'] if js else '?'}s {rss:.0f}MB", flush=True)
            mf.flush()
            # accuracy (standalone + bf16 vs pt_fp32)
            try:
                ref = np.load(f"{FIX}/fold_{p}_ref_coords.npy")
                rust = np.load(out)
                bf16 = np.load(f"{FIX}/fold_{p}_pt_bf16_coords.npy")
                mask = np.load(f"{FIX}/fold_{p}_ie_atom_attention_mask.npy").reshape(-1).astype(bool)
                rr, rm = kabsch_rmsd(rust, ref, mask); br, bm = kabsch_rmsd(bf16, ref, mask)
                aw.writerow([p, len(seq), round(rr, 6), round(rm, 4), round(br, 5), round(bm, 4)]); af.flush()
                print(f"  RMSD rust-vs-fp32={rr*1000:.4f} mA, bf16-vs-fp32={br:.4f} A", flush=True)
            except Exception as e:
                print(f"  accuracy skip: {e}", flush=True)
    print("BENCHMARK DONE", flush=True)

if __name__ == "__main__":
    main()
