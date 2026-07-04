"""Refill crambin+ubiquitin pt_bf16 (clean timing) + their accuracy rows."""
import os, re, json, subprocess, csv, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, ".."))
FIX = os.path.join(ROOT, "fixtures"); RES = os.path.join(ROOT, "results")
VENV_PY = os.path.join(ROOT, "..", "esmfold2_venv", "bin", "python")

def timed(cmd, cwd):
    p = subprocess.run(["/usr/bin/time", "-v"] + cmd, cwd=cwd, capture_output=True, text=True)
    rss = 0; m = re.search(r"Maximum resident set size \(kbytes\): (\d+)", p.stderr)
    if m: rss = int(m.group(1)) / 1024.0
    js = None
    for line in p.stdout.splitlines():
        s = line.strip()
        if s.startswith("{"):
            try: js = json.loads(s)
            except Exception: pass
    return js, rss

def kabsch(P, Q, mask):
    P = P[mask]; Q = Q[mask]; Pc = P - P.mean(0); Qc = Q - Q.mean(0); H = Pc.T @ Qc
    U, S, Vt = np.linalg.svd(H); d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T; diff = Pc @ R.T - Qc
    return float(np.sqrt((diff ** 2).sum() / len(P))), float(np.abs(diff).max())

for p in ["crambin46", "ubiquitin76"]:
    js, rss = timed([VENV_PY, "bench_pytorch.py", p, "bf16"], HERE)
    if js:
        with open(os.path.join(RES, "metrics.csv"), "a", newline="") as f:
            csv.writer(f).writerow([p, js["L"], "pt_bf16", js["fold_s"], round(rss, 1),
                                    js["plddt_mean"], js["ptm"], js["complex_plddt"]])
        print(f"{p} pt_bf16: {js['fold_s']}s {rss:.0f}MB")
    # accuracy (append rust_vs_fp32 + bf16_vs_fp32)
    ref = np.load(f"{FIX}/fold_{p}_ref_coords.npy"); rust = np.load(f"{FIX}/fold_{p}_rust_coords.npy")
    bf16 = np.load(f"{FIX}/fold_{p}_pt_bf16_coords.npy")
    mask = np.load(f"{FIX}/fold_{p}_ie_atom_attention_mask.npy").reshape(-1).astype(bool)
    rr, rm = kabsch(rust, ref, mask); br, bm = kabsch(bf16, ref, mask)
    with open(os.path.join(RES, "accuracy.csv"), "a", newline="") as f:
        csv.writer(f).writerow([p, len(mask), round(rr, 5), round(rm, 4), round(br, 5), round(bm, 4)])
    print(f"{p} RMSD rust={rr*1000:.3f} mA bf16={br:.3f} A")
print("FILL DONE")
