"""Element-wise compare two [n_atoms,3] coord npy files (same atom order), like
results/esmfold2_config_sweep/compare_sweep.py. Prints raw RMSD, max dev, Kabsch.
Usage: python cmp_coords.py <label> <a.npy> <b.npy>
"""
import sys, numpy as np

def kabsch(P, Q):
    Pc = P - P.mean(0); Qc = Q - Q.mean(0); H = Pc.T @ Qc
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    return float(np.sqrt(((Pc @ R.T - Qc) ** 2).sum(1).mean()))

label, fa, fb = sys.argv[1], sys.argv[2], sys.argv[3]
a = np.load(fa).reshape(-1, 3); b = np.load(fb).reshape(-1, 3)
d = np.sqrt(((a - b) ** 2).sum(1))
raw = float(np.sqrt((d ** 2).mean()))
print(f"{label:24s} n={a.shape[0]:4d}  raw_RMSD={raw:.6e} A  max={d.max():.6e} A  Kabsch={kabsch(a,b):.6e} A")
