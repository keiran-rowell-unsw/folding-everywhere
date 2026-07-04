"""Compare two PDB files atom-for-atom (matched by residue index + atom name).
Reports raw RMSD (no superposition), Kabsch-superposed RMSD, CA RMSD, max dev."""
import sys
import numpy as np


def parse(path):
    d = {}
    for line in open(path):
        if not line.startswith("ATOM"):
            continue
        f = line.split()
        # ATOM serial name resName chain resSeq x y z occ b element
        name = f[2]
        resseq = int(f[5])
        x, y, z = float(f[6]), float(f[7]), float(f[8])
        d[(resseq, name)] = (x, y, z)
    return d


def kabsch_rmsd(P, Q):
    Pc = P - P.mean(0)
    Qc = Q - Q.mean(0)
    H = Pc.T @ Qc
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1, 1, d])
    R = Vt.T @ D @ U.T
    Pr = Pc @ R.T
    return float(np.sqrt(((Pr - Qc) ** 2).sum(1).mean()))


def main():
    a, b = sys.argv[1], sys.argv[2]  # rust, pytorch
    da, db = parse(a), parse(b)
    keys = sorted(set(da) & set(db))
    if not keys:
        print("NO COMMON ATOMS")
        sys.exit(2)
    P = np.array([da[k] for k in keys])
    Q = np.array([db[k] for k in keys])
    per = np.sqrt(((P - Q) ** 2).sum(1))
    raw = float(np.sqrt((per ** 2).mean()))
    kab = kabsch_rmsd(P, Q)
    ca_idx = [i for i, k in enumerate(keys) if k[1] == "CA"]
    ca = float(np.sqrt((per[ca_idx] ** 2).mean())) if ca_idx else float("nan")
    print(f"matched atoms      : {len(keys)} (rust {len(da)}, pytorch {len(db)})")
    print(f"raw RMSD (no super): {raw:.5f} A")
    print(f"Kabsch RMSD        : {kab:.5f} A")
    print(f"CA RMSD (raw)      : {ca:.5f} A")
    print(f"max atom deviation : {per.max():.5f} A")
    # verdict
    ok = kab < 0.5 and raw < 1.0
    print("VERDICT:", "MATCH" if ok else "MISMATCH")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
