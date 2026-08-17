"""Compare two PDBs atom-for-atom, matching by (residue-ORDER-index, atom-name).
Using residue *order* (0..L-1 by first appearance) rather than the literal resSeq
means a constant numbering offset (e.g. Rust ESMFold2 starts residues at 0, the
reference at 1) does not affect the match. Reports raw RMSD, Kabsch-superposed
RMSD, CA RMSD, and max atom deviation.

Usage: python compare_by_order.py <rust.pdb> <reference.pdb>
"""
import sys
import numpy as np


def parse(path):
    atoms = {}          # (res_order, atom_name) -> (x,y,z)
    order = {}          # resSeq -> order index
    for line in open(path):
        if not line.startswith("ATOM"):
            continue
        name = line[12:16].strip()
        resseq = int(line[22:26])
        x = float(line[30:38]); y = float(line[38:46]); z = float(line[46:54])
        if resseq not in order:
            order[resseq] = len(order)
        atoms[(order[resseq], name)] = (x, y, z)
    return atoms, len(order)


def kabsch_rmsd(P, Q):
    Pc = P - P.mean(0); Qc = Q - Q.mean(0)
    H = Pc.T @ Qc
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    return float(np.sqrt(((Pc @ R.T - Qc) ** 2).sum(1).mean()))


def main():
    ra, na = parse(sys.argv[1])   # rust
    rb, nb = parse(sys.argv[2])   # reference
    keys = sorted(set(ra) & set(rb))
    if not keys:
        print("NO COMMON ATOMS"); sys.exit(2)
    P = np.array([ra[k] for k in keys])
    Q = np.array([rb[k] for k in keys])
    per = np.sqrt(((P - Q) ** 2).sum(1))
    raw = float(np.sqrt((per ** 2).mean()))
    kab = kabsch_rmsd(P, Q)
    ca = [i for i, k in enumerate(keys) if k[1] == "CA"]
    ca_rmsd = float(np.sqrt((per[ca] ** 2).mean())) if ca else float("nan")
    print(f"residues           : rust {na}, reference {nb}")
    print(f"atoms              : rust {len(ra)}, reference {len(rb)}, matched {len(keys)}")
    print(f"unmatched          : rust-only {len(ra)-len(keys)}, ref-only {len(rb)-len(keys)}")
    print(f"raw RMSD (no super) : {raw*1000:.4f} mA  ({raw:.6f} A)")
    print(f"Kabsch RMSD         : {kab*1000:.4f} mA  ({kab:.6f} A)")
    print(f"CA RMSD (raw)       : {ca_rmsd*1000:.4f} mA  ({ca_rmsd:.6f} A)")
    print(f"mean atom dev       : {per.mean()*1000:.4f} mA")
    print(f"max atom deviation  : {per.max()*1000:.4f} mA  ({per.max():.6f} A)")
    ok = raw < 0.01 and per.max() < 0.05   # fp32 round-off floor: <10 mA RMSD, <50 mA max
    print("VERDICT:", "AGREE (fp32 round-off)" if ok else
          ("CLOSE" if kab < 0.5 else "MISMATCH"))


if __name__ == "__main__":
    main()
