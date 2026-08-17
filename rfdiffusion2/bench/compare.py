#!/usr/bin/env python3
"""Compare one benchmark case's two .pdb files; print a single TSV row."""
import sys, hashlib
case, prot, ligs, contig, length, T, extra, fa, fb, t_ref, t_rs = sys.argv[1:12]
A = open(fa, 'rb').read(); B = open(fb, 'rb').read()
la = A.decode().splitlines(); lb = B.decode().splitlines()

def xyz(l):
    return [float(l[30+8*i:38+8*i]) for i in range(3)]

if len(la) != len(lb):
    print(f"{case}\tLINECOUNT_MISMATCH\t{len(la)} vs {len(lb)}"); sys.exit(0)
same = sum(a == b for a, b in zip(la, lb))
atoms = [(a, b) for a, b in zip(la, lb) if a.startswith(('ATOM', 'HETATM'))]
prot_rows = [(a, b) for a, b in atoms if a.startswith('ATOM')]
lig_rows = [(a, b) for a, b in atoms if a.startswith('HETATM')]
mx = lambda rows: max((max(abs(p-q) for p, q in zip(xyz(a), xyz(b))) for a, b in rows), default=0.0)
con = [(a, b) for a, b in zip(la, lb) if a.startswith('CONECT')]
L = len(atoms)
print("\t".join(str(x) for x in [
    case, prot, ligs, contig, length, T, extra, L,
    len(prot_rows), len(lig_rows), len(la), same,
    "YES" if A == B else "no",
    f"{mx(atoms):.4f}", f"{mx(prot_rows):.4f}", f"{mx(lig_rows):.4f}",
    f"{sum(a==b for a,b in con)}/{len(con)}",
    f"{float(t_ref):.1f}", f"{float(t_rs):.1f}",
    f"{float(t_ref)/max(float(t_rs),1e-9):.2f}",
    hashlib.sha256(A).hexdigest()[:12], hashlib.sha256(B).hexdigest()[:12],
]))
