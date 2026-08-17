#!/usr/bin/env python3
"""Re-compare every saved benchmark case, order-insensitively where that is the
honest thing to do.

Why not a plain byte-diff: `dev/idealize_backbone.rewrite` recovers the ligand
list from a Python `set`, so the ORDER of the HETATM block in the reference's own
output follows CPython's string hashing, not RFdiffusion2. For {NAD,OXM} that
happens to be input order; for {ZN,DUC} it is not. Comparing the block
positionally then reports a 3.26 A "error" for ligand atoms that are in fact
bit-identical. So ligand atoms are matched on (residue name, atom name), and
CONECT records are compared as a SET of bonds after mapping serial numbers
through each file's own atom table.

Emits one TSV row per case to stdout.
"""
import glob, os, sys, hashlib

COLS = ["case", "protein", "ligands", "contig", "length", "T", "extra",
        "n_prot", "n_lig", "L_tokens", "bytes_identical", "identical_mod_ligorder",
        "prot_exact", "prot_max_d", "lig_exact", "lig_max_d", "conect_match",
        "lig_order_same", "t_ref", "t_rs", "speedup", "sha_ref", "sha_rs"]

def atoms(lines, tag):
    return [l for l in lines if l.startswith(tag)]
def xyz(l):
    return [float(l[30+8*i:38+8*i]) for i in range(3)]
def key(l):
    return (l[17:20].strip(), l[12:16].strip(), l[22:27].strip())
def serial_map(lines):
    m = {}
    for l in lines:
        if l.startswith(("ATOM", "HETATM")):
            m[int(l[6:11])] = key(l)
    return m
def bonds(lines):
    sm, out = serial_map(lines), set()
    for l in lines:
        if l.startswith("CONECT"):
            f = [l[i:i+5].strip() for i in range(6, min(len(l.rstrip()), 31), 5)]
            f = [int(x) for x in f if x]
            if not f or f[0] not in sm: continue
            for o in f[1:]:
                if o in sm:
                    out.add(frozenset((sm[f[0]], sm[o])))
    return out

def designs(case_dir):
    """Every (ref, port) design pair in a case dir — num_designs>1 writes more
    than one, and the later ones exercise the per-design reseed."""
    out = []
    for fa in sorted(glob.glob(os.path.join(case_dir, "ref", "design_*-atomized-bb-False.pdb"))):
        fb = fa.replace("/ref/", "/rs/")
        if os.path.exists(fb):
            idx = os.path.basename(fa).split("_")[1].split("-")[0]
            out.append((idx, fa, fb))
    return out

def one(case_dir, meta, fa=None, fb=None):
    if fa is None:
        fa = os.path.join(case_dir, "ref", "design_0-atomized-bb-False.pdb")
        fb = os.path.join(case_dir, "rs", "design_0-atomized-bb-False.pdb")
    if not (os.path.exists(fa) and os.path.exists(fb)):
        return None
    A, B = open(fa, "rb").read(), open(fb, "rb").read()
    la, lb = A.decode().splitlines(), B.decode().splitlines()
    pa, pb = atoms(la, "ATOM"), atoms(lb, "ATOM")
    ha, hb = atoms(la, "HETATM"), atoms(lb, "HETATM")

    # protein: positional (same order both sides by construction)
    pe = sum(a == b for a, b in zip(pa, pb))
    pmax = max((max(abs(p-q) for p, q in zip(xyz(a), xyz(b))) for a, b in zip(pa, pb)),
               default=0.0)
    # ligand: matched on identity, because the block order is a hash artifact
    mb = {key(l): l for l in hb}
    le, lmax, unmatched = 0, 0.0, 0
    for l in ha:
        o = mb.get(key(l))
        if o is None:
            unmatched += 1; continue
        if l[30:54] == o[30:54]: le += 1
        lmax = max(lmax, max(abs(p-q) for p, q in zip(xyz(l), xyz(o))))
    ba, bb = bonds(la), bonds(lb)
    lig_order_same = [key(l) for l in ha] == [key(l) for l in hb]
    # byte-identity after canonicalising the ligand block order
    canon = lambda lines: "\n".join(
        atoms(lines, "ATOM") + sorted(atoms(lines, "HETATM"), key=key))
    return dict(zip(COLS, [
        meta["case"], meta["protein"], meta["ligands"], meta["contig"], meta["length"],
        meta["T"], meta["extra"], len(pa), len(ha),
        # L as the NETWORK sees it: one token per residue plus one per ligand
        # atom (not one per protein atom, which is what the .pdb line count is)
        len({l[21] + l[22:27] for l in pa}) + len(ha),
        "YES" if A == B else "no",
        "YES" if canon(la) == canon(lb) and ba == bb else "no",
        f"{pe}/{len(pa)}", f"{pmax:.4f}",
        f"{le}/{len(ha)}" + (f" (+{unmatched} unmatched)" if unmatched else ""),
        f"{lmax:.4f}",
        f"{len(ba & bb)}/{len(ba)}",
        "YES" if lig_order_same else "no",
        meta.get("t_ref", "-"), meta.get("t_rs", "-"), meta.get("speedup", "-"),
        hashlib.sha256(A).hexdigest()[:12], hashlib.sha256(B).hexdigest()[:12],
    ]))

def main():
    cases = {}
    src = ["bench/cases.tsv"] + [f for f in ("bench/cases_extra.tsv", "bench/cases_daily.tsv")
                                  if os.path.exists(f)]
    for line in [l for f in src for l in open(f).read().splitlines()[1:]]:
        if not line.strip(): continue
        c, p, l, g, ln, T, x = line.split("\t")
        cases[c] = dict(case=c, protein=p, ligands=l, contig=g, length=ln, T=T, extra=x)
    # carry timings over from the first-pass table where present
    if os.path.exists("bench/results.tsv"):
        hdr = None
        for line in open("bench/results.tsv").read().splitlines():
            f = line.split("\t")
            if hdr is None: hdr = f; continue
            if len(f) == len(hdr) and f[0] in cases:
                d = dict(zip(hdr, f))
                cases[f[0]].update(t_ref=d.get("t_ref", "-"), t_rs=d.get("t_rs", "-"),
                                   speedup=d.get("speedup", "-"))
    out = [COLS]
    for c, meta in cases.items():
        ds = designs(f"bench/runs/{c}")
        if len(ds) > 1:
            for idx, fa, fb in ds:
                m = dict(meta, case=f"{c}#d{idx}")
                r = one(f"bench/runs/{c}", m, fa, fb)
                if r: out.append([str(r[k]) for k in COLS])
        else:
            r = one(f"bench/runs/{c}", meta)
            if r: out.append([str(r[k]) for k in COLS])
    for row in out:
        print("\t".join(row))

main()
