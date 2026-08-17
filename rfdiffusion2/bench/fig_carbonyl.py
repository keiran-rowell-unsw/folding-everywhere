#!/usr/bin/env python3
"""Figure 6 — where the residual actually lives: one backbone carbonyl oxygen.

Every case in the benchmark that is not byte-identical has its largest
disagreement on a backbone carbonyl O, with that residue's CA bit-identical.
O is the only backbone atom placed through the psi torsion rather than directly
by the residue frame, so it is the most angularly sensitive atom in the chain.
"""
import math, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "results/figures"; os.makedirs(OUT, exist_ok=True)
TEAL, AMBER, INK, GRID = "#0F7268", "#9C540B", "#1B242E", "#D5DBE3"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": INK,
    "axes.labelcolor": INK, "text.color": INK, "xtick.color": INK, "ytick.color": INK,
    "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 170,
    "savefig.bbox": "tight", "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": .6, "axes.axisbelow": True})

sub = lambda a, b: [x - y for x, y in zip(a, b)]
dot = lambda a, b: sum(x * y for x, y in zip(a, b))
mag = lambda v: math.sqrt(dot(v, v))
unit = lambda v: [x / mag(v) for x in v]

def res_atoms(path):
    out = {}
    for l in open(path):
        if l.startswith("ATOM"):
            out.setdefault(int(l[22:26]), {})[l[12:16].strip()] = \
                [float(l[30 + 8 * i:38 + 8 * i]) for i in range(3)]
    return out

def dihedral(p0, p1, p2, p3):
    b0, b1, b2 = sub(p0, p1), sub(p2, p1), sub(p3, p2)
    b1 = unit(b1)
    v = [a - b * dot(b0, b1) for a, b in zip(b0, b1)]
    w = [a - b * dot(b2, b1) for a, b in zip(b2, b1)]
    cr = [b1[1]*v[2]-b1[2]*v[1], b1[2]*v[0]-b1[0]*v[2], b1[0]*v[1]-b1[1]*v[0]]
    return math.degrees(math.atan2(dot(cr, w), dot(v, w)))

CASES = ["p01_L30", "du_mm4_prod", "len25_L101", "len40_L131",
         "du_T100_mid", "p09_L82", "cfg_T10s"]
pts = []
for c in CASES:
    A = res_atoms(f"bench/runs/{c}/ref/design_0-atomized-bb-False.pdb")
    B = res_atoms(f"bench/runs/{c}/rs/design_0-atomized-bb-False.pdb")
    worst = (0, None, None)
    for r in A:
        for n in A[r]:
            if n in B.get(r, {}):
                m = mag(sub(A[r][n], B[r][n]))
                if m > worst[0]: worst = (m, r, n)
    m, r, n = worst
    d = abs((dihedral(*[B[r][k] for k in "N CA C O".split()])
             - dihedral(*[A[r][k] for k in "N CA C O".split()]) + 180) % 360 - 180)
    pts.append((c, r, n, m, d))

fig = plt.figure(figsize=(11.6, 3.6))
gs = fig.add_gridspec(1, 3, width_ratios=[1, 1.05, 1.15], wspace=.34)

# ---- A: per-atom displacement in the offending residue ---------------------
A = res_atoms("bench/runs/p01_L30/ref/design_0-atomized-bb-False.pdb")[1]
B = res_atoms("bench/runs/p01_L30/rs/design_0-atomized-bb-False.pdb")[1]
order = [a for a in ("N", "CA", "C", "O", "CB") if a in A and a in B]
disp = [mag(sub(A[a], B[a])) for a in order]
ax = fig.add_subplot(gs[0])
ax.bar(range(len(order)), disp, color=[AMBER if a == "O" else TEAL for a in order], width=.6)
ax.set_xticks(range(len(order))); ax.set_xticklabels(order, fontfamily="monospace")
ax.set_ylabel("displacement, ref vs port (Å)")
ax.set_title("A  M0636_1uaq residue 1", loc="left", fontweight="bold", fontsize=9.5)
for i, (a, v) in enumerate(zip(order, disp)):
    ax.text(i, v + max(disp) * .03, "0.000" if v < 5e-5 else f"{v:.3f}",
            ha="center", fontsize=7, color=AMBER if a == "O" else TEAL)
ax.set_ylim(0, max(disp) * 1.22)
ax.text(.5, .82, "CA and CB are\nbit-identical", transform=ax.transAxes,
        ha="center", fontsize=7.5, color=INK, style="italic")

# ---- B: Newman projection DOWN the CA->C axis ------------------------------
# The carbonyl rotates ABOUT CA->C, so the displacement is perpendicular to the
# CA-C-O plane. Projecting into that plane hides it entirely; looking down the
# bond axis is the view that shows it.
ax = fig.add_subplot(gs[1])
ca, cc = A["CA"], A["C"]
axis = unit(sub(cc, ca))
ref_perp = sub(sub(A["N"], ca), [axis[i] * dot(sub(A["N"], ca), axis) for i in range(3)])
f1 = unit(ref_perp)
f2 = [axis[1]*f1[2]-axis[2]*f1[1], axis[2]*f1[0]-axis[0]*f1[2], axis[0]*f1[1]-axis[1]*f1[0]]
def nproj(p):
    v = sub(p, ca)
    perp = [v[i] - axis[i] * dot(v, axis) for i in range(3)]
    return (dot(perp, f1), dot(perp, f2))
ax.axhline(0, color=GRID, lw=.6); ax.axvline(0, color=GRID, lw=.6)
circ = plt.Circle((0, 0), 1.25, fill=False, color=INK, lw=1.0, alpha=.5)
ax.add_patch(circ)
for lbl, S, col, mk, sz in (("reference", A, AMBER, "o", 70), ("Rust port", B, TEAL, "X", 62)):
    for a in ("N", "CB", "O"):
        if a not in S: continue
        x, y = nproj(S[a])
        ax.scatter([x], [y], s=sz, color=col, marker=mk, zorder=4,
                   label=lbl if a == "N" else None)
        ax.plot([0, x], [0, y], color=col, lw=1.0, alpha=.45, zorder=3)
for a in ("N", "CB", "O"):
    if a in A:
        x, y = nproj(A[a])
        ax.annotate(a, (x, y), textcoords="offset points", xytext=(7, 6),
                    fontsize=8, fontfamily="monospace")
ox, oy = nproj(A["O"]); px, py = nproj(B["O"])
ra = math.hypot(ox, oy)
th0, th1 = math.atan2(oy, ox), math.atan2(py, px)
arc = [(ra * math.cos(th0 + (th1 - th0) * t / 40), ra * math.sin(th0 + (th1 - th0) * t / 40))
       for t in range(41)]
ax.annotate("", xy=arc[-1], xytext=arc[-6],
            arrowprops=dict(arrowstyle="->", color=AMBER, lw=1.8))
ax.plot([q[0] for q in arc], [q[1] for q in arc], color=AMBER, lw=1.8, zorder=5)
ax.text(0, -1.68, f"Δψ = {pts[0][4]:.1f}°  →  O moves {pts[0][3]:.3f} Å",
        fontsize=8.5, color=AMBER, ha="center", fontweight="bold")
ax.scatter([0], [0], s=30, color=INK, zorder=6)
ax.annotate("CA, C\n(on axis)", (0, 0), textcoords="offset points", xytext=(-40, 8),
            fontsize=7, color=INK)
ax.set_aspect("equal"); ax.set_xlim(-1.9, 1.9); ax.set_ylim(-1.95, 1.75)
ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
ax.legend(frameon=False, fontsize=7.5, loc="upper left",
              bbox_to_anchor=(-.02, 1.02))
ax.set_title("B  looking down the CA\u2192C bond", loc="left", fontweight="bold", fontsize=9.5)

# ---- C: displacement tracks the torsion, across every case -----------------
ax = fig.add_subplot(gs[2])
ax.scatter([p[4] for p in pts], [p[3] for p in pts], s=42, color=AMBER, zorder=4)
lo, hi = 0, max(p[4] for p in pts) * 1.12
ax.plot([lo, hi], [0, hi * math.radians(1) * 1.1], color=INK, lw=1, ls="--", zorder=2)
ax.text(hi * .52, hi * math.radians(1) * 1.1 * .40,
        "lever arm ≈ 1.1 Å\n(O about the CA–C axis)", fontsize=7, color=INK, style="italic")
for c, r, n, m, d in pts:
    if m > .04:
        ax.annotate(c, (d, m), textcoords="offset points", xytext=(-6, 7),
                    fontsize=6.6, ha="right", color=INK)
ax.set_xlabel("Δψ  (N–CA–C–O dihedral, degrees)")
ax.set_ylabel("worst-atom displacement (Å)")
ax.set_title("C  all 7 non-identical cases", loc="left", fontweight="bold", fontsize=9.5)
ax.set_xlim(-.4, hi)

fig.savefig(f"{OUT}/fig6_carbonyl_oxygen.png"); plt.close(fig)
print("wrote fig6_carbonyl_oxygen.png")
for c, r, n, m, d in pts:
    print(f"  {c:14} res {r:<4}{n:3}  {m:.4f} A  Δψ {d:6.3f}°")
