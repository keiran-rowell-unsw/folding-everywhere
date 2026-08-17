#!/usr/bin/env python3
"""Figures for the rfdiffusion2-rs benchmark. Reads bench/results_final.tsv."""
import csv, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib.patches import Patch

OUT = "results/figures"; os.makedirs(OUT, exist_ok=True)
TEAL, AMBER, INK, GRID = "#0F7268", "#9C540B", "#1B242E", "#D5DBE3"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": INK,
    "axes.labelcolor": INK, "text.color": INK, "xtick.color": INK, "ytick.color": INK,
    "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 170,
    "savefig.bbox": "tight", "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": .6, "axes.axisbelow": True,
})

rows = []
for r in csv.DictReader(open("bench/results_final.tsv"), delimiter="\t"):
    if not r.get("L_tokens", "").isdigit():
        continue
    r["L"] = int(r["L_tokens"]); r["T"] = int(r["T"])
    for k in ("t_ref", "t_rs", "speedup", "prot_max_d", "lig_max_d"):
        try: r[k] = float(r[k])
        except (TypeError, ValueError): r[k] = float("nan")
    r["ok"] = r["bytes_identical"] == "YES"
    # p07_L69 was a RETRY (its reference had already run, so t_ref timed a
    # partial re-execution); cfg_varlen_s ran under batch contention and is the
    # only sub-1x row. Both are excluded from timing plots, never from parity.
    r["timing_ok"] = r["case"] not in ("p07_L69", "cfg_varlen_s")
    r["pe"] = eval(r["prot_exact"].replace("/", "/"))  # a/b -> fraction
    rows.append(r)
print(f"{len(rows)} completed cases")
short = lambda r: r["protein"].split("_")[1]

core = sorted([r for r in rows if r["case"].startswith("p0") or r["case"] == "p10_L117"],
              key=lambda r: r["L"])
daily = [r for r in rows if r["case"].startswith("du_")]
# ---- Fig 1 : per-protein agreement ----------------------------------------
if core:
    fig, ax = plt.subplots(figsize=(7.6, 3.5))
    x = list(range(len(core)))
    ax.bar(x, [100 * r["pe"] for r in core],
           color=[TEAL if r["ok"] else AMBER for r in core], width=.62)
    ax.set_xticks(x)
    ax.set_xticklabels([f'{short(r)}\nL={r["L"]}' for r in core], fontsize=7)
    lo = min(95, min(100 * r["pe"] for r in core) - 1)
    ax.set_ylim(lo, 100 + (100 - lo) * .16)
    ax.set_ylabel("protein backbone atoms identical (%)")
    ax.set_title("Rust vs pinned PyTorch — 10 proteins, T=2, contig 10,X,10",
                 loc="left", fontweight="bold", fontsize=10)
    for xi, r in zip(x, core):
        ax.text(xi, 100 * r["pe"] + (100 - lo) * .02,
                "byte-identical" if r["ok"] else f'{r["prot_max_d"]:.3f} Å',
                ha="center", fontsize=6.3, color=TEAL if r["ok"] else AMBER)
    # below the axes: every bar reaches ~100 %, so there is no empty space inside
    ax.legend(handles=[Patch(color=TEAL, label="whole file byte-identical"),
                       Patch(color=AMBER, label="residual (max |Δ| labelled)")],
              frameon=False, fontsize=7.5, loc="upper center",
              bbox_to_anchor=(.5, -.16), ncol=2)
    fig.savefig(f"{OUT}/fig1_agreement_by_protein.png"); plt.close(fig)

# ---- Fig 2 : runtime and speedup vs L --------------------------------------
t2 = [r for r in rows if r["T"] == 2 and r["timing_ok"]]
if t2:
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.0, 3.3))
    a1.scatter([r["L"] for r in t2], [r["t_ref"] for r in t2], s=28, color=AMBER,
               label="PyTorch (pinned)", zorder=3)
    a1.scatter([r["L"] for r in t2], [r["t_rs"] for r in t2], s=28, color=TEAL,
               label="Rust rfd2", zorder=3)
    a1.set_xlabel("L (tokens = residues + ligand atoms)")
    a1.set_ylabel("wall clock at T=2 (s)")
    a1.legend(frameon=False, fontsize=8)
    a1.set_title("Runtime", loc="left", fontweight="bold")
    a2.scatter([r["L"] for r in t2], [r["speedup"] for r in t2], s=28, color=TEAL, zorder=3)
    a2.axhline(1.0, color=INK, lw=.8, ls="--")
    a2.set_xlabel("L (tokens)"); a2.set_ylabel("speedup (PyTorch / Rust)")
    a2.yaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v:.1f}×"))
    a2.set_title("Port speedup", loc="left", fontweight="bold")
    fig.savefig(f"{OUT}/fig2_runtime_vs_length.png"); plt.close(fig)

# ---- Fig 3 : length scaling -------------------------------------------------
ls = [r for r in rows if r["case"].startswith("len") or r["case"] in ("p08_L71", "p06_L49")]
if len(ls) > 2:
    fig, ax = plt.subplots(figsize=(6.6, 3.3))
    by = {}
    for r in ls: by.setdefault(short(r), []).append(r)
    for i, (prot, rs) in enumerate(sorted(by.items())):
        rs.sort(key=lambda r: r["L"])
        ax.plot([r["L"] for r in rs], [r["t_ref"] for r in rs], "s--", color=AMBER,
                alpha=.55 + .45 * i, label=f"{prot} — PyTorch")
        ax.plot([r["L"] for r in rs], [r["t_rs"] for r in rs], "o-", color=TEAL,
                alpha=.55 + .45 * i, label=f"{prot} — Rust")
    ax.set_xlabel("L (tokens)"); ax.set_ylabel("wall clock at T=2 (s)")
    ax.set_title("Scaling with designed length", loc="left", fontweight="bold")
    ax.legend(frameon=False, fontsize=7)
    fig.savefig(f"{OUT}/fig3_length_scaling.png"); plt.close(fig)

# ---- Fig 4 : agreement by configuration ------------------------------------
lbl = {"p08_L71": "baseline\nT=2", "cfg_T10": "T=10", "cfg_selfcond": "self-cond",
       "cfg_varlen": "var-length\ncontig", "p06_L49": "baseline\nT=2 (2nd)",
       "cfg_T10s": "T=10\n(2nd)", "cfg_selfcond_s": "self-cond\n(2nd)",
       "cfg_varlen_s": "var-length\n(2nd)"}
cfg = [r for r in rows if r["case"] in lbl]
if cfg:
    order = [c for c in lbl if any(r["case"] == c for r in cfg)]
    cfg = sorted(cfg, key=lambda r: order.index(r["case"]))
    fig, ax = plt.subplots(figsize=(7.4, 3.1))
    x = list(range(len(cfg)))
    ax.bar(x, [100 * r["pe"] for r in cfg],
           color=[TEAL if r["ok"] else AMBER for r in cfg], width=.6)
    ax.set_xticks(x); ax.set_xticklabels([lbl[r["case"]] for r in cfg], fontsize=7)
    lo = min(95, min(100 * r["pe"] for r in cfg) - 1)
    ax.set_ylim(lo, 100 + (100 - lo) * .14)
    ax.set_ylabel("protein atoms identical (%)")
    ax.set_title("Agreement across run configurations", loc="left", fontweight="bold")
    for xi, r in zip(x, cfg):
        ax.text(xi, 100 * r["pe"] + (100 - lo) * .02,
                "byte-identical" if r["ok"] else f'{r["prot_max_d"]:.3f} Å',
                ha="center", fontsize=6.3, color=TEAL if r["ok"] else AMBER)
    fig.savefig(f"{OUT}/fig4_agreement_by_config.png"); plt.close(fig)

# ---- Fig 5 : daily-use configurations --------------------------------------
if daily:
    lab = {"du_mm2": "2-motif contig\nL=82, T=2",
           "du_mm4_prod": "PRODUCTION contig\n4 motifs, 180 res, L=230",
           "du_T100_small": "T=100\nL=31", "du_T100_mid": "T=100\nL=49"}
    daily = [r for r in daily if r["case"] in lab]
    daily.sort(key=lambda r: list(lab).index(r["case"]))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.2, 3.4),
                                 gridspec_kw={"width_ratios": [1.25, 1]})
    x = list(range(len(daily)))
    a1.bar(x, [100 * r["pe"] for r in daily],
           color=[TEAL if r["ok"] else AMBER for r in daily], width=.6)
    a1.set_xticks(x); a1.set_xticklabels([lab[r["case"]] for r in daily], fontsize=6.8)
    a1.set_ylim(90, 101.6); a1.set_ylabel("protein atoms identical (%)")
    a1.set_title("Daily-use configurations", loc="left", fontweight="bold")
    for xi, r in zip(x, daily):
        a1.text(xi, 100 * r["pe"] + .25,
                "byte-identical" if r["ok"] else f'{r["prot_max_d"]:.3f} Å',
                ha="center", fontsize=6.4, color=TEAL if r["ok"] else AMBER)
    a2.bar([i - .19 for i in x], [r["t_ref"] / 60 for r in daily], width=.36,
           color=AMBER, label="PyTorch")
    a2.bar([i + .19 for i in x], [r["t_rs"] / 60 for r in daily], width=.36,
           color=TEAL, label="Rust")
    shortlab = {"du_mm2": "2-motif", "du_mm4_prod": "production", 
                "du_T100_small": "T=100\nL=31", "du_T100_mid": "T=100\nL=49"}
    a2.set_xticks(x); a2.set_xticklabels([shortlab[r["case"]] for r in daily], fontsize=7)
    a2.set_ylabel("wall clock (min)"); a2.legend(frameon=False, fontsize=7.5)
    a2.set_title("Cost", loc="left", fontweight="bold")
    fig.savefig(f"{OUT}/fig5_daily_use.png"); plt.close(fig)

print("wrote:", ", ".join(sorted(os.listdir(OUT))))
