"""Generate comparison plots from results/{metrics,accuracy}.csv into plots/."""
import os, csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
RES = os.path.join(ROOT, "results")
PLOTS = os.path.join(ROOT, "plots"); os.makedirs(PLOTS, exist_ok=True)

def read_csv(p):
    with open(p) as f:
        return list(csv.DictReader(f))

metrics = read_csv(os.path.join(RES, "metrics.csv"))
acc = read_csv(os.path.join(RES, "accuracy.csv"))

# index metrics by (protein, variant)
M = {(r["protein"], r["variant"]): r for r in metrics}
prots = sorted({r["protein"] for r in metrics}, key=lambda p: int(M[(p, "pt_fp32")]["L"]) if (p, "pt_fp32") in M else 0)
L = [int(M[(p, "pt_fp32")]["L"]) for p in prots]
variants = ["pt_fp32", "pt_bf16", "rust_fp32"]
colors = {"pt_fp32": "#1f77b4", "pt_bf16": "#ff7f0e", "rust_fp32": "#2ca02c"}
labels = {"pt_fp32": "PyTorch fp32", "pt_bf16": "PyTorch bf16 (released)", "rust_fp32": "Rust fp32"}

# ---- Fig 1: accuracy (RMSD vs PyTorch fp32) ----
A = {r["protein"]: r for r in acc}
ap = [p for p in prots if p in A]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
rust_rmsd = [float(A[p]["rust_vs_fp32_rmsd_A"]) * 1000 for p in ap]   # mA
bf16_rmsd = [float(A[p]["bf16_vs_fp32_rmsd_A"]) for p in ap]          # A
x = np.arange(len(ap))
ax1.bar(x, rust_rmsd, color="#2ca02c")
ax1.set_xticks(x); ax1.set_xticklabels(ap, rotation=60, ha="right", fontsize=8)
ax1.set_ylabel("Cα-aligned RMSD (milli-Å)"); ax1.set_title("Rust fp32 vs PyTorch fp32 (sub-mÅ)")
ax1.axhline(1.0, color="gray", ls="--", lw=0.8, label="1 mÅ")
ax1.legend()
ax2.bar(x, bf16_rmsd, color="#ff7f0e")
ax2.set_xticks(x); ax2.set_xticklabels(ap, rotation=60, ha="right", fontsize=8)
ax2.set_ylabel("aligned RMSD (Å)"); ax2.set_title("PyTorch bf16 (released) vs PyTorch fp32")
fig.tight_layout(); fig.savefig(os.path.join(PLOTS, "fig1_accuracy.png"), dpi=200); plt.close(fig)

# ---- Fig 2: confidence agreement (pLDDT, pTM) ----
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
for v in ["rust_fp32", "pt_bf16"]:
    xs = [float(M[(p, "pt_fp32")]["plddt_mean"]) for p in prots if (p, v) in M]
    ys = [float(M[(p, v)]["plddt_mean"]) for p in prots if (p, v) in M]
    ax1.scatter(xs, ys, label=labels[v], color=colors[v], alpha=0.8)
    xs = [float(M[(p, "pt_fp32")]["ptm"]) for p in prots if (p, v) in M]
    ys = [float(M[(p, v)]["ptm"]) for p in prots if (p, v) in M]
    ax2.scatter(xs, ys, label=labels[v], color=colors[v], alpha=0.8)
for ax, t in [(ax1, "mean pLDDT"), (ax2, "pTM")]:
    lim = ax.get_xlim()
    ax.plot(lim, lim, "k--", lw=0.8); ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel(f"PyTorch fp32 {t}"); ax.set_ylabel(f"variant {t}"); ax.set_title(f"{t} agreement"); ax.legend()
fig.tight_layout(); fig.savefig(os.path.join(PLOTS, "fig2_confidence.png"), dpi=200); plt.close(fig)

# ---- Fig 3: fold time vs length ----
fig, ax = plt.subplots(figsize=(8, 5))
for v in variants:
    pp = [p for p in prots if (p, v) in M]
    xs = [int(M[(p, "pt_fp32")]["L"]) for p in pp]
    ys = [float(M[(p, v)]["fold_s"]) for p in pp]
    ax.plot(xs, ys, "o-", label=labels[v], color=colors[v])
ax.set_xlabel("sequence length L"); ax.set_ylabel("fold time (s)"); ax.set_title("Fold time vs length")
ax.legend(); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(os.path.join(PLOTS, "fig3_time.png"), dpi=200); plt.close(fig)

# ---- Fig 4: peak memory vs length ----
fig, ax = plt.subplots(figsize=(8, 5))
for v in variants:
    pp = [p for p in prots if (p, v) in M]
    xs = [int(M[(p, "pt_fp32")]["L"]) for p in pp]
    ys = [float(M[(p, v)]["peak_rss_mb"]) / 1024.0 for p in pp]  # GB
    ax.plot(xs, ys, "o-", label=labels[v], color=colors[v])
ax.set_xlabel("sequence length L"); ax.set_ylabel("peak RSS (GB)"); ax.set_title("Peak memory vs length")
ax.legend(); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(os.path.join(PLOTS, "fig4_memory.png"), dpi=200); plt.close(fig)

print("plots written to", PLOTS)
