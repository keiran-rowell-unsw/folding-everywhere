"""Generate professional figures for the Rust-vs-PyTorch ESMFold benchmark.
All figures: PNG, dpi=300, bbox_inches='tight'. Reads results/metrics.csv + the
saved atom37 dumps (Rust) and ref.safetensors (PyTorch)."""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from safetensors.numpy import load_file

from common import FIX, REPO

RESULTS = os.path.join(REPO, "results")
FIG = os.path.join(RESULTS, "figures")
DUMPS = os.path.join(RESULTS, "dumps")
os.makedirs(FIG, exist_ok=True)

TORCH_C = "#2E86C1"   # steel blue
RUST_C = "#E67E22"    # orange
plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.dpi": 300, "savefig.bbox": "tight",
})
SAVE = dict(dpi=300, bbox_inches="tight")


def metrics():
    rows = list(csv.DictReader(open(os.path.join(RESULTS, "metrics.csv"))))
    for r in rows:
        for k in ["L", "torch_time", "torch_rss_gb", "rust_time", "rust_rss_gb", "rmsd", "max_dev"]:
            r[k] = float(r[k])
    return rows


def atom37(name):
    ref = load_file(os.path.join(FIX, f"bench/{name}", "ref.safetensors"))
    pt = ref["atom37"]                       # [L,37,3]
    L = pt.shape[0]
    ru = np.fromfile(os.path.join(DUMPS, f"{name}_rust.atom37.f32"), dtype=np.float32).reshape(L, 37, 3)
    ex = ref["atom37_atom_exists"]           # [L,37]
    return pt, ru, ex


def set_equal_3d(ax, pts):
    mn, mx = pts.min(0), pts.max(0)
    c = (mn + mx) / 2
    r = (mx - mn).max() / 2
    ax.set_xlim(c[0] - r, c[0] + r); ax.set_ylim(c[1] - r, c[1] + r); ax.set_zlim(c[2] - r, c[2] + r)
    ax.set_box_aspect([1, 1, 1])


def fig_overlay(rows):
    """Fig 1: 3D Cα backbone overlay (Rust vs PyTorch) for two proteins."""
    names = [r["name"] for r in rows]
    pick = [n for n in ["flgM", "lysozyme_mature", "cytochrome_c", "thioredoxin"] if n in names][:2]
    if len(pick) < 2:
        pick = names[:2]
    fig = plt.figure(figsize=(12, 5.5))
    for k, name in enumerate(pick):
        r = next(x for x in rows if x["name"] == name)
        pt, ru, _ = atom37(name)
        pca, rca = pt[:, 1, :], ru[:, 1, :]   # CA
        ax = fig.add_subplot(1, 2, k + 1, projection="3d")
        ax.plot(pca[:, 0], pca[:, 1], pca[:, 2], color=TORCH_C, lw=3.2, alpha=0.55, label="PyTorch fp32")
        ax.plot(rca[:, 0], rca[:, 1], rca[:, 2], color=RUST_C, lw=1.6, ls="--", label="Rust fp32")
        set_equal_3d(ax, np.vstack([pca, rca]))
        ax.set_title(f"{name}  (L={int(r['L'])},  Cα-RMSD = {r['rmsd']*1000:.2f} mÅ)")
        ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])
        ax.grid(False)
        if k == 0:
            ax.legend(loc="upper left", frameon=True)
    fig.suptitle("Cα backbone traces are indistinguishable: pure-Rust fp32 vs PyTorch fp32",
                 fontsize=13, fontweight="bold")
    fig.savefig(os.path.join(FIG, "fig1_structure_overlay.png"), **SAVE)
    plt.close(fig)


def fig_scatter(rows):
    """Fig 2: all backbone atom coordinates, PyTorch vs Rust (identity), + deviation histogram."""
    idx = [0, 1, 2, 4]  # N, CA, C, O
    P, R = [], []
    for r in rows:
        pt, ru, ex = atom37(r["name"])
        m = ex[:, idx] > 0.5
        P.append(pt[:, idx, :][m].reshape(-1))
        R.append(ru[:, idx, :][m].reshape(-1))
    P = np.concatenate(P); R = np.concatenate(R)
    dev = np.abs(P - R)
    rcorr = np.corrcoef(P, R)[0, 1]

    fig, (a0, a1) = plt.subplots(1, 2, figsize=(12, 5))
    a0.scatter(P, R, s=2, alpha=0.25, color=RUST_C, rasterized=True)
    lo, hi = P.min(), P.max()
    a0.plot([lo, hi], [lo, hi], color="0.2", lw=1, ls="--", label="y = x")
    a0.set_xlabel("PyTorch fp32 coordinate (Å)")
    a0.set_ylabel("Rust fp32 coordinate (Å)")
    a0.set_title(f"All backbone atoms ({len(P):,} coords)\nPearson r = {rcorr:.8f}")
    a0.legend(loc="upper left")
    a0.set_aspect("equal")

    dm = np.clip(dev * 1000, 1e-3, None)  # mÅ, floor for log axis
    bins = np.logspace(np.log10(1e-3), np.log10(max(dm.max(), 1.0)), 60)
    a1.hist(dm, bins=bins, color=TORCH_C, edgecolor="white")
    a1.set_xscale("log"); a1.set_yscale("log")
    a1.set_xlabel("|Rust − PyTorch| per coordinate (mÅ, log)")
    a1.set_ylabel("count")
    a1.set_title(f"Coordinate deviation\nmean {dev.mean()*1000:.3f} mÅ, max {dev.max()*1000:.2f} mÅ")
    fig.suptitle("Predicted coordinates are identical to within fp32 round-off",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.subplots_adjust(top=0.80, wspace=0.25)
    fig.savefig(os.path.join(FIG, "fig2_coord_scatter.png"), **SAVE)
    plt.close(fig)


def fig_distance_maps(rows):
    """Fig 3: Cα–Cα distance maps (PyTorch, Rust, |difference|) for the longest protein."""
    name = max(rows, key=lambda r: r["L"])["name"]
    pt, ru, _ = atom37(name)
    pca, rca = pt[:, 1, :], ru[:, 1, :]
    Dp = np.linalg.norm(pca[:, None, :] - pca[None, :, :], axis=-1)
    Dr = np.linalg.norm(rca[:, None, :] - rca[None, :, :], axis=-1)
    diff = np.abs(Dp - Dr)
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.6))
    for ax, M, t in [(axs[0], Dp, "PyTorch  Cα–Cα distance (Å)"), (axs[1], Dr, "Rust  Cα–Cα distance (Å)")]:
        im = ax.imshow(M, cmap="viridis", origin="lower")
        ax.set_title(t); ax.set_xlabel("residue"); ax.set_ylabel("residue")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    im = axs[2].imshow(diff * 1000, cmap="magma", origin="lower")
    axs[2].set_title(f"|difference| (mÅ)   max {diff.max()*1000:.2f} mÅ")
    axs[2].set_xlabel("residue"); axs[2].set_ylabel("residue")
    fig.colorbar(im, ax=axs[2], fraction=0.046, pad=0.04)
    fig.suptitle(f"Distance maps identical — {name} (L={pca.shape[0]})", fontsize=13, fontweight="bold")
    fig.savefig(os.path.join(FIG, "fig3_distance_maps.png"), **SAVE)
    plt.close(fig)


def fig_accuracy(rows):
    """Fig 4: per-protein Cα-RMSD and max deviation (mÅ)."""
    rows = sorted(rows, key=lambda r: r["L"])
    names = [f"{r['name']}\n({int(r['L'])})" for r in rows]
    rmsd = [r["rmsd"] * 1000 for r in rows]
    mx = [r["max_dev"] * 1000 for r in rows]
    x = np.arange(len(rows)); w = 0.4
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.bar(x - w / 2, rmsd, w, color=RUST_C, label="RMSD")
    ax.bar(x + w / 2, mx, w, color=TORCH_C, label="max deviation")
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Rust vs PyTorch all-atom difference (mÅ, log scale)")
    ax.set_yscale("log")
    ax.axhline(1.0, color="0.5", lw=0.8, ls=":")  # 1 mÅ reference
    ax.set_title("Structural agreement across 15 proteins (1 mÅ = 0.001 Å)\n"
                 "low-confidence/disordered regions amplify fp32 round-off", fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3, which="both")
    fig.savefig(os.path.join(FIG, "fig4_accuracy_per_protein.png"), **SAVE)
    plt.close(fig)


def fig_time_memory(rows):
    """Fig 5: wall time and peak RSS per protein, both versions."""
    rows = sorted(rows, key=lambda r: r["L"])
    names = [f"{r['name']}\n({int(r['L'])})" for r in rows]
    x = np.arange(len(rows)); w = 0.4
    fig, (a0, a1) = plt.subplots(2, 1, figsize=(13, 9))

    a0.bar(x - w / 2, [r["torch_time"] for r in rows], w, color=TORCH_C, label="PyTorch fp32")
    a0.bar(x + w / 2, [r["rust_time"] for r in rows], w, color=RUST_C, label="Rust fp32")
    a0.set_ylabel("wall-clock time (s)")
    a0.set_title("Inference time per protein", fontweight="bold")
    a0.set_xticks(x); a0.set_xticklabels([])
    a0.legend(); a0.grid(axis="y", alpha=0.3)

    a1.bar(x - w / 2, [r["torch_rss_gb"] for r in rows], w, color=TORCH_C, label="PyTorch fp32")
    a1.bar(x + w / 2, [r["rust_rss_gb"] for r in rows], w, color=RUST_C, label="Rust fp32")
    a1.set_ylabel("peak memory (GB)")
    a1.set_title("Peak memory per protein", fontweight="bold")
    a1.set_xticks(x); a1.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    a1.legend(); a1.grid(axis="y", alpha=0.3)
    fig.suptitle("Resource usage: pure-Rust fp32 vs PyTorch fp32 (15 proteins, 4-core CPU)",
                 fontsize=13, fontweight="bold")
    fig.savefig(os.path.join(FIG, "fig5_time_memory.png"), **SAVE)
    plt.close(fig)


def main():
    rows = metrics()
    print(f"{len(rows)} proteins")
    fig_overlay(rows)
    fig_scatter(rows)
    fig_distance_maps(rows)
    fig_accuracy(rows)
    fig_time_memory(rows)
    print("wrote figures to", FIG)
    for f in sorted(os.listdir(FIG)):
        print("  ", f)


if __name__ == "__main__":
    main()
