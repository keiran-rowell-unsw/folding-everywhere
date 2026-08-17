"""Figures for the Rust-vs-PyTorch ProteinMPNN benchmark.

All figures: PNG, dpi=300, tight bbox. Inputs:
  results/metrics.csv           (run_benchmark.py)
  results/logprob_accuracy.csv  (compare_logprobs.py)
  results/stage_parity/*.json   (cargo test --test parity_model)
"""
import csv
import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from common import RESULTS  # noqa: E402

FIG = os.path.join(RESULTS, "figures")
os.makedirs(FIG, exist_ok=True)

TORCH_C = "#2E86C1"   # steel blue
RUST_C = "#E67E22"    # orange
OK_C = "#27AE60"      # green
GREY = "#7F8C8D"
plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.dpi": 300, "savefig.bbox": "tight",
})
SAVE = dict(dpi=300, bbox_inches="tight")

# Pipeline order for the stage figure (label -> short name).
STAGE_ORDER = [
    ("X", "coords X"),
    ("virtual Cb", "virtual Cb"),
    ("D_neighbors", "kNN distances"),
    ("RBF(D_neighbors)", "RBF(dist)"),
    ("E_input (416-d)", "edge input\n(416-d)"),
    ("E (edge features)", "edge embed\n(Linear+LN)"),
    ("h_E init (W_e)", "W_e"),
    ("enc0 h_V", "enc0 h_V"), ("enc0 h_E", "enc0 h_E"),
    ("enc1 h_V", "enc1 h_V"), ("enc1 h_E", "enc1 h_E"),
    ("enc2 h_V", "enc2 h_V"), ("enc2 h_E", "enc2 h_E"),
    ("dec0 h_V", "dec0 h_V"), ("dec1 h_V", "dec1 h_V"), ("dec2 h_V", "dec2 h_V"),
    ("logits", "logits"), ("log_probs", "log-probs"),
    ("sample probs", "sample probs"),
]


def read_metrics():
    path = os.path.join(RESULTS, "metrics.csv")
    rows = list(csv.DictReader(open(path)))
    for r in rows:
        for k, v in r.items():
            if k == "name":
                continue
            r[k] = float(v)
    rows.sort(key=lambda r: r["L"])
    return rows


def read_logprob():
    path = os.path.join(RESULTS, "logprob_accuracy.csv")
    if not os.path.exists(path):
        return []
    rows = list(csv.DictReader(open(path)))
    for r in rows:
        for k, v in r.items():
            if k != "name":
                r[k] = float(v)
    rows.sort(key=lambda r: r["L"])
    return rows


def read_stages():
    out = {}
    for p in glob.glob(os.path.join(RESULTS, "stage_parity", "*.json")):
        d = json.load(open(p))
        out[d["stage"]] = d
    return [(short, out[label]) for label, short in STAGE_ORDER if label in out]


# --------------------------------------------------------------------------
def fig1_sequence_identity(rows):
    """The headline correctness result: every designed sequence is identical."""
    names = [r["name"] for r in rows]
    n_seq = [int(r["n_seq"]) for r in rows]
    ident = [int(r["seqs_identical"]) for r in rows]
    x = np.arange(len(rows))

    fig, (ax, ax2) = plt.subplots(
        2, 1, figsize=(12, 7.2), gridspec_kw={"height_ratios": [2, 1.15]}
    )
    ax.bar(x, ident, color=OK_C, width=0.68, label="identical to PyTorch")
    miss = [n - i for n, i in zip(n_seq, ident)]
    if any(miss):
        ax.bar(x, miss, bottom=ident, color="#C0392B", width=0.68, label="differing")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{n}\nL={int(r['L'])}" for n, r in zip(names, rows)], fontsize=8)
    ax.set_ylabel("sequences per structure")
    ax.set_ylim(0, max(n_seq) * 1.28)
    tot, tid = sum(n_seq), sum(ident)
    ax.set_title(
        f"Designed sequences are identical residue-for-residue: "
        f"{tid}/{tot} sequences over {len(rows)} structures "
        f"({100*tid/tot:.1f}%)", fontweight="bold")
    ax.legend(loc="upper right", frameon=False, ncol=2)
    for xi, v in zip(x, ident):
        ax.text(xi, v + 0.12, str(v), ha="center", fontsize=8, color=OK_C, fontweight="bold")

    # score agreement (FASTA-reported, 4 decimals)
    dmax = [r["max_score_absdiff"] for r in rows]
    ax2.bar(x, dmax, color=GREY, width=0.68)
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, fontsize=8, rotation=45, ha="right")
    ax2.set_ylabel("max |Δ score|")
    ax2.set_title("Reported per-sequence scores agree to the printed precision", fontsize=11)
    if max(dmax) == 0:
        ax2.set_ylim(0, 1e-4)
        ax2.text(0.5, 0.55, "all differences exactly 0",
                 transform=ax2.transAxes, ha="center", color=OK_C, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig1_sequence_identity.png"), **SAVE)
    plt.close(fig)


def fig2_logprob_agreement(lp_rows):
    """Continuous agreement of the log-probability matrices."""
    if not lp_rows:
        return
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    names = [r["name"] for r in lp_rows]
    x = np.arange(len(lp_rows))

    ax = axes[0]
    ax.bar(x, [r["max_abs"] for r in lp_rows], color=RUST_C, width=0.7)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=90, fontsize=7)
    ax.set_ylabel("max |Δ log P|")
    ax.axhline(1.2e-7, color=GREY, ls=":", lw=1)
    ax.text(0.02, 1.3e-7, "fp32 eps", fontsize=8, color=GREY)
    ax.set_title("Largest log-probability deviation")

    ax = axes[1]
    cos = [1 - r["cosine"] for r in lp_rows]
    ax.bar(x, np.maximum(cos, 1e-16), color=TORCH_C, width=0.7)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=90, fontsize=7)
    ax.set_ylabel("1 − cosine similarity")
    ax.set_title("Cosine similarity is 1.0 to 12 decimals")

    ax = axes[2]
    agree = [100 * r["argmax_agree"] for r in lp_rows]
    ax.bar(x, agree, color=OK_C, width=0.7)
    ax.set_ylim(99.0, 100.05)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=90, fontsize=7)
    ax.set_ylabel("% positions")
    ax.set_title("argmax residue agrees at every position")

    fig.suptitle(
        "Log-probabilities agree at fp32 round-off; every discrete decision is identical",
        fontsize=13, fontweight="bold", y=1.04)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig2_logprob_agreement.png"), **SAVE)
    plt.close(fig)


def fig3_stage_parity(stages):
    """Where numerical difference enters the pipeline, layer by layer."""
    if not stages:
        return
    labels = [s[0] for s in stages]
    d = [s[1] for s in stages]
    x = np.arange(len(labels))

    # Three regimes, in pipeline order:
    #   bit-exact  ->  1-ULP transcendental (sqrt/exp)  ->  GEMM accumulation
    GEOM = {"kNN distances", "RBF(dist)", "edge input\n(416-d)"}
    ULP_C = "#F1C40F"

    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(13, 7.8), sharex=True)
    vals = [max(v["max_abs"], 1e-9) for v in d]
    colors = [
        OK_C if v["max_abs"] == 0 else (ULP_C if lab in GEOM else RUST_C)
        for lab, v in zip(labels, d)
    ]
    ax.bar(x, vals, color=colors, width=0.66)
    ax.set_yscale("log")
    ax.set_ylim(5e-10, 2e-4)
    ax.set_ylabel("max |Δ| vs PyTorch")
    ax.axhline(1.2e-7, color=GREY, ls=":", lw=1)
    ax.text(0.15, 1.5e-7, "fp32 eps", fontsize=8, color=GREY)
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=OK_C),
        plt.Rectangle((0, 0), 1, 1, color=ULP_C),
        plt.Rectangle((0, 0), 1, 1, color=RUST_C),
    ]
    ax.legend(
        handles,
        ["bit-exact",
         "≤1 ULP: PyTorch's fp32 sqrt is not correctly rounded",
         "fp32 GEMM accumulation order (MKL blocking)"],
        loc="lower right", frameon=False, fontsize=9)
    ax.set_title(
        "Numerical agreement stage by stage (5L33, L=106)\n"
        "difference enters twice — at the distance sqrt, then at the first matmul — "
        "and stops growing after that",
        fontweight="bold")

    ax2.bar(x, [100 * v["bitexact_frac"] for v in d], color=TORCH_C, width=0.66)
    ax2.set_ylabel("% values bit-identical")
    ax2.set_ylim(0, 105)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    for xi, v in zip(x, d):
        if v["bitexact_frac"] > 0.5:
            ax2.text(xi, 100 * v["bitexact_frac"] + 2,
                     f"{100*v['bitexact_frac']:.0f}", ha="center", fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig3_stage_parity.png"), **SAVE)
    plt.close(fig)


def fig4_speed(rows):
    """Wall time vs chain length, against both PyTorch threading settings."""
    L = np.array([r["L"] for r in rows])
    tt = np.array([r["torch_time"] for r in rows])
    rt = np.array([r["rust_time"] for r in rows])
    has_mt = "torch_time_mt" in rows[0]
    tmt = np.array([r["torch_time_mt"] for r in rows]) if has_mt else None
    TORCH_MT_C = "#5DADE2"

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.5))
    ax = axes[0]
    ax.plot(L, tt, "o-", color=TORCH_C, label="PyTorch, 1 thread", ms=5)
    if has_mt:
        ax.plot(L, tmt, "^-", color=TORCH_MT_C, label="PyTorch, default threads", ms=5)
    ax.plot(L, rt, "s-", color=RUST_C, label="Rust (rayon, all cores)", ms=5)
    ax.set_xlabel("chain length L")
    ax.set_ylabel("wall time (s)")
    ax.set_title(f"End-to-end run ({int(rows[0]['n_seq'])} sequences/structure)")
    ax.legend(frameon=False, fontsize=9)

    ax = axes[1]
    ax.plot(L, 1000 * tt / rows[0]["n_seq"], "o-", color=TORCH_C, ms=5, label="PyTorch, 1 thread")
    if has_mt:
        ax.plot(L, 1000 * tmt / rows[0]["n_seq"], "^-", color=TORCH_MT_C, ms=5,
                label="PyTorch, default")
    ax.plot(L, 1000 * rt / rows[0]["n_seq"], "s-", color=RUST_C, ms=5, label="Rust")
    ax.set_xlabel("chain length L")
    ax.set_ylabel("ms per sequence")
    ax.set_title("Sampling throughput")
    ax.legend(frameon=False, fontsize=9)

    ax = axes[2]
    x = np.arange(len(rows))
    w = 0.4 if has_mt else 0.7
    sp1 = tt / rt
    ax.bar(x - (w / 2 if has_mt else 0), sp1, w, color=TORCH_C, label="vs 1-thread")
    if has_mt:
        ax.bar(x + w / 2, tmt / rt, w, color=TORCH_MT_C, label="vs default threads")
    ax.axhline(1.0, color=GREY, ls="--", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels([r["name"] for r in rows], rotation=90, fontsize=7)
    ax.set_ylabel("PyTorch time / Rust time")
    title = f"Rust speedup (median {np.median(sp1):.2f}×"
    if has_mt:
        title += f" / {np.median(tmt / rt):.2f}×"
    ax.set_title(title + ")")
    ax.legend(frameon=False, fontsize=8)

    fig.suptitle(
        "Speed on the same 4-core machine. PyTorch is timed both pinned to one thread "
        "(per-core comparison)\nand at its default thread count (what a user actually gets).",
        fontsize=12, fontweight="bold", y=1.09)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig4_speed.png"), **SAVE)
    plt.close(fig)


def read_memory():
    path = os.path.join(RESULTS, "memory.csv")
    if not os.path.exists(path):
        return {}
    out = {}
    for r in csv.DictReader(open(path)):
        out[r["name"]] = (float(r["torch_rss_mb"]), float(r["rust_rss_mb"]))
    return out


def fig5_memory_and_recovery(rows):
    """Peak memory, and the biological sanity check (native sequence recovery)."""
    names = [r["name"] for r in rows]
    x = np.arange(len(rows))
    # Measured by measure_memory.py with /usr/bin/time; getrusage(RUSAGE_CHILDREN)
    # cannot attribute peak RSS per child once a bigger one has already run.
    mem = read_memory()
    tr = np.array([mem.get(n, (np.nan, np.nan))[0] for n in names])
    rr = np.array([mem.get(n, (np.nan, np.nan))[1] for n in names])

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.6))
    ax = axes[0]
    w = 0.4
    ax.bar(x - w / 2, tr, w, color=TORCH_C, label="PyTorch")
    ax.bar(x + w / 2, rr, w, color=RUST_C, label="Rust")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=90, fontsize=7)
    ax.set_ylabel("peak RSS (MB)")
    ax.set_title(f"Peak RSS (median {np.nanmedian(tr):.0f} MB vs {np.nanmedian(rr):.0f} MB, "
                 f"{np.nanmedian(tr)/np.nanmedian(rr):.1f}x lower)")
    ax.legend(frameon=False, fontsize=9)

    ax = axes[1]
    rec = 100 * np.array([r["mean_recovery"] for r in rows])
    ax.bar(x, rec, color=OK_C, width=0.7)
    ax.axhline(rec.mean(), color=GREY, ls="--", lw=1)
    ax.text(len(x) - 0.5, rec.mean() + 1, f"mean {rec.mean():.1f}%",
            fontsize=9, color=GREY, ha="right")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=90, fontsize=7)
    ax.set_ylabel("native sequence recovery (%)")
    ax.set_title("Design quality (identical for both implementations)")

    fig.suptitle("Resource use and design quality", fontsize=13, fontweight="bold", y=1.04)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig5_memory_recovery.png"), **SAVE)
    plt.close(fig)


def fig6_config_sweep():
    """Does sequence identity hold across model settings, not just one config?"""
    path = os.path.join(RESULTS, "config_sweep.csv")
    if not os.path.exists(path):
        return
    rows = list(csv.DictReader(open(path)))
    for r in rows:
        r["identical"] = int(r["identical"])
        r["total"] = int(r["total"])
        r["L"] = int(r["L"])

    # Group by the axis each configuration varies.
    def axis(tag):
        if "_model_" in tag:
            return "checkpoint"
        if "_temp_" in tag:
            return "temperature"
        if "_seed_" in tag:
            return "seed"
        if "many_seqs" in tag:
            return "long RNG stream"
        return "multi-chain"

    order = ["checkpoint", "temperature", "seed", "long RNG stream", "multi-chain"]
    groups = {a: [r for r in rows if axis(r["tag"]) == a] for a in order}
    groups = {k: v for k, v in groups.items() if v}

    fig, ax = plt.subplots(figsize=(14, 5.6))
    x, labels, colors, edges = [], [], [], []
    pos = 0
    boundaries = []
    for gi, (a, rs) in enumerate(groups.items()):
        for r in rs:
            x.append(pos)
            lab = r["tag"]
            for pre in ("6EKB_", "7NL3_"):
                lab = lab.replace(pre, "")
            labels.append(f"{lab}\n{r['pdb']} L={r['L']}")
            ok = r["identical"] == r["total"]
            colors.append(OK_C if ok else "#C0392B")
            edges.append(r["identical"] / max(r["total"], 1) * 100)
            pos += 1
        boundaries.append((pos - len(rs) - 0.5, pos - 0.5, a))
        pos += 1  # gap between groups

    ax.bar(x, edges, color=colors, width=0.72)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=90, fontsize=6.5)
    ax.set_ylabel("% sequences identical to PyTorch")
    ax.set_ylim(0, 118)
    ax.axhline(100, color=GREY, ls=":", lw=1)
    for lo, hi, a in boundaries:
        ax.text((lo + hi) / 2, 109, a, ha="center", fontsize=10, fontweight="bold",
                color=TORCH_C)
        ax.plot([lo + 0.25, hi - 0.25], [105, 105], color=TORCH_C, lw=1.2)

    tot = sum(r["total"] for r in rows)
    ident = sum(r["identical"] for r in rows)
    ax.set_title(
        f"Sequence identity holds across model settings: "
        f"{ident}/{tot} sequences over {len(rows)} configurations "
        f"(4 checkpoints, T = 0.05–1.0, 4 seeds, complexes, homo-oligomer, fixed chains)",
        fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig6_config_sweep.png"), **SAVE)
    plt.close(fig)


def main():
    rows = read_metrics()
    lp = read_logprob()
    stages = read_stages()
    fig1_sequence_identity(rows)
    fig2_logprob_agreement(lp)
    fig3_stage_parity(stages)
    fig4_speed(rows)
    fig5_memory_and_recovery(rows)
    fig6_config_sweep()
    print("figures written to", FIG)
    for f in sorted(os.listdir(FIG)):
        print("  ", f)


if __name__ == "__main__":
    main()
