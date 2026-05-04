"""Generate publication figures for PXRD-Diff paper.

Outputs (all 300 dpi PNG + matched PDF):
  fig1_ablation.{png,pdf}     -- main ablation bar chart
  fig2_training_curves.{png,pdf} -- v15 vs v16 lattice loss
  fig3_diffpxrd_validation.{png,pdf} -- Pearson histogram for diff simulator
"""
from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).parent
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "savefig.bbox": "tight",
    "savefig.dpi": 300,
})

CB = {  # Wong colorblind-safe palette
    "blue": "#0072B2", "orange": "#E69F00", "green": "#009E73",
    "yellow": "#F0E442", "red": "#D55E00", "purple": "#CC79A7",
    "skyblue": "#56B4E9", "black": "#000000",
}


def fig1_ablation():
    runs = ["v10", "v11", "v13", "v14", "v15", "v16"]
    match = [1.40, 0.90, 2.51, 0.80, 2.10, 1.80]
    pearson = [0.359, 0.365, 0.434, 0.367, 0.392, 0.368]
    rmsd = [0.17, 0.15, 0.22, 0.14, 0.21, 0.22]

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6), constrained_layout=True)
    x = np.arange(len(runs))
    colors = [CB["skyblue"]] * 2 + [CB["green"]] + [CB["red"]] * 1 + [CB["orange"]] * 2

    for ax, vals, ylabel, title in [
        (axes[0], match, "Match rate (%)", "(a) StructureMatcher"),
        (axes[1], pearson, "Pearson correlation", "(b) PXRD Pearson"),
        (axes[2], rmsd, "RMSD (Å) — matched only", "(c) Coord RMSD"),
    ]:
        bars = ax.bar(x, vals, color=colors, edgecolor=CB["black"], linewidth=0.6)
        v13_idx = 2
        bars[v13_idx].set_edgecolor(CB["green"])
        bars[v13_idx].set_linewidth(1.8)
        ax.set_xticks(x)
        ax.set_xticklabels(runs, fontsize=8.5)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(title, fontsize=9, loc="left")
        for i, v in enumerate(vals):
            ax.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=7)
        ax.margins(y=0.18)
        ax.tick_params(axis='both', labelsize=8)

    # Single shared legend below
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor=CB["skyblue"], edgecolor="k", lw=0.6, label="ε prediction"),
        Patch(facecolor=CB["green"], edgecolor="k", lw=1.8, label="x₀-residual (best)"),
        Patch(facecolor=CB["red"], edgecolor="k", lw=0.6, label="x₀ + Wyckoff + dist"),
        Patch(facecolor=CB["orange"], edgecolor="k", lw=0.6, label="x₀ + one extension"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, -0.05), frameon=False, fontsize=7.5)
    fig.suptitle("Ablation on MP-20 test (n=1000, true lattice, coord-only)",
                 fontsize=10, y=1.04)
    for ext in ("png", "pdf"):
        fig.savefig(ROOT / f"fig1_ablation.{ext}")
    plt.close(fig)
    print("wrote fig1_ablation.{png,pdf}")


def parse_curves(path: Path) -> dict[str, dict[str, list[float]]]:
    """Return {run_name: {col: [values]}} where col in {step, coord, lat, aux, debye}."""
    runs: dict[str, dict[str, list[float]]] = {}
    cur = None
    pat = re.compile(
        r"step=\s*(\d+)\s+loss=([\d.]+)\s+coord=([\d.]+)\s+lat=([\d.]+)\s+"
        r"aux=([\d.]+)\s+debye=([\d.]+)"
    )
    for line in path.read_text().splitlines():
        if line.startswith("=== "):
            cur = line.replace("=== ", "").replace(" ===", "").strip()
            runs[cur] = {k: [] for k in ("step", "loss", "coord", "lat", "aux", "debye")}
            continue
        m = pat.search(line)
        if m and cur is not None:
            d = runs[cur]
            d["step"].append(int(m.group(1)))
            d["loss"].append(float(m.group(2)))
            d["coord"].append(float(m.group(3)))
            d["lat"].append(float(m.group(4)))
            d["aux"].append(float(m.group(5)))
            d["debye"].append(float(m.group(6)))
    return runs


def smooth(y, k=5):
    y = np.asarray(y, dtype=float)
    if len(y) < k:
        return y
    kernel = np.ones(k) / k
    return np.convolve(y, kernel, mode="same")


def fig2_training_curves():
    runs = parse_curves(ROOT / "training_curves.txt")

    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.4), constrained_layout=True)

    color_map = {"gpu_v14": CB["red"], "gpu_v15": CB["orange"], "gpu_v16": CB["green"]}
    label_map = {"gpu_v14": "v14 (+Wyck +dist)", "gpu_v15": "v15 (+Wyck)",
                 "gpu_v16": "v16 (+dist) — clean lat"}

    # (a) Lattice loss
    ax = axes[0]
    for name, d in runs.items():
        ax.plot(d["step"], smooth(d["lat"], 5), color=color_map[name],
                label=label_map[name], lw=1.2)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Lattice loss")
    ax.set_title("(a) Lattice prediction", fontsize=9, loc="left")
    ax.set_xlim(0, 100000)
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=7, frameon=False, loc="upper right")
    ax.axhline(1.0, color="gray", lw=0.5, ls="--", alpha=0.5)
    ax.text(95000, 0.97, "random baseline", fontsize=6, color="gray",
            ha="right", va="bottom")

    # (b) Coord loss
    ax = axes[1]
    for name, d in runs.items():
        ax.plot(d["step"], smooth(d["coord"], 5), color=color_map[name],
                label=label_map[name], lw=1.2)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Coordinate loss")
    ax.set_title("(b) Coordinate prediction", fontsize=9, loc="left")
    ax.set_xlim(0, 100000)
    ax.set_ylim(0.06, 0.10)
    ax.legend(fontsize=7, frameon=False, loc="upper right")

    fig.suptitle("Wyckoff embedding destabilises lattice prediction",
                 fontsize=10, y=1.05)
    for ext in ("png", "pdf"):
        fig.savefig(ROOT / f"fig2_training_curves.{ext}")
    plt.close(fig)
    print("wrote fig2_training_curves.{png,pdf}")


def fig3_diffpxrd_validation():
    """Real Pearson values from scripts/04_verify_debye.py on 50 MP-20 test structures."""
    raw = ("0.9382,0.9649,0.9719,0.9386,0.9758,0.9975,0.9885,0.9946,0.9869,"
           "0.9759,0.9183,0.9325,0.9240,0.9267,0.9692,0.9487,0.9411,0.9865,"
           "0.9742,0.9745,0.9932,0.9864,0.9970,0.9092,0.9903,0.9807,0.9911,"
           "0.9459,0.9543,0.9687,0.9769,0.9786,0.9456,0.9903,0.8956,0.9821,"
           "0.9762,0.9575,0.9735,0.9616,0.9376,0.9765,0.9722,0.9469,0.9116,"
           "0.9883,0.9709,0.9575,0.9858,0.9057")
    samples = np.array([float(x) for x in raw.split(",")])

    fig, ax = plt.subplots(figsize=(4.0, 2.6), constrained_layout=True)
    ax.hist(samples, bins=15, color=CB["blue"], edgecolor=CB["black"], lw=0.6)
    ax.axvline(samples.mean(), color=CB["red"], lw=1.2, ls="-",
               label=f"mean = {samples.mean():.3f}")
    ax.axvline(0.7, color="gray", lw=0.8, ls="--",
               label="acceptance threshold (0.7)")
    ax.set_xlabel("Pearson correlation vs pymatgen.XRDCalculator")
    ax.set_ylabel("Count (n=50 structures)")
    ax.set_title("DiffPXRD vs reference simulator", fontsize=10, loc="left")
    ax.legend(fontsize=7, frameon=False, loc="upper left")
    ax.set_xlim(0.65, 1.02)
    for ext in ("png", "pdf"):
        fig.savefig(ROOT / f"fig3_diffpxrd_validation.{ext}")
    plt.close(fig)
    print("wrote fig3_diffpxrd_validation.{png,pdf}")


if __name__ == "__main__":
    fig1_ablation()
    fig2_training_curves()
    fig3_diffpxrd_validation()
    print("done.")
