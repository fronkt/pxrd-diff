"""Generate publication figures for PXRD-Diff paper.

Outputs (all 300 dpi PNG + matched PDF):
  fig1_ablation.{png,pdf}             -- main ablation bar chart (Phase 4)
  fig2_training_curves.{png,pdf}      -- v15 vs v16 lattice loss   (Phase 4)
  fig3_diffpxrd_validation.{png,pdf}  -- Pearson histogram         (Phase 4)
  fig4_indexer_bench.{png,pdf}        -- per-system indexer benchmark (Phase 9 reframe)
  fig5_threeway_headline.{png,pdf}    -- ours vs DGpt vs PXRDnet headline bars
"""
from __future__ import annotations

import json
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


def fig4_indexer_bench():
    """Per-crystal-system strict% and len_MAE from the v2 native indexer.

    Source: paper/phase9_results/index_benchmark_v2_native.json (n=1000 MP-20 test).
    """
    d = json.loads((ROOT / "phase9_results" / "index_benchmark_v2_native.json").read_text())
    ps = d["per_system"]
    order = ["cubic", "tetragonal", "hexagonal", "trigonal",
             "orthorhombic", "monoclinic", "triclinic"]
    labels = [s.capitalize() for s in order]
    ns = [ps[s]["n"] for s in order]
    strict = [ps[s]["strict_pct"] for s in order]
    consist = [ps[s]["consistent_pct"] for s in order]
    lenmae = [ps[s]["len_mae"] for s in order]
    # triclinic len_mae is NaN — substitute with 0 for plotting and annotate
    lenmae_plot = [(v if v == v else 0.0) for v in lenmae]
    overall = d["overall"]

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), constrained_layout=True)
    x = np.arange(len(order))
    width = 0.4

    # (a) strict / consistent match rate
    ax = axes[0]
    ax.bar(x - width / 2, strict, width, color=CB["blue"], edgecolor=CB["black"],
           linewidth=0.6, label="Strict")
    ax.bar(x + width / 2, consist, width, color=CB["skyblue"], edgecolor=CB["black"],
           linewidth=0.6, label="Consistent")
    ax.axhline(overall["overall_strict_pct"], color=CB["red"], lw=0.8, ls="--",
               label=f"overall strict {overall['overall_strict_pct']:.1f}%")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7.5, rotation=25, ha="right")
    ax.set_ylabel("Indexing accuracy (%)", fontsize=9)
    ax.set_title("(a) Per-system match rate", fontsize=9, loc="left")
    ax.set_ylim(0, 100)
    ax.legend(fontsize=6.5, frameon=False, loc="upper right")
    for i, (v, n) in enumerate(zip(strict, ns)):
        ax.text(i - width / 2, v + 1.5, f"{v:.0f}", ha="center", fontsize=6.5)
        ax.text(i, -8, f"n={n}", ha="center", fontsize=6, color="gray")

    # (b) lattice-length MAE (Å)
    ax = axes[1]
    bar_colors = [CB["green"] if v <= 1.0 else CB["orange"] if v <= 2.0 else CB["red"]
                  for v in lenmae_plot]
    bars = ax.bar(x, lenmae_plot, color=bar_colors, edgecolor=CB["black"], linewidth=0.6)
    ax.axhline(overall["v20_learned_head_len_mae"], color=CB["black"], lw=0.8, ls=":",
               label=f"v20 learned head ({overall['v20_learned_head_len_mae']:.2f} Å)")
    ax.axhline(overall["overall_len_mae"], color=CB["red"], lw=0.8, ls="--",
               label=f"overall ({overall['overall_len_mae']:.2f} Å)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7.5, rotation=25, ha="right")
    ax.set_ylabel("Lattice-length MAE (Å)", fontsize=9)
    ax.set_title("(b) Per-system lattice error", fontsize=9, loc="left")
    ax.set_ylim(0, 5.2)
    ax.legend(fontsize=6.5, frameon=False, loc="upper left")
    for i, v in enumerate(lenmae):
        if v != v:
            ax.text(i, 0.2, "n/a", ha="center", fontsize=7, color="gray")
        else:
            ax.text(i, v + 0.1, f"{v:.2f}", ha="center", fontsize=6.5)

    fig.suptitle(
        "Classical Q-space indexer: wins on high-symmetry, fails on low-symmetry",
        fontsize=10, y=1.04)
    for ext in ("png", "pdf"):
        fig.savefig(ROOT / f"fig4_indexer_bench.{ext}")
    plt.close(fig)
    print("wrote fig4_indexer_bench.{png,pdf}")


def fig5_threeway_headline():
    """Three-way head-to-head: ours (9.1.3) vs DiffractGPT (n=1000) vs PXRDnet (n=20).

    Sources:
      paper/phase9_results/p9_idxlat_n1000.json         (ours, n=1000)
      paper/phase9_results/baseline_diffractgpt_n1000.json
      paper/phase9_results/baseline_pxrdnet_sinc100_n20.json
    """
    def load(name):
        return json.loads((ROOT / "phase9_results" / name).read_text())

    ours = load("p9_idxlat_n1000.json")
    dgpt = load("baseline_diffractgpt_n1000.json")
    pxnt = load("baseline_pxrdnet_sinc100_n20.json")

    systems = [
        ("PXRD-Diff\n(ours, n=1000)", ours, CB["blue"]),
        ("DiffractGPT\n(n=1000)",     dgpt, CB["orange"]),
        ("PXRDnet\n(n=20)",           pxnt, CB["green"]),
    ]

    metrics = [
        ("Match rate (%)",        "match_rate (StructureMatcher)", True),
        ("All-correct (%)",       "headline_all_correct",          True),
        ("Pearson (pred vs true)","pearson_mean",                  False),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6), constrained_layout=True)
    x = np.arange(len(systems))

    for ax, (ylabel, key, pct) in zip(axes, metrics):
        vals = [s[1][key] * (100 if pct else 1) for s in systems]
        colors = [s[2] for s in systems]
        ax.bar(x, vals, color=colors, edgecolor=CB["black"], linewidth=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels([s[0] for s in systems], fontsize=7.5)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(ylabel.split("(")[0].strip(), fontsize=9, loc="left")
        for i, v in enumerate(vals):
            fmt = f"{v:.1f}" if pct else f"{v:.2f}"
            ax.text(i, v, fmt, ha="center", va="bottom", fontsize=7.5)
        ax.margins(y=0.20)
        ax.tick_params(axis="y", labelsize=8)

    fig.suptitle(
        "Open generative baselines on the same MP-20 evaluation harness",
        fontsize=10, y=1.05)
    for ext in ("png", "pdf"):
        fig.savefig(ROOT / f"fig5_threeway_headline.{ext}")
    plt.close(fig)
    print("wrote fig5_threeway_headline.{png,pdf}")


if __name__ == "__main__":
    fig1_ablation()
    fig2_training_curves()
    fig3_diffpxrd_validation()
    fig4_indexer_bench()
    fig5_threeway_headline()
    print("done.")
