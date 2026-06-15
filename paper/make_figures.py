"""Generate publication figures for PXRD-Diff paper.

Outputs (all 300 dpi PNG + matched PDF):
  fig1_ablation.{png,pdf}             -- main ablation bar chart (Phase 4)
  fig2_training_curves.{png,pdf}      -- v15 vs v16 lattice loss   (Phase 4)
  fig3_diffpxrd_validation.{png,pdf}  -- Pearson histogram         (Phase 4)
  fig4_indexer_bench.{png,pdf}        -- per-system indexer benchmark (Phase 9 reframe)
  fig5_threeway_headline.{png,pdf}    -- ours vs DGpt vs PXRDnet headline bars

Layout principles (2026-06 restructure):
  * every value label gets explicit headroom so nothing clips
  * shared legends use loc="outside ..." so they reserve their own band
  * multi-line tick labels replace colliding below-axis annotations
  * light y-grids sit behind the bars; spines trimmed
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator

ROOT = Path(__file__).parent
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.titlesize": 9.5,
    "axes.titleweight": "bold",
    "axes.labelsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "axes.axisbelow": True,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7.5,
    "legend.frameon": False,
    "grid.color": "#B0B0B0",
    "grid.linewidth": 0.5,
    "grid.alpha": 0.30,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.dpi": 300,
})

CB = {  # Wong colorblind-safe palette
    "blue": "#0072B2", "orange": "#E69F00", "green": "#009E73",
    "yellow": "#F0E442", "red": "#D55E00", "purple": "#CC79A7",
    "skyblue": "#56B4E9", "black": "#000000",
}
GRID_KW = dict(axis="y", linestyle="-")
LABEL_GREY = "#555555"


def _ygrid(ax):
    ax.grid(**GRID_KW)
    ax.set_axisbelow(True)


def _headroom(ax, vals, frac=0.20, bottom=0.0):
    """Expand the y-limit so value labels above the tallest bar never clip."""
    top = max(vals) if len(vals) else 1.0
    ax.set_ylim(bottom, top * (1.0 + frac))


def _bar_labels(ax, xs, vals, fmt="{:.2f}", dy=None, fontsize=7, color="#222222"):
    span = ax.get_ylim()[1] - ax.get_ylim()[0]
    dy = dy if dy is not None else span * 0.015
    for x, v in zip(xs, vals):
        ax.text(x, v + dy, fmt.format(v), ha="center", va="bottom",
                fontsize=fontsize, color=color)


def _save(fig, stem):
    for ext in ("png", "pdf"):
        fig.savefig(ROOT / f"{stem}.{ext}")
    plt.close(fig)
    print(f"wrote {stem}.{{png,pdf}}")


def fig1_ablation():
    runs = ["v10", "v11", "v13", "v14", "v15", "v16"]
    match = [1.40, 0.90, 2.51, 0.80, 2.10, 1.80]
    pearson = [0.359, 0.365, 0.434, 0.367, 0.392, 0.368]
    rmsd = [0.17, 0.15, 0.22, 0.14, 0.21, 0.22]

    fig, axes = plt.subplots(1, 3, figsize=(7.4, 3.0), constrained_layout=True)
    x = np.arange(len(runs))
    colors = [CB["skyblue"]] * 2 + [CB["green"]] + [CB["red"]] * 1 + [CB["orange"]] * 2
    best_idx = 2  # v13

    panels = [
        (axes[0], match, "Match rate (%)", "(a) StructureMatcher", "{:.2f}"),
        (axes[1], pearson, "Pearson correlation", "(b) PXRD Pearson", "{:.3f}"),
        (axes[2], rmsd, "Coord RMSD (Å)", "(c) Coord RMSD · matched", "{:.2f}"),
    ]
    for ax, vals, ylabel, title, fmt in panels:
        bars = ax.bar(x, vals, color=colors, edgecolor="white", linewidth=0.8, zorder=3)
        # outline the best (x0-residual) run so it reads at a glance
        bars[best_idx].set_edgecolor(CB["black"])
        bars[best_idx].set_linewidth(1.6)
        ax.set_xticks(x)
        ax.set_xticklabels(runs)
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left")
        _ygrid(ax)
        _headroom(ax, vals, frac=0.22)
        _bar_labels(ax, x, vals, fmt=fmt)

    legend_handles = [
        Patch(facecolor=CB["skyblue"], edgecolor="white", label="ε prediction"),
        Patch(facecolor=CB["green"], edgecolor=CB["black"], lw=1.6, label="x₀-residual (best)"),
        Patch(facecolor=CB["red"], edgecolor="white", label="x₀ + Wyckoff + dist"),
        Patch(facecolor=CB["orange"], edgecolor="white", label="x₀ + one extension"),
    ]
    fig.legend(handles=legend_handles, loc="outside lower center", ncol=4,
               handlelength=1.2, columnspacing=1.6, borderaxespad=0.2)
    fig.suptitle("Ablation on MP-20 test (n=1000, true lattice, coord-only)",
                 fontsize=10.5, fontweight="bold")
    _save(fig, "fig1_ablation")


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


def smooth(y, k=9):
    """Centred moving average with edge shrink (no convolution wrap artefacts)."""
    y = np.asarray(y, dtype=float)
    n = len(y)
    if n < 3:
        return y
    k = min(k, n if n % 2 else n - 1)
    half = k // 2
    out = np.empty(n)
    for i in range(n):
        lo, hi = max(0, i - half), min(n, i + half + 1)
        out[i] = y[lo:hi].mean()
    return out


def fig2_training_curves():
    runs = parse_curves(ROOT / "training_curves.txt")

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9), constrained_layout=True)

    color_map = {"gpu_v14": CB["red"], "gpu_v15": CB["orange"], "gpu_v16": CB["green"]}
    label_map = {"gpu_v14": "v14  (+Wyck +dist)", "gpu_v15": "v15  (+Wyck)",
                 "gpu_v16": "v16  (+dist) · clean lat"}
    order = ["gpu_v14", "gpu_v15", "gpu_v16"]

    handles, labels = [], []

    def draw(ax, col):
        for name in order:
            d = runs[name]
            step = np.asarray(d["step"])
            raw = np.asarray(d[col])
            ax.plot(step, raw, color=color_map[name], lw=0.6, alpha=0.18, zorder=2)
            line, = ax.plot(step, smooth(raw, 11), color=color_map[name],
                            label=label_map[name], lw=1.7, zorder=3)
            if ax is axes[0]:
                handles.append(line)
                labels.append(label_map[name])
        ax.set_xlabel("Training step")
        ax.set_xlim(0, 100000)
        ax.xaxis.set_major_locator(MaxNLocator(5))
        ax.grid(linestyle="-")
        ax.set_axisbelow(True)

    # (a) Lattice loss
    ax = axes[0]
    draw(ax, "lat")
    ax.set_ylabel("Lattice loss")
    ax.set_title("(a) Lattice prediction", loc="left")
    ax.set_ylim(0, 1.08)
    ax.axhline(1.0, color="gray", lw=0.8, ls="--", alpha=0.7, zorder=1)
    ax.text(1500, 0.985, "random baseline", fontsize=7, color="gray",
            ha="left", va="top")

    # (b) Coord loss
    ax = axes[1]
    draw(ax, "coord")
    ax.set_ylabel("Coordinate loss")
    ax.set_title("(b) Coordinate prediction", loc="left")
    ax.set_ylim(0.06, 0.10)

    # one shared legend below — keeps it off the busy curves
    fig.legend(handles, labels, loc="outside lower center", ncol=3,
               handlelength=1.6, columnspacing=2.0, borderaxespad=0.2)
    fig.suptitle("Wyckoff embedding destabilises lattice prediction",
                 fontsize=10.5, fontweight="bold")
    _save(fig, "fig2_training_curves")


def fig3_diffpxrd_validation():
    """Real Pearson values from scripts/04_verify_debye.py on 50 MP-20 test structures."""
    raw = ("0.9382,0.9649,0.9719,0.9386,0.9758,0.9975,0.9885,0.9946,0.9869,"
           "0.9759,0.9183,0.9325,0.9240,0.9267,0.9692,0.9487,0.9411,0.9865,"
           "0.9742,0.9745,0.9932,0.9864,0.9970,0.9092,0.9903,0.9807,0.9911,"
           "0.9459,0.9543,0.9687,0.9769,0.9786,0.9456,0.9903,0.8956,0.9821,"
           "0.9762,0.9575,0.9735,0.9616,0.9376,0.9765,0.9722,0.9469,0.9116,"
           "0.9883,0.9709,0.9575,0.9858,0.9057")
    samples = np.array([float(x) for x in raw.split(",")])

    fig, ax = plt.subplots(figsize=(4.4, 2.8), constrained_layout=True)
    bins = np.linspace(0.88, 1.00, 13)
    ax.hist(samples, bins=bins, color=CB["blue"], edgecolor="white", lw=0.7, zorder=3)
    ax.axvline(samples.mean(), color=CB["red"], lw=1.6, ls="-", zorder=4,
               label=f"mean = {samples.mean():.3f}")
    ax.set_xlabel("Pearson correlation vs pymatgen.XRDCalculator")
    ax.set_ylabel("Count (n = 50 structures)")
    ax.set_title("DiffPXRD vs reference simulator", loc="left")
    ax.set_xlim(0.875, 1.005)
    ax.grid(linestyle="-")
    ax.set_axisbelow(True)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    # All 50 samples sit far above the 0.7 acceptance gate — state it instead of
    # wasting a third of the axis on empty space down to 0.7.
    ax.text(0.882, ax.get_ylim()[1] * 0.92,
            f"all 50 structures ≥ {samples.min():.2f}\n(acceptance threshold = 0.70)",
            fontsize=7, color=LABEL_GREY, ha="left", va="top")
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.78))
    _save(fig, "fig3_diffpxrd_validation")


def fig4_indexer_bench():
    """Per-crystal-system strict% and len_MAE from the v2 native indexer.

    Source: paper/phase9_results/index_benchmark_v2_native.json (n=1000 MP-20 test).
    """
    d = json.loads((ROOT / "phase9_results" / "index_benchmark_v2_native.json").read_text())
    ps = d["per_system"]
    order = ["cubic", "tetragonal", "hexagonal", "trigonal",
             "orthorhombic", "monoclinic", "triclinic"]
    abbr = ["Cubic", "Tetrag.", "Hexag.", "Trigon.", "Orthor.", "Monocl.", "Tricl."]
    ns = [ps[s]["n"] for s in order]
    strict = [ps[s]["strict_pct"] for s in order]
    consist = [ps[s]["consistent_pct"] for s in order]
    lenmae = [ps[s]["len_mae"] for s in order]
    lenmae_plot = [(v if v == v else 0.0) for v in lenmae]  # NaN (triclinic) -> 0
    overall = d["overall"]
    # two-line tick labels carry the per-system n inline, no below-axis collisions
    ticklabels = [f"{a}\nn={n}" for a, n in zip(abbr, ns)]

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.3), constrained_layout=True)
    x = np.arange(len(order))
    width = 0.4

    # (a) strict / consistent match rate
    ax = axes[0]
    ax.bar(x - width / 2, strict, width, color=CB["blue"], edgecolor="white",
           linewidth=0.6, label="Strict", zorder=3)
    ax.bar(x + width / 2, consist, width, color=CB["skyblue"], edgecolor="white",
           linewidth=0.6, label="Consistent", zorder=3)
    ax.axhline(overall["overall_strict_pct"], color=CB["red"], lw=1.0, ls="--", zorder=2,
               label=f"overall strict = {overall['overall_strict_pct']:.1f}%")
    ax.set_xticks(x)
    ax.set_xticklabels(ticklabels, fontsize=7)
    ax.set_ylabel("Indexing accuracy (%)")
    ax.set_title("(a) Per-system match rate", loc="left")
    ax.set_ylim(0, 100)
    _ygrid(ax)
    ax.legend(loc="upper right", borderaxespad=0.3)
    for i, v in enumerate(strict):
        ax.text(i - width / 2, v + 1.8, f"{v:.0f}", ha="center", fontsize=6.5,
                color="#222222")

    # (b) lattice-length MAE (Å)
    ax = axes[1]
    bar_colors = [CB["green"] if v <= 1.0 else CB["orange"] if v <= 2.0 else CB["red"]
                  for v in lenmae_plot]
    ax.bar(x, lenmae_plot, color=bar_colors, edgecolor="white", linewidth=0.6, zorder=3)
    ax.axhline(overall["v20_learned_head_len_mae"], color=CB["black"], lw=1.0, ls=":",
               zorder=2, label=f"v20 learned head = {overall['v20_learned_head_len_mae']:.2f} Å")
    ax.axhline(overall["overall_len_mae"], color=CB["red"], lw=1.0, ls="--", zorder=2,
               label=f"overall = {overall['overall_len_mae']:.2f} Å")
    ax.set_xticks(x)
    ax.set_xticklabels(ticklabels, fontsize=7)
    ax.set_ylabel("Lattice-length MAE (Å)")
    ax.set_title("(b) Per-system lattice error", loc="left")
    ax.set_ylim(0, 5.6)  # headroom for the tall trigonal bar + its value label
    _ygrid(ax)
    for i, v in enumerate(lenmae):
        if v != v:
            ax.text(i, 0.12, "n/a", ha="center", va="bottom", fontsize=7, color="gray")
        else:
            ax.text(i, v + 0.08, f"{v:.2f}", ha="center", va="bottom", fontsize=6.5,
                    color="#222222")

    # colour key for the MAE quality bands
    band_handles = [
        Patch(facecolor=CB["green"], label="≤ 1 Å"),
        Patch(facecolor=CB["orange"], label="1–2 Å"),
        Patch(facecolor=CB["red"], label="> 2 Å"),
    ]
    ax.legend(handles=ax.get_legend_handles_labels()[0] + band_handles,
              loc="upper left", borderaxespad=0.3, ncol=1)

    fig.suptitle("Classical Q-space indexer: wins on high-symmetry, fails on low-symmetry",
                 fontsize=10.5, fontweight="bold")
    _save(fig, "fig4_indexer_bench")


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

    # (display label, n-string, data, colour)
    systems = [
        ("PXRD-Diff\n(ours)", "n = 1000", ours, CB["blue"]),
        ("DiffractGPT",       "n = 1000", dgpt, CB["orange"]),
        ("PXRDnet",           "n = 20",   pxnt, CB["green"]),
    ]
    labels = [s[0] for s in systems]
    nstr = [s[1] for s in systems]
    colors = [s[3] for s in systems]

    metrics = [
        ("Match rate (%)",         "(a) Match rate",  "match_rate (StructureMatcher)", True),
        ("All-correct (%)",        "(b) All-correct", "headline_all_correct",          True),
        ("Pearson (pred vs true)", "(c) Pearson",     "pearson_mean",                  False),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(7.4, 3.0), constrained_layout=True)
    x = np.arange(len(systems))

    for ax, (ylabel, title, key, pct) in zip(axes, metrics):
        vals = [s[2][key] * (100 if pct else 1) for s in systems]
        ax.bar(x, vals, color=colors, edgecolor="white", linewidth=0.8,
               width=0.66, zorder=3)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left")
        _ygrid(ax)
        _headroom(ax, vals, frac=0.22)
        _bar_labels(ax, x, vals, fmt=("{:.1f}" if pct else "{:.2f}"), fontsize=8)
        # sample-size row, parked just under the axis, clear of the tick labels
        for xi, n in zip(x, nstr):
            ax.annotate(n, xy=(xi, 0), xytext=(0, -26), textcoords="offset points",
                        ha="center", va="top", fontsize=6.5, color=LABEL_GREY,
                        annotation_clip=False)

    fig.suptitle("Open generative baselines on the same MP-20 evaluation harness",
                 fontsize=10.5, fontweight="bold")
    _save(fig, "fig5_threeway_headline")


if __name__ == "__main__":
    fig1_ablation()
    fig2_training_curves()
    fig3_diffpxrd_validation()
    fig4_indexer_bench()
    fig5_threeway_headline()
    print("done.")
