"""Generate publication figures for PXRD-Diff paper.

Outputs (all 600 dpi PNG + matched PDF; IUCr requires >= 600 dpi bitmaps):
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
    "savefig.dpi": 600,
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
    # JAC R1: no in-figure titles; the caption carries the description
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
    _save(fig, "fig2_training_curves")


def fig3_diffpxrd_validation():
    """JAC R1 revision: DiffPXRD vs pymatgen.XRDCalculator on the first 1000
    MP-20 test structures, for the submitted (legacy) form-factor expression and
    the corrected pymatgen-convention expression. Data:
    paper/submissions/JAC-R1/analysis/simulator_revalidation.json
    (produced by paper/submissions/JAC-R1/analysis/revalidate_simulator.py)."""
    src = ROOT / "submissions" / "JAC-R1" / "analysis" / "simulator_revalidation.json"
    d = json.loads(src.read_text(encoding="utf-8"))
    legacy = np.array([r["legacy"] for r in d["rows"]])
    fixed = np.array([r["pymatgen"] for r in d["rows"]])
    n = len(legacy)

    fig, ax = plt.subplots(figsize=(4.4, 2.8), constrained_layout=True)
    bins = np.linspace(0.80, 1.00, 41)
    ax.hist(legacy.clip(0.80, 1.0), bins=bins, color="#9A9A9A", edgecolor="white",
            lw=0.5, zorder=3, alpha=0.85,
            label=f"submitted expression: mean {legacy.mean():.3f}, median {np.median(legacy):.3f}")
    ax.hist(fixed.clip(0.80, 1.0), bins=bins, color=CB["blue"], edgecolor="white",
            lw=0.5, zorder=4, alpha=0.85,
            label=f"corrected expression: mean {fixed.mean():.3f}, median {np.median(fixed):.3f}")
    ax.set_xlabel("Pearson correlation vs pymatgen.XRDCalculator")
    ax.set_ylabel(f"Count (n = {n} structures)")
    ax.set_title("DiffPXRD vs reference simulator", loc="left")
    ax.set_xlim(0.795, 1.005)
    ax.grid(linestyle="-")
    ax.set_axisbelow(True)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    below = int((legacy < 0.80).sum()), int((fixed < 0.80).sum())
    ax.text(0.80, ax.get_ylim()[1] * 0.97,
            f"values below 0.80 pooled into the first bin: {below[0]} (submitted), {below[1]} (corrected)\n"
            f"corrected ≥ submitted on {int((fixed >= legacy).sum())}/{n} structures",
            fontsize=6.5, color=LABEL_GREY, ha="left", va="top")
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.80), fontsize=6.5)
    _save(fig, "fig3_diffpxrd_validation")


def fig4_indexer_bench():
    """Per-crystal-system strict% and len_MAE from the v2 native indexer.

    Source: paper/phase9_results/index_benchmark_v2_native.json (n=1000 MP-20 test).
    """
    d = json.loads((ROOT / "phase9_results" / "index_benchmark_v2_native.json").read_text())
    ps = d["per_system"]
    # JAC R1 (referee 1, 5iii): the same benchmark with the crystal system NOT supplied
    du = json.loads((ROOT / "phase15_results" / "index_cells_test1000_unknown.json").read_text())
    psu = du["per_system"]
    order = ["cubic", "tetragonal", "hexagonal", "trigonal",
             "orthorhombic", "monoclinic", "triclinic"]
    abbr = ["Cubic", "Tetrag.", "Hexag.", "Trigon.", "Orthor.", "Monocl.", "Tricl."]
    ns = [ps[s]["n"] for s in order]
    strict = [ps[s]["strict_pct"] for s in order]
    strict_unk = [psu[s]["strict_pct"] if s in psu else 0.0 for s in order]
    consist = [ps[s]["consistent_pct"] for s in order]
    lenmae = [ps[s]["len_mae"] for s in order]
    lenmae_plot = [(v if v == v else 0.0) for v in lenmae]  # NaN (triclinic) -> 0
    overall = d["overall"]
    # two-line tick labels carry the per-system n inline, no below-axis collisions
    ticklabels = [f"{a}\nn={n}" for a, n in zip(abbr, ns)]

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.3), constrained_layout=True)
    x = np.arange(len(order))
    width = 0.27

    # (a) strict (system given / unknown) and consistent match rate
    ax = axes[0]
    ax.bar(x - width, strict, width, color=CB["blue"], edgecolor="white",
           linewidth=0.6, label="Strict, system given", zorder=3)
    ax.bar(x, strict_unk, width, color=CB["orange"], edgecolor="white",
           linewidth=0.6, label="Strict, system unknown", zorder=3)
    ax.bar(x + width, consist, width, color=CB["skyblue"], edgecolor="white",
           linewidth=0.6, label="Consistent, system given", zorder=3)
    ax.axhline(overall["overall_strict_pct"], color=CB["red"], lw=1.0, ls="--", zorder=2,
               label=f"overall strict = {overall['overall_strict_pct']:.1f}% / "
                     f"{du['overall']['overall_strict_pct']:.1f}%")
    ax.set_xticks(x)
    ax.set_xticklabels(ticklabels, fontsize=7)
    ax.set_ylabel("Indexing accuracy (%)")
    ax.set_title("(a) Per-system match rate", loc="left")
    ax.set_ylim(0, 128)          # headroom so the legend clears the hexagonal bars
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    _ygrid(ax)
    ax.legend(loc="upper right", borderaxespad=0.3, fontsize=7)
    for i, (v, u) in enumerate(zip(strict, strict_unk)):
        ax.text(i - width, v + 1.8, f"{v:.0f}", ha="center", fontsize=6, color="#222222")
        ax.text(i, u + 1.8, f"{u:.0f}", ha="center", fontsize=6, color="#222222")

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

    _save(fig, "fig4_indexer_bench")


def _wilson(k, n, z=1.959964):
    """Wilson 95 % interval for k successes in n trials, as percentages."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return 100 * (centre - half), 100 * (centre + half)


def fig5_threeway_headline():
    """Head-to-head on one harness: ours (v22 learned head, 3 sampling seeds pooled) vs
    DiffractGPT (n = 990 scored of 1000), PXRDnet (n = 20) and deCIFer (n = 298 scored of 300),
    with 95 % Wilson intervals on the rate panels (JAC R1 pre-upload audit).

    Sources:
      paper/phase15_results/v22_learned_pooled.json     (ours: gpu_v22_jac, learned head,
                                                          3 x 1000, JAC R1 retrain)
      paper/phase9_results/baseline_diffractgpt_n1000.json
      paper/phase9_results/baseline_pxrdnet_sinc100_n20.json
      paper/phase13_results/baselines/decifer_n300.json
    (The submitted version plotted phase9_results/p9_idxlat_n1000.json, the v21 indexer row.)
    """
    def load(name):
        return json.loads((ROOT / "phase9_results" / name).read_text())

    ours = json.loads((ROOT / "phase15_results" / "v22_learned_pooled.json").read_text())
    dgpt = load("baseline_diffractgpt_n1000.json")
    pxnt = load("baseline_pxrdnet_sinc100_n20.json")
    dcif = json.loads((ROOT / "phase13_results" / "baselines" / "decifer_n300.json").read_text())

    # (display label, n-string, data, colour)
    systems = [  # DGpt = DiffractGPT (abbreviation defined in §1); n = 3000 is 3 seeds × 1000
        ("PXRD-Diff\n(ours)", "n = 3000", ours, CB["blue"]),
        ("DGpt",              "n = 990",  dgpt, CB["orange"]),
        ("PXRDnet",           "n = 20",   pxnt, CB["green"]),
        ("deCIFer",           "n = 298",  dcif, CB["purple"]),
    ]
    labels = [s[0] for s in systems]
    nstr = [s[1] for s in systems]
    colors = [s[3] for s in systems]
    ns = [int(s[2]["n"]) for s in systems]

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
        tops = list(vals)
        if pct:
            # Wilson 95 % intervals from the scored counts
            lo, hi = zip(*[_wilson(round(v / 100 * n), n) for v, n in zip(vals, ns)])
            yerr = np.array([np.array(vals) - np.array(lo), np.array(hi) - np.array(vals)])
            ax.errorbar(x, vals, yerr=yerr, fmt="none", ecolor="#333333", elinewidth=0.9,
                        capsize=2.5, zorder=4)
            tops = list(hi)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=7.5)
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left")
        _ygrid(ax)
        _headroom(ax, tops, frac=0.22)
        _bar_labels(ax, x, tops, fmt=("{:.1f}" if pct else "{:.2f}"), fontsize=7)
        if pct:  # relabel with the point estimate, placed above the interval cap
            for t in list(ax.texts):
                t.remove()
            span = ax.get_ylim()[1]
            for xi, v, tp in zip(x, vals, tops):
                ax.text(xi, tp + span * 0.015, f"{v:.1f}", ha="center", va="bottom",
                        fontsize=7, color="#222222")
        # sample-size row, parked just under the axis, clear of the tick labels
        for xi, n in zip(x, nstr):
            ax.annotate(n, xy=(xi, 0), xytext=(0, -26), textcoords="offset points",
                        ha="center", va="top", fontsize=6.5, color=LABEL_GREY,
                        annotation_clip=False)

    _save(fig, "fig5_threeway_headline")


if __name__ == "__main__":
    fig1_ablation()
    fig2_training_curves()
    fig3_diffpxrd_validation()
    fig4_indexer_bench()
    fig5_threeway_headline()
    print("done.")
