"""Generate paper plots from training logs and ablation results.

Produces:
  - loss_curves.png: training curves for v6 (no debye, no fixes), v10 (lattice fix), v13 (full)
  - ablation_bar.png: match rate / Pearson bar chart for v10/v11/v13

Usage:
  python scripts/06_paper_plots.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def parse_log(path: Path, want_run: str | None = None) -> dict:
    """Parse a training log. If want_run given, only keep lines from that run.

    Multi-run logs use '=== Starting <name> ===' markers between runs.
    """
    pat = re.compile(r"step=\s*(\d+)\s+loss=([\d.]+)\s+coord=([\d.]+)\s+lat=([\d.]+)\s+aux=([\d.]+)")
    start_pat = re.compile(r"=== Starting (\w+)")
    steps, losses, coords, lats, auxs = [], [], [], [], []
    current_run = None
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            ms = start_pat.search(line)
            if ms:
                current_run = ms.group(1)
                continue
            if want_run is not None and current_run != want_run:
                continue
            m = pat.search(line)
            if m:
                steps.append(int(m.group(1)))
                losses.append(float(m.group(2)))
                coords.append(float(m.group(3)))
                lats.append(float(m.group(4)))
                auxs.append(float(m.group(5)))
    return {
        "step": np.array(steps),
        "loss": np.array(losses),
        "coord": np.array(coords),
        "lat": np.array(lats),
        "aux": np.array(auxs),
    }


def loss_curves():
    runs = [
        ("v10 (eps prediction, lattice fix)", ROOT / "runs" / "v10_v11.log", "gpu_v10"),
        ("v13 (x0 residual, λ_debye=1)", ROOT / "runs" / "gpu_v13_train.log", None),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.5), sharex=True)
    colors = ["#1f77b4", "#d62728"]
    for (name, path, want_run), c in zip(runs, colors):
        if not path.exists():
            print(f"missing: {path}")
            continue
        d = parse_log(path, want_run=want_run)
        if len(d["step"]) == 0:
            continue
        axes[0].plot(d["step"], d["coord"], label=name, color=c, lw=1.5)
        axes[1].plot(d["step"], d["lat"], label=name, color=c, lw=1.5)
        axes[2].plot(d["step"], d["aux"], label=name, color=c, lw=1.5)

    axes[0].set_title("Coord noise/x0 loss")
    axes[0].set_xlabel("step")
    axes[0].set_ylabel("MSE")
    axes[0].axhline(1.0, color="gray", ls="--", lw=0.8, label="eps random baseline")
    axes[0].axhline(1/12, color="gray", ls=":", lw=0.8, label="x0 random baseline (1/12)")
    axes[0].legend(fontsize=8, loc="upper right")
    axes[0].set_ylim(0, 1.2)

    axes[1].set_title("Lattice loss (key win)")
    axes[1].set_xlabel("step")
    axes[1].axhline(1.0, color="gray", ls="--", lw=0.8, label="random baseline")
    axes[1].legend(fontsize=8, loc="upper right")
    axes[1].set_ylim(0, 1.2)

    axes[2].set_title("Aux loss (PXRD→lattice from global)")
    axes[2].set_xlabel("step")
    axes[2].set_ylim(0, 1.5)

    plt.suptitle("Training loss curves: lattice fix dramatic, coord plateau breaks with x0 residual",
                 fontsize=11)
    plt.tight_layout()
    out = ROOT / "runs" / "loss_curves.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")


def ablation_bar():
    """Headline ablation: match rate and Pearson at n=1000 (true lattice)."""
    labels = ["v10\n(eps, λ=0)", "v11\n(eps, λ=1)", "v13\n(x0+lat-fix\n+ Debye λ=1)"]
    match_pct = [1.10, 1.50, 1.91]
    pearson = [0.362, 0.367, 0.426]
    colors = ["#888888", "#888888", "#d62728"]

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    x = np.arange(len(labels))

    axes[0].bar(x, match_pct, color=colors)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, fontsize=9)
    axes[0].set_ylabel("StructureMatcher match rate (%)")
    axes[0].set_title(f"Match rate (n=1000, true lattice)")
    for xi, v in zip(x, match_pct):
        axes[0].text(xi, v + 0.05, f"{v:.2f}%", ha="center", fontsize=9)

    axes[1].bar(x, pearson, color=colors)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, fontsize=9)
    axes[1].set_ylabel("PXRD Pearson correlation")
    axes[1].set_title(f"PXRD Pearson (n=1000, true lattice)")
    for xi, v in zip(x, pearson):
        axes[1].text(xi, v + 0.005, f"{v:.3f}", ha="center", fontsize=9)

    plt.suptitle("Headline ablation: x0 residual + lattice fix + Debye loss vs eps baseline",
                 fontsize=11)
    plt.tight_layout()
    out = ROOT / "runs" / "ablation_bar.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")


def main():
    loss_curves()
    ablation_bar()


if __name__ == "__main__":
    main()
