#!/usr/bin/env python
"""Phase 12 statistics helper for PXRD-Diff match-rate claims.

Two jobs the aggregate result JSONs cannot do on their own:

1. Confidence intervals on a proportion (Wilson score interval), so every
   match / all-correct rate in the paper can be reported as `p [lo, hi]`.
2. A *paired* McNemar test across two lattice sources (learned head vs
   classical indexer) on the SAME materials, which is the correct test for
   "did the drop-in change the headline?" — the unpaired two-proportion test
   throws away the pairing and is underpowered.

Inputs are the per-structure JSONL files emitted by `03_sample.py
--per-sample-json` (one row per material: material_id, match, all_correct, ...).

Usage
-----
# CI on a single run, on a chosen flag (default: match)
python scripts/stats_ci.py ci runA_seed0.jsonl --flag match

# Pooled CI across seeds for one mode
python scripts/stats_ci.py ci idx_seed0.jsonl idx_seed1.jsonl idx_seed2.jsonl

# Paired McNemar: learned-head vs indexer, matched by material_id (per seed,
# then pooled). Pass the two modes as two comma-separated file groups.
python scripts/stats_ci.py mcnemar \
    --a learned_s0.jsonl,learned_s1.jsonl,learned_s2.jsonl \
    --b idx_s0.jsonl,idx_s1.jsonl,idx_s2.jsonl --flag match

No third-party deps beyond the stdlib (math) so it runs anywhere.
"""
from __future__ import annotations

import argparse
import json
from math import sqrt, erf


# ---------------------------------------------------------------- proportions
def wilson(x: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    """Wilson score interval. Returns (point%, lo%, hi%)."""
    if n == 0:
        return 0.0, 0.0, 0.0
    p = x / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = (z * sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / d
    return 100 * p, 100 * (centre - half), 100 * (centre + half)


def two_proportion(x1: int, n1: int, x2: int, n2: int) -> tuple[float, float]:
    """Unpaired pooled two-proportion z-test. Returns (z, two-sided p)."""
    p1, p2 = x1 / n1, x2 / n2
    p = (x1 + x2) / (n1 + n2)
    se = sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    if se == 0:
        return 0.0, 1.0
    z = (p2 - p1) / se
    pval = 2 * (1 - 0.5 * (1 + erf(abs(z) / sqrt(2))))
    return z, pval


def mcnemar(b: int, c: int) -> tuple[float, float]:
    """McNemar test on discordant counts b, c (paired binary outcomes).

    Uses the exact binomial p-value (robust for small discordant totals),
    two-sided. b = #(A wrong, B right), c = #(A right, B wrong).
    """
    n = b + c
    if n == 0:
        return 0.0, 1.0
    # exact two-sided binomial against p=0.5
    from math import comb
    k = min(b, c)
    tail = sum(comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    pval = min(1.0, 2 * tail)
    # continuity-corrected chi-square statistic for reference
    chi2 = (abs(b - c) - 1) ** 2 / n if n > 0 else 0.0
    return chi2, pval


# ----------------------------------------------------------------- file utils
def load(path: str) -> dict[str, dict]:
    """Load a per-sample JSONL into {material_id: row}."""
    rows = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            rows[r["material_id"]] = r
    return rows


def count_flag(rows: dict[str, dict], flag: str) -> tuple[int, int]:
    n = len(rows)
    x = sum(1 for r in rows.values() if r.get(flag))
    return x, n


# ----------------------------------------------------------------------- main
def cmd_ci(args):
    total_x = total_n = 0
    print(f"flag = {args.flag}")
    for path in args.files:
        rows = load(path)
        x, n = count_flag(rows, args.flag)
        p, lo, hi = wilson(x, n)
        print(f"  {path:40s}  {x:4d}/{n:<4d}  {p:5.1f}% [{lo:4.1f}, {hi:4.1f}]")
        total_x += x
        total_n += n
    if len(args.files) > 1:
        p, lo, hi = wilson(total_x, total_n)
        print(f"  {'POOLED':40s}  {total_x:4d}/{total_n:<4d}  "
              f"{p:5.1f}% [{lo:4.1f}, {hi:4.1f}]")


def cmd_mcnemar(args):
    a_files = args.a.split(",")
    b_files = args.b.split(",")
    if len(a_files) != len(b_files):
        raise SystemExit("--a and --b must list the same number of seed files")

    tot_b = tot_c = 0
    xa = na = xb = nb = 0
    print(f"flag = {args.flag}   (A = learned head, B = indexer)")
    print(f"{'seed-pair':40s}  disc(b,c)   McNemar p")
    for fa, fb in zip(a_files, b_files):
        ra, rb = load(fa), load(fb)
        common = sorted(set(ra) & set(rb))
        b = sum(1 for m in common if not ra[m].get(args.flag) and rb[m].get(args.flag))
        c = sum(1 for m in common if ra[m].get(args.flag) and not rb[m].get(args.flag))
        _, pv = mcnemar(b, c)
        print(f"  {fa.split('/')[-1]:38s}  ({b},{c})       p={pv:.3f}  (n_common={len(common)})")
        tot_b += b
        tot_c += c
        xa += sum(1 for m in common if ra[m].get(args.flag)); na += len(common)
        xb += sum(1 for m in common if rb[m].get(args.flag)); nb += len(common)

    chi2, pv = mcnemar(tot_b, tot_c)
    print("\n--- Pooled across seeds ---")
    pa, la, ha = wilson(xa, na)
    pb, lb, hb = wilson(xb, nb)
    print(f"  A (learned): {xa}/{na}  {pa:.1f}% [{la:.1f}, {ha:.1f}]")
    print(f"  B (indexer): {xb}/{nb}  {pb:.1f}% [{lb:.1f}, {hb:.1f}]")
    print(f"  discordant b(A-wrong,B-right)={tot_b}  c(A-right,B-wrong)={tot_c}")
    print(f"  McNemar exact two-sided p = {pv:.4f}   (chi2_cc={chi2:.2f})")
    verdict = "SIGNIFICANT" if pv < 0.05 else "not significant"
    print(f"  => paired indexer effect is {verdict} at alpha=0.05")


def main():
    ap = argparse.ArgumentParser(description="PXRD-Diff match-rate statistics")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_ci = sub.add_parser("ci", help="Wilson CI on a flag, per file + pooled")
    p_ci.add_argument("files", nargs="+")
    p_ci.add_argument("--flag", default="match",
                      choices=["match", "all_correct", "sg_match@0.1", "composition_ok"])
    p_ci.set_defaults(func=cmd_ci)

    p_mc = sub.add_parser("mcnemar", help="paired McNemar A vs B across seeds")
    p_mc.add_argument("--a", required=True, help="comma-separated learned-head JSONLs")
    p_mc.add_argument("--b", required=True, help="comma-separated indexer JSONLs")
    p_mc.add_argument("--flag", default="match",
                      choices=["match", "all_correct", "sg_match@0.1", "composition_ok"])
    p_mc.set_defaults(func=cmd_mcnemar)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
