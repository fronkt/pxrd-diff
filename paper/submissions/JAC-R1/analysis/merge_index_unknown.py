"""Merge the sharded `09_index_benchmark.py --system-mode unknown` chunks into one
benchmark JSON (same schema as index_benchmark_v2_native.json) and compare with
the given-system benchmark on the same 1000 test structures (JAC R1, R1.5iii).

Usage: python paper/submissions/JAC-R1/analysis/merge_index_unknown.py
Writes: paper/phase15_results/index_cells_test1000_unknown.json
        paper/phase15_results/index_unknown_vs_given.json
"""
import glob
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
A = ROOT / "paper/submissions/JAC-R1/analysis"
OUT = ROOT / "paper/phase15_results"
OUT.mkdir(parents=True, exist_ok=True)
SYSTEMS = ["cubic", "hexagonal", "trigonal", "tetragonal", "orthorhombic", "monoclinic", "triclinic"]
N_TOTAL = 1000

rows, n_seen = [], 0
for f in sorted(glob.glob(str(A / "index_unknown_chunk*.json"))):
    d = json.load(open(f))
    rows += d["rows"]
    n_seen += d["overall"]["n"]
assert n_seen == N_TOTAL, f"chunks cover {n_seen} rows, expected {N_TOTAL}"
mids = [r["mid"] for r in rows]
assert len(set(mids)) == len(mids), "duplicate mids across chunks"

given = json.load(open(ROOT / "paper/phase9_results/index_benchmark_v2_native.json"))
g_by = {r["mid"]: r for r in given["rows"]}

success = lambda r: r["vol_err"] < 0.05 and r["len_mae"] < 0.3


def summarise(rows, n_total, per_sys_counts):
    per_sys = defaultdict(list)
    for r in rows:
        per_sys[r["system"]].append(r)
    out = {}
    for s in SYSTEMS:
        recs = per_sys.get(s, [])
        n_sys = per_sys_counts.get(s, 0)
        if n_sys == 0:
            continue
        out[s] = dict(
            n=n_sys, n_indexed=len(recs),
            strict_pct=round(100.0 * sum(success(r) for r in recs) / n_sys, 1),
            consistent_pct=round(100.0 * sum(r["consistent"] for r in recs) / n_sys, 1),
            len_mae=round(float(np.mean([r["len_mae"] for r in recs])), 4) if recs else None,
            ang_mae=round(float(np.mean([r["ang_mae"] for r in recs])), 3) if recs else None,
            system_correct_pct=(round(100.0 * sum(r.get("system_correct", True) for r in recs) / n_sys, 1)),
        )
    overall = dict(
        n=n_total, n_indexed=len(rows),
        overall_strict_pct=round(100.0 * sum(success(r) for r in rows) / n_total, 1),
        overall_consistent_pct=round(100.0 * sum(r["consistent"] for r in rows) / n_total, 1),
        overall_len_mae=round(float(np.mean([r["len_mae"] for r in rows])), 4) if rows else None,
        system_correct_pct=round(100.0 * sum(r.get("system_correct", True) for r in rows) / n_total, 1),
    )
    return overall, out


# true per-system counts over the 1000 (from the given-system run, which records every
# indexable row; rows the given-system run skipped are counted from the unknown run)
counts = defaultdict(int)
seen = set()
for r in given["rows"] + rows:
    if r["mid"] in seen:
        continue
    seen.add(r["mid"])
    counts[r["system"]] += 1

u_overall, u_sys = summarise(rows, N_TOTAL, counts)
g_overall, g_sys = summarise(given["rows"], N_TOTAL, counts)
u_overall["system_mode"] = "unknown"
g_overall["system_mode"] = "given"

json.dump(dict(overall=u_overall, per_system=u_sys, rows=rows), open(OUT / "index_cells_test1000_unknown.json", "w"), indent=1)

# paired view: same mid, given vs unknown
paired = []
for r in rows:
    g = g_by.get(r["mid"])
    if g is None:
        continue
    paired.append(dict(mid=r["mid"], system=r["system"], pred_system=r["pred_system"],
                       system_correct=r["system_correct"],
                       strict_given=success(g), strict_unknown=success(r),
                       len_mae_given=g["len_mae"], len_mae_unknown=r["len_mae"]))
b = sum(p["strict_unknown"] and not p["strict_given"] for p in paired)
c = sum(p["strict_given"] and not p["strict_unknown"] for p in paired)
cmp = dict(
    n_paired=len(paired),
    strict_given=sum(p["strict_given"] for p in paired),
    strict_unknown=sum(p["strict_unknown"] for p in paired),
    unknown_gains=b, unknown_losses=c,
    system_correct_pct=round(100.0 * sum(p["system_correct"] for p in paired) / len(paired), 1),
    given=dict(overall=g_overall, per_system=g_sys),
    unknown=dict(overall=u_overall, per_system=u_sys),
)
json.dump(cmp, open(OUT / "index_unknown_vs_given.json", "w"), indent=1)

print(f"paired {len(paired)}; strict given {cmp['strict_given']} vs unknown {cmp['strict_unknown']} "
      f"(unknown gains {b}, losses {c}); system correct {cmp['system_correct_pct']} %")
print(f"{'system':<13}{'n':>5}{'given strict%':>15}{'unknown strict%':>17}{'sys-correct%':>14}{'lenMAE given':>14}{'lenMAE unk':>12}")
for s in SYSTEMS:
    if s not in u_sys and s not in g_sys:
        continue
    g = g_sys.get(s, {}); u = u_sys.get(s, {})
    print(f"{s:<13}{counts[s]:>5}{g.get('strict_pct', float('nan')):>15}{u.get('strict_pct', float('nan')):>17}"
          f"{u.get('system_correct_pct', float('nan')):>14}{g.get('len_mae', float('nan')):>14}{u.get('len_mae', float('nan')):>12}")
print(f"OVERALL   given strict {g_overall['overall_strict_pct']} %  unknown strict {u_overall['overall_strict_pct']} %  "
      f"unknown indexed {u_overall['n_indexed']}/{N_TOTAL}")
