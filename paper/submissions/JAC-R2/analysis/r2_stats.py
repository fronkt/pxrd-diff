"""JAC R2 statistics from the phase17 box run (and the v22 records for comparison).

Pooled Wilson CIs per arm, paired McNemar between arms (pairs = material x seed),
per-crystal-system match counts, and, for Q3, the match rate when the number of
formula units must be estimated from the cell volume (rows whose estimate is wrong
are counted as misses; analysis/z_from_volume.json).

Usage: python r2_stats.py REPO OUT_JSON
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

REPO = Path(sys.argv[1])
sys.path.insert(0, str(REPO / "scripts"))
from stats_ci import mcnemar, wilson   # noqa: E402

P17 = REPO / "paper/phase17_results"
P15 = REPO / "paper/phase15_results"
SEEDS = (0, 1, 2)
ARMS = {
    "v22_learned (R1, clean cell)": [P15 / f"v22_learned_s{s}.per_sample.jsonl" for s in SEEDS],
    "v22_learned_noref (D1)": [P17 / f"v22_learned_noref_s{s}.per_sample.jsonl" for s in SEEDS],
    "v22_oracle": [P15 / f"v22_oracle_s{s}.per_sample.jsonl" for s in SEEDS],
    "v22_indexer": [P15 / f"v22_indexer_s{s}.per_sample.jsonl" for s in SEEDS],
    "v23_learned": [P17 / f"v23_learned_s{s}.per_sample.jsonl" for s in SEEDS],
    "v23_oracle": [P17 / f"v23_oracle_s{s}.per_sample.jsonl" for s in SEEDS],
    "v23_indexer": [P17 / f"v23_indexer_s{s}.per_sample.jsonl" for s in SEEDS],
    "v23_indexer_unknown": [P17 / f"v23_indexer_unknown_s{s}.per_sample.jsonl" for s in SEEDS],
}
PAIRS = [
    ("v22_learned_noref (D1)", "v22_learned (R1, clean cell)"),
    ("v23_oracle", "v23_learned"),
    ("v23_indexer", "v23_learned"),
    ("v23_oracle", "v23_indexer"),
    ("v23_indexer_unknown", "v23_indexer"),
    ("v23_indexer_unknown", "v23_learned"),
    ("v23_learned", "v22_learned (R1, clean cell)"),
    ("v23_oracle", "v22_oracle"),
    ("v23_indexer", "v22_indexer"),
]


def load(files):
    rows = {}
    for f in files:
        if not f.exists():
            return None
        for line in f.read_text().splitlines():
            r = json.loads(line)
            rows[(r["material_id"], r["seed"])] = r
    return rows


z = {r["mid"]: r for r in json.loads((REPO / "paper/submissions/JAC-R2/analysis/z_from_volume.json")
                                     .read_text())["rows"]}
system = {mid: r["system"] for mid, r in z.items()}

out = {"arms": {}, "pairs": {}}
data = {name: load(files) for name, files in ARMS.items()}
for name, rows in data.items():
    if rows is None:
        out["arms"][name] = "missing"
        continue
    res = {}
    for flag in ("match", "all_correct"):
        x = sum(bool(r[flag]) for r in rows.values())
        p, lo, hi = wilson(x, len(rows))
        res[flag] = dict(x=x, n=len(rows), pct=round(p, 2), ci=[round(lo, 2), round(hi, 2)])
    res["match_by_system"] = dict(Counter(system.get(mid, "?") for (mid, _), r in rows.items() if r["match"]))
    # Q3: formula units estimated from the cell volume (true cell for the oracle,
    # indexed cell for the indexer); learned-head rows have no cell to estimate from.
    if "oracle" in name:
        ok = lambda mid: z[mid]["z_est_true_cell"] == z[mid]["z_true"]
    elif "indexer" in name and "unknown" not in name:
        ok = lambda mid: (z[mid].get("indexed") and z[mid].get("idx_same_volume")
                          and z[mid]["z_est_idx_cell"] == z[mid]["z_true"])
    else:
        ok = None
    if ok is not None:
        x = sum(bool(r["match"]) and bool(ok(mid)) for (mid, _), r in rows.items())
        p, lo, hi = wilson(x, len(rows))
        res["match_with_estimated_Z"] = dict(x=x, n=len(rows), pct=round(p, 2),
                                             ci=[round(lo, 2), round(hi, 2)])
    out["arms"][name] = res

for a, b in PAIRS:
    ra, rb = data[a], data[b]
    if ra is None or rb is None:
        continue
    keys = sorted(set(ra) & set(rb))
    fixed = sum(ra[k]["match"] and not rb[k]["match"] for k in keys)
    broken = sum(rb[k]["match"] and not ra[k]["match"] for k in keys)
    _, p = mcnemar(fixed, broken)
    out["pairs"][f"{a} vs {b}"] = dict(n_pairs=len(keys), a_only=fixed, b_only=broken, p=p)

Path(sys.argv[2]).write_text(json.dumps(out, indent=1))
for name, res in out["arms"].items():
    if res == "missing":
        print(f"{name:32s} missing")
        continue
    m = res["match"]
    extra = (f"  estimated-Z {res['match_with_estimated_Z']['pct']:.2f}%"
             if "match_with_estimated_Z" in res else "")
    print(f"{name:32s} match {m['x']:4d}/{m['n']} = {m['pct']:5.2f}% [{m['ci'][0]}, {m['ci'][1]}]"
          f"  all-correct {res['all_correct']['pct']:.2f}%{extra}  by system {res['match_by_system']}")
for k, v in out["pairs"].items():
    print(f"{k:60s} {v['a_only']:4d} / {v['b_only']:4d}  p={v['p']:.2g}")
