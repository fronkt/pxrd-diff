"""Summarise the JAC-R1 GPU runs (retrained v22 checkpoint) against the v1 (gpu_v21_p9) rows.

Reads the per-structure JSONLs written by the box pipeline into paper/phase15_results/
(v22_{learned,indexer,oracle,pool_gpool,pool_attnpool,pool_peaks,indexer_unknown}_s{0,1,2})
and the v1 Phase-12 files, and prints:
  * pooled match / all-correct rates with Wilson 95 % CIs per lattice source (v1 and v22),
  * the covered-only indexer rate (rows the indexer actually covered; fallback rows are misses),
  * paired McNemar tests on the SAME material_id x seed: learned vs each other source (v22),
    and v1 vs v22 for the learned head and the indexer (did the retrain move the numbers?).
Missing sources are skipped, so it can run before the pipeline has finished.

Usage: python paper/submissions/JAC-R1/analysis/summarise_v22.py
Writes: paper/phase15_results/v22_summary.json
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "scripts"))
from stats_ci import load, mcnemar, wilson  # noqa: E402

R15 = ROOT / "paper/phase15_results"
P12 = ROOT / "paper/phase9_results/phase12_multiseed"
SEEDS = [0, 1, 2]
V22_SOURCES = ["learned", "indexer", "oracle", "pool_gpool", "pool_attnpool", "pool_peaks", "indexer_unknown"]
V1_ORACLE_COUNTS = {0: 41, 1: 49, 2: 45}   # tasks/todo.md Phase 13 item 2 (per-sample files not kept)


def read_source(pattern):
    """pattern with {s} -> {seed: {mid: row}} for the seeds whose file exists."""
    out = {}
    for s in SEEDS:
        p = Path(str(pattern).format(s=s))
        if p.exists():
            out[s] = load(str(p))
    return out


def pooled(rows_by_seed, flag="match", covered_only=False):
    x = n = 0
    per_seed = {}
    for s, rows in rows_by_seed.items():
        rs = [r for r in rows.values() if not (covered_only and r.get("index_fallback_miss"))]
        xs = sum(bool(r.get(flag)) for r in rs)
        per_seed[s] = (xs, len(rs))
        x += xs
        n += len(rs)
    if n == 0:
        return None
    p, lo, hi = wilson(x, n)   # already in percent
    return dict(x=x, n=n, rate=round(p, 2), ci=[round(lo, 2), round(hi, 2)],
                per_seed={s: f"{a}/{b}" for s, (a, b) in per_seed.items()})


def paired(a_by_seed, b_by_seed, flag="match"):
    """McNemar on the same (seed, material_id): b = A miss & B hit, c = A hit & B miss."""
    b = c = both = neither = 0
    for s in a_by_seed:
        if s not in b_by_seed:
            continue
        for mid, ra in a_by_seed[s].items():
            rb = b_by_seed[s].get(mid)
            if rb is None:
                continue
            fa, fb = bool(ra.get(flag)), bool(rb.get(flag))
            if fa and fb:
                both += 1
            elif fa:
                c += 1
            elif fb:
                b += 1
            else:
                neither += 1
    if b + c + both + neither == 0:
        return None
    chi2, p = mcnemar(b, c)
    return dict(n_pairs=b + c + both + neither, both=both, b_gain=b, c_loss=c, chi2=round(chi2, 3), p=p)


def main():
    v22 = {src: read_source(R15 / f"v22_{src}_s{{s}}.per_sample.jsonl") for src in V22_SOURCES}
    v1 = {"learned": read_source(P12 / "learned_s{s}.per_sample.jsonl"),
          "indexer": read_source(P12 / "indexer_s{s}.per_sample.jsonl")}
    out = {"v1": {}, "v22": {}, "paired_v22": {}, "paired_v1_vs_v22": {}}

    for src, rows in v1.items():
        out["v1"][src] = dict(match=pooled(rows), all_correct=pooled(rows, "all_correct"))
    x = sum(V1_ORACLE_COUNTS.values())
    p, lo, hi = wilson(x, 3000)
    out["v1"]["oracle"] = dict(match=dict(x=x, n=3000, rate=round(p, 2), ci=[round(lo, 2), round(hi, 2)],
                                          per_seed={s: f"{c}/1000" for s, c in V1_ORACLE_COUNTS.items()}))

    for src, rows in v22.items():
        if not rows:
            continue
        d = dict(seeds=sorted(rows), match=pooled(rows), all_correct=pooled(rows, "all_correct"))
        if any(r.get("index_fallback_miss") is not None for rs in rows.values() for r in rs.values()):
            d["match_covered_only"] = pooled(rows, covered_only=True)
        out["v22"][src] = d

    if v22["learned"]:
        for src in V22_SOURCES[1:]:
            if v22[src]:
                out["paired_v22"][f"learned_vs_{src}"] = paired(v22["learned"], v22[src])
    if v22["indexer"] and v22["indexer_unknown"]:
        out["paired_v22"]["indexer_vs_indexer_unknown"] = paired(v22["indexer"], v22["indexer_unknown"])
    for src in ("learned", "indexer"):
        if v1[src] and v22[src]:
            out["paired_v1_vs_v22"][src] = paired(v1[src], v22[src])

    R15.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(R15 / "v22_summary.json", "w"), indent=1)

    def fmt(d):
        return "–" if not d else f"{d['rate']:.1f} % [{d['ci'][0]:.1f}, {d['ci'][1]:.1f}] ({d['x']}/{d['n']})"

    print("| lattice source | v1 match | v22 match | v22 all-correct | v22 seeds |")
    print("|---|---|---|---|---|")
    for src in V22_SOURCES:
        a = out["v1"].get(src, {}).get("match")
        b = out["v22"].get(src, {})
        print(f"| {src} | {fmt(a)} | {fmt(b.get('match'))} | {fmt(b.get('all_correct'))} | "
              f"{b.get('match', {}).get('per_seed', '') if b else ''} |")
        if b.get("match_covered_only"):
            print(f"| {src} (covered rows only) | | {fmt(b['match_covered_only'])} | | |")
    print("\nPaired McNemar (v22, match):")
    for k, v in out["paired_v22"].items():
        if v:
            print(f"  {k:<34} pairs {v['n_pairs']:>5}  gain {v['b_gain']:>3}  loss {v['c_loss']:>3}  p = {v['p']:.2e}")
    print("\nPaired v1 vs v22 (same material_id x seed, match):")
    for k, v in out["paired_v1_vs_v22"].items():
        if v:
            print(f"  {k:<10} pairs {v['n_pairs']:>5}  v22 gains {v['b_gain']:>3}  v22 loses {v['c_loss']:>3}  p = {v['p']:.2e}")
    print(f"\nWrote {R15 / 'v22_summary.json'}")


if __name__ == "__main__":
    main()
