"""Merge one chunked eval arm into the file names eval_v30.sbatch would have written.

Usage: python merge_chunks.py OUT TAG N CHUNK

Reads OUT/chunks/TAG.cSSSS.{json,per_sample.jsonl,structs.jsonl} for SSSS = 0, CHUNK, ...
(written by chain_debug.sbatch through 03_sample.py --start/--n) and writes
OUT/TAG.per_sample.jsonl and OUT/TAG.structs.jsonl (chunks concatenated in test order)
plus OUT/TAG.json (the per-chunk summaries, and the headline rates recomputed from the
per-sample rows, which is what analysis/r2_stats.py reads). Refuses to merge an
incomplete arm or one whose per-sample rows are not exactly N distinct structures.
"""
import json
import sys
from pathlib import Path

out, tag, n, chunk = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
starts = list(range(0, n, chunk))
bases = [out / "chunks" / f"{tag}.c{s:04d}" for s in starts]
missing = [str(b) for b in bases
           if not all(Path(f"{b}{ext}").exists() for ext in (".json", ".per_sample.jsonl", ".done"))]
if missing:
    sys.exit(f"{tag}: {len(missing)} chunk(s) not finished, e.g. {missing[0]}")

rows, structs, summaries = [], [], []
for s, b in zip(starts, bases):
    summ = json.loads(Path(f"{b}.json").read_text())
    if summ.get("config", {}).get("start") != s:
        sys.exit(f"{b}.json: config.start = {summ.get('config', {}).get('start')}, expected {s}")
    summaries.append(summ)
    rows += [json.loads(l) for l in Path(f"{b}.per_sample.jsonl").read_text().splitlines() if l.strip()]
    sp = Path(f"{b}.structs.jsonl")
    if sp.exists():
        structs += [l for l in sp.read_text().splitlines() if l.strip()]

mids = [r["material_id"] for r in rows]
if len(rows) != n or len(set(mids)) != n:
    sys.exit(f"{tag}: {len(rows)} per-sample rows, {len(set(mids))} distinct; expected {n}")
seeds = {r["seed"] for r in rows}
if len(seeds) != 1:
    sys.exit(f"{tag}: mixed seeds {sorted(seeds)}")


def rate(key):
    return sum(bool(r[key]) for r in rows) / n


rmsds = [r["rmsd"] for r in rows if r["rmsd"] is not None]
merged = {
    "tag": tag, "n": n, "chunk": chunk, "seed": seeds.pop(),
    "merged_from": [f"{b.name}.json" for b in bases],
    "rates_from_per_sample": {
        "match": rate("match"), "composition_ok": rate("composition_ok"),
        "sg_match@0.1": rate("sg_match@0.1"), "all_correct": rate("all_correct"),
        "n_index_fallback_miss": sum(bool(r.get("index_fallback_miss")) for r in rows),
        "n_unbuildable": sum(bool(r.get("unbuildable")) for r in rows),
        "mean_rmsd_of_matches": sum(rmsds) / len(rmsds) if rmsds else None,
    },
    "chunks": summaries,
}
(out / f"{tag}.per_sample.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
(out / f"{tag}.structs.jsonl").write_text("".join(l + "\n" for l in structs))
(out / f"{tag}.json").write_text(json.dumps(merged, indent=1))
print(f"{tag}: merged {len(bases)} chunks, n={n}, match={merged['rates_from_per_sample']['match']:.4f}, "
      f"{len(structs)} structures")
