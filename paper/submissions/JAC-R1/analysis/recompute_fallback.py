"""R3-3c: recompute the indexer arm EXCLUDING the 39 structures that 03_sample.py
silently fell back to the true lattice for (uncovered by the indexer)."""
import json, math, glob
from scipy.stats import binomtest
ROOT = 'paper/phase9_results'
idx = json.load(open(f'{ROOT}/index_benchmark_v2_native.json'))
covered = {r['mid'] for r in idx['rows'] if r.get('pred_params')}
def load(f): return [json.loads(l) for l in open(f)]
res = {}
for arm in ['indexer', 'learned']:
    rows = []
    for s in range(3):
        rows += load(f'{ROOT}/phase12_multiseed/{arm}_s{s}.per_sample.jsonl')
    res[arm] = rows
all_mids = {r['material_id'] for r in res['indexer']}
uncovered = all_mids - covered
print('test mids', len(all_mids), 'covered', len(all_mids & covered), 'uncovered', len(uncovered))
def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return round(100 * (c - h), 2), round(100 * (c + h), 2)
ind = res['indexer']
ind_cov = [r for r in ind if r['material_id'] in covered]
ind_unc = [r for r in ind if r['material_id'] not in covered]
for name, rows in [('indexer ALL (as published)', ind), ('indexer COVERED only', ind_cov), ('indexer FALLBACK=true lattice', ind_unc)]:
    k = sum(r['match'] for r in rows); ka = sum(r['all_correct'] for r in rows); n = len(rows)
    print(f'{name:32s} n={n:5d} match={k:3d} ({100*k/n:.2f}% CI {wilson(k,n)})  all_correct={ka}')
for s in range(3):
    a = [r for r in ind_cov if r['seed'] == s]; b = [r for r in ind_unc if r['seed'] == s]
    print(f'  seed {s}: covered match {sum(r["match"] for r in a)}/{len(a)}   fallback match {sum(r["match"] for r in b)}/{len(b)}')
learned = {(r['material_id'], r['seed']): r for r in res['learned']}
print('learned scored rows', len(learned), 'learned matches', sum(r['match'] for r in res['learned']))
for label, rows in [('published pairing (all)', ind), ('covered-only pairing', ind_cov)]:
    b = c = n = b_ac = c_ac = 0
    for r in rows:
        k = (r['material_id'], r['seed'])
        if k not in learned: continue
        n += 1; l = learned[k]
        b += r['match'] and not l['match']; c += l['match'] and not r['match']
        b_ac += r['all_correct'] and not l['all_correct']; c_ac += l['all_correct'] and not r['all_correct']
    p = binomtest(b, b + c, 0.5).pvalue if b + c else float('nan')
    p_ac = binomtest(b_ac, b_ac + c_ac, 0.5).pvalue if b_ac + c_ac else float('nan')
    print(f'{label:26s} paired n={n} match: b={b} c={c} p={p:.2e} | all_correct: b={b_ac} c={c_ac} p={p_ac:.3f}')
# uncovered structures: which crystal systems?
print('uncovered mids sample:', sorted(uncovered)[:10])
