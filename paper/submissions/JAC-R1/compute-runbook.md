# JAC HAT5032 revision — GPU compute runbook (rented RTX 5090, Vast.ai)

Budget estimate: C1 ≈ 1.7 h train + ~2 h eval; C2 downstream ≈ 1 h; C3 ≈ 1 h; C4 optional
≈ 18 h. At ~$0.40–0.60/h a 5090 → **C1–C3 ≈ $3–6, C4 ≈ +$8–11.** Frank's call to rent.
Vast.ai notes (from memory): torch via the cu128 index for RTX 5090; cap workers on many-core
boxes; check Inet-down ≥ 200 Mbit/s before pulling MP-20; repo `fronkt/pxrd-diff` is public
so no SSH key is needed for the clone. Push local first, pull on the box.

## 0. Box setup (~10 min)
```bash
git clone https://github.com/fronkt/pxrd-diff.git && cd pxrd-diff
pip install -r requirements.txt   # then torch cu128 wheel if the image's torch lacks sm_120
python scripts/00_pull_mp20.py
python scripts/01_simulate_pxrd.py --splits test val train --workers 8
mkdir -p runs paper/phase15_results
```
Bring `E:\gpu_v21_p9\ckpt_final.pt` up as `runs/gpu_v21_p9/ckpt_final.pt` (scp; 44 MB) — it
is the v1 checkpoint and is needed for the v1-vs-v2 eval rows and for C2.

## C1. Retrain v21 with the corrected simulator and the fixed x0-mode Debye loss (R2.1, R3.3a/b)
Exact v21 configuration from the checkpoint's stored args: steps 100000, bs 64, **lr 1e-3**
(the manuscript said 5e-4 — corrected in §4), d_model 256, n_layers 3, n_heads 4, lat_weight
0.1, aux_weight 0.5, debye_weight 1.0, predict_x0 True, everything else default.
```bash
python scripts/02_train.py --run-name gpu_v22_jac --steps 100000 --bs 64 --lr 1e-3 \
    --predict-x0 --debye-weight 1.0 --debye-form-factor pymatgen
```
Then the three lattice sources at three seeds, identical flags to Phase 12 (the default
`--index-fallback miss` now scores the 39 unindexed patterns as misses):
```bash
CK=runs/gpu_v22_jac/ckpt_final.pt
bash scripts/run_phase12_multiseed.sh $CK 1000 "0 1 2"        # learned head + given-system indexer
for s in 0 1 2; do
  python scripts/03_sample.py --ckpt $CK --n 1000 --n-samples 20 --ensemble-eta 0.0 \
      --refine-steps 200 --refine-lr 1e-3 --ddim-steps 50 --true-lattice --seed $s \
      --out-json paper/phase15_results/oracle_s$s.json \
      --per-sample-json paper/phase15_results/oracle_s$s.per_sample.jsonl
done
```
Move the Phase-12 outputs to `paper/phase15_results/v22_*` (the script writes into
`paper/phase9_results/phase12_multiseed/` and skips rows whose per-sample file exists —
either pass a fresh OUT_DIR or move the v1 files aside first). Stats:
`python scripts/stats_ci.py` as in run_phase12_multiseed.sh, plus
`paper/submissions/JAC-R1/analysis/recompute_fallback.py` adapted to the new paths.

Also worth one extra run if time allows: the v1 checkpoint re-evaluated with
`--index-fallback miss` (should reproduce 46/2883 exactly; it is a determinism check).

## C2. Pooling ablation (R2.2) — heads on the frozen encoder, then end-to-end
Head training is CPU-feasible and is being run locally (`scripts/13_pooling_ablation.py`);
on the box run it against the C1 checkpoint too:
```bash
python scripts/13_pooling_ablation.py --ckpt $CK --epochs 3 --bs 64 --n-test 1000 \
    --out paper/phase15_results/pooling_ablation_v22
```
End-to-end: substitute each head's cells for the lattice channel (same mechanism as the
indexer drop-in; `pred_params` schema):
```bash
for arm in gpool attnpool peaks; do for s in 0 1 2; do
  python scripts/03_sample.py --ckpt $CK --n 1000 --n-samples 20 --ensemble-eta 0.0 \
      --refine-steps 200 --refine-lr 1e-3 --ddim-steps 50 --seed $s \
      --lat-from-index paper/phase15_results/pooling_ablation_v22/${arm}_cells.json \
      --out-json paper/phase15_results/pool_${arm}_s$s.json \
      --per-sample-json paper/phase15_results/pool_${arm}_s$s.per_sample.jsonl
done; done
```

## C3. Indexer WITHOUT the crystal system (R1.5iii) — end-to-end
Cells come from the local CPU run (`09_index_benchmark.py --system-mode unknown`, merged to
`paper/phase15_results/index_cells_test1000_unknown.json`):
```bash
for s in 0 1 2; do
  python scripts/03_sample.py --ckpt $CK --n 1000 --n-samples 20 --ensemble-eta 0.0 \
      --refine-steps 200 --refine-lr 1e-3 --ddim-steps 50 --seed $s \
      --lat-from-index paper/phase15_results/index_cells_test1000_unknown.json \
      --out-json paper/phase15_results/indexer_unknown_s$s.json \
      --per-sample-json paper/phase15_results/indexer_unknown_s$s.per_sample.jsonl
done
```

## C4 (optional). Phase 4 ablation at three seeds (R1.5ii)
Six configs (v10, v11, v13, v14, v15, v16) × seeds 1, 2 with the corrected simulator; the
flag sets are in tasks/todo.md Phase 4. ~1.5 h each → ~18 GPU-h. If skipped, Table 1 stays
labelled single-seed / exploratory (already done in the text).

## Declined
PXRDnet n = 200 (≈ 33 GPU-days at ~4 GPU-h/material). The ordering claim it would have
supported has been removed from the abstract instead; §5.3/§6 keep it as a hypothesis.

## After the box
Copy `paper/phase15_results/` back, commit it, destroy the instance, record spend in
tasks/todo.md Phase 15, then replace every `[PENDING C*]` tag in paper.md.

## Actuals (run 2026-09-21, vast.ai instance 51833313, RTX 5090 at $0.518/h)
- Box: 32 vCPU offer (cgroup quota 30.7), 456/370 Mbit/s, image pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime.
  Rented 00:55 UTC; setup 00:57–01:05 (clone, pip, MP-20 pull, 45 k patterns simulated at 24 workers in 2 min,
  form-factor tests green). The v1 checkpoint was NOT uploaded (E: drive unmounted locally), so the optional
  v1 determinism re-check was skipped; the v22 checkpoint was pulled back to `runs/gpu_v22_jac/` (md5-verified).
- Everything ran from one box-side `pipeline.sh` in tmux (idempotent; status file with rc per run; all 22 runs rc=0).
- C1: train 01:05–03:03 (6 896 s, final EMA 0.602 vs v21 0.726); nine evals 03:03–04:09 (~7.4 min each, no
  StructureMatcher hangs, 3 000/3 000 scored per arm). Result: learned 1.9 %, indexer 1.6 %, oracle 5.5 %;
  learned-vs-indexer McNemar p = 0.35 (submitted 27/0, p = 1.5e-8 does not survive the fix); oracle p = 3e-18.
- C2: heads at 10 epochs (not 3: arm (ii) was unconverged locally) 04:09–04:17; nine end-to-end runs 04:17–05:25
  (0.7 / 0.5 / 1.2 % for gpool / attnpool / peaks; every match cubic). Extra: attnpool at 30 epochs after the
  pipeline (a concurrent launch was stopped: the sampler holds 26.6 GB of the 32 GB card).
- C3: the unknown-system cells were recomputed ON THE BOX CPUs during training (20 shards × 50, 01:16–01:26)
  after a subagent audit showed the first max-M20 rule (24.3 %) was naive; the de Wolff/TREOR acceptance rule gives
  43.4 % strict vs 48.8 % given. Three end-to-end runs 05:25–05:48: 1.4 % vs 1.6 % (0 gained / 6 lost).
- C4: declined (budget: $6.35 credit at rent).
- Wall time ≈ 5.5 h through the pipeline; cost recorded in tasks/todo.md Phase 15 at teardown.
