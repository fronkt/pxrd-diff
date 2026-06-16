# Phase 13 — prediction-target ablation, oracle firm-up, deCIFer baseline, indexer robustness

Batch of GPU experiments run **2026-06-16** on a vast.ai RTX 5090 (32-core) box,
feeding the Phase-12 journal submission (Digital Discovery, RSC). Driven by three
chained scripts (see `provenance/`): `matrix_driver.sh` → `finalize.sh` → `finalize2.sh`.

All structure-domain scoring goes through the project harness
(`pxrd_diff.eval` StructureMatcher, ltol 0.2 / stol 0.3 / angle_tol 5).

## 1. Prediction-target ablation (`ablation_3seed/`)

3 training seeds × 3 architectures, each trained from scratch (100k steps, lr 5e-4,
bs 64), evaluated true-lattice at n=1000 (n_samples=1, no refine). Tests whether the
ε-vs-x₀ prediction target or the Debye guidance weight matter.

| variant | match_rate | all_correct | rwp | pearson |
|---|---|---|---|---|
| v11  (ε,  λ=1) | 0.0110 ± 0.0000 | 0.0020 | 7.107 ± 0.008 | 0.359 ± 0.000 |
| v13_l1 (x₀, λ=1) | 0.0127 ± 0.0017 | 0.0020 | 7.211 ± 0.234 | 0.392 ± 0.023 |
| v13_l0 (x₀, λ=0) | 0.0123 ± 0.0012 | 0.0020 | 7.276 ± 0.180 | 0.389 ± 0.024 |

**NEGATIVE result:** the three variants are statistically indistinguishable (overlapping
error bars on every metric, identical 0.002 all_correct). The prediction target and the
Debye-guidance weight do not affect structure recovery.

## 2. Oracle 3-seed firm-up (`oracle_3seed/`)

v21 checkpoint (`runs/gpu_v21_p9/ckpt_final.pt`), true-lattice, n=1000, n_samples=20,
refine_steps=200, across 3 sampling seeds — the paper's "5.6% oracle protocol".

| metric | mean ± std | seeds |
|---|---|---|
| match_rate | **0.0450 ± 0.0033** | 0.041 / 0.049 / 0.045 |
| all_correct | 0.0060 ± 0.0008 | 0.007 / 0.006 / 0.005 |
| rwp | 5.595 ± 0.057 | — |
| pearson | 0.644 ± 0.002 | — |

The 3-seed match_rate firms to **4.5%**, below the previously reported single-seed
**5.6%** (56/1000 was a high draw). Still well above the no-true-lattice indexer (~1.2%)
and learned head (~0%); the oracle gap stands, now tighter. Same multi-seed
regression-to-reality already seen for the learned head (single-seed 1.0% → 3-seed 0%).

## 3. deCIFer external baseline (`baselines/`)

Published deCIFer_v1 (Johansen et al. 2025, arXiv:2502.02189), conditioned on XRD +
composition, n=300 MP-20 test, re-scored through our harness (`decifer_n300.json`,
n=298 after 2 parse fails):

- match_rate **0.738**, all_correct **0.695**, rwp 6.63, pearson 0.551, sg_match@0.1 0.836

Establishes the external upper-reference: a full autoregressive CIF model recovers ~74%,
vs our best oracle protocol at 4.5% — the coordinate-decoding/encoder bottleneck, not the
lattice alone, is the deep limit.

## 4. Classical-indexer robustness (`index_robustness/`)

`scripts/09b_index_robustness.py` — classical Q-space auto-indexer vs systematic
displacement and zero-shift perturbations (n=300, 287 indexed). v20 learned-head len_MAE
is a fixed 1.37 (encoder is perturbation-independent).

| condition | strict % | consistent % | len_MAE (Å) |
|---|---|---|---|
| base | 50.7 | 60.0 | 1.237 |
| disp 0.10° | 35.3 | 40.0 | 1.624 |
| disp 0.20° | 25.0 | 27.7 | 1.839 |
| zshift 0.05° | 43.7 | 50.7 | 1.403 |
| zshift 0.10° | 35.7 | 40.3 | 1.596 |
| zshift 0.20° | 26.0 | 28.3 | 1.842 |

**Caveat (addresses M4):** the indexer beats the learned head (1.24 vs 1.37 len_MAE) only
on clean patterns. Under realistic instrumental displacement/zero-shift (≥0.10°) it
degrades past the learned head — its advantage is contingent on well-calibrated input.
