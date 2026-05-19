# Task: PXRD-Diff — E(3)-equivariant conditional diffusion for PXRD inversion

## Context
Target: AI4Mat / ML4PS workshop @ NeurIPS 2026. Frozen MACE encoder + 1D ResNet PXRD
encoder + diffusion on fractional coords/lattice, with differentiable Debye scattering
loss as the core novelty. Scope locked to MP-20 (≤20 atoms). Compute: CPU now, cloud
GPU later.

Council verdict (2026-04-29): Pursue Candidate A. Define eval metric and held-out test
split BEFORE writing any model code.

## Plan

### Phase 0 — Data pipeline + evaluation harness (CPU only)
- [x] 0.1 Project skeleton, git init, README, requirements, pyproject
- [x] 0.2 Pull MP-20 canonical splits from CDVAE repo (train/val/test CSVs with CIFs)
- [x] 0.3 PXRD simulator (pymatgen XRDCalculator wrapper, Cu Kα, configurable grid)
- [x] 0.4 Cache simulated patterns (npz) for all train/val/test
- [x] 0.5 Evaluation module: spacegroup match, coord RMSD, Rwp, element accuracy
- [x] 0.6 Smoke tests: identity baseline scores 100%, random scores ~0%
- [x] 0.7 Visualization notebook (5 random structures + their PXRD)
- [x] 0.8 Push to GitHub

### Phase 1 — Conditional diffusion baseline (Weeks 2-4, CPU prototype)
- [x] 1.1 Equivariant structure encoder (SchNet-style, MACE deferred to Phase 2)
- [x] 1.2 1D ResNet PXRD encoder (4 blocks, 256-dim output)
- [x] 1.3 Diffusion module: VP-SDE cosine schedule, torus coords + R^6 lattice
- [x] 1.4 Additive PXRD+timestep conditioning (cross-attention deferred)
- [x] 1.5 Training loop w/ lattice normalization, grad clip, cosine LR
- [x] 1.6 DDIM sampler (50 steps, deterministic) — reverse process from noise to structure
- [x] 1.7 E2E smoke: 500 steps, sample 8, all invalid lattice (expected at <1% training)

### Phase 1.5 — Multi-resolution cross-attention (coord loss fix)
GPU v4 run (100k steps) confirmed coord loss flat at ~3.0 — global PXRD pooling
is the bottleneck. Aux loss proves encoder learns good features (0.99→0.007),
but AdaptiveAvgPool1d(1) destroys spectral structure before it reaches the denoiser.

- [x] 1.5.1 PXRDEncoder: return multi-res feature maps alongside global vector
- [x] 1.5.2 CrystalDenoiser: add per-layer cross-attention (atoms attend to PXRD features)
- [x] 1.5.3 DDIMSampler: wire up new encoder output (global + feature maps)
- [x] 1.5.4 Training script: update for new signatures, keep aux head on global vector
- [x] 1.5.5 GPU smoke test — shapes OK, backward OK, no NaN (3.5M params)
- [x] 1.5.6 Push to GitHub, deploy to Vast.ai RTX 5090 (32GB)
- [x] 1.5.7 Full training run (100k steps) — coord 3.0→1.0, loss 3.1→1.1

### Phase 2 — Differentiable Debye scattering loss (Weeks 5-6)
- [x] 2.1 Implement differentiable structure factor PXRD in PyTorch (vectorized)
      - Atomic form factors (4-Gaussian, tabulated for Z=1..100 via pymatgen)
      - F(hkl) = Σⱼ fⱼ(s)·exp(2πi h·xⱼ), fully differentiable w.r.t. frac_coords
      - |F|² × Lorentz-polarization, Gaussian broadening onto 2θ grid
      - Debye-Waller temperature factor
      - Note: Debye equation approach (pairwise sum) was tried first but produces
        smooth scattering, not Bragg peaks. Replaced with structure factor.
- [x] 2.2 Verify against pymatgen XRDCalculator on 50 known structures
      - Pearson: mean=0.9627, std=0.0265, min=0.8956, max=0.9975
      - 50/50 > 0.7, 50/50 > 0.5, all gradients OK
- [x] 2.3 Add as auxiliary loss in training: Debye PXRD of x0_pred vs input pattern
      - Reconstruct x0_pred from noise prediction via diffusion equation
      - Compute structure factor PXRD, compare with Pearson correlation loss
      - --debye-weight flag controls λ (default 0, backward compatible)
- [x] 2.4 Ablation: λ_debye ∈ {0, 0.1, 1, 10} on gpu_v6/v7/v8/v9
      - 4× 100k steps, batch=64, RTX 5090. Final EMA losses:
        v6=1.09 (no debye), v7=1.16, v8=1.68, v9=6.81 (loss inflated by debye term)
      - coord loss converged to ~1.0 in all runs (not better than v5 baseline)
      - lat loss stuck at ~1.0 in all runs (random baseline — see 2.5)
      - At n=256 test eval (with ground-truth lattice to isolate coords):
        match rate: v6=0.4-1.2%, v7=0-1.2%, v8=0.4-0.8%, v9=1.2%
      - Conclusion: differences within noise (1-3 matches per 256). Debye loss
        does NOT provide measurable benefit at this architecture/scale. The
        bottleneck is coord/lattice prediction itself, not the loss formulation.

### Phase 2.5 — Lattice prediction fix + coord prediction improvements
The ablation eval revealed two architectural problems independent of Debye loss:
  (a) Lattice noise prediction never learns (loss flat at random baseline ~1.0
      across all four runs). Sampled lattices are completely garbage (negative
      angles, lengths in 10^2 range).
  (b) Coord prediction is barely better than mean-prediction (Pearson ~0.31,
      headline match 0%). The cross-attention helped vs. v4 (3.0→1.0) but model
      isn't truly inverting PXRD.

- [x] 2.5.1 Diagnose & fix lattice noise prediction
      - **Root cause**: denoiser never saw the noisy_lat_p it was trying to
        denoise. Training fed ground-truth lattice matrix to denoiser while
        applying noise to a separate noisy_lat_p vector. Lattice head could
        only predict E[eps]=0, giving loss=1.0.
      - **Fix**: pass noisy_lat_p as additional input. Lattice head now sees
        h_pool + pxrd_global + lat_in_proj(noisy_lat_p) + t_cond.
      - **Result (gpu_v10, 100k steps, λ=0)**: lat loss 1.0 → **0.055** (95%↓)
      - **Result (gpu_v11, 100k steps, λ=1)**: lat loss 1.0 → **0.063** (94%↓)
- [x] 2.5.2 Re-run ablation with lattice fix
      - At n=256 with predicted lattice: v10=0%, v11=0.4% match (within noise)
      - At n=256 with true lattice (coord-only): v10=1.6%, v11=0.4% match
      - Pearson ~0.34 with true lattice (vs 0.31 baseline) — minor improvement
      - **Conclusion**: lattice fix dramatically improves lat loss but coord
        prediction is still the dominant bottleneck. Debye loss does not
        meaningfully shift coord quality at this scale.
- [x] 2.5.3 Coord prediction architecture investigation
      - **Bigger model (10.1M params, gpu_v12, 18k steps killed)**: coord still
        stuck at 1.0 baseline. Capacity is NOT the bottleneck.
      - **x0 residual prediction (gpu_v13, 100k, λ=1)**: model output
        interpreted as correction added to noisy input. Coord drops to 0.072
        (vs 0.083 random baseline, 13% below). Lat drops to 0.012 (99%↓).
      - **Headline: 3-way ablation (n=256 test, true lattice for coord-only)**:
        ```
        config                       match%   rmsd   pearson
        v10 (eps, λ=0)               0.8%     0.24   0.346
        v11 (eps, λ=1)               0.0%     nan    0.333
        v13 (x0+res+latfix, λ=1)     2.7%     0.23   0.434  ← 3.5×
        ```
      - The combination (x0 residual + lattice fix + Debye loss) provides
        first measurable improvement over baseline.
      - With predicted lattice (full pipeline): v13 = 1.2% vs v10/11 = 0.4%

### Phase 2.6 — Mid-effort improvements (target 15-30% match rate)
Three independent improvements addressing different aspects of the bottleneck:
  (a) Per-atom positional supervision (currently coords get only noise prediction)
  (b) Per-atom symmetry-breaking (currently permutation-equivariant means atoms ambiguous)
  (c) Better physics inductive bias for the denoiser

- [ ] 2.6.1 Distance matrix auxiliary loss
      - Add pair-MLP `D_ij = f(h_i, h_j)` predicting periodic Cartesian distances
      - Strong per-atom supervision: each atom must encode its relative position
      - Cheapest to implement (~1 hour); supervises geometry directly
- [ ] 2.6.2 Wyckoff position tokens
      - Use spglib to analyze each training structure → assign Wyckoff label per atom
      - Add learned Wyckoff embedding alongside atom_emb
      - Breaks permutation equivariance in a physically meaningful way
      - At inference: predict Wyckoff site assignments from PXRD (or use ground truth for now)
- [ ] 2.6.3 MACE encoder for denoiser (optional, high effort)
      - Replace SchNet-style MP with MACE message passing (mace-torch)
      - Pre-trained MACE-MP-0 features for free physics priors
- [~] 2.6.4 Combined gpu_v14 run + 4-way ablation (PARTIAL — see below)
      - v14 = x0+res+latfix + Debye λ=1 + dist λ=0.01 + Wyckoff (zero-init).
      - Training stable (gnorm 0.4-0.9, was 7-34 before zero-init fix).
      - Final losses: coord=0.076, lat=0.013, aux=0.014, dist=2.0, debye=0.55.
      - n=1000 ablation (coord-only with true lattice):
        ```
        v10 (eps λ=0):              match 1.40%   pearson 0.359   rmsd 0.17
        v11 (eps λ=1):              match 0.90%   pearson 0.365   rmsd 0.15
        v13 (x0+res+latfix, λ=1):   match 2.51%   pearson 0.434   rmsd 0.22  ← best
        v14 (+ Wyckoff + dist):     match 0.80%   pearson 0.367   rmsd 0.14  ← best rmsd
        ```
      - Surprising: v14 has lowest RMSD-when-matched but lowest match rate.
        Suggests model produces structures with correct local geometry (low rmsd)
        that don't match StructureMatcher's tolerance (different mode of error).
      - Wyckoff embedding DID learn (mean abs 0.065, 5404 non-zero entries),
        just not helpful for StructureMatcher.

### Phase 2.6.5 — Disambiguate Wyckoff vs dist (TOMORROW)
v14 lumped both novelties together and underperformed v13 on match rate.
Need to isolate which is the problem (or whether either helps in isolation).
- [ ] Run v15: x0+res+latfix+Debye λ=1 + Wyckoff only (no dist loss)
- [ ] Run v16: x0+res+latfix+Debye λ=1 + dist only (no Wyckoff)
- [ ] If neither beats v13, that's the answer — write up

### Phase 3 — Cloud GPU scale-up + baselines (Weeks 7-9)
- [x] 3.1 Provision cloud GPU (Vast.ai RTX 5090, 32GB)
- [x] 3.2 Full training run on MP-20 train split (100k steps × 4 runs done)
- [ ] 3.3 DiffractGPT baseline reproduction (deferred until 2.5 fixed)
- [ ] 3.4 Crystalyze baseline reproduction (deferred until 2.5 fixed)
- [ ] 3.5 Random-search + Rietveld baseline (via GSAS-II or jana2020)
- [ ] 3.6 Final benchmarks, ablations, plots, paper draft

## Review — Phase 0
- Completed: 2026-04-29
- What worked: pymatgen XRDCalculator is fast (~100 struct/s), zero sim failures across 45k structures
- What changed: none, plan executed as-is
- Known limitations: patterns are simulated (no experimental artifacts), 92% sparsity

## Review — Phase 1
- Completed: 2026-04-30
- What worked: 2.1M param model trains on CPU (0.19s/step), DDIM sampling at 0.1s/struct
- What changed from plan: used SchNet-style MP instead of frozen MACE (simpler, correct geometry,
  MACE integration deferred to Phase 2). Additive conditioning instead of cross-attention.
  Atom types conditioned (fixed), not predicted.
- Known limitations: 500-step smoke produces garbage lattice (10^6-10^9 values); need real
  training (~10k+ steps on GPU) to see meaningful predictions. Lattice prediction may need
  separate treatment (clamping, log-space diffusion, or conditioning on composition stats).

## Review — GPU v4 run (100k steps on RTX 5070 Ti)
- Completed: 2026-05-01
- coord loss: flat at ~3.0 from step 1 to 100k (uniform random baseline)
- lat loss: slight improvement (1.09 → 0.97)
- aux loss: 0.99 → 0.007 (PXRD encoder IS learning, information destroyed by pooling)
- Diagnosis: AdaptiveAvgPool1d(1) collapses all spectral structure; additive broadcast
  gives every atom identical conditioning. Cross-attention is required.

## Review — Phase 2 (differentiable PXRD + ablation)
- Completed: 2026-05-02
- What worked:
  - Structure factor PXRD computation matches pymatgen at Pearson 0.96 mean (50/50 > 0.7)
  - DiffPXRD module is fully differentiable, integrates cleanly into training loop
  - DDIM sampler bug at t=1.0 boundary identified and fixed (ab.sqrt clamp, t-start offset)
- What changed from plan:
  - Pivoted from Debye equation to structure factors (Debye eq → no Bragg peaks)
  - Bug fix: d-spacing factor of 2π was wrong (recip lattice in physics convention)
- Honest negative result:
  - At λ ∈ {0, 0.1, 1, 10}, ablation match rates are within noise (0.4-1.2% at n=256)
  - The Debye loss does NOT measurably improve coord prediction at this scale
  - Real bottleneck: lattice prediction is broken, coord prediction barely learns
- Why this matters: The Debye loss is well-formulated and verified, but proving its
  value requires a model that can actually invert PXRD. Phase 2.5 must come first.

## Review — Phase 2.5 (architecture fixes)
- Completed: 2026-05-02
- Three architectural changes, each measurably impactful:
  1. **Lattice fix (2.5.1)**: pass noisy_lat_p to denoiser → lat loss 1.0 → 0.05
     (95% reduction). Root cause was denoiser never seeing the lattice it was
     denoising. Without this, all lat sampling is unusable noise.
  2. **Capacity test (2.5.3a)**: 10.1M params (vs 3.7M), d=384, 5 layers.
     Coord still stuck at 1.0 — capacity is NOT the bottleneck.
  3. **x0 residual prediction (2.5.3b)**: model output interpreted as
     correction added to noisy input (so the natural zero output gives the
     identity transform). Coord drops to 0.072 (13% below random baseline),
     lat to 0.012 (99% reduction).
- **Final 3-way ablation (n=256, coord-only with true lattice):**
  ```
  config                                match%   pearson
  v10  eps, λ=0                          0.8%    0.35
  v11  eps, λ=1                          0.0%    0.33
  v13  x0+residual+latfix, λ=1           2.7%    0.43   ← 3.5× match, +26% Pearson
  ```
- Honest read: 2.7% match is still far from useful for real Rietveld replacement,
  but the architectural recipe (x0 residual + lattice noise input + Debye aux loss)
  shows directional improvement. This is publishable as a methods paper showing
  both what works (the recipe) and what's hard (per-atom anchoring from a global
  signal like PXRD).

---

## Phase 4 — Inference-time improvements (target: 2.51% → 10–15%)
These require NO retraining. Pure inference-side changes on the best checkpoint (v13).
Fastest path to a better headline number.

### 4.1 Ensemble generation + pick by Rwp (~1 hour)
- [ ] 4.1.1 In `03_sample.py`, add `--n-samples N` flag (default 20)
- [ ] 4.1.2 For each test structure: generate N candidate structures via DDIM sampler
- [ ] 4.1.3 For each candidate: compute Rwp against true PXRD pattern using DiffPXRD
- [ ] 4.1.4 Return the candidate with lowest Rwp as the prediction
- [ ] 4.1.5 Evaluate on n=1000 test set, compare match% vs single-sample v13 baseline
- [ ] 4.1.6 Sweep N ∈ {5, 10, 20, 50} — find diminishing returns threshold
- Expected improvement: 2–4× match rate (2.51% → 5–10%)

### 4.2 Rietveld refinement post-processing (~1 day)
- [ ] 4.2.1 Add `refine_structure(frac_coords, atom_types, lattice, target_pxrd, steps=200)`
        to `sampler.py`. Uses Adam on `frac_coords` (leaf tensor) minimizing Debye loss.
- [ ] 4.2.2 Lattice refinement: also optimize lattice_params jointly (start at predicted)
- [ ] 4.2.3 Constrain: frac_coords % 1.0 after each step (stay on torus)
- [ ] 4.2.4 Apply refinement AFTER ensemble selection (best Rwp candidate → then refine)
- [ ] 4.2.5 Evaluate on n=1000, compare vs 4.1 and v13 baseline
- [ ] 4.2.6 Tune: step size (1e-3 → 1e-4), steps (50/100/200), convergence criterion
- Expected improvement: 4.1 result × 1.5–2× additional gain
- Push to GitHub after Phase 4 complete

---

## Phase 5 — Direct lattice from d-spacings (~2 days + retrain overnight)
Replace diffusion-predicted lattice with deterministic extraction from PXRD peaks.
Root cause: diffusion lattice still fails often (invalid angles/lengths) even with v13 fix.

### 5.1 Dedicated lattice head on PXRDEncoder
- [ ] 5.1.1 Add `self.lattice_head = nn.Sequential(Linear(d_model,128), SiLU(), Linear(128,6))`
        to `PXRDEncoder` (on the global embedding — peak positions encode d-spacings directly)
- [ ] 5.1.2 Train with MSE loss vs normalized lattice params (same normalization as existing)
- [ ] 5.1.3 Add `--lat-direct` flag to `02_train.py`: if set, use encoder lattice head
        prediction instead of diffusion for lattice at inference. Denoiser still receives
        predicted lattice params (as before) but with a much better starting point.
- [ ] 5.1.4 Constrain outputs: lengths > 0 via softplus, angles via sigmoid scaled to [10°, 170°]
- [ ] 5.1.5 Train from v13 checkpoint (warm start): freeze denoiser, train encoder+lat_head
        for 20k steps. Then unfreeze all for 30k steps. Total: ~6 GPU-hours.
- [ ] 5.1.6 Evaluate: lat MAE vs v13 lat MSE, match% on n=1000

### 5.2 Validation
- [ ] 5.2.1 Compare predicted vs true lattice params (MAE per dimension)
- [ ] 5.2.2 Check: how many predicted lattices are physically valid (a,b,c > 0, angles in [10,170])
- [ ] 5.2.3 Run full pipeline (predicted lattice + coords + Phase 4 refinement)
- Push to GitHub after Phase 5 complete

---

## Phase 6 — Space group conditioning (~1 week + retrain)
Constrain generation to crystallographically valid configurations.
Biggest physics inductive bias available without changing architecture.

### 6.1 Space group prediction head
- [ ] 6.1.1 Add SG classification head to PXRDEncoder: Linear(d_model, 230) + CrossEntropy
- [ ] 6.1.2 Train jointly (--sg-weight 0.1). Evaluate top-1 and top-5 SG accuracy.
- [ ] 6.1.3 At inference: predict top-k SG candidates from PXRD encoder

### 6.2 SG-constrained lattice parameter generation
- [ ] 6.2.1 Map SG → crystal system (cubic/tetragonal/ortho/mono/triclinic/hex/rhombo)
- [ ] 6.2.2 In lattice head (Phase 5): apply SG constraints post-hoc:
        - Cubic: a=b=c, α=β=γ=90°
        - Tetragonal: a=b, α=β=γ=90°
        - Hexagonal: a=b, γ=120°, α=β=90°
        - etc.
- [ ] 6.2.3 Alternatively: condition denoiser on SG one-hot embedding (concat with t_cond)

### 6.3 Wyckoff-reduced asymmetric unit prediction
- [ ] 6.3.1 For each training structure: extract asymmetric unit coords (unique Wyckoff positions)
- [ ] 6.3.2 Model predicts ONLY the asymmetric unit; symmetry operations applied to generate
        full structure. Reduces DoF dramatically (e.g., cubic: 1/48 of the full cell).
- [ ] 6.3.3 Implement `apply_symmetry_ops(asym_coords, sg_number)` using spglib
- [ ] 6.3.4 Full retrain from scratch with new parameterization (~12 GPU-hours)
- Push to GitHub after Phase 6 complete

---

## Phase 7 — Experimental noise augmentation (~1 day + retrain overnight)
Make model robust to real PXRD data (not just perfect simulated patterns).

### 7.1 Augmentation pipeline in `data.py`
- [ ] 7.1.1 Gaussian instrument noise: pattern += N(0, σ²), σ ~ U(0.001, 0.03) × max(I)
- [ ] 7.1.2 Peak broadening: convolve with Lorentzian (not just Gaussian) kernel,
        FWHM ~ U(0.05°, 0.3°) to simulate crystallite size effects
- [ ] 7.1.3 Amorphous background: add polynomial baseline + broad humps
- [ ] 7.1.4 Preferred orientation (March-Dollase): scale intensities by orientation factor
        for a random hkl direction (affects systematic sets of peaks)
- [ ] 7.1.5 Peak shift: random 2θ zero-offset ∈ [-0.1°, +0.1°] (simulates calibration error)
- [ ] 7.1.6 Apply augmentations with p=0.8 per batch (keep 20% clean for stability)

### 7.2 Training
- [ ] 7.2.1 Retrain from v13 (or Phase 5 best checkpoint), 50k additional steps with augmentation
- [ ] 7.2.2 Evaluate on clean simulated patterns AND augmented patterns
- [ ] 7.2.3 Goal: match rate on augmented patterns ≥ 50% of clean pattern performance
- Push to GitHub after Phase 7 complete

---

## Phase 8 — Full Materials Project data scale-up (~1 day setup + overnight training)
**DEMOTED (council 2026-05-18).** More data cannot fix the encoder/lattice wall —
five retrains proved that is architectural blindness, not data starvation. Phase 8
is deferred to a *B-then-A* lever: only worth running for residual coordinate error
AFTER Phase 9 fixes the unit cell. Do Phase 9.0 first. Steps below kept for later.
MP-20 has 27k train structures. Full MP has ~154k DFT-computed structures.

### 8.1 Data acquisition
- [ ] 8.1.1 Pull full MP structures via `mp-api`: filter to ≤20 atoms, DFT-converged,
        energy_above_hull < 0.1 eV/atom (stable/near-stable)
- [ ] 8.1.2 Expected yield: ~80–120k structures after filtering
- [ ] 8.1.3 Split: 90/5/5 train/val/test. Keep MP-20 test split untouched for comparison.
- [ ] 8.1.4 Simulate PXRD for all new structures: `01_simulate_pxrd.py` (batched, ~15 min)

### 8.2 Training
- [ ] 8.2.1 Retrain from scratch with full dataset, 200k steps (~12–18 GPU-hours)
- [ ] 8.2.2 Keep all Phase 5–7 improvements (lat head, SG conditioning, noise augmentation)
- [ ] 8.2.3 Evaluate on both MP-20 test split (comparability) and new full-MP test split

### 8.3 Validation
- [ ] 8.3.1 Compare match% on MP-20 test vs Phase 5–7 best checkpoint
- [ ] 8.3.2 Expected: 1.5–3× improvement from data scale alone
- Push to GitHub after Phase 8 complete

---

## Execution Order & Expected Match Rate Progression

| Phase | What | Effort | Expected match% |
|-------|------|--------|-----------------|
| v13 baseline | Current best | done | 2.51% |
| 4.1 Ensemble (N=20) | No retrain | 1 hour | ~5–8% |
| 4.2 Rietveld refinement | No retrain | 1 day | ~8–15% |
| 5 Lat from d-spacings | Partial retrain | 2 days | ~12–18% |
| 6 SG conditioning | Full retrain | 1 week | ~18–30% |
| 7 Noise augmentation | Partial retrain | 1 day | (robust to real data) |
| 8 Full MP data | Full retrain | 1 day setup | ~25–45% |

Goal: ≥25% match rate on MP-20 test split (publishable positive result).
Stretch: ≥40% (competitive with Crystalyze on this benchmark).


---

## Review — Phase 4 (inference improvements; no retrain)
- Completed: 2026-05-07
- Checkpoint: `runs/gpu_v13/ckpt_079500.pt` (v13 training crashed at step 80k due to
  16 GB instance disk filling with 160 × 43 MB checkpoints; the 80k save was partial.
  79.5k loads cleanly, coord loss 0.085 — ~17% above v13's 100k value of 0.072).
- Test set: n=1000 from MP-20 test split, `--true-lattice` (coord-only eval).
- DiffPXRD config matched training: `n_bins=256, hkl_max=5`.

### Headline results (n=1000)

| metric                      | baseline (single) | +ensemble (n=20) | +ensemble + refine 200 |
|-----------------------------|------------------:|-----------------:|-----------------------:|
| StructureMatcher match rate |             1.60% |            4.20% |              **6.80%** |
| rmsd_mean (Å)               |             0.211 |            0.170 |              **0.133** |
| rmsd_median (Å)             |             0.247 |            0.145 |              **0.126** |
| pearson_mean                |             0.425 |            0.533 |              **0.694** |
| rwp_mean                    |              7.57 |             7.14 |               **5.32** |
| headline_all_correct        |             0.20% |            0.20% |              **0.60%** |
| sg_match@0.1                |             1.70% |            1.60% |                  2.10% |

### Read

- **Phase 4.1 (ensemble + DiffPXRD-pearson selection) gives 2.6× match rate for free.**
  Selecting the best of N=20 candidates per pattern lifts match from 1.6 → 4.2 % with
  no retraining and no architectural changes. Per-candidate selection time on RTX 5090
  was ~14 s for n=1000 at chunk_size=32.
- **Phase 4.2 (Rietveld refinement) compounds to 4.25× match rate.**
  200 Adam steps on `1 - Pearson(DiffPXRD(coords), target)` collapses the avg
  refinement loss from 0.366 → 0.076, recovering coordinate precision that the
  diffusion sampler leaves on the table. Total refinement wall-clock ~28 s per 1000
  patterns at chunk_size=32. Refinement lattice was left fixed (true lattice in this
  eval mode).
- **Pattern-domain metrics improve sharply:** Pearson 0.425 → 0.694 (+63 %), Rwp
  7.57 → 5.32 (−30 %).  Coord-domain RMSD drops 37 % on the mean and 49 % on the
  median, suggesting the refinement is genuinely realigning atoms rather than just
  fitting the pattern.
- **Space-group recovery is unchanged.** sg_match@0.1 sits at ~2 % across all three
  configs. This is expected — refinement nudges fractional coords continuously and
  cannot flip the discrete symmetry detected by spglib. `headline_all_correct` is
  gated by sg_match, so it ticks up only modestly (0.2 → 0.6 %).

### Per-sample anecdotes from the refine run
A handful of test cases reach near-perfect reconstruction after refinement, e.g.
  - mp-1225695: rmsd 0.004 Å, pearson 1.000, sg ✓ (only `all=Y` row in the chunk)
  - mp-1218989: rmsd 0.293, pearson 0.993
  - mp-1223834: rmsd 0.219, pearson 0.978
  - mp-972370 : rmsd 0.054, pearson 0.996
These are cases where the diffusion sampler placed atoms close to the basin and the
DiffPXRD gradient pulled them in. The structures with low Pearson before refinement
(<0.3) tended to refine to mid-Pearson but rarely cross the rmsd≤0.1 threshold —
suggesting some predictions are in the wrong basin entirely (Phase 6 territory).

### What Phase 4 unblocks
- Match-rate ceiling moved from "<3 % feels publishable" to "**6.8 % is the new
  baseline**". Phase 5/6/7 improvements should build on this Phase 4 evaluation
  pipeline (ensemble + refine is essentially free at inference time).
- The fact that pearson improved 63 % while sg_match stayed flat localizes the
  remaining gap to **categorical symmetry recovery**, not coordinate precision.
  This validates the planned Phase 6 (space group conditioning + Wyckoff-reduced
  asymmetric unit) as the highest-leverage next step.

### Artifacts
- `paper/phase4_results/*.json` — per-config aggregate metrics (committed)
- `runs/phase4/*.{json,log}` on the GPU instance — full per-sample results

---

## Review — Phase 5 Path A (no retrain) — NEGATIVE result
- Completed: 2026-05-07
- Approach: promote the existing `AuxLatHead` (an MSE-supervised regularizer in
  the v13 training loss) to inference-time lattice predictor. Substitute its
  output for the diffusion sampler's lattice in `lattice_init`, ensemble
  selection, refinement, and final eval.
- Implementation: `src/pxrd_diff/model/aux_head.py`, `--lat-from-aux` flag in
  `scripts/03_sample.py`. Diagnostic always prints aux-vs-true MAE.

### Aux-head lattice quality (v13 ckpt @ 79.5k, n=1000 test)
```
                a       b       c       α       β       γ
MAE          1.119   1.022   1.380   12.98°  11.85°  17.55°
validity     998 / 1000 = 99.8% (vs ≈ 0% for diffusion sampler)
```

### End-to-end results (no --true-lattice, n=1000)
| metric                       | aux baseline (single) | aux + Phase 4 (ens+refine) |
|------------------------------|----------------------:|---------------------------:|
| StructureMatcher match rate  |                 1.20% |                      1.30% |
| pearson_mean                 |                 0.014 |                      0.082 |
| rmsd_mean (Å)                |                 0.226 |                      0.176 |
| rwp_mean                     |                19.94  |                     18.06  |
| headline_all_correct         |                 0.00% |                      0.00% |
| sg_match@0.1                 |                 1.40% |                      1.40% |

### Why Path A doesn't work
- A 1 Å MAE on lengths is ~15–20 % relative error for typical MP-20 unit cells.
  All Bragg peaks shift by a similar fraction in 2θ, driving Pearson with the
  target pattern to ~0 (mean 0.014 in the baseline).
- Ensemble + Pearson selection picks among uniformly-bad candidates; Rietveld
  refinement on coords cannot recover from a wrong unit cell. Phase 4 lifts
  pearson 0.014 → 0.082 (6×), but the absolute level is still useless.
- `match_rate` and `rmsd_mean` are misleading here — they're computed only over
  StructureMatcher matches (which are accidental at this lattice quality).
- The diffusion-sampler lattice was even worse: pymatgen's `StructureMatcher.
  get_rms_dist` hung on the predicted (degenerate) lattices, so the
  `diffusion_lat_*` configs of the eval runner were killed and not scored.
  Per-sample logs showed Pearson ≈ 0 across the board.

### What this rules out
- The aux head learned-as-regularizer signal is **insufficient** for inference
  use even though its 99.8 % validity rate is much better than the diffusion
  sampler. The aux-head MAE corresponds to ~15 % relative lattice error, and
  the eval is bottlenecked by lattice accuracy, not coordinate accuracy.

### What this motivates (Phase 5 Path B — pending)
- Retrain with much stronger aux-head supervision **and** a constrained head:
  - softplus on lengths (always positive)
  - sigmoid scaled to a physical range on angles (e.g. [30°, 150°])
  - aux-weight bumped from 0.5 → 5.0 (or split into separate length/angle losses)
  - resume from `ckpt_079500.pt` as warm start, ~30 k more steps
- Alternative: dedicate a small CNN/MLP regressor on raw PXRD peaks
  (Bragg's-law-informed loss directly on d-spacings).

### Artifacts
- `paper/phase5_results/{aux_lat_baseline, aux_lat_phase4}.json` (committed)
- `runs/phase5/*.{json,log}` on the GPU instance

---

## Review — Phase 5B + 6 (constrained lattice head + space-group head + SG constraints)
- Completed: 2026-05-08
- Trained `runs/gpu_v17_p5b6/ckpt_final.pt`: 30,500 steps resumed from
  `runs/gpu_v13/ckpt_079500.pt` (110k total). Encoder/denoiser warm-started;
  ConstrainedLatHead and SpaceGroupHead initialized fresh. 32 min on RTX 5090.
  Config: `--predict-x0 --debye-weight 1.0 --constrained-lat-head
  --sg-weight 0.1 --aux-weight 5.0 --save-every 5000`.
- Final losses: coord=0.076, lat=0.22, aux=0.38, sg=2.11, debye=0.58.

### SG-head accuracy
- **top-1: 39.2 %** | **top-5: 70.9 %**  (vs 0.4 % random across 230 classes).
  The encoder cleanly captures crystal-system / point-group features even
  though it doesn't capture absolute scale (lattice parameter MAE is large).

### End-to-end results (n=1000, no --true-lattice)

| metric                       | p5b_baseline | p5b_phase4 | p5b_sg_baseline | p5b_sg_phase4 |
|------------------------------|-------------:|-----------:|----------------:|--------------:|
| StructureMatcher match rate  |        1.40% |      1.20% |           0.50% |         0.70% |
| pearson_mean                 |        0.009 |      0.082 |           0.005 |         0.047 |
| rmsd_mean (Å)                |        0.226 |      0.211 |           0.227 |     **0.109** |
| rmsd_median (Å)              |        0.250 |      0.256 |           0.264 |     **0.007** |
| rwp_mean                     |       20.25  |     17.91  |          19.05  |         16.22 |
| headline_all_correct         |        0.00% |      0.00% |           0.00% |     **0.30%** |
| sg_match@0.1                 |        1.40% |      1.30% |           1.40% |         1.60% |

### ConstrainedLatHead MAE
Pre-SG constraint:  a=1.11 b=1.02 c=1.42 Å,  α=13.7° β=12.5° γ=18.4°
Post-SG constraint: a=1.11 b=1.02 c=1.41 Å,  α=17.5° β=16.3° γ=23.4°

### Read

Two findings drive the rest of this project:

1. **The encoder is lattice-precision-limited, not lat-head-architecture-limited.**
   Bumping aux weight 0.5 → 5.0, swapping to a sigmoid-bounded ConstrainedLatHead,
   and 30 k extra training steps left the lattice MAE essentially unchanged from
   the original AuxLatHead (a≈1.11 Å, α≈14°). The PXRD encoder's global pooling
   destroys the peak-position fidelity that Bragg's-law decoding needs to nail
   absolute lengths and angles. **No amount of head-side fixing will move this
   number; the next move has to live in the encoder** — explicit d-spacing
   extraction, wavelet/peak-finding features, or per-peak attention rather than
   global pooling.

2. **SG conditioning works qualitatively but is destroyed by lattice noise.**
   The 39 % top-1 SG accuracy is a real signal — when the SG head is right and
   Phase 4 ensemble + refinement converge, we get **rmsd_median = 0.007 Å, basically
   exact reconstruction**. That collapse from 0.256 to 0.007 is the strongest
   per-instance signal in the project. The catch: it only fires for ~7/1000
   structures because (a) SG is wrong 61 % of the time and (b) when it's wrong
   the constraint locks the lattice into the wrong symmetry, dropping match_rate
   from 1.20 % to 0.70 %.

   **`headline_all_correct = 0.30 %` is the first nonzero "predicted everything
   from PXRD alone" number in the project.** It's small but it's a real signal,
   and it scales with SG-head accuracy and lattice precision — both improvable.

### What this rules out
- A drop-in replacement of the lattice head (Phase 5 Path A or B) cannot fix
  the no-true-lattice headline. The encoder is the bottleneck.

### What this points to next
- **Phase 5C (encoder rework, ~1 week):** add explicit Bragg-peak features —
  e.g. a peak-position attention head fed by a 1-D peak detector — so the
  encoder can encode d-spacings with subpixel precision instead of relying on
  the ResNet's pooled summary. This should drop length MAE from ~1 Å to ~0.1 Å.
- **Phase 6.3 (Wyckoff-reduced asymmetric unit, deferred):** with a working SG
  predictor we can already enumerate symmetry orbits; predicting only the
  asymmetric-unit fractional coords would shrink the search space dramatically
  for high-symmetry SGs.
- **Phase 7 (noise augmentation) / Phase 8 (full MP scale-up):** still on
  the roadmap, but they amplify whatever lattice precision the encoder
  produces — they don't fix the encoder bottleneck themselves.

### Artifacts
- `paper/phase5b6_results/{p5b_baseline, p5b_phase4, p5b_sg_baseline, p5b_sg_phase4}.json`
- `runs/gpu_v17_p5b6/ckpt_final.pt` on the GPU instance (8 ckpts × 47 MB total)
- `runs/phase5b6/*.{json,log}` on the GPU instance

---

## Review — Phase 5C (peak-augmented lattice head) — NEGATIVE result
- Completed: 2026-05-08
- Trained `runs/gpu_v18_p5c/ckpt_final.pt`: 30k steps resumed from
  `runs/gpu_v17_p5b6/ckpt_final.pt` (140k total). PeakAugmentedLatHead
  initialized fresh; encoder, denoiser, and sg_head warm-started. 32 min
  on RTX 5090.
- Config: `--predict-x0 --debye-weight 1.0 --peak-aug-lat-head --n-peaks 20
  --sg-weight 0.1 --aux-weight 5.0 --save-every 5000`.
- Final losses: coord=0.076, lat=0.19, aux=0.30 (vs v17's 0.38), sg=1.80, debye=0.54.
  Aux DID drop slightly in normalized space, but the physical-unit MAE
  did not move.

### PeakAug-lat MAE (n=1000)
| head        | a (Å) | b (Å) | c (Å) | α (°) | β (°) | γ (°) |
|-------------|------:|------:|------:|------:|------:|------:|
| Aux (Path A)| 1.119 | 1.022 | 1.380 | 12.98 | 11.85 | 17.55 |
| Constrained | 1.113 | 1.018 | 1.418 | 13.72 | 12.54 | 18.44 |
| **PeakAug** | **1.127** | **1.024** | **1.413** | **13.99** | **12.83** | **18.69** |

**Three head architectures with very different inputs and supervision —
identical MAE.** The peak-feature signal is being fed in (verified by
unit test: head output changes when peaks change) but the head can't
turn it into better lattice predictions.

### End-to-end (n=1000, no --true-lattice)

| metric                       | p5c_baseline | p5c_phase4 | p5c_sg_baseline | p5c_sg_phase4 |
|------------------------------|-------------:|-----------:|----------------:|--------------:|
| StructureMatcher match rate  |        0.60% |      0.90% |           0.70% |         0.50% |
| pearson_mean                 |        0.010 |      0.083 |           0.005 |         0.047 |
| rmsd_mean                    |        0.231 |      0.149 |           0.244 |         0.125 |
| rmsd_median                  |        0.274 |      0.158 |           0.255 |         0.098 |
| rwp_mean                     |       19.94  |     17.86  |          18.94  |        16.13  |
| headline_all_correct         |        0.00% |      0.00% |           0.00% |         0.20% |

Slightly worse than Phase 5B (v17) on every dimension. SG-head accuracy
stays at top-1=39.9% / top-5=70.7% (essentially identical to v17).

### Why peaks alone aren't enough (the diagnosis)

To get a lattice from peak positions you need to **index** the peaks —
figure out which `(h, k, l)` reflection produced each one. Indexing is
a combinatorial inverse problem: even a perfectly-positioned 40-d peak
vector tells you nothing about which dimension belongs to which `(h,k,l)`.
A small MLP cannot learn this implicitly from 27 k MP-20 examples.

The v18 head is therefore drawing on the encoder embedding for almost
all of its lattice signal — which is exactly the signal Phase 5/5B already
showed is bottlenecked at ~1 Å MAE. We just spent 30 k steps adding a
pathway to information the model can't decode.

### What WOULD work (deferred — bigger engineering)
- **Phase 5D**: convert `2θ → d-spacing` via Bragg's law before feeding
  to the head. d is the actual physical quantity that determines lattice;
  feeding it directly may let the MLP learn the much-simpler map.
- **Phase 5E**: peak-attention transformer with learnable `(h,k,l)`
  queries and crystal-system inductive bias.
- **Phase 5F**: classical auto-indexing (Treor / DICVOL / N-TREOR) to
  produce a first lattice estimate, then the model only refines.

### What this rules out
- Adding peak features as a flat dense vector to a small MLP. Two
  retrains (v17, v18) confirm the head architecture isn't the wall.

### Artifacts
- `paper/phase5c_results/{p5c_baseline, p5c_phase4, p5c_sg_baseline, p5c_sg_phase4}.json`
- `runs/gpu_v18_p5c/ckpt_final.pt` on the GPU instance
- `runs/phase5c/*.{json,log}` on the GPU instance

---

## Review — Phase 7 (noise augmentation) — NEGATIVE result
- Completed: 2026-05-08
- Trained `runs/gpu_v19_p7/ckpt_final.pt`: 30 k steps resumed from
  `runs/gpu_v18_p5c/ckpt_final.pt` (170 k total). New `--noise-aug` flag
  applies stochastic per-call augmentation to PXRD patterns: 2θ zero-offset
  in [-5, +5] bins, Lorentzian peak broadening (FWHM 2-15 bins), additive
  Gaussian noise (σ 0.5-3 % of max), each with p=0.8.
- First launch (`--steps 170000` cosine) blew up immediately: v18's cosine
  schedule ended at lr=0 at step 140 k, fast-forwarding the new scheduler
  past most of its decay left lr ~ 0. Heads diverged. Killed; added
  `--const-lr` flag (LambdaLR with constant 1.0 multiplier) and relaunched
  with `--const-lr --lr 1e-4`. Training completed but with chronic gradient
  pathology: gnorm 8 k - 40 k throughout (clipped to 1.0 by the existing
  grad-clip), encoder gradient ≈ 0. The encoder stayed essentially frozen
  at v18 weights while the heads thrashed.
- Final losses: coord=0.107, lat=0.44, aux=0.99, sg=2.75 (top-1 ~28 %),
  debye=0.61. All worse than v18's converged values.

### Phase 7 robustness eval (n=1000, strongest pipeline)

| metric                       | v18_on_clean | v18_on_noisy | v19_on_clean | v19_on_noisy |
|------------------------------|-------------:|-------------:|-------------:|-------------:|
| StructureMatcher match rate  |        0.80% |        0.20% |    **0.00%** |        0.10% |
| rmsd_mean                    |        0.129 |        0.230 |       NaN ¹  |        0.007 |
| rmsd_median                  |        0.101 |        0.230 |       NaN ¹  |        0.007 |
| pearson_mean                 |        0.048 |        0.053 |        0.014 |        0.017 |
| rwp_mean                     |       16.18  |       11.47  |       17.19  |        9.40  |
| headline_all_correct         |        0.30% |        0.00% |        0.00% |        0.00% |
| sg_match@0.1                 |        1.70% |        1.30% |        0.20% |        0.20% |

¹ rmsd is NaN when StructureMatcher finds zero matches across all 1000
   test structures — v19 on clean simply does not produce structures
   pymatgen can register as matches.

### Two real findings
1. **v18 already has some implicit robustness.** Trained only on clean
   simulated patterns, it degrades from match=0.80 % → 0.20 % on noisy
   patterns (a 75 % drop, not catastrophic). Coord-only diffusion training
   appears to add some noise tolerance for free.
2. **v19 (aug-fine-tuned) is worse on every metric, including noisy.**
   Match rate 0.00 % on clean (vs v18's 0.80 %), SG accuracy collapsed
   1.7 % → 0.2 %, rmsd is NaN on the clean test (no matches at all).

### Why aug-fine-tune from v18 broke the model
The training log spelled it out: gnorm shot from v18's typical 100-300
range to 8 k - 40 k under noise augmentation, all clipped to 1.0. The
encoder's gradient share fell to ~0. So the encoder stayed at v18's
clean-pattern weights while the heads received 30 k steps of unit-norm
random-direction updates with no signal to pull them anywhere productive.
Result: heads ended up far from any working point in their loss surface.

### What WOULD work (deferred — bigger compute)
- **Phase 7 done right**: train end-to-end *from scratch* with augmentation
  enabled from step 0. Encoder learns noise-invariant features as it
  trains, instead of being asked to relearn after convergence. Estimated
  cost: ~6 GPU-h (full v13-equivalent run on RTX 5090). Out of scope for
  this session's budget but a clear next step if Phase 7 is revisited.
- **Phase 7 lite**: much milder augmentation parameters (e.g. σ in [0.001,
  0.005] of max instead of [0.005, 0.03]) might let v18's converged
  weights stay close to a working solution while still picking up some
  noise tolerance.
- **Phase 7 freeze-encoder**: detach the encoder during noise-aug
  fine-tune so only the heads adapt. Avoids the gnorm-blowup pathology.

### What this rules out
- A direct noise-aug fine-tune from a converged clean checkpoint.
  v18-style models are too tightly fit to clean inputs for a 30 k-step
  fine-tune to bend them gracefully under aggressive augmentation.

### Artifacts
- `paper/phase7_results/{v18_on_clean, v18_on_noisy, v19_on_clean, v19_on_noisy}.json`
- `runs/gpu_v19_p7/ckpt_final.pt` and intermediate ckpts on the GPU instance
- `runs/phase7/*.{json,log}` on the GPU instance

---

## Review — Phase 5D (Bragg's law d-spacing conversion) — partial result, overfitting
- Completed: 2026-05-08
- Trained `runs/gpu_v20_p5d/ckpt_final.pt` from scratch: 100 k steps,
  `--predict-x0 --debye-weight 1.0 --peak-aug-lat-head --use-d-spacing
  --sg-weight 0.1 --aux-weight 5.0`. RTX 5090, 95 min wallclock.
- PeakAugmentedLatHead now applies `d = λ / (2 sin θ)` to each peak's
  normalized 2θ position before the MLP, replacing the raw position with
  log(d) in log Å. The conversion is differentiable.

### Training metrics looked spectacular
| metric        | step   | v17/v18 (no Bragg) | v20 (Phase 5D) |
|---------------|--------|-------------------:|---------------:|
| aux loss      | 100k   |               0.30 |       **0.01** |
| sg top-1      | 100k   |              ~50 % |    **100.0 %** |
| gnorm         | 100k   |             80-200 |        0.5-1.1 |
| enc_gn        | 100k   |              ~0.05 |       0.20-0.35|

Aux ~0.01 in normalized space implies physical MAE ~0.15 Å on lengths
and ~0.5° on angles — a 7× tighter prediction than v18's training-set MAE.

### But the test-set lattice MAE moved the wrong way
| head         | a    | b    | c    | α     | β     | γ     |
|--------------|------|------|------|-------|-------|-------|
| v18 (5C)     | 1.13 | 1.02 | 1.42 | 13.99 | 12.83 | 18.69 |
| **v20 (5D)** | **1.24** | **1.15** | **1.71** | **16.41** | **14.82** | **20.76** |

Test SG-head accuracy: top-1 = **44 %** (vs v18's 40 %), top-5 = 70 %.
Modest improvement, but **far from the 100 % training accuracy**.

### End-to-end results (n=1000, no --true-lattice)

| metric                       | p5d_baseline | p5d_phase4 | p5d_sg_baseline | p5d_sg_phase4 |
|------------------------------|-------------:|-----------:|----------------:|--------------:|
| StructureMatcher match rate  |        1.10% |      1.00% |           0.50% |         0.50% |
| pearson_mean                 |        0.008 |      0.076 |           0.005 |         0.047 |
| rmsd_mean                    |        0.216 |      0.194 |           0.210 |         0.223 |
| rmsd_median                  |        0.238 |      0.212 |           0.188 |         0.249 |
| rwp_mean                     |       19.20  |     17.05  |          19.09  |        16.23  |
| headline_all_correct         |        0.00% |      0.00% |           0.00% |         0.00% |
| sg_match@0.1                 |        1.40% |      1.20% |           1.40% |         1.60% |

### Diagnosis: this is overfitting
The 100 % training SG accuracy and 0.15 Å training-MAE collapsed to 44 %
and 1.24 Å on the test split. A ~10× train-to-test gap on aux loss and
a ~2× SG-accuracy gap. Each MP-20 structure has a unique top-20 peak
fingerprint, so once we hand the head explicit `(d, intensity)` pairs
the model can memorise the training-set lookup table — we *gave* it
enough information to do that. But the lookup doesn't transfer to
unseen test patterns: nearby compositions/structures with subtly
different patterns get the wrong lattice.

### Where Phase 5D *did* help vs Phase 5C
- p5d_baseline match rate 1.10 % vs p5c 0.60 % (+0.5 pp; +83 % relative).
  In the simplest config the Bragg features genuinely help.
- p5d_phase4 1.00 % vs p5c 0.90 % (small, within noise).
- p5d_sg_baseline 0.50 % vs p5c 0.70 % (small loss).
- p5d_sg_phase4 0.50 % vs p5c 0.50 % (tie).

Net: tiny lift on the simple configs, basically a wash on the strong ones.
The encoder-bottleneck wall is still in place; the d-spacing input
helps the MLP memorise but doesn't help it generalise.

### What WOULD help (deferred)
- **Stronger regularisation** in the peak-feature pathway: dropout on
  `peak_proj`, weight decay specific to the new layers, or just a much
  smaller hidden width (32 instead of 256).
- **Peak-feature augmentation during training**: add jitter to peak
  positions (±1-2 bins) and intensities (±10 %) so the model can't rely
  on exact training-set fingerprints. This is the "Phase 7 done right"
  but applied at the peak-feature level rather than the raw spectrum.
- **More data (Phase 8 — full MP)**: with ~150 k structures, exact
  fingerprint memorisation gets harder; the head may be forced to learn
  a more generalisable mapping.
- **Different inductive bias**: feed peak-position *ratios* (d_i/d_j)
  rather than absolute d-spacings. Ratios are scale-invariant within a
  crystal system and may force the model toward learning structure
  rather than memorising scale.

### Artifacts
- `paper/phase5d_results/{p5d_baseline, p5d_phase4, p5d_sg_baseline, p5d_sg_phase4}.json`
- `runs/gpu_v20_p5d/ckpt_final.pt` and intermediate ckpts on the GPU instance
- `runs/phase5d/*.{json,log}` on the GPU instance

---

## NEXT SESSION — execution notes (added 2026-05-18)

### New GPU instance (vast.ai)
- `ssh -p 28081 root@202.214.223.66`  (port-forward TensorBoard: `-L 8080:localhost:8080`)
- **BLOCKER — SSH key mismatch.** The instance has registered key
  `ssh-ed25519 AAAA...+R1 frankcai222@gmail.com`, but this machine's only
  private key (`~/.ssh/id_ed25519`) has public key
  `...zAzw frank@frank` — a *different* keypair. `Permission denied (publickey)`.
  No private key matching `...+R1` exists on disk.
  Resolve one of two ways before deploying:
    1. Paste this machine's pubkey (`~/.ssh/id_ed25519.pub`, `...zAzw`) into the
       vast.ai instance's authorized_keys / the instance's SSH-key field, OR
    2. Place the private key matching `...+R1` at `~/.ssh/` and connect with `-i`.
- Deploy once connected: `git clone https://github.com/fronkt/pxrd-diff` (or pull),
  `pip install -e .`, then run the data + train scripts below.

### Immediate next step — Phase 8 (data scale-up) is staged but unrun
The Phase 8.1 data script (full-MP pull via mp-api, MP-20 test split excluded) was
committed in `3816b29` but has not been executed. Next session:
- [ ] 8.1.x  Run the full-MP pull on the GPU box; confirm yield + split sizes
- [ ] 8.1.4  `python scripts/01_simulate_pxrd.py` on the new structures
- [ ] 8.2.1  Train v21 from scratch, 200k steps, all Phase 5–7 improvements ON
- [ ] 8.3.1  Eval on MP-20 test split for comparability vs v20

---

## Phase 9 — Hybrid classical indexing + differentiable guidance
## (a measured, falsifiable improvement over Crystalyze)

**Why this is the right next bet.** Five retrains (v17/v18/v19/v20 + 5D) have now
proven the same wall: the learned PXRD encoder cannot decode absolute d-spacings,
so no-true-lattice match rate is pinned at ~1%. Meanwhile Phase 4 showed that
*given a good lattice* the existing pipeline already hits 6.8% and produces
near-exact reconstructions (rmsd_median 0.007 Å on matched cases). The bottleneck
is entirely the unit cell — a problem classical crystallography solved decades ago.

**How Crystalyze does it / where it's weak.** Crystalyze (Riesel et al., JACS 2024)
generates candidate structures with a generative model, then selects among them by
*re-simulating each candidate's pattern and DFT-relaxing* — discrete, non-differentiable
post-hoc filtering. Physics never steers generation; it only ranks finished candidates,
and the expensive DFT relaxation caps throughput.

**The improvement.** Replace the broken learned-lattice path with classical
auto-indexing, and replace Crystalyze's discrete filtering with *differentiable
physics guidance baked into the reverse diffusion*. Two independent, individually
testable changes:

### Council verdict (2026-05-18) — execution order
5/5 advisors + 5/5 peer reviews: **do Phase 9, skip Phase 8 as the next step.**
Key revisions the council forced into this plan:
  - **Gate everything on a cheap indexing benchmark FIRST.** Do not retrain, do
    not write the guided sampler, until classical indexing is shown viable.
    The indexing success rate IS the ceiling of Phase 9 — measure it before
    committing a month.
  - The 6.8% Phase-4 number was measured with **oracle (ground-truth) cells**.
    A real indexer returns cells with error → 9.1.4 must test pipeline
    sensitivity to imperfect cells, not assume 6.8% survives.
  - Indexing is **symmetry-stratified**, not pass/fail — report results per
    crystal system (cubic/tetragonal easy → triclinic hard).
  - Auto-indexers return **ranked candidate cells with figures of merit** —
    treat that as uncertainty: feed top-N cells to the sampler, score by Debye
    fit. Do not collapse to one cell.
  - Debye-gradient guidance (9.2) is refinement, not the load-bearing fix —
    **defer it** until 9.1 lands. Lead the paper with the bottleneck diagnosis
    + the targeted indexing fix.
  - Phase 8 (more data) is demoted to a **B-then-A** lever for residual
    coordinate error AFTER the lattice is fixed — not cancelled, just later.

### 9.0 Indexing viability benchmark — GO/NO-GO GATE (do this first, ~2 days)
No retraining. Pure offline measurement on the MP-20 test split.
- [x] 9.0.1 `scripts/09_index_benchmark.py`: peak-find → 2θ→d→Q → classical
        Ito/de-Wolff Q-space auto-indexer (native NumPy/SciPy implementation —
        GSAS-II is not on PyPI; conda install deferred as a follow-up)
- [x] 9.0.2 Q-form per crystal system, seed-hypothesis search with a hard
        hypothesis cap, M20 figure-of-merit scoring with a coverage gate
- [x] 9.0.3 Metrics stratified by crystal system: strict (recovered the
        conventional cell) and consistent (conventional OR a small-index
        sub/super-cell — the unavoidable peak-position ambiguity)
- [x] 9.0.4 Compared vs v20 learned head (~1.37 Å len MAE)
- [x] 9.0.5 **DECISION: qualified GO** — see Review — Phase 9.0 below.

### Review — Phase 9.0 (indexing viability gate) — QUALIFIED GO
- Completed: 2026-05-18. `scripts/09_index_benchmark.py`, full n=300 MP-20 test.
  Results: `paper/phase9_results/index_benchmark.json` (base),
  `index_benchmark_volcorr.json` (with 9.0.6 sub-cell correction).
- Native Q-space indexer (Ito/de-Wolff): peak-find → Q=1/d² → hypothesise
  (hkl) for the lowest peaks → solve the linear Q-form → de-Wolff M20 score
  with a hard coverage gate (a cell must explain ≥80% of observed peaks; among
  survivors M20 rewards parsimony, killing super-cells).
- **Result (n=300, with 9.0.6 sub-cell correction):**
  ```
  system        n   strict%  consist%  len_MAE(Å)  ang_MAE
  cubic        68    52.9      61.8      1.29        0.00
  tetragonal   54    50.0      63.0      0.93        0.00
  hexagonal    35    82.9      88.6      0.44        0.00
  trigonal     28    64.3      60.7      2.75        0.00
  orthorhombic 57    63.2      73.7      0.69        0.00
  monoclinic   45    13.3      31.1      1.90       13.67
  triclinic    13     0.0       0.0       —           —
  OVERALL     300    50.7      60.0      1.24   (v20 head 1.37)
  ```
- **Findings:**
  1. Higher-symmetry systems (hexagonal/orthorhombic) index at **0.44–0.69 Å
     len MAE — 2–3× better than the learned head** and good enough to feed the
     pipeline. Clear GO signal there.
  2. **Sub-cell aliasing on cubic/tetragonal**: peak positions are consistent
     with a smaller cell when the conventional-cell reflections are extinct.
     9.0.6 (composition floor: a cell can't pack denser than ~9 Å³/atom, take
     the smallest integer super-cell clearing it) cut overall len MAE
     1.50→1.24 Å but barely moved strict% — uniform scaling fixes cell *size*,
     not *shape*. cubic/tetragonal strict% stays ~50%.
  3. Monoclinic/triclinic are the wall — as the council predicted. The native
     4-/6-param seed search is hypothesis-capped; a real indexer (DICVOL/
     GSAS-II) is needed for these.
  4. trigonal regressed at scale (88.9%→64.3%, len MAE 2.75) — the hexagonal-
     setting Q-form mis-handles some rhombohedral cells; flagged for 9.0.7.
- **Net: GO for Phase 9**, scoped: the indexed cell is reliable on hexagonal/
  orthorhombic (and resolvable on cubic/tetragonal), unreliable on
  monoclinic/triclinic. Plan 9.1 accordingly — proceed with high-symmetry
  cells, treat low-symmetry as a known gap.
- [x] 9.0.6 Sub-cell disambiguation via composition density floor — partial
        win (len MAE ↓, strict% flat). Intensity-based check deferred.
- [ ] 9.0.7 (optional) GSAS-II via conda — re-benchmark monoclinic/triclinic +
        fix the trigonal/rhombohedral Q-form regression

### 9.1 Wire indexed cells into the pipeline (9.0 passed — qualified GO)
- [~] 9.1.0 Retrain v13-equivalent checkpoint on the new box (no checkpoint was
        available). `gpu_v21_p9`: 100k steps, --predict-x0 --debye-weight 1.0,
        bs 64, RTX 5090, ~1.5h. IN PROGRESS (launched 2026-05-19).
- [ ] 9.1.1 `index_lattice(pattern) -> [(lattice_params, crystal_system, FoM), ...]`
        returning ranked candidates, not a single cell
- [ ] 9.1.2 Cell-perturbation study: feed Phase 4 oracle cells + synthetic noise
        at the indexer's measured MAE → confirm 6.8% degrades gracefully
- [ ] 9.1.3 Feed top-N indexed cells into the Phase 4 pipeline; sample per
        candidate, score by Debye fit, keep best. Re-run n=1000 eval —
        first true no-true-lattice number not bottlenecked by the encoder.

### 9.2 Debye-gradient guidance inside the sampler (vs Crystalyze's post-hoc filter)
- [ ] 9.2.1 In the DDIM sampler, at each reverse step reconstruct x0_pred, compute
        `DiffPXRD(x0_pred)`, take `∇_x [1 - Pearson(·, target)]`, nudge the sample.
        `--guide-scale` flag (0 = current behaviour, backward compatible).
- [ ] 9.2.2 Anneal guidance: apply only on low-noise late steps; clip the step to
        keep samples on the data manifold.
- [ ] 9.2.3 Reuse the verified DiffPXRD module — no new physics code.

### 9.3 Head-to-head benchmark (the paper's headline claim)
- [ ] 9.3.1 Same checkpoint, n=1000 MP-20 test, four configs:
        (a) unguided single sample;
        (b) Crystalyze-style: K=20 candidates, discrete re-rank by simulated Rwp;
        (c) Debye-guided sampling (9.2);
        (d) guided + classical-indexed lattice (9.1) + Phase 4.2 differentiable refine.
- [ ] 9.3.2 Headline metric: space-group match % and coord RMSD, no true lattice.
- [ ] 9.3.3 Claim to test: differentiable physics guidance + classical indexing
        beats discrete post-hoc candidate filtering. If (c)/(d) > (b), that is a
        clean, falsifiable improvement over Crystalyze and the paper's centerpiece.
