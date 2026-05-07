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
MP-20 has 27k train structures. Full MP has ~154k DFT-computed structures.
More data is the safest scaling lever for a diffusion model at this size.

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
