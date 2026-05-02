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
- [ ] 2.5.3 Coord prediction architecture investigation
      - Hypothesis: at high noise, atoms have no positional anchoring to
        specific PXRD features. Cross-attention provides global context but
        no way to associate "this peak corresponds to this atom".
      - Try predicting x0 directly instead of eps (more stable at low SNR)
      - Try larger model (more layers, bigger d_model)

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
