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

- [ ] 1.5.1 PXRDEncoder: return multi-res feature maps alongside global vector
- [ ] 1.5.2 CrystalDenoiser: add per-layer cross-attention (atoms attend to PXRD features)
- [ ] 1.5.3 DDIMSampler: wire up new encoder output (global + feature maps)
- [ ] 1.5.4 Training script: update for new signatures, keep aux head on global vector
- [ ] 1.5.5 Local smoke test (CPU, 200 steps) — verify shapes, gradients, no NaN
- [ ] 1.5.6 Push to GitHub, deploy to new Vast.ai RTX 5090 instance
- [ ] 1.5.7 Full training run (100k steps), compare coord loss trajectory to v4

### Phase 2 — Differentiable Debye scattering loss (Weeks 5-6)
- [ ] 2.1 Implement differentiable Debye scattering in PyTorch (vectorized)
- [ ] 2.2 Verify against pymatgen XRDCalculator on 50 known structures
- [ ] 2.3 Add as auxiliary loss with λ_debye hyperparameter
- [ ] 2.4 Ablation: λ_debye ∈ {0, 0.1, 1, 10}

### Phase 3 — Cloud GPU scale-up + baselines (Weeks 7-9)
- [ ] 3.1 Provision cloud GPU (Vast.ai / RunPod RTX 3090, 24 GB)
- [ ] 3.2 Full training run on MP-20 train split
- [ ] 3.3 DiffractGPT baseline reproduction
- [ ] 3.4 Crystalyze baseline reproduction
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
