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
- [ ] 0.2 Pull MP-20 canonical splits from CDVAE repo (train/val/test CSVs with CIFs)
- [ ] 0.3 PXRD simulator (pymatgen XRDCalculator wrapper, Cu Kα, configurable grid)
- [ ] 0.4 Cache simulated patterns (npz) for all train/val/test
- [ ] 0.5 Evaluation module: spacegroup match, coord RMSD, Rwp, element accuracy
- [ ] 0.6 Smoke tests: identity baseline scores 100%, random scores ~0%
- [ ] 0.7 Visualization notebook (5 random structures + their PXRD)
- [ ] 0.8 Push to GitHub

### Phase 1 — Conditional diffusion baseline (Weeks 2-4, CPU prototype)
- [ ] 1.1 Frozen MACE-MP-0 encoder integration; cache embeddings
- [ ] 1.2 1D ResNet PXRD encoder
- [ ] 1.3 Diffusion module: fractional coords on flat torus + lattice in R^6
- [ ] 1.4 Cross-attention conditioning between PXRD embedding and structure tokens
- [ ] 1.5 EDM-style denoising training loop (small-batch CPU smoke)
- [ ] 1.6 Sampler (DDIM / Euler) — reverse process from noise to structure
- [ ] 1.7 End-to-end smoke: train 100 steps on 16 structures, sample, eval

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

## Review
<!-- Filled in after Phase 0 -->
- Completed:
- What worked:
- What changed from plan:
- Known limitations: