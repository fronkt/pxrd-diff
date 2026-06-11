# PXRD-Diff

E(3)-equivariant conditional diffusion for inverting 1D powder X-ray diffraction (PXRD) patterns into 3D crystal structures.

**Core novelty:** differentiable Debye scattering loss as physics-informed training signal alongside denoising score matching.

## Status
Phase 0 — Data pipeline + evaluation harness (no model code yet).

## Scope lock
- MP-20 dataset only (canonical CDVAE splits: 27,136 train / 9,047 val / 9,046 test)
- ≤20 atoms / unit cell
- Frozen pretrained MACE encoder for structure embeddings
- Differentiable Debye loss is the protected contribution — do not dilute

## Falsifiable claim
> E(3)-equivariant conditional diffusion with differentiable Debye loss recovers crystal
> structures from simulated PXRD patterns at X% space-group match rate and Y Å coordinate
> RMSD on the MP-20 test set, outperforming DiffractGPT and Crystalyze baselines.

## Quickstart
```bash
pip install -e .
python scripts/00_pull_mp20.py
python scripts/01_simulate_pxrd.py
pytest tests/
```

## Layout
```
src/pxrd_diff/   # library code
scripts/         # one-shot pipelines (data pull, simulation, training, eval)
data/raw/        # MP-20 CSVs from CDVAE
data/cache/      # cached simulated PXRD patterns
configs/         # training/sim configs (yaml)
tests/           # pytest smoke tests
notebooks/       # exploration
tasks/           # todo.md, lessons.md (agentic-eng-workflow)
```

## Compute
- Phase 0–1: CPU only (Windows Python 3.12)
- Phase 2+: rented cloud GPU (Vast.ai/RunPod RTX 3090)
- Phase 3+: Cloud GPU 5090. 100+ DP
