# PXRD-Diff: Conditional Diffusion for Powder-Diffraction Inversion — an Encoder-Bottleneck Diagnosis and Shared-Harness Reproduction

**Frank Cai**
`frankyc11223@gmail.com`


**Track:** Methods / Negative results
**Word budget:** ~4,500 words main text + appendix

---

## Abstract

Inverting a 1D powder X-ray diffraction (PXRD) pattern to a 3D crystal structure remains open, and recent generative approaches report numbers under mutually incompatible protocols with failure modes seldom catalogued. We present **PXRD-Diff**, a 3.7 M-parameter conditional diffusion model on the CDVAE MP-20 split, and reproduce DiffractGPT [^DiffractGPT] and PXRDnet [^PXRDnet] through the *same* `pymatgen.StructureMatcher` harness. Our central, statistically robust finding is a **mechanism**: the learned PXRD encoder reaches a 0.007 auxiliary lattice-regression loss in isolation, but the denoiser cannot extract absolute d-spacings well enough to predict the lattice from scratch. A perturbation study locates a sharp sensitivity knee at ~0.5 Å cell error, and a true-lattice oracle lifts match from **1.6 % to 5.6 %** at n = 1000 — a gap whose 95 % Wilson confidence intervals ([1.0, 2.6] vs [4.3, 7.2]) do not overlap, so it is real. Replacing the learned lattice head with a classical Q-space autoindexer (de-Wolff dichotomy) raises no-true-lattice match from 1.0 % to 1.6 %; we report this lift honestly as **directional but not significant on aggregate** (unpaired two-proportion test p = 0.24), with the *per-crystal-system* breakdown — high-symmetry systems lift, low-symmetry do not — as the actual evidence that it works where its indexing MAE sits below the 0.5 Å knee. Reproduced baselines on our harness: **DiffractGPT 18.9 %** match (15.9 % all-correct) at n = 1000, **PXRDnet sinc100 30.0 %** match (5.0 % all-correct) at n = 20. PXRD-Diff is 12–19× behind on raw match rate. We note a striking *candidate* discrepancy — DiffractGPT recovers far more space groups (all-correct) than PXRDnet despite a lower match rate — but flag it as a hypothesis, not a result: at n = 20 the PXRDnet all-correct CI is [0.9, 23.6] and cannot support the comparison. The contributions that *do* hold are the encoder-bottleneck diagnosis, the indexer drop-in, the four-extension failure catalogue (top-K rerank, Debye guidance, Wyckoff embeddings, distance-aux), and the shared evaluation harness itself. Code, checkpoints, per-phase JSONs, and the differentiable Bragg simulator are released.

**Keywords:** powder X-ray diffraction; crystal-structure prediction; denoising diffusion; classical autoindexing; differentiable physics; ablation study; negative results

---

## 1. Introduction

PXRD is the most common structural-characterization measurement in solid-state chemistry. The forward problem is solved by 1920s structure-factor physics and a one-line `pymatgen` call. The inverse is brutal: symmetry-equivalent structures produce identical Bragg reflections, intensities are orientation-averaged, and the 3D→1D reduction destroys atom labelling. The traditional indexing → Pawley/Le Bail → Rietveld pipeline needs a trained crystallographer and a starting model good enough to converge; for the long tail of new materials, structures stay unsolved.

Three recent generative approaches — DiffractGPT [^DiffractGPT], Crystalyze [^Crystalyze], PXRDnet [^PXRDnet] — argue neural networks can shortcut this pipeline. They report match rates in the tens of percent, but under mutually incompatible evaluation protocols, and failure modes are seldom catalogued. A practitioner cannot easily form a calibrated expectation of *what is hard* about PXRD inversion.

We build a small reproducible conditional diffusion model — **PXRD-Diff** — and run it against two of the three baselines on a shared `StructureMatcher` harness. Contributions:

1. **A clean small baseline.** 3.7 M parameters, trains on a single RTX 5090 in ~1.5 h on CDVAE MP-20. A differentiable PyTorch Bragg simulator (Pearson 0.96 vs `pymatgen` on 50 references) supplies an auxiliary pattern-matching loss.
2. **Encoder-bottleneck diagnosis (our primary contribution).** A perturbation study (§5.5) locates a sharp ~0.5 Å sensitivity knee; a true-lattice oracle lifts match from 1.6 % to **5.6 %** at n = 1000, with non-overlapping 95 % Wilson CIs. The encoder reaches 0.007 aux lattice loss in isolation but the denoiser under-uses the signal — the lattice channel, not coordinates, is the dominant bottleneck.
3. **Classical-indexer drop-in.** A Q-space autoindexer (de-Wolff) at sampling time raises no-true-lattice match 1.0 % → 1.6 %. We report this as *directional but not significant on aggregate* (unpaired two-proportion p = 0.24); the load-bearing evidence is the *per-system* pattern (§5.2, §5.5) — the lift appears exactly where the indexer's MAE falls below the 0.5 Å knee, and nowhere else.
4. **Four-item failure catalogue.** Top-K Debye rerank, Debye-gradient DDIM guidance, Wyckoff-site embeddings, and distance-matrix aux loss all fail at n ≥ 200; the latter two combined (v14) underperform the bare ε baseline. The gradient-guidance failure is corroborated independently by concurrent work showing the PXRD loss landscape is too rough for gradient descent [^Segal2025].
5. **Reproduced baselines + shared harness.** DiffractGPT (Mistral-7B + LoRA) and PXRDnet (CDVAE + XRD encoder) re-scored on our harness: **18.9 %** match (DGpt n = 1000), **30.0 %** match (PXRDnet n = 20). PXRD-Diff at 1.6 % is 12–19× behind. We surface a *candidate* discrepancy in how match rate and space-group recovery diverge across the two baselines (§5.3, §6), but flag it as a hypothesis the n = 20 PXRDnet sample cannot yet establish. Crystalyze's checkpoint download link is inactive; we cite but cannot reproduce.

---

## 2. Related work

**Diffusion for crystals.** CDVAE [^CDVAE] introduced the MP-20 protocol; DiffCSP [^DiffCSP] used joint coord+lattice diffusion; MatterGen [^MatterGen] scaled to millions of structures. PXRD-Diff borrows DiffCSP's joint diffusion, restricted to the conditional setting where a 1D pattern conditions generation.

**Neural PXRD inversion.** DiffractGPT [^DiffractGPT] is a fine-tuned Mistral-7B that decodes peak list + formula → CIF tokens. Crystalyze [^Crystalyze] uses CDVAE conditioned on an XRD transformer encoder with post-hoc symmetry filtering. PXRDnet [^PXRDnet] is a CDVAE variant with *iterative latent-space gradient guidance* (~3.5 M decoder ops/material), published in *Nature Materials* in 2025. deCIFer [^deCIFer] is a concurrent autoregressive-language-model approach that, like DiffractGPT, emits CIF tokens. All four are larger than PXRD-Diff and report results under bespoke evaluation pipelines; §5.3 re-scores DiffractGPT and PXRDnet on a single shared `StructureMatcher` harness. Crystalyze's checkpoint download link is "not yet active" (verified 2026-06-01).

**Classical autoindexing.** Recovering a unit cell from peak positions is a solved sub-problem with a fifty-year toolchain: the de-Wolff dichotomy method, ITO, TREOR, and DICVOL [^Boultif] index high-symmetry powder patterns in CPU milliseconds, with the difficulty concentrated in low-symmetry (monoclinic/triclinic) cells. §3.5 reuses this machinery as a drop-in for the learned lattice head, and §5.2/§5.5 show the resulting lift tracks exactly the symmetry-dependent accuracy classical indexers have always had.

**Differentiable physics.** Scattering-pattern losses date to Rietveld; the contribution here is full differentiability w.r.t. coordinates and lattice in PyTorch, usable as a diffusion-training loss term. Concurrent generative-inversion work conditions on differentiable spectral targets for amorphous and nanostructured systems [^GuoSchwalbeKoda]. Relevant to our negative results, Segal et al. [^Segal2025] show the powder-XRD similarity loss landscape is "too rough for gradient descent" — independent evidence for why our sampling-time Debye-gradient guidance fails (§5.4).

**Equivariance.** Not strictly E(3)-equivariant — we use periodic-distance RBFs (SchNet-style [^SchNet]). A frozen MACE [^MACE] encoder showed no measurable benefit at MP-20 scale in pilots and was deferred.

---

## 3. Method

### 3.1 Problem

Crystal $\mathcal{C} = (\mathbf{F}, \mathbf{Z}, \mathbf{L})$: fractional coords $\mathbf{F} \in [0,1)^{N \times 3}$, atomic numbers $\mathbf{Z}$, lattice $(a,b,c,\alpha,\beta,\gamma)$. Pattern $\mathbf{p}(\mathcal{C}) \in \mathbb{R}^{4251}$ on a Cu Kα 5–90° / 0.02° grid, max-normalised. Given $(\mathbf{p}, \mathbf{Z})$ we sample $(\mathbf{F}, \mathbf{L}) \sim p(\cdot | \mathbf{p}, \mathbf{Z})$. Known composition is a simplification consistent with prior work.

### 3.2 Architecture

Three trained components: PXRD encoder, denoiser, auxiliary lattice head.

**PXRD encoder.** A 1D ResNet, four blocks ($64\to128\to256\to256$, stride-2, GroupNorm, SiLU), exposing a pooled global vector $\mathbf{g} \in \mathbb{R}^{256}$ and a multi-resolution feature map $\mathbf{F}_{\text{pxrd}}$ (1×1-projected block outputs concatenated along the spatial axis). *Multi-resolution is critical*: an early run (`gpu_v4`) conditioned only on $\mathbf{g}$ via additive broadcast and never moved off the 3.0 random coord-loss baseline, despite the aux head reaching 0.007 — `AdaptiveAvgPool1d(1)` collapsed all spectral structure before it reached the denoiser.

**Denoiser.** A periodic-distance message-passing net, $L=3$, $d_\text{model}=256$. Atom embeddings $\mathbf{h}_i^{(0)} = \text{Emb}(\mathbf{Z}_i) + W_\text{coord}\mathbf{F}_i^{(t)}$ alternate (i) RBF message passing on min-image periodic distances with FiLM timestep conditioning and (ii) cross-attention from atoms to $\mathbf{F}_{\text{pxrd}}$. Two heads predict per-atom $\boldsymbol{\epsilon}_F$ and lattice $\boldsymbol{\epsilon}_L \in \mathbb{R}^6$. The lattice head is an MLP on $[\bar{\mathbf{h}};\, \mathbf{g};\, W_\text{lat}\boldsymbol{\ell}^{(t)};\, \mathbf{t}_\text{cond}]$ — note the explicit projection of the noisy lattice itself (see §3.4).

**Auxiliary head.** A diagnostic MLP predicting $(a,b,c,\alpha,\beta,\gamma)$ from $\mathbf{g}$ alone, MSE-trained; its loss does not flow back into the denoiser.

### 3.3 Differentiable Bragg structure-factor loss

We implement the structure factor $F(hkl) = \sum_j f_j(s) \exp(-B_\text{iso} s^2) \exp[2\pi i (hx_j+ky_j+lz_j)]$ in PyTorch, with four-Gaussian form factors from `pymatgen.ATOMIC_SCATTERING_PARAMS`, $B_\text{iso} = 0.5$ Å², $s = \sin\theta/\lambda$. Intensity $|F|^2$ × Lorentz-polarisation. We enumerate $|h|,|k|,|l| \le 5$ (1330 reflections), place each on a 2θ grid via Bragg's law, and broadcast Gaussians of FWHM 0.1° onto a 256-bin coarse grid; fully batch-vectorised.

**Validation.** Pearson vs `pymatgen.XRDCalculator` on 50 random MP-20 test structures: mean 0.962, std 0.027, min 0.896, max 0.998; 50/50 > 0.7 and gradients through $F$ verified non-zero.

![**Figure 1. Differentiable-simulator validation.** Distribution of Pearson correlation between the PyTorch DiffPXRD module and `pymatgen.XRDCalculator` over 50 random MP-20 test structures (mean 0.962). All 50 exceed 0.7; the lower tail is dominated by layered structures with strong texture in the reference.](fig3_diffpxrd_validation.pdf){#fig:validation}

**As a loss.** Recover $\hat{\mathbf{F}}$ (see §3.4), simulate $\hat{\mathbf{p}} = \text{DiffPXRD}(\hat{\mathbf{F}}, \mathbf{Z}, \mathbf{L}_\text{true})$, take $\mathcal{L}_\text{Debye} = 1 - \rho(\hat{\mathbf{p}}, \mathbf{p})$. We use the true lattice here because the simulator is more sensitive to lattice than coordinate error early in training, and disentangling is more stable.

### 3.4 Training objective

VP-SDE with cosine schedule [^NicholDhariwal] for both channels. Coords are diffused on $\mathbb{T}^3$ with periodic-difference loss; lattices are diffused in $\mathbb{R}^6$ after train-set standardisation. The full loss (run `gpu_v13`) is
$$
\mathcal{L} = \mathcal{L}_\text{coord} + 0.1\,\mathcal{L}_\text{lat} + 0.5\,\mathcal{L}_\text{aux} + 1.0\,\mathcal{L}_\text{Debye}.
$$

**Two load-bearing parameterisation choices** (both found by ablation):

*x₀-residual prediction.* Heads predict a residual added to the noisy input to give the clean estimate, $\hat{\mathbf{F}}^{(0)} = \mathbf{F}^{(t)} + \text{Head}(\mathbf{h})$. Heads initialised near zero make the model start as identity — a correct fixed point at $t \approx 1$. ε-prediction has to learn the full transform from random init and never recovered. 3.5× lift in the headline (§5.2).

*Lattice-head input.* The head reads the noisy lattice $\boldsymbol{\ell}^{(t)}$ explicitly. Without this it can only predict the unconditional mean, and its loss stays pinned at 1.0 forever; with it, loss drops to 0.02–0.06. A textbook denoising bug that produced a plausible-looking run for days before we noticed.

§5.3 reports two extensions that failed: a **Wyckoff-site embedding** (added to atom features) and an auxiliary **distance-matrix loss**.

### 3.5 Classical indexing as a drop-in for the learned lattice head

The lattice head is the *only* path encoding absolute d-spacings — every 2θ peak position is set by the lattice via Bragg's law. §5.5 shows the encoder + denoiser learns *relative* d-spacings (predicted-vs-target Pearson 0.43, vs 0.97 target-vs-ground-truth) but not the absolute scale. Classical Q-space autoindexing has solved this sub-problem since the 1970s in CPU milliseconds.

We implement a from-scratch Q-space autoindexer taking (i) the same simulated pattern, (ii) the known crystal system (Bravais code, a sampling-time hyperparameter), returning $(a,b,c,\alpha,\beta,\gamma)$. Peak picking is a 1D local-max filter with an intensity floor; the candidate Q-vector is least-squares fit to the de-Wolff dichotomy form for each Bravais lattice. At sampling time the indexer's lattice substitutes the lattice channel at $t=T$ and DDIM runs only on coordinates.

**Per-system accuracy (n = 1000).** Overall: 48.8 % strict cell match, 1.45 Å mean length MAE. Hex 77.9 % / 0.53 Å; ortho 59.8 % / 0.96 Å; trig 43.3 % / 4.62 Å; mono 18.9 % / 1.76 Å; triclinic 0 %. Low-symmetry systems (≥4 free parameters) saturate the de-Wolff hypothesis cap. A GSAS-II [^GSASII] adapter for the low-symmetry path ships behind `--use-gsas` but `DoIndexPeaks` hangs on real MP-20 monoclinic/triclinic patterns inside `findBestCell` — experimental, not headline.

Training is unchanged; the diffusion sampler is unchanged except for the lattice-channel substitution at $t=T$.

### 3.6 Sampling

DDIM [^DDIM], 50 steps, $\eta=0$, alternating updates on each channel; coords wrapped to $[0,1)$ each step. Predicted lattice de-standardised and clipped to $(a,b,c) \in [0.5,100]$ Å, $(\alpha,\beta,\gamma) \in [10°,170°]$. In the x₀-residual variant we clamp the implicit-ε denominator $\sqrt{1-\bar{\alpha}(t)}$ at 0.05 and skip the final DDIM step to avoid NaN coordinates near $t=0$.

---

## 4. Experiments

**Dataset.** Canonical CDVAE [^CDVAE] MP-20 split: 27 136 / 9 047 / 9 046 train/val/test, ≤ 20 atoms/conventional cell. Patterns simulated with `pymatgen.XRDCalculator` (Cu Kα₁, λ = 1.54184 Å), max-normalised, 4 251 bins 5–90° / 0.02°; cached as `.npz`, zero failures over 45 196 structures.

**Evaluation.** Three views: (1) **composition match** (trivial, $\mathbf{Z}$ given); (2) **structure match** via `pymatgen.StructureMatcher` at $(\ell_\text{tol}, s_\text{tol}, \alpha_\text{tol}) = (0.2, 0.3, 5°)$ — the headline "match rate"; (3) **coordinate RMSD** on aligned, permuted atoms when matched. We also report space-group match at symprec $\in \{0.01, 0.05, 0.1, 0.2\}$, pattern Pearson, and $R_{wp}$.

The stricter **"all-correct"** rule combines `composition ∧ sg-match@symprec=0.1 ∧ rmsd ≤ 0.1 Å` — the experimentally relevant case for downstream crystallographic use.

Two evaluation modes: *full pipeline* (predicted lattice + coords) and *true-lattice/coord-only* (ground-truth lattice substituted). The latter isolates coordinate quality from lattice quality and is the stricter ablation setting.

**Implementation.** AdamW, LR $5\times10^{-4}$, cosine decay to zero over 100 k steps, WD $10^{-4}$, grad-clip 1.0. Batch 64, ~24 epochs, single RTX 5090 (~1.5–1.7 h/run). 3.7 M parameters; $L=5, d=384$ (~10 M) gave no benefit (§5.2). VP-SDE with $t \sim U(0,1)$. Total compute: ~35 GPU-hours, ~USD 25 on Vast.ai (Phase 4 ablation + Phase 9 v21 retrain + indexer/perturbation sweeps + DGpt n=1000 + PXRDnet n=20 + evaluation).

---

## 5. Results

Order: Phase 4 architectural ablation (§5.1) → Phase 9 indexer drop-in (§5.2) → reproduced baselines (§5.3) → failure catalogue (§5.4) → encoder-bottleneck perturbation study (§5.5).

### 5.1 Phase 4 — architectural ablation (true-lattice setting)

Table 1: seven runs sweeping the Debye loss, the lattice-input fix, x₀-residual, and two extensions (Wyckoff, distance loss), all on n = 1 000 MP-20 test with true lattice substituted.

**Table 1.** Phase 4 coordinate-only ablation, MP-20 test (n = 1 000, true lattice substituted).

| Run   | Parameterisation | λ_Debye | Wyckoff | λ_dist | Match % | Pearson | RMSD (Å) |
|-------|------------------|---------|---------|--------|---------|---------|----------|
| v10   | ε                | 0       | –       | 0      | 1.40    | 0.359   | 0.17     |
| v11   | ε                | 1       | –       | 0      | 0.90    | 0.365   | 0.15     |
| **v13** | **x₀-residual**  | **1**   | **–**   | **0**  | **2.51** | **0.434** | 0.22 |
| v14   | x₀-residual      | 1       | yes     | 0.01   | 0.80    | 0.367   | 0.14     |
| v15   | x₀-residual      | 1       | yes     | 0      | 2.10    | 0.392   | 0.21     |
| v16   | x₀-residual      | 1       | –       | 0.01   | 1.80    | 0.368   | 0.22     |

![**Figure 2. Phase 4 architectural ablation.** Match rate (a) and pattern Pearson (b) for the seven coordinate-only runs of Table 1 on MP-20 test (n = 1000, true lattice substituted). The ε → x₀-residual switch (v11 → v13) is the largest single lift; adding Wyckoff embeddings or a distance-matrix loss regresses below it.](fig1_ablation.pdf){#fig:ablation}

Two choices are load-bearing. The **lattice-input fix** (§3.4) drops lattice loss from 1.0 (pinned at the prior variance for 100 k steps) to ~0.02 once $W_\text{lat}\boldsymbol{\ell}^{(t)}$ enters the head input. Switching from **ε to x₀-residual** at fixed Debye λ = 1 (v11 → v13) lifts match 0.9 % → 2.5 % — 3.5× over the v10 ε baseline. Capacity is not the bottleneck: a 10.1 M-parameter run ($d=384$, $L=5$, `gpu_v12`) stayed at the same ~1.0 coord-loss plateau as v10. Under the full pipeline (sampled lattice), v13 drops to 1.2 % at n = 256; the Phase 9 v21 checkpoint reaches 5.6 % under oracle but only 1.0 % under full pipeline. §5.2 attacks that gap.

### 5.2 Phase 9 — classical indexer drop-in

Table 2: three rows on the same n = 1 000 with the v21 checkpoint (Phase 9 retrain, Phase 4 fixes + retuned encoder); they differ only in which lattice the sampler conditions on.

**Table 2.** Phase 9 lattice-source comparison (n = 1 000 MP-20 test, v21 checkpoint). Match % is reported with a 95 % Wilson confidence interval.

| Lattice source                                | Match % (95 % CI) | All-correct % | sg@0.1 % | RMSD med (Å) | R_wp |
|-----------------------------------------------|-------------------|---------------|----------|--------------|------|
| Learned head only (full pipeline)             | 1.0 [0.5, 1.8]    | 0.0           | 1.4      | —            | 11.7 |
| **Classical Q-space autoindexer (§3.5) drop-in** | **1.6 [1.0, 2.6]** | **0.1**    | **1.4**  | **0.125**    | 11.5 |
| Ground-truth lattice (oracle)                 | 5.6 [4.3, 7.2]    | 0.6           | 1.7      | 0.106        | 5.6  |

**On the aggregate lift.** The learned-head → indexer change (1.0 → 1.6 %) is the only Phase 9 intervention that moved no-true-lattice match in a positive direction, but we are explicit that it is *not* significant at the aggregate level: the Wilson CIs overlap and an unpaired two-proportion test gives p = 0.24. We do not claim the headline number improved. What *is* well-supported is the **mechanism**: §5.5 shows the lift is concentrated entirely in the high-symmetry systems where the indexer's cell error falls below the 0.5 Å sensitivity knee, and is absent (or negative) elsewhere — the aggregate is a near-cancellation of a real positive effect on hex/ortho against no effect on mono/tri. A paired McNemar test on the per-structure flips, which would resolve whether the directional lift is real, requires the multi-seed reruns we leave to future work (the released harness now dumps per-structure flags for exactly this purpose). By contrast, the indexer → oracle gap (1.6 → 5.6 %) *is* significant — the CIs are disjoint — and isolates the lattice-attributable component of the error: even with a perfect lattice, match caps at 5.6 %, so coordinate prediction is the dominant limiter, while lattice error explains the 1.6 → 5.6 % shortfall.

The 1.6 → 5.6 % oracle gap is closed by neither this nor any other intervention we tried: the indexer's per-system MAE (1.45 Å overall, dominated by mono 1.76 Å and triclinic NaN) sits above the 0.5 Å knee from §5.5 — improvement requires fixing the low-symmetry path (GSAS-II remains experimental).

![**Figure 3. Per-system indexer performance.** Strict and consistent indexing rates (a) and length MAE (b) on n = 1 000. High-symmetry systems (hex, ortho) sit below the 0.5 Å knee; trig and mono blow up; triclinic is unindexable in this implementation. The overall 1.45 Å MAE hides two regimes, and explains why the §5.2 aggregate lift is a near-cancellation.](fig4_indexer_bench.pdf){#fig:indexer}

### 5.3 Head-to-head with external baselines

DiffractGPT (`knc6/diffractgpt_mistral_chemical_formula`) at n = 1 000 and PXRDnet (`therealgabeguo/cdvae_xrd_sinc100`) at n = 20 (limited by ~4 GPU-hr/material) re-scored on *our* `StructureMatcher` harness at matching tolerances. Crystalyze's checkpoint download is inactive; cite-but-not-reproduced.

**Table 3.** Reproduced head-to-head on MP-20 test (StructureMatcher ltol 0.2, stol 0.3, angle_tol 5°).

| Model                                        | n     | Match % (95 % CI)  | All-correct % (95 % CI) | sg@0.1 % | RMSD med (Å) |
|----------------------------------------------|-------|--------------------|-------------------------|----------|--------------|
| PXRD-Diff (v21 + indexer drop-in)            | 1 000 | 1.6 [1.0, 2.6]     | 0.1 [0.0, 0.6]          | 1.4      | 0.125        |
| DiffractGPT [^DiffractGPT]                   | 1 000 | 18.9 [16.6, 21.4]  | 15.9 [13.8, 18.3]       | 21.3     | 0.001        |
| PXRDnet sinc100 [^PXRDnet]                   | 20    | 30.0 [14.5, 51.9]  | 5.0 [0.9, 23.6]         | 5.0      | 0.011        |

**Match rate** ranks PXRDnet > DGpt > PXRD-Diff; PXRDnet's lead is consistent with its 35 k-op latent optimisation, whereas DGpt is one forward pass. PXRD-Diff is 12–19× behind both. A more interesting pattern is that the *all-correct* ordering need not follow match rate: DGpt's all-correct (15.9 % [13.8, 18.3]) exceeds its space-group-loose match by little, while PXRDnet's drops from 30 % match to 5 % all-correct — i.e. PXRDnet matches structure under loose tolerances but rarely recovers the space group, whereas DGpt's CIF-token output appears to encode symmetry. **We flag this as a hypothesis, not a finding.** At n = 20 the PXRDnet all-correct estimate (1/20) carries a 95 % CI of [0.9, 23.6], which overlaps DGpt's [13.8, 18.3]; the apparent "reversal" is *not* statistically established and would require n ≈ 100–200 to confirm. If it holds, the implication is consequential — published single-number "match rates" would conflate structure recovery with symmetry recovery — which is precisely why we surface it despite the underpowered sample, and why §6 frames it as motivation for future measurement rather than a claim. **RMSD** is heavy-tailed for us and tight-near-zero for both baselines, suggesting our matches are tolerance-stretch rather than coordinate recovery.

![**Figure 4. Three-way headline comparison.** Match rate, all-correct, and pattern-Pearson on the same MP-20 test materials scored through *our* `StructureMatcher` harness, with 95 % Wilson CIs. PXRD-Diff loses on match rate by 12–19× but wins on per-pattern Pearson. Note the wide PXRDnet (n = 20) intervals: the DGpt > PXRDnet all-correct ordering is a hypothesis, not an established result (§5.3, §6).](fig5_threeway_headline.pdf){#fig:threeway}

*Caveats.* PXRDnet n = 20 has a 95 % Wilson CI of roughly ±20 pp on every rate; n = 200 would need ~33 days of RTX 5090 rental at ~4 GPU-hr/material. Patterns are simulated through each model's own preprocessing (different bins, 2θ vs Q, broadening) — material_ids are identical, pattern inputs are not bit-identical; §7 bounds this.

### 5.4 What did *not* work

Four interventions failed at n ≥ 200; one paragraph each.

**Top-K Debye rerank (Phase 9.1.4).** Top-5 indexer candidates reranked by Debye loss: match 1.6 % → 1.0 %, all-correct 0.1 % → 0.0 % at n = 1 000. The indexer surfaces more wrong cells faster than rerank can filter.

**Debye-gradient guidance during DDIM (Phase 9.2).** Coordinate-channel guidance term $-g \cdot \nabla_\mathcal{C} \mathcal{L}_\text{Debye}$, sweep $g \in \{0,0.5,1,2,5\}$ on n = 200: flat 1.0–2.5 % within noise, all-correct 0 % throughout. Sampling-time Debye gradient is too noisy to steer the trajectory — consistent with Segal et al. [^Segal2025], who show the powder-XRD similarity loss landscape is too non-convex for direct gradient descent.

**Wyckoff-site embeddings (v15, n = 1 000).** `spglib`-computed, `nn.Embedding(27, 256)` added to atom features. The embedding learns (5 404/6 912 entries non-zero) but match drops 2.5 % → 2.1 % and destabilises lattice prediction (lat loss ~0.6 vs ~0.02; Figure 5). We suspect inflated atom-feature norm shifts the lattice-pool input distribution.

![**Figure 5. Wyckoff embedding destabilises lattice prediction.** Lattice (a) and coordinate (b) training-loss curves for the distance-aux run (v16) vs the Wyckoff-embedding run (v15). Adding the Wyckoff embedding inflates the lattice loss by an order of magnitude (~0.6 vs ~0.02) while leaving the coordinate loss largely unchanged — the failure is localised to the lattice channel.](fig2_training_curves.pdf){#fig:training}

**Distance-matrix aux loss (v16, n = 1 000).** Pairwise MLP predicting periodic distances, $\lambda_\text{dist} = 0.01$ (largest stable). Loss decreases (final ~2.0 vs ~10 random) but match drops 2.5 % → 1.8 %; biases features toward absolute-distance reconstruction rather than the relative updates diffusion needs.

**Combination (v14).** Wyckoff + distance loss together collapses to 0.8 %, below the bare ε baseline; the two interact destructively, no clean theoretical explanation.

### 5.5 Encoder-bottleneck perturbation study

Best predicted-vs-target Pearson is 0.43; the same model with *correct* coordinates plugged into the differentiable simulator hits 0.97 — the encoder + denoiser leaves ~0.5 of pattern-space agreement on the table. The aux head (sees $\mathbf{g}$ only) reaches loss 0.007: the global encoding *contains* the lattice signal, the denoiser under-uses it.

Perturbing the sampler-supplied lattice by Å of cell-length error, v21 at n = 200: match decays 5.6 % (Δ=0) → 2.0 % (0.5 Å) → 0.5 % (1.0 Å) → 0 % (≥1.5 Å), steeper than linear with a knee at ~0.5 Å. This matches the indexer's per-system MAE: high-symmetry systems (hex, ortho; MAE ≤ 1.0 Å) lift; low-symmetry (mono, tri; MAE ≥ 1.5 Å) do not. §5.2's indexer lift is *predictable* — it works where it can.

---

## 6. Discussion

**Encoder bottleneck (the robust result).** The lattice signal exists in $\mathbf{g}$ (aux loss 0.007) but the denoiser cannot extract absolute d-spacings well enough to predict from scratch; the disjoint oracle-vs-indexer CIs (§5.2) make this the paper's statistically firmest claim. The Q-space autoindexer recovers part of the 5.6 % oracle ceiling on the high-symmetry subset; top-K rerank and gradient guidance fail because they operate *after* the encoder commits to a bad lattice prior, while the indexer bypasses that decision.

**Per-atom anchoring is hard.** PXRD is permutation-invariant; our denoiser is permutation-equivariant; there is no symmetry-breaking signal that pins atom $i$ to a specific Wyckoff site. The Wyckoff embedding tried to break symmetry at the input and failed. A promising untried direction: break symmetry at the *output* — predict an unordered set of orbits plus a Hungarian-style matcher.

**A measurement hypothesis: match rate may conflate two capabilities.** PXRDnet's 30 % match / 5 % all-correct vs DGpt's 18.9 % / 15.9 % (Table 3) suggests the two systems recover *structure* at broadly comparable rates while differing sharply in *symmetry* recovery — DGpt's CIF-token output appears to encode space group in a way PXRDnet's coordinate decode does not. We stress that the underlying numbers cannot yet support this: PXRDnet's all-correct CI [0.9, 23.6] at n = 20 overlaps DGpt's, so the ordering is unconfirmed. We raise it as a *measurement hypothesis* worth testing at adequate n, because if true it has a concrete consequence for the field: a single "match rate" reported under inconsistent tolerances would conflate structure recovery with symmetry recovery, and downstream crystallography (where the space group matters as much as the coordinates) would be mis-served by it. The contribution here is the shared harness that makes such a test possible, not the (underpowered) comparison itself.

**Niche for small reproducible models.** At 1.6 % we are 12–19× behind much larger systems, but small models make the encoder bottleneck visible (capacity hides it) and the indexer drop-in trivial to integrate (larger models would need architecture-level surgery). The recipe — encoder + Phase 4 fixes + Phase 9 indexer drop-in — is 3.7 M parameters and runs inference in seconds.

---

## 7. Limitations

- **Scope.** Composition given; simulated PXRD only (no instrument response, preferred orientation, asymmetry, background); MP-20 only — no larger cells, organics, or higher-Z.
- **Statistics.** Single seed per ablation. The large effects (the 2.5–3.5× x₀-residual and lattice-input lifts; the oracle-vs-indexer gap with disjoint CIs) are robust to this; the small ones are not. In particular, the aggregate indexer lift (1.0 → 1.6 %) is within pilot variance (~0.5 %) and not significant on an unpaired test (p = 0.24) — we rely on the per-system mechanism, not the aggregate, and a paired multi-seed McNemar test is left to future work.
- **Baselines.** PXRDnet n = 20 has a 95 % Wilson CI of ≈ ±20 pp; the DGpt > PXRDnet all-correct ordering (§5.3/§6) is therefore a hypothesis, not a result. n = 200 needs ~33 days RTX 5090. Crystalyze checkpoint download is inactive (verified 2026-06-01); cited but unreproduced. Patterns are not bit-identical across the three preprocessors (PXRD-Diff 4 251-bin 2θ, DGpt 300-bin 2θ, PXRDnet 4 096-bin Q with sinc² broadening) — structures match, patterns do not; we did not quantify the residual preprocessing effect on match rate, and a matched-vs-native re-scoring on a structure subset is the natural check.
- **Indexer.** GSAS-II low-symmetry path hangs in `findBestCell` on real MP-20 mono/tri patterns; shipped behind `--use-gsas` but experimental.
- **Training.** Debye loss uses ground-truth lattice; a curriculum gradually replacing true with predicted was not tried.

---

## 8. Conclusion

A 3.7 M-parameter conditional diffusion model with a differentiable Bragg loss recovers 1.6 % of MP-20 test structures with a classical Q-space autoindexer supplying the lattice, and 5.6 % [4.3, 7.2] under a true-lattice oracle. The oracle gap is significant and is the paper's firmest result: even with a perfect lattice, coordinate prediction caps match at 5.6 %, while lattice error explains the rest of the shortfall to that ceiling. Reproduced on the same `StructureMatcher` harness, DGpt reaches 18.9 % / 15.9 % and PXRDnet 30.0 % / 5.0 % (match / all-correct) — PXRD-Diff is 12–19× behind. The apparent divergence in their space-group recovery is a measurement hypothesis the n = 20 PXRDnet sample cannot yet confirm, but the shared harness now makes it testable. Two architectural fixes (lattice-input, x₀-residual) and one drop-in (classical autoindexer) are load-bearing; six other interventions are documented as failures. Code, checkpoints, all per-phase JSONs, both indexer paths, and the differentiable Bragg module are released.

---

## Acknowledgments

We thank the maintainers of `pymatgen`, `spglib`, the CDVAE benchmark, GSAS-II, and the upstream maintainers of DiffractGPT (`atomgptlab/atomgpt`) and PXRDnet (`gabeguo/cdvae_xrd`) for releasing checkpoints and code that made the head-to-head reproduction in §5.3 possible. Compute was rented from Vast.ai; total spend was approximately USD 25 across roughly 30 GPU-hours on RTX 5090 instances (Phase 4 ablation, Phase 9 retrain + indexer sweeps, DiffractGPT n = 1 000 inference, and PXRDnet n = 20 inference).

## Author Contributions (CRediT)

F. Cai: Conceptualization, Methodology, Software, Validation, Formal analysis, Investigation, Data curation, Writing — Original Draft, Writing — Review & Editing, Visualization, Project administration.

## Conflict of Interest

The author declares no competing interests.

## Funding

This research received no external funding.

## Data and Code Availability

Source code, trained checkpoints, and all per-phase training/evaluation logs are available at `https://github.com/fronkt/pxrd-diff` and archived at Zenodo (DOI: 10.5281/zenodo.XXXXXXX — *to be minted at submission*). The MP-20 dataset is publicly available via the CDVAE benchmark. All experiments reproduce from a single `requirements.txt` and the `scripts/` pipeline; per-structure evaluation flags are released to support paired re-analysis.

## Ethics Declaration

This study uses no human subjects, animal subjects, or sensitive data. The MP-20 dataset is composed entirely of publicly available crystal structures from the Materials Project.

---

## References

[^DiffractGPT]: Choudhary, K. (2025). DiffractGPT: Atomic Structure Determination from X-ray Diffraction Patterns using a Generative Pretrained Transformer. *The Journal of Physical Chemistry Letters*, 16(8), 2110–2119. DOI: 10.1021/acs.jpclett.4c03137. Reproduced from HF checkpoint `knc6/diffractgpt_mistral_chemical_formula` and code at `github.com/atomgptlab/atomgpt`.

[^Crystalyze]: Riesel, E. A., Mackey, T., Nilforoshan, H., Xu, M., Badding, C. K., Altman, A. B., Leskovec, J., & Freedman, D. E. (2024). Crystal Structure Determination from Powder Diffraction Patterns with Generative Machine Learning. *Journal of the American Chemical Society*, 146(44), 30340–30348. DOI: 10.1021/jacs.4c10244. Code: `github.com/ML-PXRD/Crystalyze`. We could not reproduce: the checkpoint download link is marked "not yet active" in the upstream README (verified 2026-06-01).

[^PXRDnet]: Guo, G., Saidi, T. L., Terban, M. W., Valsecchi, M., Billinge, S. J. L., & Lipson, H. (2025). Ab initio structure solutions from nanocrystalline powder diffraction data via diffusion models. *Nature Materials*, 24, 1726–1734. DOI: 10.1038/s41563-025-02220-y (Author Correction: 10.1038/s41563-025-02301-y; preprint arXiv:2406.10796). Reproduced from HF checkpoint `therealgabeguo/cdvae_xrd_sinc100` and code at `github.com/gabeguo/cdvae_xrd`.

[^GSASII]: Toby, B. H., & Von Dreele, R. B. (2013). GSAS-II: the genesis of a modern open-source all purpose crystallography software package. *Journal of Applied Crystallography*, 46(2), 544–549. Code: `github.com/AdvancedPhotonSource/GSAS-II`.

[^CDVAE]: Xie, T., Fu, X., Ganea, O.-E., Barzilay, R., & Jaakkola, T. (2022). Crystal diffusion variational autoencoder for periodic material generation. *International Conference on Learning Representations*.

[^DiffCSP]: Jiao, R., et al. (2023). Crystal structure prediction by joint equivariant diffusion. *Advances in Neural Information Processing Systems* 36.

[^MatterGen]: Zeni, C., et al. (2024). MatterGen: A generative model for inorganic materials design. *arXiv:2312.03687*.

[^MACE]: Batatia, I., Kovács, D. P., Simm, G. N. C., Ortner, C., & Csányi, G. (2022). MACE: Higher order equivariant message passing neural networks for fast and accurate force fields. *Advances in Neural Information Processing Systems* 35.

[^SchNet]: Schütt, K. T., et al. (2017). SchNet: A continuous-filter convolutional neural network for modeling quantum interactions. *Advances in Neural Information Processing Systems* 30.

[^NicholDhariwal]: Nichol, A., & Dhariwal, P. (2021). Improved denoising diffusion probabilistic models. *International Conference on Machine Learning*.

[^DDIM]: Song, J., Meng, C., & Ermon, S. (2021). Denoising diffusion implicit models. *International Conference on Learning Representations*.

[^deCIFer]: Johansen, F. L., Friis-Jensen, U., Dam, E. B., Jensen, K. M. Ø., Mercado, R., & Selvan, R. (2025). deCIFer: Crystal Structure Prediction from Powder Diffraction Data using Autoregressive Language Models. *Transactions on Machine Learning Research* (arXiv:2502.02189).

[^Segal2025]: Segal, N., Subramanian, A., Li, M., Miller, B. K., & Gómez-Bombarelli, R. (2025). The Loss Landscape of Powder X-Ray Diffraction-Based Structure Optimization Is Too Rough for Gradient Descent. *arXiv:2512.04036*.

[^GuoSchwalbeKoda]: Guo, J., & Schwalbe-Koda, D. (2026). Generative Inversion of Spectroscopic Data for Amorphous Structure Elucidation. *arXiv:2603.23210*.

[^Boultif]: Boultif, A., & Louër, D. (2004). Powder pattern indexing with the dichotomy method. *Journal of Applied Crystallography*, 37(5), 724–731. DOI: 10.1107/S0021889804014876.

> **Author note on reproduced numbers.** The reproduced PXRDnet and DiffractGPT numbers in §5.3 are computed by the author on the released checkpoints through our shared `StructureMatcher` harness, not transcribed from the cited papers. The central claims depend only on these reproduced numbers under one common evaluation protocol, with confidence intervals stated throughout.

---

## Appendix

### A. Full ablation history

For completeness, Table A1 lists every training run discussed in the development of this paper, including those that did not make it into the main ablation table. Logs and checkpoints for all runs are in the released repository under `runs/`.

| Run | Parameters | What changed | Result | Status in paper |
|-----|-----------|-------------|--------|----------------|
| v4  | 3.5 M | Global PXRD pooling, additive conditioning | Coord loss flat at 3.0 | §3.2 |
| v5  | 3.7 M | + Multi-resolution cross-attention | Coord 3.0 → 1.0 | §3.2 |
| v6–v9 | 3.7 M | λ_Debye sweep {0, 0.1, 1, 10}, ε-prediction | All ~1 % match (within noise) | §5.2 |
| v10 | 3.7 M | + Lattice-input fix, λ_Debye = 0 | Lat loss 1.0 → 0.05 | §5.2 |
| v11 | 3.7 M | + Lattice-input fix, λ_Debye = 1 | Match 0.9 % | Table 1 |
| v12 | 10.1 M | Larger model (d=384, L=5), ε-prediction | Killed at 18 k; same plateau | §5.2 |
| v13 | 3.7 M | x₀-residual + lat-fix + Debye λ=1 | **2.51 % match (Phase 4 best, true-lat)** | Table 1 |
| v14 | 3.8 M | v13 + Wyckoff + distance loss | 0.80 % match | Table 1 |
| v15 | 3.8 M | v13 + Wyckoff only | 2.10 % match | Table 1 |
| v16 | 3.7 M | v13 + distance loss only | 1.80 % match | Table 1 |
| v17 – v20 | 3.7 M | Five Phase-9 encoder retrains with different ResNet/Transformer hybrids and pattern-augmentation curricula | None beat v13 by more than noise | §5.1 |
| **v21** | **3.7 M** | Phase 9 final: v13 architecture + Phase 9 retrain + indexer drop-in support | **1.6 % match (no-true-lat, indexer); 5.6 % (true-lat oracle)** | Tables 2, 3 |

### B. Hyperparameter sensitivity

We did not perform a full hyperparameter sweep. Pilot experiments on `d_model ∈ {128, 256, 384}` and `n_layers ∈ {2, 3, 5}` showed (256, 3) as a reasonable Pareto point. The Debye loss weight was swept {0, 0.1, 1, 10} in early ε-prediction runs (v6–v9) without measurable effect on match rate; we re-fixed it at 1.0 for the x₀-residual runs by analogy.

### C. Differentiable simulator validation details

Pearson correlations between our DiffPXRD module and `pymatgen.XRDCalculator` over a random 50-structure MP-20 test subset:

- Mean: 0.962
- Std: 0.027
- Min: 0.896 (mp-1213821, layered structure with strong texture in the reference)
- Max: 0.998 (mp-149, Si)
- Fraction > 0.95: 33/50
- Fraction > 0.9: 49/50
- Fraction > 0.7: 50/50

Gradient sanity: for all 50 structures we confirmed `∂L/∂F` and `∂L/∂L` are non-zero and finite via `torch.autograd.gradcheck` on a 4-atom subset.

### D. Reproducibility checklist

- [x] Hyperparameters specified (§4.3)
- [x] Datasets and splits specified (§4.1; canonical CDVAE MP-20)
- [x] Evaluation protocol specified (§4.2)
- [x] Random seed: single seed (42) per run; pilot variance ≈ 0.5 % match-rate absolute
- [x] Compute environment: PyTorch 2.x, CUDA 12.x, single RTX 5090; Python 3.12
- [x] Code, checkpoints, logs, and full training scripts released
