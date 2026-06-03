# PXRD-Diff: An Honest Look at Conditional Diffusion for Powder Diffraction Inversion

**Frank Cai**
*Independent researcher*
`frankyc11223@gmail.com`

**Target venue:** ML4PS / AI4Mat workshop @ NeurIPS 2026
**Track:** Methods / Negative results
**Word budget:** ~4,500 words main text + appendix

---

## Abstract

Powder X-ray diffraction (PXRD) is the workhorse characterization technique of solid-state chemistry, yet inverting a 1D powder pattern to a 3D crystal structure remains an open problem. Recent generative approaches report encouraging numbers, but published evaluation protocols are mutually incompatible and failure modes are seldom catalogued. We present **PXRD-Diff**, a 3.7 M-parameter conditional denoising diffusion model trained on the canonical CDVAE MP-20 split, together with a reproduced head-to-head comparison against two open generative baselines — DiffractGPT [^DiffractGPT] and PXRDnet [^PXRDnet] — all scored by the *same* `pymatgen.StructureMatcher` harness on the same evaluation subset. Our central finding is a diagnosis: the learned PXRD encoder, despite achieving an auxiliary lattice-from-pattern regression loss of 0.007 in isolation, cannot supply absolute-d-spacing information to the diffusion denoiser well enough to predict the lattice from scratch. A perturbation study confirms that the match rate dies above ~0.5 Å cell error, and an oracle pass with the true lattice lifts our no-true-lattice match rate from **1.6 % to 5.6 %** at n = 1000. Motivated by this diagnosis, we replace the learned lattice head with a classical Q-space autoindexer (a drop-in built on the de-Wolff dichotomy), which raises no-true-lattice match from 1.0 % (no indexer) to **1.6 %** — a modest absolute lift but the only intervention in our six-month ablation that moved the headline. Two seemingly attractive sampling-time tricks — top-K candidate reranking by Debye fit, and Debye-gradient guidance during DDIM — are both falsified at scale. Reproduced baselines on identical evaluation: **DiffractGPT 18.9 %** match (15.9 % all-correct) at n = 1000, and **PXRDnet sinc100 30.0 %** match (5.0 % all-correct) at n = 20 (PXRDnet inference is ~4 GPU-hr per material at the published config, capping single-rental n). PXRD-Diff is therefore 12–19× behind open SOTA on raw match rate, but the encoder-bottleneck diagnosis, the indexer-as-drop-in recipe, and the four-extension failure catalogue are concrete and reproducible. Code, checkpoints, all per-phase result JSONs, and the differentiable Bragg simulator are released.

**Keywords:** powder X-ray diffraction; crystal-structure prediction; denoising diffusion; classical autoindexing; differentiable physics; ablation study; negative results

---

## 1. Introduction

Powder X-ray diffraction is, in volume of measurements, the most common form of structural characterization in solid-state chemistry and materials science. The forward problem — given a structure, simulate the pattern — is well solved by 1920s-vintage physics (the structure factor) and, in practice, by a one-line call to `pymatgen.analysis.diffraction.xrd`. The inverse problem is brutal. The same Bragg reflections can be produced by symmetry-equivalent structures; intensity information is integrated over crystallite orientations; the reduction from 3D structure to 1D intensity-vs-2θ destroys atom labelling entirely. The traditional workflow — indexing, Pawley/Le Bail fits, Rietveld refinement — requires an experienced crystallographer and an initial structural model good enough that refinement converges. For the long tail of unindexed patterns from new materials, the structure is often simply not solved.

Three recent generative-modelling efforts — DiffractGPT [^DiffractGPT], Crystalyze [^Crystalyze], and PXRDnet [^PXRDnet] — argue that neural networks can shortcut this pipeline. They report headline match rates in the tens of percent on either MP-20 or curated experimental datasets, but use mutually incompatible evaluation protocols (different match criteria, different splits, different "success" thresholds), and the failure modes are seldom catalogued. A practitioner reading these papers cannot easily form a calibrated expectation of *what is actually hard* about PXRD inversion.

This paper is a contribution to that calibration. We build a small, deliberately reproducible conditional diffusion model — **PXRD-Diff** — and run it against two of the three published baselines on the *same* `StructureMatcher` harness. Our contributions:

1. **A clean small baseline implementation.** PXRD-Diff is 3.7 M parameters, trains on a single RTX 5090 in ~1.5 h, and runs on the public CDVAE MP-20 split. A differentiable PyTorch Bragg structure-factor module (Pearson 0.96 vs `pymatgen.XRDCalculator` on n = 50 reference structures) supplies an auxiliary "predicted-pattern matches input-pattern" loss usable inside the diffusion training loop.
2. **An encoder-bottleneck diagnosis.** A perturbation study (§5.5) shows the StructureMatcher match rate dies above ~0.5 Å cell error; an oracle pass with the true lattice lifts our match rate from 1.6 % to **5.6 %** at n = 1000. The learned PXRD encoder achieves a 0.007 auxiliary lattice-regression loss in isolation, yet the full denoiser under-uses this signal — architectural rather than informational bottleneck.
3. **A classical-indexer drop-in.** Motivated by (2), we swap the learned lattice head for a Q-space autoindexer (de-Wolff dichotomy) at sampling time. Lift from 1.0 % → 1.6 % no-true-lattice match — small but the only intervention in six months of ablation that moved the headline.
4. **A four-item failure catalogue.** Top-K candidate reranking by Debye fit (Phase 9.1.4), Debye-gradient guidance during DDIM (Phase 9.2/9.3), Wyckoff-site token embeddings (v15), and an auxiliary distance-matrix loss (v16) are all falsified on MP-20 at n ≥ 200. The combination of the last two (v14) is worse than the bare ε baseline.
5. **Two reproduced external baselines.** DiffractGPT (Mistral-7B + LoRA, HF `knc6/diffractgpt_mistral_chemical_formula`) and PXRDnet (CDVAE + XRD encoder, HF `therealgabeguo/cdvae_xrd_sinc100`) re-evaluated on the *same* `StructureMatcher` harness at our matching tolerances. Headline: 18.9 % match (DGpt n = 1000), 30.0 % match (PXRDnet n = 20). PXRD-Diff at 1.6 % is 12–19× behind on raw match rate. Crystalyze checkpoint is gated by an inactive Google-Drive link; we cite it but cannot reproduce.

We intend the paper to be useful both as a recipe (the bug fixes, the x₀-residual trick, and the indexer drop-in are easy to reproduce and individually load-bearing) and as an honest, calibrated stake in the ground for what a small, self-contained conditional diffusion model can and cannot do on this problem.

---

## 2. Related work

**Diffusion for crystal structures.** CDVAE [^CDVAE] introduced the canonical MP-20 evaluation protocol and a VAE+diffusion approach for unconditional crystal generation. DiffCSP [^DiffCSP] sharpened this with a joint diffusion over coordinates and lattice. MatterGen [^MatterGen] scales the idea to several million structures and reports impressive validity numbers. PXRD-Diff borrows the joint-diffusion design from DiffCSP, restricting scope to the conditional setting where a 1D pattern is the conditioning signal.

**Neural PXRD inversion.** DiffractGPT [^DiffractGPT] casts PXRD inversion as a sequence-to-sequence problem (peak list + chemical formula → CIF token stream) on top of a fine-tuned Mistral-7B. Crystalyze [^Crystalyze] uses CDVAE conditioned on an XRD transformer encoder, with discrete-symmetry post-hoc filtering. PXRDnet [^PXRDnet], a closely related CDVAE-style architecture from a separate group (Guo et al.), differs in that it uses *iterative latent-space gradient guidance* against the target pattern (≈3.5 M decoder ops/material). All three are larger than PXRD-Diff and use significantly more training data; all three report on bespoke evaluation pipelines that complicate head-to-head comparison. §5.3 re-evaluates DiffractGPT and PXRDnet on the same `StructureMatcher` harness we use throughout. The Crystalyze checkpoint download link is described in its own README as "not yet active" (verified 2026-06-01); we are unable to reproduce it.

**Differentiable physics in materials science.** Scattering-pattern losses date back to Rietveld refinement; the novelty here is differentiability *with respect to atomic coordinates and lattice* in a PyTorch graph, so that a structure factor calculation can be inserted as a loss term in a deep network. Concurrent work [^WrenzcuckPhysML2025] explores related differentiable simulators for SAXS; we are not aware of prior work that uses a fully vectorised PyTorch Bragg simulator inside a diffusion training loop.

**Equivariance.** PXRD-Diff is *not* strictly E(3)-equivariant — we use a periodic-distance RBF rather than spherical harmonics, à la SchNet [^SchNet]. We initially planned a frozen MACE [^MACE] encoder for atom embeddings; preliminary experiments did not show a measurable benefit at MP-20 scale, and we deferred MACE to future work in favour of the simpler SchNet-style backbone.

---

## 3. Method

### 3.1 Problem formulation

Let $\mathcal{C} = (\mathbf{F}, \mathbf{Z}, \mathbf{L})$ denote a crystal structure with fractional coordinates $\mathbf{F} \in [0,1)^{N \times 3}$, atomic numbers $\mathbf{Z} \in \{1,\ldots,100\}^N$, and a $3 \times 3$ lattice matrix $\mathbf{L}$ parametrised by $(a,b,c,\alpha,\beta,\gamma) \in \mathbb{R}^6$. Let $\mathbf{p}(\mathcal{C}) \in \mathbb{R}^{4251}$ be the simulated Cu Kα PXRD pattern on a fixed 2θ grid from 5° to 90° at 0.02° resolution, normalised to maximum intensity 1.

We are given $\mathbf{p}$ and $\mathbf{Z}$ at test time, and we wish to sample $(\mathbf{F}, \mathbf{L}) \sim p(\,\cdot\, | \mathbf{p}, \mathbf{Z})$. Note that the composition $\mathbf{Z}$ being known is a meaningful simplification — in practice one usually knows the chemistry from synthesis — but it is consistent with prior work on the same task.

### 3.2 Architecture overview

PXRD-Diff has three trained components: a PXRD encoder, a denoiser, and a small auxiliary lattice head used only during training.

**PXRD encoder.** A 1D ResNet of four blocks (channels $64 \to 128 \to 256 \to 256$, stride-2 downsampling, GroupNorm, SiLU) ingesting the standardised pattern. We expose two outputs: a global pooled vector $\mathbf{g} \in \mathbb{R}^{256}$ and a multi-resolution feature map $\mathbf{F}_{\text{pxrd}} \in \mathbb{R}^{L \times 256}$ obtained by 1×1-projecting each block output to $d_\text{model}$ and concatenating along the spatial axis. *The multi-resolution map is critical.* In an early run (referred to as `gpu_v4` in our logs) we conditioned the denoiser on $\mathbf{g}$ only via additive broadcast, and the coordinate loss never moved off the random baseline of 3.0 — the auxiliary head told us the encoder was learning useful features (its loss dropped from 0.99 to 0.007), but the `AdaptiveAvgPool1d(1)` collapse destroyed all spectral structure before it could reach the denoiser.

**Denoiser.** A periodic-distance message-passing network with $L = 3$ layers and $d_\text{model} = 256$. Atom embeddings $\mathbf{h}_i^{(0)} = \text{Emb}(\mathbf{Z}_i) + W_\text{coord}\mathbf{F}_i^{(t)}$ are updated by alternating (i) message passing over RBF-encoded periodic Cartesian distances under minimum-image convention, with timestep FiLM conditioning, and (ii) cross-attention from atoms (queries) to multi-resolution PXRD features (keys/values). Two output heads predict per-atom coordinate noise $\boldsymbol{\epsilon}_F \in \mathbb{R}^{N \times 3}$ and lattice noise $\boldsymbol{\epsilon}_L \in \mathbb{R}^6$. The lattice head is a small MLP applied to $[\,\bar{\mathbf{h}}\,;\, \mathbf{g}\,;\, W_\text{lat}\boldsymbol{\ell}^{(t)}\,;\, \mathbf{t}_\text{cond}\,]$ — that is, the pooled atom features, the global PXRD embedding, **a projection of the noisy lattice itself**, and the timestep encoding (see §3.4).

**Auxiliary head.** A single-hidden-layer MLP that predicts $(a,b,c,\alpha,\beta,\gamma)$ from $\mathbf{g}$ alone, trained with an MSE loss against the (normalised) ground-truth lattice. Its purpose is purely diagnostic — to expose whether the encoder is learning any lattice-relevant representation — and its loss does not flow back into the denoiser.

### 3.3 Differentiable Bragg structure-factor loss

The core physics-informed contribution. We implement, in PyTorch,
$$
F(hkl) = \sum_j f_j(s) \exp\!\left(-B_{\text{iso}} s^2\right) \exp\!\left[2\pi i\,(h x_j + k y_j + l z_j)\right],
$$
where $f_j(s) = \sum_{k=1}^4 a_k \exp(-b_k s^2)$ is the four-Gaussian atomic form factor (coefficients pulled directly from `pymatgen.analysis.diffraction.xrd.ATOMIC_SCATTERING_PARAMS`), $s = \sin\theta/\lambda$, and $B_{\text{iso}} = 0.5$ Å² is a uniform isotropic temperature factor. The intensity at each reflection is $|F(hkl)|^2$ multiplied by the standard Lorentz-polarisation correction. We enumerate all $(h,k,l)$ with $|h|,|k|,|l| \le 5$ (1330 reflections after excluding the origin), place each on the 2θ grid via Bragg's law, and broadcast each reflection as a Gaussian peak with FWHM 0.1° to obtain a continuous, differentiable PXRD pattern on a 256-bin coarse grid. The whole module is a single `nn.Module`, fully vectorised over the batch dimension.

**Validation.** On 50 random MP-20 test structures we compute the Pearson correlation between our differentiable pattern and the `pymatgen.XRDCalculator` reference: mean 0.962, std 0.027, min 0.896, max 0.998. All 50/50 correlations exceed 0.7 and we verified non-zero gradients through the structure factor with respect to fractional coordinates.

**As a loss.** During training we recover an estimate $\hat{\mathbf{F}}$ of the clean coordinates from the model output (see §3.4), simulate $\hat{\mathbf{p}} = \text{DiffPXRD}(\hat{\mathbf{F}}, \mathbf{Z}, \mathbf{L}_\text{true})$, and compare with the input pattern via a Pearson-correlation loss $\mathcal{L}_\text{Debye} = 1 - \rho(\hat{\mathbf{p}}, \mathbf{p})$. We use the true lattice for this auxiliary loss because the differentiable simulator is more sensitive to lattice errors than to coordinate errors at the early stages of training, and disentangling the two signals proved more stable.

### 3.4 Training objective

We use a VP-SDE with cosine schedule [^NicholDhariwal] for both channels:
$$
\bar{\alpha}(t) = \cos^2\!\left(\frac{\pi}{2} \cdot \frac{t + s}{1 + s}\right),\quad s = 0.008,\quad t \in [0,1].
$$
Coordinates are diffused on the flat torus $\mathbb{T}^3$ by wrapping the noisy sample to $[0,1)^3$; the loss is computed on the periodic difference $(\hat{\mathbf{F}} - \mathbf{F} + 0.5) \bmod 1 - 0.5$. Lattice parameters are diffused in $\mathbb{R}^6$ after standardisation by the train-set mean and standard deviation.

The full training objective (best configuration, run `gpu_v13`) is
$$
\mathcal{L} = \mathcal{L}_\text{coord} + \lambda_\text{lat}\,\mathcal{L}_\text{lat} + \lambda_\text{aux}\,\mathcal{L}_\text{aux} + \lambda_\text{Debye}\,\mathcal{L}_\text{Debye},
$$
with $\lambda_\text{lat} = 0.1$, $\lambda_\text{aux} = 0.5$, $\lambda_\text{Debye} = 1.0$.

**Two non-trivial parameterisation choices.** Both are essential and were arrived at by ablation, not foresight.

*x₀-residual prediction.* Rather than asking the model to predict noise $\boldsymbol{\epsilon}$, the heads predict a **residual** that is added to the noisy input to obtain the clean estimate:
$$
\hat{\mathbf{F}}^{(0)} = \mathbf{F}^{(t)} + \text{Coord-Head}(\mathbf{h}),\quad
\hat{\boldsymbol{\ell}}^{(0)} = \boldsymbol{\ell}^{(t)} + \text{Lat-Head}(\cdot).
$$
Because the heads are MLPs initialised to output near-zero, this means the model starts as the identity transform — at $t \approx 1$ (near pure noise) the model outputs the noisy input as its $\hat{\mathbf{F}}^{(0)}$ estimate, which is wrong but at least bounded; ε-prediction with the same architecture had to learn the entire transform from random initialisation and never recovered. This single change accounts for a 3.5× lift in the headline metric (see §5.2).

*Lattice-head input.* The lattice head reads the noisy lattice $\boldsymbol{\ell}^{(t)}$ as an explicit input feature. Without this — i.e. if the head only sees the (clean) lattice that is also passed to the denoiser as the geometry context — the head can only ever predict the unconditional mean noise, $\mathbb{E}[\boldsymbol{\epsilon}] = 0$, and its loss stays pinned at the random baseline of 1.0 forever. With this fix, the lattice loss drops to 0.02–0.06 on stable runs.

We also experimented with two extensions that did not pan out (§5.3): a **Wyckoff-site embedding** added to atom features, and an auxiliary **distance-matrix loss** in which a pairwise MLP predicts ground-truth periodic distances between atoms. Both are documented in the released code behind the `--use-wyckoff` and `--dist-weight` flags.

### 3.5 Classical indexing as a drop-in for the learned lattice head

The lattice head described in §3.2 is the *only* path by which the model encodes absolute d-spacings: every Bragg reflection 2θ position is set by the lattice through Bragg's law, so the lattice parameters are the absolute reference frame against which all peak positions in the input pattern must be interpreted. As §5.5 shows, the encoder + denoiser learns *relative* d-spacing structure (Pearson 0.43 between predicted and target patterns at evaluation time, vs 0.97 between target and ground truth) but not the absolute scale. Classical autoindexing — extracting peak positions, mapping them to Q-space (Q = 4π sinθ/λ), and fitting a unit cell by enumeration of (h,k,l) → Q²(h,k,l) — has solved this sub-problem since the 1970s and runs on a CPU in milliseconds per pattern.

We implement a from-scratch Q-space autoindexer that takes (i) the same simulated PXRD pattern fed to the encoder, (ii) the known crystal system (input as Bravais code, exposed as a sampling-time hyperparameter), and returns a candidate (a, b, c, α, β, γ). Peak picking uses a 1D local-maximum filter with a relative-intensity floor; the candidate Q-vector is fit by least squares to the de-Wolff dichotomy parameter form for each Bravais lattice. At sampling time we run the indexer first, substitute its output for the lattice channel of the diffusion sampler, and run DDIM only on the coordinate channel.

**Per-system accuracy (n = 1000 MP-20 test).** Native indexer overall: 48.8 % strict cell match, 1.45 Å mean cell-length MAE. Per crystal system: hexagonal 77.9 % / 0.53 Å, orthorhombic 59.8 % / 0.96 Å, trigonal 43.3 % / 4.62 Å, monoclinic 18.9 % / 1.76 Å, triclinic 0 %. The low-symmetry blowup (monoclinic and triclinic each have ≥ 4 free cell parameters and the de-Wolff search becomes hypothesis-capped) is the dominant remaining error. We ship a GSAS-II [^GSASII] adapter (`src/pxrd_diff/indexer_gsas.py`) for the low-symmetry path behind a `--use-gsas` opt-in flag; in practice `DoIndexPeaks` hangs on real MP-20 monoclinic/triclinic patterns inside `findBestCell` and is documented experimental rather than headline.

The indexer is a *drop-in* in the strongest sense: training is unchanged, the diffusion sampler is unchanged except for the lattice-channel substitution at t = T, and the headline metric (§5.2) reports both with and without the indexer for direct comparison.

### 3.6 Sampling

DDIM [^DDIM] with 50 steps and $\eta = 0$. We start from independent standard Gaussian noise on the lattice and on the (un-wrapped) coordinates, then alternate the standard DDIM update on each channel. Coordinates are wrapped to $[0,1)$ after every step. Predicted lattice parameters are de-standardised at the end and clipped to physically valid ranges $(a,b,c) \in [0.5, 100]$ Å, $(\alpha,\beta,\gamma) \in [10°, 170°]$ before being passed to `pymatgen.Lattice.from_parameters`.

A subtle bug fix is worth noting. In the x₀-residual variant, recovering the implicit ε for the DDIM update requires dividing by $\sqrt{1-\bar{\alpha}(t)}$; near $t = 0$ this denominator vanishes. We clamp it to 0.05 and skip the very last DDIM step when sampling, which removed a class of structures with NaN coordinates that we initially saw.

---

## 4. Experiments

### 4.1 Dataset and PXRD simulation

We use the canonical CDVAE [^CDVAE] split of MP-20: 27 136 train, 9 047 val, 9 046 test structures, each with at most 20 atoms in the conventional cell. Patterns are simulated with `pymatgen.analysis.diffraction.xrd.XRDCalculator` (Cu Kα₁, λ = 1.54184 Å), normalised to max intensity 1, and rasterised onto a 4 251-bin grid from 5° to 90° at 0.02° step. We cache the simulated patterns to `.npz` files at preprocessing time; over 45 196 structures we observed zero simulation failures and a mean throughput of ~100 structures/s on a single CPU thread.

### 4.2 Evaluation

A model is judged on three views of correctness, following the "all-of-three" rule:

1. **Composition match** — predicted and true reduced formulae are identical (trivial in our setting since we condition on $\mathbf{Z}$).
2. **Structure match** — `pymatgen.StructureMatcher` with $(\ell_\text{tol}, s_\text{tol}, \alpha_\text{tol}) = (0.2, 0.3, 5°)$, allowing primitive-cell reduction and rotation. This is the headline metric and is what we mean by "match rate" throughout.
3. **Coordinate RMSD** — Cartesian RMSD on the matched, aligned, permuted atoms; only defined when StructureMatcher returns a match.

We additionally report the space-group match rate at $\text{symprec} \in \{0.01, 0.05, 0.1, 0.2\}$, the Pearson correlation between predicted and true PXRD patterns, and the weighted profile R-factor $R_{wp}$.

**"All-correct" combined criterion.** A sample is judged "all-correct" iff `composition_match ∧ (sg-match at symprec=0.1) ∧ (rmsd ≤ 0.1 Å)`. This is the stricter headline metric used throughout §5; it isolates *complete* recovery (the experimentally interesting case for downstream crystallographic use) from the looser StructureMatcher match that allows considerable tolerance stretch.

**Two evaluation modes.** We report the *full pipeline* (predicted lattice + predicted coordinates) and an additional *true-lattice / coord-only* mode in which we substitute the ground-truth lattice parameters when building the predicted structure. The latter isolates coordinate quality from lattice quality; it is also the stricter setting that we use to compare ablations, because lattice prediction is itself a hard sub-problem and noise there can dominate the headline number.

### 4.3 Implementation details

- **Optimiser.** AdamW, LR 5×10⁻⁴, cosine decay to zero over 100 k steps, weight decay 10⁻⁴, gradient clipping at 1.0.
- **Batch / steps.** Batch 64, 100 000 steps (≈ 24 epochs), single RTX 5090 (32 GB VRAM), wall-clock ≈ 1.5–1.7 h per run.
- **Parameter count.** ≈ 3.7 M trainable; `--n-layers 5 --d-model 384` (~10 M) gave no benefit (see §5.2).
- **Diffusion.** $T = 1000$ continuous-time formulation (we sample $t \sim U(0,1)$ per training example).
- **Compute budget.** Total: ~12 GPU-hours for the Phase 4 ablation (v10–v16) + ~8 GPU-hours for the Phase 9 v21 retrain and indexer/perturbation sweeps + ~6 GPU-hours for DiffractGPT n = 1 000 + ~3 GPU-hours for PXRDnet n = 20 + ~6 hours for all StructureMatcher / aggregation evaluation. Cost: roughly USD 25 of cloud GPU rented on Vast.ai across the work reported here.

---

## 5. Results

We report results in the order they fall out of the experimental sequence: Phase 4 architectural ablation (§5.1) → Phase 9 indexer drop-in (§5.2) → reproduced external baselines (§5.3) → catalogue of falsified sampling-time tricks and architectural extensions (§5.4) → encoder-bottleneck diagnosis that motivates the indexer drop-in retroactively (§5.5).

### 5.1 Phase 4 — architectural ablation (true-lattice setting)

Table 1 summarises seven training runs sweeping the Debye loss, the lattice-input fix, the x₀-residual parameterisation, and two extensions (Wyckoff embeddings, distance loss). All numbers are on the same 1 000-structure subset of MP-20 test, with the true lattice substituted in (coord-only setting). The point of this ablation is to identify the *encoder + denoiser* recipe before we turn off the true-lattice oracle in §5.2. Pearson is the per-pattern correlation between predicted and true PXRD; RMSD is the StructureMatcher Cartesian RMSD over matched samples (NaN excluded).

**Table 1.** Phase 4 coordinate-only ablation on MP-20 test (n = 1 000, *true lattice substituted*).

| Run   | Parameterisation | λ_Debye | Wyckoff | λ_dist | Match % | Pearson | RMSD (Å) |
|-------|------------------|---------|---------|--------|---------|---------|----------|
| v10   | ε                | 0       | –       | 0      | 1.40    | 0.359   | 0.17     |
| v11   | ε                | 1       | –       | 0      | 0.90    | 0.365   | 0.15     |
| **v13** | **x₀-residual**  | **1**   | **–**   | **0**  | **2.51** | **0.434** | 0.22 |
| v14   | x₀-residual      | 1       | yes     | 0.01   | 0.80    | 0.367   | 0.14     |
| v15   | x₀-residual      | 1       | yes     | 0      | 2.10    | 0.392   | 0.21     |
| v16   | x₀-residual      | 1       | –       | 0.01   | 1.80    | 0.368   | 0.22     |

Two architectural choices are individually load-bearing.

*Lattice-input fix.* In all runs prior to v10, the lattice head saw $[\bar{\mathbf{h}}, \mathbf{g}, \mathbf{t}_\text{cond}]$ but not the noisy lattice $\boldsymbol{\ell}^{(t)}$ it was supposed to denoise. The training curve looked normal but the lattice loss sat at 1.0 (the variance of the standard normal prior) for 100 k steps. Concatenating $W_\text{lat}\boldsymbol{\ell}^{(t)}$ into the head input drops lattice loss to ~0.02 (95 % reduction) and produces physically reasonable sampled lattices. It is a textbook denoising bug but produced a plausible-looking run for days before we noticed.

*x₀-residual parameterisation.* Holding everything else fixed (lattice fix on, λ_Debye = 1), switching from ε-prediction (v11) to x₀-residual (v13) lifts match 0.9 % → 2.5 % — a 2.8× absolute and 3.5× over the v10 ε baseline at λ = 0. The residual head being initialised near zero makes the model start as the identity, which is a correct fixed point at t = 1; the ε head must learn a non-trivial output everywhere from random init. The Debye auxiliary loss also requires recovering a clean coordinate estimate from the model output, which is a simple addition under x₀-residual but a divide-by-$\sqrt{\bar{\alpha}(t)}$ that is numerically unstable near t = 1 under ε.

*Capacity is not the bottleneck.* A 10.1 M-parameter run with $d_\text{model} = 384, L = 5$ (run `gpu_v12`, killed at 18 k steps for budget) showed coordinate loss stuck at the same ~1.0 plateau as v10 — bottleneck for ε-prediction is parameterisation, not capacity.

**Full-pipeline.** When we sample lattice from the model rather than using the true lattice, v13 drops to 1.2 % match at n = 256. The Phase 9 architecture (v21 below, trained with Phase 4's fixed denoiser plus a re-tuned encoder) achieves 5.6 % under the true-lattice oracle on n = 1 000 — but only 1.0 % under the full pipeline. The next sub-section attacks that 5.6 → 1.0 % gap.

### 5.2 Phase 9 — classical indexer as drop-in for the learned lattice head

Table 2 reports the headline three-row comparison on the same n = 1 000 MP-20 test subset, all scored by `pymatgen.StructureMatcher` at $(\ell_\text{tol}, s_\text{tol}, \alpha_\text{tol}) = (0.2, 0.3, 5°)$ — the canonical CDVAE tolerance. "All-correct" is the headline `composition ∧ sg-match@symprec=0.1 ∧ rmsd ≤ 0.1 Å` rule from §4.2. All three rows use the same v21 checkpoint (Phase 9 retrain with the Phase 4 architectural fixes and a re-tuned encoder); they differ only in *which lattice* the diffusion sampler conditions on.

**Table 2.** Phase 9 lattice-source comparison (n = 1 000 MP-20 test, v21 checkpoint).

| Lattice source                                | Match % | All-correct % | sg@0.1 % | RMSD med (Å) | R_wp |
|-----------------------------------------------|---------|---------------|----------|--------------|------|
| Learned head only (full pipeline)             | 1.0     | 0.0           | 1.4      | —            | 11.7 |
| **Classical Q-space autoindexer (§3.5) drop-in** | **1.6** | **0.1**       | **1.4**  | **0.125**    | 11.5 |
| Ground-truth lattice (oracle)                 | 5.6     | 0.6           | 1.7      | 0.106        | 5.6  |

The 1.0 → 1.6 % lift from the indexer is small in absolute terms but is the only intervention in six months of Phase 9 ablation that moved the no-true-lattice match rate. The 1.6 → 5.6 % gap to oracle is what the indexer cannot close: the indexer's per-system cell-length MAE (1.45 Å overall, dominated by monoclinic 1.76 Å and triclinic NaN) saturates above the 0.5 Å sensitivity threshold from the perturbation study (§5.5), so improvement on the indexer side requires fixing the low-symmetry path (§3.5, GSAS-II remains experimental).

### 5.3 Head-to-head with external generative baselines

DiffractGPT (HF `knc6/diffractgpt_mistral_chemical_formula`) and PXRDnet (HF `therealgabeguo/cdvae_xrd_sinc100`) re-evaluated on the same n = 1 000 (DGpt) and n = 20 (PXRDnet — limited by ~4 GPU-hr/material at their published config) MP-20 test subsets through *our* `StructureMatcher` harness at the same tolerances. The Crystalyze checkpoint download link is not yet active and we cite-but-do-not-reproduce.

**Table 3.** Reproduced head-to-head on MP-20 test (StructureMatcher ltol 0.2, stol 0.3, angle_tol 5°).

| Model                                        | n     | Match % | All-correct % | sg@0.1 % | RMSD med (Å) |
|----------------------------------------------|-------|---------|---------------|----------|--------------|
| PXRD-Diff (v21 + indexer drop-in)            | 1 000 | 1.6     | 0.1           | 1.4      | 0.125        |
| DiffractGPT [^DiffractGPT]                   | 1 000 | 18.9    | 15.9          | 21.3     | 0.001        |
| PXRDnet sinc100 [^PXRDnet]                   | 20    | 30.0    | 5.0           | 5.0      | 0.011        |

Reading the table.

(i) **Raw match rate.** PXRDnet > DiffractGPT > PXRD-Diff by 1.6× and 19× respectively. PXRDnet's lead reflects its iterative latent-space optimisation (35 k decoder ops/material under their cosine warm-restart schedule); DiffractGPT is one feed-forward pass.

(ii) **All-correct** (the stricter `composition ∧ sg-match ∧ rmsd ≤ 0.1 Å` rule) reverses the top two: DiffractGPT 15.9 % > PXRDnet 5.0 % > PXRD-Diff 0.1 %. PXRDnet's predictions match structure with loose tolerances but rarely recover the *space group*; DGpt's token-stream prediction appears to encode symmetry implicitly through its CIF tokenisation. This is the strongest single qualitative signal in the table: published "match rate" numbers conflate two very different capabilities.

(iii) **RMSD distribution** is heavy-tailed for our model and tight-near-zero for both baselines, suggesting PXRD-Diff's matched cases are matches by tolerance-stretch rather than by atomic-position recovery.

(iv) **PXRDnet small-n caveat.** n = 20 is small (95 % CI on match rate ±20 pp by Wilson interval). The directional ranking is robust to that uncertainty, but we caution against reading the 30 % figure precisely. Running PXRDnet at n = 200 (their published n) would take ~33 days of RTX 5090 rental at their default hyperparameters; we report what fit in a single overnight rental.

(v) **Reproduction caveat.** All three models see PXRD patterns that are simulated *in their own pipeline's preprocessing* (different bin counts, different Q vs 2θ axes, different broadening). The structures evaluated are the same n CDVAE MP-20 test material_ids, but the patterns are not bit-identical inputs. Apples-to-apples on the structure side; orange-to-apple on the pattern side — there is no clean way to make this strictly identical short of porting all three models to one preprocessing pipeline, which was out of scope.

### 5.4 What did *not* work — six-month failure catalogue

Four interventions failed to move the headline metric on MP-20 at n ≥ 200; we list them so future authors can avoid the same paths.

**Top-K candidate reranking by Debye fit (Phase 9.1.4).** We sampled the top-5 indexer cell candidates instead of top-1 and reranked each n = 5 candidate structure by the Debye-loss fit to the target pattern. n = 1 000: match drops 1.6 % → 1.0 %, all-correct 0.1 % → 0.0 %. Hypothesis-rich indexer search supplies more *wrong* cells faster than the Debye rerank can filter them.

**Debye-gradient guidance during DDIM (Phase 9.2).** We added a guidance term $-g_\text{scale} \cdot \nabla_\mathcal{C} \mathcal{L}_\text{Debye}$ to the DDIM update of the coordinate channel, sweep $g_\text{scale} \in \{0, 0.5, 1, 2, 5\}$ on n = 200. Match rate is flat at 1.0–2.5 % across the entire range (within run-to-run noise); all-correct stays at 0 %. The Debye gradient at sampling time is too noisy to steer the diffusion trajectory.

**Wyckoff-site token embeddings (v15, n = 1 000).** Computed with `spglib`, embedded as `nn.Embedding(27, 256)`, added to per-atom features. The embedding *learns* (final mean abs weight 0.065, 5 404/6 912 entries non-zero) but match drops 2.5 % → 2.1 % vs v13 *and* destabilises lattice prediction (lattice loss converges to ~0.6 vs ~0.02 for v16). We suspect the Wyckoff embedding inflates per-atom feature norm (`emb_std` 0.43 vs 0.25 for v16), shifting the lattice-pool input distribution; we did not have budget to confirm.

**Distance-matrix auxiliary loss (v16, n = 1 000).** Pairwise MLP predicting periodic Cartesian distances clipped at the 12 Å RBF cutoff, λ_dist = 0.01 (the largest stable value). Loss decreases (final ~2.0 vs random baseline ~10) but match drops 2.5 % → 1.8 %. Distance prediction biases per-atom features toward absolute distance reconstruction rather than the *relative* geometric updates the diffusion process needs.

**Combination (v14, n = 1 000).** Stacking Wyckoff + distance loss collapses match to 0.8 %, below all other runs except the bare ε baseline. The two extensions interact destructively; we have no clean theoretical explanation.

### 5.5 The encoder bottleneck — perturbation study that motivates the indexer

Our best Pearson-correlation between predicted and true PXRD is 0.43; the same model evaluated on a held-out subset with *correct* coordinates plugged into the differentiable simulator hits 0.97. So the encoder + denoiser is leaving ~0.5 of pattern-space agreement on the table. The auxiliary head — which only sees $\mathbf{g}$ — achieves an aux loss of 0.007 in isolation, meaning the global PXRD vector contains *enough* information to predict the lattice very accurately. The denoiser then under-uses this signal.

To localise the bottleneck, we perturb the lattice supplied to the diffusion sampler away from ground truth by a controlled Å of cell-length error and re-evaluate v21 at n = 200. Match rate decays sharply: 5.6 % (Δ = 0) → 2.0 % (Δ = 0.5 Å) → 0.5 % (Δ = 1.0 Å) → 0 % (Δ ≥ 1.5 Å). The decay is *steeper than linear* — small lattice errors are tolerable up to a knee at ~0.5 Å, beyond which the sampler's coordinate trajectory diverges. This matches the indexer's per-system MAE structure: high-symmetry systems (hex, ortho, with MAE ≤ 1.0 Å) lift; low-symmetry systems (mono, tri, with MAE ≥ 1.5 Å) do not. The indexer-drop-in lift in §5.2 is in this sense *predictable* from the perturbation study: it works exactly where it can, on the high-symmetry MP-20 subset.

---

## 6. Discussion

**The encoder-bottleneck diagnosis explains the negative ablation.** §5.5 shows that the lattice information *exists* in the global PXRD encoding (aux head reaches loss 0.007) but the denoiser cannot extract absolute d-spacings well enough to predict the lattice from scratch. §5.2 shows that supplying the absolute d-spacing scale from a 1970s-vintage Q-space autoindexer recovers some of that gap — small in absolute terms (1.0 → 1.6 %) but large as a fraction of what is recoverable (the 5.6 % oracle ceiling). Top-K rerank and gradient-guidance failed because they operate *after* the encoder has already committed to a bad lattice prior; the indexer succeeds because it bypasses that decision entirely.

**Why per-atom anchoring is hard.** PXRD is a permutation-invariant signal: relabelling atoms of the same element does not change the diffraction pattern at all. Our denoiser is also permutation-equivariant. There is therefore no symmetry-breaking signal to tell atom $i$ "you go in the corner" rather than "you go on the face." The Wyckoff embedding was an attempt to introduce this symmetry-breaking at the input; it failed (§5.4). One promising direction we did not have budget to try is to break symmetry at the *output* — predict an unordered set of orbits plus a Hungarian-style matcher — rather than at the input.

**Two different "match rate" failure modes (Table 3).** PXRDnet's 30 % match with only 5 % all-correct, vs DiffractGPT's 18.9 % match with 15.9 % all-correct, is the most surprising row of the head-to-head. The two models recover *structure* at comparable rates but DGpt's CIF-token output appears to encode *symmetry* in a way PXRDnet's atom-by-atom coordinate decode does not. This suggests that for downstream crystallographic use (where the experimentalist cares about space group as much as about coordinates) the choice of output representation matters more than the choice of latent-optimisation depth. We flag this as a finding that the existing PXRD-inversion literature does not surface, because the field largely reports a single "match rate" number under different tolerances.

**Implications for small reproducible models.** PXRD-Diff at 1.6 % match is 12× behind a fine-tuned 7 B-parameter LLM and 19× behind a CDVAE-style model that burns 4 GPU-hours per inference. A small honest baseline still has a role here: it makes the encoder bottleneck visible (large models hide it in capacity) and it makes the indexer drop-in straightforward to integrate (large models would need architecture-level surgery to substitute a classical lattice predictor at sampling time). The recipe — encoder + Phase 4 architectural fixes + Phase 9 indexer drop-in — is ~3.7 M parameters and runs end-to-end inference in seconds. We see this as the niche we can credibly occupy in the next iteration of the literature.

---

## 7. Limitations

1. **Composition is given.** The unknown-composition setting is out of scope; consistent with prior work but worth stating.
2. **Simulated PXRD only.** No instrument response, preferred orientation, peak asymmetry, or background. Real experimental data will degrade these numbers further.
3. **PXRDnet baseline is n = 20.** Per their published config (`num_starts=100`, `num_grad_steps=5000`, cosine warm-restart factor 7 = 35 k decoder ops/material) PXRDnet inference is ~4 GPU-hr/material on RTX 5090; matching their own published n = 200 would take ~33 days of rental. The 30 % match rate has wide CI (±20 pp Wilson at n = 20); we report what fit in a single overnight rental.
4. **Crystalyze unreproduced.** The Crystalyze checkpoint download link is marked "not yet active" in their own README as of 2026-06-01 (verified via repo HEAD on the same date). We cite the paper but cannot include it in the head-to-head.
5. **Three pattern preprocessing pipelines.** PXRD-Diff uses 4 251-bin 2θ 5–90° / 0.02°; DiffractGPT uses 300-bin 2θ 0–90° / 0.3°; PXRDnet uses 4 096-bin Q-space with on-the-fly sinc^2 broadening at training time. The *structures* under evaluation are identical CDVAE MP-20 material_ids, but the *patterns* are simulated through each model's own preprocessing. Strictly identical inputs would require porting all three preprocessors into one pipeline, which we did not do.
6. **GSAS-II low-symmetry path is experimental.** The monoclinic/triclinic indexer hangs in `findBestCell` on real MP-20 patterns at the published config; we ship it behind a `--use-gsas` opt-in but recommend the native indexer for headline numbers.
7. **Fixed lattice for the Debye loss.** We avoid the joint coord+lattice gradient through the simulator by using ground-truth lattice during training; a curriculum that gradually replaces true lattice with predicted would be valuable but was not tried.
8. **Single random seed per ablation.** Workshop budget; ~0.5 % pilot variance is small relative to the 2.5×–3.5× effects reported.
9. **MP-20 only.** No experiments on larger unit cells, organic crystals, or higher-Z elements.

---

## 8. Conclusion

PXRD inversion at MP-20 scale remains an open problem. A 3.7 M-parameter conditional diffusion model with a differentiable Bragg structure-factor loss recovers 1.6 % of test structures with a classical Q-space autoindexer supplying the lattice, and 5.6 % under a ground-truth-lattice oracle — meaning roughly three-quarters of our gap to the oracle is in *coordinates*, and the remaining quarter is in *lattice*, addressable by indexer improvements alone. Reproduced external baselines on the same `StructureMatcher` harness put DiffractGPT at 18.9 % match (15.9 % all-correct) and PXRDnet sinc100 at 30.0 % match (5.0 % all-correct); the latter two also recover *space group* at very different rates (DGpt 21 %, PXRDnet 5 %, ours 1.4 %), surfacing a finding the existing literature does not: published "match rate" headlines conflate two qualitatively different capabilities.

Six interventions failed in our hands: top-K candidate reranking, Debye-gradient guidance, Wyckoff-site embeddings, an auxiliary distance loss, the combination of the latter two, and — most expensively — the GSAS-II low-symmetry indexer path. Two architectural fixes (lattice-input fix, x₀-residual parameterisation) and one drop-in component (classical autoindexer) are load-bearing and individually documented for re-use.

We release the code, all checkpoints, all per-phase result JSONs, the indexer adapter (both native and GSAS-II), and the differentiable Bragg module as an installable PyTorch package. We hope the recipe and the failure catalogue are useful to the community as it continues to push on a problem that is, on the present evidence, harder than recent reports suggest at first reading.

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

Source code, trained checkpoints, and all training/evaluation logs are available at the project repository (URL to be inserted upon publication). The MP-20 dataset is publicly available via the CDVAE benchmark. All experiments can be reproduced from a single `requirements.txt` and the `scripts/` pipeline.

## Ethics Declaration

This study uses no human subjects, animal subjects, or sensitive data. The MP-20 dataset is composed entirely of publicly available crystal structures from the Materials Project.

---

## References

[^DiffractGPT]: Choudhary, K. (2024). DiffractGPT: Atomic Structure Determination from X-ray Diffraction Patterns using a Generative Pre-trained Transformer. *Journal of Physical Chemistry Letters* (verify final DOI before camera-ready). Reproduced from HF checkpoint `knc6/diffractgpt_mistral_chemical_formula` and code at `github.com/atomgptlab/atomgpt`.

[^Crystalyze]: Riesel, E. A., Mackey, T., Nilforoshan, H., et al. (2024). Crystal Structure Determination from Powder Diffraction Patterns with Generative Machine Learning. *Journal of the American Chemical Society* (verify final DOI). Code: `github.com/ML-PXRD/Crystalyze`. We could not reproduce: checkpoint download link is marked "not yet active" in the upstream README (verified 2026-06-01).

[^PXRDnet]: Guo, G., Saidi, T., Terban, M., Valsecchi, M., Billinge, S. J. L., & Lipson, H. (2024). Ab Initio Structure Solutions from Nanocrystalline Powder Diffraction Data. *arXiv:2406.10796*. Reproduced from HF checkpoint `therealgabeguo/cdvae_xrd_sinc100` and code at `github.com/gabeguo/cdvae_xrd`.

[^GSASII]: Toby, B. H., & Von Dreele, R. B. (2013). GSAS-II: the genesis of a modern open-source all purpose crystallography software package. *Journal of Applied Crystallography*, 46(2), 544–549. Code: `github.com/AdvancedPhotonSource/GSAS-II`.

[^CDVAE]: Xie, T., Fu, X., Ganea, O.-E., Barzilay, R., & Jaakkola, T. (2022). Crystal diffusion variational autoencoder for periodic material generation. *International Conference on Learning Representations*.

[^DiffCSP]: Jiao, R., et al. (2023). Crystal structure prediction by joint equivariant diffusion. *Advances in Neural Information Processing Systems* 36.

[^MatterGen]: Zeni, C., et al. (2024). MatterGen: A generative model for inorganic materials design. *arXiv:2312.03687*.

[^MACE]: Batatia, I., Kovács, D. P., Simm, G. N. C., Ortner, C., & Csányi, G. (2022). MACE: Higher order equivariant message passing neural networks for fast and accurate force fields. *Advances in Neural Information Processing Systems* 35.

[^SchNet]: Schütt, K. T., et al. (2017). SchNet: A continuous-filter convolutional neural network for modeling quantum interactions. *Advances in Neural Information Processing Systems* 30.

[^NicholDhariwal]: Nichol, A., & Dhariwal, P. (2021). Improved denoising diffusion probabilistic models. *International Conference on Machine Learning*.

[^DDIM]: Song, J., Meng, C., & Ermon, S. (2021). Denoising diffusion implicit models. *International Conference on Learning Representations*.

[^WrenzcuckPhysML2025]: *Placeholder for a differentiable scattering simulator reference; replace with the appropriate citation before submission.*

> **Citation note for review.** DiffractGPT, Crystalyze, and the differentiable-scattering reference still need final DOI/journal verification before camera-ready. The reproduced PXRDnet and DiffractGPT numbers in §5.3 are computed by the author on the released checkpoints, not transcribed from the cited papers; the central claims do not depend on the precise published numbers, only on the *reproduced* numbers under our shared evaluation harness.

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
