# PXRD-Diff: diagnosing the lattice-recovery bottleneck in diffusion-based powder-diffraction structure inversion

**Frank Cai**

Purdue University, West Lafayette, IN, USA

frankyc11223@gmail.com

*Manuscript type: Research Paper (Journal of Applied Crystallography).*

**Synopsis:** A deliberately small conditional diffusion model for powder-diffraction structure inversion is used as a diagnostic instrument: a true-lattice oracle lifts its match rate from 1.9 to 5.5 %, while a controlled ablation shows that restoring absolute peak positions to the auxiliary lattice head reduces the cell error by at most 13 % and closes none of that gap. A classical autoindexer serves as a control, three external models are re-scored on one harness, and a review-prompted code audit corrects three implementation errors.

---

## Abstract

Inverting a one-dimensional powder X-ray diffraction (PXRD) pattern to a crystal structure remains an open problem. This paper is a diagnostic study. PXRD-Diff, a 3.7 M-parameter conditional diffusion model trained on the MP-20 split of the CDVAE benchmark, is kept deliberately small so that its failures are visible; it is not proposed as a competitive tool. On one pymatgen StructureMatcher harness, DiffractGPT, PXRDnet and deCIFer reach 18.9, 30.0 and 73.8 % match; PXRD-Diff reaches 1.9 %. The central finding is a paired oracle gap that replicates across checkpoints and seeds: a true-lattice oracle lifts match to 5.5 % (n = 1000, three sampling seeds on one trained checkpoint; 95 % Wilson interval [4.7, 6.4]) against 1.9 % [1.5, 2.5] for the learned lattice (paired McNemar p = 3 × 10⁻¹⁸), and pattern agreement falls tenfold once the cell-length error reaches ~0.4 Å. The encoder's auxiliary head fits the lattice to a 0.007 normalised training loss yet leaves a 1.1–1.3 Å test error, and a controlled ablation on the frozen encoder shows that handing it absolute peak positions, by position-aware pooling or explicit d-spacings, reduces that error by at most 13 % and recovers no more structures end to end. The submitted reading that global pooling discards the d-spacings is withdrawn (§5.6). A classical autoindexer given the true crystal system reaches 1.6 % [1.2, 2.1] (1.4 % without it) and is not distinguishable from the learned head (p = 0.35). A code audit prompted by review corrected three implementation errors; retraining with them raised the learned head from 0 to 1.9 % and removed the indexer's lead of the submitted version.

**Keywords:** powder diffraction; autoindexing; crystal structure determination; conditional diffusion models; machine learning; differentiable simulation; reproducibility

---

## 1. Introduction

PXRD is the most common structural-characterisation measurement in solid-state chemistry. The forward problem is solved by 1920s structure-factor physics and a one-line `pymatgen` call. The inverse is brutal: symmetry-equivalent structures produce identical Bragg reflections, intensities are orientation-averaged, and the 3D→1D reduction destroys atom labelling. The traditional indexing → Pawley/Le Bail → Rietveld pipeline needs a trained crystallographer and a starting model good enough to converge; for the long tail of new materials, structures stay unsolved.

Three recent generative approaches (DiffractGPT: Choudhary, 2025; Crystalyze: Riesel *et al.*, 2024; PXRDnet: Guo *et al.*, 2025) argue neural networks can shortcut this pipeline. They report match rates in the tens of percent, but under mutually incompatible evaluation protocols, and failure modes are seldom catalogued. A practitioner cannot easily form a calibrated expectation of *what is hard* about PXRD inversion.

We build a small reproducible conditional diffusion model (**PXRD-Diff**) and use it as a diagnostic instrument. Its size is a design choice: a 3.7 M-parameter model exposes a lattice-recovery failure that the larger systems of Table 3 evidently do not share, whether through capacity, architecture, training data or inference compute (not separated here). It is not proposed as an alternative to the larger systems it is compared with. Contributions:

1. **A small, fully released baseline.** 3.7 M parameters, trains on a single RTX 5090 in ~1.9 h on the MP-20 split of the CDVAE benchmark (Crystal Diffusion Variational Autoencoder; Xie *et al.*, 2022). A PyTorch port of the `pymatgen` structure-factor calculation, differentiable with respect to coordinates and lattice, supplies an auxiliary pattern-matching loss. The port introduces no new physics (§3.3).
2. **Lattice-recovery bottleneck diagnosis (the primary contribution).** A true-lattice oracle lifts match to **5.5 %** at n = 1000 (three sampling seeds on one trained checkpoint; Wilson 95 % CI [4.7, 6.4]) from 1.9 % [1.5, 2.5] for the learned head (paired McNemar p = 3 × 10⁻¹⁸), and a perturbation study (§5.5) shows that recovery is not graceful: pattern agreement falls tenfold by ~0.4 Å of cell-length error, so ~0.5 Å is used as a working tolerance. The encoder's auxiliary head reaches a 0.007 *normalised* training loss, yet its lattice recovers only 1.2 % end to end, a 1.1 Å error well above that tolerance (§5.6). A controlled pooling ablation (§5.6) finds that neither position-aware pooling nor explicit peak-position features reduce the error by more than 13 % or recover more structures end to end, so the best-supported reading, offered as a hypothesis, is the regression formulation rather than the pooled encoding. The 5.5 % oracle ceiling shows coordinates remain a co-limiter.
3. **A classical control.** A Q-space autoindexer, supplied with the true crystal system, replaces the learned lattice head at sampling time and reaches 1.6 % [1.2, 2.1]. With the submitted checkpoint the learned head recovered nothing and the indexer's lead was significant (27 fixed, 0 broken); with the checkpoint retrained under the corrected loss the two tie at three seeds (42 fixed, 52 broken, p = 0.35), the indexer fixing hexagonal and tetragonal structures and the learned head cubic ones, a post-hoc per-system split (§5.2). This is a sanity check on the diagnosis, not a method: a fifty-year-old algorithm given extra prior information is expected to match or beat the learned head. Without the crystal system the indexer indexes 43.4 % of cells against 48.8 % and recovers 1.4 % [1.0, 1.9] downstream (§3.5).
4. **Four-item failure catalogue.** Top-K Debye rerank, Debye-gradient guidance of the sampler, Wyckoff-letter embeddings and a distance-matrix aux loss did not improve match rate under the conditions tested: the two Debye-based interventions ran with the submitted simulator and were not rerun (§5.4, §5.7), and the Wyckoff and distance-loss runs are single-seed comparisons against a baseline later shown to be a high draw (§7). The robust signal is the order-of-magnitude lattice-loss destabilisation under the Wyckoff embedding, which is read narrowly: the embedding carried a label, not a constraint (§5.4). The gradient-guidance failure agrees with independent work showing the PXRD loss landscape is too rough for gradient descent (Segal *et al.*, 2026).
5. **Reproduced baselines and a shared harness.** DiffractGPT (Mistral-7B with low-rank adaptation, LoRA), PXRDnet (CDVAE + XRD encoder) and deCIFer are re-scored on our harness: **18.9 %** match (DiffractGPT, hereafter DGpt; n = 990 parseable of 1000), **30.0 %** (PXRDnet, n = 20) and **73.8 %** (deCIFer, n = 298). These numbers set the scale of the diagnostic model; they are not a claim of competitiveness. A candidate divergence between match rate and space-group recovery across two baselines is raised in §5.3 and §6 as a hypothesis that the n = 20 PXRDnet sample cannot establish.
6. **A code audit.** Review of the released code identified three implementation errors: the atomic form factor was evaluated in the wrong convention, the Debye loss in x₀-residual mode was evaluated on the wrong tensor, and patterns the indexer could not index were silently given the true lattice. A systematic text-versus-record check found two further discrepancies. §5.7 reports each item, its effect on every published number, and what was re-run.

---

## 2. Related work

**Diffusion for crystals.** CDVAE (Xie *et al.*, 2022) introduced the MP-20 protocol; DiffCSP (Jiao *et al.*, 2023) used joint coord+lattice diffusion; MatterGen (Zeni *et al.*, 2025) scaled to millions of structures. PXRD-Diff borrows DiffCSP's joint diffusion, restricted to the conditional setting where a 1D pattern conditions generation.

**Neural PXRD inversion.** DiffractGPT (Choudhary, 2025) is a fine-tuned Mistral-7B that decodes peak list + formula → Crystallographic Information File (CIF) tokens. Crystalyze (Riesel *et al.*, 2024) uses CDVAE conditioned on an XRD transformer encoder with post-hoc symmetry filtering. PXRDnet (Guo *et al.*, 2025) is a CDVAE variant with *iterative latent-space gradient guidance* (~3.5 M decoder ops/material). deCIFer (Johansen *et al.*, 2025) is a concurrent autoregressive-language-model approach that, like DiffractGPT, emits CIF tokens. All four are larger than PXRD-Diff and report results under bespoke evaluation pipelines; §5.3 re-scores DiffractGPT, PXRDnet, and deCIFer on a single shared `StructureMatcher` harness. Crystalyze's checkpoint download link is "not yet active" (verified 2026-06-01).

**Where the unit cell comes from.** Recent approaches differ in whether the unit cell is generated jointly with the coordinates or supplied as a separate input. PXRDGen (Li *et al.*, 2025) generates lattice and coordinates jointly by default, but can accept an independently determined cell, from a cell network or from conventional indexing, as an additional condition. XtalNet (Lai *et al.*, 2025) is an end-to-end equivariant approach, reported on metal–organic-framework benchmarks. Parackal *et al.* (2024) enumerate symmetry-confined Wyckoff arrangements and rank them with a learned energy model before fitting the pattern. XRDSol (Yu *et al.*, 2026) denoises coordinates with an equivariant diffusion model conditioned on the pattern, and takes the stoichiometry and the unit cell as inputs. That last design choice is the one this paper examines from the other side: XRDSol assumes an indexed cell, and the cell is exactly what our learned head fails to recover. Chitturi *et al.* (2021) regress lattice parameters from simulated patterns with a convolutional network and analyse how impurities, noise and broadening degrade the prediction, a diagnostic aim close to ours. None is reproduced here; they place PXRD-Diff, a diagnostic instrument rather than a competing solver, within the range of current designs.

**Classical structure solution from powder data.** Recovering a unit cell from peak positions is a solved sub-problem with a fifty-year toolchain: ITO, TREOR (Werner *et al.*, 1985) and DICVOL (Boultif & Louër, 2004), ranking solutions with the de Wolff (1968) figure of merit M20, index high-symmetry powder patterns in CPU milliseconds, with the difficulty concentrated in low-symmetry (monoclinic and triclinic) cells. Indexing is only the first stage of classical structure solution. Once the cell and space group are fixed, direct-space search, simulated annealing and charge flipping complete the structure, as implemented in programs such as FOX (Favre-Nicolin & Černý, 2002) and EXPO (Cuocci *et al.*, 2022). §3.5 reuses only the indexing stage, as a control for the learned lattice head. The indexer is given the true crystal system, which classical practice must itself infer; §3.5 also reports the run without it. §5.2 and §5.5 show the resulting lift tracks the symmetry-dependent accuracy classical indexers have always had.

**Differentiable physics.** Scattering-pattern losses date to Rietveld. The module used here is a PyTorch port of the `pymatgen` structure-factor computation, differentiable with respect to coordinates and lattice so that it can serve as a diffusion-training loss term; it adds no physics beyond that computation (§3.3). Concurrent generative-inversion work conditions on differentiable spectral targets for amorphous and nanostructured systems (Guo & Schwalbe-Koda, 2026). Relevant to our negative results, Segal *et al.* (2026) show the powder-XRD similarity loss landscape is "too rough for gradient descent", independent evidence for why our sampling-time Debye-gradient guidance fails (§5.4).

**Equivariance.** Not strictly E(3)-equivariant: we use periodic-distance RBFs (SchNet-style; Schütt *et al.*, 2017). A frozen MACE (Batatia *et al.*, 2022) encoder showed no measurable benefit at MP-20 scale in pilots and was deferred.

---

## 3. Method

### 3.1 Problem formulation

Let $\mathcal{C} = (\mathbf{F}, \mathbf{Z}, \mathbf{L})$ denote a crystal structure with fractional coordinates $\mathbf{F} \in [0,1)^{N \times 3}$, atomic numbers $\mathbf{Z} \in \{1,\ldots,100\}^N$, and a $3 \times 3$ lattice matrix $\mathbf{L}$ parameterised by $(a,b,c,\alpha,\beta,\gamma) \in \mathbb{R}^6$. Let $\mathbf{p}(\mathcal{C}) \in \mathbb{R}^{4251}$ be the simulated Cu Kα PXRD pattern on a fixed 2θ grid from 5° to 90° at 0.02° resolution, normalised to maximum intensity 1.

We are given $\mathbf{p}$ and $\mathbf{Z}$ at test time, and we wish to sample $(\mathbf{F}, \mathbf{L}) \sim p(\,\cdot\, | \mathbf{p}, \mathbf{Z})$. Note that the composition $\mathbf{Z}$ being known is a meaningful simplification (in practice one usually knows the chemistry from synthesis), but it is consistent with prior work on the same task.

### 3.2 Architecture overview

PXRD-Diff has three trained components: a PXRD encoder, a denoiser, and a small auxiliary lattice head used only during training.

**PXRD encoder.** A 1D ResNet of four blocks (channels $64 \to 128 \to 256 \to 256$, stride-2 downsampling, GroupNorm, SiLU) ingesting the standardised pattern. We expose two outputs: a global pooled vector $\mathbf{g} \in \mathbb{R}^{256}$ and a multi-resolution feature map $\mathbf{F}_{\text{pxrd}} \in \mathbb{R}^{L \times 256}$ obtained by 1×1-projecting each block output to $d_\text{model}$ and concatenating along the spatial axis. *The multi-resolution map is critical.* In an early run (referred to as `gpu_v4` in our logs) we conditioned the denoiser on $\mathbf{g}$ only via additive broadcast, and the coordinate loss never moved off the random baseline of 3.0; the auxiliary head told us the encoder was learning useful features (its loss dropped from 0.99 to 0.007), but the `AdaptiveAvgPool1d(1)` collapse destroyed all spectral structure before it could reach the denoiser.

**Denoiser.** A periodic-distance message-passing network with $L = 3$ layers and $d_\text{model} = 256$. Atom embeddings $\mathbf{h}_i^{(0)} = \text{Emb}(\mathbf{Z}_i) + W_\text{coord}\mathbf{F}_i^{(t)}$ are updated by alternating (i) message passing over RBF-encoded periodic Cartesian distances under minimum-image convention, with timestep FiLM conditioning, and (ii) cross-attention from atoms (queries) to multi-resolution PXRD features (keys/values). Two output heads predict per-atom coordinate noise $\boldsymbol{\epsilon}_F \in \mathbb{R}^{N \times 3}$ and lattice noise $\boldsymbol{\epsilon}_L \in \mathbb{R}^6$. The lattice head is a small multilayer perceptron (MLP) applied to $[\,\bar{\mathbf{h}}\,;\, \mathbf{g}\,;\, W_\text{lat}\boldsymbol{\ell}^{(t)}\,;\, \mathbf{t}_\text{cond}\,]$: that is, the pooled atom features, the global PXRD embedding, **a projection of the noisy lattice itself**, and the timestep encoding (see §3.4).

**Auxiliary head.** A single-hidden-layer MLP that predicts $(a,b,c,\alpha,\beta,\gamma)$ from $\mathbf{g}$ alone, trained with an MSE loss against the (normalised) ground-truth lattice. Its purpose is purely diagnostic, to expose whether the encoder is learning any lattice-relevant representation, and its loss does not flow back into the denoiser. Throughout, "learned head" denotes the denoiser's lattice channel, whose sampled cell the full pipeline uses; "auxiliary head" (or "aux head") denotes this diagnostic regressor.

### 3.3 Differentiable Bragg structure-factor loss

This module is an engineering component, not a physical contribution. It is a PyTorch port of the structure-factor calculation in `pymatgen.analysis.diffraction.xrd.XRDCalculator`, written so that gradients flow to fractional coordinates and lattice. We implement
$$
F(hkl) = \sum_j f_j(s) \exp\!\left(-B_{\text{iso}} s^2\right) \exp\!\left[2\pi i\,(h x_j + k y_j + l z_j)\right],
$$
with the atomic form factor in the convention `pymatgen` uses for its tabulated coefficients,
$$
f_j(s) = Z_j - 41.78214\, s^2 \sum_{k=1}^{4} a_k \exp(-b_k s^2),
$$
where $(a_k, b_k)$ are read from `ATOMIC_SCATTERING_PARAMS`, $Z_j$ is the atomic number, $s = \sin\theta/\lambda$, and $B_{\text{iso}} = 0.5$ Å² is a uniform isotropic temperature factor. The intensity at each reflection is $|F(hkl)|^2$ multiplied by the standard Lorentz-polarisation correction. We enumerate all $(h,k,l)$ with $|h|,|k|,|l| \le 5$ (1330 reflections after excluding the origin), place each on the 2θ grid via Bragg's law, and broadcast each reflection as a Gaussian peak with FWHM 0.1° to obtain a continuous, differentiable pattern on a 256-bin coarse grid. The module treats ideal powders only: no preferred orientation, peak asymmetry, instrument profile or background is modelled. It is a single `nn.Module`, vectorised over the batch dimension.

**Correction made during review.** The submitted version evaluated the form factor as the bare sum $\sum_k a_k \exp(-b_k s^2)$ with the same coefficients, which is not the expression they were fitted for: it gives the wrong element weighting (for silicon $f(0) = 5.8$ where $Z = 14$) and decays too quickly with $s$. Both referees who inspected the code identified the discrepancy. The module was corrected and re-validated (below), and the production checkpoint was retrained from scratch with the corrected module and the corrected x₀-mode loss of §3.4 (v22, behind Tables 2, 2a and 3 and Fig. 4); §5.7(a) gives the accounting, and the exploratory Phase-4 rows of Table 1 were not retrained (§5.1).

**Validation.** On the first 1000 MP-20 test structures we compute the Pearson correlation between the differentiable pattern and the `pymatgen.XRDCalculator` reference, both on the 256-bin coarse grid (Fig. 1). With the corrected form factor the mean is 0.988 and the median 0.999; 98.0 % of structures exceed 0.9. With the submitted version's form factor the same structures give a mean of 0.952 and a median of 0.969; 90.7 % exceed 0.9. The corrected expression scores at least as high on every one of the 1000 structures. On the first 50 structures the means are 0.939 (submitted form factor) and 0.974 (corrected). The submitted version reported 0.962 on 50 structures with the validation script's default reflection range ($|h|,|k|,|l| \le 10$), wider than the $\le 5$ training grid used above. Appendix C reproduces that value for the submitted expression (0.963 on the first 50, 0.960 on all 1000), gives 0.999 for the corrected expression at both sizes, and shows that the lower tail is a truncation effect of the $\le 5$ grid on small cells rather than a property of the form factor. Gradients through the structure factor with respect to fractional coordinates were verified non-zero.

![**Figure 1. Differentiable-simulator validation.** Distribution of Pearson correlation between the PyTorch DiffPXRD module and `pymatgen.XRDCalculator` over the first 1000 MP-20 test structures, with the corrected form factor (mean 0.988, median 0.999) and with the form factor of the submitted version (mean 0.952, median 0.969). The corrected expression scores at least as high on every structure. The residual lower tail is a truncation effect of the $|h|,|k|,|l| \le 5$ reflection range on small cells (Appendix C).](fig3_diffpxrd_validation.pdf){#fig:validation}

**As a loss.** During training we recover an estimate $\hat{\mathbf{F}}$ of the clean coordinates from the model output (see §3.4), simulate $\hat{\mathbf{p}} = \text{DiffPXRD}(\hat{\mathbf{F}}, \mathbf{Z}, \mathbf{L}_\text{true})$, and compare with the input pattern via a Pearson-correlation loss $\mathcal{L}_\text{Debye} = 1 - \rho(\hat{\mathbf{p}}, \mathbf{p})$. We use the true lattice for this auxiliary loss because the differentiable simulator is more sensitive to lattice errors than to coordinate errors at the early stages of training, and disentangling the two signals proved more stable.

### 3.4 Training objective

We use a variance-preserving stochastic differential equation (VP-SDE) with cosine schedule (Nichol & Dhariwal, 2021) for both channels:
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
Because the heads are MLPs initialised to output near-zero, this means the model starts as the identity transform: at $t \approx 1$ (near pure noise) the model outputs the noisy input as its $\hat{\mathbf{F}}^{(0)}$ estimate, which is wrong but at least bounded; ε-prediction with the same architecture had to learn the entire transform from random initialisation and never recovered. At single seed this change appeared to lift the headline match metric 2.8×; a multi-seed checkpoint sweep (§7) shows that match-rate lift does not replicate, but the parameterisation does reproducibly improve the pattern-Pearson (≈ 0.36 → 0.40, see §5.1).

**Correction made during review.** In x₀-residual mode the Debye loss must be evaluated on the clean-coordinate estimate $\hat{\mathbf{F}}^{(0)} = \mathbf{F}^{(t)} + \text{Coord-Head}(\mathbf{h})$; the submitted code evaluated it on the residual $\text{Coord-Head}(\mathbf{h})$ alone, so in every x₀-residual run (v13, v14, v15, v16 and the Phase 9 checkpoint v21) the Debye term compared the pattern of the wrong tensor with the target. The ε-mode runs v10 and v11 were unaffected, as were the coordinate and lattice losses and every evaluation-time number. The code is fixed and the production checkpoint was retrained with the corrected loss and the corrected simulator (v22, §3.3); its final exponential-moving-average (EMA) loss fell from 0.726 to 0.602. §5.7(b) gives the accounting; the Phase-4 rows v13–v16 were not retrained and remain labelled exploratory single-seed runs carrying this defect (§5.1).

*Lattice-head input.* The lattice head reads the noisy lattice $\boldsymbol{\ell}^{(t)}$ as an explicit input feature. Without this, i.e. if the head only sees the (clean) lattice that is also passed to the denoiser as the geometry context, the head can only ever predict the unconditional mean noise, $\mathbb{E}[\boldsymbol{\epsilon}] = 0$, and its loss stays pinned at the random baseline of 1.0 forever. With this fix, the lattice loss drops to 0.02–0.06 on stable runs.

We also experimented with two extensions that did not pan out (§5.4): a **Wyckoff-letter embedding** added to atom features, and an auxiliary **distance-matrix loss** in which a pairwise MLP predicts ground-truth periodic distances between atoms. Both are documented in the released code behind the `--use-wyckoff` and `--dist-weight` flags.

### 3.5 Classical indexing as a control for the learned lattice head

The denoiser's lattice head (§3.2) is the *only* path by which the model encodes absolute d-spacings: every Bragg reflection 2θ position is set by the lattice through Bragg's law, so the lattice parameters are the absolute reference frame for every peak position in the input pattern. As §5.5 shows, the encoder + denoiser learns *relative* d-spacing structure (v13, Table 1: Pearson 0.43 between predicted and target patterns at evaluation time, vs 0.97 between target and ground truth) but not the absolute scale. Classical autoindexing, which extracts peak positions, maps them to Q-space (Q = 4π sinθ/λ), and fits a unit cell by enumeration of (h,k,l) → Q²(h,k,l), has solved this sub-problem since the 1970s and runs on a CPU in milliseconds per pattern.

We implement a from-scratch Q-space autoindexer that takes (i) the same simulated PXRD pattern fed to the encoder and (ii) the true crystal system of the target structure, supplied as a Bravais code, and returns a candidate (a, b, c, α, β, γ). Item (ii) is prior information beyond the composition that the learned head receives; it makes the indexer an optimistic control rather than a fair competitor (§7; the unknown-system run below removes it). Peak picking uses a 1D local-maximum filter with a relative-intensity floor; the candidate Q-vector is fit by least squares to the de-Wolff dichotomy parameter form for each Bravais lattice. At sampling time we run the indexer first, substitute its output for the lattice channel of the diffusion sampler, and run the DDIM sampler (§3.6) only on the coordinate channel.

**Per-system accuracy (n = 1000 MP-20 test).** Native indexer overall (Fig. 3): 48.8 % strict cell match, 1.45 Å mean absolute error (MAE) in cell length. Per crystal system: cubic 52.9 % / 1.14 Å, tetragonal 52.8 % / 0.96 Å, hexagonal 77.9 % / 0.53 Å, orthorhombic 59.8 % / 0.96 Å, trigonal 43.3 % / 4.62 Å, monoclinic 18.9 % / 1.76 Å, triclinic 0 %. The low-symmetry blowup (monoclinic and triclinic each have ≥ 4 free cell parameters and the de-Wolff search becomes hypothesis-capped) is the dominant remaining error. A GSAS-II (Toby & Von Dreele, 2013) adapter (`src/pxrd_diff/indexer_gsas.py`) for the low-symmetry path ships behind a `--use-gsas` opt-in flag; it hangs on real MP-20 monoclinic/triclinic patterns inside `findBestCell` and is experimental (§7).

**Without the crystal system.** At a referee's request the indexer was also run with no crystal system supplied. Cubic, tetragonal, hexagonal/trigonal, orthorhombic and monoclinic cells are searched in that order, highest symmetry first. A lattice type is accepted by the rule the classical programs use (de Wolff, 1968; Werner *et al.*, 1985; Boultif & Louër, 2004): the first type whose best cell indexes every observed line with M20 ≥ 10 is kept, and if none does, the highest-M20 cell among those indexing every line. The reported system is the metric symmetry of the returned cell.

On the same n = 1000 this reaches **43.4 %** strict cell match [40.4, 46.5] with a 1.67 Å mean cell-length MAE, against 48.8 % [45.7, 51.9] and 1.45 Å with the system given, and it recovers the lattice type itself for 65.4 % of the patterns (68.1 % of the 961 that both modes index). Per crystal system, strict success with the system given → unknown: cubic 52.9 → 49.2 %, hexagonal 77.9 → 65.4 %, trigonal 43.3 → 30.9 %, tetragonal 52.8 → 46.7 %, orthorhombic 59.8 → 53.6 %, monoclinic 18.9 → 19.6 %; two of the 39 triclinic structures, unindexed with the system given, are matched within tolerance by a higher-symmetry cell when the system is unknown. Paired on the same structures, removing the system loses 65 cells and gains 9 (McNemar p < 10⁻¹⁰): under ideal peak positions the crystal-system prior is worth about five percentage points to the indexer, not the difference between working and failing.

A first implementation that simply kept the highest M20 across lattice types reached only 24.3 %, choosing the correct system for 40.7 % of patterns and cubic for 12 of the 238 cubic structures. M20 carries no penalty for free parameters, so on a pattern with a handful of exactly placed lines a four-parameter monoclinic sub-cell out-scores the true cubic cell (ScAlRh₂, mp-867922: M20 = 2182 against 1231, both indexing every line). That run is released beside the final one (`--unknown-rule maxm20`) because the comparison is itself informative: the choice of acceptance rule (24.3 → 43.4 %) matters more than the prior it replaces (43.4 → 48.8 %). Substituted into the sampler at three seeds, these cells recover 1.4 % [1.0, 1.9] (42 / 3000) against 1.6 % [1.2, 2.1] with the system given: six structures lost and none gained on the same seeds (McNemar p = 0.03), with identical pattern Pearson (0.41), and no significant difference from the learned head's 1.9 % (36 fixed, 52 broken, p = 0.11). Table 2 carries this as its third row.

The indexer is a drop-in in the mechanical sense: training is unchanged, and the diffusion sampler is unchanged except for the lattice-channel substitution at t = T. Patterns for which the indexer returns no cell (39 of the 1000 test patterns) are scored as misses; the submitted version had given them the true lattice (§5.7). §5.2 reports both with and without the indexer.

### 3.6 Sampling

Denoising diffusion implicit model (DDIM) sampling (Song *et al.*, 2021) with 50 steps and $\eta = 0$. We start from independent standard Gaussian noise on the lattice and on the (un-wrapped) coordinates, then alternate the standard DDIM update on each channel. Coordinates are wrapped to $[0,1)$ after every step. Predicted lattice parameters are de-standardised at the end and clipped to physically valid ranges $(a,b,c) \in [0.5, 100]$ Å, $(\alpha,\beta,\gamma) \in [10°, 170°]$ before being passed to `pymatgen.Lattice.from_parameters`.

A subtle bug fix is worth noting. In the x₀-residual variant, recovering the implicit ε for the DDIM update requires dividing by $\sqrt{1-\bar{\alpha}(t)}$; near $t = 0$ this denominator vanishes. We clamp it to 0.05 and skip the very last DDIM step when sampling, which removed a class of structures with NaN coordinates that we initially saw.

---

## 4. Experiments

**Dataset.** Canonical CDVAE (Xie *et al.*, 2022) MP-20 split: 27 136 / 9047 / 9046 train/val/test, ≤ 20 atoms/conventional cell. Patterns simulated with `pymatgen.XRDCalculator` (Cu Kα₁, λ = 1.54184 Å), max-normalised, 4251 bins 5–90° / 0.02°; cached as `.npz`, zero failures over 45 196 structures.

**Evaluation.** Three views: (1) **composition match** (trivial, $\mathbf{Z}$ given); (2) **structure match** via `pymatgen.StructureMatcher` at $(\ell_\text{tol}, s_\text{tol}, \alpha_\text{tol}) = (0.2, 0.3, 5°)$, the headline "match rate"; (3) **coordinate root-mean-square deviation (RMSD)** on aligned, permuted atoms when matched. `StructureMatcher` is run with `scale=True`, so the match rate is invariant to a uniform scaling of the cell and tests cell shape and coordinates; absolute scale enters through pattern Pearson and $R_{wp}$. We also report space-group match at symprec $\in \{0.01, 0.05, 0.1, 0.2\}$, pattern Pearson, and $R_{wp}$.

The stricter **"all-correct"** rule combines `composition ∧ sg-match@symprec=0.1 ∧ rmsd ≤ 0.1 Å`, the experimentally relevant case for downstream crystallographic use.

Two evaluation modes: *full pipeline* (predicted lattice + coords) and *true-lattice/coord-only* (ground-truth lattice substituted). The latter isolates coordinate quality from lattice quality and is the stricter ablation setting. In the indexer mode (§3.5) a pattern the indexer cannot index counts as a miss; it is never scored with the true lattice.

**Implementation.** AdamW, learning rate $10^{-3}$ (the value stored in the released v21 checkpoint; §5.7(d) on the submitted text's $5\times10^{-4}$), cosine decay to zero over 100 k steps, WD $10^{-4}$, grad-clip 1.0. Batch 64, ~236 epochs, single RTX 5090 (~1.7–1.9 h/run). 3.7 M parameters; $L = 5, d = 384$ (~10 M) gave no benefit over the steps it ran (§5.1). VP-SDE with $t \sim U(0,1)$. Total compute before the revision: ~35 GPU-hours, ~USD 25 on Vast.ai (Phase 4 ablation + Phase 9 v21 retrain + indexer/perturbation sweeps + DGpt n = 1000 + PXRDnet n = 20 + evaluation); the revision runs of §5.2, §5.6 and §3.5 added about 5.5 GPU-hours (Acknowledgements).

---

## 5. Results

Order: Phase 4 architectural ablation (§5.1) → classical indexer control (§5.2) → reproduced baselines (§5.3) → failure catalogue (§5.4) → lattice-perturbation study (§5.5) → pooling ablation (§5.6) → code audit (§5.7).

### 5.1 Phase 4: architectural ablation (true-lattice setting)

Table 1 (Fig. 2): six runs sweeping the Debye loss, the lattice-input fix, x₀-residual, and two extensions (Wyckoff, distance loss), all on n = 1000 MP-20 test with true lattice substituted. Every row is a single training seed. The table is exploratory: it guided the design, and only the ε-vs-x₀-residual contrast was later replicated across seeds (§7). Runs v11 and v13–v16 were trained with the submitted version's simulator, and v13–v16 with the x₀-mode Debye defect (§3.3, §3.4). These rows were not retrained: only the production checkpoint was (v22, §5.2), and Table 1 is retained as the exploratory, single-seed record it always was, with the defects it carries stated here rather than corrected.

**Table 1.** Phase 4 coordinate-only ablation, MP-20 test (n = 1000, true lattice substituted). Single seed per row; exploratory.

| Run   | Parameterisation | λ_Debye | Wyckoff | λ_dist | Match % | Pearson | RMSD (Å) |
|-------|------------------|---------|---------|--------|---------|---------|----------|
| v10   | ε                | 0       | –       | 0      | 1.40    | 0.359   | 0.17     |
| v11   | ε                | 1       | –       | 0      | 0.90    | 0.365   | 0.15     |
| **v13** | **x₀-residual**  | **1**   | **–**   | **0**  | **2.51** | **0.434** | 0.22 |
| v14   | x₀-residual      | 1       | yes     | 0.01   | 0.80    | 0.367   | 0.14     |
| v15   | x₀-residual      | 1       | yes     | 0      | 2.10    | 0.392   | 0.21     |
| v16   | x₀-residual      | 1       | –       | 0.01   | 1.80    | 0.368   | 0.22     |

![**Figure 2. Phase 4 architectural ablation (single seed per run; exploratory).** Match rate (a), pattern Pearson (b) and coordinate RMSD of matched structures (c) for the six coordinate-only runs of Table 1 on MP-20 test (n = 1000, true lattice substituted). The ε → x₀-residual switch (v11 → v13) shows the largest single-seed match lift (panel a), but that lift does not replicate across seeds; only the pattern-Pearson gain (panel b) does (§7). The Wyckoff and distance-loss runs land below v13, within single-seed noise.](fig1_ablation.pdf){#fig:ablation}

The **lattice-input fix** (§3.4) is load-bearing: it drops lattice loss from 1.0 (pinned at the prior variance for 100 k steps) to ~0.02 once $W_\text{lat}\boldsymbol{\ell}^{(t)}$ enters the head input. A second, apparent lift did **not** survive replication: switching from **ε to x₀-residual** at fixed Debye λ = 1 (v11 → v13) lifted single-seed match 0.9 % → 2.5 %, but a multi-seed checkpoint sweep (two seeds × five checkpoints, 20 k–100 k) found ε and x₀-residual indistinguishable on match rate (§7), so the 2.8× match lift was a single-seed fluctuation. The x₀-residual parameterisation does reproducibly improve pattern-Pearson (0.36 → 0.40), the part of the Phase-4 signal that survives. Capacity is not obviously the limiter: a 10.1 M-parameter run ($d = 384$, $L = 5$, `gpu_v12`) showed the same ~1.0 coord-loss plateau as v10 over its first 18 k steps before it was stopped; a full-length run was not made. Under the full pipeline (sampled lattice), v13's own rate falls to 1.2 % at n = 256; the Phase 9 checkpoint reaches 5.5 % under oracle (v22, three seeds; the submitted v21 gave 4.5 %) but only 1.9 % under the full pipeline. §5.2 attacks that gap.

### 5.2 Phase 9: classical indexer control

Table 2 compares four lattice sources on the same n = 1000 at three sampling seeds with the v22 checkpoint (the v21 configuration retrained under the corrected simulator and x₀-mode Debye loss, §3.3, §3.4, §5.7); the rows differ only in the lattice the sampler conditions on. The indexer rows are controls with extra prior information (§3.5). The submitted version's v21 numbers are retained in §5.7.

**Table 2.** Phase 9 lattice-source comparison (n = 1000 MP-20 test × three sampling seeds on the single v22 training run, 3000 structure-seeds pooled). The indexer receives the true crystal system; the 117 structure-seeds it cannot index are scored as misses. Match % with 95 % Wilson CI; the other columns are means over seeds.

| Lattice source                                | Match % (95 % CI) | All-correct % | sg@0.1 % | RMSD med (Å) | R_wp | Pearson |
|-----------------------------------------------|-------------------|---------------|----------|--------------|------|---------|
| Learned head only (full pipeline)             | 1.9 [1.5, 2.5]    | 0.0           | 1.4      | 0.232        | 17.0 | 0.18    |
| **Classical Q-space autoindexer (§3.5), crystal system given** | **1.6 [1.2, 2.1]** | **0.2**    | **0.3**  | **0.035**    | 11.4 | 0.41 |
| Classical Q-space autoindexer, crystal system unknown | 1.4 [1.0, 1.9]    | 0.2           | 1.6      | 0.038        | 12.2 | 0.41    |
| Ground-truth lattice (oracle)                 | 5.5 [4.7, 6.4]    | 0.8           | 2.2      | 0.044        | 5.1  | 0.71    |

**A three-seed paired test.** Because both arms see the same materials, a paired McNemar test on per-structure outcomes, not overlapping Wilson intervals, decides whether two lattice sources differ. Every source was run at three seeds (n = 1000 each); with v22 every arm scores all 3000 structure-seeds (the submitted v21 checkpoint sampled degenerate cells that hung `StructureMatcher`/`spglib`, leaving 1949 scorable learned-head rows). The indexer recovers **42** structures the learned head misses and the learned head recovers **52** the indexer misses (Table 2a).

**Table 2a.** Paired per-structure outcomes, learned head vs indexer control (v22, three seeds pooled; match = `StructureMatcher` hit).

| | indexer **match** | indexer **miss** | row total |
|---|---|---|---|
| **learned match** | 6 | 52 | 58 |
| **learned miss**  | 42 | 2900 | 2942 |
| **col total** | 48 | 2952 | 3000 |

The discordant cells are *b* = 42 (indexer fixes), *c* = 52 (indexer breaks); a two-sided McNemar test gives **p = 0.35**: the two lattice sources are not distinguishable on match rate. The submitted version reported the opposite: with v21 the learned head matched 0 of 1949 scored structure-seeds against the indexer's 46 of 2883, 27 fixed and none broken (p = 1.5 × 10⁻⁸). That lead did not survive retraining under the corrected loss (§3.4): paired on the same seeds, the retrain gains 32 learned-head structures and loses none (p = 4.7 × 10⁻¹⁰), while the indexer arm moves by 12 gained and 10 lost (p = 0.83). v21 and v22 are one training run each, so this attribution rests on the two corrections being the only change between them; the correction and training-seed variance are not separated (§7). The claim that the control improves on the learned head is withdrawn as a general statement.

What survives is a per-system pattern, identified post hoc. The indexer still fixes hexagonal (16 fixed, 0 broken) and tetragonal (7, 2) structures, the systems where its cell error is smallest (0.53 and 0.96 Å; §3.5), but the corrected learned head wins the cubic cases (18 fixed, 47 broken; learned 52 / 714, indexer 23 / 714, oracle 96 / 714), where the indexer's error is the largest among the high-symmetry systems (1.14 Å); orthorhombic cells, indexed as accurately as tetragonal ones, show no gain. Where the indexer does index a cell, its downstream pattern agreement is higher (Pearson 0.41 against 0.18) and its matched structures closer (RMSD median 0.035 against 0.232 Å, over different matched subsets). On the stricter all-correct metric the indexer fixes 6 and breaks 0 (p = 0.03).

What neither lattice source does is close the gap to the oracle: **5.5 % [4.7, 6.4]** at three seeds (165 / 3000; 4.5 % [3.8, 5.3] with the submitted checkpoint, unpaired p = 0.08). Paired against the learned head it fixes 134 structures and breaks 27 (p = 3 × 10⁻¹⁸), and against the indexer the interval is disjoint. This is the effect that replicates across checkpoints and seeds: the cell explains the 1.9 → 5.5 % shortfall, and coordinates cap the rest, since even a perfect lattice leaves match at 5.5 %. The indexer's per-system MAE (1.45 Å overall, dominated by mono 1.76 Å and triclinic NaN) sits above the ~0.5 Å tolerance of §5.5; closing the rest requires fixing the low-symmetry path (GSAS-II remains experimental).

![**Figure 3. Per-system indexer performance.** (a) Strict indexing rate with the crystal system given (blue) and with the system unknown (orange; acceptance rule of §3.5), and the consistent rate with the system given (light blue: the returned cell indexes at least 90 % of the observed lines, whether or not it is the reference cell), on n = 1000; dashed line, overall strict rate with the system given (48.8 %; 43.4 % unknown). (b) Cell-length MAE with the system given; dotted line, the v20 learned head of §7 (1.37 Å). Hexagonal, tetragonal and orthorhombic cells carry mean length errors of 1 Å or less, trigonal and monoclinic errors exceed 1.7 Å, and triclinic cells are not indexed with the system given (two of 39 are matched by a higher-symmetry cell when it is unknown). Removing the crystal system costs a few points in every system except monoclinic and triclinic and never changes the regime.](fig4_indexer_bench.pdf){#fig:indexer}

### 5.3 Head-to-head with external baselines

DiffractGPT (`knc6/diffractgpt_mistral_chemical_formula`) at n = 1000 (990 parseable outputs), PXRDnet (`therealgabeguo/cdvae_xrd_sinc100`) at n = 20 (limited by ~4 GPU-hr/material), and deCIFer (deCIFer_v1, conditioned on XRD + composition) at n = 300 (298 parseable) re-scored on *our* `StructureMatcher` harness at matching tolerances. Crystalyze's checkpoint download is inactive; cite-but-not-reproduced. Unparseable external outputs are excluded from those denominators, whereas PXRD-Diff's own unindexed patterns count as misses (§4).

**Table 3.** Reproduced head-to-head on MP-20 test (StructureMatcher ltol 0.2, stol 0.3, angle_tol 5°). The first PXRD-Diff row is the learned head, which receives the pattern and composition only, as the external models do; the second is the indexer control, additionally supplied with the true crystal system (§3.5). Both are the v22 checkpoint at three sampling seeds (§5.2). n is the number of scored structures; unparseable external outputs are excluded (DGpt 10 of 1000, deCIFer 2 of 300). The rows set the scale of the diagnostic model and are not a competitiveness claim.

| Model                                        | n     | Match % (95 % CI)  | All-correct % (95 % CI) | sg@0.1 % | RMSD med (Å) |
|----------------------------------------------|-------|--------------------|-------------------------|----------|--------------|
| PXRD-Diff (v22, learned head; three seeds pooled) | 3000 | 1.9 [1.5, 2.5] | 0.0 [0.0, 0.1]      | 1.4      | 0.232        |
| PXRD-Diff + indexer control, crystal system given (v22; three seeds pooled) | 3000 | 1.6 [1.2, 2.1] | 0.2 [0.1, 0.4] | 0.3 | 0.035 |
| DiffractGPT (Choudhary, 2025)                | 990   | 18.9 [16.6, 21.4]  | 15.9 [13.8, 18.3]       | 21.3     | 0.001        |
| PXRDnet sinc100 (Guo *et al.*, 2025)         | 20    | 30.0 [14.5, 51.9]  | 5.0 [0.9, 23.6]         | 5.0      | 0.011        |
| deCIFer (Johansen *et al.*, 2025)            | 298   | 73.8 [68.6, 78.5]  | 69.5 [64.0, 74.4]       | 83.6     | 0.009        |

**Match rate** (Fig. 4) ranks deCIFer ≫ {PXRDnet, DGpt} > PXRD-Diff. deCIFer, a concurrent autoregressive CIF-token model conditioned on XRD + composition, is the strongest external reference at 73.8 % match / 69.5 % all-correct (n = 298); its all-correct nearly equals its match, the same CIF-token signature seen in DGpt. PXRDnet and DGpt are not separated at n = 20. PXRD-Diff, at 1.9 %, sits an order of magnitude below all three, as expected for a 3.7 M-parameter diagnostic model. The *all-correct* ordering need not follow match rate: DGpt's all-correct (15.9 % [13.8, 18.3]) sits close to its match, while PXRDnet's drops from 30 % match to 5 % all-correct, i.e. PXRDnet matches structure under loose tolerances but rarely recovers the space group, whereas DGpt's CIF-token output appears to encode symmetry. **We flag this as a hypothesis, not a finding.** At n = 20 the PXRDnet all-correct estimate (1/20) carries a 95 % CI of [0.9, 23.6], which overlaps DGpt's [13.8, 18.3]; confirming the reversal would require n ≈ 100–200. At the looser symprec = 0.2 the gap narrows (PXRDnet 25 %, DGpt 21.4 %), so the divergence may reflect coordinate noise in a diffusion decode defeating tight symmetry detection rather than a failure to recover symmetry. §6 says why it would matter. **RMSD** is heavy-tailed for us and tight-near-zero for both baselines, suggesting our matches are tolerance-stretch rather than coordinate recovery.

**What the gap does and does not reflect.** All four systems are given the composition (§3.1). The learned-head row of PXRD-Diff receives nothing else; the indexer-control row additionally gives its indexer the true crystal system (§3.5), so on prior information that row is favoured. On the pattern itself the asymmetry runs the other way: DiffractGPT is conditioned on a **coarse 300-bin pattern** (2θ ∈ [0, 90°], 0.3° bins, rendered as a `;`-separated string), an order of magnitude lower resolution than PXRD-Diff's 4251-bin 0.02° input, yet it still reaches 18.9 %, so the external models do not win by seeing more of the pattern. The most conspicuous differences are **capacity** (DiffractGPT is a 7 B-parameter large language model; PXRD-Diff is 3.7 M, three orders of magnitude) and **inference compute** (PXRDnet's ~35 k-op latent optimisation vs our single DDIM pass); training data, output representation and any overlap between the external models' training sets and MP-20 test are not controlled here.

![**Figure 4. Head-to-head on one harness.** Match rate, all-correct and pattern Pearson on MP-20 test materials scored through *our* `StructureMatcher` harness; error bars are 95 % Wilson intervals. PXRD-Diff is the v22 checkpoint with its learned lattice head, no prior beyond composition, three sampling seeds pooled (n = 3 × 1000); its indexer control reaches 1.6 % and its true-lattice oracle 5.5 % (§5.2). It sits an order of magnitude below the external systems on match rate; its pattern Pearson (0.18) lies between DGpt's 0.02 and deCIFer's 0.55. The PXRDnet (n = 20) intervals are wide: the DGpt > PXRDnet all-correct ordering is a hypothesis, not an established result (§5.3, §6).](fig5_threeway_headline.pdf){#fig:threeway}

*Caveats.* PXRDnet's n = 20 intervals (roughly ±20 pp) and the preprocessing differences between the three pipelines (identical material_ids, pattern inputs not bit-identical) are bounded in §7.

### 5.4 What did *not* work

Four interventions did not improve match rate under the conditions tested (n ≥ 200); one paragraph each. The two Debye-based interventions ran with the submitted form factor (§5.7(a)), and the rerank additionally with the indexer fallback (§5.7(c)); neither was rerun, so both are recorded as failures of the submitted implementation, not of the corrected one. The Wyckoff and distance-loss runs are single seeds compared against v13, whose 2.51 % §7 shows to be a high draw.

**Top-K Debye rerank (Phase 9.1.4).** Top-5 indexer candidates reranked by Debye loss: match 1.6 % → 1.0 %, all-correct 0.1 % → 0.0 % at n = 1000. The indexer surfaces more wrong cells faster than rerank can filter.

**Debye-gradient guidance during DDIM (Phase 9.2).** Coordinate-channel guidance term $-g \cdot \nabla_{\mathcal{C}} \mathcal{L}_\text{Debye}$, sweep $g \in \{0,0.5,1,2,5\}$ on n = 200: flat 1.0–2.5 % within noise, all-correct 0 % throughout. Sampling-time Debye gradient is too noisy to steer the trajectory, consistent with Segal *et al.* (2026), who show the powder-XRD similarity loss landscape is too non-convex for direct gradient descent.

**Wyckoff-letter embeddings (v15, n = 1000).** Each atom received an `nn.Embedding(27, 256)` token for its `spglib`-assigned Wyckoff letter. A Wyckoff letter is meaningful only within a given space group: the same letter denotes different multiplicities, site symmetries and coordinate constraints in different groups (Dauter & Jaskolski, 2010; de la Flor *et al.*, 2023), so the embedding carried a label, not a constraint, and nothing in the loss or the sampler enforced the fixed or symmetry-related coordinates a Wyckoff position implies. The embedding learns (5404 of 6912 entries non-zero), but match drops from 2.5 % to 2.1 % and lattice prediction destabilises (lattice loss ~0.6 vs ~0.02; Fig. 5), presumably because the inflated atom-feature norm shifts the lattice-pool input distribution. The negative result is narrow: a space-group-agnostic label token does not help and may hurt training dynamics; whether symmetry imposed as a constraint, on the coordinates or on the space group, would help was not tested.

![**Figure 5. Wyckoff embedding destabilises lattice prediction.** Lattice (a) and coordinate (b) training-loss curves for the distance-aux run (v16), the Wyckoff-embedding run (v15) and their combination (v14), one run each. Adding the Wyckoff embedding inflates the lattice loss by an order of magnitude (~0.6 vs ~0.02) while leaving the coordinate loss largely unchanged; the failure is localised to the lattice channel.](fig2_training_curves.pdf){#fig:training}

**Distance-matrix aux loss (v16, n = 1000).** Pairwise MLP predicting periodic distances, $\lambda_\text{dist} = 0.01$ (largest stable). Loss decreases (final ~2.0 vs ~10 random) but match drops 2.5 % → 1.8 %; biases features towards absolute-distance reconstruction rather than the relative updates diffusion needs.

**Combination (v14).** Wyckoff + distance loss together falls to 0.8 % (single seed), below the bare ε baseline; no clean explanation.

### 5.5 Lattice-perturbation study

Best predicted-vs-target Pearson is 0.43 (v13, Table 1); the same model with *correct* coordinates plugged into the differentiable simulator hits 0.97, so the encoder + denoiser leaves about half of the pattern-space agreement on the table. This section measures how much cell error the sampler tolerates; §5.6 asks whether the lattice is the part that is lost.

Perturbing the true cell by Gaussian noise on the lengths (σ_ang = 0; v21, n = 300 per level, single sampling seed; `paper/phase9_results/cell_perturb_sensitivity.json`): at zero error match is 2.0 % (6 / 300) and pattern Pearson 0.52; at a realised length error of 0.40 Å match is 0.33 % (1 / 300) and Pearson 0.05; at 0.80 Å, 0.33 % and 0.04; from 0.99 Å upwards, 0 %. Recovery is therefore not graceful, but with six matches at zero error the match curve cannot resolve a knee; the tenfold fall in pattern agreement by 0.4 Å is the resolved signal, and ~0.5 Å is used throughout as a working tolerance rather than a located threshold. The indexer's per-system errors sit on both sides of it: hexagonal 0.53 Å, tetragonal and orthorhombic 0.96 Å, cubic 1.14 Å, monoclinic 1.76 Å and trigonal 4.62 Å (Fig. 3), which is consistent with the indexer's paired gains falling in the hexagonal and tetragonal systems (§5.2), though orthorhombic cells of equal accuracy show none.

### 5.6 Why the learned lattice fails: neither the head nor the pooling repairs the scale

The natural reading of the 0.007 aux loss, that the lattice is already recoverable from the encoder and only the denoiser is at fault, is testable, and the tests below do not support it. Two experiments (released as `paper/phase5_results/` and `paper/phase5b6_results/`) promote the auxiliary head from a training-time regulariser to the inference-time lattice source and measure what it actually delivers.

**Aux head as lattice predictor (no retrain).** Substituting the v13 aux head's predicted $(a,b,c,\alpha,\beta,\gamma)$ for the sampled lattice channel at n = 1000 gives a length MAE of (1.12, 1.02, 1.38) Å and an angle MAE of (13.0°, 11.9°, 17.6°) at **99.8 %** cell validity, far better conditioned than the diffusion sampler, whose predicted cells were near-0 % valid for v13 (the v22 sampler's cells are all valid, §5.2), but still ~15 % relative length error. End to end this recovers **1.2 %** match (1.3 % with Phase 4 ensembling + coordinate refinement) and 0.0 % all-correct, with pattern Pearson collapsing to **0.013**: a 15 % length error shifts every Bragg reflection by a similar fraction in 2θ, so the simulated pattern no longer overlaps the target.

**Stronger, constrained supervision (retrain).** Bounding the head (softplus lengths, sigmoid-scaled angles), raising the aux weight 0.5 → 5.0, and warm-starting 30 k further steps leaves the MAE essentially unchanged (a ≈ 1.11 Å, α ≈ 14°). No head reading $\mathbf{g}$ recovers the absolute scale. The submitted version read this as a property of the *pooled global encoding*, on the grounds that the `AdaptiveAvgPool1d(1)` collapse §3.2 identified for coordinates would also discard the peak-position fidelity Bragg's-law decoding needs; the controlled ablation below tests that reading and does not support it. The 0.007 figure is a training-batch loss; on test the same head has a 1.3 Å length error (Table 4). A generalisation gap of the encoder features is therefore an alternative to the regression reading below that a frozen-encoder ablation cannot exclude; the head's training-set MAE, not reported here, would separate them, and a direct comparison with the convolutional lattice-parameter regressor of Chitturi *et al.* (2021) on comparable cells is the natural external test.

This reconciles the diagnosis with the indexer result. The 0.007 normalised aux loss is a *relative* fit, enough to place the crystal system (a space-group head on the same encoder reaches 39 % top-1 / 71 % top-5 across 230 classes, released under `phase5b6_results/`) but not the absolute scale. The autoindexer fixes the hexagonal and tetragonal cases, where peak positions over-determine the cell; we read the contrast with the regression heads as a difference between explicit index assignment and pointwise regression (a hypothesis), consistent with the head-side fixes of Phase 5 here and the top-K rerank of §5.4 not helping. The ablation below asks whether it is the summary, rather than the regression, that fails.

**Controlled pooling ablation.** The two experiments above vary the head and its supervision with the encoder fixed. They do not by themselves isolate global average pooling as the cause: the encoder, the information content of its representation, the head parameterisation and optimisation difficulty remain confounded, as one referee noted. To separate them, a controlled ablation holds the encoder, the denoiser, the training objective and the schedule fixed and changes only how the auxiliary head reads the encoder: (i) the global average pool $\mathbf{g}$ of the submitted version; (ii) a position-aware pooling that retains the 2θ coordinate of each feature; (iii) explicit peak-position features extracted from the input pattern and concatenated to $\mathbf{g}$. If absolute d-spacings are lost at the pooling stage, variants (ii) and (iii) should reduce the ~1.1 Å length error while (i) does not. Table 4 gives the result on both encoders.

**Table 4.** Controlled pooling ablation (n = 1000 MP-20 test). Three heads trained under an identical loss, AdamW 10⁻³, batch 64, sigmoid-bounded outputs, on a frozen encoder: the submitted v21 encoder at three epochs and the retrained v22 encoder (§5.2) at ten, one training seed per head. Length and angle MAE are means over (a, b, c) and (α, β, γ); "< 0.5 Å" is the fraction of structures whose mean length error lies below the ~0.5 Å tolerance of §5.5; the last column substitutes the head's cells into the sampler at three seeds (n = 3000 structure-seeds; the full pipeline reaches 1.9 % and the indexer 1.6 %, Table 2). The checkpoint rows are the head trained jointly for 100 k steps, without retraining.

| Head | Params | Encoder, epochs | Length MAE / Å | Angle MAE / ° | Median length error / Å | < 0.5 Å | End-to-end match % [95 % CI] |
|---|---|---|---|---|---|---|---|
| Checkpoint aux head | – | v21, – | 1.32 | 15.6 | 0.90 | 0.33 | – |
| (i) global average pool $\mathbf{g}$ | 67 k | v21, 3 | 1.29 | 16.0 | 0.87 | 0.29 | – |
| (ii) position-aware attention pooling | 531 k | v21, 3 | 1.31 | 16.4 | 0.94 | 0.24 | – |
| (iii) explicit peak features + $\mathbf{g}$ | 209 k | v21, 3 | 1.28 | 16.1 | 0.88 | 0.28 | – |
| Checkpoint aux head | – | v22, – | 1.28 | 15.0 | 0.90 | 0.31 | – |
| (i) global average pool $\mathbf{g}$ | 67 k | v22, 10 | 1.29 | 15.5 | 0.91 | 0.28 | 0.7 [0.5, 1.1] |
| (ii) position-aware attention pooling | 531 k | v22, 10 | 1.21 | 15.4 | 0.88 | 0.29 | 0.5 [0.3, 0.9] |
| (ii) position-aware attention pooling | 531 k | v22, 30 | 1.12 | 14.6 | 0.75 | 0.36 | 0.8 [0.5, 1.2] |
| (iii) explicit peak features + $\mathbf{g}$ | 209 k | v22, 10 | 1.28 | 15.4 | 0.90 | 0.30 | 1.2 [0.9, 1.7] |

The prediction is not borne out on either encoder. On v21 the three arms lie within 0.03 Å of one another (~2 % of the error). On v22, with ten epochs, arms (i) and (iii) converge by the seventh epoch to 1.29 and 1.28 Å, and arm (ii) reaches 1.21 Å at ten epochs and 1.12 Å at thirty, where it is converged (flat over the last six epochs), a 13 % reduction. Every arm sits more than 2.2× above the ~0.5 Å tolerance of §5.5, and arm (i) reproduces the checkpoint head on both encoders, so the budget suffices to match the baseline. Each head is one training seed, so differences of a few hundredths of an ångström between arms are not resolved.

End to end, the substituted cells recover 0.7 %, 0.5 % and 1.2 % for (i), (ii) and (iii), each below the 1.9 % of the full pipeline (paired McNemar p = 7 × 10⁻⁷, 3 × 10⁻⁹ and 8 × 10⁻³) and the indexer's 1.6 %, with pattern Pearson 0.08 in every arm. The head-level gain of arm (ii) does not carry through: its converged thirty-epoch cells recover 0.8 % [0.5, 1.2] (24 / 3000, 22 of them cubic), level with arm (i). Nearly every match obtained through a regression head is a cubic structure (18 of 22, 16 of 16 and 34 of 37), the one-parameter case; none is hexagonal and one (in the thirty-epoch arm) is tetragonal, the systems where the indexer's search succeeds (§5.2).

Restoring absolute peak positions to the head therefore does not repair the absolute scale, and the submitted reading that global pooling discards the d-spacing information is withdrawn. Arm (iii) is the informative one: it receives the same peak positions the autoindexer reads, yet a head trained to regress six cell parameters did not turn them into a cell here. We read this as the peak-to-index assignment, a combinatorial step the autoindexer performs by explicit search, not being learned by a pointwise-loss regressor. This reading is the best-supported one, not a demonstrated cause: it accounts for every head-side failure (the two experiments above, the top-K rerank of §5.4), for the space-group head's success on relative features and for the indexer succeeding exactly where the assignment is over-determined (§5.2), but the generalisation-gap alternative raised above is not excluded. On this evidence the more promising route to the cell is search, or a model that performs the assignment explicitly, rather than a different pooling or a better head; once the cell is fixed, the denoiser is the limiter (§5.2).

Three caveats remain: the encoder is frozen, so an encoder trained from scratch under position-aware pooling could behave differently (not run); the heads are trained on the frozen encoder's features rather than jointly, so a jointly trained position-aware head is untested; and each arm is a single training seed. Runs: `paper/phase15_results/pooling_ablation_v21/`, `pooling_ablation_v22/`, `v22_pool_*`.

### 5.7 Code audit and corrections

Review of the released code, prompted by two referees, identified three implementation errors and, in the systematic check the referees asked for, two discrepancies between the text and the released records (items d and e). The check compared each equation of §3 with the function implementing it, each setting of §4 with the arguments stored in the released checkpoints, and each quoted result with its released record. Each item is listed with its effect on the numbers in this paper.

(a) *Atomic form factor.* The differentiable simulator evaluated $\sum_k a_k \exp(-b_k s^2)$ with coefficients fitted for $Z - 41.78214\,s^2 \sum_k a_k \exp(-b_k s^2)$ (§3.3). Affected: the Debye training loss in every run with $\lambda_\text{Debye} > 0$ (v11, v13–v16, v21); the Debye-gradient guidance and top-K rerank of §5.4; and the validation figure. Not affected: all match, all-correct, space-group and RMSD numbers, which `pymatgen` scores on the predicted structures; the indexer; the oracle; the perturbation study; the aux-head substitution of §5.6; and the external baselines. The module was corrected and re-validated (mean Pearson 0.952 → 0.988 over 1000 structures). The production checkpoint was retrained from scratch with the corrected module and the fix of (b), under otherwise identical settings (v22; final EMA loss 0.602 against 0.726); Tables 2, 2a and 3 and Fig. 4 now report it at three seeds, and §5.2 gives the paired before/after accounting (learned head 0 / 1949 → 58 / 3000, indexer 46 / 3000 → 48 / 3000, oracle 135 → 165 / 3000; the indexer's lead became a tie). The Phase-4 rows of Table 1 and the §5.4 experiments were not rerun (~18 GPU-hours for the six configurations at three seeds); they keep their exploratory labels.

(b) *Debye loss in x₀-residual mode.* The loss was evaluated on the residual tensor rather than on the clean-coordinate estimate (§3.4). Affected: the Debye term of runs v13–v16 and v21. Not affected: the ε-mode runs v10 and v11, and every evaluation-time number. Fixed; the retrained runs of (a) also carry this fix.

(c) *Indexer fallback.* When the classical indexer returned no cell, the sampling script substituted the true lattice and the pattern was scored as if indexed. This affected 39 of the 1000 test patterns in every indexer run, 117 structure-seeds over three seeds. Recomputing from the released per-structure records, none of the 117 fallback structure-seeds matched, so the submitted McNemar cells (27 fixed, 0 broken) and the all-correct cells (4, 0) were unchanged by the fallback. The submitted match rate becomes 46 / 2883 = 1.6 % [1.2, 2.1] on the indexed structure-seeds, in place of 46 / 3000 = 1.5 % [1.2, 2.0]. The v21 checkpoint was not re-sampled under the corrected script; these figures are recomputed from the records of the submitted runs, and they are superseded by the v22 retrain of (a), which is what §5.2 now reports. The script now scores unindexed patterns as misses and records their count. The top-K rerank experiment of §5.4 used the same fallback; because the fallback rows did not match, its 1.0 % result can only be lower, and its conclusion (no improvement) stands.

(d) *Reported learning rate.* The submitted §4 gave the learning rate as $5\times10^{-4}$. The arguments stored in the released v21 checkpoint give $10^{-3}$, which is the value that was used; §4 now says so. No result depends on this, but the referees asked for a systematic check of the description against the implementation and this is what it found.

(e) *Perturbation-study figures.* The submitted §5.5 quoted match rates (5.6 → 2.0 → 0.5 → 0 % at n = 200) that do not correspond to the released record of that experiment, which is n = 300 per level with 6, 1, 1 and 0 matches (§5.5); the 5.6 % was the single-seed oracle figure later replaced by the three-seed 4.5 %. §5.5 now reports the record, and the "knee" of the submitted version is reworded as a working tolerance. No other number depends on it.

The corrected code, the per-structure records used for (c), the re-validation script and a regression test that pins the form-factor convention (`tests/test_debye_form_factor.py`) are in the released repository.

---

## 6. Discussion

**Lattice-recovery bottleneck (the robust result).** A 0.007 *normalised* aux loss shows $\mathbf{g}$ carries lattice-*relevant* structure, but it is a training-batch, relative fit: feeding the aux head's own lattice to the sampler recovers only 1.2 % (§5.6), and neither stronger constrained supervision nor restoring absolute peak positions to the head reduces its 1.1–1.3 Å test error by more than 13 %, so the pooling reading of the submitted version is withdrawn in favour of the regression formulation, offered as a hypothesis beside a generalisation-gap alternative (§5.6). The paired oracle-vs-learned-head test (134 fixed, 27 broken, p = 3 × 10⁻¹⁸; §5.2) is the paper's statistically firmest claim and the one result that replicates across the submitted and the corrected checkpoint. The Q-space autoindexer, which reads peak positions directly, recovers part of the 5.5 % oracle ceiling on the hexagonal and tetragonal subset; with the corrected checkpoint it no longer leads the learned head in aggregate, but where it indexes a cell its downstream pattern agreement is higher (Pearson 0.41 against 0.18). Top-K rerank failed because the then-defective Debye loss did not separate the correct cell from the indexer's alternatives, and gradient guidance failed on a loss landscape too rough for descent (Segal *et al.*, 2026); neither was rerun with the corrected simulator (§5.7). The oracle ceiling is itself only 5.5 %, so even a perfect lattice leaves coordinates as a co-limiter (§8).

**Per-atom anchoring is hard.** PXRD is permutation-invariant; our denoiser is permutation-equivariant; there is no symmetry-breaking signal that pins atom $i$ to a specific Wyckoff site. The Wyckoff-letter embedding tried to break symmetry at the input with a label rather than a constraint, and failed (§5.4). A promising untried direction: break symmetry at the *output*: predict an unordered set of orbits plus a Hungarian-style matcher.

**A measurement hypothesis: match rate may conflate two capabilities.** PXRDnet's 30 % match / 5 % all-correct vs DGpt's 18.9 % / 15.9 % (Table 3) suggests the two systems recover *structure* at broadly comparable rates while differing sharply in *symmetry* recovery: DGpt's CIF-token output appears to encode space group in a way PXRDnet's coordinate decode does not. The numbers cannot yet support this (PXRDnet's all-correct CI [0.9, 23.6] at n = 20 overlaps DGpt's), so we raise it as a *measurement hypothesis* worth testing at adequate n. If true, a single "match rate" reported under inconsistent tolerances would conflate structure recovery with symmetry recovery, and downstream crystallography, where the space group matters as much as the coordinates, would be mis-served by it. The contribution here is the shared harness that makes such a test possible, not the underpowered comparison itself.

**Niche for small reproducible models.** At 1.9 % PXRD-Diff sits an order of magnitude below much larger systems, and that is the point of its size. A small model makes the lattice-recovery failure visible and the indexer control trivial to wire in; it trains in under two hours and runs inference in seconds, so every experiment here can be repeated.

---

## 7. Limitations

- **Scope.** Composition given; simulated PXRD only (no instrument response, preferred orientation, asymmetry, background); MP-20 only, with no larger cells, organics, or higher-Z.
- **Statistics.** The Phase 4 ablations in Table 1 are single-seed. A multi-seed checkpoint sweep of the ε-vs-x₀-residual contrast (two seeds × five checkpoints, 20 k–100 k, true-lattice n = 1000) shows the reported 2.8× *match-rate* lift does **not** replicate: pooled match is 1.20 % (ε) vs 1.35 % (x₀-residual), and x₀ never exceeds 1.5 % at any checkpoint, including the 80 k checkpoint that matches the original v13's 79.5 k, so the single-seed 2.51 % was a high draw, not a checkpoint-selection effect. We therefore demote the *match-rate* lift to a single-seed artefact. What does reproduce is the pattern-Pearson improvement (ε 0.359 vs x₀ 0.403, with disjoint ranges across both seeds), consistent with Table 1. The lattice-input fix remains load-bearing as a training-dynamics result (lattice loss 1.0 → 0.02), and the oracle-vs-learned-head gap is the effect that replicates (§5.2). The indexer-vs-learned-head comparison is the cautionary case: significant with the submitted checkpoint (27 fixed, 0 broken), a tie with the corrected one (42, 52; p = 0.35), so a paired test at three sampling seeds protects against sampling noise but not against a training defect, or a training-seed draw, that moves one arm (§5.2, §5.7). Each checkpoint (v21, v22) is a single training run; the three seeds of §5.2 and §5.6 are sampling seeds, so training-seed variance of the production model was not measured.
- **Baselines.** PXRDnet n = 20 has a 95 % Wilson CI of ≈ ±20 pp; the DGpt > PXRDnet all-correct ordering (§5.3/§6) is therefore a hypothesis, not a result. n = 200 needs ~33 days RTX 5090. Crystalyze checkpoint download is inactive (verified 2026-06-01); cited but unreproduced. Patterns are not bit-identical across the three preprocessors (PXRD-Diff 4251-bin 2θ, DGpt 300-bin 2θ, PXRDnet 4096-bin Q with sinc² broadening): structures match, patterns do not; we did not quantify the residual preprocessing effect on match rate, and a matched-vs-native re-scoring on a structure subset is the natural check. Parse failures are excluded from the external denominators (DGpt 10 of 1000, deCIFer 2 of 300), whereas PXRD-Diff's unindexed patterns count as misses; overlap between the external models' training sets and MP-20 test was not assessed.
- **Indexer realism.** Two assumptions make the indexer's accuracy an optimistic ceiling. (i) *Crystal system.* The main runs supply the true Bravais code to the de-Wolff fit (§3.5); the unknown-system run removes it at a cost of about five percentage points at the indexing stage (48.8 → 43.4 % strict) and six of 48 downstream matches (1.4 % against 1.6 %), so the assumption changes little at either stage. (ii) *Patterns are ideal.* All patterns are clean `pymatgen` simulations with no zero-point shift, sample-displacement error, or peak asymmetry, precisely the perturbations that drive real-world indexing failure. The 48.8 % native-indexer match and the downstream 1.6 % should be read as upper bounds under ideal peak positions. A zero-shift/displacement robustness sweep (n = 300, v20 checkpoint, run before the audit) confirms the concern: the indexer's length-MAE rises from 1.24 Å on those clean patterns (a 300-structure subset; 1.45 Å on the full n = 1000) past the v20 learned head's 1.37 Å once displacement or zero-shift exceeds ~0.10°, so the indexer's exact cells on high-symmetry patterns depend on well-calibrated input.
- **Idealised conditions and the bottleneck conclusion.** Every pattern is an ideal simulation. Instrument profile, zero shift, sample displacement, preferred orientation and background all move or reshape peaks. Each would further degrade an encoder that already fails to fix absolute d-spacings on clean input. The lattice-recovery bottleneck diagnosis is therefore a lower bound on the difficulty of real data, not an estimate of it. The ~0.5 Å tolerance is a property of the sampler under clean cells; under real peak shifts it may move, and the indexer's own degradation past ~0.10° suggests the whole curve shifts towards smaller tolerable errors.
- **Perturbation study and ablation heads.** The perturbation study (§5.5) is v21, n = 300 per level, single sampling seed, and was not rerun on v22; the Table 4 heads are one training seed each with no interval on their MAE; the 0.007 aux loss is a training-batch value and the head's training-set error was not measured; an encoder trained from scratch under position-aware pooling was not run.
- **Match metric.** `StructureMatcher` is run with `scale=True`, so the headline match rate is invariant to a uniform scaling of the cell and tests cell shape and coordinates; the absolute scale that the lattice discussion concerns enters the results through pattern Pearson and $R_{wp}$, and through the anisotropic errors of the perturbation study.
- **Indexer (low-symmetry path).** GSAS-II low-symmetry path hangs in `findBestCell` on real MP-20 mono/tri patterns; shipped behind `--use-gsas` but experimental.
- **Training.** Debye loss uses ground-truth lattice; a curriculum gradually replacing true with predicted was not tried.

---

## 8. Conclusion

A 3.7 M-parameter conditional diffusion model, used as a diagnostic instrument, locates the largest correctable failure in this model's PXRD inversion in the recovery of the unit cell from the pattern, upstream of the denoiser. The encoder's pooled representation supports a 0.007 normalised lattice-regression training loss yet delivers a 1.1–1.3 Å absolute cell error that no head reading it reduces by more than 13 %, whether constrained, position-aware or handed explicit peak positions; the pooling reading is withdrawn, and we read the failure as one of the regression formulation, which leaves the peak-to-index assignment unsolved (a hypothesis, §5.6). A true-lattice oracle caps match at 5.5 % [4.7, 6.4] (three sampling seeds) against 1.9 % [1.5, 2.5] for the learned head (paired McNemar p = 3 × 10⁻¹⁸): the cell explains the 1.9 → 5.5 % shortfall, and coordinates cap the rest. A classical Q-space autoindexer, supplied with the true crystal system, serves as a control: it reaches 1.6 % [1.2, 2.1] and, with the corrected checkpoint, is not distinguishable from the learned head at three seeds (42 fixed, 52 broken, p = 0.35), its gains confined to the hexagonal and tetragonal systems where its cell error is smallest. Re-scored on one `StructureMatcher` harness, deCIFer reaches 73.8 % / 69.5 %, DGpt 18.9 % / 15.9 % and PXRDnet 30.0 % / 5.0 % (match / all-correct); these set the scale of the diagnostic model and are not a ranking claim, and the apparent divergence in their space-group recovery is a measurement hypothesis the n = 20 PXRDnet sample cannot yet confirm. A code audit corrected three implementation errors; retraining the production checkpoint under the corrected loss raised the learned head from 0 to 1.9 % and removed the indexer's aggregate lead of the submitted version, leaving the oracle gap as the result that replicates. The lattice-input fix is load-bearing; the x₀-residual switch improves pattern-Pearson but its single-seed match-rate lift did not replicate (§7); the remaining interventions did not improve match rate under the conditions tested (§5.4). Code, checkpoints, all per-phase JSONs, per-structure records, both indexer paths and the differentiable Bragg module are released.

---

## Acknowledgements

*Use of AI tools.* The author used a large language model (Claude, Anthropic) as a coding and writing assistant throughout this work, for drafting and editing text and code under the author's direction. The author reviewed all code, analyses and text, and takes full responsibility for their content.

We thank the maintainers of `pymatgen`, `spglib`, the CDVAE benchmark, GSAS-II, and the upstream maintainers of DiffractGPT (`atomgptlab/atomgpt`), PXRDnet (`gabeguo/cdvae_xrd`) and deCIFer for releasing checkpoints and code that made the head-to-head reproduction in §5.3 possible. Compute was rented from Vast.ai; total spend was approximately USD 28 across roughly 40 GPU-hours on RTX 5090 instances (Phase 4 ablation, Phase 9 retrain + indexer sweeps, DiffractGPT n = 1000 inference, PXRDnet n = 20 inference, and the revision's retrain, pooling-ablation and unknown-system runs of §5.2, §5.6 and §3.5, about 5.5 GPU-hours).

## Author Contributions (CRediT)

F. Cai: Conceptualization, Methodology, Software, Validation, Formal analysis, Investigation, Data curation, Writing – Original Draft, Writing – Review & Editing, Visualization, Project administration.

## Conflict of Interest

The author declares no competing interests.

## Funding

This research received no external funding.

## Data and Code Availability

Source code, trained checkpoints, and all per-phase training/evaluation logs are openly available in the GitHub repository at https://github.com/fronkt/pxrd-diff and archived at Zenodo under the concept DOI [10.5281/zenodo.20738993](https://doi.org/10.5281/zenodo.20738993), which resolves to the latest version; the version archived for this revision (source at the revised commit, the corrected v22 checkpoint, the submitted v21 checkpoint and the pooling-ablation heads) is [10.5281/zenodo.22973311](https://doi.org/10.5281/zenodo.22973311). The MP-20 dataset is publicly available via the CDVAE benchmark. All experiments reproduce from a single requirements.txt and the scripts/ pipeline; per-structure evaluation flags are released to support paired re-analysis. The corrected simulator, the audit scripts of §5.7 and the per-structure records behind Table 2a are included.

## Ethics Declaration

This study uses no human subjects, animal subjects, or sensitive data. The MP-20 dataset is composed entirely of publicly available crystal structures from the Materials Project.

---

## References

Batatia, I., Kovács, D. P., Simm, G. N. C., Ortner, C. & Csányi, G. (2022). MACE: higher order equivariant message passing neural networks for fast and accurate force fields. *Advances in Neural Information Processing Systems*, Vol. 35.

Boultif, A. & Louër, D. (2004). *J. Appl. Cryst.* **37**, 724–731. DOI: 10.1107/S0021889804014876.

Chitturi, S. R., Ratner, D., Walroth, R. C., Thampy, V., Reed, E. J., Dunne, M., Tassone, C. J. & Stone, K. H. (2021). *J. Appl. Cryst.* **54**, 1799–1810. DOI: 10.1107/S1600576721010840.

Choudhary, K. (2025). *J. Phys. Chem. Lett.* **16**, 2110–2119. DOI: 10.1021/acs.jpclett.4c03137. Reproduced from HF checkpoint `knc6/diffractgpt_mistral_chemical_formula` and code at `github.com/atomgptlab/atomgpt`.

Cuocci, C., Corriero, N., Dell'Aera, M., Falcicchio, A., Rizzi, R. & Altomare, A. (2022). *Comput. Mater. Sci.* **210**, 111465. DOI: 10.1016/j.commatsci.2022.111465.

Dauter, Z. & Jaskolski, M. (2010). *J. Appl. Cryst.* **43**, 1150–1171. DOI: 10.1107/S0021889810026956.

de la Flor, G., Kroumova, E., Hanson, R. M. & Aroyo, M. I. (2023). *J. Appl. Cryst.* **56**, 1824–1840. DOI: 10.1107/S1600576723009068.

de Wolff, P. M. (1968). *J. Appl. Cryst.* **1**, 108–113. DOI: 10.1107/S002188986800508X.

Favre-Nicolin, V. & Černý, R. (2002). *J. Appl. Cryst.* **35**, 734–743. DOI: 10.1107/S0021889802015236.

Guo, G., Saidi, T. L., Terban, M. W., Valsecchi, M., Billinge, S. J. L. & Lipson, H. (2025). *Nat. Mater.* **24**, 1726–1734. DOI: 10.1038/s41563-025-02220-y (Author Correction: 10.1038/s41563-025-02301-y; preprint arXiv:2406.10796). Reproduced from HF checkpoint `therealgabeguo/cdvae_xrd_sinc100` and code at `github.com/gabeguo/cdvae_xrd`.

Guo, J. & Schwalbe-Koda, D. (2026). Generative inversion of spectroscopic data for amorphous structure elucidation. arXiv:2603.23210.

Jiao, R., Huang, W., Lin, P., Han, J., Chen, P., Lu, Y. & Liu, Y. (2023). Crystal structure prediction by joint equivariant diffusion. *Advances in Neural Information Processing Systems*, Vol. 36.

Johansen, F. L., Friis-Jensen, U., Dam, E. B., Jensen, K. M. Ø., Mercado, R. & Selvan, R. (2025). deCIFer: crystal structure prediction from powder diffraction data using autoregressive language models. *Transactions on Machine Learning Research* (arXiv:2502.02189).

Lai, Q., Xu, F., Yao, L., Gao, Z., Liu, S., Wang, H., Lu, S., He, D., Wang, L., Zhang, L., Wang, C. & Ke, G. (2025). *Adv. Sci.* **12**, 2410722. DOI: 10.1002/advs.202410722.

Li, Q., Jiao, R., Wu, L., Zhu, T., Huang, W., Jin, S., Liu, Y., Weng, H. & Chen, X. (2025). *Nat. Commun.* **16**, 7428. DOI: 10.1038/s41467-025-62708-8.

Nichol, A. & Dhariwal, P. (2021). Improved denoising diffusion probabilistic models. *Proceedings of the 38th International Conference on Machine Learning*, PMLR **139**, 8162–8171.

Parackal, A. S., Goodall, R. E. A., Faber, F. A. & Armiento, R. (2024). *Phys. Rev. Mater.* **8**, 103801. DOI: 10.1103/PhysRevMaterials.8.103801.

Riesel, E. A., Mackey, T., Nilforoshan, H., Xu, M., Badding, C. K., Altman, A. B., Leskovec, J. & Freedman, D. E. (2024). *J. Am. Chem. Soc.* **146**, 30340–30348. DOI: 10.1021/jacs.4c10244. Code: `github.com/ML-PXRD/Crystalyze`. We could not reproduce: the checkpoint download link is marked "not yet active" in the upstream README (verified 2026-06-01).

Schütt, K. T., Kindermans, P.-J., Sauceda, H. E., Chmiela, S., Tkatchenko, A. & Müller, K.-R. (2017). SchNet: a continuous-filter convolutional neural network for modeling quantum interactions. *Advances in Neural Information Processing Systems*, Vol. 30.

Segal, N., Subramanian, A., Li, M., Miller, B. K. & Gómez-Bombarelli, R. (2026). *Digital Discovery* **5**, 1590–1599. DOI: 10.1039/D6DD00017G. (Preprint: arXiv:2512.04036.)

Song, J., Meng, C. & Ermon, S. (2021). Denoising diffusion implicit models. *International Conference on Learning Representations*.

Toby, B. H. & Von Dreele, R. B. (2013). *J. Appl. Cryst.* **46**, 544–549. DOI: 10.1107/S0021889813003531. Code: `github.com/AdvancedPhotonSource/GSAS-II`.

Werner, P.-E., Eriksson, L. & Westdahl, M. (1985). *J. Appl. Cryst.* **18**, 367–370. DOI: 10.1107/S0021889885010512.

Xie, T., Fu, X., Ganea, O.-E., Barzilay, R. & Jaakkola, T. (2022). Crystal diffusion variational autoencoder for periodic material generation. *International Conference on Learning Representations*.

Yu, D., Zhu, Z., Leng, F. & Zhu, Y. (2026). *Nat. Commun.* **17**, 3274. DOI: 10.1038/s41467-026-70035-9.

Zeni, C., Pinsler, R., Zügner, D., Fowler, A., Horton, M., Fu, X., Wang, Z., Shysheya, A., Crabbé, J., Ueda, S., Sordillo, R., Sun, L., Smith, J., Nguyen, B., Schulz, H., Lewis, S., Huang, C.-W., Lu, Z., Zhou, Y., Yang, H., Hao, H., Li, J., Yang, C., Li, W., Tomioka, R. & Xie, T. (2025). *Nature* **639**, 624–632. DOI: 10.1038/s41586-025-08628-5.


---

## Appendix

### A. Full ablation history

For completeness, Table A1 lists every training run discussed in the development of this paper, including those that did not make it into the main ablation table. Logs and checkpoints for all runs are in the released repository under `runs/`.

| Run | Parameters | What changed | Result | Status in paper |
|-----|-----------|-------------|--------|----------------|
| v4  | 3.5 M | Global PXRD pooling, additive conditioning | Coord loss flat at 3.0 | §3.2 |
| v5  | 3.7 M | + Multi-resolution cross-attention | Coord 3.0 → 1.0 | §3.2 |
| v6–v9 | 3.7 M | λ_Debye sweep {0, 0.1, 1, 10}, ε-prediction | All ~1 % match (within noise) | Appendix B |
| v10 | 3.7 M | + Lattice-input fix, λ_Debye = 0 | Lat loss 1.0 → 0.05 | Table 1 |
| v11 | 3.7 M | + Lattice-input fix, λ_Debye = 1 | Match 0.9 % | Table 1 |
| v12 | 10.1 M | Larger model (d = 384, L = 5), ε-prediction | Stopped at 18 k; same plateau | §5.1 |
| v13 | 3.7 M | x₀-residual + lat-fix + Debye λ = 1 | **2.51 % match (Phase 4 best, true-lat)** | Table 1 |
| v14 | 3.8 M | v13 + Wyckoff + distance loss | 0.80 % match | Table 1 |
| v15 | 3.8 M | v13 + Wyckoff only | 2.10 % match | Table 1 |
| v16 | 3.7 M | v13 + distance loss only | 1.80 % match | Table 1 |
| v17 | 3.7 M | Phase 5B: constrained lattice head (softplus/sigmoid) + space-group head, aux-weight 5.0, +30 k warm-start steps | Lattice MAE unchanged (a ≈ 1.11 Å); SG head 39 % top-1 | §5.6 |
| v18 – v20 | 3.7 M | Phase-9 encoder retrains with different ResNet/Transformer hybrids and pattern-augmentation curricula | None beat v13 by more than noise | §7 (v20) |
| v21 | 3.7 M | Phase 9 final: v13 architecture + Phase 9 retrain + indexer-control support (submitted version; trained with the §5.7 (a) and (b) defects) | 0 % (learned head), 1.6 % (indexer, crystal system given), 4.5 % (true-lattice oracle), all 3-seed | §5.7 |
| **v22** | **3.7 M** | v21 configuration retrained from scratch with the corrected form factor and the corrected x₀-mode Debye loss (§5.7) | **1.9 % (learned head), 1.6 % (indexer, crystal system given), 5.5 % (true-lattice oracle), all 3-seed** | Tables 2, 2a, 3 |

### B. Hyperparameter sensitivity

We did not perform a full hyperparameter sweep. Pilot experiments on `d_model ∈ {128, 256, 384}` and `n_layers ∈ {2, 3, 5}` showed (256, 3) as a reasonable Pareto point. The Debye loss weight was swept {0, 0.1, 1, 10} in early ε-prediction runs (v6–v9) without measurable effect on match rate; we re-fixed it at 1.0 for the x₀-residual runs by analogy.

### C. Differentiable simulator validation details

Pearson correlations between the DiffPXRD module and `pymatgen.XRDCalculator` on the 256-bin coarse grid, first 1000 MP-20 test structures (per-structure values in `paper/submissions/JAC-R1/analysis/simulator_revalidation.json`; script `revalidate_simulator.py`). "Submitted" is the bare Gaussian sum of the submitted version; "corrected" is the pymatgen convention of §3.3. The $\le 5$ range is the training configuration; the $\le 10$ range is the default of `scripts/04_verify_debye.py`, behind the 0.962 reported in the submitted version.

| Reflection range | Expression | n | Mean | Median | Min | Fraction > 0.9 |
|---|---|---|---|---|---|---|
| $|h|,|k|,|l| \le 5$ | submitted | 1000 | 0.952 | 0.969 | 0.23 | 0.907 |
| $|h|,|k|,|l| \le 5$ | corrected | 1000 | 0.988 | 0.999 | 0.25 | 0.980 |
| $|h|,|k|,|l| \le 5$ | submitted | 50 | 0.939 | 0.971 | 0.49 | 0.90 |
| $|h|,|k|,|l| \le 5$ | corrected | 50 | 0.974 | 0.998 | 0.52 | 0.92 |
| $|h|,|k|,|l| \le 10$ | submitted | 1000 | 0.960 | 0.972 | 0.54 | – |
| $|h|,|k|,|l| \le 10$ | corrected | 1000 | 0.999 | 1.000 | 0.58 | – |
| $|h|,|k|,|l| \le 10$ | submitted | 50 | 0.963 | 0.972 | 0.90 | – |
| $|h|,|k|,|l| \le 10$ | corrected | 50 | 0.999 | 1.000 | 0.98 | – |

The corrected expression scores at least as high as the submitted one on every one of the 1000 structures. Gradients through the structure factor with respect to fractional coordinates were verified non-zero and finite. The regression test `tests/test_debye_form_factor.py` pins $f(0) = Z$, agreement with the `pymatgen` expression at further $(Z, s)$ points, rejection of the legacy convention, and agreement with `XRDCalculator` on NaCl (Pearson > 0.99).

### D. Reproducibility checklist

- [x] Hyperparameters specified (§4, Implementation)
- [x] Datasets and splits specified (§4, Dataset; canonical CDVAE MP-20)
- [x] Evaluation protocol specified (§4, Evaluation)
- [x] Random seeds: training seed 42 for every run (one training run per configuration); sampling seeds 0, 1, 2 for §5.2, §5.6 and Table 3; pilot variance ≈ 0.5 % match-rate absolute
- [x] Compute environment: PyTorch 2.x, CUDA 12.x, single RTX 5090; Python 3.12
- [x] Code, checkpoints, logs, and full training scripts released
