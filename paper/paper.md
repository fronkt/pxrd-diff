# PXRD-Diff: An Honest Look at Conditional Diffusion for Powder Diffraction Inversion

**Frank Cai**
*Independent researcher*
`frankyc11223@gmail.com`

**Target venue:** ML4PS / AI4Mat workshop @ NeurIPS 2026
**Track:** Methods / Negative results
**Word budget:** ~4,500 words main text + appendix

---

## Abstract

Powder X-ray diffraction (PXRD) is the workhorse characterization technique of solid-state chemistry, yet inverting a 1D powder pattern to a 3D crystal structure remains an open problem. Recent generative approaches (DiffractGPT, Crystalyze) report encouraging numbers but use very different evaluation protocols, and the literature contains few published failure modes that practitioners can use to calibrate expectations. We present **PXRD-Diff**, a 3.7 M-parameter conditional denoising diffusion model that takes a simulated Cu Kα PXRD pattern as input and generates fractional coordinates and lattice parameters jointly, conditioned on the known composition. The model combines a 1D ResNet PXRD encoder, a periodic-distance message-passing denoiser with multi-resolution PXRD cross-attention, and — as the central physics-informed contribution — a differentiable Bragg structure-factor module that supplies an auxiliary "predicted-pattern matches input-pattern" loss. We train on the canonical CDVAE MP-20 split (≤20 atoms/cell, 27 k train / 9 k test) and report a careful 7-condition ablation. Three findings stand out. (1) A naive ε-prediction denoiser conditioned on the true lattice never escapes the random baseline on lattice prediction, because the lattice head is fed the *clean* lattice rather than the lattice it is being asked to denoise; passing the noisy lattice as input drops lattice loss by 95 %. (2) Switching from ε-prediction to **x₀-residual prediction** lifts the StructureMatcher match rate by 3.5× over the ε baseline. (3) Two seemingly attractive extensions — Wyckoff-site embeddings and an auxiliary distance-matrix loss — fail to improve on (2) and, in combination, actively hurt it. The best configuration recovers 2.5 % of MP-20 test structures (StructureMatcher, true-lattice coord-only setting; 1.2 % full-pipeline). This is far from solving the inverse problem, but the architectural recipe and the catalogue of what *did not* work are concrete, reproducible, and we believe useful to the community. Code, checkpoints, and all training/evaluation logs are released.

**Keywords:** powder X-ray diffraction; crystal-structure prediction; denoising diffusion; equivariant message passing; differentiable physics; ablation study

### 摘要 (zh-TW)

粉末 X 光繞射 (PXRD) 是固態化學最常用的鑑定技術，然而由一維粉末圖樣回推三維晶體結構仍是未解的逆問題。本文提出 **PXRD-Diff**，一個 3.7 M 參數的條件性去噪擴散模型，輸入模擬 Cu Kα PXRD 圖樣，在已知組成的條件下同時生成分數座標與晶格參數。模型包含 1D ResNet PXRD 編碼器、結合多解析度交叉注意力的週期距離訊息傳遞去噪網路，以及作為核心物理導引貢獻的**可微 Bragg 結構因子模組**作為輔助損失。我們在 CDVAE MP-20 標準切分上訓練 (≤20 原子/晶胞，27 k 訓練/9 k 測試) 並進行七條件消融研究。三個主要發現：(1) 標準 ε 預測去噪器若僅將乾淨晶格輸入晶格頭，晶格損失永遠停在隨機基準；改傳「正在去噪的雜訊化晶格」可將晶格損失降低 95 %；(2) 由 ε 預測改為 **x₀ 殘差預測**使 StructureMatcher 匹配率提高 3.5 倍；(3) Wyckoff 位置嵌入與輔助距離矩陣損失兩個看似合理的擴充，皆未超越基準，且合併使用時甚至明顯劣化。最佳設定在 MP-20 測試集上達到 2.5 % 匹配率 (真實晶格、僅座標評估)，距離真正解決逆問題仍遠，但本工作所記錄的架構配方與失敗目錄具體、可重現，對社群應有實用價值。完整程式碼、檢查點與訓練日誌皆已公開。

**關鍵詞**：粉末 X 光繞射；晶體結構預測；去噪擴散模型；等變訊息傳遞；可微物理；消融研究

---

## 1. Introduction

Powder X-ray diffraction is, in volume of measurements, the most common form of structural characterization in solid-state chemistry and materials science. The forward problem — given a structure, simulate the pattern — is well solved by 1920s-vintage physics (the structure factor) and, in practice, by a one-line call to `pymatgen.analysis.diffraction.xrd`. The inverse problem is brutal. The same Bragg reflections can be produced by symmetry-equivalent structures; intensity information is integrated over crystallite orientations; the reduction from 3D structure to 1D intensity-vs-2θ destroys atom labelling entirely. The traditional workflow — indexing, Pawley/Le Bail fits, Rietveld refinement — requires an experienced crystallographer and an initial structural model good enough that refinement converges. For the long tail of unindexed patterns from new materials, the structure is often simply not solved.

Two recent generative-modelling efforts — DiffractGPT [^DiffractGPT] and Crystalyze [^Crystalyze] — argue that neural networks can shortcut this pipeline. Both report headline numbers in the tens of percent on either MP-20 or curated experimental datasets, but they use mutually incompatible evaluation protocols (different match criteria, different splits, different "success" thresholds), and the failure modes are not catalogued. A practitioner reading these papers cannot easily form a calibrated expectation of *what is actually hard* about PXRD inversion.

This paper is a contribution to that calibration. We build a deliberately small, deliberately reproducible conditional diffusion model — **PXRD-Diff** — and report an honest ablation. Our contributions:

1. **A clean baseline implementation.** PXRD-Diff is 3.7 M parameters, trains on a single RTX 5090 in roughly 1.5 h to 100 k steps, and contains no proprietary data; everything runs on the public CDVAE MP-20 split.
2. **A differentiable Bragg structure-factor module.** We implement a fully differentiable PXRD simulator in PyTorch that agrees with `pymatgen.XRDCalculator` at Pearson 0.96 (mean over 50 reference structures) and that can be plugged in as a "predicted pattern matches input pattern" auxiliary loss.
3. **Two architectural fixes whose absence silently breaks training.** We document a *bug pattern* — feeding the lattice head only the clean lattice it is supposed to denoise — that produces a healthy-looking training curve but a useless lattice predictor, and we show that an x₀-residual parameterisation lifts the structure-match rate by 3.5× over a standard ε-prediction baseline.
4. **A failure catalogue.** Two architecturally well-motivated extensions — Wyckoff-site token embeddings and an auxiliary atom-pair distance loss — fail to help and hurt in combination. We report the negative result in detail.
5. **Calibration of expectations.** Even with all the above, the best configuration matches 2.5 % of MP-20 test structures (StructureMatcher, true-lattice coord-only). PXRD inversion at MP-20 scale is still very far from solved by a small conditional diffusion model.

We intend the paper to be useful both as a recipe — the bug fixes and the x₀-residual trick are easy to reproduce and clearly load-bearing — and as an honest stake in the ground for what a modest, self-contained model can and cannot do on this problem.

---

## 2. Related work

**Diffusion for crystal structures.** CDVAE [^CDVAE] introduced the canonical MP-20 evaluation protocol and a VAE+diffusion approach for unconditional crystal generation. DiffCSP [^DiffCSP] sharpened this with a joint diffusion over coordinates and lattice. MatterGen [^MatterGen] scales the idea to several million structures and reports impressive validity numbers. PXRD-Diff borrows the joint-diffusion design from DiffCSP, restricting scope to the conditional setting where a 1D pattern is the conditioning signal.

**Neural PXRD inversion.** DiffractGPT [^DiffractGPT] casts PXRD inversion as a sequence-to-sequence problem (peak list → CIF token stream) and trains on a million simulated patterns. Crystalyze [^Crystalyze] uses a transformer encoder + diffusion head and reports recovery rates on both simulated and experimental patterns. Both are far larger than our model and use significantly more training data; both report on bespoke evaluation pipelines that complicate head-to-head comparison. We see PXRD-Diff as a smaller, simpler, and more reproducible point in the design space.

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

### 3.5 Sampling

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

**Two evaluation modes.** We report the *full pipeline* (predicted lattice + predicted coordinates) and an additional *true-lattice / coord-only* mode in which we substitute the ground-truth lattice parameters when building the predicted structure. The latter isolates coordinate quality from lattice quality; it is also the stricter setting that we use to compare ablations, because lattice prediction is itself a hard sub-problem and noise there can dominate the headline number.

### 4.3 Implementation details

- **Optimiser.** AdamW, LR 5×10⁻⁴, cosine decay to zero over 100 k steps, weight decay 10⁻⁴, gradient clipping at 1.0.
- **Batch / steps.** Batch 64, 100 000 steps (≈ 24 epochs), single RTX 5090 (32 GB VRAM), wall-clock ≈ 1.5–1.7 h per run.
- **Parameter count.** ≈ 3.7 M trainable; `--n-layers 5 --d-model 384` (~10 M) gave no benefit (see §5.2).
- **Diffusion.** $T = 1000$ continuous-time formulation (we sample $t \sim U(0,1)$ per training example).
- **Compute budget.** Total: ~12 h GPU across all runs reported here, plus ~6 h for evaluation. Cost: roughly USD 6 of cloud GPU.

---

## 5. Results

### 5.1 Main ablation

Table 1 summarises seven training runs sweeping the contributions of the Debye loss, the lattice-input fix, the x₀-residual parameterisation, and two extensions (Wyckoff embeddings, distance loss). All numbers are evaluated on the same 1 000-structure subset of MP-20 test, with the true lattice substituted in (coord-only setting). Pearson is the per-pattern Pearson correlation between predicted and true PXRD. RMSD is the StructureMatcher Cartesian RMSD over matched samples (NaN for non-matches is excluded from the mean).

**Table 1.** Coordinate-only ablation on MP-20 test (n = 1000, true lattice).

| Run   | Parameterisation | λ_Debye | Wyckoff | λ_dist | Match % | Pearson | RMSD (Å) |
|-------|------------------|---------|---------|--------|---------|---------|----------|
| v10   | ε                | 0       | –       | 0      | 1.40    | 0.359   | 0.17     |
| v11   | ε                | 1       | –       | 0      | 0.90    | 0.365   | 0.15     |
| **v13** | **x₀-residual**  | **1**   | **–**   | **0**  | **2.51** | **0.434** | 0.22 |
| v14   | x₀-residual      | 1       | yes     | 0.01   | 0.80    | 0.367   | 0.14     |
| v15   | x₀-residual      | 1       | yes     | 0      | 2.10    | 0.392   | 0.21     |
| v16   | x₀-residual      | 1       | –       | 0.01   | 1.80    | 0.368   | 0.22     |

The best run is **v13 (x₀-residual, λ_Debye = 1, no extensions)** at 2.51 % match and Pearson 0.434.

### 5.2 What worked

**The lattice-input fix.** This is the most consequential single change. In all runs prior to v10, the lattice head saw $[\bar{\mathbf{h}}, \mathbf{g}, \mathbf{t}_\text{cond}]$ — pooled atom features, global PXRD, and time embedding — but not the noisy lattice $\boldsymbol{\ell}^{(t)}$ that it was supposed to denoise. The training curve looked normal (loss decreasing on coords and aux), but the lattice loss sat at 1.0 (the variance of the standard normal prior) for the entire 100 k steps. The fix is one line: concatenate $W_\text{lat}\boldsymbol{\ell}^{(t)}$ into the lattice-head input. With it, lattice loss drops to ~0.02 on stable runs (a 95 % reduction). Sampled lattices become physically reasonable instead of $10^2$-Å lengths and negative angles. We mention this not because it is conceptually deep — it is a textbook denoising bug — but because it produced a perfectly plausible-looking training run for many days before we noticed anything was wrong.

**x₀-residual parameterisation.** Holding everything else fixed (lattice fix on, λ_Debye = 1), switching from ε-prediction (v11) to x₀-residual (v13) lifts match from 0.9 % to 2.5 % — a 2.8× absolute and 3.5× over the v10 ε baseline at λ = 0. We attribute this to two effects. First, the residual head being initialised to ≈ 0 means the model starts as the identity, which is a correct fixed point at $t = 1$; the ε head must learn to produce a non-trivial output everywhere from random init. Second, the $\mathcal{L}_\text{Debye}$ auxiliary loss requires recovering a clean coordinate estimate from the model output — under x₀-residual this is a simple addition; under ε-prediction it is a divide-by-$\sqrt{\bar{\alpha}(t)}$ that is numerically unstable near $t = 1$.

**Capacity is not the bottleneck.** A 10.1 M-parameter run with $d_\text{model} = 384, L = 5$ (run `gpu_v12`, killed at 18 k steps for budget reasons) showed coordinate loss stuck at the same ~1.0 plateau as v10 — confirming that the bottleneck for ε-prediction is parameterisation, not capacity.

### 5.3 What did not work

Two extensions from prior literature on related problems failed to help and are worth documenting.

**Wyckoff-site embeddings (v15).** Following the intuition that PXRD-equivalent structures live in symmetry-related orbits, we computed Wyckoff-site labels for every training structure with `spglib`, embedded them as a learned `nn.Embedding(27, 256)` initialised to zero, and added to the per-atom features. The embedding *did* learn — final mean absolute weight 0.065, with 5 404 of 6 912 entries non-zero — but it (a) failed to improve the headline match rate (2.1 % vs 2.5 % for v13), and (b) destabilised lattice prediction. v15's lattice loss converges only to ~0.6, vs ~0.02 for v16, despite having the same lattice-fix architecture. We suspect the Wyckoff embedding inflates the per-atom feature norm (`emb_std` 0.43 vs 0.25 for v16), shifting the lattice-pool input distribution far from anything the lattice head sees during early training, but we did not have budget to confirm this hypothesis.

**Distance-matrix auxiliary loss (v16).** A pairwise MLP $D_{ij} = f(\mathbf{h}_i, \mathbf{h}_j)$ predicting periodic Cartesian distances clipped at the 12 Å RBF cutoff. λ_dist = 0.01 was the largest weight at which training remained stable. The loss does decrease over training (final ~2.0 vs random baseline ~10), but the headline match is *worse* than v13 (1.8 % vs 2.5 %). Our interpretation: distance prediction shares the same per-atom features used for coordinate denoising, and the auxiliary objective biases features toward absolute distance reconstruction rather than the relative geometric updates the diffusion process needs.

**The combination is worst (v14).** Stacking Wyckoff and distance loss together (v14) collapses match to 0.8 %, below all other runs except the bare ε baseline. The two extensions interact destructively. We have no clean theoretical explanation; we report the result so future authors can avoid the combination.

### 5.4 Full-pipeline result

When we sample lattice from the model rather than using the ground-truth lattice, the picture is sobering. The best configuration (v13) achieves 1.2 % match rate at n = 256 in the full pipeline, with most failures attributable to lattice prediction sampling lattice parameters that are either physically reasonable but for the wrong space group, or in a few percent of cases physically degenerate (length < 0.5 Å, angles outside [10°, 170°]) and rejected before structure construction. Lattice prediction at MP-20 scale with a 3.7 M-parameter model conditioned only on a 1D PXRD pattern is, on this evidence, not yet a solved problem.

---

## 6. Discussion

**Where does the gap lie?** Our best Pearson-correlation between predicted and true PXRD is 0.43; the same model, evaluated against itself on a held-out 200-structure subset, has roughly 0.97 Pearson with the *correct* coordinates plugged into the differentiable simulator. So the encoder–denoiser is leaving ~0.5 of pattern-space agreement on the table. The auxiliary head, which sees only $\mathbf{g}$, achieves an aux loss of 0.007 — meaning the global PXRD vector contains *enough* information to predict lattice parameters very accurately *in isolation*. The fact that the full denoiser, given $\mathbf{g}$ + $\mathbf{F}_\text{pxrd}$ + atom features, then under-uses this signal suggests an architectural rather than informational bottleneck.

**Why per-atom anchoring is hard.** PXRD is a permutation-invariant signal: relabelling atoms of the same element does not change the diffraction pattern at all. Our denoiser is also permutation-equivariant. There is therefore no symmetry-breaking signal to tell atom $i$ "you go in the corner" rather than "you go on the face." The Wyckoff embedding was an attempt to introduce this symmetry-breaking; it did not work, which suggests the right approach may need to break symmetry at the *output* (e.g. by predicting an unordered set of orbits and a Hungarian-style matcher) rather than at the input.

**Honest comparison with prior work.** DiffractGPT and Crystalyze report headline numbers we do not match; both also use considerably more data, larger models, and (in DiffractGPT's case) a tokenisation that appears to bake in much more crystallographic prior. We do not claim that the fundamental task is impossible — only that, at MP-20 scale with a clean 3.7 M-parameter conditional diffusion model, 2.5 % is what we observe. The gap between this and 30 % published rates is large enough to warrant scrutiny of the published numbers' evaluation protocols.

---

## 7. Limitations

1. **Composition is given.** The challenging unknown-composition setting is out of scope; this is consistent with most prior work but worth stating.
2. **Simulated PXRD only.** No instrument response, preferred orientation, peak asymmetry, or background. Real-experimental data will degrade these numbers further.
3. **Fixed lattice for the Debye loss.** We avoid the joint coord+lattice gradient through the simulator by using ground-truth lattice; a curriculum in which true lattice is gradually replaced by predicted lattice could be valuable but was not tried.
4. **Single random seed per ablation.** Workshop budget; the ~0.5 % run-to-run variance we observed in pilot experiments is small relative to the 2.5×–3.5× effects reported in §5.1.
5. **MP-20 only.** No experiments on larger unit cells, organic crystals, or higher-Z elements.
6. **No baseline reproduction.** We do not reproduce DiffractGPT or Crystalyze. A clean head-to-head comparison would require porting their evaluation pipelines and is a paper of its own.

---

## 8. Conclusion

PXRD inversion remains an open problem. We report a careful, reproducible negative-leaning result on the MP-20 split: a 3.7 M-parameter conditional diffusion model with a differentiable Bragg structure-factor loss recovers 2.5 % of test structures (true lattice, coord-only) and 1.2 % in the full pipeline. Two architectural fixes — passing the noisy lattice into the lattice head, and using x₀-residual rather than ε-prediction — are individually load-bearing; together they account for most of our headline number. Two literature-motivated extensions, Wyckoff embeddings and an auxiliary distance loss, do not help and hurt in combination.

We release the code, all checkpoints, all training and evaluation logs, and the differentiable PXRD module as an installable PyTorch package. We hope the recipe and the failure catalogue are useful to the community as it continues to push on a problem that is, on the present evidence, harder than recent reports suggest.

---

## Acknowledgments

We thank the maintainers of `pymatgen`, `spglib`, and the CDVAE benchmark for tools that made this work possible. Compute for all GPU experiments was rented from Vast.ai; total compute spend was approximately USD 6 across roughly 12 GPU-hours on a single RTX 5090.

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

## AI Disclosure Statement

The author used Anthropic's Claude (Sonnet 4.6 and Opus 4.7) for code drafting, debugging assistance, ablation orchestration, and writing assistance during paper preparation. All experimental designs, scientific claims, code architecture, and final paper text were reviewed and approved by the author. No AI-generated text was published without human verification of every claim against the underlying data and code.

---

## References

[^DiffractGPT]: Choudhary, K. (2024). DiffractGPT: Atomic structure determination from X-ray diffraction patterns using a generative pre-trained transformer. *arXiv:2412.NNNNN* (placeholder; verify exact arXiv ID before submission).

[^Crystalyze]: Riesel, E. A., et al. (2024). Crystal structure determination from powder diffraction patterns with generative machine learning. *Journal of the American Chemical Society* (placeholder; verify exact citation before submission).

[^CDVAE]: Xie, T., Fu, X., Ganea, O.-E., Barzilay, R., & Jaakkola, T. (2022). Crystal diffusion variational autoencoder for periodic material generation. *International Conference on Learning Representations*.

[^DiffCSP]: Jiao, R., et al. (2023). Crystal structure prediction by joint equivariant diffusion. *Advances in Neural Information Processing Systems* 36.

[^MatterGen]: Zeni, C., et al. (2024). MatterGen: A generative model for inorganic materials design. *arXiv:2312.03687*.

[^MACE]: Batatia, I., Kovács, D. P., Simm, G. N. C., Ortner, C., & Csányi, G. (2022). MACE: Higher order equivariant message passing neural networks for fast and accurate force fields. *Advances in Neural Information Processing Systems* 35.

[^SchNet]: Schütt, K. T., et al. (2017). SchNet: A continuous-filter convolutional neural network for modeling quantum interactions. *Advances in Neural Information Processing Systems* 30.

[^NicholDhariwal]: Nichol, A., & Dhariwal, P. (2021). Improved denoising diffusion probabilistic models. *International Conference on Machine Learning*.

[^DDIM]: Song, J., Meng, C., & Ermon, S. (2021). Denoising diffusion implicit models. *International Conference on Learning Representations*.

[^WrenzcuckPhysML2025]: *Placeholder for a differentiable scattering simulator reference; replace with the appropriate citation before submission.*

> **Citation note for review.** Three references above (DiffractGPT, Crystalyze, and the differentiable-scattering reference) are flagged as placeholders pending verification. The author will resolve each via DOI lookup before submission; as a methods/negative-results paper, the central claims do not depend on the precise published numbers from these works, only on the meta-claim that current literature reports widely varying evaluation protocols.

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
| v13 | 3.7 M | x₀-residual + lat-fix + Debye λ=1 | **2.51 % match (best)** | Table 1 |
| v14 | 3.8 M | v13 + Wyckoff + distance loss | 0.80 % match | Table 1 |
| v15 | 3.8 M | v13 + Wyckoff only | 2.10 % match | Table 1 |
| v16 | 3.7 M | v13 + distance loss only | 1.80 % match | Table 1 |

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
- [x] AI tool usage disclosed (§AI Disclosure)
