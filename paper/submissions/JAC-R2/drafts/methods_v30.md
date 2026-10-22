<!-- DRAFT 2026-10-22 for the JAC R2 rewrite: replacement text for paper.md Sections 3.2 (denoiser and
     distance-features paragraphs), 3.4 and 3.6, describing the v30 model (model_family "csp").
     Written from the code (src/pxrd_diff/torus.py, model/csp_denoiser.py, sampler_pc.py,
     scripts/14_train_csp.py) before any v30 result exists. Numbers that depend on results are tagged
     [PENDING v30-*]. Not merged into paper.md until gate G2 passes. -->

### 3.2 Architecture overview (denoiser paragraphs, v30)

**Denoiser.** A periodic message-passing network in the style of the CSPNet of DiffCSP (Jiao *et al.*, 2023), with $L = 6$ layers and $d_\text{model} = 256$. Every atom starts from $\mathbf{h}_i^{(0)} = \text{Emb}(Z_i) + \mathbf{c}$, where the conditioning vector
$$
\mathbf{c} = \text{TimeEmb}(t) + W_g\,\text{LN}(\mathbf{g}) + W_\ell\,[\,\log a, \log b, \log c, \cos\alpha, \cos\beta, \cos\gamma\,;\ \tilde{\boldsymbol{\ell}}\,;\ \kappa\,]
$$
combines the diffusion time, the layer-normalised global pattern embedding, invariant features of the cell the network is shown, its standardised parameters $\tilde{\boldsymbol{\ell}}$, and a flag $\kappa$ that is 1 when that cell is supplied and clean and 0 when it is being denoised. Each layer (i) sends a message between every ordered pair of real atoms,
$$
\mathbf{m}_{ij} = \phi_m\!\left(\text{LN}(\mathbf{h}_i),\ \text{LN}(\mathbf{h}_j),\ \boldsymbol{\psi}(\mathbf{f}_j - \mathbf{f}_i),\ \boldsymbol{\lambda}\right),\qquad
\boldsymbol{\psi}(\mathbf{d}) = \big[\sin(2\pi k d_a),\ \cos(2\pi k d_a)\big]_{a = 1..3,\ k = 1..10},
$$
averages the messages each atom receives, and adds a FiLM-modulated (by $\mathbf{c}$) update to $\mathbf{h}_i$; then (ii) lets each atom cross-attend to the pattern, $\mathbf{h}_i \leftarrow \mathbf{h}_i + \text{MHA}(\text{LN}(\mathbf{h}_i), \text{LN}(\mathbf{F}_\text{pxrd}), \text{LN}(\mathbf{F}_\text{pxrd}))$. A coordinate head maps each final atom state to a three-component score, and a lattice head maps the pooled atom states and $\mathbf{c}$ to the lattice noise.

Because the fractional differences enter only through $\boldsymbol{\psi}$, the output is unchanged by a global translation of the structure and by adding an integer to any single coordinate, and it is permutation-equivariant; padding atoms neither send nor receive messages. The test suite checks each property (`tests/test_csp_model.py`). There is no absolute-coordinate input. The v10–v24 denoiser embedded the raw fractional coordinates linearly, which is neither periodic nor independent of the arbitrary origin of each CIF.

**Pattern conditioning, corrected during the second review.** In v10–v24 the cross-attention read the encoder's feature map without normalisation. Its tokens reach norms of about 1800, so the softmax saturated onto a single token (attention entropy 0.03 nats against 8.36 for uniform attention) and the block's output, the same for every atom and about 10³ times the atom state, erased the atom state at the following LayerNorm. Normalising keys, values and queries and adding the output to the residual stream (above) removes this (§5.7).

**Cell features and Q2.** The v30 denoiser uses no Cartesian distances, so the image-selection question raised by the referee does not arise: the cell enters through its invariant features, from which the message network can form any metric quantity it needs. The minimum-image distance function remains in the code, corrected and tested against `pymatgen`, for the legacy checkpoints (§5.7(g)).

### 3.4 Training objective (v30)

**Coordinates: wrapped-normal diffusion.** Fractional coordinates are diffused on the torus with variance-exploding wrapped Gaussian noise,
$$
\mathbf{F}^{(t)} = \big(\mathbf{F}^{(0)} + \sigma_t\,\boldsymbol{\epsilon}\big) \bmod 1,\qquad \sigma_t = \sigma_\text{min}^{\,1-t}\,\sigma_\text{max}^{\,t},\quad \sigma_\text{min} = 0.005,\ \sigma_\text{max} = 0.5,\ t \sim \mathcal{U}(0,1).
$$
The noise scale is set by the period of the coordinates: $\sigma_\text{max} = 0.5$ is already close to uniform on the torus, and $\sigma_\text{min} = 0.005$ is about 0.03 Å in a 6 Å cell. The network predicts the score of the wrapped normal,
$$
\mathbf{s}(\mathbf{F}^{(t)}\mid\mathbf{F}^{(0)}) = \nabla \log \sum_{\mathbf{k}\in\mathbb{Z}^3} \mathcal{N}\!\big(\mathbf{F}^{(t)} - \mathbf{F}^{(0)} + \mathbf{k};\ \mathbf{0},\ \sigma_t^2 \mathbf{I}\big),
$$
evaluated with $|k| \le 10$ images per axis and divided by its root-mean-square value at $\sigma_t$, computed once by quadrature. The loss is the mean squared error of the normalised score over real atoms, $\mathcal{L}_\text{coord}$; predicting zero scores 1.0 at every noise level. The target depends on $\mathbf{F}^{(t)} - \mathbf{F}^{(0)}$ only modulo 1, so it is unchanged by the choice of origin.

**Why the schedule was changed.** Up to v24 the coordinates used the same variance-preserving cosine schedule as the lattice, with unit-variance noise in fractional units. For $t > 0.19$ that noise exceeds $0.3$ cell lengths, at which a wrapped Gaussian keeps less than 17 % of its first Fourier component: for 81 % of training timesteps, and 40 of the 50 sampling steps, the noisy coordinates carried essentially no information about the clean ones. The do-nothing predictor $\hat{\mathbf{F}}^{(0)} = \mathbf{F}^{(t)}$ already sat at the random-guess ceiling of 1/12 above $t = 0.25$. Its loss over the training distribution is 0.0719; v22 ended training at 0.0707, 1.7 % below it, and v23 and v24 stayed at or above it (`analysis/denoiser_collapse/vp_torus_check.txt`, `identity_loss.txt`, training logs). §5.7(h) gives the accounting.

**Lattice.** The six standardised lattice parameters keep the variance-preserving cosine schedule, which is appropriate in $\mathbb{R}^6$, with noise prediction and loss $\mathcal{L}_\text{lat}$. For half of each training batch, chosen at random, the clean cell is supplied instead ($\kappa = 1$) and no lattice loss is taken, so one network serves both the learned-lattice mode and the given-cell modes (oracle, indexer).

**Pattern dropout.** For 10 % of each batch the pattern embedding and feature map are replaced by a learned null pattern. This gives, from the same weights, a composition-only generator used as a control (§5).

**Full objective.**
$$
\mathcal{L} = \mathcal{L}_\text{coord} + \lambda_\text{lat}\,\mathcal{L}_\text{lat} + \lambda_\text{aux}\,\mathcal{L}_\text{aux},\qquad \lambda_\text{lat} = \lambda_\text{aux} = 1,
$$
with AdamW (learning rate $10^{-3}$, weight decay $10^{-4}$, 1000-step linear warm-up, cosine decay over 100 000 steps), batch 256, gradient clipping at 1.0 and bfloat16 autocast. The differentiable simulator of §3.3 is not used as a training loss in v30; it is used at sampling time for candidate selection and refinement (§3.6).

### 3.6 Sampling (v30)

Predictor-corrector sampling (Song *et al.*, 2021; Jiao *et al.*, 2023) over $T = 1000$ noise levels $t_i = i/T$. Coordinates start uniform on the torus and the lattice from standard Gaussian noise. At each level, a Langevin corrector step
$$
\mathbf{F} \leftarrow \mathbf{F} + \gamma_i\,\hat{\mathbf{s}} + \sqrt{2\gamma_i}\,\mathbf{z},\qquad \gamma_i = 10^{-5}\,(\sigma_{t_i}/\sigma_\text{min})^2,
$$
is followed by the ancestral predictor step from $\sigma_{t_i}$ to $\sigma_{t_{i-1}}$ (with $\sigma_{t_0} = 0$),
$$
\mathbf{F} \leftarrow \mathbf{F} + (\sigma_{t_i}^2 - \sigma_{t_{i-1}}^2)\,\hat{\mathbf{s}} + \sqrt{\sigma_{t_{i-1}}^2(\sigma_{t_i}^2 - \sigma_{t_{i-1}}^2)/\sigma_{t_i}^2}\;\mathbf{z},
$$
each wrapped to $[0,1)$, with $\hat{\mathbf{s}}$ re-evaluated before each step. The lattice follows the ancestral DDPM update with the predicted noise. In the learned-lattice mode the network sees only the cell being sampled: the reference cell is never read, and a regression test confirms that the output is identical whatever cell, including an all-NaN one, is passed to the sampler. In the given-cell modes the lattice is not sampled; the given cell is shown clean with $\kappa = 1$ and is the output cell.

The rest of the pipeline is unchanged from the submitted version: 20 candidates per pattern, the one whose differentiable pattern correlates best with the input is kept, and its coordinates are refined for 200 Adam steps against the same correlation (§3.3). One sampled structure takes [PENDING v30-timing] on an A100 GPU.
