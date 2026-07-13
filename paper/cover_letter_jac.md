# Cover letter — Journal of Applied Crystallography

Dear Editors,

Please consider the enclosed manuscript, "PXRD-Diff: diagnosing the encoder bottleneck in diffusion-based powder-diffraction structure inversion, with classical autoindexing as a drop-in remedy", for publication as a Research Paper in the Journal of Applied Crystallography.

The manuscript addresses a question at the centre of current powder-diffraction methodology: why do neural generative models, despite headline match rates in the tens of percent, fail to deliver the absolute d-spacing accuracy that structure solution from powder data requires? Using a deliberately small (3.7 M-parameter) conditional diffusion model as a diagnostic instrument, we localise the failure to the pattern encoder's global pooling, which discards absolute peak-position information, and we quantify the sensitivity with a perturbation study that finds a sharp knee at ~0.5 Å of cell-length error.

We believe the paper is a natural fit for the Journal for three reasons.

1. **It connects modern machine learning to the classical indexing literature the Journal itself established.** Replacing the model's learned lattice head with a classical Q-space autoindexer in the de Wolff/DICVOL lineage (Boultif & Louër, 2004, *J. Appl. Cryst.* **37**, 724–731) yields a statistically significant improvement (three-seed paired McNemar test, p < 10⁻⁴; 27 structures fixed, none broken), concentrated exactly in the high-symmetry systems where classical indexing has always excelled. The result is directly actionable for anyone designing hybrid classical/ML pipelines.

2. **It puts three published generative systems on one evaluation protocol.** DiffractGPT, PXRDnet and deCIFer are re-scored from their released checkpoints through a single pymatgen StructureMatcher harness with confidence intervals throughout, and the paper surfaces a measurement question of general interest: single-number "match rates" may conflate structure recovery with symmetry recovery.

3. **It is fully reproducible.** All code, trained checkpoints, per-structure evaluation flags and the differentiable Bragg simulator are openly available (GitHub: github.com/fronkt/pxrd-diff; archived at Zenodo: doi.org/10.5281/zenodo.20738994), and the paper includes an explicit catalogue of negative results to save the community repeated effort.

The manuscript is original, is not under consideration elsewhere, and has not been published previously in any form. I am the sole author; there are no competing interests, and the work received no external funding. I would like to publish via the standard (non-open-access) route.

Thank you for your consideration.

Yours faithfully,

Frank Cai
Purdue University, West Lafayette, IN, USA
frankyc11223@gmail.com
ORCID: 0009-0003-0041-1459
