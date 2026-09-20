# Referee-suggested references — verified 2026-09-20

Method: every DOI resolved through the Crossref API (full author lists, volume, pages);
content cross-checked from Crossref abstracts, arXiv, Europe PMC full text (both Nature
Communications papers), the Geneva open archive and search snippets. Publisher landing pages
(IUCr, Wiley, Elsevier, APS) returned HTTP 403 to automated fetches. Nothing here is from memory.

## Citations (IUCr author–date, full author lists) — paste into paper.md References

- Chitturi, S. R., Ratner, D., Walroth, R. C., Thampy, V., Reed, E. J., Dunne, M., Tassone, C. J. & Stone, K. H. (2021). *J. Appl. Cryst.* **54**, 1799–1810. https://doi.org/10.1107/S1600576721010840
- Cuocci, C., Corriero, N., Dell'Aera, M., Falcicchio, A., Rizzi, R. & Altomare, A. (2022). *Comput. Mater. Sci.* **210**, 111465. https://doi.org/10.1016/j.commatsci.2022.111465
- Dauter, Z. & Jaskolski, M. (2010). *J. Appl. Cryst.* **43**, 1150–1171. https://doi.org/10.1107/S0021889810026956
- de la Flor, G., Kroumova, E., Hanson, R. M. & Aroyo, M. I. (2023). *J. Appl. Cryst.* **56**, 1824–1840. https://doi.org/10.1107/S1600576723009068
- Favre-Nicolin, V. & Černý, R. (2002). *J. Appl. Cryst.* **35**, 734–743. https://doi.org/10.1107/S0021889802015236
- Lai, Q., Xu, F., Yao, L., Gao, Z., Liu, S., Wang, H., Lu, S., He, D., Wang, L., Zhang, L., Wang, C. & Ke, G. (2025). *Adv. Sci.* **12**, 2410722. https://doi.org/10.1002/advs.202410722
- Li, Q., Jiao, R., Wu, L., Zhu, T., Huang, W., Jin, S., Liu, Y., Weng, H. & Chen, X. (2025). *Nat. Commun.* **16**, 7428. https://doi.org/10.1038/s41467-025-62708-8
- Parackal, A. S., Goodall, R. E. A., Faber, F. A. & Armiento, R. (2024). *Phys. Rev. Mater.* **8**, 103801. https://doi.org/10.1103/PhysRevMaterials.8.103801
- Segal, N., Subramanian, A., Li, M., Miller, B. K. & Gómez-Bombarelli, R. (2026). *Digital Discovery* **5**, 1590–1599. https://doi.org/10.1039/D6DD00017G  (replaces the arXiv 2512.04036 "Segal et al. (2025)" entry; same title, same five authors)
- Yu, D., Zhu, Z., Leng, F. & Zhu, Y. (2026). *Nat. Commun.* **17**, 3274. https://doi.org/10.1038/s41467-026-70035-9

## What each paper actually does (write no stronger than this)

| Marker | Claim safe to make in the text |
|---|---|
| Chitturi2021 | 1D CNNs regress lattice parameters from ~10⁶ simulated patterns (~10 % MAPE); analyses how impurities, noise and broadening degrade the prediction; proposes ML estimate + refinement. |
| Segal2026 | Gradient descent on PXRD similarity alone from distorted starts; common XRD similarity metrics give a highly non-convex, ill-posed landscape; constraining to the true crystal family helps substantially. Published version of the preprint already cited. |
| PXRDGen | End-to-end: contrastively pretrained XRD encoder + diffusion/flow generator (DiffCSP/FlowMM lineage) conditioned on composition + PXRD + automatic Rietveld refinement; 82 % (1-sample) / 96 % (20-sample) match on MP-20. **Default generates lattice jointly**; it *can optionally accept* an independently determined cell (CellNet or conventional indexing) as an extra condition. Do NOT call it a two-stage architecture. |
| XtalNet | Equivariant end-to-end PXRD→structure with contrastive PXRD–crystal alignment; benchmarks are MOF datasets (hMOF-100/400: 90.2 % / 79 % top-10), not MP-20. |
| Parackal2024 | Enumerates symmetry-confined (Wyckoff-based) arrangements, ranks with a Wyckoff-representation energy regressor, fits atoms to the pattern, validates with DFT; identifies structures beyond known prototypes. |
| XRDSol | Equivariant GNN diffusion that denoises coordinates conditioned on PXRD; **requires stoichiometry AND the unit cell as inputs** (cell supplied, not solved); 82.3 % MP-20 simulated / 81.6 % ICDD-20 experimental. Directly relevant: it assumes an indexed cell. |
| FOX | Free open-source ab initio structure determination from powder data by direct-space global optimisation (simulated annealing, parallel tempering), modular building blocks. |
| EXPO | Case-study paper on direct-space (simulated annealing) solutions in EXPO; practical guidance. Not the primary software paper (that may be Cuocci et al. 2022 *J. Appl. Cryst.* 10.1107/S160057672200245X, NOT verified). |
| Dauter2010 | Tutorial on reading ITA Vol. A for non-specialists; covers general/special positions, multiplicity, Wyckoff letter and site symmetry; the Wyckoff letter is a naming convention. Good citation for "a Wyckoff letter is a label, not a constraint". |
| delaFlor2023 | Describes the IUCr Symmetry Database (space-group/point-group data + tools); Wyckoff-position data are in the database but not the paper's stated topic. Cite as the database reference, paired with Dauter2010. |

## Referee-description mismatches to phrase around
- PXRDGen: separation of cell determination is optional, not the architecture.
- de la Flor 2023: database paper, not a Wyckoff discussion.
- EXPO 2022: worked-examples paper.
- FOX: second author is Černý (Crossref renders it wrongly).
