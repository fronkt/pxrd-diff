# HAT5032 revision 1 — change log for `paper/paper.md`

Prepared 2026-09-20. Referee items are R1.n (referee 1), R2.n (referee 2), R3.n (referee 3), matching
the numbering in `referee_reports/hat5032_authorreview.txt`. Every `[PENDING Cx]` tag in the manuscript
marks a place where a GPU or long-CPU result is still to be inserted; `[REF-PENDING: key]` marks a
citation awaiting verified metadata before it enters the References list.

## Title
- **R1.1, R1.3, R1.5i.** "…, with classical autoindexing as a drop-in remedy" → clause dropped. New title:
  *PXRD-Diff: diagnosing the encoder bottleneck in diffusion-based powder-diffraction structure inversion.*
  **Frank's call to revert**; the indexer is now framed as a control, so the old clause overstated it.

## Synopsis
- **R1.1, R2.2, R2.3, R3.3.** Rewritten: diagnostic instrument; pooling claim (was "destroys") replaced by
  the ablation outcome: "restoring absolute peak positions to the lattice head … does not reduce its ~1.3 Å
  cell error; the evidence points to the regression formulation … not to the pooling stage"; indexer
  "supplied with the true crystal system, serves as a control"; code audit mentioned.

## Abstract
- **R1.1, R1.5i.** Opens with "This paper is a diagnostic study"; model "kept deliberately small"; "not
  proposed as a competitive inversion tool".
- **R1.4.** DiffractGPT-vs-PXRDnet all-correct ordering removed from the abstract (stays in §5.3/§6 as a
  hypothesis).
- **R1.1.** "leaving PXRD-Diff 12–46 times behind" → external rates given once as scale, PXRD-Diff 1.6 %.
- **R2.2.** "The limiter is the global pooling" → the ablation outcome ("leaves the error unchanged
  (1.28–1.31 Å across three arms …) so the earlier reading that global pooling discards the d-spacings is
  withdrawn") and the revised reading (regression head does not perform the peak-to-index assignment).
- **R2.3, R1.3.** Indexer sentence now says "supplied with the true crystal system" and "classical control".
- **R3.3c.** "1.5 % [1.2, 2.0]" → "1.6 % [1.2, 2.1]" (46/2883, unindexed patterns scored as misses).
- **R2.1, R3.3a.** New sentences on the code audit and the re-validated simulator (median 0.999, n = 1000)
  with `[PENDING C1]` for the retrained numbers.
- "the classical-indexer drop-in" removed from the contributions list.

## §1 Introduction
- **R1.1, R1.5i.** Framing paragraph: "use it as a diagnostic instrument … not proposed as an alternative".
- **R1.2.** Contribution 1: simulator described as "a PyTorch port of the pymatgen structure-factor
  calculation … introduces no new physics"; the "Pearson 0.96 on 50 references" claim removed.
- **R2.2.** Contribution 2: "so the pooled encoding … fails" → "A controlled pooling ablation (§5.6) finds
  that neither position-aware pooling nor explicit peak-position features reduce that error … so the pooled
  encoding is not the stage at fault; the evidence points instead to the regression formulation"; indexer
  CI updated to [1.2, 2.1].
- **R1.3, R2.3, R1.5iii.** Contribution 3 retitled "A classical control": crystal system supplied; "sanity
  check on the diagnosis, not a method"; unknown-system run announced `[PENDING C3]`.
- **R3.2.** Contribution 4: "Wyckoff-site" → "Wyckoff-letter"; "read narrowly: label, not a constraint".
- **R1.1, R1.4.** Contribution 5: "PXRD-Diff at 1.6 % is 12–19× behind" removed; "These numbers set the scale
  … not a claim of competitiveness"; deCIFer added to the list; ordering kept as a hypothesis.
- **R3.3.** New contribution 6: the code audit, pointing to §5.7.

## §2 Related work
- **R3.1.** New paragraph "Structure solution as a staged problem" citing `[REF-PENDING: PXRDGen]`,
  `[XtalNet]`, `[Parackal2024]`, `[XRDSol]`, `[Chitturi2021]`; wording deliberately generic.
- **R3.1.** "Classical autoindexing" → "Classical structure solution from powder data": adds the
  direct-space / simulated-annealing / charge-flipping stage with `[REF-PENDING: FOX]`, `[EXPO]`; states
  the indexer is given the true crystal system "which classical practice must itself infer".
- **R1.2.** "Differentiable physics": "the contribution here is full differentiability" → "PyTorch port …
  adds no physics beyond that computation".
- **R3.1.** Segal et al. noted as "since published in Digital Discovery `[REF-PENDING: Segal2026]`".

## §3.3 Differentiable Bragg loss
- **R1.2.** "The core physics-informed contribution" → "an engineering component, not a physical
  contribution"; ideal-powder scope stated (no preferred orientation, asymmetry, profile, background).
- **R2.1, R3.3a.** Form factor rewritten as f(s) = Z − 41.78214 s² Σ a_k exp(−b_k s²), pymatgen convention.
- **R2.1, R3.3a.** New paragraph "Correction made during review": bare sum was used; Si f(0) = 5.8 vs Z = 14;
  corrected and re-validated; affected runs retrained `[PENDING C1]`.
- **R2.1.** Validation paragraph rewritten to n = 1000, hkl_max 5: corrected mean 0.988 / median 0.999
  (98.0 % > 0.9); submitted form factor 0.952 / 0.969 (90.7 % > 0.9); improved for 1000/1000; first-50 means
  0.939 vs 0.974; note that the submitted 0.962 used the validation script's default |hkl| ≤ 10.
- Fig. 1 caption updated to the n = 1000 two-convention numbers (figure file to be regenerated).

## §3.4 Training objective
- **R3.3b.** New paragraph "Correction made during review": x₀-mode Debye loss was evaluated on the residual
  rather than on noisy + residual for v13–v16 and v21; ε-mode v10/v11 unaffected; fixed; retrained
  `[PENDING C1]`.

## §3.5 Classical indexing
- **R1.3, R2.3.** Heading "drop-in for" → "control for"; input (ii) now "the true crystal system of the target
  structure … prior information beyond the composition … optimistic control rather than a fair competitor".
- **R1.5iii.** New paragraph "Without the crystal system": all lattice types searched, best M20 kept;
  `[PENDING B2]` (indexing accuracy) and `[PENDING C3]` (downstream match, three seeds).
- **R3.3c.** Final paragraph: "drop-in in the strongest sense" → "in the mechanical sense"; unindexed
  patterns (39/1000) scored as misses; submitted version's fallback disclosed with pointer to §5.7.

## §4 Experiments
- **R3.3c.** Evaluation-modes paragraph: "a pattern the indexer cannot index counts as a miss; it is never
  scored with the true lattice."

## §5.1 Phase 4 ablation
- **R1.4, R1.5ii.** Intro sentence and Table 1 caption: "Single seed per row; exploratory"; only the
  ε-vs-x₀ contrast replicated; note which rows used the flawed simulator / Debye defect `[PENDING C1]`.
- Fig. 2 caption: "(single seed per run; exploratory)".

## §5.2 Indexer control
- **R1.3, R2.3.** Heading "drop-in" → "control"; intro sentence and Table 2 caption state the crystal system
  is supplied; Table 2 row label "…, crystal system given"; `[PENDING C1]` on the caption for a retrained
  snapshot.
- **R3.3c.** Accounting: indexer arm 2 883 structure-seeds (117 unindexed, scored as misses; none matched
  under the old fallback); 86 unindexed rows inside the 1 949-row paired set enter as concordant misses;
  marginal count 46 / 3 000 → 46 / 2 883.
- **R1.3.** "So the drop-in reproducibly improves" → "So the control reproducibly improves … expected outcome
  for a classical algorithm handed the crystal system … sanity check".
- **R3.3c.** "1.5 % [1.2, 2.0]" → "1.6 % [1.2, 2.1]"; "1.5 → 4.5 %" → "1.6 → 4.5 %" (twice).

## §5.3 External baselines
- **R2.3, R1.1.** Table 3 caption: PXRD-Diff row uses the indexer with the true crystal system; external
  models get pattern + composition only; "not a competitiveness claim". Row label "v21 + indexer control,
  crystal system given".
- **R1.1.** "~46× above PXRD-Diff" and "PXRD-Diff is 12–19× behind" removed → "an order of magnitude below
  all three, as expected for a 3.7 M-parameter diagnostic model".
- **R2.3.** "The gap is capability and compute, not input information" → "What the gap does and does not
  reflect": PXRD-Diff's indexer row is *favoured* on prior information; pattern-resolution asymmetry kept;
  "reported to set the scale, not to rank".
- Fig. 4 caption: "loses on match rate by 12–19×" → "order of magnitude below … indexer supplied with the
  crystal system".

## §5.4 Failure catalogue
- **R3.2.** Wyckoff paragraph rewritten: letter meaningful only within a space group `[REF-PENDING:
  Dauter2010]` `[delaFlor2023]`; embedding carried a label, not a constraint; coordinates generated freely;
  negative result "narrow": label token does not help; says nothing about symmetry imposed as a constraint.

## §5.5 Perturbation study
- **R2.2.** "the limiter is the pooled encoding" → "§5.6 shows this is not repaired by giving the head the
  peak positions, and reads it as a failure of the regression formulation rather than of the pooled encoding
  or the denoiser".

## §5.6 Pooling bottleneck
- **R2.2.** Retitled "Why the learned lattice fails: neither the head nor the pooling repairs the scale".
- **R2.2.** "The error is a property of the pooled global encoding" → "The submitted version read this as a
  property of the pooled global encoding … the controlled ablation below tests that reading directly and
  does not support it"; the indexer reconciliation now says it "assigns Miller indices to peak positions by
  explicit search rather than regressing six numbers"; "The architectural implication is concrete" removed.
- **R2.2.** New paragraph "Controlled pooling ablation" + **new Table 4**: design (i) global average pool,
  (ii) position-aware attention pooling, (iii) explicit peak-position features, all else fixed; result on the
  frozen v21 encoder (mean length MAE 1.29 / 1.31 / 1.28 Å vs 1.32 Å for the checkpoint head; spread 0.03 Å;
  all ~2.5× above the 0.5 Å knee); pooling reading withdrawn; revised reading = regression formulation does
  not perform the peak-to-index assignment; three caveats (frozen encoder, arm (ii) not converged, v1
  encoder trained with the R3.3(b) Debye term); `[PENDING C2-box]` for the 10-epoch rerun on the retrained
  checkpoint and the end-to-end match.

## §5.7 Code audit and corrections (NEW)
- **R3.3a/b/c, R2.1.** Lists the three errors, what each affected and did not affect, the re-validation
  numbers, the fallback recomputation (0/117 matched; McNemar cells unchanged; 46/2883 = 1.6 % [1.2, 2.1]),
  the top-K rerank consequence (can only be lower; conclusion stands), `[PENDING C1]` for retrains.

## §6 Discussion
- **R2.2.** "The limiter is therefore the global pooling" → "The controlled ablation of §5.6 shows that
  restoring absolute peak positions to the head does not move it either, so the pooling reading of the
  submitted version is withdrawn; the evidence now points to the regression formulation".
- **R3.2.** Wyckoff sentence: "with a label rather than a constraint".
- **R1.1.** "Niche": "12–19× behind" removed; "that is the point of its size"; "diagnostic instrument, not a
  candidate solver"; "indexer drop-in" → "indexer control".

## §7 Limitations
- **R2.3, R1.5iii.** Indexer-realism bullet: "downstream 1.5 %" → "1.6 %"; "and a known crystal system; the
  unknown-system run of §3.5 removes the second assumption `[PENDING C3]`".
- **R1.5iv.** New bullet "Idealised conditions and the bottleneck conclusion": instrument effects, preferred
  orientation, background would degrade the encoder further; diagnosis is a lower bound; knee may move.

## §8 Conclusion
- **R1.1, R2.2, R2.3, R3.3.** Rewritten: diagnostic instrument; "neither position-aware pooling nor explicit
  peak-position features reduce that error, so the pooling reading is withdrawn; the evidence points instead
  to the regression formulation"; indexer as control with crystal system given, 1.6 % [1.2, 2.1]; external rates "set the
  scale … not a ranking claim"; code audit sentence `[PENDING C1]`; Wyckoff "read narrowly"; "12–46×" and
  "classical-autoindexer drop-in are load-bearing" removed.

## Data and Code Availability
- **R3.3.** Added: corrected simulator, audit scripts of §5.7 and per-structure records behind Table 2a.

## Not changed
- References list (markers to be resolved after verification); Tables 1, 2a, 3 numbers other than labels;
  §3.1, §3.2, §3.6, Appendix; figure files (Fig. 1 to be regenerated with both conventions).

## Readability pass (R1.6)
- Applied only inside the sections above: shorter sentences, one number per sentence where possible, no
  em dashes introduced, British spelling. Untouched sections retain their original prose.
