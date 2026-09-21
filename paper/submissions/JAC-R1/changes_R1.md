# HAT5032 revision 1 — change log for `paper/paper.md`

Prepared 2026-09-20. Referee items are R1.n (referee 1), R2.n (referee 2), R3.n (referee 3), matching
the numbering in `referee_reports/hat5032_authorreview.txt`. Every `[PENDING Cx]` tag in the manuscript
marks a place where a GPU or long-CPU result is still to be inserted; `[REF-PENDING: key]` marks a
citation awaiting verified metadata before it enters the References list.

## Title
- **R1.1, R1.3, R1.5i.** "…, with classical autoindexing as a drop-in remedy" → clause dropped. New title:
  *PXRD-Diff: diagnosing the lattice-recovery bottleneck in diffusion-based powder-diffraction structure inversion.*
  **Frank's call to revert**; the indexer is now framed as a control, so the old clause overstated it.

## Synopsis
- **R1.1, R2.2, R2.3, R3.3.** Rewritten: diagnostic instrument; pooling claim (was "destroys") replaced by
  the ablation outcome: "restoring absolute peak positions to the lattice head … does not reduce its ~1.3 Å
  cell error; the evidence points to the regression formulation … not to the pooling stage"; indexer
  "supplied with the true crystal system, serves as a control"; code audit mentioned.
- **R3.3 (retrain).** Closing sentence added: retraining with the corrected loss lifts the learned head from 0 to
  1.9 %, level with the control, oracle 5.5 %; "the gap to the oracle is the effect that replicates".

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
- **R2.1, R3.3a.** New sentences on the code audit and the re-validated simulator (median 0.999, n = 1000).
- **R3.3 (retrain, v22).** "PXRD-Diff reaches 1.6 %" → "1.9 %"; oracle "4.5 % [3.8, 5.3]" → "5.5 % [4.7, 6.4] …
  against 1.9 % [1.5, 2.5] for the learned head (paired McNemar p < 10⁻¹⁷)"; the McNemar sentence "gain over
  the learned head is significant (p < 10⁻⁴; 27 fixed, none broken) and confined to the high-symmetry
  systems" → "ties the learned head on match rate (42 fixed, 52 broken, p = 0.35), fixing hexagonal and
  tetragonal structures and losing cubic ones"; "a Debye loss evaluated on the wrong tensor" added to the
  audit sentence; closing sentence: retrain raised the learned head 0 → 1.9 %, removed the indexer's
  aggregate lead, oracle gap is the result that replicates.
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
- **R3.3 (retrain).** Contribution 2: oracle 4.5 % → 5.5 % [4.7, 6.4] with the paired p against the learned
  head; "4.5 % oracle ceiling" → "5.5 %". Contribution 3: "where the learned head recovers nothing. A
  three-seed paired McNemar test confirms the difference (p < 10⁻⁴; 27 fixed, 0 broken)" → submitted vs
  corrected checkpoint stated ("the two tie at three seeds (42 fixed, 52 broken, p = 0.35), the indexer
  winning the hexagonal and tetragonal structures … the learned head winning the cubic ones"); "expected to
  beat" → "expected to match or beat".
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
  corrected and re-validated; the production checkpoint retrained from scratch (v22, behind Tables 2, 2a, 3
  and Figs 4, 5); Phase-4 rows not retrained; learned head 0 → 1.9 %, control and oracle within noise.
- **R2.1.** Validation paragraph rewritten to n = 1000, hkl_max 5: corrected mean 0.988 / median 0.999
  (98.0 % > 0.9); submitted form factor 0.952 / 0.969 (90.7 % > 0.9); improved for 1000/1000; first-50 means
  0.939 vs 0.974; note that the submitted 0.962 used the validation script's default |hkl| ≤ 10.
- Fig. 1 caption updated to the n = 1000 two-convention numbers (figure file to be regenerated).

## §3.4 Training objective
- **R3.3b.** New paragraph "Correction made during review": x₀-mode Debye loss was evaluated on the residual
  rather than on noisy + residual for v13–v16 and v21; ε-mode v10/v11 unaffected; fixed; production
  checkpoint retrained (v22: EMA loss 0.726 → 0.602, learned head 0 → 1.9 %); v13–v16 not retrained, labelled.

## §3.5 Classical indexing
- **R1.5iii (C3 on the box).** Unknown-system cells substituted at three seeds: 1.4 % [1.0, 1.9] vs 1.6 % given
  (0 gained / 6 lost, p = 0.03; vs learned head p = 0.11). New third row of Table 2; abstract, §1 contribution 2
  and §7 bullet (i) carry the number.
- **R1.3, R2.3.** Heading "drop-in for" → "control for"; input (ii) now "the true crystal system of the target
  structure … prior information beyond the composition … optimistic control rather than a fair competitor".
- **R1.5iii.** New paragraph "Without the crystal system": lattice types searched highest symmetry first,
  accepted by the de Wolff / TREOR / DICVOL rule (all lines indexed, M20 ≥ 10, system from metric symmetry);
  43.4 % strict vs 48.8 % given, lattice type recovered 65.4 %, paired loss 65 / gain 9; the first-pass max-M20
  rule (24.3 %) reported alongside. §1 sentence and §7 bullet (i) updated; refs de Wolff (1968) and Werner
  et al. (1985) added. `[PENDING C3]` (downstream match, three seeds) remains. Fig. 3 (fig4_indexer_bench)
  panel (a) gains the system-unknown strict bars and the dual overall line; caption updated.
- **R3.3c.** Final paragraph: "drop-in in the strongest sense" → "in the mechanical sense"; unindexed
  patterns (39/1000) scored as misses; submitted version's fallback disclosed with pointer to §5.7.

## §4 Experiments
- **R3.3c.** Evaluation-modes paragraph: "a pattern the indexer cannot index counts as a miss; it is never
  scored with the true lattice."

## §5.1 Phase 4 ablation
- **R1.4, R1.5ii.** Intro sentence and Table 1 caption: "Single seed per row; exploratory"; only the
  ε-vs-x₀ contrast replicated; states which rows used the flawed simulator / Debye defect and that these rows
  were not retrained ("only the production checkpoint was (v22, §5.2)").
- **R3.3 (retrain).** Closing sentence of §5.1: "v21 checkpoint reaches 4.5 % under oracle … only 1.0 %
  under full pipeline" → "5.5 % under oracle (v22, three seeds; the submitted v21 gave 4.5 %) but only 1.9 %".
- Fig. 2 caption: "(single seed per run; exploratory)".

## §5.2 Indexer control
- **R1.3, R2.3.** Heading "drop-in" → "control"; intro sentence and Table 2 caption state the crystal system
  is supplied; Table 2 row label "…, crystal system given".
- **R3.3 (retrain, v22).** Intro: "with the v21 checkpoint … single-seed" → "at three seeds with the v22
  checkpoint (the v21 configuration retrained under the corrected simulator and the corrected x₀-mode Debye
  loss)"; v21 numbers "retained in §5.7 for the record". **Table 2 replaced**: three-seed pooled v22 rows
  (learned 1.9 [1.5, 2.5] / indexer 1.6 [1.2, 2.1] / oracle 5.5 [4.7, 6.4]; all-correct, sg@0.1, RMSD median,
  R_wp and a new Pearson column). Paragraph "The lift is real: a three-seed paired test" → "A three-seed
  paired test": eval-hang accounting reduced to one parenthesis (v22 scores 3 000/3 000; the 1 949-row
  intersection is moot). **Table 2a replaced** with the v22 2×2 (6 / 52 / 42 / 2 900). Result paragraph
  rewritten: b = 42, c = 52, p = 0.35, the two sources tie; submitted 0/1 949 vs 46/2 883 (27 fixed, 0
  broken, p = 1.5 × 10⁻⁸) quoted once and attributed to the defective checkpoint (retrain gains 32 / loses 0
  learned-head structures, p = 4.7 × 10⁻¹⁰; indexer 12 / 10, p = 0.83); "control reproducibly improves"
  claim withdrawn as a general statement; per-system pattern reported (hexagonal 16/0, tetragonal 7/2,
  cubic 18/47 with per-system counts); indexer cells closer (Pearson 0.41 vs 0.18, RMSD median 0.035 vs
  0.232 Å); all-correct 6/0 (p = 0.03). Oracle paragraph: "4.5 % [3.8, 5.3] … disjoint from the indexer's
  1.6 %" → "5.5 % [4.7, 6.4] (165 / 3 000; 4.5 % with the submitted checkpoint, unpaired p = 0.08)"; paired
  oracle-vs-learned 134 / 27, p = 3 × 10⁻¹⁸ "the effect that replicates"; "1.6 → 4.5 %" → "1.9 → 5.5 %".
- **R3.3c.** Unindexed patterns (117 structure-seeds) scored as misses, stated in the Table 2 caption.
- Fig. 3 caption: "indexer's significant match-rate gain … concentrated entirely in the high-symmetry
  systems" → "paired gains over the learned head … confined to the hexagonal and tetragonal systems".

## §5.3 External baselines
- **R2.3, R1.1, R3.3 (retrain).** Table 3 now carries two PXRD-Diff rows, both v22 at three seeds (n = 3 000):
  "learned head" (1.9 [1.5, 2.5]; pattern + composition only, the same information as the external models)
  and "+ indexer control, crystal system given" (1.6 [1.2, 2.1]); caption and the "What the gap does and does
  not reflect" paragraph say which row is favoured on prior information; "PXRD-Diff, at 1.6 %" → "1.9 %".
- **R1.1.** "~46× above PXRD-Diff" and "PXRD-Diff is 12–19× behind" removed → "an order of magnitude below
  all three, as expected for a 3.7 M-parameter diagnostic model".
- **R2.3.** "The gap is capability and compute, not input information" → "What the gap does and does not
  reflect": PXRD-Diff's indexer row is *favoured* on prior information; pattern-resolution asymmetry kept;
  "reported to set the scale, not to rank".
- Fig. 4 (fig5_threeway_headline) **regenerated**: the PXRD-Diff bar is now the v22 learned head, three seeds
  pooled (`phase15_results/v22_learned_pooled.json`; was the v21 indexer row `p9_idxlat_n1000.json`); n-label
  "n = 3 × 1000". Caption: "loses on match rate by 12–19×" → "order of magnitude below"; states the bar is the
  learned head with no prior beyond composition, indexer control 1.6 % and oracle 5.5 % in the caption; Pearson
  0.18 against 0.02 and 0.16.

## §5.4 Failure catalogue
- **R3.2.** Wyckoff paragraph rewritten: letter meaningful only within a space group `[REF-PENDING:
  Dauter2010]` `[delaFlor2023]`; embedding carried a label, not a constraint; coordinates generated freely;
  negative result "narrow": label token does not help; says nothing about symmetry imposed as a constraint.

## §5.5 Perturbation study
- **R2.2.** "the limiter is the pooled encoding" → "§5.6 shows this is not repaired by giving the head the
  peak positions, and reads it as a failure of the regression formulation rather than of the pooled encoding
  or the denoiser".

## §5.6 Pooling bottleneck
- **R2.2 (C2 on the box).** Table 4 restructured to carry both encoders (v21 at 3 epochs, v22 at 10) and an
  end-to-end column (0.7 / 0.5 / 1.2 % for (i)/(ii)/(iii) vs 1.9 % full pipeline); every regression-head match is
  cubic; pooling reading stays withdrawn; caveats reduced to two. Arm (ii) also at 30 epochs (1.12 Å, converged):
  end-to-end 0.8 % [0.5, 1.2], level with arm (i); extra Table 4 row.
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
  numbers, the fallback recomputation (0/117 matched; submitted McNemar cells unchanged by the fallback;
  46/2883 = 1.6 % [1.2, 2.1]; "superseded by the v22 retrain of (a)"), the top-K rerank consequence (can only
  be lower; conclusion stands).
- **R3.3 (retrain).** Item (a) now records the retrain: v22 from scratch, EMA 0.602 vs 0.726; what moved
  (learned 0/1 949 → 58/3 000; indexer 46/2 883 → 48/3 000, paired 12/10, p = 0.83; oracle 135 → 165/3 000,
  p = 0.08; indexer lead 27/0 → tie 42/52, p = 0.35) and what was not rerun (Table 1, §5.4; ~18 GPU-h).

## §6 Discussion
- **R2.2.** "The limiter is therefore the global pooling" → "The controlled ablation of §5.6 shows that
  restoring absolute peak positions to the head does not move it either, so the pooling reading of the
  submitted version is withdrawn; the evidence now points to the regression formulation".
- **R3.2.** Wyckoff sentence: "with a label rather than a constraint".
- **R1.1.** "Niche": "12–19× behind" removed; "that is the point of its size"; "diagnostic instrument, not a
  candidate solver"; "indexer drop-in" → "indexer control"; "At 1.6 %" → "At 1.9 %".
- **R3.3 (retrain).** First paragraph: "disjoint oracle-vs-indexer CIs … firmest claim" → "paired
  oracle-vs-learned-head test (134 fixed, 27 broken, p = 3 × 10⁻¹⁸) … the one result that replicates across
  the submitted and the corrected checkpoint"; indexer "recovers part of the 5.5 % oracle ceiling on the
  hexagonal and tetragonal subset; … no longer leads the learned head in aggregate … cells stay far closer
  (Pearson 0.41 against 0.18)"; "4.5 %" → "5.5 %" (twice).

## §7 Limitations
- **R2.3, R1.5iii.** Indexer-realism bullet: "downstream 1.5 %" → "1.6 %"; "and a known crystal system; the
  unknown-system run of §3.5 removes the second assumption `[PENDING C3]`"; "the indexer's advantage over the
  learned head" → "the indexer's cell-accuracy advantage over the learned head (§5.2)".
- **R1.5ii, R3.3 (retrain).** Statistics bullet: "lift confirmed significant by a paired McNemar test
  (p < 10⁻⁴) … superseding the earlier underpowered unpaired test" → the comparison is "the cautionary case":
  significant with the submitted checkpoint (27/0), a tie with the corrected one (42/52, p = 0.35); "a paired
  test at three seeds protects against seed noise but not against a training defect that moves one arm";
  oracle "4.5 % [3.8, 5.3] … firming the single-seed 5.6 %" → "5.5 % [4.7, 6.4] … (4.5 % with the submitted
  one)"; oracle-vs-learned-head gap named as "the effect that replicates".
- **R1.5iv.** New bullet "Idealised conditions and the bottleneck conclusion": instrument effects, preferred
  orientation, background would degrade the encoder further; diagnosis is a lower bound; knee may move.

## §8 Conclusion
- **R1.1, R2.2, R2.3, R3.3.** Rewritten: diagnostic instrument; "neither position-aware pooling nor explicit
  peak-position features reduce that error, so the pooling reading is withdrawn; the evidence points instead
  to the regression formulation"; indexer as control with crystal system given, 1.6 % [1.2, 2.1]; external rates "set the
  scale … not a ranking claim"; Wyckoff "read narrowly"; "12–46×" and
  "classical-autoindexer drop-in are load-bearing" removed.
- **R3.3 (retrain).** Oracle "4.5 % [3.8, 5.3]" → "5.5 % [4.7, 6.4] … against 1.9 % [1.5, 2.5] for the learned
  head (paired McNemar p = 3 × 10⁻¹⁸)"; indexer sentence "where the learned head recovers nothing (… p < 10⁻⁴),
  and only in the high-symmetry systems" → "ties the learned head at three seeds (42 fixed, 52 broken,
  p = 0.35), its gains confined to the hexagonal and tetragonal systems"; code-audit sentence closes with the
  retrain outcome (learned head 0 → 1.9 %, indexer's aggregate lead removed, oracle gap replicates).

## Appendix C (run table)
- **R3.3 (retrain).** v21 row de-emphasised and labelled "submitted version; trained with the §5.7 (a) and (b)
  defects" with its three-seed numbers (0 / 1.6 / 4.5 %); new bold **v22** row (retrained from scratch with the
  corrected form factor and x₀-mode Debye loss; 1.9 / 1.6 / 5.5 %, all 3-seed; Tables 2, 2a, 3).

## Data and Code Availability
- **R3.3.** Added: corrected simulator, audit scripts of §5.7 and per-structure records behind Table 2a.

## Not changed
- Table 1 numbers (rows not retrained; labels updated); §3.1, §3.2, §3.6; §5.4 experiments (not rerun; the
  fallback consequence is stated). References list markers resolved (see references_verified.md).

## Readability pass (R1.6)
- Applied only inside the sections above: shorter sentences, one number per sentence where possible, no
  em dashes introduced, British spelling. Untouched sections retain their original prose.
