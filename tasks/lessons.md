# Lessons

(Captured corrections and validated patterns. Update after every user correction or
successful surprising approach. Use format:
  ## YYYY-MM-DD — short title
  - Mistake / Pattern: what happened
  - Rule: what to do differently / keep doing
  - Context: when this rule applies
)
## 2026-09-20 — JAC referee 3 read the code and found three real defects (HAT5032)
- Mistake / Pattern: (a) a "no-true-lattice" arm silently substituted the TRUE lattice for
  the 3.9 % of patterns the indexer could not index; (b) pymatgen's ATOMIC_SCATTERING_PARAMS
  were dropped into a bare Gaussian sum although they belong to f = Z − 41.78 s² Σ a e^{−b s²}
  (Si: f(0) = 5.8 instead of 14) and a 0.96 pattern-Pearson "validation" hid it; (c) in
  x0-residual mode the Debye loss consumed the residual tensor, not noisy + residual; (d) the
  manuscript quoted lr 5e-4 while the checkpoint args say 1e-3.
- Rule: before ANY submission, run a consistency audit of code against the paper's equations
  and experimental conditions: (1) every "no-oracle" arm asserts zero oracle leakage and
  scores uncovered rows as misses, with the count in the log AND the per-sample file;
  (2) any borrowed coefficient table is checked NUMERICALLY against its source implementation
  at a known point (f(0) must equal Z), not by a downstream correlation; (3) every auxiliary
  loss consumes the same reconstructed tensor as the primary loss (assert identity, not intent
  in a comment); (4) hyperparameters in the text are read from the checkpoint's stored args,
  never from memory. A verify_claims-style script pinning manuscript numbers to artefacts
  (as in mlip-dynstab) would have caught (a) and (d).
- Context: any paper whose main claim is "why the method fails". An implementation error is
  indistinguishable from a physical cause until the audit is done; referees will do it for you.

## 2026-09-20 — Recompute before you apologise
- Pattern: the fallback defect (a) looked like it could contaminate the headline; the
  per-structure flags showed 0/117 fallback structure-seeds matched, so the McNemar result
  stood and only the denominator changed (46/3000 → 46/2883).
- Rule: when a defect is found, quantify its effect from the per-sample record first, then
  write the disclosure with the measured effect. Always ship per-sample flags with results
  so this recomputation is possible without a GPU.
- Context: revision letters; disclosure paragraphs.

## 2026-09-21 — A "best figure of merit" is not the classical rule
- Pattern: the first unknown-crystal-system indexer kept the highest M20 across lattice types
  and scored 24.3 % strict with cubic chosen for 12/238 cubic structures. M20 has no penalty for
  free parameters, so a 4-parameter monoclinic sub-cell out-scores the true cubic cell on a few
  exact lines. The TREOR/DICVOL acceptance rule (highest symmetry first, all lines indexed,
  M20 ≥ 10, system from metric symmetry) gives 43.4 % on the same patterns.
- Rule: when implementing a classical baseline, reproduce the program's decision rule, not just
  its figure of merit; before reporting a negative result for a classical method, audit one
  failing case by hand (a subagent replay of 110 structures settled it in 15 min).
- Context: any "classical control" in an ML paper; referee 1's point 5(iii).

## 2026-09-21 — Test the mechanism you wrote before a referee asks
- Pattern: the paper's central reading (global pooling discards d-spacings) was withdrawn by a
  2-hour frozen-encoder ablation on the laptop CPU; the retrain then erased the "indexer beats
  learned head" claim (0 → 1.9 %, McNemar p 1.5e-8 → 0.35). Both were cheap to test before
  submission.
- Rule: every mechanistic sentence in an abstract needs a controlled test in the results, run
  before submission; when a code defect touches training, re-measure every affected claim
  rather than disclose-and-keep. Budget the GPU hour for it up front.
- Context: diagnostic papers; §5.6 / §5.2 of hat5032.

## 2026-09-21 — Do not share a 32 GB GPU between the sampler and a side job
- Pattern: 03_sample.py held 26.6 GB; launching the 30-epoch head training beside it risked an
  OOM that the pipeline would have recorded as a silent rc≠0 and skipped. Killed the side job
  and queued it after PIPELINE DONE (12 min, no loss).
- Rule: on a rented box, queue side jobs behind the pipeline unless nvidia-smi shows ≥ 2× the
  side job's peak free; monitor filters must include "out of memory" (torch's text), not only
  "OOM"; make pipeline stages fail loudly (status line with rc) so a skipped run is visible.
- Context: vast.ai runs with a chained pipeline.sh.

## 2026-09-26 — pre-upload audit of the JAC revision
- **Every quoted result must be traced to a released file before upload, not just the new ones.** The four review agents
  recomputed all revision numbers (all matched) but the most-cited number in the paper (the "0.5 Å knee", §5.5) came from
  the submitted version and matched no record (n = 200 / 5.6 % vs the record's n = 300 / 2.0 %). Rule: before any upload,
  grep every percentage/count in paper.md and map each to a JSON/log path; anything unmapped is a defect to disclose.
- **Captions are claims.** Three captions described figures that were not the ones shipped (CIs not drawn, "seven runs" for
  six, "below the knee" for 0.53–0.96 Å bars) and one in-figure title carried a withdrawn claim. Rule: after regenerating
  figures, read each PNG and check every clause of its caption against it; never keep suptitles inside figure files.
- **"Three seeds" needs its noun.** Sampling seeds on one training run are not training seeds; say which, once, in the
  abstract and in every table caption that pools them.
- **Hedge consistently across sections.** A hypothesis labelled as such in §5.6 was stated as the finding in the synopsis,
  abstract and §8. Rule: grep the abstract/synopsis/conclusion for every causal verb and confirm the body uses the same
  strength.
- **Public-surface actions are Frank's.** Zenodo publish was (rightly) blocked by the classifier; prepare the draft fully and
  hand over the publish click, and cite the reserved DOI so the paper does not need a second edit.
- **A file open in Word blocks the docx rebuild.** Close that one document through the Word COM object (Documents by
  FullName) rather than killing WINWORD — other documents were open.

