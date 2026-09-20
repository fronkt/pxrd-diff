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
