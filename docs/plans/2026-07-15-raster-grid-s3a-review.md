# Raster-grid PUs — S3a multi-agent design review — synthesis

**Date:** 2026-07-15
**Reviewed:** `2026-07-15-raster-grid-s3a-design.md` + `...-s3a-implementation.md`
**Lenses:** Architect, Codebase-grounding+parity (RAN the parity harness), Independent re-design.

## Verdict

**Architecturally sound; one HIGH mislabel (delta bit-identity) + doc/test refinements.** The
grounding agent empirically proved the exact claims on the actual repo (build_csr==dense incl.
duplicate/unknown rows; delta==dense at `n_feat=4` over 12 problems; `apply_flip_to_held`
bit-identical across 1456 swap pairs; probmode gating; CSC `feat_uses_pu` incl. cancellation/
negative edges; all 4 call sites; `compare=False` necessity; `validate_marxan_parity.py` →
MIP 35.0, SA 43, greedy 45). The independent re-design converged point-for-point AND caught the
one real defect the `n_feat=4` test regime hid.

## Findings folded in

- **HIGH — delta is NOT bit-identical at raster `n_feat` (mislabel + blind test).** The per-nonzero-
  column `np.dot(_det_spf[cols], Δsf[cols])` omits exact-zero terms but regroups the retained
  terms into a different (non-associative) reduction vs the full-width dot. `n_feat=4` → 0 diffs;
  `n_feat=64/512/2000` with float targets/SPF → ~38% differ by ≤ a few ULP (same-sign → bounded
  drift, worst rel-diff 6.3e-16; harmless — anchor integer-exact, MIP unaffected, `compute_held`
  already drifts identically). *Folded:* (a) relabel the **delta** everywhere from "bit-identical"
  to "identical for integer-amount problems; ≤ a few ULP for arbitrary floats (same float-sum
  regime as `compute_held`)"; (b) the delta test uses a **relative tolerance**, not `==`; (c) add
  a **wide-`n_feat` (≥128)** plain-problem delta test so the raster regime is actually covered.
  `apply_flip_to_held` stays genuinely bit-identical (element-wise scatter, no dot regrouping —
  grounding-verified) and keeps its `==` test.

- **MEDIUM (architect) — densify scale-cliff undocumented.** A clumping/separation/probmode-3
  problem at raster scale materializes the dense matrix on first `pu_feat_matrix[:, j]` access —
  reverting to the ~2.4 GB footprint and now holding CSR **and** dense (net *more* than before).
  *Folded:* documented as a KNOWN LIMITATION in the design + the property docstring, and a
  one-time `warnings.warn` on the first densify above a size threshold.

- **MEDIUM (architect/independent) — the raster-scale story is only "plain SA/II".** `clumping.py`
  (its own `build_pu_feature_matrix` at :295), `analysis/{ferrier_importance,irreplaceability}`,
  and `zones/cache.py` (separate dense `pu_feat_matrix`) still densify. *Folded:* stated explicitly
  in the design's out-of-scope so the CHANGELOG/story is honest.

- **LOW — `compare=False` / "hashable" wording.** The cache is *already* non-hashable (its ndarray
  fields), so `compare=False` on the CSR is hygiene (scipy `==` returns a sparse matrix; `hash`
  raises), not something that "makes it hashable". *Folded:* reworded the rationale; the design's
  "hashable/eq-safe" test line → "construction-safe" (the impl test already only asserts
  construction doesn't raise — kept).

- **LOW — `sum_duplicates()` is load-bearing for `apply_flip_to_held`.** `held[cols] += sign*data`
  needs unique `cols` per row (numpy fancy in-place add is last-wins, not accumulate).
  `build_pu_feature_csr` calls `sum_duplicates()`; *folded:* a precondition comment on both.

- **LOW — polish.** `compute_held` uses `np.flatnonzero(selected)` (unambiguous across scipy
  versions) instead of boolean row indexing; `build_pu_feature_csr` gets a real return annotation
  (`TYPE_CHECKING` import, no import-cost); update the `cache.py` class Attributes block +
  inverse-index-discipline note (CSR is source of truth, `pu_feat_matrix` now a property,
  `feat_uses_pu` from CSC); add edge tests (feature in no PU, PU with no features, negative
  amount); the no-densify test also calls `compute_full_objective` (also once per run); soften the
  CHANGELOG "no change to solver results" to the integer-exact / float-sum-order nuance.

## Noted, no action

- Option (b) — a full-width O(n_feat) delta for true bit-identity — was considered and rejected:
  `compute_held` already reorders float sums, so float-trajectory bit-reproducibility isn't
  achievable regardless, and the O(nnz) delta is faster with only bounded sub-ULP drift. Keep O(nnz)
  + honest labeling.
