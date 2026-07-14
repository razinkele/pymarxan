# Zonation Phase A design review — synthesis

**Date:** 2026-07-14
**Reviewed:** `2026-07-14-zonation-phase-a-design.md` + `...-implementation.md`
**Method:** four parallel perspectives (architect, codebase-grounding, scientific-accuracy via scite + the open Moilanen 2007 PDF, independent re-design).

## Verdict

**Approve to implement after folding in the findings below.** The architecture,
CAZ math, cost/rank/status/warp semantics, and both hand-oracles are verified
correct — the codebase-grounding agent materialized the plan's verbatim code and
ran ruff/mypy/all-12-tests, and the independent re-designer reproduced all eight
test scenarios and converged on the plan's core loop. Two findings need real
changes (one CRITICAL-as-labeled, one HIGH), plus an empirically-confirmed lint
fix and untested-guard coverage.

## Findings absorbed

### CRITICAL (scientific) — ABF is mislabeled, not mis-computed
The plan calls ABF `δ_i = Σ_j(w_j·q_ij/Q_j)` a "linear benefit function; the
Moilanen-2007 default" and "verified against Moilanen 2007." That labeling is
**wrong**: with `q_ij/Q_j = (q_ij/T_j)/R_j`, this formula is the *proportional /
logarithmic* marginal-benefit member of Moilanen's ABF family (`B'_j(R)=1/R`),
**not** linear. Strictly-linear ABF uses the *original* total `T_j`
(`Σ_j w_j·q_ij/T_j`) and is **static** — the marginal loss never changes as cells
are removed, collapsing ABF to a one-shot sort (the independent agent's decisive
point; the scientific agent read the power-benefit form `w_j·z·R_j^(z-1)·q_ij/T_j`
from the primary PDF). **Resolution (both agents accept this):** keep the
adaptive `Q_j` formula (it is the standard Zonation-software ABF and switching to
static `T_j`-linear would be a regression), but **drop the "linear," "Moilanen-2007
default," and "verified against Moilanen 2007" claims for ABF**, describe it as the
proportional/remaining-sum ABF, and note the concave power-benefit generalization
(`R_j^z`, user-set exponent) as a documented future extension. CAZ's "verified
against Moilanen 2007 Eq. 1a" is confirmed correct and stays.

### HIGH (independent) — performance-curve x-axis is by-count, ranking is by-cost
When `use_cost=True` the removal order is `δ/cost`, but the curve's
`prop_landscape_remaining` counts *cells*, so "retained X% after removing 50% of
the landscape" silently means 50%-by-count, not 50%-by-budget — misleading exactly
the variable-cost users, and it's a `ZonationResult` schema decision (breaking to
change later). **Fix:** add a `prop_cost_remaining` column (Σ remaining cost / Σ
total cost, using the ranking's cost vector) alongside `prop_landscape_remaining`,
always present. No breaking change later.

### MEDIUM (codebase-grounding, EMPIRICALLY CONFIRMED) — import order fails ruff
The plan's `from pymarxan.models.problem import (ConservationProblem,
STATUS_LOCKED_IN, STATUS_LOCKED_OUT)` produces `I001` under ruff's
`order-by-type=true` (verified by running ruff; convention: `spatial/wdpa.py:13`).
**Constants must come before the class:** `STATUS_LOCKED_IN, STATUS_LOCKED_OUT,
ConservationProblem`. (This corrects a wrong edit made earlier in planning.)

### MEDIUM (independent) — two numeric guards are untested
`T_j=0` (feature in zero PUs → retained 1.0, excluded from δ) and mid-run
extinction (`Q_j` hits 0 while cells remain → the `pos=Q>0` guard) are the fiddly
float edge cases; the plan implements both but tests neither. **Add two tests.**
(+2 tests → 14 total.)

### MEDIUM (architect) — perf hygiene + document the complexity contract
Allocating a full-width `contrib = np.zeros((cand, n_feat))` and scatter-filling
each removal is avoidable churn; compute δ directly on the candidate slice
(`r = q[cand]*(w/Q_safe); r.max(1)/r.sum(1)`), and `Q -= q[idx]` in place. Add a
docstring note: the O(n²·n_feat) recompute is *inherent* (removing a cell shifts
every `Q_j`, so the Marxan per-flip delta model doesn't apply), `warp` is the
scaling knob, and this suits vector PUs (hundreds–thousands), not million-cell
rasters (the separate raster-PU gap). A future maintainer must not "optimize" the
recompute into incorrectness.

### LOW (absorbed)
- Cross-reference `analysis.rank_importance` (Jung 2021, selected-PUs, Marxan
  objective) vs `zonation.rank_removal` (all-PUs, biological loss) in both
  docstrings — same "rank" word, different questions.
- The warp test asserts exact `removal_order` equality, stricter than the design's
  "coarse bucket agreement"; it passes on P1 coincidentally — add a comment that P1
  is order-preserving, don't over-generalize.
- Document that `priority_rank` values are unique by construction (so
  `top_fraction` is deterministic) and that curves include locked-out
  (unprotectable) stock.

## Not absorbed (with reason)
- **Switch ABF denominator to `T_j` / add a `z` exponent now** — `T_j`-linear is a
  static-sort regression; a `z`-parameterized power benefit is the faithful general
  form but over-scopes Phase A. Deferred as a documented future extension; the
  accurate relabel resolves the CRITICAL without it.
- **A `bench` marker test** — phylo/temporal/rivers set the precedent of no bench
  test for a new subpackage; the complexity docstring note covers the intent.
- **Mask-out locked-out cells instead of remove-first** — the independent agent
  showed the normal/locked-in ranking is identical either way, and remove-first
  composes better with `top_fraction`/Phase B. Keep remove-first; document it.
