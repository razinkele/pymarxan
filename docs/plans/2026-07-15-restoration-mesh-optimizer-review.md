# restoration MESH optimizer — design review synthesis

**Date:** 2026-07-15
**Lenses:** Codebase-grounding (RAN the greedy, 10/11), Architect + Independent-redesign. (Science
settled in the MESH review.)

## Verdict

**Algorithm correct — the independent redesign reproduced it on every hard decision (recompute-all,
`current = best_mesh` anti-drift, `+inf` zero-cost guard, stop condition), and grounding verified
every hand-reasoned pick against running code.** But grounding caught a **real `criterion` bug** in
the plan's code and the architect caught a **HIGH result-shape gap**. Fold both before TDD.

## Fixes to fold

- **HIGH (grounding) — `criterion` is a no-op bug.** The plan's `optimize.py` scores
  `score = np.inf if cc == 0.0 else gain / cc` with **no branch on `criterion`**, so
  `criterion="gain"` behaves identically to `gain_per_cost` and fails the plan's own
  `test_gain_per_cost_vs_gain_differ` (proved: both give `{2,3}`; `gain` should give `{1}`).
  *Fix (confirmed by re-run):* `score = gain if criterion == "gain" else (np.inf if cc == 0.0
  else gain / cc)`.
- **HIGH (architect) — `mesh_curve` is step-indexed, not the cost-indexed budget–MESH frontier it's
  named.** Under non-uniform costs `step ≠ cost`, so a caller can't plot the frontier against budget
  or reconstruct a sub-budget plan — and the deferred `min_restore` dual ("smallest cost to reach
  MESH ≥ T") can't be read off it. Diverges from the cited `ZonationResult` precedent (which carries
  `removal_order` + a `prop_cost_remaining` cost axis). *Fix (additive):* add to the result, aligned
  to `mesh_curve` (length `n_restored+1`): `cost_curve: np.ndarray` (cumulative cost, `[0]=0.0`) and
  `order: list[int]` (PU indices in pick sequence). Both accumulate free in the loop.
- **MEDIUM (architect) — rename `RestorationResult` → `MeshRestorationResult`.** The generic name over
  MESH-only fields (`mesh`/`baseline_mesh`/`mesh_curve`) collides with the forecast IIC/PC optimizer
  (its own `compute_*`+result). Rename now (free, pre-ship); matches the `MeshResult`/`compute_mesh`
  naming; leaves `RestorationResult` open as a future base. (The deferred SA refiner is MESH-based so
  it can share `MeshRestorationResult`.)
- **LOW (architect) — document the grid-size ceiling.** `O(n_restorable² × n_pu)` full-grid
  relabeling per candidate; ~1s at 30×30/450-restorable but minutes at 100×100. Mirror
  `rank_removal`'s explicit docstring ceiling so callers don't hand it a national-scale raster.

## Not changed (verified fine)

Module-function + result-dataclass shape (matches `zonation.rank_removal`); benefit-cost greedy with
`gain_per_cost` default (reduces to `gain` under uniform cost); recompute-all every iteration
(required — MESH supermodular, gains can't be cached); calling `compute_mesh` directly (skips
`restore_mesh`'s per-call validation, sanctioned by problem.py); deferring the union-find delta and
the SA refiner; honesty that MESH is supermodular so no greedy variant has a (1−1/e) guarantee;
`budget<0` / bad-`criterion` `ValueError`s; parity untouched (pure new subpackage). No shared
greedy/SA eval helper needed — `compute_mesh(grid, existing | restored)` *is* the shared seam.
