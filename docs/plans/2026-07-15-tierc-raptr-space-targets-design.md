# Tier C — raptr-style space/adequacy targets — design

**Date:** 2026-07-15
**Status:** Approved (brainstorm), pending spec review → plan → design review (scientific lens
is load-bearing here). Full pipeline.
**Scope:** A first `pymarxan.adequacy` piece — the representative-**and-adequate** model:
per-feature *space targets* (spatial adequacy) on top of the existing *amount targets*
(representation), evaluated by `compute_space_held` and enforced as an SA/greedy penalty.
Named-competitor gap: **raptr** (Hanson et al. 2018, MEE, doi:10.1111/2041-210x.12862).

## ⚠ Scientific-accuracy gate

The **space-held** formula and **demand-point generation** below are my best reading of raptr;
they are the crux of correctness and MUST be verified by the design-review **scite lens** against
the raptr paper (doi:10.1111/2041-210x.12862) + the raptr package docs (and the surrogate
application doi:10.1073/pnas.1711009114) before TDD. If the paper's normalization differs, the
formula changes — treat the maths here as provisional. (Both DOIs verified real via scite; full
text not yet extracted.)

## Motivation

Marxan/pymarxan secures a target *amount* of each feature but is blind to spatial **adequacy** —
a feature can hit its amount target in one clustered corner of its range. raptr adds **space
targets**: the selected sites must be *spread across the feature's distribution in an attribute
space* (geographic and/or environmental), so the reserve is representative of the feature's
variation, not just its quantity. This closes the raptr gap and adds a genuinely new target type.

## The model

**Attribute space (per the brainstorm: general, default geographic).** Each PU has a position in
an `n_dim` attribute space:
- Default: **geographic centroids** via `get_pu_coordinates(problem)` (the S4a three-tier
  resolver: geometry → xloc/yloc → grid centroids), giving `(n_pu, 2)`.
- Optional: user-supplied **attribute columns** on `planning_units` (e.g. environmental variables),
  standardized (z-scored per column) so distances aren't dominated by one axis' units.

**Space targets.** A feature carries an optional **`space_target ∈ [0, 1]`** column on the
`features` DataFrame (a proportion; absent / all-zero = no space targets — behaviour unchanged),
alongside `target`/`spf`. The attribute-space configuration (which `planning_units` columns define
the space, `max_demand_points`, whether to z-score) is carried by a small **`SpaceSpec`** dataclass
passed to `compute_space_held` / the solver, defaulting to geographic centroids + z-scoring. All
attribute dimensions (geographic *and* user columns) are **z-scored** by default so no axis
dominates the distance metric. **[scite/design-review: confirm whether raptr z-scores the geographic
default.]**

**Demand points (per feature).** Discretize the feature's distribution into demand points in
attribute space. First cut: each PU where the feature occurs (amount > 0) is a demand point at
its attribute position, with **demand weight = the feature amount there**. (raptr samples points
from the feature's distribution; occupied-PUs-as-demand-points is a faithful, deterministic
discretization — the scite lens confirms whether raptr's sampling materially differs. An optional
`max_demand_points` sub-samples large distributions.)

## `compute_space_held` (the crux)

For feature `f` with demand points `D_f` (weights `w_d`, positions `p_d`) and the set `S_f` of
**selected** PUs carrying `f` (positions), define the assignment cost as the weighted squared
distance from each demand point to its **nearest selected PU carrying f**:
```
cost(S_f) = Σ_{d ∈ D_f} w_d · min_{i ∈ S_f} ‖p_d − pos_i‖²
```
**Space held** normalizes this between the best and worst achievable, so it is a proportion in
`[0, 1]` (1 = as adequate as selecting *all* relevant PUs, 0 = worst single PU):
```
space_held_f = clip( (cost_worst − cost(S_f)) / (cost_worst − cost_best), 0, 1 )
cost_best  = cost(all PUs carrying f)          # every demand point served by its own PU
cost_worst = max_i cost({i})                    # the single worst PU serves everything
```
Empty `S_f` (feature absent from the selection) → `space_held_f = 0`. `cost_worst == cost_best`
(degenerate: one candidate PU) → `space_held_f = 1` when `S_f` non-empty. **[scite-verify the
worst/best normalization — raptr may define held as a proportion-of-variation-captured with a
different denominator.]**

**Simplification for the occupied-PU discretization.** When demand points *are* the occupied PUs
(each `p_d` = that PU's attribute position), `cost_best = cost(all occupied PUs) = 0` — every
demand point is served by itself at distance 0. So the formula collapses to a clean, div-by-zero-
free measure:
```
space_held_f = 1 − cost(S_f) / cost_worst
```
i.e. `cost(S_f) = Σ_{d not selected} w_d · dist²(d, nearest selected occupied PU)` — how far the
*unselected* occupied cells sit from the nearest selected one (0 when all occupied cells are
selected → held 1). This is the concrete formula the implementation uses; the general worst/best
form above is the fallback if the scite lens shows raptr sub-samples demand points (making
`cost_best ≠ 0`).

Returns a `dict[feature_id, space_held]`. Pure numpy; `O(|D_f| · |S_f|)` per feature (nearest of
selected). `pos` in the standardized attribute space.

## Solver integration (SA / greedy penalty; MILP deferred)

A **space-target penalty** parallel to the amount penalty, added to the objective:
```
space_penalty = Σ_f  spf_f · max(0, space_target_f − space_held_f)
```
(`spf_f` reuses the feature's SPF, or a dedicated `space_spf`; scite/design-review decides.)
- **Greedy** evaluates candidates via the full objective → the space penalty slots in directly.
  **[verify in the plan/grounding: whether pymarxan's greedy scores candidates by full objective or
  by a marginal amount-benefit heuristic — the space integration is a clean slot-in only for the
  former; if marginal, greedy needs a marginal-space-held term, which shifts the "greedy is simpler
  than SA" scope-fallback.]**
- **SA**: the space term is added to `compute_full_objective`. Because `space_held` is a
  nearest-facility function (no cheap additive delta — flipping a PU re-assigns demand points), an
  exact per-flip delta is **deferred**; when any feature has a space target, the SA path recomputes
  the affected features' `space_held` on a flip (or full-objective evaluation), a documented perf
  cost. An incremental `SpaceState` (mirroring the existing Phase-20 `SepState` geographic-delta
  machinery) is the natural Phase-B optimization.

Non-space problems pay **zero** cost (no `space_target` column / all zero → the whole path is
skipped), so the 35.0 min-set anchor and all existing solver behaviour are untouched.

Out of scope (deferred): the exact p-median MILP facility-location formulation (per the brainstorm);
the incremental `SpaceState` SA delta; raptr's probabilistic MVNorm space-usage variant.

**Scope fallback.** The load-bearing, verifiable deliverable is the scientifically-grounded model +
`compute_space_held`. If the design review (scite or architect) finds the space-held maths needs
substantial rework, or the SA full-objective-recompute integration is too large/risky for one
piece, this ships as **Phase A = model + demand points + `compute_space_held` + greedy penalty**
(greedy is a clean full-objective fit), with **SA integration as Phase B**. The science gets
verified either way before any solver code lands.

## Testing strategy (TDD)

- **`compute_space_held` bounds + monotonicity:** all-PUs-selected → `1.0`; empty selection → `0.0`;
  adding a PU that improves coverage never *decreases* space held; a clustered selection scores
  lower than a spread one on the same amount.
- **Attribute space:** geographic default (via `get_pu_coordinates`) works on a vector, xloc/yloc,
  and grid problem; user attribute columns are z-scored (a large-unit column doesn't dominate).
- **Demand points:** occupied-PU discretization; `max_demand_points` sub-sampling is deterministic.
- **Penalty + solver:** a problem with a space target that a clustered optimum violates gets a
  higher objective for the clustered solution than for a spread one; greedy/SA prefer the spread
  reserve; a problem with **no** space target is bit-identical to today (35.0 anchor via
  `validate_marxan_parity.py`).
- **Scientific:** `compute_space_held` on a hand-worked tiny example matches the raptr definition
  the scite lens confirms.

**Target:** ~14–20 tests, `make check` green, parity 35.0 intact. Scientifically validated by the
design-review scite lens before merge.

## References

raptr: Hanson, Rhodes, Fuller & Possingham (2018) *Methods Ecol. Evol.*
doi:10.1111/2041-210x.12862 (**scite-verify the space-held maths**). Surrogate application:
doi:10.1073/pnas.1711009114. Reuses `solvers/separation.py::get_pu_coordinates` (S4a three-tier
attribute-space default) and the SPF/penalty machinery. Precedent for a geographic SA-delta:
Phase-20 `SepState` (`solvers/separation.py`).
