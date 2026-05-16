# Phase 20 — Multi-agent review synthesis

**Date:** 2026-05-16
**Inputs:** four parallel subagent reviews of the draft `2026-05-16-phase20-design.md` + `2026-05-16-phase20-implementation.md`:

- **Architect** (`a72e1369814b54514`) — design coherence + Phase 19 pattern fidelity.
- **Codebase-grounding** (`a48d486afd689ac9a`) — verifies every "matches Phase 19 pattern" claim against actual `src/`.
- **Scientific-accuracy** (`ab0ee7d22b61057c5`) — line-by-line check against Marxan v4 master source (`clumping.cpp`, `computation.hpp`, `score_change.cpp`).
- **Independent-redesign** (`ad821c93f56e29cdd`) — clean-room counter-design.

The reviews agreed in spirit; they diverged on **two parity-critical math points** that the draft got wrong, and on **three plan-grounding bugs** that block mechanical execution. All patches land in the design + implementation plan in this commit.

---

## CRITICAL (parity-breaking — must fix before any code)

### C1 — Penalty formula is wrong

**Found by:** scientific-accuracy (verbatim source quote of `computation.hpp:15–27`).

Draft says:

```
penalty_j = baseline · SPF · max(0, (sepnum − count) / sepnum)        # linear
```

Marxan v4 `computeSepPenalty` is **hyperbolic**, not linear:

```c
fval = (double)ival / (double)itarget;          // count / sepnum
if (!ival) fval = 1.0 / (double)itarget;        // bump fval up so the hyperbola doesn't blow up
return (1 / (7 * fval + 0.2) - (1 / 7.2));
```

Properties:
- `fval = 1` (target met) → penalty = `0` exactly.
- `fval → 0` → penalty → `1/0.2 − 1/7.2 ≈ 4.86` ... **wait** — actually with the `count=0` bump to `fval = 1/sepnum`, the maximum value is bounded.
- For `sepnum=3, count=0`: `fval = 1/3`; penalty = `1/(7/3 + 0.2) − 1/7.2 = 1/2.533 − 0.139 ≈ 0.255`.
- For `sepnum=3, count=1`: `fval = 1/3`; same `0.255` (the count=0 bump is exactly `1/sepnum`).
- For `sepnum=3, count=2`: `fval = 2/3`; penalty = `1/4.867 − 0.139 ≈ 0.067`.
- For `sepnum=3, count=3`: `fval = 1`; penalty = `0`.

This is a different curve from `(sepnum − count)/sepnum` and the difference is observable in SA acceptance behaviour. Must fix.

**Resolution:** Replace the draft formula with `compute_sep_penalty(count, sepnum)` mirroring `computation.hpp:15` exactly. New helper in `pymarxan.solvers.separation`:

```python
def compute_sep_penalty(count: int, sepnum: int) -> float:
    if sepnum <= 0:
        return 0.0
    fval = count / sepnum if count > 0 else 1.0 / sepnum
    return 1.0 / (7.0 * fval + 0.2) - 1.0 / 7.2
```

Then per-feature total = `baseline_penalty_j · SPF_j · compute_sep_penalty(count_j, sepnum_j)`.

### C2 — Greedy ordering is **PU-id (insertion order)**, not descending amount

**Found by:** scientific-accuracy (reading `clumping.cpp::CountSeparation2`, `makelist`, `SepDealList`).

Draft says:

> Greedy order (TBC against Marxan source):
> - Likely sorted by descending `amount_ij` (priorities the largest contributors).

Marxan v4 does **no sort by amount**. `makelist` (`clumping.cpp:1173-1220`) builds the candidate list in `spec[isp].head` order — which is the order PUs were inserted as features were added during read time. `SepDealList` (`clumping.cpp:1225-1279`) walks this list head-to-tail and greedily admits any candidate that is ≥ `sepdistance` from all already-admitted.

For pymarxan this maps to **iterating selected candidates in PU-id ascending order** (since our `pu_feat_matrix` rows are PU-id indexed). This is the deterministic, parity-correct rule.

This also incidentally resolves the architect's "HIGH" tie-breaker concern — PU-id order is inherently stable.

**Resolution:** `count_separation` iterates candidates in ascending PU-id order. No `np.argsort` on amount needed. Bedrock test 4 ("greedy order regression") now has a fully deterministic expected output.

### C3 — Clamp `sep_count_j = min(actual, sepnum_j)`

**Found by:** scientific-accuracy (`clumping.cpp:1113-1116, 1167`).

Marxan's `CountSeparation2` short-circuits as soon as it has found `sepnum` separated PUs — it never returns more than `sepnum`. Counting higher doesn't reduce the penalty further (the hyperbolic curve plateaus at 0 for `fval=1`) but it does waste compute. pymarxan should match: stop the greedy scan as soon as `len(kept) == sepnum_j`.

**Resolution:** `count_separation` accepts a `sepnum` parameter and returns `min(actual_count, sepnum)`. Implementation-side, the greedy loop breaks early.

---

## HIGH (must fix before solver wiring)

### H1 — Deterministic penalty mask must exclude separation features

**Found by:** architect.

Phase 19 added this mask to `cache.compute_full_objective` (cache.py:379) and `compute_delta_objective` (cache.py:483):

```python
det_spf = self.feat_spf * (self.feat_target2 <= 0)
```

A feature with `target2 == 0` AND `sepnum > 1` would currently fire the deterministic `SPF * max(0, target*MISSLEVEL - held)` penalty AND have `SepState.delta_penalty` added on top — double-counting.

**Resolution:** Extend the mask to a compound:

```python
det_spf = self.feat_spf * (self.feat_target2 <= 0) * (self.feat_sepnum <= 1)
```

(The `<= 1` threshold, not `<= 0`, matches Marxan's "sepnum ≤ 1 is disabled" convention — see M3 below.) This is a single-line change at two sites in `cache.py`, in Task 7 of the plan.

### H2 — Type-4 + sepnum composition (routing)

**Found by:** scientific-accuracy.

Marxan's `NewPenalty4` (`clumping.cpp:783-930`) computes a **single** per-feature penalty term that absorbs both the clumping shortfall AND `computeSepPenalty(iseparation, spec[isp].sepnum)`. pymarxan computes them as two parallel terms (ClumpState + SepState). The numeric sum is **identical** as long as the compound mask in H1 prevents the deterministic path from double-counting. Both ClumpState and SepState compute their own per-feature contribution multiplied by `baseline·SPF`; summed they equal Marxan's single-term value.

**Resolution:** No structural change. Document the routing equivalence in the design doc so reviewers don't flag it again. The parallel pipelines are correct because (a) the mask in H1 prevents the deterministic path firing for type-4-and/or-sep features, and (b) each penalty path uses the same `baseline_penalty_j · SPF_j` scale.

### H3 — Task 10b is misleading: no "R6 gate" exists to remove

**Found by:** codebase-grounding.

`grep -r "sepnum\|sepdistance\|separation_active\|SepState" src/` returns zero matches. Phase 19 never shipped a `target2 > 0 AND sepnum > 0` NotImplementedError check — that "R6 gate" was a forward-looking name in the Phase 19 design that never materialised.

**Resolution:** Rewrite Task 10b as: "Add a regression test verifying `Solution.sep_shortfalls` AND `Solution.clump_shortfalls` are both populated when a problem has both `target2 > 0` and `sepnum > 1`. No guard removal — there is no guard."

### H4 — `_EDITABLE_INT_COLUMNS` validation bug

**Found by:** codebase-grounding.

`validate_feature_edit` in `feature_table.py` currently restricts `_EDITABLE_INT_COLUMNS` values to `{0, 1, 2}` (the clumptype rule). Adding `"sepnum"` to that tuple without splitting validation would wrongly reject `sepnum=3`.

**Resolution:** Split the int-validation logic. Either (a) introduce a per-column validator dict, or (b) special-case `clumptype` and let other int columns validate as non-negative ints. Approach (b) is simpler and matches the Phase 18 / Phase 19 pattern of growing the function in-place.

### H5 — `ProblemCache` dataclass field declarations are an explicit touch point

**Found by:** codebase-grounding.

`ProblemCache` is a `frozen=True` dataclass. The four new fields (`feat_sepdistance`, `feat_sepnum`, `pu_centroids`, `separation_active`) must be declared in the class body (cache.py around line 90), not just computed in `from_problem`.

**Resolution:** Task 7 description explicitly mentions both touch points (class-body declaration + `from_problem` computation).

---

## MEDIUM (worth doing in v1, not v2)

### M1 — `sepnum > 1` not `sepnum > 0` as the active threshold

**Found by:** independent-redesign (against Marxan source).

Marxan treats `sepnum ≤ 1` as disabled — a `sepnum=1` requirement is trivially met by any PU containing the feature (since one PU is always trivially "separated" from itself). The draft uses `sepnum > 0`; tightening to `sepnum > 1` matches Marxan exactly and avoids a degenerate "always-passes" branch.

**Resolution:** `separation_active = bool(np.any((feat_sepnum > 1) & (feat_sepdistance > 0)))`. Default `sepnum` value in `read_spec` becomes `1` (not `0`). Writer omits when `(sepnum <= 1).all()`.

### M2 — Three-tier PU coordinate resolution

**Found by:** independent-redesign.

Marxan reads `xloc` / `yloc` from `pu.dat` directly. pymarxan currently relies on `planning_units.geometry.centroid`. To support Marxan-format projects without a GeoDataFrame conversion, add a fallback:

1. If `has_geometry(problem)` (geometry column exists): use `planning_units.geometry.centroid.x` and `.y`.
2. Else if `planning_units` has both `xloc` AND `yloc` columns: use those.
3. Else raise `ValueError` with a clear message at `ProblemCache.from_problem` when `separation_active`.

**Resolution:** Helper `get_pu_coordinates(problem) -> np.ndarray (n_pu, 2)` in `pymarxan.solvers.separation` (not `pymarxan.spatial.coordinates` — keeps separation-specific logic local to the module that uses it).

### M3 — CRS sanity warning in `validate()`

**Found by:** independent-redesign + architect.

If `planning_units` has a geographic CRS (degrees, e.g. EPSG:4326) AND any feature has `sepdistance > 0`, distance values are in degrees — typically meaningless. Emit a `UserWarning` in `ConservationProblem.validate()`. Don't error — users may have reasons; just warn.

### M4 — `sepdistance == 0 AND sepnum > 1` should warn

**Found by:** architect.

If `sepdistance == 0` the constraint collapses (every pair is trivially ≥ 0 apart), so any selected PU "separates" from any other. `count_separation` returns `min(|candidates|, sepnum)`, which equals `sepnum` whenever the feature is at all represented — penalty is always 0. This is a no-op constraint; warn in `validate()`.

### M5 — Vectorised `count_separation`

**Found by:** architect.

The greedy loop must use a precomputed pairwise-distance matrix (or `scipy.spatial.cKDTree`) per call, not a Python-level pair scan. Per-flip cost in `SepState.delta_penalty` is otherwise prohibitive on large problems (architect estimated 4 s per SA run at 10⁶ iterations × 10 features × 200 candidates).

**Resolution:** `count_separation` precomputes `dist_matrix = scipy.spatial.distance.squareform(...)` once per call. Greedy admission then masks already-rejected candidates with vectorised boolean ops.

### M6 — MIP `socp` strategy is not a separation strategy

**Found by:** independent-redesign.

PROBMODE 3 has `"socp"` as a sensible (future) strategy because chance constraints are conic. Separation is purely combinatorial — `"socp"` makes no sense and should be rejected in `__init__`, not raised as `NotImplementedError` at solve time.

**Resolution:** `MIPSolver.__init__` validates `mip_sep_strategy in ("drop", "big_m")`; `"socp"` triggers a `ValueError` immediately. `"big_m"` still raises `NotImplementedError` at solve time (deferred to a future phase).

---

## LOW (defer)

### L1 — `Solution` is accumulating optional attrs

**Found by:** architect.

After Phase 20, `Solution` will carry six nullable analytics attrs: `prob_shortfalls`, `prob_penalty`, `clump_shortfalls`, `clump_penalty`, `sep_shortfalls`, `sep_penalty`. Any future constraint type will tip this into noise. A `SolutionMetrics` named-tuple as an optional `Solution.metrics` attr would be cleaner.

**Resolution:** Defer to v0.3. Listed in the realignment plan backlog as `SolutionMetrics refactor`.

### L2 — Incremental KD-tree `SepState`

**Found by:** independent-redesign + architect.

For problems with `|P_j| > ~500` selected candidates per feature, the per-flip full-recompute is the bottleneck. An incremental KD-tree-backed `SepState` that only re-evaluates the changed neighbourhood cuts per-flip cost from `O(n²)` to `O(n)`.

**Resolution:** Ship v1 with full-recompute. Add a benchmark; optimise only if `make bench` exceeds budget on a 5000-PU separation-active problem.

---

## Patches landed in this commit

- **`docs/plans/2026-05-16-phase20-design.md`** — rewritten formulation §, risks/assumptions updated, formula corrected to hyperbolic, ordering pinned to PU-id, three-tier coordinate resolution documented, `sepnum > 1` threshold adopted.
- **`docs/plans/2026-05-16-phase20-implementation.md`** — Task 4 pins ordering and adds the hyperbolic `compute_sep_penalty` helper, Task 7 explicitly lists dataclass declarations + compound det_spf mask, Task 10b rewritten (no guard removal), Task 11 documents the validation-split for `sepnum`, new Task 4b for `validate()` warnings, new Task 7b for the coordinate-resolution helper.

After patches, the plan is mechanically executable. Open questions OQ1–OQ5 from the draft are all resolved by the review pass.
