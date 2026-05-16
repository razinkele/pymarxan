# Phase 20 — Implementation Plan (v4, post-three-rounds-of-review)

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans. Each task TDD: failing test → minimal impl → passing test. Commit after each batch.

**Goal:** Ship per-feature SEPDISTANCE / SEPNUM ("separation distance") across SA, IterativeImprovement, MIP-drop, heuristic (post-hoc reporting only). Same shape as Phase 19 — pure-functional math + mutable `SepState` companion + post-hoc Solution attrs.

**Design doc:** `2026-05-16-phase20-design.md` (v3)
**Round-1 review:** `2026-05-16-phase20-review.md` (Marxan-source parity bugs + plan grounding)
**Round-2 review:** `2026-05-16-phase20-review-round2.md` (adversarial, integration, UX, performance)
**Round-3 review:** `2026-05-16-phase20-review-round3.md` (failure-mode, test-design, compatibility, maintainability)
**Tech stack:** Python, NumPy, scipy, pandas, pytest.

## Batch 0 — Cross-phase observability fix (1 task, ships with Batch 1)

### Task 0: `read_spec` unrecognised-column whitelist warning

**Round-3 H10 fix.** Catches typo'd column names (`sepnnum`, `targt2`, `ptraget`, `clumptpe`) for Phases 18 + 19 + 20 in one shot. ~8 LOC in `io/readers.py`:

```python
_KNOWN_SPEC_COLS = {"id", "name", "target", "prop", "spf", "ptarget",
                    "target2", "clumptype", "sepdistance", "sepnum"}
extra = set(df.columns) - _KNOWN_SPEC_COLS
if extra:
    warnings.warn(
        f"spec.dat has unrecognised columns {sorted(extra)} — these will be "
        f"ignored. Did you mean one of {sorted(_KNOWN_SPEC_COLS)}?",
        UserWarning, stacklevel=2,
    )
```

Test: `tests/pymarxan/io/test_io_separation.py::test_unrecognised_spec_columns_warn` — read a spec with `sepnnum` column, assert `UserWarning` fires.

## Batch 1 — Schema, I/O, and validation (5 tasks)

### Task 1: `spec.dat` accepts optional `sepdistance` column

Float ≥ 0; default `0.0`. `read_spec` coerces dtype, fills default, AND validates non-negativity — `if (df["sepdistance"] < 0).any(): raise ValueError("sepdistance values must be >= 0; got negative at feature_id [...]")` (round-2 H6 fix). `write_spec` omits when `(df["sepdistance"] == 0).all()`. Identical pattern to the existing `ptarget` / `target2` handling, plus the new negative-value guard.

### Task 2: `spec.dat` accepts optional `sepnum` column

Int ≥ 0; default `1` (Marxan's "disabled" sentinel, not 0). `read_spec` coerces dtype, fills default, rejects negative values with a clear `ValueError`, AND rejects non-integer floats with `if not (df["sepnum"] == df["sepnum"].astype(int)).all(): raise ValueError("sepnum must be integer; got non-integer at feature_id [...]")` (round-2 M6 fix — prevents `2.7` silently becoming `2`). `write_spec` omits when `(df["sepnum"] <= 1).all()`. This is a distinct validation block from `clumptype`'s `{0, 1, 2}` set-membership check — do NOT reuse that block.

### Task 3: `validate()` smoke test

Confirm `validate()` accepts the new columns without raising; identical mechanical pattern to Phase 18 + Phase 19 schema-validation tests.

### Task 4: Warnings for no-op configurations (in `ProblemCache.from_problem`, NOT `validate()`)

**Round-2 CR2 fix + round-3 CR4 visibility fix.** The v2 plan put these warnings in `validate()`, but `validate()` only fires on Shiny upload — a user editing the feature_table grid never triggers it. Move them to `ProblemCache.from_problem`, which runs on every solve.

`ProblemCache.from_problem` emits `UserWarning` via `warnings.warn(msg, UserWarning, stacklevel=2)` when:
- Any feature has `sepdistance > 0` AND the GeoDataFrame has a geographic CRS — distance values in degrees are nearly meaningless. Detect via `crs.is_geographic` when CRS is present.
- Any feature has `sepnum > 1` AND `sepdistance == 0` — constraint trivially satisfied.

**Round-3 CR4 fix**: use `stacklevel=2` so the source location points to user code, not pymarxan internals (matches Phase 19's `compute_baseline_penalty` precedent). Document in `help_content.py` that pymarxan emits `UserWarning` from `ProblemCache.from_problem` and that strict-mode users (`python -W error` or pytest `filterwarnings = error::UserWarning`) should filter explicitly: `warnings.filterwarnings("always", category=UserWarning, module="pymarxan")`.

Then in `run_panel._run_solver`, wrap the solve in `warnings.catch_warnings(record=True)` and replay each captured `UserWarning` via `ui.notification_show(message=str(w.message), type="warning")` so the user actually sees them. Same surface the Phase 19 `compute_baseline_penalty` warning uses.

Test (round-3 CR4): `tests/integration/test_phase20_smoke.py::test_geographic_crs_emits_warning` — `with pytest.warns(UserWarning): ProblemCache.from_problem(geographic_crs_problem)`.

### Task 5: `get_pu_coordinates(problem)` helper with NaN guard + custom exception

New helper in `pymarxan.solvers.separation`. Define a custom exception subclass at the top of the module (round-3 M8 fix — lets `build_solution` catch only this specific class, not all `ValueError`):

```python
class PUCoordinatesUnavailableError(ValueError):
    """PU coordinates required for separation evaluation but not derivable
    from the problem (no geometry, no xloc/yloc, or NaN/invalid centroids)."""
```

`get_pu_coordinates` returns `np.ndarray (n_pu, 2)` of float64 coordinates. Three-tier resolution with NaN protection (round-2 H3 fix):

1. If `problem.has_geometry()`:
   - Compute `coords = np.column_stack([gdf.geometry.centroid.x, gdf.geometry.centroid.y])`.
   - If `np.isnan(coords).any()`: raise `PUCoordinatesUnavailableError("PU geometry contains N empty or invalid rows at indices [...]; cannot compute centroids for separation")`.
   - Return `coords`.
2. Else if `planning_units` has both `xloc` AND `yloc` columns:
   - Build the same array from those columns.
   - Same NaN guard.
3. Else raise `PUCoordinatesUnavailableError("PU coordinates required for separation-active problems. Either pass a GeoDataFrame planning_units with a geometry column, or include xloc/yloc columns. See pymarxan.spatial.importers.import_planning_units for converting Marxan-format pu.dat with coordinates.")`.

Module-level `__all__` exports (round-3 compat §6): `["compute_sep_penalty", "count_separation", "compute_sep_penalty_from_scratch", "evaluate_solution_separation", "get_pu_coordinates", "SepState", "PUCoordinatesUnavailableError"]`. **Not** re-exported at `pymarxan.solvers` package level — matches Phase 19 `clumping` module pattern.

**Commit message:**
```
Phase 20 Batch 1: SEPDISTANCE + SEPNUM schema, I/O, validation
```

## Batch 2 — Separation math (4 tasks)

### Task 6: `compute_sep_penalty(count, sepnum)` — Marxan hyperbolic curve

New module `src/pymarxan/solvers/separation.py`. Implements verbatim from `computation.hpp::computeSepPenalty:15-27`:

```python
def compute_sep_penalty(count: int, sepnum: int) -> float:
    if sepnum <= 0:
        return 0.0
    fval = count / sepnum if count > 0 else 1.0 / sepnum
    return 1.0 / (7.0 * fval + 0.2) - 1.0 / 7.2
```

Tests:
- `count == sepnum` → exactly `0.0` (allow ≤ 1e-12 floating tolerance).
- `count == 0, sepnum == 3` → ≈ `0.255` (hand-computed against the formula).
- `count == 2, sepnum == 3` → ≈ `0.067`.
- Monotone non-increasing in `count` for fixed `sepnum`.

### Task 7: `count_separation` (Marxan-faithful greedy admission)

```python
def count_separation(
    selected: np.ndarray,        # (n_pu,) bool
    feat_amounts: np.ndarray,    # (n_pu,) float for one feature
    pu_coords: np.ndarray,       # (n_pu, 2) float
    sepdistance: float,
    sepnum: int,
) -> int:
    """Greedy admission in ascending PU-id order; capped at sepnum.

    Mirrors clumping.cpp::CountSeparation2 + makelist + SepDealList.
    Allocates a k x k squared-distance matrix on the CANDIDATE sub-array
    only (NOT the full n_pu x n_pu — round-2 CR1 fix). Memory and CPU
    per call scale with the selection footprint, not problem size."""
```

Behaviour:
- Candidates `C = np.where(selected & (feat_amounts > 0))[0]` are already in ascending PU-id order (since `np.where` is sorted).
- Compute `dist_sq = scipy.spatial.distance.squareform(pdist(pu_coords[C], "sqeuclidean"))` — **k×k allocation only, never n_pu×n_pu**. With early-exit at `|kept|==sepnum`, the typical call doesn't even materialise the full k×k.
- For each candidate `i` in `C` (ascending PU-id), admit if `dist_sq[i_local, kept_local].min() >= sepdistance**2` (vectorised). The mask `kept_local` is a small bool array of length `|C|`.
- Stop and return `len(kept)` as soon as `len(kept) == sepnum`.

**Critical:** do NOT precompute a global `(n_pu, n_pu)` distance matrix on `ProblemCache`. At n_pu=50000 that's 20 GB. Per-call k×k is bounded by the selection footprint (rarely > 1000 PUs containing a single feature simultaneously).

Tests: three hand-computed scenarios (linear arrangement, square layout, no candidates); `sepdistance=0` returns `min(|candidates|, sepnum)`; ordering regression with two equal-amount candidates verifies PU-id (not amount) is the tiebreaker.

### Task 8: `compute_sep_penalty_from_scratch` and `evaluate_solution_separation`

Reference impl + post-hoc evaluator paralleling Phase 19's `compute_clump_penalty_from_scratch` / `evaluate_solution_clumping`. Uses the same `baseline_penalty_j` from Phase 19's `compute_baseline_penalty`. Per-feature loop, multiply each `compute_sep_penalty(count_j, sepnum_j)` by `baseline_j · SPF_j`, sum.

### Task 9: Wire into `build_solution` (with narrow try/except for coordinate-less heuristic users)

`build_solution` in `solvers/utils.py` gains a third optional block (parallel to PROBMODE 3 and TARGET2). **Round-2 H2 fix + round-3 M8 narrowing**: catch only `PUCoordinatesUnavailableError`, not all `ValueError`. Internal bugs in `count_separation` raising `ValueError` propagate; only the "no coordinates" case is swallowed gracefully.

```python
if "sepnum" in p.features.columns and (p.features["sepnum"] > 1).any():
    from pymarxan.solvers.separation import (
        evaluate_solution_separation,
        PUCoordinatesUnavailableError,
    )
    try:
        sep_short, sep_pen = evaluate_solution_separation(p, selected)
        sol.sep_shortfalls = sep_short
        sol.sep_penalty = sep_pen
    except PUCoordinatesUnavailableError as exc:
        import warnings
        warnings.warn(
            f"Separation evaluation skipped: {exc}",
            UserWarning,
            stacklevel=2,
        )
        # sep_shortfalls / sep_penalty stay None; deterministic solution still returns.
```

Single source of truth — every solver path inherits.

**Commit message:**
```
Phase 20 Batch 2: separation math (Marxan hyperbolic penalty + greedy admission)
```

## Batch 3 — Solver integration (5 tasks)

### Task 10: `ProblemCache` stores separation params

**Three-part change** + round-2 additions.

**First**, add five new fields to the `@dataclass(frozen=True)` class body (cache.py around line 90, parallel to the Phase 19 clumping fields):

```python
feat_sepdistance: np.ndarray         # (n_feat,) float64
feat_sepnum: np.ndarray              # (n_feat,) int32
pu_coords: np.ndarray                # (n_pu, 2) float64
separation_active: bool
pu_to_sep_feats: list[np.ndarray]    # PU idx → sep-active feature column indices
```

The new `pu_to_sep_feats` (round-2 H5 fix) is the inverse index: `pu_to_sep_feats[i]` is the array of separation-active feature column indices that contain PU `i`. Built once at `from_problem`. `SepState.delta_penalty` iterates this for O(features-at-PU) outer cost instead of O(n_feat).

**Second**, compute them in `from_problem`:
- `separation_active = bool(np.any((feat_sepdistance > 0) & (feat_sepnum > 1)))`.
- Call `get_pu_coordinates(problem)` only when `separation_active`; raise its `ValueError` when needed.
- Build `pu_to_sep_feats` as `[np.where(pu_feat_matrix[i, sep_mask] > 0)[0] for i in range(n_pu)]` where `sep_mask = (feat_sepdistance > 0) & (feat_sepnum > 1)`.
- **Round-2 CR2 fix**: emit `UserWarning` via `warnings.warn` when (a) geographic CRS + any `sepdistance > 0`, OR (b) any feature has `sepnum > 1` AND `sepdistance == 0`.
- Default-inactive when columns missing.

**Third**, extract the deterministic-penalty mask as a `functools.cached_property` (round-3 H14 fix — eliminates the "edit-one-of-two" bug class):

```python
from functools import cached_property

@cached_property
def _det_spf(self) -> np.ndarray:
    """Deterministic-path SPF mask — features that get the standard
    SPF · max(0, target·MISSLEVEL − held) penalty path. Excludes
    features handled by ClumpState (target2 > 0) and SepState (sepnum > 1)
    to prevent double-counting. When adding a new constraint type, add its
    exclusion factor here, not at every call site."""
    return self.feat_spf * (self.feat_target2 <= 0) * (self.feat_sepnum <= 1)
```

Replace the two former mask call sites in `compute_full_objective` (cache.py:379) AND `compute_delta_objective` (cache.py:483) with `self._det_spf`. Round-1 H1 fix — prevents double-counting when a feature has only `sepnum > 1` set.

**Fourth (round-3 H15)**, add a module-level docstring on `ProblemCache` documenting the "inverse-index discipline":

```
When adding a new constraint type (Phase 21 importance, Phase 22 boundary,
etc.), precompute the inverse PU→feature index at `from_problem` time as
a frozen-dataclass field. NEVER do `np.where(pu_feat_matrix[idx] > 0)` per
flip in the SA hot loop — it is O(n_feat) per flip and silently blows the
performance budget. Pattern: see `feat_uses_pu` (Phase 19 clumping) and
`pu_to_sep_feats` (Phase 20 separation).
```

### Task 11: `SepState` mutable companion class

`from_selection` / `delta_penalty` / `apply_flip` / `penalty_total`. Same shape as `ClumpState`. v1 implementation: per-affected-feature full recompute via vectorised `count_separation`. Affected features for a flip on PU `i` = `cache.pu_to_sep_feats[i]` (round-2 H5: precomputed inverse index, NOT a fresh `np.where` per flip).

**Class docstring** (round-3 F9): include `"NOT thread-safe; private to a single solver call frame. Do NOT expose to progress observers without explicit snapshotting."` Backport the same docstring to `ClumpState`.

**Bedrock test** (round-3 H13 parametrization): delta-matches-full, parametrized over `(n_pu, sep_density, seed)` with at least one configuration where `sep_density=1.0` (every PU contributes to a sep-active feature) and one where `sep_density=0.2`. Each config runs 200 random flips. Catches future incremental-KD-tree drift that the v1 full-recompute hides under sparse density.

**Additional invariant test** (round-2 A4): after a random flip sequence + `apply_flip` chain, `SepState.from_selection(cache, self.selected)` produces identical state. Trivially true for the v1 pure-recompute implementation, but pins the contract so the v0.3 incremental KD-tree variant cannot silently drift.

### Task 11b: `bench_sep.py` performance benchmark (round-3 M12)

Create `tests/benchmarks/bench_sep.py` mirroring `bench_sa.py`. One benchmark function: `bench_sepstate_delta_5000pu_10features` — assert per-flip cost ≤ 200 µs on `n_pu=5000, n_sep_features=10, avg_candidates=300`. Pins the round-2 CR1 memory-shape claim against actual measurement.

Run via `make bench` (on-demand), NOT `make check`. Add to Makefile if not already there. ~30 LOC.

### Task 11c: Byte-identical legacy regression test (round-3 H12)

The v2 design's test-strategy item 12 said "feat_sepnum == 0 is byte-identical to pre-Phase-20 — golden-fixture regression guard". v3 dropped it. Reinstate.

`tests/integration/test_phase20_smoke.py::test_sa_objective_unchanged_when_sepnum_disabled`:

```python
def test_sa_objective_unchanged_when_sepnum_disabled():
    """Anti-test: a problem where every feature has sepnum=1 (disabled)
    must produce the same SA objective as the same problem with no sepnum
    column. Pins the round-2 H1 compound-mask correctness — separation
    code paths must be no-ops when no feature is sep-active."""
    p_legacy = make_test_problem()  # no sepnum column
    p_disabled = p_legacy.clone()
    p_disabled.features["sepnum"] = 1
    p_disabled.features["sepdistance"] = 0.0
    sol_legacy = SASolver().solve(p_legacy, SolverConfig(seed=42))[0]
    sol_disabled = SASolver().solve(p_disabled, SolverConfig(seed=42))[0]
    assert sol_legacy.objective == pytest.approx(sol_disabled.objective)
```

~15 LOC.

### Task 12: SA + IterativeImprovement wire `SepState`

Identical pattern to Phase 19's `ClumpState` wiring. Build `SepState` alongside `ClumpState` (both conditional on `cache.separation_active` / `cache.clumping_active`). In the inner loop add `sep_state.delta_penalty(...)` to the total delta; on accept, call `sep_state.apply_flip(...)`. Both removal and addition passes in `iterative_improvement.py`.

### Task 13: `MIPSolver` gains `mip_sep_strategy` (+ shared validation helper)

**Round-3 M7 correction**: round-2 M5 claimed `mip_clump_strategy` "currently accepts any string" — wrong. `mip_solver.py:57-61` already validates at `__init__`. The actual change is wording-tightening for consistency, plus extracting a shared helper.

Add `mip_sep_strategy: str = "drop"` to `MIPSolver.__init__` and `ZoneMIPSolver.__init__`. Extract a shared helper:

```python
def _validate_mip_strategy(
    name: str,
    value: str,
    allowed: tuple[str, ...],
    rejected_with_reason: dict[str, str] | None = None,
) -> None:
    """Validate a MIP strategy kwarg at __init__ time. `rejected_with_reason`
    is `{strategy_name: explanation}` for values that ARE recognised but
    explicitly rejected with a different reason than 'not in allowed set'
    (e.g. 'socp' for separation)."""
    if rejected_with_reason and value in rejected_with_reason:
        raise ValueError(f"{name}={value!r} is not valid — {rejected_with_reason[value]}")
    if value not in allowed:
        raise ValueError(f"{name} must be one of {allowed}, got {value!r}")
```

Then:

```python
_validate_mip_strategy(
    "mip_sep_strategy", mip_sep_strategy, ("drop", "big_m"),
    rejected_with_reason={
        "socp": "separation is a combinatorial constraint (greedy maximum "
                "independent set), not a conic/probabilistic one. Use 'drop' "
                "(default; gap reported on Solution.sep_shortfalls) or "
                "'big_m' (deferred to a future phase).",
    },
)
```

Apply the same helper to `mip_clump_strategy` and `mip_chance_strategy` validation — replaces three ad-hoc validation blocks with one consistent helper.

At solve time, if `mip_sep_strategy == "big_m"` and any feature has `sepnum > 1`, raise `NotImplementedError("big_m separation strategy deferred to v0.3")`. The default `"drop"` strategy is a no-op at solve time — the deterministic MIP runs unchanged and `build_solution` populates `Solution.sep_shortfalls` / `sep_penalty` post-hoc.

Add `Solver.supports_separation()` method on the base ABC in `base.py`, alongside `supports_probmode3()` and `supports_clumping()`. Default `return True`. All four non-zone solver subclasses inherit. Zone solvers override (see Task 13b).

### Task 13b: Zone solvers raise on separation-active problems

**Round-2 H1 fix.** Zone solvers (`ZoneSASolver`, `ZoneIterativeImprovementSolver`, `ZoneHeuristicSolver`, `ZoneMIPSolver`) construct `Solution(...)` directly — they don't call `build_solution`. On a sep-active problem they would silently produce wrong-because-incomplete results (no penalty, no reporting). Add:

```python
def supports_separation(self) -> bool:
    return False
```

…to each zone solver class. In each `.solve()`, after problem unpacking, add:

```python
if "sepnum" in problem.features.columns and "sepdistance" in problem.features.columns:
    sep_active = (
        (problem.features["sepnum"] > 1)
        & (problem.features["sepdistance"] > 0)
    ).any()
    if sep_active:
        raise NotImplementedError(
            "Zone solvers don't honour SEPDISTANCE/SEPNUM; deferred to v0.3. "
            "Use the non-zone solvers (SASolver, IterativeImprovementSolver, "
            "MIPSolver, HeuristicSolver) or set sepnum<=1 on all features."
        )
```

### Task 14: Combined-constraints regression + scenario plumbing

(a) Combined-constraints test — verify that on a problem with `target2 > 0 AND sepnum > 1 AND ptarget > 0` for the same features, a single SA run produces a `Solution` with all three of `prob_shortfalls`, `clump_shortfalls`, and `sep_shortfalls` populated AND none is `None`. Replaces the misleading "Task 10b lift R6 gate" from v1.

(b) Round-2 M2 fix: extend `_OVERRIDABLE_FIELDS` in `pymarxan/models/problem.py:334` from `{"target", "spf", "prop"}` to `{"target", "spf", "prop", "target2", "clumptype", "ptarget", "sepdistance", "sepnum"}`. Enables `ScenarioSet` to sweep over sepnum values. Backports Phase 18 + 19 coverage in the same commit.

### Task 14b: Scenario clone-with-sepnum test (round-3 H11)

Round-2 M2 extended `_OVERRIDABLE_FIELDS` but the v3 plan never tested it. Without coverage the extension can silently regress.

`tests/pymarxan/models/test_problem.py::test_scenario_set_clones_with_sepnum_override`:

```python
def test_scenario_set_clones_with_sepnum_override():
    """clone_scenario({"sepnum": {1: 5}}) produces a problem whose cache
    rebuilds with the new sepnum. Pins the round-2 M2 _OVERRIDABLE_FIELDS
    extension against silent regression."""
    p = make_sep_active_problem()
    p.features.loc[0, "sepnum"] = 3
    scenarios = ScenarioSet(p, {"baseline": {}, "stricter": {"sepnum": {1: 5}}})
    stricter = scenarios.get("stricter")
    assert int(stricter.features.loc[stricter.features["id"] == 1, "sepnum"].iloc[0]) == 5
```

~15 LOC.

**Commit message:**
```
Phase 20 Batch 3: SepState + ProblemCache + solver integration
```

## Batch 4 — Shiny UI + integration smoke (3 tasks)

### Task 15: `feature_table.py` Shiny editing surface

Four touch points (codebase-grounding agent counted these precisely against the current source):

- **`_COLUMN_ORDER`**: append `"sepdistance"`, `"sepnum"`.
- **`_EDITABLE_FLOAT_COLUMNS`**: append `"sepdistance"`.
- **`_EDITABLE_INT_COLUMNS`**: append `"sepnum"` — BUT split the validator. The existing branch validates int columns against `{0, 1, 2}` (the `clumptype` rule); `sepnum` validates as `>= 0`. Either (a) introduce a per-column rule dict, or (b) special-case `clumptype` and let other int columns validate as non-negative. Approach (b) is simpler.
- **The three `for opt in ("ptarget", "target2", "clumptype")` loops** inside the server function (render, patch fn, `_save` handler) — extend the tuple to `("ptarget", "target2", "clumptype", "sepdistance", "sepnum")` in all three places.

**Round-2 H9 fixes (apply to both Phase 19 + Phase 20 columns):**

- **Invalid-edit toast**: in `_save`, when `validate_feature_edit` returns `None`, call `ui.notification_show(f"Invalid {column} value '{raw}'; expected {rule}", type="warning")`. Currently the edit is silently dropped — confusing.
- **Extend the `ui.p(...)` description block** at feature_table.py:48-61: add one sentence on `sepdistance`/`sepnum`. Suggested text: *"`sepdistance` / `sepnum`: require N geographically separated PUs (≥ sepdistance apart in CRS units) to count toward the feature's target. Leave sepnum=1 to disable."*

Tests parallel `test_clumping_ui.py`:
- `validate_feature_edit("sepnum", "3")` returns `3` (NOT `None`).
- `validate_feature_edit("sepnum", "-1")` returns `None`.
- `validate_feature_edit("sepdistance", "1500.5")` returns `1500.5`.
- `_COLUMN_ORDER` contains both new columns.
- Source-text grep confirms the three optional-column tuples were updated.
- File-presence check for the new help description sentence.

### Task 16: `target_met.py` shows sep_short when active

Add a third `has_separation` branch parallel to `has_clumping`. Columns when active: `sepdistance`, `sepnum`, `sep_short`. Show "—" when the feature has `sepnum <= 1`. File-based string-presence test.

### Task 16b: Extend `write_mvbest` with separation columns (and Phase 19 backport)

**Round-2 H7 fix.** Marxan's reference `mvbest.csv` includes separation-met columns. pymarxan's `write_mvbest` (writers.py:179-226) currently emits only `Feature_ID, Feature_Name, Target, Amount_Held, Target_Met, Shortfall`. Extend:

- When `solution.sep_shortfalls is not None`: add `Separation_Count` (`sepnum - shortfall`) and `Separation_Met` (boolean) columns per feature.
- **Phase 19 backport**: in the same commit, add `Clump_Short` when `solution.clump_shortfalls is not None`. Zero-cost win since the writer surface is right here.
- **Phase 18 backport optional but recommended**: `Prob_Gap` when `solution.prob_shortfalls is not None`.

Test: round-trip a sep-active problem through SA + `write_mvbest`, read the CSV, assert the new columns appear and match `solution.sep_shortfalls`.

### Task 17: `help_content.py` documents SEPDISTANCE/SEPNUM

New section with:
- Marxan source-of-truth pointer (`computation.hpp::computeSepPenalty` + `clumping.cpp::CountSeparation2`).
- The hyperbolic penalty formula in plain English.
- A note on the unit mismatch (round-2 L3): `sep_shortfalls` is integer count, `clump_shortfalls` is float amount, `prob_shortfalls` is float probability — reading the three side-by-side requires unit awareness.
- A note that the heuristic and MIP-drop strategies are separation-blind during scoring; the penalty is post-hoc reported only.
- Citations: Watts et al. (2009, *EMS* 24:1513-1521) + Watts, Stewart, Martin (2017, *Learning landscape ecology* tutorial).

File-based test pins all three citation strings.

### Task 17b: `run_panel.py` — sep banner, warning replay, crash traceback, constraint-gap summary

Four sub-changes:

**(a) Round-2 H8 — sep_mip_notice banner.** Phase 18 added `probmode3_mip_notice` banner in `run_panel.py:217-245` to warn when MIP + PROBMODE 3 are combined. Mirror line-for-line as `sep_mip_notice`:

- Shows when `solver_type == "MIP"` AND any feature has `sepnum > 1 AND sepdistance > 0`.
- Text: *"MIP with separation distance: the deterministic relaxation is solved exactly; separation gap is reported post-hoc on Solution.sep_shortfalls. For exact separation-aware optimisation use SA or IterativeImprovement."*

**(b) Round-2 CR2 — warning replay.** Wrap the solve in `warnings.catch_warnings(record=True)` and replay any captured `UserWarning` via `ui.notification_show(message=str(w.message), type="warning")`.

**(c) Round-3 CR3 — SA crash traceback visibility.** In `run_panel._run`'s `except Exception as e` (run_panel.py:134-136), include the full traceback so users see file + line:

```python
except Exception as e:
    import traceback
    progress.status = "error"
    progress.error = f"{e!r}\n{traceback.format_exc()}"
    progress.message = "Solver failed (see Error below)"
```

**(d) Round-3 M9 — constraint-gap summary line.** Extend the run_panel summary line so users can tell when constraint shortfalls are non-zero:

```python
parts = [f"Cost: {best.cost:.2f}", f"Targets met: {met}/{total}"]
if best.sep_shortfalls is not None:
    n_active = sum(1 for v in best.sep_shortfalls.values() if v > 0)
    parts.append(f"Sep gap: {n_active}/{len(best.sep_shortfalls)}")
if best.clump_shortfalls is not None:
    n_active = sum(1 for v in best.clump_shortfalls.values() if v > 0)
    parts.append(f"Clump gap: {n_active}/{len(best.clump_shortfalls)}")
if best.prob_shortfalls is not None:
    n_active = sum(1 for v in best.prob_shortfalls.values() if v > 0)
    parts.append(f"Prob gap: {n_active}/{len(best.prob_shortfalls)}")
summary = ", ".join(parts)
```

Document explicitly in the design Acceptance Criteria that `Solution.all_targets_met` remains amount-only (consistent with Phases 18 + 19); the per-constraint shortfalls expose the rest.

Total Task 17b cost: ~70 LOC.

### Task 18: Integration smoke test

`tests/integration/test_phase20_smoke.py`:
- All four solvers (SA, II, heuristic, MIP-drop) run end-to-end on a separation-active problem and populate `Solution.sep_shortfalls` + `sep_penalty`.
- Combined PROBMODE 3 + TARGET2 + SEPNUM problem produces a `Solution` with all six analytics attrs populated.
- `save_project` + `load_project` round-trip preserves the new columns (including the round-trip case where `sepdistance=0 AND sepnum>1`).
- `MIPSolver(mip_sep_strategy="socp")` raises `ValueError` at construction.
- `MIPSolver(mip_sep_strategy="big_m")` raises `NotImplementedError` at solve time.
- **Round-2 H1 cases**: `ZoneSASolver`, `ZoneIISolver`, `ZoneHeuristicSolver`, `ZoneMIPSolver` each raise `NotImplementedError` on a sep-active problem.
- **Round-2 H2 case**: heuristic on a no-geometry problem with `sepnum > 1` produces a solution with `sep_shortfalls=None` and emits a `UserWarning` — does NOT crash.
- **Round-2 H3 case**: GeoDataFrame with an empty geometry row + sep-active raises `ValueError` from `get_pu_coordinates` (clear message).

### Task 18b: Marxan-binary cross-validation (skipif-unavailable)

**Round-2 M3 fix + round-3 test-design tightening.** Phase 20's purpose is Marxan parity; the smoke test should include a cross-check against the reference C++ binary when available.

```python
@pytest.mark.skipif(
    shutil.which("marxan") is None,
    reason="Marxan binary not on PATH; skipping cross-validation",
)
def test_phase20_marxan_binary_agreement(tmp_path):
    # Construct sep-active problem with INTEGER xloc/yloc (no FP centroid
    # noise) and deterministic seed. Use a fixture where no two pairwise
    # distances are equal modulo sepdistance — eliminates greedy-ordering
    # tie ambiguity. Run pymarxan SA + MarxanBinarySolver on the same
    # problem; assert sep_count matches EXACTLY per feature.
```

**Round-3 test-design fix**: the original "±1 tolerance" wording could hide a real off-by-one (e.g. `>` vs `>=` in `CheckDistance`). Require exact equality by constructing the fixture without distance ties; only allow ±1 if you must use geometry.centroid (which introduces FP noise — avoid for this test).

Optional but high-value test — pins the Marxan-source claims with a numerical check.

### Task 18c: Solver-capability-matrix programmatic test (round-3 H16)

`tests/integration/test_solver_capability_matrix.py`:

```python
@pytest.mark.parametrize("solver_cls,problem_kind,should_support", [
    # Phase 18, 19, 20 capability cross-product
    (SASolver, "probmode3", True),
    (SASolver, "clumping", True),
    (SASolver, "separation", True),
    (ZoneSASolver, "probmode3", True),     # Zone supports probmode3 (per-zone)
    (ZoneSASolver, "clumping", False),     # Phase 19 doesn't ship zone clumping
    (ZoneSASolver, "separation", False),   # Phase 20 doesn't ship zone separation
    # ... full cross-product for all 8 solvers × 3 constraint types ...
])
def test_solver_capability_matrix(solver_cls, problem_kind, should_support):
    """Programmatic check: every (solver, constraint) pair either reports
    supports_X() == True AND .solve() succeeds, OR supports_X() == False
    AND .solve() raises NotImplementedError. Catches Phase 21+ regressions
    where someone forgets the override-to-False on zone solvers."""
    p = make_problem_with_constraint(problem_kind)
    solver = solver_cls()
    cap = getattr(solver, f"supports_{problem_kind}")()
    assert cap == should_support
    if should_support:
        solver.solve(p, SolverConfig(seed=42, num_solutions=1))  # no raise
    else:
        with pytest.raises(NotImplementedError):
            solver.solve(p, SolverConfig(seed=42, num_solutions=1))
```

~30 LOC + matrix data. Brittle bookkeeping wants a regression test.

**Commit message:**
```
Phase 20 Batch 4: Shiny UI + integration smoke (Phase 20 COMPLETE)
```

## Verification

`make check` after each batch. Targets:
- Tests: 1212 → ~1252 (v4 adds round-3 tests: typo-column warning, parametrized bedrock density, sepnum-disabled byte-identical, scenario clone with sepnum, capability matrix, parametrized `compute_sep_penalty` table, parametrized `count_separation` table)
- Coverage: ≥91 %
- Per-flip SepState benchmark (`tests/benchmarks/bench_sep.py`, run via `make bench`): under 200 µs on 5000-PU × 10-sep-features × 300-candidate problem (sanity check on round-2 CR1 memory-shape fix).

## CHANGELOG entry (adopt verbatim into CHANGELOG.md `[Unreleased]` section at v0.2.0a3 cut)

```markdown
## [Unreleased]

### Added
- **Phase 20 — SEPDISTANCE / SEPNUM (separation distance).** Per-feature
  geographic-spread constraints, validated line-by-line against Marxan v4
  `computation.hpp::computeSepPenalty` and `clumping.cpp::CountSeparation2`.
  - Optional `sepdistance` (float ≥ 0, default 0) and `sepnum` (int ≥ 1,
    default 1) columns on `spec.dat`. A feature is separation-active iff
    `sepdistance > 0 AND sepnum > 1`. Writers omit both when all-default,
    so legacy projects round-trip byte-identical.
  - PU coordinates resolve via three-tier fallback: GeoDataFrame
    `.geometry.centroid` → `pu.dat` `xloc`/`yloc` columns →
    `PUCoordinatesUnavailableError` at `ProblemCache.from_problem` (only
    when separation-active).
  - New `pymarxan.solvers.separation` module exposes `compute_sep_penalty`
    (Marxan hyperbolic curve), `count_separation` (greedy admission in
    ascending PU-id order, capped at `sepnum`),
    `compute_sep_penalty_from_scratch`, `evaluate_solution_separation`,
    `get_pu_coordinates`, the mutable `SepState` companion, and the
    `PUCoordinatesUnavailableError` exception class.
  - `MIPSolver` and `ZoneMIPSolver` gain `mip_sep_strategy` kwarg
    (default `"drop"`). `"socp"` rejected at `__init__` (separation is
    combinatorial, not conic); `"big_m"` raises `NotImplementedError`
    at solve time.
  - New `Solver.supports_separation()` capability method (default `True`).
  - `Solution` gains `sep_shortfalls: dict[int, int] | None` and
    `sep_penalty: float | None` attributes (purely additive — all existing
    keyword-only construction patterns unchanged).
  - `write_mvbest` now emits `Separation_Count` / `Separation_Met` columns
    when separation-active, plus `Clump_Short` (Phase 19 backport) and
    `Prob_Gap` (Phase 18 backport) when those constraint paths are active.
  - Shiny UI: `sepdistance` / `sepnum` editable in `feature_table` (split
    int-validator — `sepnum ≥ 0`, distinct from `clumptype ∈ {0,1,2}`);
    `target_met` shows `sep_short` column when active; `run_panel`
    shows a MIP-with-separation banner; help content documents the
    hyperbolic penalty curve with citations.
  - `ScenarioSet._OVERRIDABLE_FIELDS` extended to include `sepdistance`,
    `sepnum` (plus `target2`, `clumptype`, `ptarget` backports — Phase 18
    + 19 coverage gap).
  - References: Watts et al. (2009). *Environmental Modelling & Software*
    24(12): 1513–1521. https://doi.org/10.1016/j.envsoft.2009.06.005.
    Watts, Stewart & Martin (2017). *Learning landscape ecology*, 211–227.
    https://doi.org/10.1007/978-1-4939-6374-4_13

### Changed
- Zone solvers (`ZoneSASolver`, `ZoneIterativeImprovementSolver`,
  `ZoneHeuristicSolver`, `ZoneMIPSolver`) raise `NotImplementedError` on
  separation-active problems (previously would silently no-op). Per-zone
  SEPDISTANCE deferred to v0.3.
- `ProblemCache.from_problem` emits `UserWarning` for two no-op
  separation configurations (sepdistance>0 on geographic CRS; sepnum>1
  with sepdistance=0). The Shiny `run_panel` captures and replays these
  via `ui.notification_show`.
- `read_spec` emits `UserWarning` for unrecognised columns in `spec.dat`
  — catches typos like `sepnnum`, `targt2`, `ptraget`, `clumptpe`
  (Phases 18 + 19 + 20 column whitelist).
- `run_panel` SA crash error display now includes full traceback (file +
  line); the solver-failed state visually leaves the running card.
- `MIPSolver` strategy-kwarg validation moved to a shared
  `_validate_mip_strategy` helper for consistency across
  `mip_chance_strategy`, `mip_clump_strategy`, `mip_sep_strategy`.
- When any feature has a non-default `sepdistance` or `sepnum`, the
  column is written for *all* features in `spec.dat` (same behavior as
  Phase 18 `ptarget` and Phase 19 `target2` / `clumptype`).

### Notes
- `Solution` and `ProblemCache` serialized state produced under v0.2.0a3+
  cannot be deserialized by v0.2.0a2 or earlier (one-way forward
  compatibility). Same as the Phase 18 / 19 alpha bumps.
- Users running with `python -W error` or pytest `filterwarnings =
  error::UserWarning` should filter pymarxan warnings explicitly:
  `warnings.filterwarnings("always", category=UserWarning, module="pymarxan")`.
```

## Open questions

All v1 OQs resolved by the review pass (see `2026-05-16-phase20-review.md`):

- ~~OQ1 (greedy ordering)~~ → ascending PU-id (insertion order).
- ~~OQ2 (distance metric)~~ → Euclidean squared in native CRS, per `CheckDistance`.
- ~~OQ3 (coordinate source)~~ → three-tier resolution: geometry.centroid → xloc/yloc → ValueError.
- ~~OQ4 (TARGET2 + SEPNUM composition)~~ → parallel additive pipelines; compound det_spf mask prevents double-counting.
- ~~OQ5 (sepdistance validation)~~ → `< 0` rejected at read time; `> 0` + geographic CRS warns at validate time; `0` + `sepnum > 1` warns at validate time.

Residual deferrals (tracked in v0.3 backlog, see review doc L1/L2):
- `SolutionMetrics` named-tuple refactor (six nullable Solution attrs accumulating).
- Incremental KD-tree-backed `SepState` (if benchmarks demand).
