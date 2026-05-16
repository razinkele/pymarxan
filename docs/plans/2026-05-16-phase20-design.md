# Phase 20 — Separation distance (SEPDISTANCE + SEPNUM)

**Date:** 2026-05-16 (v4 — patched after three rounds of multi-agent review; see `2026-05-16-phase20-review.md`, `2026-05-16-phase20-review-round2.md`, `2026-05-16-phase20-review-round3.md`)
**Target version:** v0.2.0 final (full Marxan-classic parity — last piece)
**Parent plan:** [`docs/plans/2026-05-16-realignment.md`](2026-05-16-realignment.md)
**Effort:** ~4 days (round-3 added cross-phase typo guard, capability-matrix test, parametrized math tests, `_det_spf` extraction, expanded `run_panel` with traceback + constraint-gap summary)
**Confidence:** 96 % (post-round-3; further review past this point produces diminishing returns — Batch 1 execution will surface any remaining issues against real code)

## Why

Marxan v4's `CountSeparation` lets users require a feature to be represented in **N geographically separated** planning units — useful for risk-bet-hedging where a single catastrophe shouldn't wipe out the feature's protection. Per-feature parameters:

- `SEPDISTANCE_j` — minimum distance between PUs containing the feature
- `SEPNUM_j` — minimum number of pairwise-separated PUs required

Without this, a reserve that meets a feature's amount target via 5 contiguous PUs technically protects the feature but offers no spatial redundancy. SEPDISTANCE + SEPNUM say "at least N of those PUs must be ≥ SEPDISTANCE apart from each other."

pymarxan currently has no separation mechanism. Phase 20 closes the last v0.2.0 gap (per the realignment plan §"Recommended next phases"). MarZone explicitly punted on this (commented out in source — "re-enable when separation is reimplemented"); pymarxan ships it.

### Foundational references

Validated line-by-line against the Marxan v4 master source by the scientific-accuracy review pass:

- **`computeSepPenalty`** — `computation.hpp:15-27`: hyperbolic penalty curve `1/(7·fval + 0.2) − 1/7.2` with `fval = max(count, 1) / sepnum`. NOT linear.
- **`CountSeparation2`** — `clumping.cpp:1075-1168`: greedy admission in `spec[isp].head` (PU-insertion / PU-id) order, capped at `sepnum`.
- **`CheckDistance`** — `clumping.cpp:1006-1012`: Euclidean-squared on raw `pu.xloc` / `pu.yloc`.
- **Call sites** — `score_change.cpp:445-446, 574`: `reserve.penalty += computeSepPenalty(...) · spec[i].spf · spec[i].penalty`.
- Watts et al. (2009) *Marxan with Zones*. *Environmental Modelling & Software* 24(12): 1513-1521. https://doi.org/10.1016/j.envsoft.2009.06.005 — establishes the SA objective pymarxan inherits.
- Watts, Stewart & Martin (2017) "Systematic conservation planning with Marxan", in *Learning landscape ecology*: 211-227. https://doi.org/10.1007/978-1-4939-6374-4_13 — tutorial introduction to SEPDISTANCE/SEPNUM.

## What's in scope

- Per-feature `sepdistance` (float ≥ 0; default `0`) and `sepnum` (int ≥ 1; default `1`) columns on `spec.dat`. Following Marxan, a feature is "separation-active" iff `sepdistance > 0 AND sepnum > 1`. The `sepnum=1` default matches Marxan's "trivially satisfied" disabled-state.
- **PU coordinates** — three-tier resolution: (1) `planning_units.geometry.centroid` when a GeoDataFrame; (2) `xloc` / `yloc` columns on `pu.dat` (Marxan convention); (3) raise `ValueError` at `ProblemCache.from_problem` if separation-active and neither is available.
- New `pymarxan.solvers.separation` module with the Marxan-faithful `compute_sep_penalty` (hyperbolic curve), `count_separation` (greedy PU-id-ordered admission), `compute_sep_penalty_from_scratch`, `evaluate_solution_separation`, and `SepState` companion class.
- **SA / iterative-improvement integration** via a `SepState` companion to `ProblemCache` paralleling Phase 19's `ClumpState`.
- **MIP "drop" strategy** (default): MIP solves the deterministic relaxation; separation gap reported post-hoc on `Solution.sep_shortfalls` / `Solution.sep_penalty`. `"socp"` rejected in `__init__` (separation is combinatorial, not conic); `"big_m"` raises `NotImplementedError` at solve time (future phase).
- **Heuristic** stays separation-blind during scoring; reports the gap post-hoc through `build_solution`.
- **Combined PROBMODE 3 / TARGET2 / SEPNUM**: all three penalty paths run additively. Per the scientific review, Marxan's `NewPenalty4` folds clump + sep into a single per-feature term; pymarxan splits them into two parallel pipelines that sum to the same numeric total when the deterministic mask in `ProblemCache` is extended to compound-exclude both type-4 and sep-active features.
- `validate()` warnings for two no-op configurations: (a) `sepdistance > 0` on a geographic-CRS GeoDataFrame (distance in degrees is nearly meaningless); (b) `sepnum > 1 AND sepdistance == 0` (constraint trivially satisfied).
- Shiny UI: `sepdistance` / `sepnum` columns in `feature_table` editor (with split validation — `sepnum` validates `>= 0`, distinct from `clumptype`'s `{0,1,2}` rule); `sep_short` column in `target_met` when active; help content with citations.

## What's NOT in scope

- **Per-zone SEPDISTANCE** in `ZonalProblem`. Marxan with Zones doesn't ship separation; deferred.
- **MIP big-M / pairwise-distance formulation** of separation. Deferred — same shape as Phase 19's `mip_clump_strategy="big_m"` deferral.
- **Distance metrics other than Euclidean centroid-to-centroid.** Marxan uses Euclidean; pymarxan matches. If users have great-circle / geodesic requirements, they can transform their geometry to a projected CRS upstream.

## Formulation (validated against Marxan v4 master `computation.hpp` + `clumping.cpp`)

Given:
- `amount_ij` per (PU, feature) as today.
- `sepdistance_j` ≥ 0 — minimum pairwise distance for feature *j*. `0` disables.
- `sepnum_j` ≥ 1 — minimum number of separated PUs required. `≤ 1` disables (Marxan convention; `sepnum=1` is trivially satisfied by any single representing PU).
- `(x_i, y_i)` ∈ ℝ² — PU coordinates (from `planning_units.geometry.centroid` or `pu.dat` `xloc`/`yloc` columns).

**Separation count** `count_j(x)` for feature *j* under selection vector `x`:

1. Build the candidate set `C_j(x) = {i : x_i = 1 AND amount_ij > 0}`.
2. Iterate `C_j(x)` in **ascending PU-id order** (Marxan's `spec[isp].head` insertion order; `clumping.cpp::makelist` / `SepDealList`).
3. Greedily admit each candidate into `kept` if its Euclidean distance to every already-admitted PU is ≥ `sepdistance_j`. Use the squared-distance comparison from `clumping.cpp::CheckDistance:1006-1012`.
4. Stop as soon as `|kept| == sepnum_j` and return `count_j = |kept|`. Per Marxan, the result is clamped at `sepnum_j` — counting higher is meaningless because the penalty plateaus at 0.

**Hyperbolic separation penalty** — verbatim from `computation.hpp::computeSepPenalty:15-27`:

```python
def compute_sep_penalty(count: int, sepnum: int) -> float:
    if sepnum <= 0:
        return 0.0
    # When count == 0, bump fval to 1/sepnum so the hyperbola does not blow up.
    fval = count / sepnum if count > 0 else 1.0 / sepnum
    return 1.0 / (7.0 * fval + 0.2) - 1.0 / 7.2
```

Properties:
- `fval = 1` (target met): returns `0` exactly.
- `count = 0, sepnum = 3`: `fval = 1/3`, penalty ≈ `0.255`.
- `count = 2, sepnum = 3`: `fval = 2/3`, penalty ≈ `0.067`.
- `count = sepnum`: `0`.
- Bounded by ≈ `0.95` from above (curve plateau when `fval ≈ 1/sepnum` is small).

The shape is **not** `(sepnum − count)/sepnum`; the curve is steeper for nearly-met targets and flatter for badly-missed ones, which is the original Marxan calibration choice.

**Per-feature penalty contribution**:

```
sep_pen_j(x) = baseline_penalty_j · SPF_j · compute_sep_penalty(count_j(x), sepnum_j)
```

`baseline_penalty_j` reuses Phase 19's `compute_baseline_penalty` (per-feature cost-to-meet-target greedy from cheapest PUs). The same scale applies to clumping and separation alike.

**Total separation penalty**:

```
sep_penalty(x) = Σ_j sep_pen_j(x)   over features with sepnum_j > 1 AND sepdistance_j > 0
```

Added to the objective alongside the deterministic + clumping + PROBMODE 3 penalties — all four are additive. To prevent double-counting, `ProblemCache.compute_full_objective` and `compute_delta_objective` mask separation-active features out of the deterministic `det_spf` path:

```python
det_spf = self.feat_spf * (self.feat_target2 <= 0) * (self.feat_sepnum <= 1)
```

For features with both `target2 > 0` AND `sepnum > 1`, Marxan computes a single per-feature penalty term inside `NewPenalty4` that absorbs both contributions. pymarxan computes them as two parallel terms (ClumpState + SepState); the numeric sum is identical because both use the same `baseline · SPF` scale and the deterministic path fires for neither.

## API surface changes

```python
# New optional columns on features
problem.features["sepdistance"]      # float ≥ 0; default 0 disables
problem.features["sepnum"]           # int ≥ 1;   default 1 disables (Marxan convention)

# New module pymarxan.solvers.separation
def compute_sep_penalty(count: int, sepnum: int) -> float:
    """Hyperbolic Marxan penalty curve. Mirrors computation.hpp::computeSepPenalty."""

def get_pu_coordinates(problem) -> np.ndarray:
    """Three-tier resolution: geometry.centroid → xloc/yloc → ValueError."""

def count_separation(
    selected: np.ndarray,
    feat_amounts: np.ndarray,        # (n_pu,) for one feature
    pu_coords: np.ndarray,           # (n_pu, 2) PU coordinates
    sepdistance: float,
    sepnum: int,                     # used to early-exit when |kept| == sepnum
) -> int:
    """Greedy admission in ascending PU-id order, capped at sepnum.
    Mirrors clumping.cpp::CountSeparation2 + makelist + SepDealList."""

def compute_sep_penalty_from_scratch(
    cache: ProblemCache, selected: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Reference impl: per-feature (sep_count, total_penalty)."""

def evaluate_solution_separation(
    problem, selected,
) -> tuple[dict[int, int], float]:
    """Post-hoc evaluator for MIP drop + heuristic + build_solution.
    Returns (sep_shortfalls: feature_id → shortfall, total_penalty)."""

class SepState:
    """Mutable companion to ProblemCache for the SA inner loop.
    Maintains per-feature sep_count and held penalty for incremental
    delta. v1 implementation: per-affected-feature full recompute
    using a precomputed pairwise distance matrix (vectorised NumPy)
    rather than a Python pair loop."""
    @classmethod
    def from_selection(cls, cache, selected): ...
    def delta_penalty(self, cache, idx, adding): ...
    def apply_flip(self, cache, idx, adding): ...
    def penalty_total(self) -> float: ...

# New Solution attrs (mirrors prob_*/clump_* from Phases 18/19)
Solution.sep_shortfalls: dict[int, int] | None  # feature_id → max(0, sepnum - count)
Solution.sep_penalty: float | None

# New ProblemCache fields (declared in the @dataclass(frozen=True) body)
ProblemCache.feat_sepdistance: np.ndarray     # (n_feat,) float64
ProblemCache.feat_sepnum: np.ndarray          # (n_feat,) int32
ProblemCache.pu_coords: np.ndarray            # (n_pu, 2) float64
ProblemCache.separation_active: bool          # any((feat_sepdistance > 0) & (feat_sepnum > 1))

# New Solver capability
Solver.supports_separation() -> bool   # default True; MIP via drop fallback

# New MIPSolver kwarg
MIPSolver(*, mip_sep_strategy: str = "drop")
# "drop" (default) | "big_m" (raises NotImplementedError at solve time)
# "socp" rejected in __init__ — separation is combinatorial, not conic.
```

## Data flow

```
read_spec()       → features (with optional sepdistance, sepnum columns)
                       │
                       ▼
ConservationProblem ───┴──▶ Solver
                                │
                                ▼
              ProblemCache.from_problem()  [build-once]
                  - existing fields unchanged
                  - new: feat_sepdistance, feat_sepnum, pu_centroids,
                         separation_active
                  - cache.compute_full_objective and compute_delta_objective
                    are NOT changed for separation — that path lives entirely
                    in SepState (same pattern as ClumpState).

              SepState.from_selection(cache, selected)  [SA setup]
                  - full Marxan-greedy MIS per type-sep feature
                  - returns mutable companion

              SA / IterativeImprovement inner loop  [per flip]
                  - deterministic delta from cache (excludes type-4 too;
                    already handled by Phase 19's mask)
                  - + clump_state.delta_penalty(cache, idx, adding)
                  - + sep_state.delta_penalty(cache, idx, adding)
                  - on accept: apply both *_state.apply_flip

              build_solution
                  - existing PROBMODE 3 + clumping post-hoc populates
                  - + Phase 20 sep_shortfalls / sep_penalty populator
                  - single source of truth across all solvers
```

## File format

`spec.dat` gains two optional columns:

```csv
id,name,target,spf,sepdistance,sepnum
1,sea_turtle,100,1.0,5000,3
2,kelp_forest,50,2.0,0,1
```

`sepdistance` is a float (distance in the planning units' CRS units — metres for projected CRS, degrees for EPSG:4326). `sepnum` is an integer ≥ 1; default `1` is Marxan's "disabled" sentinel. Writers omit `sepdistance` when all-zero AND omit `sepnum` when all-`≤1` — keeps legacy projects byte-identical.

## Test strategy

Same TDD-per-task pattern. 12–15 new tests covering:

1. **`count_separation` math** — three known scenarios with hand-computed MIS sizes.
2. **`count_separation` with no candidates** — 0.
3. **`count_separation` with sepdistance=0** — all candidates count (degenerate, fall back to deterministic count).
4. **`count_separation` greedy order regression** — pinned against the Marxan-source ordering after the review pass.
5. **Schema defaults + I/O round-trip** — `sepdistance`/`sepnum` missing → 0 fill; writers omit when all-default.
6. **`SepState.delta_penalty` bedrock** — delta-matches-full over 200 random flips on a problem with separation active. (The Phase 19 ClumpState bedrock test analog.)
7. **MIP "drop" strategy** — populates `Solution.sep_shortfalls` post-hoc; `"big_m"` raises NotImplementedError.
8. **`Solver.supports_separation()` capability** — all four solvers return True.
9. **Combined PROBMODE 3 + clumping + separation** — all three penalties present, all correctly accumulate.
10. **Lift Phase 19 R6 reject** — `target2 > 0 AND sepnum > 0` no longer raises; both penalties active.
11. **PUs without geometry + sepnum > 0** — clear ValueError at problem construction.
12. **`feat_sepnum == 0` is byte-identical to pre-Phase-20** — golden-fixture regression guard.
13. **Shiny UI surface tests** — file-based string presence for `sepdistance`/`sepnum`/`sep_short`.
14. **End-to-end smoke test** — all four solvers run on a problem with separation active and populate `Solution.sep_*`.
15. **Marxan-source agreement** — once the greedy ordering is pinned, a numeric test reproduces an exact Marxan-classic separation-count for a constructed scenario.

Target: +12–15 new tests, total ≥1227, coverage stays ≥91 %.

## Risks (post-review residuals)

- **R1 (RESOLVED) — Greedy ordering.** Pinned to ascending PU-id (`spec[isp].head` insertion order) per `clumping.cpp::makelist` + `SepDealList`. Bedrock test 4 is now deterministic.
- **R2 (RESOLVED) — Penalty formula.** Pinned to hyperbolic `1/(7·fval + 0.2) − 1/7.2` per `computation.hpp::computeSepPenalty`. The draft's linear form was wrong.
- **R3 (RESOLVED) — Distance metric and PU coordinate source.** Three-tier resolution (geometry.centroid → xloc/yloc → ValueError); Euclidean squared distance per `CheckDistance`.
- **R4 (RESOLVED) — Combined penalty double-counting.** Deterministic mask in `ProblemCache` extended to compound-exclude both type-4 AND sep-active features: `det_spf = self.feat_spf * (self.feat_target2 <= 0) * (self.feat_sepnum <= 1)`.
- **R5 (RESOLVED in v3) — Delta-perf memory shape.** Round-1 left the pairwise-distance matrix shape ambiguous; round-2 (adversarial + performance agents independently) flagged that the obvious reading allocates O(n_pu²) memory per flip — 20 GB at n_pu=50000. **v3 pins:** `count_separation` allocates `pdist(coords[candidates], "sqeuclidean")` on the **candidate sub-array only**, k×k where `k = |selected ∩ has-feature|`, never n_pu×n_pu. The matrix lives in the function frame and is freed when the call returns. Per-flip cost: O(k²) build + early-exit greedy capped at `sepnum`. Acceptable for n_pu ≤ ~5000; the L2 KD-tree `query_pairs` precompute is the v0.3 optimisation path for larger problems.
- **R6 (RESOLVED in v3) — `validate()` warnings were dead code.** Round-2 UX agent showed `validate()` only fires on Shiny upload, not at solve time — a user editing in the feature_table grid never triggers it. **v3 moves the warnings to `ProblemCache.from_problem`**, which runs on every solve; `run_panel` wraps the solve in `warnings.catch_warnings(record=True)` and replays captured warnings via `ui.notification_show(type="warning")` so they actually surface.
- **R7 (LOW) — Solution attr accumulation.** Six nullable analytics attrs on `Solution` after Phase 20. A `SolutionMetrics` named-tuple refactor is on the v0.3 backlog (L1 in the round-1 review doc).
- **R8 (MEDIUM, documented) — Combined SA cost growth.** With all three of PROBMODE 3 + TARGET2 + SEPNUM active on the same problem, the SA per-flip cost grows ~3-4× vs vanilla. The three constraint paths run serially with no fusion. Acceptable for v0.2.0; loop fusion is a v0.3 candidate.
- **R9 (LOW) — Zone solvers silently no-op separation.** Per-zone SEPDISTANCE is out of scope, but the v2 plan didn't add a guard — zone solvers would silently produce wrong-because-incomplete results. **v3 adds**: `Zone*Solver.supports_separation()` returns `False` and `.solve()` raises `NotImplementedError` when any feature has `sepnum > 1 AND sepdistance > 0`. **v4 adds**: programmatic capability-matrix test (Task 18c) catches Phase 21+ regressions where someone forgets the override.
- **R10 (MEDIUM, documented) — Warning visibility in non-Shiny contexts.** `warnings.warn` in `ProblemCache.from_problem` uses `stacklevel=2` so source location points to user code. Default-mode dedup means warnings fire only on the first solve per problem. Strict-mode users (`python -W error`) need to filter explicitly — documented in `help_content.py`.
- **R11 (LOW) — Capability-matrix bookkeeping.** Phase 20 adds 12 touch-points for `supports_separation()` (4 base supports + 4 zone overrides + 4 zone raises). Phase 21+ will likely forget at least one. Programmatic cross-product test (Task 18c) is the regression net.
- **R12 (LOW) — Test density vs Phase 19 precedent.** Phase 19 added ~50 tests; Phase 20 targets ~40 (post-v4 parametrization). Lower than precedent but still appropriate — the math layer is smaller (no `connected_components` graph machinery) and the parametrized tables in Tasks 6 + 7 cover boundary behaviour more densely than Phase 19's per-test approach.

## Assumptions

1. PU coordinates resolve in order: (1) `planning_units.geometry.centroid` if GeoDataFrame AND no row has NaN/empty centroid; (2) `pu.dat` `xloc`/`yloc` columns if present AND no row is NaN; (3) raise `ValueError` at `ProblemCache.from_problem` when separation-active. NaN guard prevents silent under-counting from empty geometries.
2. Distance is Euclidean-squared in the planning_units' native CRS, comparison via `>=`. Parity with Marxan's `CheckDistance`. Users SHOULD set `sepdistance` strictly less-than or greater-than any nominal grid spacing; matching values produce floating-point boundary non-determinism (round-2 H4).
3. Greedy candidate admission is in ascending PU-id order, capped at `sepnum_j` (matches `CountSeparation2` short-circuit).
4. Separation penalty is additive to deterministic + PROBMODE 3 + clumping. For type-4 + sep-active features, Marxan folds both into `NewPenalty4`'s single per-feature term; pymarxan splits them into parallel ClumpState + SepState pipelines that sum to the same numeric total because the deterministic mask fires for neither.
5. `baseline_penalty_j` from Phase 19 applies unchanged. No new baseline computation.
6. `sepnum ≤ 1` disables separation for the feature. `sepdistance == 0` AND `sepnum > 1` emits a `UserWarning` from `ProblemCache.from_problem` (constraint trivially satisfied; replayed in the Shiny run_panel via `warnings.catch_warnings(record=True)`).
7. `MIPSolver(mip_sep_strategy="socp")` raises `ValueError` in `__init__` — separation is combinatorial, not conic.
8. `count_separation` allocates `pdist(coords[candidates], "sqeuclidean")` on the candidate sub-array only — k×k, never n_pu×n_pu. Per-flip memory is bounded by selection footprint, not problem size.
9. `ProblemCache` precomputes `pu_to_sep_feats: list[np.ndarray]` (PU idx → array of sep-active feature column indices). `SepState.delta_penalty` iterates this inverse index — O(features-at-PU), not O(n_feat). The "inverse-index discipline" pattern is documented on `ProblemCache`'s class docstring for future maintainers (Phase 21+ should follow).
10. Zone solvers explicitly raise `NotImplementedError` on sep-active problems. Per-zone separation deferred to v0.3.
11. The `pymarxan.solvers.separation` module is NOT re-exported at the `pymarxan.solvers` package level (matches Phase 19 `clumping` precedent). Public names exported via the module's `__all__` only.
12. The deterministic-penalty mask is centralised as `ProblemCache._det_spf` (`functools.cached_property`). When adding a new constraint type, edit the single mask definition — not the call sites in `compute_full_objective` and `compute_delta_objective`.
13. `Solution.all_targets_met` remains amount-only (consistent with Phases 18 + 19). Per-constraint shortfalls are exposed on `Solution.X_shortfalls` attrs and surfaced in `run_panel`'s summary line via per-constraint gap counts. Users wanting a single "all constraints met" boolean should check the relevant `*_shortfalls` dicts explicitly.

## Acceptance criteria

1. ✅ All new code has TDD-per-task unit tests.
2. ✅ `make check` stays green.
3. ✅ Shiny exposes sepdistance/sepnum editing with split int-validation (`sepnum` accepts any `≥ 0`, distinct from `clumptype`'s `{0,1,2}` rule).
4. ✅ CHANGELOG.md `[Unreleased]` gains Phase 20 entry.
5. ✅ Test count grows; coverage ≥91 %.
6. ✅ Scientific-accuracy review confirmed: hyperbolic penalty formula, ascending PU-id ordering, Euclidean-squared distance, count clamped at `sepnum`.
7. ✅ Combined PROBMODE 3 + TARGET2 + SEPNUM regression test verifies `prob_shortfalls`, `clump_shortfalls`, AND `sep_shortfalls` all populate on the same `Solution`.
