# Phase 19 — Implementation Plan (revised post-review)

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Each task follows TDD: failing test → minimal impl → passing test. Commit after each batch.

**Goal:** Ship per-feature TARGET2 / CLUMPTYPE (Marxan "type-4 species" / clumping) across SA / iterative-improvement natively via an incremental `ClumpState`; heuristic + MIP use post-hoc clump reporting (same shape as Phase 18 MIP drop).

**Design doc:** `2026-05-16-phase19-design.md` (also revised post-review)
**Review doc:** `2026-05-16-phase19-review.md` — explains the v1 → v2 changes.
**Tech stack:** Python, NumPy, SciPy (`scipy.sparse.csgraph.connected_components` used in the one-time full build and in tests), pandas, pytest.

Key changes from v1 plan:
- CLUMPTYPE formulas are Marxan-faithful (1 = amount/2, 2 = amount²/target2 quadratic).
- Shortfall is fractional `(T−a)/T` × baseline_penalty × SPF (Marxan strict, D1 from review).
- MISSLEVEL stays in the in-objective shortfall (pymarxan convention, D2 from review).
- New `ClumpState` class with incremental union-find / bounded local BFS handles the SA hot loop. Naive scipy recompute is reference-only (used in `compute_clump_penalty_from_scratch` for tests and the one-time full build).
- Heuristic stays clumping-blind during scoring; post-hoc reports via `build_solution` (honest scope).
- `ZoneMIPSolver` gets a brand-new `__init__` (currently has none).
- `feature_table` editing needs both `_COLUMN_ORDER` and `validate_feature_edit` whitelist updated.

---

## Batch 1 — Schema + I/O (3 tasks)

### Task 1: spec.dat accepts optional `target2` column

**Files:** `src/pymarxan/io/readers.py::read_spec`, `src/pymarxan/io/writers.py::write_spec`, `tests/pymarxan/io/test_io_clumping.py` (new).

**Failing tests:**
- `target2` column round-trips through write/read.
- Reader fills `target2=0.0` when column absent.
- Writer omits `target2` when every value is 0 (preserves legacy bit-identity).

### Task 2: spec.dat accepts optional `clumptype` column

Symmetric to Task 1. Default `0` (binary CLUMPTYPE).

### Task 3: validate() smoke test

`ConservationProblem.validate()` doesn't flag the new columns. One assertion.

**Commit message for Batch 1:**
```
Phase 19 Batch 1: TARGET2 + CLUMPTYPE schema and I/O

Adds optional target2 and clumptype columns to spec.dat. target2=0
(default) disables clumping for the feature; clumptype defaults to 0
(binary). Writers omit columns when all-default. Legacy projects
round-trip byte-identical.
```

---

## Batch 2 — Clumping math (3 tasks)

### Task 4: `compute_feature_components` pure function

**File:** new `src/pymarxan/solvers/clumping.py`.

Uses `scipy.sparse.csgraph.connected_components` on a sparse CSR adjacency masked to (selected ∩ has_feature).

**Failing tests:**
- Empty selection → no components.
- Selected isolated PUs (no edges) → each PU its own component.
- Linear chain → one component.
- Two disconnected chains → two components.
- Feature absent from some selected PUs → those PUs not in the feature's components.

### Task 5: `PartialPen4` (Marxan-faithful) — CLUMPTYPE 0/1/2

In `solvers/clumping.py`, implement `partial_pen4(occ, target2, clumptype) -> float` directly matching `clumping.cpp::PartialPen4`:

```python
def partial_pen4(occ: float, target2: float, clumptype: int) -> float:
    if occ >= target2:
        return occ
    if clumptype == 0:
        return 0.0
    if clumptype == 1:
        return occ / 2.0
    if clumptype == 2 and target2 > 0:
        return (occ * occ) / target2
    return 0.0
```

**Failing tests (6+, the distinguishing ones):**
- CLUMPTYPE 1 at occ=30, target2=50 → exactly 15.0 (NOT 30, that would be the v1 wrong formula).
- CLUMPTYPE 2 at occ=30, target2=50 → exactly 18.0 (= 30²/50, NOT 0 and NOT 30 — distinguishes from both v1 wrong formulas and from CLUMPTYPE 0).
- Each CLUMPTYPE at occ ≥ target2 → exactly `occ` (full credit).
- CLUMPTYPE 0 sub-target → 0.0.

### Task 6: `held_eff` and post-hoc clump evaluation

In `solvers/clumping.py`, implement:

```python
def compute_clump_penalty_from_scratch(
    cache: ProblemCache,
    selected: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Reference impl — used in tests and one-time full builds.
    NOT used in the SA hot loop. Returns (held_eff vector, total_penalty).
    """

def evaluate_solution_clumping(
    problem: ConservationProblem,
    selected: np.ndarray,
) -> tuple[dict[int, float], float]:
    """Post-hoc evaluator for the MIP 'drop' strategy and heuristic
    Solution.clump_shortfalls / clump_penalty population."""
```

Each per-feature contribution uses **fractional shortfall × baseline_penalty × SPF** (Marxan-faithful, D1):

```python
if target_j > 0:
    fractional = max(0.0, (target_j * misslevel - held_eff_j) / target_j)
else:
    fractional = 0.0
penalty_j = baseline_penalty_j * spf_j * fractional
```

Note: this path is the SA's **full-recompute reference** and the `build_solution` post-hoc reporter. The SA inner loop uses `ClumpState.delta_penalty` (Task 8) for performance. Both must agree numerically.

**Failing tests:**
- Two-component problem: hand-computed `held_eff` matches `compute_clump_penalty_from_scratch`.
- Same problem, all-zero `target2` → `held_eff == held` exactly; reduces to non-clumping path.
- Combined PROBMODE 3 + clumping → both penalties accumulate independently in the objective.

**Commit message for Batch 2:**
```
Phase 19 Batch 2: clumping math (pure functions + penalty integration)

Pure-functional layer in new pymarxan.solvers.clumping module:
compute_feature_components uses scipy.sparse.csgraph for component
detection on the (selected ∩ has_feature) subgraph;
compute_clump_adjusted_amounts applies CLUMPTYPE 0/1/2 rules.
compute_penalty in solvers/utils.py uses the clumping-adjusted
amount in place of the raw sum when target2_j > 0 for feature j.
```

---

## Batch 3 — Solver integration (4 tasks)

### Task 7: ProblemCache stores clumping params + baseline penalty

Add to `ProblemCache`:
- `feat_target2: np.ndarray (n_feat,)` — float64, default 0.
- `feat_clumptype: np.ndarray (n_feat,)` — int8, default 0.
- `feat_baseline_penalty: np.ndarray (n_feat,)` — float64. **NEW** in Task 7: greedy precompute. For each feature, sort PUs by `cost/amount_ij` ascending; accumulate amount until target is met; record the cost. This is the Marxan `compute_penalties()` semantics.
- `feat_uses_pu: list[np.ndarray]` — j → sorted PU indices where `amount_ij > 0` (used by `ClumpState` for fast neighbour filtering).
- `clumping_active: bool` — `any(feat_target2 > 0)`. Zero-cost gate.

Default-zero/inactive when columns missing — zero-cost for non-clumping problems.

**Failing tests:**
- `feat_baseline_penalty[j]` matches hand-computed greedy cost on a 4-PU, 2-feature constructed problem.
- `clumping_active == False` for legacy problems with no target2 column.
- `feat_uses_pu[j]` excludes PUs with amount==0 for feature j.

### Task 8: `ClumpState` incremental delta (the SA hot loop)

**New module sections** in `solvers/clumping.py`:

```python
@dataclass
class ClumpState:
    """Mutable companion to ProblemCache. Maintained alongside `held`
    and `total_cost` through the SA / iterative-improvement inner loop.

    Per-feature state:
      comp_of[j]: np.ndarray of length n_pu; -1 if PU not in any clump for j
      comp_occ[j]: dict[int, float] component_id -> occupancy
      held_eff[j]: float — cached Σ_c PartialPen4(occ_c, target2_j, ...)
      next_comp_id[j]: int — monotone id generator
    """

    @classmethod
    def from_selection(cls, cache, selected) -> "ClumpState":
        """O(edges + n_features × n_components) full build using scipy.csgraph.
        Called once before SA starts."""

    def delta_penalty(self, cache, idx, adding) -> float:
        """Returns the change in clumping penalty (baseline · spf · Δfractional)
        for flipping PU at `idx`. Does NOT mutate state. Cost: O(degree²)
        per feature in PU `idx`."""

    def apply_flip(self, cache, idx, adding) -> None:
        """Commits the flip to internal state via union-find on add /
        bounded local BFS on remove. Cost: O(degree²) per type-4 feature."""

    def held_effective(self) -> np.ndarray:
        """Returns the current held_eff vector for verification / build_solution."""
```

Wire `ClumpState` into the existing SA inner loop in `SimulatedAnnealingSolver.solve`:
- Allocate one `ClumpState.from_selection(cache, selected_init)` after `held` is initialised.
- On each candidate flip:
  ```python
  delta = cache.compute_delta_objective(idx, selected, held, total_cost, blm)
  if cache.clumping_active:
      delta += clump_state.delta_penalty(cache, idx, adding)
  ```
- On accept: `clump_state.apply_flip(cache, idx, adding)`.

`IterativeImprovementSolver` uses the same path through its own existing reuse of `ProblemCache.compute_delta_objective` — wire identically.

**Critical tests (mirrors Phase 18 cache delta-correctness):**
- For 200 random flips on a 50-PU grid with 2 type-4 features and target2 active:
  `clump_state.delta_penalty(...) ≈ from_scratch_after_penalty - from_scratch_before_penalty`
  to within 1e-9.
- After all 200 flips, `clump_state.held_effective()` matches `compute_clump_penalty_from_scratch(cache, selected)[0]` to within 1e-9 (no floating-point drift).
- `clumping_active == False` problems: SA produces byte-identical `Solution.objective` and `Solution.selected` to pre-Phase-19 (regression guard).

### Task 9: HeuristicSolver — post-hoc clump reporting (scoring is clump-blind)

**Honest scope, per architect review concern H2.** HeuristicSolver continues to score by raw rarity (clump-blind during construction); the existing `build_solution` call at the end populates `Solution.clump_shortfalls` and `Solution.clump_penalty` via the post-hoc evaluator. Same shape as Phase 18's MIP "drop" — report the gap, don't optimise against it. A future phase can add a clump-aware heuristic scorer.

**No solver-source changes for Task 9.** The work happens entirely in Task 10 (which wires `build_solution`'s post-hoc reporter).

`help_content.py` for the heuristic explainer documents this limitation explicitly.

### Task 10: MIP "drop" + Solver.supports_clumping + build_solution

`MIPSolver` gains `mip_clump_strategy: str = "drop"` kwarg. Default solves the deterministic problem; `build_solution` evaluates clumping post-hoc.

`ZoneMIPSolver` does **not currently have an `__init__`** (codebase agent flagged this). Create one from scratch:

```python
def __init__(self, *, mip_clump_strategy: str = "drop") -> None:
    if mip_clump_strategy not in ("drop", "big_m"):
        raise ValueError(...)
    self.mip_clump_strategy = mip_clump_strategy
```

Both raise `NotImplementedError` for `"big_m"` with a phase pointer.

`Solver.supports_clumping(self) -> bool` — base method returns True (default).

`build_solution` (`solvers/utils.py`) gains the post-hoc clumping populator, mirroring how Phase 18 populates `prob_shortfalls` / `prob_penalty`. **Single source of truth — every solver path inherits clumping reporting through `build_solution`.**

```python
if int(problem.parameters.get("PROBMODE", 0)) == 3:
    # existing Phase 18 prob_* populate
    ...
if "target2" in problem.features.columns and (problem.features["target2"] > 0).any():
    clump_shortfalls, clump_penalty = evaluate_solution_clumping(problem, selected)
    sol.clump_shortfalls = clump_shortfalls
    sol.clump_penalty = clump_penalty
```

Also: enforce `R6` (sepnum × target2 reject) in `validate()` or at problem construction.

**Failing tests:**
- `MIPSolver().solve(target2_problem)` runs; `Solution.clump_shortfalls` populated.
- `MIPSolver(mip_clump_strategy="big_m").solve(target2_problem)` raises NotImplementedError.
- `MIPSolver(mip_clump_strategy="bogus")` raises ValueError at __init__.
- All four solvers (SA, MIP, Heuristic, IterativeImprovement) populate `Solution.clump_shortfalls` end-to-end.
- `Solver.supports_clumping()` returns True for all current solvers.

**Commit message for Batch 3:**
```
Phase 19 Batch 3: clumping wired into ProblemCache + MIP drop strategy

ProblemCache.from_problem reads target2/clumptype from features and
sets clumping_active mask + has_any_clumping flag. When clumping is
active, compute_full_objective and compute_delta_objective use the
clumping-adjusted amount for type-4 features; SA + iterative-improvement
inherit automatically. MIPSolver/ZoneMIPSolver gain mip_clump_strategy
kwarg (default "drop"): deterministic solve + post-hoc gap on
Solution.clump_shortfalls / clump_penalty. New Solver.supports_clumping()
capability method.
```

---

## Batch 4 — Shiny UI + integration smoke (2 tasks)

### Task 11: Shiny UI surface

- `modules/data/feature_table.py`: add `target2` (numeric) and `clumptype` (int 0/1/2) columns to the editable feature table. **Per codebase agent**, this requires touching THREE places in the file:
  1. `_COLUMN_ORDER` constant — append `"target2"` and `"clumptype"`.
  2. The `feature_grid` render function — extend the `p.features[...]` column selection.
  3. `validate_feature_edit` — current whitelist `("target", "spf")` must be extended to `("target", "spf", "target2", "clumptype")` and the integer-only check for clumptype added.
- `modules/results/target_met.py`: when any feature has `target2 > 0`, add a `clump_short` column showing the clumping shortfall (or "—" for non-type-4 features).
- `modules/help/help_content.py`: explainer entry for type-4 species + CLUMPTYPE with the foundational citations (Ball-Possingham-Watts 2009, Metcalfe 2015).

File-based tests pin that the strings `"target2"`, `"clumptype"`, `"clump_short"`, and the Marxan-faithful CLUMPTYPE formulas appear in the right modules (Review 6 H4 pattern).

### Task 12: Integration smoke

`tests/integration/test_phase19_smoke.py`:

1. **All four single-zone solvers run end-to-end** under TARGET2. "Four solvers" = explicit list:
   - `SimulatedAnnealingSolver`
   - `MIPSolver` (default `mip_clump_strategy="drop"`)
   - `HeuristicSolver`
   - `IterativeImprovementSolver`
   Each must produce a `Solution` with `clump_shortfalls` and `clump_penalty` populated.
2. **Feature with `target2=0`** behaves identically with/without the new code path (the regression guard — pinned by hash of `Solution.objective` and `Solution.selected`).
3. **Feature with `target2 > 0`** produces a *different* solution under TARGET2 vs the same seed without TARGET2.
4. **save_project + load_project** preserves target2 and clumptype columns; PROBMODE 3 + TARGET2 round-trip together.
5. **Sepnum × target2 reject**: a problem with both `target2 > 0` and `sepnum > 0` raises `NotImplementedError` with a Phase 20 pointer.

**Commit message for Batch 4:**
```
Phase 19 Batch 4: Shiny UI + integration smoke (Phase 19 COMPLETE)

feature_table editable columns for target2/clumptype; target_met shows
clump_short column when any feature has target2>0; help content
documents the type-4 species formulation. Integration smoke exercises
all four solvers under TARGET2.
```

---

## Verification

After each batch: `make check`. After Batch 4: also `/opt/micromamba/envs/shiny/bin/pytest tests/integration/test_phase19_smoke.py -v`.

Targets:
- Tests: 1152 → ~1167
- Coverage: ≥91 %
- `make check` green
- `CHANGELOG.md` `[Unreleased]` gains Phase 19 entry

## Open questions — resolved by review

All OQs from the v1 plan are now resolved (see review doc §"TL;DR"):

- **OQ1 (CLUMPTYPE semantics)**: ✅ Resolved by scientific agent quoting `clumping.cpp::PartialPen4`:
  - CLUMPTYPE 0 = `0` (sub-target); CLUMPTYPE 1 = `amount/2`; CLUMPTYPE 2 = `amount²/target2` (quadratic).
- **OQ2 (heuristic scoring)**: ✅ Resolved as honest scope. Heuristic stays clumping-blind during scoring; reports the gap post-hoc via `build_solution` (Task 9). A future phase can add a clump-aware scorer.
- **OQ3 (PROBMODE 3 × TARGET2)**: ✅ Resolved by scientific agent: Marxan source treats both as independent additive shortfalls. Test plan item 11 pins per-feature routing.
- **OQ4 (delta perf)**: ✅ Resolved by adopting incremental `ClumpState` (architect + independent re-design convergence). Delta ≈ O(degree²) per flip. No benchmark gate needed at acceptance.
- **OQ5 (adjacency)**: ✅ Resolved by scientific agent: Marxan uses boundary file only. pymarxan matches; document explicitly in user-facing help that `connectivity.dat` does NOT contribute to clump contiguity.

## User decisions absorbed (from review §"Recommended next actions")

- **D1 (shortfall normalisation)** = **Marxan strict**. Fractional `(T−a)/T` × `baseline_penalty` × SPF. Baseline-penalty precompute added to `ProblemCache.from_problem` in Task 7.
- **D2 (MISSLEVEL placement)** = **Keep pymarxan extension**. MISSLEVEL stays in-objective in the clumping shortfall, consistent with the existing deterministic path. Documented as a divergence from Marxan classic in design doc §Assumptions item 6.
- **H1 (delta perf)** = **Adopt incremental ClumpState**. Task 8 implements union-find on add + bounded local BFS on remove.
