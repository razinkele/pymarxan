# Stage 3 (v0.41.0): exact multi-objective optimisation — design

**Date:** 2026-10-15 · **Baseline:** v0.40.0 (`8f8ea76`) · **Roadmap:**
`docs/plans/2026-10-11-prioritizr-parity-roadmap.md` Stage 3 · **Reference:** prioritizr 9.0.1
source (`R/multi_problem.R`, `R/multi_compile.R`, `R/add_hier_approach.R`,
`R/add_wtd_sum_approach.R`, `R/add_ref_point_approach.R`, `R/approach_*_matrix.R`,
`R/Solver-class.R`), read 2026-10-15. Research notes:
`scratchpad/stage3-prioritizr-multiobj.md` (session scratch; the findings that matter are
restated here).

## 1. Intent

Planners can trade cost against representation **exactly**: "the cheapest plan among the
best-representing ones" (hierarchical) and "the cost-vs-representation frontier"
(weighted-sum sweep), with prioritizr 9's API concepts.

**User decisions (brainstorm, 2026-10-15):**

| Question | Decision |
|---|---|
| Driving use case | A: cost vs representation (Baltic / ROWER); D (prioritizr parity) shapes the API |
| API shape | Approach A: one `ConservationProblem`, a list of objective specs, an approach object (`MultiObjectiveMIP`) — not prioritizr's list-of-problems (`multi_problem`) |
| Approaches | Hierarchical + weighted sum, with matrix sweeps; reference point deferred |

**Research findings that shape the design (prioritizr 9.0.1):**
- A `min_set` member's targets are hard constraints in every stage, so "min_set vs
  min_shortfall" cannot trade cost against representation. prioritizr's idiom is a pure-cost
  member (min_penalties + cost penalties, no targets) plus a representation member with
  `budget = NULL`. → pymarxan needs a `MinCost` objective and budget-optional representation
  objectives in the multi-objective path.
- Hierarchical caps are `v·(1±r)` (`Solver-class.R:214-220`): zero slack at v = 0, infeasible
  at v < 0. Gurobi indexes `rel_tol` by problem, the generic path by solve order.
- Weighted sum: maximise `Σ (w/Σw)·s·f`, s = −1 for min objectives; no objective scaling; an
  all-zero weight row gives NaN coefficients.
- Reference point is broken in 9.0.1: its max rows are written `t − w·d ≤ 0` instead of
  `≥ 0` (`rcpp_convert_ref_point_method.cpp:80-97`), so the min-max stage always gives t = 0
  and the method minimises a weighted sum of normalised shortfalls. → deferred.
- The `objective` attribute is read from raw auxiliary variables, which can misstate members
  not being optimised at that stage. → pymarxan recomputes from the selection.

**Success criteria (roadmap acceptance):** hierarchical on a 2-objective toy matches the
enumerated lexicographic optimum; the weighted-sum sweep reproduces the convex hull of an
enumerated frontier; single-objective results unchanged (`tests/data/simple` = 35.0).

## 2. Refactor of `MIPSolver` (prerequisite)

`MIPSolver._build` (`src/pymarxan/solvers/mip_solver.py`) is split:

1. **`_build_base(problem, config) -> _BaseModel`** — everything an objective does not own:
   decision variables `x`, locked-in / locked-out bounds, every existing constraint family
   (contiguity, feature contiguity, min-neighbours, linear, separation / clumping strategies,
   PROBMODE 3 chance constraints, no-good cuts), and the expressions `cost_expr`,
   `boundary_expr` (BLM-weighted terms), `conn_expr`, `prob_expr`, `amount_exprs`, and the
   target terms (`_core_target_terms`). It does **not** add hard target rows, the cost budget
   row, or an objective.
2. **Objective attachment** — single-objective `MIPSolver` keeps its exact current behaviour
   by composing the base with: hard targets (min_set, as today), `COSTBUDGET` (budgeted
   objectives, required as today), and the objective from the builder table.

All six single objectives go through builders in `_mip_objectives.py`; the remaining
`if/elif` blocks for `max_features`, `max_weighted_features` and absolute
`min_largest_shortfall` are replaced by the existing `BUILDERS` entries. Two builders are
new:
- **`build_min_cost`** — `cost_expr + blm·boundary_expr + conn_expr + prob_expr`, exactly the
  terms `min_set` minimises today (so `min_set` = `min_cost` + hard targets and the 35.0
  anchor is unchanged by construction).
- **`build_min_boundary`** — the boundary term alone (unweighted by BLM: compactness as its own
  objective).
- **`build_max_wtd_sum`** — `Σ_j w_j · achieved_j` (maximise weighted amount held; prioritizr
  `max_wtd_sum`, backlog item folded in because the prioritizr parity tests use it). Feature
  weights default to 1.

Builders that need expressions beyond `cost_expr` and target terms read them from an extended
`ObjectiveContext` (new optional fields `boundary_expr`, `penalty_expr`, `amount_exprs`, all
defaulting to empty expressions so `ZoneMIPSolver`'s use of the context is unchanged).

**Golden tests first:** before `_build` is touched, record `MIPSolver` results (selected set,
cost, objective, targets_met, metadata["objective"]) for every objective × shortfall mode,
with and without BLM, PROBMODE 2 and 3, locks, and a no-good cut. The refactor must reproduce
them exactly.

## 3. Objective specs (`pymarxan.multiobjective`)

New module `src/pymarxan/multiobjective.py` (core layer, no UI). Frozen dataclasses:

| Spec | Builder | Sense | Notes |
|---|---|---|---|
| `MinCost()` | `build_min_cost` | min | cost + BLM boundary + connectivity + probability penalty |
| `MinBoundary()` | `build_min_boundary` | min | boundary only |
| `MinShortfall(shortfall="relative")` | `build_min_shortfall` | min | prioritizr min_shortfall (relative default) |
| `MinLargestShortfall(shortfall="relative")` | `build_min_largest_shortfall` | min | |
| `MaxFeatures()` | `build_max_features` | max | count of targets met |
| `MaxWeightedFeatures()` | `build_max_weighted_features` | max | SPF-weighted count |
| `MaxWtdSum(weights=None)` | `build_max_wtd_sum` | max | `weights`: None (all 1) or `{feature_id: w}` (finite, any sign; id-keyed; missing id → ValueError) |

Each spec has `name` (e.g. `"min_cost"`, `"min_shortfall"`) used as the key in reported
objective values; two specs with the same name in one problem must be distinguished by an
explicit `label=` argument (all specs accept `label: str | None = None`), otherwise
`ValueError`.

**Targets and budget inside the multi-objective model:**
- Targets are **soft**: no hard target rows. Only representation objectives use targets (as
  slack / met variables via `ObjectiveContext`).
- `COSTBUDGET` is **optional**: if `problem.parameters` has it, `cost_expr ≤ budget` is added
  once to the base model and binds every stage; otherwise there is no budget row.
  Single-objective `MIPSolver` keeps requiring it for its budgeted objectives.
- Locks and every constraint family in the base model bind every stage and every run.
- `MISSLEVEL` and SPF enter through the target terms exactly as in single-objective builders.

## 4. Approaches

### 4.1 `Hierarchical(rel_tol, abs_tol=1e-9)`
- Priority = order of the `objectives` list (first is solved first). No separate `priority`
  argument.
- `rel_tol`: a float (applied to every stage), a sequence of n−1 floats, or a 2-D sequence
  (one row of n−1 per run). Each value finite and ≥ 0; values > 1 allowed.
- Stage k (k < n): solve with objective k; let v be its value; add the cap row
  `"hier_cap_{k}"`:
  - minimise: `f_k ≤ v + r_k·|v| + abs_tol`
  - maximise: `f_k ≥ v − (r_k·|v| + abs_tol)`
  Then stage k+1. The last stage's solution is returned.
- v is the solver's value at the configured `MIP_GAP` (default 0 → proven optimum; documented).
- Implementation: generalise `solve_lexicographic` to `solve_hierarchical(model, terms,
  rel_tols, abs_tol, solver)` over `ObjectiveTerm`s with mixed senses. `solve_lexicographic`
  becomes a two-term wrapper with its current signature and behaviour (min_penalties
  unchanged; its tests are the guard).
- Stage 1 with no solution → `RuntimeError` naming the stage and suggesting
  `presolve_check(problem)`. A later stage with no solution (only numerically possible — the
  previous solution satisfies the cap) → restore the previous stage's values, one
  `UserWarning`, `metadata["stage_reached"] = k`.
- Cap rows are removed after each run so sweep rows are independent.

### 4.2 `WeightedSum(weights)`
- `weights`: a sequence of n floats, or a 2-D sequence (one row per run). Each finite and ≥ 0;
  a row summing to 0 → `ValueError` (prioritizr gives NaN).
- One solve per row: minimise `Σ_o (w_o / Σw) · s_o · f_o`, s = +1 for min, −1 for max.
- **No objective scaling** (prioritizr parity). Docstring and tutorial state that objectives
  in different units need weights scaled to their magnitudes (Das & Dennis 1997).

### 4.3 Matrix helpers (prioritizr parity)
- `weights_matrix(n_objectives, n_values, *, include_zeros=True, include_extremes=True)` —
  prioritizr `approach_weights_matrix` algorithm exactly (§ research notes 2.4): grid of
  `linspace(0,1,n_values)` (or `n_values+1` points without 0 when `include_zeros=False`),
  drop all-equal rows, drop rows with < 2 entries > 1e-6, prepend `identity(n)` if
  `include_extremes`, always prepend a row of ones. Row order follows R's `expand.grid`
  (first column varies fastest). Pinned: (2,5) → the 15 rows listed in the notes; (3,3) → 22
  rows; (3,5) → 112 rows.
- `rel_tol_matrix(n_objectives, n_values, max, *, include_zeros=True, order=True)` — prioritizr
  `approach_rel_tol_matrix`: grid over n−1 columns of `linspace(0,max,n_values)` (or without
  0); if `order`, keep rows non-increasing across columns. Pinned: (3,3,1) →
  `[0,0],[.5,0],[1,0],[.5,.5],[1,.5],[1,1]`; (3,5,1) → 15 rows; (3,5,1,order=False) → 25.
- Both return `list[list[float]]` (no numpy dependency in the signature); docstrings note
  prioritizr's doc overclaim (proportional duplicates are kept).

### 4.4 Out of scope (follow-ons)
Reference point (broken in prioritizr 9.0.1; a correct Chebyshev/achievement version can be
built on the same builders later); zones (`ZoneMIPSolver`); per-member features / targets /
costs (prioritizr's list-of-problems model); Gurobi native multi-objective; solution pools /
deduplication beyond what `nondominated` gives.

## 5. `MultiObjectiveMIP`

```python
MultiObjectiveMIP(objectives: Sequence[ObjectiveSpec], approach: Hierarchical | WeightedSum,
                  *, mip_backend="auto", mip_chance_strategy="drop",
                  mip_clump_strategy="drop", mip_sep_strategy="drop",
                  warn_above_pu=200_000, presolve_check=False)
    .solve(problem: ConservationProblem, config: SolverConfig | None = None) -> list[Solution]
```
- Not a `Solver` subclass (it returns one solution per sweep row and needs specs); it reuses
  `MIPSolver`'s strategy validation, backend factory, `MIP_TIME_LIMIT` / `MIP_GAP`
  parameters, presolve hook and `_build_base`.
- Validation at construction: ≥ 1 objective (n = 1 is allowed and equals a single solve);
  label uniqueness; approach type known; `rel_tol` / weights shapes match n.
- Builds the base model once per `solve`; each run adds its objective / cap rows and removes
  them afterwards.
- Single-zone problems only: a `ZonalProblem` → `TypeError` naming the follow-on.

**Outputs** — one `Solution` per run, in row order:
- `selected`, `cost`, `boundary`, `targets_met`, `objective` exactly as `MIPSolver` fills them
  (Marxan-style recomputation), so every existing consumer works.
- `metadata["objective_values"]`: `{label_or_name: value}` for every objective,
  **recomputed from the selection** (each builder's expression evaluated at the selected `x`,
  with slack / met variables set to their implied values, not read from the solver).
- `metadata["approach"]` (`"hierarchical"` / `"weighted_sum"`), `["row"]`, `["rel_tol"]` or
  `["weights"]`, `["stage_reached"]` (hierarchical), `["mip_backend"]`.
- `nondominated(solutions) -> list[Solution]` (module-level): keeps solutions not dominated
  under each objective's sense using `objective_values`; identical value vectors keep the
  first. Same domination rule as `calibration.pareto.pareto_frontier`, generalised.

## 6. Errors

All `ValueError` naming the argument unless stated: empty `objectives`; duplicate names
without labels; `rel_tol` length ≠ n−1 or weights length ≠ n (per row); negative / non-finite
`rel_tol`, `abs_tol` or weights; all-zero weights row; `MaxWtdSum` weights missing a feature
id; unknown approach (`TypeError`); `ZonalProblem` (`TypeError`); stage-1 infeasible
(`RuntimeError`).

## 7. Testing

1. **Golden tests before the refactor** (§2): every single-objective configuration listed
   there, recorded on the unmodified source; anchor 35.0; existing `min_penalties`
   lexicographic tests unchanged.
2. **Enumeration oracles** on a 6–8 PU toy (brute force over 2ⁿ selections):
   - hierarchical (MinCost → MinShortfall and MinShortfall → MinCost), rel_tol 0 and > 0,
     equals the enumerated lexicographic optimum value pair;
   - weighted-sum sweep reproduces every vertex of the enumerated (cost, shortfall) lower
     convex hull;
   - `nondominated` over the sweep equals the enumerated Pareto points that the sweep found.
3. **prioritizr parity** on its 6-PU fixture (cost 1 each; ft1 = (5,.5,0,0,0,0),
   ft2 = (0,0,0,0,3,2), ft3 = (0,0,10,1,.5,0); COSTBUDGET 2; prioritizr's two max_wtd_sum
   problems (ft1 only, ft2 only) map to `MaxWtdSum(weights={1: 1, 2: 0, 3: 0}, label="ft1")`
   and `MaxWtdSum(weights={1: 0, 2: 1, 3: 0}, label="ft2")` on one problem holding all three):
   w = (1,1) → x = 100010, values (5, 3); rows [(1,0),(0,1),(1,1),(1,5)] →
   x = 110000, 000011, 100010, 000011, values (5.5,0), (0,5), (5,3), (0,5). Matrix helpers
   pinned as in §4.3. (Hierarchical has no literal prioritizr cases; the oracle covers it.)
   The plan re-derives these expected values by enumeration on the fixture before pinning
   them, so a mapping mistake shows up as an oracle mismatch rather than a silent pass.
4. **Bug-fix regressions:** v = 0 cap with rel_tol > 0 (zero-cost locked-in PUs) still allows
   the tolerance via abs_tol / |v|; v < 0 (negative MaxWtdSum weights or negative cost) is
   feasible; a maximised objective with tolerance; all-zero weights row; recomputed
   `objective_values` differ from raw auxiliary values in a constructed case.
5. **Budget / targets:** COSTBUDGET honoured when set, absent otherwise; a target unmet at the
   optimum (soft targets).
6. **Mutation spot-checks:** flip the cap inequality, drop `|v|`, flip the weighted-sum sign —
   each must fail a test.

## 8. Docs

- `docs/TUTORIAL.md` section "Multi-objective planning (v0.41)": weighted-sum sweep of
  MinCost vs MinShortfall with scaled weights → `nondominated` → `Hierarchical` pick (cheapest
  plan within 5% of the best shortfall). Pinned by `tests/test_tutorial_examples.py`.
- CHANGELOG `[Unreleased]`: Added (MultiObjectiveMIP, specs, approaches, helpers,
  `nondominated`, `MinCost`/`MinBoundary`/`MaxWtdSum` builders); Changed (`MIPSolver` internals
  refactored onto builders, behaviour unchanged); deviations from prioritizr (abs tolerance and
  |v| in caps; priority = list order; objective values recomputed; all-zero weights rejected;
  reference point omitted as broken in 9.0.1). Sources: Williams & Kendall 2017
  (10.1016/j.ecolmodel.2016.10.010), Das & Dennis 1997 (10.1007/BF01197559), Schuster et al.
  2023 (10.1111/cobi.14048), Hanson et al. 2024 prioritizr (10.1111/cobi.14376).
- README feature list; roadmap Stage 3 marked as shipped at release.

## 9. Risks

- The `_build` split touches the most-used solver — golden tests (§7.1) are the guard; the
  plan records them in their own first task.
- Unscaled weighted sums are easy to misuse — docstring + tutorial show scaled weights.
- PuLP model mutation across sweep rows (adding / removing cap and objective rows) must leave
  the base model identical; a test solves the same row twice and compares.
