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
  not being optimised at that stage. → pymarxan recomputes reported values from the selection.

**Success criteria (roadmap acceptance):** hierarchical on a 2-objective toy matches the
enumerated lexicographic optimum; the weighted-sum sweep reproduces the convex hull of an
enumerated frontier; single-objective results unchanged (`tests/data/simple` = 35.0).

## 2. Refactor of `MIPSolver` (prerequisite)

`MIPSolver._build` (`src/pymarxan/solvers/mip_solver.py`) is split:

1. **`_build_base(problem, config, *, need_boundary=False) -> _BaseModel`** — exactly what
   `_build` does today minus the objective-specific parts (inventory verified against the code
   in spec review round 1):
   - strategy gating (`mip_*_strategy`; "drop" adds nothing, the others raise
     `NotImplementedError` as today);
   - `x` variables and locked-in / locked-out rows;
   - `cost_expr`;
   - the **unweighted** boundary expression `boundary_expr` (y / abs rows) — built when a
     boundary table exists and (BLM > 0 **or** `need_boundary`); today it is built only when
     BLM > 0;
   - connectivity z rows and `conn_expr` (symmetric and asymmetric);
   - PROBMODE 1 `prob_expr` and PROBMODE 2 amount scaling (`prob_map_m2`), keeping today's
     `PROBMODE` defaults exactly (a golden case covers a probability table with no PROBMODE);
   - `amount_exprs` and the target terms (`_core_target_terms`);
   - no-good cuts from `config.metadata["forbidden_selections"]`.
   It does **not** add hard target rows, ghost-feature bounds, the cost budget row, or an
   objective. (Contiguity, feature contiguity, min-neighbours and linear constraints are not
   wired into `MIPSolver` today — their `apply_to_mip` is only called from tests — so they are
   not part of the base; wiring them is out of scope.)
2. **Objective attachment** — single-objective `MIPSolver` keeps its exact current behaviour
   by composing the base with: hard targets (min_set, as today), `COSTBUDGET` (budgeted
   objectives, required as today), and the objective from the builder table.

All six single objectives go through builders in `_mip_objectives.py`; the remaining
`if/elif` blocks for `max_features`, `max_weighted_features` and absolute
`min_largest_shortfall` are replaced by the existing `BUILDERS` entries. **One deliberate
behaviour change** follows (found in spec review): the legacy absolute `min_largest_shortfall`
adds slack rows only for features with `pu_vs_features` rows, so a ghost feature (target > 0,
no rows) is silently ignored — on the simple fixture plus a ghost with target 50 and COSTBUDGET
30 it reports a largest shortfall of 5.0, while the ghost's whole target (50) is unmet. The
builder counts it (m ≥ R), consistent with relative mode, with `ZoneMIPSolver`, and with the
v0.39 ghost fix for `max_*`. CHANGELOG `### Fixed`; a golden case pins the new value. Every
other golden case must reproduce exactly. Three builders are new:
- **`build_min_cost`** — `cost_expr + blm·boundary_expr + conn_expr + prob_expr`, exactly the
  terms `min_set` minimises today (so `min_set` = `min_cost` + hard targets and the 35.0
  anchor is unchanged by construction). With BLM > 0 it already counts boundary, so a weighted
  sum of `MinCost` and `MinBoundary` counts boundary twice — documented.
- **`build_min_boundary`** — the boundary term alone (unweighted by BLM: compactness as its own
  objective).
- **`build_max_wtd_sum`** — `Σ_j w_j · achieved_j` (maximise weighted amount held; prioritizr
  `max_wtd_sum`, backlog item folded in because the prioritizr parity tests use it). Feature
  weights default to 1.

**Builder protocol (fixed here so both solvers keep working).**
- `ObjectiveContext` gains these fields, each with a default, so `ZoneMIPSolver`'s
  `ObjectiveContext(model, cost_expr, terms)` call is unchanged:
  - `blm: float = 0.0`
  - `boundary_expr: LpAffineExpression` (unweighted), default empty
  - `conn_expr: LpAffineExpression`, default empty
  - `prob_expr: LpAffineExpression`, default empty
  - `amount_exprs: dict[int, LpAffineExpression]` keyed by feature id, default `{}`

  `MIPSolver` fills them from `_build_base`. (There is no `penalty_expr`.)
- Every builder has the signature `builder(ctx, shortfall, *, owner="", **params) -> ObjectiveTerm`.
  `BUILDERS[name](ctx, shortfall)` therefore still works, and `zones/mip_solver.py:251` is
  untouched. The only extra param is `weights` (`build_max_wtd_sum`).
- **Name ownership:**
  - A variable or constraint the builder *owns* is named `f"{ctx.prefix}{owner}{stem}"`, e.g.
    `max_shortfall`, `maxs_{key}`. With the default `owner=""` every name is exactly today's.
  - The per-target slack and met variables stay `ctx.slack()` / `ctx.met()`. They are named with
    `ctx.prefix` only and are shared.
  - The multi-objective path passes `owner=f"{key}__"`, where `key` is the spec's effective key
    (§3).
- New builders:
  - `build_min_cost`: `cost_expr + ctx.blm·boundary_expr + conn_expr + prob_expr`.
  - `build_min_boundary`: `boundary_expr`.
  - `build_max_wtd_sum(ctx, shortfall, *, owner="", weights)`: `Σ_j weights[j]·amount_exprs[j]`.
    It owns no variables.
- **Renames and LP changes.** Replacing `MIPSolver`'s legacy blocks with builders changes
  the exported LP.
  - Variables: `feat_met_{id}` → `y_f{id}`, and `slack_{id}` → `s_f{id}`.
  - Constraints: the relaxed `target_{id}` → `met_f{id}`, `slack_{id}` → `short_f{id}`, and
    `max_shortfall_{id}` → `maxs_f{id}`.
  - The objective name `"objective"` is dropped.
  - The hard-target `target_{id}` rows of `min_set` keep their name.
  - Absolute `min_largest_shortfall` also changes structure: the builder adds no slack row
    for a feature with R ≤ 0 (target 0 or MISSLEVEL 0), where the legacy code did.

  Solutions are unaffected, except the ghost case. `build_model` / LP-file output changes, so
  the CHANGELOG "Changed" section says so.
- **Builder typing and defaults:**
  - `BUILDERS` is typed with a `Protocol`
    (`__call__(ctx, shortfall, /, *, owner: str = "", **params) -> ObjectiveTerm`), so calls
    with `owner=` pass `make types`.
  - `build_max_wtd_sum` takes `weights: Mapping[int, float] | None = None` (None = all 1).
  - It reads `ctx.amount_exprs.get(j, empty)`, so a ghost feature contributes 0, not a
    KeyError.
  - The expression fields of `ObjectiveContext` use `field(default_factory=pulp.LpAffineExpression)`,
    and `amount_exprs` uses `default_factory=dict`.

**Golden tests first:** before `_build` is touched, record `MIPSolver` results (selected set,
cost, objective, targets_met, metadata["objective"]) for every objective × shortfall mode,
with and without BLM, connectivity (symmetric and asymmetric), PROBMODE 1 and 2, a
probability table with no PROBMODE set, locks, a no-good cut, and a ghost feature with
MISSLEVEL ≠ 1. The refactor must reproduce them exactly, except in two places.

- **The existing v0.38 golden file** (`tests/pymarxan/solvers/test_stage1_golden_v038.py`,
  data `tests/data/golden/stage1_v038.json`) pins canonical LP models by name. Its two entries
  `core_mls_abs_relabs` and `core_mls_abs_infeasible` are regenerated. The written reason,
  which that file's docstring requires, is: absolute `min_largest_shortfall` moved onto
  `build_min_largest_shortfall`, which renames its rows and variables and drops R ≤ 0 slack
  rows (see the renames above). The selection pinned there (`[1, 0, 0]`) must not change.
  Every other entry in that file must stay byte-identical.
- **The absolute `min_largest_shortfall` ghost case above.** For that case the golden test pins only the new
objective value (50.0), with a comment, and not the selection. The ghost holds m at 50, so
every selection ties, and CBC returns the empty plan where the legacy code returned a cost-23
one. The CHANGELOG `### Fixed` entry says that the plan degenerates when a ghost is present
and points to `presolve_check` (P5).

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
| `MaxWtdSum(weights=None)` | `build_max_wtd_sum` | max | `weights`: None (all 1) or `{feature_id: w}` covering every feature in the problem (finite, any sign; an unlisted problem feature or an id not in the problem → ValueError) |

**Keys.**
- Each spec has a `name`, e.g. `"min_cost"` or `"min_shortfall"`.
- All specs accept `label: str | None = None`.
- A spec's **effective key** is `label or name`. Effective keys must be unique within one
  `MultiObjectiveMIP`, otherwise `ValueError`. So
  `[MinShortfall(), MinShortfall(label="min_shortfall")]` is rejected.
- The effective key is used:
  - as the key in reported objective values;
  - as the owner prefix (`f"{key}__"`, §2) of every variable and constraint the spec's builder
    owns, so two `MinLargestShortfall` specs do not collide. The per-target slack and met
    variables are shared (see below).
- Labels must be Python identifiers (`str.isidentifier()`), because PuLP rewrites characters
  such as space and `-` to `_`, and distinct labels could then collide.
- Each spec has a method `build(ctx) -> ObjectiveTerm` that calls its builder with
  `shortfall`, `owner` and, for `MaxWtdSum`, `weights`.
- Each spec has `sense` (`"min"` / `"max"`).

**Terms are built once.** `MultiObjectiveMIP.solve` builds every spec's `ObjectiveTerm` once,
on one shared `ObjectiveContext` (so slack / met variables are shared between objectives that
use the same target), right after the base model. Each run only calls `setObjective` and adds
/ deletes its `hier_cap_*` rows (`del model.constraints[name]`, verified clean with CBC in
review). Builders must never be called twice on one context (they add named rows).

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
- Stages are numbered 1..n. In stage k (k < n): solve with objective k; let v be its value;
  add the cap row `"hier_cap_{k}"`:
  - minimise: `f_k ≤ v + r_k·|v| + abs_tol`
  - maximise: `f_k ≥ v − (r_k·|v| + abs_tol)`
  Then stage k+1. `r_k` is indexed by **solve order**: the k-th tolerance caps the k-th stage.
  The last stage's solution is returned.
- **What v is:** the **better**, in objective k's sense, of two values. For a minimised
  objective it is `min(v_model, v_rec)`; for a maximised one, `max(v_model, v_rec)`.
  - `v_model` is the model's value of objective k's expression (`term.expr`) at the accepted
    solution. It is read after every variable whose `varValue` is still `None` has been set
    to 0. A PU variable that is never written to the LP (zero cost, no rows) keeps `None`
    after the reset, and `pulp.value` of an expression containing a `None` variable is
    `None`.
  - `v_rec` is the exact recomputation from the rounded selection (§5 rule).
  - **Both values are attained by the stage's own plan.** The model value is attained by the
    solver's solution. The recomputed value is attained by the same rounded x with every
    auxiliary set to its implied value: s = max(0, R − A) satisfies s ≥ R − A, y = 1 iff
    A ≥ R satisfies A ≥ R·y, and m, boundary y and connectivity z are set the same way. So a
    cap built from the better of the two stays feasible, and it is never looser than either.
    Using either value alone loses targets:
    - **Recompute only** (round 4, executed): amounts 0.7 + 0.1 = 0.7999999999999999 against
      target 0.8. CBC sets y = 1 within tolerance, so `v_model` = 1 but `v_rec` = 0. The cap
      `f ≥ 0 − tol` constrains nothing, and `[MaxFeatures, MinCost]` returned the empty plan
      at cost 0 instead of cost 2 with the target met.
    - **Model only** (round 5, executed): an accepted incumbent (`sol_status` 2, from a time
      limit or `MIP_GAP` > 0) need not have tight auxiliaries. CBC returned y = 0 for
      targets whose amount reached the target. In one run `[MaxFeatures, MinCost]` capped at
      `v_model` = 31 while the plan met 36 targets, so stage 2 gave up 5 targets.
  - The raw-auxiliary misreporting prioritizr suffers from (objectives *not* being optimised)
    is a reporting concern only. `objective_values` always uses the recomputation (§5).
  - With `MIP_GAP` > 0, or a time limit, v is an incumbent's value, not a proven optimum
    (documented).

  **Solution acceptance (every hierarchical stage and every weighted-sum row):**
  before each solve every `varValue` is reset to `None`; a solve is accepted only if
  `model.sol_status` is `LpSolutionOptimal` (1) or `LpSolutionIntegerFeasible` (2). Anything
  else is "no solution" — CBC can leave non-None, constraint-violating values after a failed
  solve, and values left from the previous stage would trivially satisfy its caps, so no
  value-based heuristic is used. Tests mock `sol_status` for both paths.
- At v = 0 the cap allows only `abs_tol`, whatever `rel_tol` is. prioritizr has no abs_tol
  and allows zero slack there. Users who want slack at v = 0 raise `abs_tol`.
- Our caps differ from prioritizr's `v·(1±r)` in two ways:
  - by `abs_tol` at every v;
  - when v < 0, prioritizr's cap is tighter than the optimum, so the problem is infeasible.
    Ours is not.
- `MIP_TIME_LIMIT` applies per stage, so a run takes up to n × T and a sweep up to
  rows × n × T (documented). Stages are not warm-started, so a time-limited later stage can
  end with no incumbent (handled below). Warm starts, and reusing stage 1 across sweep rows,
  are follow-ons.
- **Implementation:** `solve_lexicographic` is **left untouched**. It is not a wrapper. It
  differs from the new loop in its acceptance rule, cap formula, source of v and stage-1
  failure behaviour, and its tests mock only `status`.
- `solve_hierarchical(model, terms, rel_tols, abs_tol, solver, evaluate) -> HierarchicalResult`
  is new code in `_mip_objectives.py`. `evaluate: Callable[[int], float]` maps a 0-based
  objective index to its `v_rec` at the current variable values; `MultiObjectiveMIP`
  supplies it, so the helper itself needs no problem data. The helper computes `v_model`
  itself and takes the better of the two. It works over `ObjectiveTerm`s with mixed senses
  and returns
  `HierarchicalResult(status: int, stage_reached: int)`. It raises nothing itself;
  `MultiObjectiveMIP` turns the result into errors and warnings.
- `stage_reached` is the number of accepted stages (n on full success).
- After a stage with no solution, the loop stops.
- Stage 1 with no solution (`stage_reached == 0`) → `RuntimeError` naming the stage and
  suggesting `presolve_check(problem)`.
- A later stage with no solution → the previous stage's snapshot of variable values is
  restored, one `UserWarning` is issued, and `metadata["stage_reached"] = k − 1`. This can
  happen numerically, or when a time limit is hit with no incumbent; otherwise the previous
  solution satisfies the cap.
- A weighted-sum row with no solution → `RuntimeError` naming the row. This deliberately
  aborts the whole sweep (documented); warn-and-skip is a follow-on.
- Cap rows are removed after each run (also on failure) so sweep rows are independent.

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
  prioritizr's doc overclaim (proportional duplicates are kept). Both require
  `n_objectives ≥ 2` and `n_values ≥ 2` (`ValueError` otherwise; prioritizr's
  `weights_matrix(1, …)` returns a meaningless `[[1],[1]]`).

### 4.4 Out of scope (follow-ons)
Reference point (broken in prioritizr 9.0.1; a correct Chebyshev/achievement version can be
built on the same builders later — the roadmap's reference-point acceptance item is recorded
there as an explicit descope); a connectivity objective on its own (connectivity stays a
penalty inside `MinCost`); wiring contiguity / min-neighbour / linear constraints into the
MIP; zones (`ZoneMIPSolver`); per-member features / targets /
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
  effective-key uniqueness; approach type known; `rel_tol` / weights shapes match n.
- Builds the base model and every objective term once per `solve` (§3); each run sets its
  objective and adds / deletes its cap rows. `config.num_solutions` is ignored (one solution
  per row, documented); `config.verbose` is honoured.
- The API is named `MultiObjectiveMIP` (not the roadmap's `MultiObjectiveProblem`): it is a
  solver-like object over an ordinary `ConservationProblem`, not a new problem type.
- Single-zone problems only: a `ZonalProblem` → `TypeError` naming the follow-on.

**Outputs** — one `Solution` per run, in row order:
- `selected`, `cost`, `boundary`, `targets_met`, `objective` exactly as `MIPSolver` fills them
  (Marxan-style recomputation), so every existing consumer works.
- `metadata["objective_values"]`: `{label_or_name: value}` for every objective,
  **recomputed from the selection** with one fixed rule per builder, from the selected `x`
  rounded to {0,1}: achieved A_j = Σ amount·x (PROBMODE-2 scaled as in the model); slack
  s_j = max(0, R_j − A_j); met y_j = 1 iff A_j ≥ R_j exactly (the `targets_met` /
  `check_targets` rule); largest shortfall m = max over R_j > 0 of s_j/R_j (relative) or s_j
  (absolute), 0 when no R_j > 0; cost / boundary / connectivity / probability terms evaluated
  directly. A PU whose `varValue` is `None` counts as 0, as in `MIPSolver`. These values
  are reported in `objective_values` and are also `v_rec` in hierarchical caps (§4.1).
  - Under PROBMODE 2 the recomputed met uses scaled amounts, while `Solution.targets_met`
    (`check_targets`) uses unscaled ones, so the two can differ there (documented).
  - **Invariant:** each recomputed value is attained by the selected `x`, with every
    auxiliary (s, m, y, boundary y, connectivity z) at its implied value. It is the best
    value of the objective's model expression over the exact implied auxiliaries. Two
    consequences:
    - a stage's own plan always satisfies its own cap, within solver tolerance;
    - at a proven optimum, away from target boundaries, the recomputed value equals the model
      value.

    A test checks both on oracle data.
  - **Tolerance caveat:** on tolerance-boundary data the reported met count can sit below the
    cap's v (round-4 case: the model has the target met, the recompute does not).
    `objective_values` stays consistent with `targets_met`. This is documented, and a
    regression test pins the cap behaviour (§7.4).
- `metadata["approach"]` (`"hierarchical"` / `"weighted_sum"`)
- `metadata["row"]`
- `metadata["rel_tol"]` or `metadata["weights"]`
- `metadata["stage_reached"]` (hierarchical only; see §4.1)
- `metadata["status"]`: the PuLP status string of the final accepted solve.
- `metadata["sol_status"]`: its PuLP solution status, int 1 or 2. 2 marks an incumbent that is
  not proven optimal: CBC reports status "Optimal" with sol_status 2 when a time limit is hit
  with an incumbent.
- `metadata["objective_senses"]`: `{key: "min" | "max"}`.
- `metadata["mip_backend"]`
- `metadata["solver"] = "MultiObjectiveMIP"`

There is no `metadata["objective"]`: single-objective `MIPSolver` sets it, and here
`objective_values` replaces it.
- `nondominated(solutions) -> list[Solution]` (module-level) keeps the solutions that are
  not dominated, using `objective_values` and `objective_senses`.
  - A solution is dominated if another is at least as good on every key and strictly better
    on one, with "better" taken in each key's sense.
  - Values are compared exactly, with no tolerance.
  - Identical value vectors keep the first, in input order.
  - Input order is preserved.
  - An empty input gives `[]`.
  - Solutions whose key sets or senses differ, or that lack these metadata keys, raise
    `ValueError`.
  - It applies the same domination rule as `calibration.pareto.pareto_frontier`, but is new
    code: `pareto_frontier` works on a 2-objective `BLMResult`, not on solutions. Unlike
    `pareto_frontier`, which keeps duplicate points, `nondominated` keeps only the first of
    identical value vectors.
- **Public names** (`pymarxan.multiobjective.__all__`): `MultiObjectiveMIP`, `Hierarchical`,
  `WeightedSum`, `ObjectiveSpec`, `MinCost`, `MinBoundary`, `MinShortfall`,
  `MinLargestShortfall`, `MaxFeatures`, `MaxWeightedFeatures`, `MaxWtdSum`, `weights_matrix`,
  `rel_tol_matrix`, `nondominated`. The module is added to the subpackage table in
  `.github/copilot-instructions.md`.

## 6. Errors

Every error is a `ValueError` naming the argument, unless another type is stated:
- empty `objectives`;
- duplicate effective keys (`label or name`);
- a label that is not an identifier;
- `rel_tol` length ≠ n−1, or weights length ≠ n (checked per row);
- negative or non-finite `rel_tol`, `abs_tol` or weights;
- an all-zero weights row;
- `MaxWtdSum` weights that do not cover every problem feature, or that name an unknown id;
- matrix helpers with n < 2 or n_values < 2, or a `rel_tol_matrix` `max` that is negative or
  non-finite;
- an unknown approach (`TypeError`);
- a `ZonalProblem` (`TypeError`);
- stage 1, or a weighted-sum row, with no solution (`RuntimeError`);
- `nondominated` given solutions whose key sets or senses differ, or that lack those metadata
  keys.

## 7. Testing

1. **Golden tests before the refactor** (§2): every single-objective configuration listed
   there, recorded on the unmodified source; anchor 35.0; existing `min_penalties`
   lexicographic tests unchanged.
2. **Enumeration oracles** on a 6–8 PU toy (brute force over 2ⁿ selections):
   - hierarchical with 2 objectives (MinCost → MinShortfall and MinShortfall → MinCost):
     - rel_tol 0: the result equals the enumerated lexicographic optimum value pair.
     - rel_tol r > 0: assert stage 2's value equals the enumerated `min f2 s.t.
       f1 ≤ v + r|v| + abs_tol`, and assert the cap inequality holds. Do not assert the pair,
       because f1 can tie within the cap.
   - **3 objectives with distinct tolerances:** `rel_tol = [0, r]` and `[r, 0]` (r > 0) on a
     fixture where the two give different enumerated results. This pins indexing by solve
     order.
   - **Mixed-sense hierarchy:** stage 1 maximised (MaxWtdSum) with r > 0, then stage 2
     minimised (MinCost), against enumeration. This pins the `≥` cap and
     `−(r|v| + abs_tol)`.
   - **Locks and no-good cuts:** the hierarchical oracle is repeated with one locked-in PU, one
     locked-out PU and one forbidden selection. Every stage must respect them.
   - **Mixed-sense weighted sum:** MinCost + MaxWtdSum rows against enumeration. This catches
     a sign `s` taken from the wrong objective, which the (min, min) oracle and the
     (max, max) parity cases cannot.
   - weighted-sum sweep: enumerate the Pareto-supported (cost, shortfall) hull, whose vertices
     run from the lexicographic (min cost, then min shortfall) point to the lexicographic
     (min shortfall, then min cost) point. Derive one weight
     row strictly inside each vertex's normal cone (midpoint of the adjacent edge normals; for
     the two endpoint vertices, a normal strictly between the axis and the single adjacent
     edge's normal — the pure axis rows tie across all minimum-cost / minimum-shortfall
     selections), and assert each row returns exactly that vertex's values; for any other rows (e.g. a
     `weights_matrix` grid) assert only the optimal weighted value, since edge-normal rows tie;
   - **Hierarchical sweep:** two different `rel_tol` rows in one call. Each row matches its
     own oracle, and the second row is not affected by the first row's caps.
   - **Cap-row removal after a failed stage:** a mocked failure in stage 2 of row 1, then
     row 2. Row 2 must equal a fresh solve, and no `hier_cap_*` row may remain in the model.
   - `nondominated` over the sweep equals the enumerated Pareto points that the sweep found.
   - `nondominated` unit tests on hand-built Solutions:
     - tied value vectors (the first is kept);
     - a weakly dominated point (dropped);
     - mixed min / max senses;
     - empty input;
     - mismatched key sets (`ValueError`).
3. **prioritizr parity** on its 6-PU fixture (cost 1 each; ft1 = (5,.5,0,0,0,0),
   ft2 = (0,0,0,0,3,2); COSTBUDGET 2; prioritizr's two max_wtd_sum problems (ft1 only, ft2
   only) map to `MaxWtdSum(weights={1: 1, 2: 0}, label="ft1")` and
   `MaxWtdSum(weights={1: 0, 2: 1}, label="ft2")` on one problem holding both features; spec
   review enumerated every row and found a unique optimal x for each):
   w = (1,1) → x = 100010, values (5, 3); rows [(1,0),(0,1),(1,1),(1,5)] →
   x = 110000, 000011, 100010, 000011, values (5.5,0), (0,5), (5,3), (0,5). Matrix helpers
   pinned as in §4.3. (Hierarchical has no literal prioritizr cases; the oracle covers it.)
   The plan re-derives these expected values by enumeration on the fixture before pinning
   them, so a mapping mistake shows up as an oracle mismatch rather than a silent pass.
4. **Bug-fix regressions:** v < 0 (negative MaxWtdSum weights) with rel_tol > 0 is feasible
   (prioritizr's cap would be infeasible); v = 0 allows exactly `abs_tol`, pinned with a large
   `abs_tol` (e.g. 0.5) on a fixture where that slack changes the stage-2 optimum;
   a maximised objective with tolerance; all-zero weights row; recomputed `objective_values`
   differ from raw auxiliary values in a constructed case; two `MinLargestShortfall` specs with
   different labels build without name collisions; solving the same sweep row twice gives the
   same result (row independence); **tolerance-boundary cap** (the round-4 case): amounts 0.7
   and 0.1, target 0.8, unit costs, `Hierarchical(rel_tol=0)` over `[MaxFeatures(),
   MinCost()]` returns cost 2 with the feature's model target met, not the empty plan;
   a PU with zero cost and no amounts (`varValue` stays `None`) does not break v or the
   recomputation; **incumbent cap** (round-5 case, deterministic): a stubbed stage-1 solve
   returns `sol_status` 2 with x meeting both targets but one met binary y = 0
   (`v_model` = 1, `v_rec` = 2). The `hier_cap_1` row must use v = 2, and stage 2 must keep
   both targets met. This test together with the tolerance-boundary test kills both "model
   value only" and "recompute only"; `stage_reached` is n on success and k − 1 when stage k fails (mocked
   `sol_status`), and the stage-1 failure raises `RuntimeError`.
5. **Budget / targets / boundary:** COSTBUDGET honoured when set, absent otherwise; a target
   unmet at the optimum (soft targets); `MinBoundary` is non-trivial when BLM = 0.
6. **Mutation spot-checks:** flip the cap inequality, drop `|v|`, flip the weighted-sum sign, take v from only the model value or only the recomputed value, index `rel_tol` by objective instead of stage —
   each must fail a test.

## 8. Docs

- `docs/TUTORIAL.md` section "Multi-objective planning (v0.41)": weighted-sum sweep of
  MinCost vs MinShortfall with scaled weights → `nondominated` → `Hierarchical` pick (cheapest
  plan within 5% of the best shortfall). The example sets a `COSTBUDGET` that forces a non-zero
  best shortfall. Without one the best shortfall is 0, and `rel_tol` gives no slack at v = 0.
  The text mentions `abs_tol` for that case. Pinned by `tests/test_tutorial_examples.py`.
- CHANGELOG `[Unreleased]`: Added (MultiObjectiveMIP, specs, approaches, helpers,
  `nondominated`, `MinCost`/`MinBoundary`/`MaxWtdSum` builders); Changed (`MIPSolver` internals
  refactored onto builders; solutions unchanged except the next line; LP variable /
  constraint names in exported models changed, §2); Fixed (absolute
  `min_largest_shortfall` now counts ghost features, §2); deviations from prioritizr (abs tolerance and
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
- HiGHS is not installed locally (only CBC); CI covers whatever backend `auto` resolves to
  there. Tests use `mip_backend="cbc"` explicitly where results are pinned.
- The presolve check (P10) inspects objective coefficients; normalised weights w/Σw can make
  small coefficients trip it once per row — `presolve_check` runs once per `solve` on the base
  model with the first row's objective only.
- Time: `MIP_TIME_LIMIT` per stage multiplies across stages and rows (§4.1).
