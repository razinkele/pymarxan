# Stage 4 (v0.42.0): proportion and semicontinuous decisions — design

**Date:** 2026-10-17 · **Baseline:** v0.41.1 (`e7b90fe`) · **Roadmap:**
`docs/plans/2026-10-11-prioritizr-parity-roadmap.md` Stage 4 · **Reference:** prioritizr 9.0.1
source (commit `f2f5628`): `R/add_proportion_decisions.R`, `R/add_semicontinuous_decisions.R`,
`src/rcpp_apply_decisions.cpp`, `src/rcpp_apply_boundary_penalties.cpp`,
`src/rcpp_apply_connectivity_penalties.cpp`, `src/rcpp_apply_asym_connectivity_penalties.cpp`,
`src/rcpp_apply_locked_constraints.cpp`, `src/rcpp_boundary.cpp`,
`R/eval_target_coverage_summary.R`, `R/eval_n_summary.R`, read 2026-10-17. The findings that
matter are restated here.

> **As-built amendments (4-lens design review, 2026-10-17; synthesis in
> `docs/plans/2026-10-17-stage4-review.md`):**
> - **§5.3 tolerance is purely relative:** `tol(R) = 1e-7·|R|` (`TARGET_ABS_TOL` removed). The
>   `1e-6` absolute floor made an empty fractional plan "meet" every target below ~1e-6. A
>   relative-only rule passed 600 of 600 LP rows at amount scales 1e-6 to 1e8 under CBC.
> - **§3.2 locked-in = 1.0 (user decision):** locked-in PUs are fixed at `x = 1.0` in every mode.
>   `upper` and the semicontinuous band apply only to free PUs, because status 2 usually encodes
>   existing, fully protected reserves. Presolve follows: P9a's floor is the unscaled locked cost,
>   and P6 / P11 count locked units in full.
> - **§4 snapping:** there is no upward snap to `upper`. A value just below the bound is a genuine
>   simplex value, and snapping it up can breach COSTBUDGET. Extraction keeps the clip to each PU's
>   own bound, dust removal (≤ 1e-9 becomes 0), the semicontinuous snap to `lower`, and `s = 0` → 0.
> - **Input contract:** `build_solution` / `binarize_solution` validate `selected`. The length
>   must match; floats must be finite and in [0, 1]; int 0/1 arrays become bool (they were
>   fancy-indexed before, a pre-existing bug); other ints raise.
> - **Layout:** decision validation and extraction live in the leaf module
>   `solvers/_decisions.py`, and `presolve_check` validates its bound arguments. The helpers
>   `compute_held` and `require_binary_solver` are public.

## 1. Intent

A planning unit can be **partly** selected: partial protection, restoration shares, land-use
fractions. Every number downstream of the solver (cost, held amounts, targets met, shortfall,
boundary, connectivity, objective, written files, maps) is consistent with the fractions.
Binary results do not change. Tools that only make sense for 0/1 plans refuse fractional
plans with a clear error; **nothing rounds silently**.

**User decisions (brainstorm, 2026-10-17):**

| Question | Decision |
|---|---|
| Semicontinuous semantics | **Both**: `decisions="proportion", upper=u` reproduces prioritizr's `add_semicontinuous_decisions(u)`; `decisions="semicontinuous"` is the true floor `x = 0 or lower ≤ x ≤ upper` |
| Downstream scope | **Core + key readers** fraction-aware; every other `Solution.selected` reader guarded (§6) |
| Boundary/connectivity reported for fractional plans | **Match the optimiser's linearisation** (`|x_i − x_j|`, `min`), not prioritizr's `eval_*` product form. *Narrowed by spec review R1 (§5.3):* exact for boundary; for connectivity the existing **binary** evaluator already differs from the MIP (pre-existing, kept so binary numbers don't change), so connectivity uses the fractional generalisation of the existing evaluator |

**Research findings that shape the design (prioritizr 9.0.1):**
- `add_proportion_decisions` = vtype `"C"`, bounds `[0, 1]`; `add_binary_decisions` = `"B"`,
  `[0, 1]` (`rcpp_apply_decisions(x$ptr, vtype, 0, ub)`).
- `add_semicontinuous_decisions(upper_limit)` is **not** semicontinuous: vtype `"C"`, bounds
  `[0, upper_limit]`, scalar only, asserted `0 ≤ upper_limit ≤ 1`. No lower bound exists. The
  data are not rescaled.
- Boundary and connectivity auxiliaries are `"C"` in `[0, 1]` for every decision type. With a
  positive penalty in a min problem only `b ≤ x_i`, `b ≤ x_j` are written, so the shared-edge
  credit is `min(x_i, x_j)`. prioritizr's boundary objective is
  `Σ x_i·total_i − 2·Σ_{i>j} B_ij·b_ij`; with Marxan's boundary file
  (`total_i = b_ii + Σ_j b_ij`) that is algebraically `Σ b_ii·x_i + Σ b_ij·|x_i − x_j|`
  (since `x_i + x_j − 2·min(x_i, x_j) = |x_i − x_j|`), which is exactly pymarxan's existing MIP
  boundary term.
- prioritizr's evaluators disagree with its optimiser on fractions: `rcpp_boundary` and
  `rcpp_connectivity` use the product `x_i·x_j`. We follow the optimiser (user decision).
- Locked-in sets only `lb = 1`; with `upper_limit < 1` the model is infeasible (reported by the
  presolve check as "both locked in and locked out"). There are no decision-type compatibility
  checks anywhere: cuts portfolios, contiguity and gap portfolios compile with proportion
  decisions though they are meaningless.
- `eval_feature_representation_summary`: held = `rij %*% x`; `eval_n_summary` = `sum(x)`
  (documented as possibly non-integral); `eval_target_coverage_summary`: met iff shortfall
  `< 1e-10`.

**Grounding probe (2026-10-17, `tests/data/simple`, BLM 1):** relaxing `x` to `[0, 1]` gives
LP objective 35.583 (`x = {1: .274, 2: 1, 3: 0, 4: .117, 5: .106, 6: 1}`) vs the binary 43.0;
every target row is active at `held − T ≈ +1e-7` (CBC's printed precision). An exact `>=`
evaluator would flip on solver noise → §5.3 tolerance.

## 2. Public API

```python
MIPSolver(..., decisions="binary", upper=1.0, lower=0.0)
MultiObjectiveMIP(objectives, approach, ..., decisions="binary", upper=1.0, lower=0.0)
```

| `decisions` | Variable `x_i` | Extra | Valid `upper`, `lower` |
|---|---|---|---|
| `"binary"` (default) | Binary | — | `upper == 1.0`, `lower == 0.0` |
| `"proportion"` | Continuous `[0, upper]` | — | `0 < upper ≤ 1`, `lower == 0.0` |
| `"semicontinuous"` | Continuous `[0, upper]` | Binary `s_i`; rows `x_i − upper·s_i ≤ 0`, `x_i − lower·s_i ≥ 0` | `0 < lower ≤ upper ≤ 1` |

- `upper` and `lower` are finite scalar floats (bools rejected). Both solvers store them as
  public attributes `decisions`, `upper`, `lower` (the §6.2 solver guards read
  `solver.decisions`). Per-PU bounds
  (prioritizr's `add_manual_bounded_constraints`) are backlog.
- Validation happens in `__init__` (both `MIPSolver` and `MultiObjectiveMIP`, the latter by
  passing the arguments to its inner `MIPSolver`) with `ValueError`s naming the offending argument
  (`decisions must be one of 'binary', 'proportion', 'semicontinuous', got 'x'`;
  `lower only applies to decisions='semicontinuous'`; `upper must be in (0, 1], got 1.5`; …).
- `ZoneMIPSolver` is out of scope (zone allocations are categorical; fractional zone shares are
  a separate design).
- `"semicontinuous"` is still a MILP: it is not faster than binary. `"proportion"` is a pure LP
  whenever the objective adds no binaries (`min_set`, `min_shortfall`,
  `min_largest_shortfall`, `min_penalties`); `max_features` / `max_weighted_features` keep their
  binary met-indicators `y` (as prioritizr's `max_n_targets_met` does).

## 3. Model changes (`MIPSolver._build_base`)

The binary model is **byte-identical** to v0.41.1 (MPS from `build_model` unchanged; pinned by a
test). Non-binary changes:

1. **Decision variables.** `x_i = LpVariable(f"x_{pid}", lowBound=0, upBound=upper,
   cat="Continuous")`. Semicontinuous adds `s_i = LpVariable(f"s_{pid}", cat="Binary")` and rows
   `sc_ub_{pid}: x_i − upper·s_i ≤ 0`, `sc_lb_{pid}: x_i − lower·s_i ≥ 0`.
2. **Locks.** Locked-out (status 3): `x_i == 0` (as today). Locked-in (status 2): `x_i == upper`
   (and `s_i == 1` via the rows). *Documented deviation:* prioritizr's lock-in gives an infeasible
   model when `upper < 1`; "locked in" here means "selected as fully as allowed". Status 1
   (initial include) stays ignored by the MIP.
3. **Symmetric connectivity `z`** (`zc_i_j`, today `cat="Binary"`) becomes Continuous `[0, 1]`
   in non-binary modes. A binary `z` with fractional `x` is forced to 0 and the reward silently
   vanishes. The three AND rows stay; with the reward in a min problem `z = min(x_i, x_j)`.
4. **Boundary `y` and asymmetric `z`** are already continuous: no change.
5. **Rejected combinations** (`ValueError` at the **top of `_build_base`**, before the existing
   `NotImplementedError` strategy checks and before any variable is created — so it covers
   `MIPSolver.solve`, `MIPSolver.build_model` **and** `MultiObjectiveMIP.solve`, which calls
   `_build_base` directly), each message naming the cause and suggesting `decisions="binary"`:
   PROBMODE 3; any `target2 > 0` (clumping); any separation-active feature
   (`sepdistance > 0 and sepnum > 1`); any `space_target > 0`; a non-empty
   `config.metadata["forbidden_selections"]` (no-good cuts are only valid for 0/1 variables);
   any negative connectivity `value` when `CONNECTIVITY_WEIGHT > 0` (a negative symmetric value
   flips the sign of the `z` coefficient, so the LP drives `z` to `max(0, x_i + x_j − 1)`, not
   `min`, and the implied rule would be wrong).
   PROBMODE 1 (linear risk premium) and PROBMODE 2 (probability-scaled amounts) are linear and
   allowed.
6. **Implied rules** (`evaluate_implied`, used by hierarchical pass checks) change for all modes:
   `_and_rule → min(x_i, x_j)`, `_src_without_sink_rule → max(0, x_i − x_j)`, `_abs_rule`
   unchanged. These equal today's products on 0/1 values, so binary results are unchanged.
   Semicontinuous registers `s_i`'s implied rule `1.0 if x_i > 0 else 0.0`.
   In non-binary modes `_met_rule` and `_slack_rule` apply the §5.3 tolerance (met iff
   `A ≥ R − tol(R)`; slack `max(0, R − A)` zeroed when `≤ tol(R)`): every active row of an LP
   optimum sits at `R − O(1e-7)`, so exact rules would systematically undercount
   `max_features` / `max_weighted_features` and leave `1e-7` shortfalls in `objective_values`.
   `evaluate_implied` keeps its absolute `valid(1e-6)` row check: with the tolerant rules the
   `~1e-7·R` residual on `A ≥ R·y` / `s ≥ R − A` marks the implied point infeasible once `R` is
   above roughly 10–200, so `evaluate()` returns `None` and the hierarchical cap uses the model
   value. That fallback is by design and expected in fractional mode; **do not** widen `valid()`
   (it would touch binary). Final `objective_values` ignore feasibility.
   The tolerance is chosen when the context is built (`ObjectiveContext` gains a
   `fractional: bool` field, default False); binary rules are unchanged.
7. **Presolve data checks scale with the bounds.** `presolve_check(problem, *,
   decisions="binary", upper=1.0, lower=0.0)` and `_presolve_data(flag, problem, solver_name,
   *, decisions="binary", upper=1.0, lower=0.0)` gain keyword-only bounds (defaults keep every
   binary message byte-identical); `MIPSolver` / `MultiObjectiveMIP` pass their own. In
   `_core_checks`: P6 compares `T·MISSLEVEL` with `upper·avail`; P9a's floor is
   `upper·Σ c_locked_in`; P9b's cheapest option is `lower·min c_free` for semicontinuous and is
   skipped for proportion (any positive budget buys a sliver); P11's ceiling is
   `upper·Σ c_not_locked_out`. Without this, `presolve_check="strict"` raises P9a on a feasible
   model whenever `upper·Σc_locked ≤ COSTBUDGET < Σc_locked`. `_zone_checks` is untouched.
8. **Model carries `s`.** `_BaseModel` and `_BuiltModel` gain `s: dict[int, LpVariable]`
   (empty unless semicontinuous) and `decisions`, `upper`, `lower` so extraction (§4) can read them.

## 4. Extraction

`MIPSolver.solve` and `MultiObjectiveMIP.solve` (today `bool(round(v))` / `rounded()`) build
`selected` through one helper,
`_extract_selected(x_values, s_values, decisions, upper, lower) -> np.ndarray` (`x_values`,
`s_values`: PU-ordered lists of `varValue`s, `s_values` empty unless semicontinuous):

- `binary`: unchanged (`bool(round(v))`, dtype bool).
- `proportion` / `semicontinuous`: dtype `float64`; `None` → 0.0; clip to `[0, upper]`.
  Snapping only ever **raises** a value or removes solver dust, so it can never take held
  amount away from a target (spec review R1: snapping toward 0 removes `r·Δx`, which can exceed
  the target tolerance at large magnitudes):
  - `x ≤ ZERO_TOL = 1e-9` → 0.0;
  - `upper − SNAP_TOL ≤ x < upper` → `upper` (`SNAP_TOL = 1e-6`);
  - semicontinuous, `lower − SNAP_TOL ≤ x < lower` → `lower`;
  - semicontinuous, `round(s_i) == 0` → 0.0 (the rows force `x ≤ upper·s ≈ 0`).
- `MultiObjectiveMIP`'s `evaluate`/`evaluate_implied` receive the extracted (snapped) values,
  as `dict(zip(base.pu_ids, arr))`, instead of `rounded()` for non-binary modes.
- Known limit of `ZERO_TOL`: a genuine optimum `x* ≤ 1e-9` (one PU holding ≥ 1e9× a target) is
  snapped to 0 and that target is missed; documented, not handled.

## 5. Solution model and evaluators

### 5.1 `Solution`

- `selected` is `bool` for binary runs, `float64` in `[0, 1]` otherwise. **The dtype is the
  discriminator** (`selected.dtype.kind == "f"`), not the values: an LP that happens to be
  integral still yields a fractional-typed solution, which keeps behaviour predictable.
- New property `is_fractional -> bool`.
- `n_selected` stays `int`: the count of PUs with `x > 0`. New property
  `selected_sum -> float` = `Σ x` (prioritizr's `eval_n_summary`); equals `n_selected` for
  binary.
- The `selected` comment becomes "bool array, or float64 proportions in [0, 1] when solved with
  non-binary decisions".

### 5.2 Guard and explicit conversion

- `FractionalSolutionError(ValueError)` and `require_binary(solution, context: str) -> None` in
  `pymarxan.solvers.base`. Message: `"{context} needs a binary solution, but this solution holds
  proportion decisions. Convert explicitly with binarize_solution(problem, solution,
  threshold=...)."`
- `binarize_solution(problem, solution, threshold=0.5) -> Solution` in `pymarxan.solvers.utils`:
  `x ≥ threshold` → selected; rebuilt through `build_solution(problem, mask,
  blm=float(problem.parameters.get("BLM", 0.0)), metadata={**solution.metadata,
  "binarized_threshold": threshold})` (`space_spec` defaults, as in `MIPSolver`).
  `0 < threshold ≤ 1`, else `ValueError`. Binary input is returned unchanged (same object).
  This is the only rounding path.

### 5.3 Fraction-linear evaluators (`pymarxan.solvers.utils`)

Every formula reduces to today's on 0/1 input; the binary code path keeps its exact
comparisons, so every existing number is unchanged.

| Helper | Fractional form |
|---|---|
| `_compute_achieved` | `held_f = Σ_i r_if·x_i` — raw amounts, as today (the binary evaluator already ignores PROBMODE 2's `(1 − p)` weights that `_core_amount_exprs` and the heuristic apply; that gap is pre-existing and is **not** closed on the fractional path only, or the two paths would diverge) |
| cost (`build_solution`, `compute_objective_terms`) | `Σ c_i·x_i` |
| `compute_boundary` | `Σ b_ii·x_i + Σ_{i≠j rows} b_ij·|x_i − x_j|` (row-wise, as today) — identical to the MIP's boundary term |
| `compute_connectivity_penalty` | per row (no de-duplication, as today): `−v·min(x_i, x_j) + v·|x_i − x_j|`; on 0/1 this is today's `−v` (both) / `+v` (exactly one) |
| PROBMODE 1 risk premium | `Σ w·p_i·c_i·x_i` |
| `check_targets` | met iff `held ≥ R − tol(R)`, `R = T·MISSLEVEL` |
| `compute_feature_shortfalls` | today's `max(0, R − held)`, set to 0 when the feature is met by the rule above (so `met ⇔ shortfall == 0` holds exactly) |

**Connectivity divergence (pre-existing, documented, not fixed here).** The binary
`compute_connectivity_penalty` already differs from the MIP's connectivity term: the MIP's
symmetric term is reward-only (`−w·c·z`, pairs de-duplicated) and its asymmetric term charges
`x_i ∧ ¬x_j`, while the evaluator rewards both-selected, charges exactly-one, never reads
`ASYMMETRIC_CONNECTIVITY` and counts `(i, j)` and `(j, i)` rows separately (review R1 probe on
`tests/data/simple`: symmetric `Solution.objective` 40.0 vs MIP 37.0, asymmetric 40.0 vs 43.0, binary
selections). Fixing it would change binary numbers, which this stage promises not to do; it goes
to the roadmap backlog. Consequently `Solution.objective == mip_objective_value` holds only for
problems without connectivity, in both modes.

`tol(v) = TARGET_ABS_TOL + TARGET_REL_TOL·|v|` with `TARGET_ABS_TOL = 1e-6`,
`TARGET_REL_TOL = 1e-7` (the Stage 3 lesson: an absolute tolerance alone fails at large
magnitudes). Review R1 probe: 20 random LPs at amount scales 1, 1e4, 1e8 under CBC gave worst
`held − T ≈ −4.9e-9·|T|`, so `1e-9` failed 24 of 160 rows and `1e-7` leaves a 20× margin.
Applied only when `selected.dtype.kind == "f"` (the helpers take arrays, not `Solution`s).
HiGHS is not installed locally; its check runs in CI (§8).

`build_solution` with fractional input raises `ValueError` for the same combinations as §3.5
(PROBMODE 3, clumping, separation, space), so a hand-built fractional array cannot reach the
binary-only post-hoc evaluators. The PROBMODE 3 check sits in `compute_probability_penalty`
(reached by both `compute_objective_terms` and `build_solution`), because
`_compute_probmode3_zscore_penalty` selects PUs by truthiness and `compute_objective` callers
such as `evaluate_plans_across_scenarios` judge plans under arbitrary scenario problems (review
R3); clumping, separation and space are only evaluated inside `build_solution`.

## 6. Readers of `Solution.selected`

### 6.1 Fraction-aware

| Reader | Fractional behaviour |
|---|---|
| `io/output.py` `write_run`, `write_best`, `write_solution_matrix` | Solution column written with `format(v, ".10g")`; binary unchanged (ints) |
| `io/output.py` `write_summary`, `io/writers.py` `write_sum` | `n_selected` (count, as today); no format change |
| `io/writers.py` `write_mvbest` | held from `_compute_achieved` (§5.3), `Target_Met` from `check_targets(problem, solution.selected, pu_index, _achieved=held)` — the shared rule (exact for binary, tolerant for fractional) recomputed against the current problem (today it builds a truthiness `sel_set` and recomputes `held >= T·MISSLEVEL` exactly). Not `solution.targets_met`: an existing test (`test_write_mvbest_respects_misslevel`) changes MISSLEVEL after the solve and expects the file to follow it |
| `io/writers.py` `write_ssoln` | `Number` = Σ x over solutions (float when any solution is fractional; ints otherwise, as today) |
| `analysis/selection_freq.py`, `io/spatial_export.py` `export_frequency_spatial` | sums as float; frequency = mean of x |
| `io/readers.py` `read_ssoln` | `Number` kept int when every value is integral, float otherwise |
| `io/exporters.py` `export_solution_csv`; `io/spatial_export.py` `export_solution_spatial` | float column (`.10g`) when fractional |
| `io/exporters.py` `export_summary_csv` | held from `_compute_achieved`, met from `check_targets(..., _achieved=held)` |
| `analysis/representation.py` `compute_representation` | represented amount `Σ r·x` |
| `analysis/equity.py` `compute_equity` | group value `Σ x·value` |
| `analysis/robustness.py` `evaluate_plans_across_scenarios` | drop the `np.asarray(..., dtype=bool)` cast (line 134, a silent rounding today); it only calls `compute_objective`, which is fraction-aware |
| `analysis/portfolio.py` `selection_frequency`, `summary_statistics` | already float / count |
| `pymarxan_shiny` `results/summary_table.py` | logic pulled out of the `@render.ui` closure into a pure helper `_target_rows(problem, solution) -> list[dict]` (testable); held from `_compute_achieved`, met from `check_targets(..., _achieved=held)` (today: truthiness + exact `>=` without MISSLEVEL) |
| `pymarxan_shiny` `mapping/solution_map.py` | colour interpolated between unselected and selected by `x`; listing shows the fraction |
| Shiny solver config + `pymarxan_app/app.py` | MIP panel gets `decisions` select and `upper` / `lower` inputs passed to `MIPSolver` |

### 6.2 Guarded with `require_binary`

SA / iterative improvement `improve()` and `run_mode` chains; `analysis/portfolio_cuts.py`;
`analysis/portfolio.py` `solution_diversity`; `analysis/scenarios.py` `overlap_matrix`;
`analysis/posthoc_clusters.py`; `analysis/replacement_cost.py`; `analysis/rank_importance.py`;
`phylo/diversity.py`
`compute_phylogenetic_diversity` (Faith PD is presence-based — "a tip is represented if its
feature occurs in a selected PU"; a fractional analogue is a research choice, so it is guarded,
which narrows the brainstorm's list); `pymarxan_shiny` `mapping/comparison_map.py` (four-colour
categorical); `zones/writers.py` fallbacks.

Guard placement for the two that take a **solver**, not a `Solution`:
`compute_replacement_cost` and `generate_portfolio_cuts` raise `FractionalSolutionError` up front
when `getattr(solver, "decisions", "binary") != "binary"` (otherwise `portfolio_cuts` would die on
iteration 2 with the §3.5 `forbidden_selections` `ValueError`).

### 6.3 Internal state and count-only readers (no change)

- `solvers/clumping.py`, `solvers/separation.py`, `solvers/space_state.py`: their own
  `self.selected` state for SA/II (reached only through guarded solvers / rejected
  combinations).
- `solvers/base.py`: defines `Solution`.
- `calibration/sensitivity.py`, `calibration/sweep.py`, `pymarxan_shiny` `results/export.py`,
  `spatial_export/spatial_export.py`, `run_control/run_panel.py`: read `n_selected` only (a
  count, still correct).

### 6.4 Completeness test

`tests/pymarxan/test_fractional_readers.py` greps `src/` for `\.selected\b` and `n_selected`
and asserts every containing module appears in exactly one of four module-level tables in the
test — `FRACTION_AWARE`, `GUARDED` (including `solvers/iterative_improvement.py`),
`INTERNAL_OR_COUNT`, `MIXED` — so a new reader cannot slip in unclassified. Classification is **per module**; modules that mix categories
(`analysis/portfolio.py`, `analysis/scenarios.py`, `io/writers.py`) are listed as `MIXED` with
the guarded function names, and the per-function guard tests (§8) pin the split.

### 6.5 Binary behaviour change (one, intended)

`results/summary_table.py` today recomputes met as `held >= target` **without MISSLEVEL**; taking
met from `check_targets` makes it honour MISSLEVEL for binary solutions too. This is a
display fix, recorded under CHANGELOG *Fixed*. `write_mvbest` and `export_summary_csv` already
apply MISSLEVEL exactly, so their binary output is unchanged.

## 7. Errors (summary)

| Condition | Error |
|---|---|
| bad `decisions` / `upper` / `lower` | `ValueError` in `__init__` (both solvers) |
| non-binary + PROBMODE 3 / target2 / separation / space / forbidden_selections / negative connectivity | `ValueError` at the top of `_build_base` (all entry points) |
| fractional array into `build_solution` with PROBMODE 3 / target2 / separation / space, or into `compute_objective_terms` with PROBMODE 3 | `ValueError` |
| non-binary `MIPSolver` passed to `compute_replacement_cost` / `generate_portfolio_cuts` | `FractionalSolutionError` up front |
| binary-only reader given a fractional solution | `FractionalSolutionError` |
| `binarize_solution` threshold outside `(0, 1]` | `ValueError` |
| infeasible model (e.g. semicontinuous floor vs budget) | `[]` from `MIPSolver.solve`, as today |

## 8. Testing

- **Binary unchanged:** `build_model` MPS for binary byte-identical to v0.41.1 on
  `tests/data/simple` and a connectivity fixture; binary `presolve_check` messages unchanged;
  every v0.37–v0.41 golden and the 35.0 anchor stay green untouched.
- **LP bound:** `decisions="proportion", upper=1.0` objective ≤ binary objective on `tiny_problem` and on
  ≥ 20 seeded random problems (boundary + symmetric + asymmetric connectivity on).
- **Integral LP:** a named fixture with one feature per PU (identity `rij`, targets = amounts)
  whose LP optimum is integral; proportion objective == binary objective and every `x ∈ {0, 1}`.
- **Semicontinuous floor:** every `x_i ∈ {0} ∪ [lower, upper]`; with `upper=1.0` the objective
  lies between the proportion and binary objectives (review R2 probe: 35.58 ≤ 37.08 / 39.9 /
  43.0 ≤ 43.0 for lower 0.3 / 0.6 / 0.9). With `upper < 1` neither bound holds (upper 0.5 gave
  45.5 > 43.0).
- **Upper cap:** `proportion, upper=0.5` → every `x ≤ 0.5`; locked-in PUs at exactly 0.5.
- **Hand calculation (roadmap acceptance):** a 3-PU fixture with given fractions: held, cost,
  boundary (`|x_i − x_j|`), connectivity (`−v·min + v·|Δ|` per row),
  `compute_representation` and `compute_equity` checked against hand-written numbers.
- **Objective agreement:** `objective="min_set"`, no connectivity, no COSTTHRESH (the MIP does
  not model it): `Solution.objective` equals `mip_objective_value` within
  `1e-6·max(1, |obj|) + SNAP_TOL·Σc` for proportion and semicontinuous solves (other objectives'
  `mip_objective_value` is a count or a shortfall, not `Solution.objective`).
- **Targets met at the LP optimum:** `all_targets_met` is True and `shortfall == 0` for the
  `min_set` proportion solve on `tiny_problem` and on 20 random problems at amount scales 1,
  1e4 and 1e8 (CBC locally; the same test runs under HiGHS where installed, i.e. CI).
- **Snapping:** `_extract_selected` unit tests — dust → 0, near-upper → upper, near-lower →
  lower, `s = 0` → 0, `x > upper` → upper; it never lowers a value in any other case.
- **Connectivity reward survives:** with symmetric connectivity, fractional `z > 0` occurs and
  the objective is below the same model with `z` forced binary.
- **Presolve:** `strict` with `upper=0.5` and `0.5·Σc_locked ≤ COSTBUDGET < Σc_locked` does
  not raise P9a; P6 fires when `T > upper·avail`.
- **Guards:** every §6.2 function raises `FractionalSolutionError` (including a proportion
  `MIPSolver` passed to `generate_portfolio_cuts` / `compute_replacement_cost`); a fractional
  plan under a PROBMODE 3 scenario in `evaluate_plans_across_scenarios` raises `ValueError`; every §3.5 combination
  raises from `MIPSolver.solve`, `MIPSolver.build_model` and `MultiObjectiveMIP.solve`.
- **Multi-objective:** a hierarchical `[MaxFeatures, MinCost]` run with
  `decisions="proportion"` and a COSTBUDGET at which the LP optimum has active target rows (the
  multi-objective path has no hard targets, so `MinCost` first would select nothing): float
  selections; the `MaxFeatures` value in `objective_values` is > 0, equals the number of `True`s
  the tolerant `check_targets` gives on the returned selection, and is not below the model's
  `Σ y`.
- **Readers:** completeness test (§6.4); `write_mvbest` `Target_Met` column equals
  `check_targets(...)` on a fractional solution, and `test_write_mvbest_respects_misslevel` stays
  green; `write_ssoln` → `read_ssoln` float
  round-trip; `_target_rows` on a fractional solution.

## 9. Docs

CHANGELOG `[Unreleased]`; `docs/` user guide section "Partial selection" (proportion vs
semicontinuous, the prioritizr mapping `add_semicontinuous_decisions(u)` ≙
`decisions="proportion", upper=u`, the lock-in and evaluator deviations, the pre-existing connectivity evaluator divergence, the
guarded list);
roadmap Stage 4 as-built note and a backlog line "align `compute_connectivity_penalty` with the
MIP connectivity term (changes binary numbers)"; `.github/copilot-instructions.md` solver matrix row.

## 10. Risks

- **Widest blast radius of the roadmap.** Mitigated by the dtype discriminator (binary code
  paths are untouched), the completeness test, and the design review's blast-radius lens.
- **Solver tolerance.** CBC leaves active rows at about `−5e-9·|T|` (review R1 probe); §5.3's
  tolerance has a 20× margin at CBC, and HiGHS is checked in CI only.
- **Connectivity evaluator vs MIP** (pre-existing divergence, §5.3) — documented and
  backlogged, not fixed in this stage.
- **Users reading `n_selected` as area.** Documented; `selected_sum` is the fraction total.
