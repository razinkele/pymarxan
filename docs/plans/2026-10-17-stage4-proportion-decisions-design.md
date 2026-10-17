# Stage 4 (v0.42.0): proportion and semicontinuous decisions — design

**Date:** 2026-10-17 · **Baseline:** v0.41.1 (`e7b90fe`) · **Roadmap:**
`docs/plans/2026-10-11-prioritizr-parity-roadmap.md` Stage 4 · **Reference:** prioritizr 9.0.1
source (commit `f2f5628`): `R/add_proportion_decisions.R`, `R/add_semicontinuous_decisions.R`,
`src/rcpp_apply_decisions.cpp`, `src/rcpp_apply_boundary_penalties.cpp`,
`src/rcpp_apply_connectivity_penalties.cpp`, `src/rcpp_apply_asym_connectivity_penalties.cpp`,
`src/rcpp_apply_locked_constraints.cpp`, `src/rcpp_boundary.cpp`,
`R/eval_target_coverage_summary.R`, `R/eval_n_summary.R`, read 2026-10-17. The findings that
matter are restated here.

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

- `upper` and `lower` are finite scalar floats (bools rejected). Per-PU bounds
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
- `MultiObjectiveMIP`'s `evaluate`/`evaluate_implied` receive the extracted (snapped) values
  instead of `rounded()` for non-binary modes.

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
| `compute_feature_shortfalls` | today's `max(0, T − held)`, set to 0 when the feature is met by the rule above (so `met ⇔ shortfall == 0` holds exactly) |

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
binary-only post-hoc evaluators.

## 6. Readers of `Solution.selected`

### 6.1 Fraction-aware

| Reader | Fractional behaviour |
|---|---|
| `io/output.py` `write_run`, `write_best`, `write_solution_matrix` | Solution column written with `format(v, ".10g")`; binary unchanged (ints) |
| `io/output.py` `write_summary`, `io/writers.py` `write_sum` | `n_selected` (count, as today); no format change |
| `io/writers.py` `write_mvbest` | held from `_compute_achieved` (§5.3), `Target_Met` from `solution.targets_met` — no local `>=` recomputation (today it builds a truthiness `sel_set` and recomputes `held >= T·MISSLEVEL` exactly) |
| `io/writers.py` `write_ssoln` | `Number` = Σ x over solutions (float when any solution is fractional; ints otherwise, as today) |
| `analysis/selection_freq.py`, `io/spatial_export.py` `export_frequency_spatial` | sums as float; frequency = mean of x |
| `io/readers.py` `read_ssoln` | `Number` kept int when every value is integral, float otherwise |
| `io/exporters.py` `export_solution_csv`; `io/spatial_export.py` `export_solution_spatial` | float column (`.10g`) when fractional |
| `io/exporters.py` `export_summary_csv` | held from `_compute_achieved`, met from `solution.targets_met` |
| `analysis/representation.py` `compute_representation` | represented amount `Σ r·x` |
| `analysis/equity.py` `compute_equity` | group value `Σ x·value` |
| `analysis/portfolio.py` `selection_frequency`, `summary_statistics` | already float / count |
| `pymarxan_shiny` `results/summary_table.py` | held from `_compute_achieved`, met from `solution.targets_met` (today: truthiness + exact `>=` without MISSLEVEL) |
| `pymarxan_shiny` `mapping/solution_map.py` | colour interpolated between unselected and selected by `x`; listing shows the fraction |
| Shiny solver config + `pymarxan_app/app.py` | MIP panel gets `decisions` select and `upper` / `lower` inputs passed to `MIPSolver` |

### 6.2 Guarded with `require_binary`

SA / iterative improvement `improve()` and `run_mode` chains; `analysis/portfolio_cuts.py`;
`analysis/portfolio.py` `solution_diversity`; `analysis/scenarios.py` `overlap_matrix`;
`analysis/posthoc_clusters.py`; `analysis/replacement_cost.py`; `analysis/rank_importance.py`;
`analysis/robustness.py` `robustness_matrix`; `phylo/diversity.py`
`compute_phylogenetic_diversity` (Faith PD is presence-based — "a tip is represented if its
feature occurs in a selected PU"; a fractional analogue is a research choice, so it is guarded,
which narrows the brainstorm's list); `pymarxan_shiny` `mapping/comparison_map.py` (four-colour
categorical); `zones/writers.py` fallbacks.

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
and asserts every containing module appears in exactly one of three module-level tables in the
test — `FRACTION_AWARE`, `GUARDED`, `INTERNAL_OR_COUNT` — so a new reader cannot slip in
unclassified. Classification is **per module**; modules that mix categories
(`analysis/portfolio.py`, `analysis/scenarios.py`, `io/writers.py`) are listed as `MIXED` with
the guarded function names, and the per-function guard tests (§8) pin the split.

### 6.5 Binary behaviour change (one, intended)

`results/summary_table.py` today recomputes met as `held >= target` **without MISSLEVEL**; taking
met from `solution.targets_met` makes it honour MISSLEVEL for binary solutions too. This is a
display fix, recorded under CHANGELOG *Fixed*. `write_mvbest` and `export_summary_csv` already
apply MISSLEVEL exactly, so their binary output is unchanged.

## 7. Errors (summary)

| Condition | Error |
|---|---|
| bad `decisions` / `upper` / `lower` | `ValueError` in `__init__` (both solvers) |
| non-binary + PROBMODE 3 / target2 / separation / space / forbidden_selections / negative connectivity | `ValueError` at the top of `_build_base` (all entry points) |
| fractional array into `build_solution` with PROBMODE 3 / target2 / separation / space | `ValueError` |
| binary-only reader given a fractional solution | `FractionalSolutionError` |
| `binarize_solution` threshold outside `(0, 1]` | `ValueError` |
| infeasible model (e.g. semicontinuous floor vs budget) | `[]` from `MIPSolver.solve`, as today |

## 8. Testing

- **Binary unchanged:** `build_model` MPS for binary byte-identical to v0.41.1 on
  `tests/data/simple` and a connectivity fixture; binary `presolve_check` messages unchanged;
  every v0.37–v0.41 golden and the 35.0 anchor stay green untouched.
- **LP bound:** `decisions="proportion"` objective ≤ binary objective on `tiny_problem` and on
  ≥ 20 seeded random problems (boundary + symmetric + asymmetric connectivity on).
- **Integral LP:** a named fixture with one feature per PU (identity `rij`, targets = amounts)
  whose LP optimum is integral; proportion objective == binary objective and every `x ∈ {0, 1}`.
- **Semicontinuous floor:** every `x_i ∈ {0} ∪ [lower, upper]`; objective between the
  proportion and binary objectives.
- **Upper cap:** `proportion, upper=0.5` → every `x ≤ 0.5`; locked-in PUs at exactly 0.5.
- **Hand calculation (roadmap acceptance):** a 3-PU fixture with given fractions: held, cost,
  boundary (`|x_i − x_j|`), connectivity (`−v·min + v·|Δ|` per row),
  `compute_representation` and `compute_equity` checked against hand-written numbers.
- **Objective agreement:** on problems **without connectivity**, `Solution.objective` equals
  the MIP objective value within `1e-6·max(1, |obj|)` for proportion and semicontinuous solves.
- **Targets met at the LP optimum:** `all_targets_met` is True and `shortfall == 0` for the
  `min_set` proportion solve on `tiny_problem` and on 20 random problems at amount scales 1,
  1e4 and 1e8 (CBC locally; the same test runs under HiGHS where installed, i.e. CI).
- **Snapping:** `_extract_selected` unit tests — dust → 0, near-upper → upper, near-lower →
  lower, `s = 0` → 0, never lowers a value above `ZERO_TOL`.
- **Connectivity reward survives:** with symmetric connectivity, fractional `z > 0` occurs and
  the objective is below the same model with `z` forced binary.
- **Presolve:** `strict` with `upper=0.5` and `0.5·Σc_locked ≤ COSTBUDGET < Σc_locked` does
  not raise P9a; P6 fires when `T > upper·avail`.
- **Guards:** every §6.2 function raises `FractionalSolutionError`; every §3.5 combination
  raises from `MIPSolver.solve`, `MIPSolver.build_model` and `MultiObjectiveMIP.solve`.
- **Multi-objective:** a hierarchical `MinCost` + `MaxFeatures` run with
  `decisions="proportion"` returns float selections whose `objective_values` match
  recomputation (feature count not undercounted).
- **Readers:** completeness test (§6.4); `write_mvbest` `Target_Met` column equals
  `solution.targets_met` on a fractional solution; `write_ssoln` → `read_ssoln` float
  round-trip; Shiny summary table on a fractional solution.

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
