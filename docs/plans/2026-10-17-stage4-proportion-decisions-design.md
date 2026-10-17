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
| Boundary/connectivity reported for fractional plans | **Match the optimiser** (`|x_i − x_j|`, `min`, `max(0, ·)`), not prioritizr's `eval_*` product form |

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
- Validation happens in `__init__` with `ValueError`s naming the offending argument
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
5. **Rejected combinations** (`ValueError` at the top of `solve`/`build_model`, before any model
   is built), each message naming the feature and suggesting `decisions="binary"`:
   PROBMODE 3; any `target2 > 0` (clumping); any separation-active feature
   (`sepdistance > 0 and sepnum > 1`); any `space_target > 0`; a non-empty
   `config.metadata["forbidden_selections"]` (no-good cuts are only valid for 0/1 variables).
   PROBMODE 1 (linear risk premium) and PROBMODE 2 (probability-scaled amounts) are linear and
   allowed.
6. **Implied rules** (`evaluate_implied`, used by hierarchical pass checks) change for all modes:
   `_and_rule → min(x_i, x_j)`, `_src_without_sink_rule → max(0, x_i − x_j)`, `_abs_rule`
   unchanged. These equal today's products on 0/1 values, so binary results are unchanged.
   Semicontinuous registers `s_i`'s implied rule `1.0 if x_i > 0 else 0.0`.

## 4. Extraction

`MIPSolver.solve` and `MultiObjectiveMIP.solve` (today `bool(round(v))` / `rounded()`) build
`selected` through one helper, `_extract_selected(values, decisions, upper, lower)`:

- `binary`: unchanged (`bool(round(v))`, dtype bool).
- `proportion` / `semicontinuous`: dtype `float64`; `None` → 0.0; clip to `[0, upper]`; snap
  any value within `SNAP_TOL = 1e-6` of `0`, `lower` (semicontinuous) or `upper` to that value.
  Semicontinuous: if the rounded `s_i` is 0 the value is 0.0.
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
  `x ≥ threshold` → selected; rebuilt through `build_solution` with the original metadata plus
  `binarized_threshold`. `0 < threshold ≤ 1`. Binary input is returned unchanged (same object).
  This is the only rounding path.

### 5.3 Fraction-linear evaluators (`pymarxan.solvers.utils`)

Every formula reduces to today's on 0/1 input; the binary code path keeps its exact
comparisons, so every existing number is unchanged.

| Helper | Fractional form |
|---|---|
| `_compute_achieved` | `held_f = Σ_i r_if·x_i` (weighted sum; PROBMODE 2 weights kept) |
| cost (`build_solution`, `compute_objective_terms`) | `Σ c_i·x_i` |
| `compute_boundary` | `Σ b_ii·x_i + Σ_{i≠j rows} b_ij·|x_i − x_j|` (row-wise, as today) |
| `compute_connectivity_penalty`, symmetric | credit `min(x_i, x_j)` where today `x_i ∧ x_j` |
| `compute_connectivity_penalty`, asymmetric | charge `max(0, x_i − x_j)` where today `x_i ∧ ¬x_j` |
| PROBMODE 1 risk premium | `Σ w·p_i·c_i·x_i` |
| `check_targets` | met iff `held ≥ T·MISSLEVEL − tol(T·MISSLEVEL)` |
| `compute_feature_shortfalls` | `max(0, T − held)`, set to 0 when `≤ tol(T)` |

`tol(v) = TARGET_ABS_TOL + TARGET_REL_TOL·|v|` with `TARGET_ABS_TOL = 1e-6`,
`TARGET_REL_TOL = 1e-9` (the Stage 3 lesson: an absolute tolerance alone fails at large
magnitudes). Applied only when `solution.is_fractional`; the grounding lens confirms the values
against CBC and HiGHS on random problems.

`build_solution` with fractional input raises `ValueError` for the same combinations as §3.5
(PROBMODE 3, clumping, separation, space), so a hand-built fractional array cannot reach the
binary-only post-hoc evaluators.

## 6. Readers of `Solution.selected`

### 6.1 Fraction-aware

| Reader | Fractional behaviour |
|---|---|
| `io/output.py` `write_run`, `write_best`, `write_solution_matrix` | Solution column written with `format(v, ".10g")`; binary unchanged (ints) |
| `io/output.py` `write_summary`, `io/writers.py` `write_sum` | `n_selected` (count, as today); no format change |
| `io/writers.py` `write_mv` | uses the fractional held/shortfall from §5.3 |
| `io/writers.py` `write_ssoln`, `analysis/selection_freq.py`, `io/spatial_export.py` `export_frequency_spatial` | sums as float: frequency = mean fraction; integral counts still written as ints |
| `io/readers.py` `read_ssoln` | `Number` kept int when every value is integral, float otherwise |
| `io/exporters.py` `export_solution_csv`, `export_summary_csv`; `io/spatial_export.py` `export_solution_spatial` | float column / weighted totals |
| `analysis/representation.py` `compute_representation` | represented amount `Σ r·x` |
| `analysis/equity.py` `compute_equity` | group value `Σ x·value` |
| `analysis/portfolio.py` `selection_frequency`, `summary_statistics` | already float / count |
| `pymarxan_shiny` `results/summary_table.py` | held `amount·x` |
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

### 6.3 Completeness test

`tests/pymarxan/test_fractional_readers.py` greps `src/` for `.selected` and asserts every
containing module is listed in a table of fraction-aware or guarded modules in the test, so a
new reader cannot slip in unclassified.

## 7. Errors (summary)

| Condition | Error |
|---|---|
| bad `decisions` / `upper` / `lower` | `ValueError` in `__init__` |
| non-binary + PROBMODE 3 / target2 / separation / space / forbidden_selections | `ValueError` before the build |
| binary-only reader given a fractional solution | `FractionalSolutionError` |
| `binarize_solution` threshold outside `(0, 1]` | `ValueError` |
| infeasible model (e.g. semicontinuous floor vs budget) | `[]` from `MIPSolver.solve`, as today |

## 8. Testing

- **Binary unchanged:** `build_model` MPS for binary byte-identical to v0.41.1 on
  `tests/data/simple` and a connectivity fixture; every v0.37–v0.41 golden and the 35.0 anchor
  stay green untouched.
- **LP bound:** `decisions="proportion"` objective ≤ binary objective on `tiny_problem` and on
  ≥ 20 seeded random problems (boundary + symmetric + asymmetric connectivity on).
- **Integral LP:** a named fixture with one feature per PU (identity `rij`, targets = amounts)
  whose LP optimum is integral; proportion objective == binary objective and every `x ∈ {0, 1}`.
- **Semicontinuous floor:** every `x_i ∈ {0} ∪ [lower, upper]`; objective between the
  proportion and binary objectives.
- **Upper cap:** `proportion, upper=0.5` → every `x ≤ 0.5`; locked-in PUs at exactly 0.5.
- **Hand calculation:** a 3-PU fixture with given fractions: held, cost, boundary
  (`|x_i − x_j|`), symmetric (`min`) and asymmetric (`max(0, ·)`) connectivity checked against
  hand-written numbers; `Solution.objective` equals the MIP objective value within 1e-6.
- **Targets met at the LP optimum:** `all_targets_met` is True and `shortfall == 0` for the
  `min_set` proportion solve on `tiny_problem` and the random set.
- **Connectivity reward survives:** with symmetric connectivity, fractional `z > 0` occurs and
  the objective is below the same model with `z` forced binary.
- **Guards:** every §6.2 reader raises `FractionalSolutionError`; every §3.5 combination raises.
- **Multi-objective:** a 2-objective hierarchical run with `decisions="proportion"` returns
  float selections whose `objective_values` match recomputation.
- **Readers:** completeness test (§6.3); writer round-trips (`write_ssoln` → `read_ssoln`
  float), Shiny summary table on a fractional solution.

## 9. Docs

CHANGELOG `[Unreleased]`; `docs/` user guide section "Partial selection" (proportion vs
semicontinuous, the prioritizr mapping `add_semicontinuous_decisions(u)` ≙
`decisions="proportion", upper=u`, the lock-in and evaluator deviations, the guarded list);
roadmap Stage 4 as-built note; `.github/copilot-instructions.md` solver matrix row.

## 10. Risks

- **Widest blast radius of the roadmap.** Mitigated by the dtype discriminator (binary code
  paths are untouched), the completeness test, and the design review's blast-radius lens.
- **Solver tolerance.** CBC/HiGHS primal tolerances leave active rows at `±1e-7`; §5.3's
  tolerance and §4's snapping must be checked by the grounding lens on both backends.
- **Users reading `n_selected` as area.** Documented; `selected_sum` is the fraction total.
