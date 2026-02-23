# Codebase Review 2 — Remediation Design

**Date:** 2026-02-23
**Scope:** Fix all CRITICAL + HIGH findings from second codebase review (23 fixes in 3 batches)
**Out of scope:** MEDIUM issues (duplicate panels, type annotations, dead code, temp file leak) and LOW issues (seed default, unreachable fallbacks)

## Context

Three parallel review agents audited the entire codebase across core, Shiny, and test layers. They found 8 CRITICAL bugs (wrong answers, crashes, broken UI), 15 HIGH issues (silent data loss, race conditions, O(n²) hotspots), and ~12 MEDIUM/LOW items.

This design addresses the 23 CRITICAL + HIGH findings.

---

## Batch 1 — Solver Correctness (8 fixes)

Bugs that produce **wrong answers or crash**.

### 1.1 MIP solver missing SPF penalty

`mip_solver.py:148` computes `objective = total_cost + blm * total_boundary` but omits the SPF penalty for unmet features. Same bug pattern fixed in `build_solution` during Review 1, but MIP builds its Solution directly.

**Fix:** Call `compute_penalty()` and `compute_cost_threshold_penalty()` from `solvers/utils.py`, matching `build_solution`'s logic. Also apply MISSLEVEL to MIP constraint targets (line 113).

### 1.2 Guard `pulp.value()` returning None

`mip_solver.py:134`: `bool(round(pulp.value(x[pid])))` crashes with `TypeError: type NoneType doesn't define __round__` when the solver is infeasible or times out.

**Fix:** Check `model.status != pulp.constants.LpStatusOptimal` before extracting. Return empty list for infeasible.

### 1.3 HeuristicSolver ignores MISSLEVEL

`heuristic.py:196`: `remaining[fid] = float(row["target"])` uses raw target. Should be `target * misslevel`.

**Fix:** Read MISSLEVEL from `problem.parameters`, apply to initial remaining targets.

### 1.4 `write_mvbest` ignores MISSLEVEL for Target_Met

`writers.py:176`: `target_met = amount_held >= target` uses raw target. `compute_feature_shortfalls` already applies MISSLEVEL (fixed in Review 1), but the Target_Met column doesn't.

**Fix:** Apply MISSLEVEL: `target_met = amount_held >= target * misslevel`.

### 1.5 `write_sum` penalty reconstruction ignores BLM

`writers.py:243`: `penalty = max(0.0, sol.objective - sol.cost - sol.boundary)` is wrong because `sol.boundary` is the raw boundary value, not BLM-multiplied. When BLM != 1, the reconstructed penalty is incorrect.

**Fix:** Add a `penalty` field to the `Solution` dataclass. Populate it in `build_solution()` and in each solver that builds Solution directly. `write_sum` reads `sol.penalty` directly.

### 1.6 Handle STATUS_INITIAL_INCLUDE (status=1)

`STATUS_INITIAL_INCLUDE` (status=1) is defined but no solver handles it. Per Marxan spec, status=1 PUs should start selected but remain swappable. Currently they're treated as status=0 (start unselected).

**Fix:** In SA solver, MIP solver, heuristic, and iterative improvement: initialize status=1 PUs as selected but allow swapping. MIP: no constraint (already correct by omission). SA: set initial selection. Heuristic: pre-select. Iterative: pre-select.

### 1.7 Export missing writer functions

`io/__init__.py` exports readers (`read_mvbest`, `read_ssoln`, `read_sum`) but not the corresponding writers.

**Fix:** Add `write_mvbest`, `write_ssoln`, `write_sum` to imports and `__all__`.

### 1.8 Add `penalty` field to Solution

Supporting fix for 1.5. Add `penalty: float = 0.0` to Solution dataclass. Update `build_solution()` to populate it. Update all direct Solution constructors.

---

## Batch 2 — Shiny UX Bugs (7 fixes)

Broken or misleading UI elements.

### 2.1 Fix feature_table CellPatch keys

`feature_table.py:58`: `patch["column_id"]` raises KeyError. Shiny's `CellPatch` has `row_index`, `column_index`, `value` only.

**Fix:** Map `column_index` to column name using the rendered column order `["id", "name", "target", "spf"]`.

### 2.2 Feature table write-back

Edits in the DataGrid never persist to `problem().features`. Solvers always use original values.

**Fix:** Add "Save Changes" button that writes `feature_grid.data_view()` back to `problem().features` and calls `problem.set(p)`.

### 2.3 Progress bar polling for all solvers

`run_panel.py`: `progress.status` is only set to `"running"` by SA/Zone SA solvers. MIP, greedy, iterative, pipeline show frozen UI.

**Fix:** Set `progress.status = "running"` in `run_panel._run_solver()` before starting the thread.

### 2.4 Forward mip_verbose to SolverConfig

`run_panel.py:76`: `verbose=False` hardcoded. The `mip_verbose` checkbox value is in `config_dict` but never read.

**Fix:** `verbose=bool(config_dict.get("mip_verbose", False))`.

### 2.5 deepcopy problem before solver thread

`run_panel.py:64`: `p.parameters["BLM"] = ...` mutates the shared problem from the background thread while the UI thread may be reading it.

**Fix:** `import copy; p = copy.deepcopy(problem())` before any mutations, pass copy to solver.

### 2.6 spatial_grid.py orphaned map_summary

`spatial_grid.py:94-102`: `map_summary` renderer registered unconditionally, orphaned when `_HAS_IPYLEAFLET=False`.

**Fix:** Move inside `if _HAS_IPYLEAFLET:` block, matching the other 4 map modules.

### 2.7 Calibration error handling

`spf_explorer.py` and `sweep_explorer.py` lack try/except. Solver errors crash the Shiny session.

**Fix:** Wrap calibration calls in try/except with `ui.notification_show` error reporting.

---

## Batch 3 — Performance + CI (8 fixes)

O(n²) hotspots and CI enforcement gaps.

### 3.1 Iterative improvement: use ProblemCache

`iterative_improvement.py`: calls `compute_objective()` for every flip attempt — O(n) per call, O(n²) per pass.

**Fix:** Create ProblemCache once, use `compute_delta_objective()` for O(degree) per flip.

### 3.2 ZoneSASolver: dict lookup instead of list.index()

`zones/solver.py:70`: `pu_ids.index(int(row["id"]))` is O(n) inside an O(n) loop.

**Fix:** Build `pu_id_to_idx = {pid: i for i, pid in enumerate(pu_ids)}` once, use dict lookup.

### 3.3 GapResult.to_dataframe: dict lookup

`gap_analysis.py:33`: `self.feature_ids.index(fid)` is O(n) inside O(n) loop.

**Fix:** Build `fid_to_name` dict once.

### 3.4 CI: remove redundant networkx install

`ci.yml:41`: `pip install -e ".[all]" networkx` — networkx is already in `[shiny]` deps since Review 1.

**Fix:** Remove `networkx` from the pip install line.

### 3.5 CI: enforce coverage threshold

`ci.yml:42`: `pytest tests/ -v --cov --cov-report=term-missing` has no `--cov-fail-under`.

**Fix:** Add `--cov-fail-under=75`.

### 3.6 Fix test_solutions_are_different

The test asserts `cost >= 0` which always passes. Doesn't actually verify solution variation.

**Fix:** Assert that not all solutions have identical `selected` arrays.

### 3.7 Calibration type annotations

`blm_explorer.py:43` and `sensitivity_ui.py:73` annotate `solver` as `reactive.Value` but receive `reactive.Calc`.

**Fix:** Change to `reactive.Calc`.

### 3.8 Remove dead solver_thread reactive value

`run_panel.py:46,107`: `solver_thread` is set but never read.

**Fix:** Remove both lines.

---

## Testing Strategy

- Each batch runs the full test suite before committing
- New tests for: MIP penalty, MIP infeasible guard, MISSLEVEL in heuristic, Solution.penalty field, feature_table patch fix, iterative improvement with cache
- Full regression + lint + mypy after all batches
- Coverage must remain >= 75%

## Out of Scope (deferred)

- Duplicate summary_table/target_met panels (MEDIUM — cosmetic)
- export.py temp file leak (MEDIUM — harmless on Linux)
- Seed default contradiction (LOW)
- Unreachable `or N` fallbacks (LOW)
- Background threading for calibration modules (architectural — future phase)
