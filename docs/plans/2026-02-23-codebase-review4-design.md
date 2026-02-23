# Codebase Review 4 — Design

**Date:** 2026-02-23
**Scope:** Full codebase audit — 9,025 lines of source, 7,763 lines of tests
**Context:** Post Reviews 1-3 and Phases 14-17. 622 tests, 77.51% coverage.

## Approach

Three parallel review agents audited:
1. Core models, solvers, calibration, analysis, I/O, zones
2. Spatial modules, Shiny UI modules, assembled app
3. Test quality, coverage gaps, assertion strength

Only issues at >80% confidence are included.

## Findings: 22 Issues

### CRITICAL (5) — Wrong results or crashes

**C1: `prop` (proportional target) column read but never resolved to effective targets**
`io/readers.py:78-80` reads `prop` from spec.dat, but no solver or utility computes `effective_target = max(target, prop * total_amount)`. In Marxan, when `target=0` and `prop=0.3`, the effective target is 30% of total available amount. All solvers, `ProblemCache`, and `check_targets` use only the `target` column. Fix: resolve `prop` into `target` during `load_project()` by computing `max(target, prop * sum_of_amounts)` per feature.

**C2: Zone SA solver never sets `penalty` or `shortfall` on Solution**
`zones/solver.py:184-200`. Constructs `Solution(...)` directly without calling `build_solution`, so `penalty` and `shortfall` always default to 0.0 even when zone targets are unmet. `write_sum` uses `sol.shortfall` directly, producing incorrect output. Fix: compute zone penalty/shortfall before constructing Solution.

**C3: Zone objective functions ignore MISSLEVEL**
`zones/objective.py:127,165`. Both `check_zone_targets` and `compute_zone_penalty` compare `achieved` directly against `target`, not `target * misslevel`. Non-zone solvers all apply MISSLEVEL. Fix: read MISSLEVEL from `problem.parameters` and multiply targets.

**C4: `apply_cost_from_vector` crashes when `planning_units.crs` is None**
`spatial/cost_surface.py:36`. CRS guard checks `cost_layer.crs is not None` but NOT `planning_units.crs is not None`. Calls `to_crs(None)` which raises `CRSError`. Same bug pattern as importers.py (fixed in Review 3) but missed in this file. Fix: check both CRS values are not None.

**C5: `run_with_overrides` unguarded `min()` on empty solutions**
`analysis/scenarios.py:159`. `best = min(solutions, key=lambda s: s.objective)` without checking if `solutions` is empty. Review 3 fixed this in all 5 calibration functions but missed `scenarios.py`. Fix: guard with `if not solutions: raise RuntimeError(...)`.

### HIGH (8) — Incorrect behavior, UX issues, or missing guards

**H1: `_sync_solver_params` causes full UI re-render cascade**
`pymarxan_app/app.py:188-217`. The reactive effect reads `problem()` and `solver_config()`, deep-copies, mutates, then calls `problem.set(updated)`. Every solver config change (dropdown, BLM slider) re-invalidates ALL modules that depend on `problem()` (maps, feature table, spatial grid, etc.). Fix: remove `_sync_solver_params`; sync parameters into the deepcopy inside `run_panel._run_solver` at solve time.

**H2: `feature_table_server._save` mutates problem in-place**
`pymarxan_shiny/modules/data/feature_table.py:66-69`. Gets `p = problem()` (no copy), mutates `p.features["target"]` directly, then calls `problem.set(p)`. Modules that captured a reference to `p` see mutated data before invalidation fires. Fix: deepcopy before mutating.

**H3: `run_panel_server` sets reactive values from background thread**
`pymarxan_shiny/modules/run_control/run_panel.py:90-91`. Calls `current_solution.set()` and `all_solutions.set()` from a daemon thread. Shiny's reactive system relies on the async event loop — cross-thread `set()` may not fire invalidation correctly. Fix: store results on the `SolverProgress` object and update reactive values from the polling callback on the main thread.

**H4: `_two_step` (ITIMPTYPE=2) does not loop until convergence**
`solvers/iterative_improvement.py:212-222`. Does one removal pass then one addition pass, then stops. Marxan reference alternates until neither pass improves. Fix: wrap in a `while True` loop that breaks when no improvement.

**H5: RunModePipeline iterative improvement defaults to no-op**
`solvers/run_mode.py:118`. RUNMODE 2/3/5 create `IterativeImprovementSolver()` with default `itimptype=0` (no-op). `improve()` reads `problem.parameters["ITIMPTYPE"]` to override, but if not set, improvement is silently skipped. Fix: default to `itimptype=2` when constructing the sub-solver in pipeline mode.

**H6: `read_pu` and `read_spec` don't default missing optional columns**
`io/readers.py:56-81`. In Marxan, `status` in pu.dat defaults to 0, `spf` in spec.dat defaults to 1.0, `name` is auto-generated. Current code doesn't add these defaults. Downstream code (`ProblemCache`, `write_mvbest`, `gap_analysis`) accesses these columns directly — crashes with `KeyError`. Fix: add defaults after reading.

**H7: `generate_planning_grid` infinite loop when `cell_size <= 0`**
`spatial/grid.py:79-84,101-108`. Both square and hex generators use `while` loops that increment by `cell_size`. If `cell_size <= 0`, loops never terminate. Grid builder UI has `min=0.001` but that's client-side only. Fix: validate at the top of `generate_planning_grid`.

**H8: `planning_unit.py` missing status 1 (INITIAL_INCLUDE)**
`models/planning_unit.py:3-7`. Defines `AVAILABLE=0, LOCKED_IN=2, LOCKED_OUT=3` and `VALID_STATUSES={0,2,3}`. Missing status 1. Any validation against `VALID_STATUSES` would reject valid status=1 PUs. Fix: add `INITIAL_INCLUDE = 1` and include it in `VALID_STATUSES`.

### MEDIUM (9) — Robustness, performance, completeness

**M1: `export_summary_csv` ignores MISSLEVEL in target checking**
`io/exporters.py:56`. Computes `met: achieved >= target` without MISSLEVEL. Solvers and output writers all use MISSLEVEL. Fix: apply `target * misslevel`.

**M2: ZoneSASolver `zone_options` includes 0 (unassigned), wasting iterations**
`zones/solver.py:64`. In a 2-zone problem, 1/3 of SA moves target "unassigned" — almost always rejected (increases penalty), reducing effective iteration count. Fix: exclude 0 from zone_options or make it configurable.

**M3: `fetch_wdpa` only fetches first 50 results, no pagination**
`spatial/wdpa.py:37-48`. Countries with many protected areas (Australia: 10,000+) are severely under-represented. Fix: add pagination loop.

**M4: `compute_adjacency` is O(n²) without spatial index**
`spatial/grid.py:143-154`. For 5,000 PUs, does ~12.5M pairwise geometric tests. Runs synchronously in the Shiny event loop. Fix: use `sindex` spatial index to prefilter candidates.

**M5: `connectivity_to_matrix` produces asymmetric matrices for undirected input**
`connectivity/io.py:19-24`. Only sets `matrix[i, j]`, not `matrix[j, i]`. Marxan boundary is typically undirected. Fix: add `symmetric=True` parameter.

**M6: `cost_upload_server` doesn't validate cost column exists in uploaded file**
`pymarxan_shiny/modules/spatial/cost_upload.py:66-68`. User-specified column name not validated before `apply_cost_from_vector`. Produces confusing `KeyError` inside overlay. Fix: validate column exists before calling.

**M7: `ScenarioSet.remove()` silently succeeds if name doesn't exist**
`analysis/scenarios.py:66-67`. No error raised for misspelled names. Fix: raise `KeyError` when nothing removed.

**M8: `summary_table_server` uses `iterrows()` for feature achievement — very slow for large datasets**
`pymarxan_shiny/modules/results/summary_table.py:24-34`. O(features × PUs) via Python row iteration. Fix: vectorized pandas groupby.

**M9: `gadm_picker` boundary never connected to `grid_builder` as clip polygon**
`pymarxan_app/app.py:256`. GADM boundary and grid builder sit on same Data tab but are disconnected. `generate_planning_grid` supports `clip_to` but the UI has no way to pass it. Fix: pass `gadm_boundary` reactive to `grid_builder_server`.

### Test Gaps (6 highest-priority)

**T1:** `compute_delta_objective` (ProblemCache) — never unit-tested in isolation; sign error would silently degrade SA convergence.
**T2:** `compute_feature_shortfalls` — never tested directly; used by penalty, build_solution, write_mvbest.
**T3:** Zone SA solution penalty/shortfall accuracy — untested (currently always 0.0, see C2).
**T4:** `apply_cost_from_vector` sum/max aggregation modes — only `area_weighted_mean` tested.
**T5:** HeuristicSolver boundary accuracy when `blm > 0` — no test verifies `sol.boundary` is correct.
**T6:** `write_mvbest` with MISSLEVEL < 1.0 — Target_Met column never tested with relaxed MISSLEVEL.

## Implementation Plan

**Batch 1 (5 CRITICAL):** C1-C5 — prop resolution, Zone SA penalty, zone MISSLEVEL, cost CRS, scenario guard
**Batch 2 (8 HIGH):** H1-H8 — reactive cascade, mutations, thread safety, convergence loop, pipeline defaults, reader defaults, grid guard, status constants
**Batch 3 (9 MEDIUM):** M1-M9 — exporter MISSLEVEL, zone SA efficiency, WDPA pagination, adjacency index, connectivity symmetry, cost validation, scenario remove, summary perf, GADM integration
**Batch 4 (6 test gaps):** T1-T6 — delta unit test, shortfalls, Zone SA output, cost aggregation, heuristic boundary, MISSLEVEL mvbest
