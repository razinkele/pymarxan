# Codebase Review 5 — Design

**Date:** 2026-02-23
**Scope:** Full codebase audit — post Reviews 1-4. 643 tests, 77.34% coverage.
**Context:** Reviews 1-4 fixed 68 issues total. This review targets remaining correctness bugs, missing edge cases, and test quality.

## Approach

Three parallel review agents audited:
1. Core models, solvers, calibration, analysis, I/O, zones
2. Spatial modules, Shiny UI modules, assembled app
3. Test quality, coverage gaps, assertion strength

Only issues at >80% confidence are included.

## Findings: 19 Issues

### CRITICAL (3) — Wrong results or crashes

**C1: ZoneProblemCache ignores MISSLEVEL — zone SA optimizes wrong targets**
`zones/cache.py:157-167,455-464`. The cache stores raw zone targets without applying MISSLEVEL. All cache-based penalty computations (`_compute_zone_penalty`, `_penalty_delta`) compare held amounts against raw targets. Meanwhile, `zones/objective.py` (fixed in Review 4) correctly applies MISSLEVEL. When MISSLEVEL < 1.0, the SA penalizes more harshly than intended during optimization, and the final solution's objective (from cache) disagrees with the stored penalty (from `compute_zone_penalty`). Fix: multiply zone targets by MISSLEVEL when building the cache.

**C2: Zone SA crashes with ValueError when all PUs are locked**
`zones/solver.py:78-81,146`. When all PUs are locked-in/out, `swappable` is empty and `n_swappable = 0`. Calls to `rng.integers(n_swappable)` at lines 113 and 146 raise `ValueError: high <= 0`. The regular SA solver handles this at lines 86-95, but Zone SA has no guard. Fix: add early-return guard building a forced-assignment solution.

**C3: Feature table save corrupts data when user sorts the DataGrid**
`pymarxan_shiny/modules/data/feature_table.py:68-71`. `data_view()` returns data in the user's current sort/filter order. Positional assignment to `updated.features["target"]` applies values to the wrong features after sorting. Fix: join on the `id` column instead of positional assignment.

### HIGH (5) — Incorrect behavior or missing guards

**H1: Calibration functions crash on empty solver results**
`calibration/blm.py:61`, `spf.py:55`, `sweep.py:95`, `sensitivity.py:70`, `parallel.py:38`. All five call `min(sols, ...)` without guarding empty `sols`. The MIP solver returns `[]` when infeasible. Review 3 memory says this was fixed, but the guards are absent in the current code. Fix: add `if not sols: continue` before each `min()`.

**H2: Zone SA cooling schedule skips same-zone iterations**
`zones/solver.py:149-177`. When `new_zone == old_zone`, `continue` at line 150 skips `step_count` and `iter_count` increments (lines 169-177). For 2 zones, 50% of iterations are skipped, so SA cools half as fast as intended. The regular SA doesn't have this problem (it doesn't skip same-PU flips). Fix: move counter increments before the `continue`.

**H3: Zone SA missing STATUS_INITIAL_INCLUDE (status=1) handling**
`zones/solver.py:68-77`. Zone SA handles status=2 (locked-in) and status=3 (locked-out) but ignores status=1 (initial include). PUs with status=1 get random zone assignment instead of starting in a non-zero zone. Fix: treat status=1 like starting selected (assign to first zone) while keeping them swappable.

**H4: grid_builder crashes with TypeError when numeric inputs are cleared**
`pymarxan_shiny/modules/spatial/grid_builder.py:61-73`. Shiny's `input_numeric` returns `None` when cleared. No None guard or try/except. Fix: validate inputs and show notification.

**H5: MIP solver and SA all-locked path return aliased solutions**
`solvers/mip_solver.py:142`, `solvers/simulated_annealing.py:95`. `[sol] * config.num_solutions` creates references to the same object. Mutating one solution's metadata affects all. Fix: use `[copy.deepcopy(sol) for _ in range(config.num_solutions)]` or build fresh each time.

### MEDIUM (7) — Robustness, performance, completeness

**M1: SA alpha > 1 when initial_temp < 0.001**
`solvers/simulated_annealing.py:154`, `zones/solver.py:133`. When average delta is tiny, `initial_temp < 0.001`, making `alpha = (0.001 / initial_temp) ** (1/steps) > 1`. Temperature increases instead of decreasing. Fix: `initial_temp = max(initial_temp, 0.001)`.

**M2: network_view adds O(n²) Polyline elements, freezing the browser**
`pymarxan_shiny/modules/mapping/network_view.py:132-144`. Dense 500-PU matrix creates ~250K polylines. Fix: cap at 5000 edges or use a single GeoJSON layer.

**M3: HeuristicSolver includes locked-out PUs in total_available for rarity**
`solvers/heuristic.py:191-194`. Locked-out PUs inflate the denominator, underestimating rarity/irreplaceability of remaining PUs. Fix: exclude locked-out PUs from `total_available`.

**M4: cost_surface apply_cost_from_vector O(n*m) update loop**
`spatial/cost_surface.py:71-72`. Per-PU `result.loc[result["id"] == pu_id]` scans full DataFrame. Fix: use `.map()` or set_index for O(1) lookup.

**M5: combine_cost_layers silently truncates mismatched weights/layers**
`spatial/cost_surface.py:107`. `zip(layers, weights)` drops extras silently. Fix: validate `len(weights) == len(layers)`.

**M6: validate() crashes with KeyError if required columns missing from pu_vs_features**
`models/problem.py:146-160`. Cross-reference ID checks access columns before checking if they exist. Fix: guard with `if not missing_puvspr`.

**M7: gadm.py fetch_gadm raises opaque KeyError on API format changes**
`spatial/gadm.py:121`. `meta["gjDownloadURL"]` gives unhelpful error. Fix: use `.get()` with descriptive ValueError.

### Test Gaps (4 highest-priority)

**T1:** `build_solution` COSTTHRESH code path (`utils.py:186-192`) — never tested, used by all solvers.
**T2:** ZoneSASolver type guard (`zones/solver.py:41-43`) — rejects non-ZonalProblem, never tested.
**T3:** SA history monotonicity — never verified that `best_objective` is non-increasing or `temperature` is decreasing.
**T4:** MarxanBinarySolver at 51% coverage — subprocess, timeout, and error paths untested.

## Implementation Plan

**Batch 1 (3 CRITICAL):** C1-C3 — zone cache MISSLEVEL, zone SA all-locked, feature table sort
**Batch 2 (5 HIGH):** H1-H5 — calibration guards, cooling schedule, status=1, grid builder, solution aliasing
**Batch 3 (7 MEDIUM):** M1-M7 — alpha clamp, network cap, heuristic rarity, cost perf, weights validation, validate guard, gadm error
**Batch 4 (4 test gaps):** T1-T4 — COSTTHRESH, type guard, history monotonicity, binary solver
