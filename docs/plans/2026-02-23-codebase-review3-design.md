# Codebase Review 3 — Design

**Date:** 2026-02-23
**Scope:** Full codebase audit — all 6,048 lines across pymarxan-core, pymarxan-shiny, pymarxan-app
**Context:** Post Phases 14-17 (MaPP feature porting). 597 tests, 77.44% coverage.

## Approach

Three parallel review agents audited:
1. Core + solvers + calibration + analysis + I/O
2. Spatial + Shiny modules + app.py
3. Test suite gaps and weak assertions

Only issues at >80% confidence are included.

## Findings: 20 Issues

### CRITICAL (5) — Incorrect results or crashes

**C1: MIP solver skips self-boundary in objective**
`solvers/mip_solver.py:81-88`. The MIP objective `continue`s on self-edges (`id1 == id2`), but `compute_boundary()` in `utils.py` and `cache.py` both include self-boundary. The MIP minimizes a different boundary cost than what SA/heuristic report. Fix: add self-boundary term `bval * blm * x[id1]` to the MIP cost expression.

**C2: `min()` on empty solution list crashes in 5 calibration functions**
`calibration/blm.py:61`, `sweep.py:95`, `sensitivity.py:70`, `spf.py:55`, `parallel.py:38`. `solver.solve()` returns `[]` when infeasible (MIP solver line 133). All call sites do `min(sols, key=...)` without guarding, producing `ValueError`. Fix: guard with `if not sols: continue`.

**C3: WDPA apply_wdpa_status has no CRS reprojection**
`spatial/wdpa.py:108`. Computes intersection between `pu_gdf` and `wdpa` without checking CRS match. If PUs are in a projected CRS (e.g. UTM) and WDPA in EPSG:4326, overlap ratios are nonsense. Every other spatial function guards CRS. Fix: reproject `wdpa` to `pu_gdf.crs` before `union_all()`.

**C4: Irreplaceability denominator includes zero-target features**
`analysis/irreplaceability.py:50`. Score = `critical_count / n_features` where `n_features` is total count. But the loop skips features with `target <= 0`, so `critical_count` can never reach `n_features`. Max score < 1.0 whenever any feature has zero target. Fix: use count of positive-target features as denominator.

**C5: HeuristicSolver boundary loop skips self-edges**
`solvers/heuristic.py:255-262`. `selected[i] != selected[j]` is always `False` when `id1 == id2` (i.e. `i == j`), so self-boundary is never added. Underreports boundary cost vs. SA and `compute_boundary()`. Fix: add explicit self-edge branch matching `utils.py`.

### HIGH (7) — Incorrect behavior or performance

**H1: `_sync_solver_params` mutates problem without `problem.set()`**
`pymarxan_app/app.py:188-214`. Mutates `p.parameters` in-place but never calls `problem.set(p)`. Shiny reactive graph not notified. `run_panel.py` does `deepcopy(problem())` which captures the stale value. Fix: call `problem.set(p)` after mutations (or use deepcopy + set pattern).

**H2: Hex grid row step produces non-tessellating hexagons**
`spatial/grid.py:95-105`. Row step `h = cell_size * sqrt(3) / 2` equals the full hex height. For flat-top tessellation, the vertical step between row centers should be `3/4 * full_height` to create the overlap. Current code produces gaps. Fix: change row step to `h * 3/4` (i.e. `cell_size * sqrt(3) * 3 / 8`).

**H3: Zone boundary costs not stored symmetrically**
`zones/cache.py:505-513`. `zone_boundary_costs` populated from data file with only one direction `(zcol1, zcol2)`. Delta computation looks up `(old_col, zcol_j)` which may not exist if only `(zcol_j, old_col)` was stored. Fix: populate both directions during `from_zone_problem()`.

**H4: `write_sum` Shortfall column = SPF-weighted penalty, not raw shortfall**
`io/writers.py:244`. `shortfall = penalty` but `penalty = SPF * shortfall_per_feature`. Only correct when all SPF = 1.0. Fix: compute actual shortfall from solution data or add `shortfall` field to Solution.

**H5: `import_features_from_vector` CRS guard missing `planning_units.crs is not None` check**
`spatial/importers.py:90`. `to_crs(None)` crashes when PU GDF has no CRS. Fix: add `planning_units.crs is not None` to the guard.

**H6: `apply_cost_from_vector` O(n^2) geometry lookup**
`spatial/cost_surface.py:52`. Per-PU `planning_units.loc[planning_units["id"] == pu_id]` inside a groupby loop. Fix: precompute `pu_area_by_id = dict(zip(planning_units["id"], planning_units.geometry.area))`.

**H7: `scenarios.py` forward references without `from __future__ import annotations`**
`analysis/scenarios.py:114-118`. Uses `# noqa: F821` suppressions for unimported types. `get_type_hints()` raises `NameError`. Fix: add `from __future__ import annotations`.

### Test Gaps (8)

**T1:** `data_input/upload.py` — zero tests for the main data loading module.
**T2:** `summary_table.py`, `export.py`, `zone_config.py`, `blm_explorer.py` — no test files.
**T3:** ~10 Shiny modules only have `assert callable()` smoke tests.
**T4:** `compute_zone_penalty` — critical penalty path never tested with unmet targets.
**T5:** `overlap_matrix` Jaccard — only identical selections tested, formula unverifiable.
**T6:** `write_sum` test — tautological (tests default `penalty=0.0`, not any behavior).
**T7:** `irreplaceability` score values never verified numerically.
**T8:** No roundtrip test for `write_mvbest` / `read_mvbest`.

## Implementation Plan

**Batch 1 (5 CRITICAL):** C1-C5 — solver correctness, spatial CRS, irreplaceability
**Batch 2 (7 HIGH):** H1-H7 — app sync, hex tessellation, zone cache, writers, spatial guards
**Batch 3 (8 test gaps):** T1-T8 — missing tests, weak assertions, roundtrip tests
