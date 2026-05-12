# Codebase Review 6 — Design

**Date:** 2026-02-23
**Scope:** Full codebase audit — post Reviews 1-5. 683 tests, 78.47% coverage.
**Context:** Reviews 1-5 fixed 87 issues total. This review targets remaining correctness bugs, missing edge cases, performance issues, and test quality.

## Approach

Three parallel review agents audited disjoint subsystems:
1. Core models, solvers, calibration, analysis, I/O, zones, connectivity
2. Spatial subpackage, Shiny UI modules, assembled app
3. Test suite quality, coverage gaps, assertion strength

Only issues at >80% confidence are included. Findings that overlapped with Reviews 1-5 (e.g., MISSLEVEL in `compute_zone_objective`, status=1 handling in zone SA) were filtered out. Two agent claims were rejected on verification:
- *Boundary double-counting in `ProblemCache`* — the explicit comment at `cache.py:411-414` shows the iterate-only-selected approach is correct for standard upper-triangular boundary files. The cache only double-counts if the input itself contains both (i,j) and (j,i) rows — addressed via Medium-4 below.
- *`_check_results` polling stops between solver runs* — agent itself flagged this as borderline; the reactive effect re-arms on every status change.

## Findings: 22 Issues

### CRITICAL (3) — Wrong results, crashes, or fundamental brokenness

**C1: `run_sweep_parallel` crashes on first infeasible point, losing all completed results**
`calibration/parallel.py:38-39,82-87`. `_solve_single` raises `ValueError("Solver returned no solutions (infeasible)")` when the solver returns `[]`. The `future.result()` call at line 83 re-raises this exception uncaught, propagating out of `run_sweep_parallel` and losing every completed sweep point. Additionally, `indexed_results[i]` at line 87 raises `KeyError` for any failed future because the dict was never populated for those indices. The sequential `run_sweep` correctly skips infeasible points (`if not sols: continue` — added in Review 5). The parallel version must mirror this: catch exceptions per-future, place a sentinel (None) in the result dict, and skip None entries when assembling final lists.

**C2: `CoolingSchedule.lundy_mees.temperature()` is O(step) — total cost O(num_steps²)**
`solvers/cooling.py:31-35`. The method iterates `for _ in range(step)` on every call. With `num_temp_steps=10_000`, the cooling loop calls `temperature(step)` for each `step` from 0 to 10_000, doing a cumulative ≈10⁸ iterations purely on temperature updates. SA runs with `COOLING=lundy_mees` will be ~100x slower than geometric. Fix: maintain temperature state incrementally — store `t` in the dataclass and update it on each call, or precompute all temperatures once on construction.

**C3: Shapefile upload broken — single-file upload cannot include sidecar `.shx`/`.dbf`**
`pymarxan_shiny/modules/spatial/import_wizard.py:38-44,116`, `cost_upload.py:35-40,119`. The UI exposes `.shp` as an `input_file` option with `multiple=False` (default). A Shapefile requires at minimum `.shp`, `.shx`, and `.dbf` — Fiona raises an error when called with just the `.shp`. Even the help text explicitly warns sidecar files are required, confirming the path was never exercised end-to-end. Fix: accept only `.zip` for shapefile uploads (extract before passing to `gpd.read_file`), or set `multiple=True` and reconstruct a temp directory with shared file stems. ZIP is simpler and matches the existing project-upload pattern.

### HIGH (6) — Incorrect behavior, missing guards, silent silent divergences

**H1: MIP solvers misclassify time-limit-with-feasible-solution as infeasible**
`solvers/mip_solver.py:232`, `zones/mip_solver.py:142`. Both check `model.status != pulp.constants.LpStatusOptimal` and return `[]` on mismatch. CBC sets `status=LpStatusNotSolved` or similar when the time limit is hit, even when a feasible-but-non-optimal integer solution exists. With `MIP_TIME_LIMIT > 0`, any non-trivial problem that fails to prove optimality silently returns no solutions. Fix: also accept `pulp.LpStatusNotSolved` when `pulp.value(model.objective)` is not None AND all decision vars have integer values — i.e., the relaxation has integer-feasible incumbent values.

**H2: `build_pu_feature_matrix` silently drops duplicate `(pu, species)` rows**
`models/problem.py:120`. Uses `matrix[ri, ci] = float(pv_am[k])` (assignment) — last write wins. But `feature_amounts()` at line 131 sums via `groupby("species")["amount"].sum()`. With any duplicate puvspr rows, SA's `ProblemCache` (which uses the matrix) optimizes a different objective than `build_solution` reports (which sums). Marxan reference behavior is to sum. Fix: change line 120 to `matrix[ri, ci] += float(pv_am[k])`.

**H3: `create_geo_map` does not reproject — non-WGS84 GeoDataFrames render off-coast of Africa**
`pymarxan_shiny/modules/mapping/map_utils.py:95-118`, `network_view.py:151-152`. ipyleaflet requires EPSG:4326 lat/lon. The helper passes `row.geometry.__geo_interface__` directly and computes `center` from `gdf.total_bounds`. For grids/imports with projected CRS (UTM, equal-area), polygons land at meaningless lat/lon. Same issue for centroid computation in `network_view`. Fix: at the top of `create_geo_map`, if `gdf.crs is not None and gdf.crs.to_epsg() != 4326`, call `gdf = gdf.to_crs("EPSG:4326")`. Apply the same guard in `network_view` before centroid extraction.

**H4: Download tempfiles never deleted — long sessions accumulate megabytes in /tmp**
`pymarxan_shiny/modules/results/export.py:57-71`, `modules/spatial_export/spatial_export.py:93-114`. Four download handlers create `NamedTemporaryFile(delete=False)`, write data, return the path. Nothing ever removes them. A user exporting 50 GeoPackages over a session accumulates ~50 × file-size in /tmp permanently. Fix: register cleanup via `session.on_ended` to `unlink` the paths, or use a session-scoped `TemporaryDirectory` whose teardown handles all paths.

**H5: `import_features_from_vector` skips CRS reprojection when source has no CRS**
`spatial/importers.py:89-93`. The guard at lines 90-93 reprojects only when both CRS are defined and differ. If `features_gdf.crs is None` (common for hand-edited GeoJSONs) while `planning_units.crs` is set, the function proceeds with `gpd.overlay()` on mismatched coordinate frames, producing silently empty or wrong intersections. Fix: if one CRS is None and the other is not, raise `ValueError("Feature file lacks CRS; cannot reproject. Set CRS explicitly or upload a file with .prj")`.

**H6: `ZoneSASolver` ignores `COOLING` parameter — always uses geometric**
`zones/solver.py:183-189`. Hardcoded `alpha = (0.001 / initial_temp) ** (1.0 / num_temp_steps)` and `temp *= alpha` (line ~225). The `COOLING` parameter is honoured by `SimulatedAnnealingSolver` but silently ignored here. Users selecting `linear` or `lundy_mees` for zone runs get geometric without warning. Fix: build a `CoolingSchedule` from `params.get("COOLING", "geometric")` and call `.temperature(step_count)` instead of multiplying `temp *= alpha`.

### MEDIUM (8) — Robustness, performance, completeness

**M1: `compute_irreplaceability` ignores MISSLEVEL and includes locked-out PUs in denominator**
`analysis/irreplaceability.py:38,52`. Line 38 sums `total_per_feat` across all PUs including status=3 (locked-out), so a unique provider that happens to be locked-out inflates "available total" and underestimates other PUs' criticality. Line 52 compares `remaining < feat_targets` against raw target, while all solver code uses `target * misslevel`. Fix: filter `pu_feat_matrix` to exclude locked-out PUs before summing; multiply `feat_targets` by `misslevel` in the comparison.

**M2: `gap_analysis.compute_gap_analysis` ignores MISSLEVEL — inconsistent with solver and exporter**
`analysis/gap_analysis.py:79-81`. `shortfall = max(targets[fid] - protected_amount[fid], 0.0)` uses raw target. The solver and `export_summary_csv` apply `target * misslevel`. UI gap analysis will display features as unmet that the solver considers met. Fix: scale `targets[fid]` by misslevel before computing shortfall.

**M3: `SweepResult.best` raises `ValueError` on empty objectives**
`calibration/sweep.py:53-55`. `min(range(len(self.objectives)), ...)` raises on empty range. If every sweep point was infeasible (post-C1 fix, this becomes reachable), `.best` crashes. Fix: return `None` or raise a descriptive `ValueError("No feasible solutions in sweep")`. Pick the latter and update the type hint to `Solution | None` or just raise.

**M4: `connectivity_to_matrix` silently overwrites both directions for explicit-bidirectional edge lists**
`connectivity/io.py:47-49`. Vectorized assignment `matrix[idx1, idx2] = val_col` overwrites — if the edge list has `(A, B, v1)` and `(B, A, v2)`, only the latter is kept for the `[A][B]` cell. Then the `symmetric` line overwrites again. For users who provide both directions to express asymmetric connectivity, the matrix loses half the data. Fix: use `np.add.at(matrix, (idx1, idx2), val_col)` for accumulation if duplicates are summed, OR detect duplicates and raise. Sum is the safer default.

**M5: `apply_wdpa_status` raises raw `TopologicalError` on invalid unioned geometry**
`spatial/wdpa.py:124-133`. `wdpa_union = wdpa.geometry.union_all()` can produce invalid geometries for complex coastlines. Subsequent `.intersection()` calls raise `shapely.errors.TopologicalError` that bubbles out of the Shiny effect as a raw stack trace. Fix: wrap in `try/except` and call `wdpa_union = wdpa_union.buffer(0)` to clean before intersecting; on second failure raise `ValueError` with actionable message.

**M6: `comparison_map_server._update_choices` resets Solution B to "1" on every new run**
`pymarxan_shiny/modules/mapping/comparison_map.py:101-110`. The effect fires on every change to `all_solutions()` and unconditionally calls `ui.update_select("sol_b", selected="1")`. If the user manually selected run 3 for B, the next solver execution clobbers it. Fix: preserve current selection when still valid: `selected=input.sol_b() if input.sol_b() in [c[0] for c in choices] else "1"`.

**M7: `fetch_gadm` does not strip whitespace from `admin_name` — `" "` matches everything**
`spatial/gadm.py:138-139`. `df[col].str.contains(admin_name, case=False, na=False)` with `admin_name=" "` matches every row containing a space (most place names). The Shiny caller uses `input.admin_name() or None` which correctly converts `""` to `None`, but a stray space character bypasses the guard. Fix: `admin_name = admin_name.strip() if admin_name else None` at top of the filter block.

**M8: `fetch_wdpa` blocks main Shiny thread for large countries**
`spatial/wdpa.py:37-58`. No bounding-box parameter is sent to the API; for large countries (Russia, Canada), the pagination loop fetches thousands of records on the main thread, freezing the UI for minutes with no feedback. Fix: at minimum, emit a `ui.notification_show` warning at page 5+ and switch to a `Notification(type="warning")` with progress text. A full fix would move the call to a background thread mirroring the solver pattern, but the lighter fix is acceptable as a Medium.

### Test Gaps (5) — Highest priority

**T1: `ProblemCache.compute_delta_objective` with `MISSLEVEL != 1.0` never tested**
`tests/pymarxan/solvers/test_cache.py`. `TestComputeDeltaObjective` only runs at `misslevel=1.0`. The delta logic at `cache.py:334` multiplies `feat_targets * self.misslevel` — if MISSLEVEL ≠ 1, delta computation diverges from full objective. Since SA's inner loop relies entirely on delta for accept/reject decisions, an incorrect delta under MISSLEVEL would silently produce suboptimal solutions with all existing tests passing. Add a test: with `MISSLEVEL=0.5`, for 10 random flips assert `compute_delta_objective ≈ full_after - full_before`.

**T2: `ScenarioSet.clone_scenario` mutation isolation for `feature_overrides` not verified**
`tests/pymarxan/analysis/test_scenario_overrides.py`. Existing tests check the clone has merged overrides; none verify the source's `feature_overrides` dict is untouched. A shallow copy regression would only surface for nested-dict overrides. Add a test: source has `feature_overrides={1: {"target": 10.0}}`, clone with overrides on feature 2; assert source's dict still has only key 1.

**T3: `ZoneSASolver` all-locked branch doesn't verify zone assignments respect locks**
`tests/pymarxan/zones/test_solver.py`. `test_all_locked_in_returns_solution` only asserts `len == 1` and `cost ≥ 0`. The forced-assignment loop (`solver.py:92-128`) could assign locked-in PUs to any zone or fail to lock-out without detection. Add tests: (a) lock all status=2 → assert every assignment > 0; (b) lock all status=3 → assert every assignment == 0.

**T4: `IterativeImprovementSolver` ITIMPTYPE=2 multi-pass convergence not regression-tested**
`tests/pymarxan/solvers/test_iterative_improvement.py`. The convergence loop added in Review 4 only has a single-pass-passing test. A regression to single-pass implementation would not be caught. Construct a problem where removal in pass 1 unlocks addition in pass 2; assert final solution differs from single-pass result.

**T5: `export_summary_csv` `achieved` column values never numerically verified**
`tests/pymarxan/io/test_exporters.py`. Tests assert column names and `met` boolean, but never read back `achieved` and compare to expected sum. A regression in the summing logic could keep `met` correct while emitting wrong `achieved` values. Add a test that builds a known small problem, runs export, and verifies `achieved` equals manually-summed amounts for selected PUs.

## Implementation Plan

**Batch 1 (3 CRITICAL):** C1 (parallel sweep), C2 (cooling), C3 (shapefile upload)
**Batch 2 (6 HIGH):** H1 (MIP status), H2 (puvspr accumulation), H3 (CRS reproject), H4 (tempfile cleanup), H5 (CRS None guard), H6 (zone cooling)
**Batch 3 (8 MEDIUM):** M1 (irreplaceability), M2 (gap MISSLEVEL), M3 (best on empty), M4 (connectivity dup), M5 (wdpa topological), M6 (comparison preserve), M7 (gadm strip), M8 (wdpa warning)
**Batch 4 (5 test gaps):** T1-T5

Total: 22 fixes, expected delta ~705 tests, ~79% coverage.
