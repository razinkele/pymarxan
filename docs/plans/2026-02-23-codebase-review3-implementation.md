# Codebase Review 3 — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix 20 issues found in codebase review 3 — 5 critical, 7 high, 8 test gaps.

**Architecture:** All fixes are surgical edits to existing files. No new modules. Test-first for each fix.

**Tech Stack:** Python, pytest, numpy, geopandas, pulp, shiny

---

## Batch 1 — CRITICAL Fixes (5 Tasks)

### Task 1: C1 — MIP solver self-boundary in objective

**Files:**
- Modify: `src/pymarxan/solvers/mip_solver.py:81-88`
- Test: `tests/pymarxan/solvers/test_mip_solver.py`

**Step 1: Write the failing test**

Add to `tests/pymarxan/solvers/test_mip_solver.py`:

```python
def test_mip_includes_self_boundary(simple_problem):
    """MIP objective must include self-boundary (external boundary) terms."""
    import pandas as pd
    from pymarxan.solvers.mip_solver import MIPSolver

    # Add self-boundary entries (external boundary for each PU)
    self_boundary = pd.DataFrame({
        "id1": [1, 2, 3],
        "id2": [1, 2, 3],
        "boundary": [5.0, 5.0, 5.0],
    })
    if simple_problem.boundary is not None:
        simple_problem.boundary = pd.concat(
            [simple_problem.boundary, self_boundary], ignore_index=True,
        )
    else:
        simple_problem.boundary = self_boundary
    simple_problem.parameters["BLM"] = 1.0

    solver = MIPSolver()
    sols = solver.solve(simple_problem)
    assert len(sols) > 0
    sol = sols[0]
    # Self-boundary should contribute to boundary cost
    # Each selected PU adds 5.0 to boundary
    assert sol.boundary > 0.0, "Self-boundary must be included in MIP solution"
```

**Step 2: Run test to verify it fails**

Run: `/opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/solvers/test_mip_solver.py::test_mip_includes_self_boundary -v`
Expected: FAIL — MIP skips self-edges, so boundary reported by `build_solution` includes them but the MIP objective optimized without them (solution may differ).

**Step 3: Fix MIP solver**

In `src/pymarxan/solvers/mip_solver.py`, replace lines 81-88:

```python
                if id1 == id2:
                    # External/diagonal boundary: contributes when PU is NOT selected
                    # In Marxan, this is the perimeter cost when PU is selected but
                    # adjacent PU is not. For self-edges, it represents external boundary.
                    # Actually skip self-edges in the MIP objective since they represent
                    # the external boundary. We handle them in the actual boundary calculation.
                    # For the MIP, we only linearize the off-diagonal pairs.
                    continue
```

With:

```python
                if id1 == id2:
                    # Self-boundary: external boundary cost when PU is selected
                    boundary_expr += bval * x[id1]
```

**Step 4: Run test to verify it passes**

Run: `/opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/solvers/test_mip_solver.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add src/pymarxan/solvers/mip_solver.py tests/pymarxan/solvers/test_mip_solver.py
git commit -m "fix(mip): include self-boundary terms in MIP objective"
```

---

### Task 2: C2 — Guard empty solution lists in calibration

**Files:**
- Modify: `src/pymarxan/calibration/blm.py:52-65`
- Modify: `src/pymarxan/calibration/spf.py:53-56`
- Modify: `src/pymarxan/calibration/sensitivity.py:69-70`
- Modify: `src/pymarxan/calibration/sweep.py:86-99`
- Modify: `src/pymarxan/calibration/parallel.py:37-38`
- Test: `tests/pymarxan/calibration/test_blm.py`

**Step 1: Write the failing test**

Add to `tests/pymarxan/calibration/test_blm.py`:

```python
def test_calibrate_blm_handles_infeasible(simple_problem):
    """calibrate_blm should skip BLM values where solver returns no solutions."""
    from unittest.mock import MagicMock
    from pymarxan.calibration.blm import calibrate_blm
    from pymarxan.solvers.base import SolverConfig

    mock_solver = MagicMock()
    # First call returns empty (infeasible), second returns a solution
    from pymarxan.solvers.base import Solution
    import numpy as np
    sol = Solution(
        selected=np.array([True, True, True, True, True, True]),
        cost=10.0, boundary=5.0, objective=15.0,
        targets_met={1: True, 2: True, 3: True},
    )
    mock_solver.solve.side_effect = [[], [sol]]

    result = calibrate_blm(
        simple_problem, mock_solver,
        blm_values=[0.0, 1.0],
        config=SolverConfig(num_solutions=1),
    )
    # Should have 1 result (the feasible one), not crash
    assert len(result.blm_values) == 1
    assert result.blm_values[0] == 1.0
```

**Step 2: Run test to verify it fails**

Run: `/opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/calibration/test_blm.py::test_calibrate_blm_handles_infeasible -v`
Expected: FAIL with `ValueError: min() arg is an empty sequence`

**Step 3: Fix all 5 calibration functions**

In `src/pymarxan/calibration/blm.py`, replace the loop body (lines 52-65):

```python
    result_blm_values: list[float] = []

    for blm in blm_values:
        modified = ConservationProblem(
            planning_units=problem.planning_units,
            features=problem.features,
            pu_vs_features=problem.pu_vs_features,
            boundary=problem.boundary,
            parameters={**problem.parameters, "BLM": blm},
        )
        sols = solver.solve(modified, config)
        if not sols:
            continue
        best = min(sols, key=lambda s: s.objective)
        result_blm_values.append(blm)
        costs.append(best.cost)
        boundaries.append(best.boundary)
        objectives.append(best.objective)
        solutions_list.append(best)

    return BLMResult(
        blm_values=result_blm_values,
```

In `src/pymarxan/calibration/spf.py`, add guard after line 54:

```python
        sols = solver.solve(modified, config)
        if not sols:
            continue
        best = min(sols, key=lambda s: s.objective)
```

In `src/pymarxan/calibration/sensitivity.py`, add guard after line 69:

```python
            sols = solver.solve(modified, solver_config)
            if not sols:
                continue
            best = min(sols, key=lambda s: s.objective)
```

In `src/pymarxan/calibration/sweep.py`, replace lines 86-99 with:

```python
    valid_param_dicts: list[dict] = []

    for params in param_dicts:
        modified = ConservationProblem(
            planning_units=problem.planning_units,
            features=problem.features,
            pu_vs_features=problem.pu_vs_features,
            boundary=problem.boundary,
            parameters={**problem.parameters, **params},
        )
        sols = solver.solve(modified, solver_config)
        if not sols:
            continue
        best = min(sols, key=lambda s: s.objective)
        valid_param_dicts.append(params)
        solutions.append(best)
        costs.append(best.cost)
        boundaries.append(best.boundary)
        objectives.append(best.objective)

    return SweepResult(
        param_dicts=valid_param_dicts,
```

In `src/pymarxan/calibration/parallel.py`, add guard in `_solve_single` after line 37:

```python
    sols = solver.solve(modified, solver_config)
    if not sols:
        return (index, None)
    best = min(sols, key=lambda s: s.objective)
    return (index, best)
```

And in `run_sweep_parallel`, filter None results (after line 81):

```python
        for future in as_completed(futures):
            idx, sol = future.result()
            if sol is not None:
                indexed_results[idx] = sol

    # Reassemble in order, skipping infeasible points
    valid_indices = sorted(indexed_results.keys())
    solutions = [indexed_results[i] for i in valid_indices]
    valid_param_dicts = [param_dicts[i] for i in valid_indices]
    costs = [s.cost for s in solutions]
    boundaries = [s.boundary for s in solutions]
    objectives = [s.objective for s in solutions]

    return SweepResult(
        param_dicts=valid_param_dicts,
```

**Step 4: Run tests**

Run: `/opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/calibration/ -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add src/pymarxan/calibration/blm.py src/pymarxan/calibration/spf.py \
  src/pymarxan/calibration/sensitivity.py src/pymarxan/calibration/sweep.py \
  src/pymarxan/calibration/parallel.py tests/pymarxan/calibration/test_blm.py
git commit -m "fix(calibration): guard against empty solution lists in all calibration functions"
```

---

### Task 3: C3 — WDPA CRS reprojection

**Files:**
- Modify: `src/pymarxan/spatial/wdpa.py:106-108`
- Test: `tests/pymarxan/spatial/test_wdpa.py`

**Step 1: Write the failing test**

Add to `tests/pymarxan/spatial/test_wdpa.py`:

```python
def test_apply_wdpa_status_reprojects_crs():
    """apply_wdpa_status must reproject WDPA to PU CRS before intersection."""
    import copy
    import geopandas as gpd
    import numpy as np
    from shapely.geometry import box
    from pymarxan.models.problem import ConservationProblem
    from pymarxan.spatial.wdpa import apply_wdpa_status

    # PUs in a projected CRS (EPSG:32610 — UTM zone 10N)
    pu_gdf = gpd.GeoDataFrame(
        {"id": [1, 2], "cost": [1.0, 1.0], "status": [0, 0]},
        geometry=[box(500000, 5000000, 501000, 5001000),
                  box(501000, 5000000, 502000, 5001000)],
        crs="EPSG:32610",
    )
    problem = ConservationProblem(
        planning_units=pu_gdf,
        features=gpd.GeoDataFrame({"id": [1], "name": ["f"], "target": [1.0], "spf": [1.0]}),
        pu_vs_features=gpd.GeoDataFrame({"species": [1], "pu": [1], "amount": [1.0]}),
    )

    # WDPA polygon covering PU 1, but in EPSG:4326
    # Convert PU 1 bounds to lon/lat
    pu1_4326 = gpd.GeoDataFrame(
        geometry=[box(500000, 5000000, 501000, 5001000)],
        crs="EPSG:32610",
    ).to_crs("EPSG:4326")
    wdpa = gpd.GeoDataFrame(
        {"name": ["PA1"], "desig": ["NP"], "iucn_cat": ["II"]},
        geometry=pu1_4326.geometry.values,
        crs="EPSG:4326",
    )

    result = apply_wdpa_status(problem, wdpa, overlap_threshold=0.5)
    # PU 1 should be locked in (status=2)
    assert result.planning_units.iloc[0]["status"] == 2
    # PU 2 should remain unchanged
    assert result.planning_units.iloc[1]["status"] == 0
```

**Step 2: Run test to verify it fails**

Run: `/opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/spatial/test_wdpa.py::test_apply_wdpa_status_reprojects_crs -v`
Expected: FAIL — without reprojection, intersection produces wrong results

**Step 3: Fix WDPA CRS**

In `src/pymarxan/spatial/wdpa.py`, add CRS reprojection before line 108:

```python
    pu_gdf = result.planning_units

    # Reproject WDPA to PU CRS if they differ
    if wdpa.crs is not None and pu_gdf.crs is not None and wdpa.crs != pu_gdf.crs:
        wdpa = wdpa.to_crs(pu_gdf.crs)

    wdpa_union = wdpa.geometry.union_all()
```

**Step 4: Run tests**

Run: `/opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/spatial/test_wdpa.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add src/pymarxan/spatial/wdpa.py tests/pymarxan/spatial/test_wdpa.py
git commit -m "fix(wdpa): reproject WDPA to PU CRS before intersection"
```

---

### Task 4: C4 — Irreplaceability denominator

**Files:**
- Modify: `src/pymarxan/analysis/irreplaceability.py:30,50`
- Test: `tests/pymarxan/analysis/test_irreplaceability.py`

**Step 1: Write the failing test**

Add to `tests/pymarxan/analysis/test_irreplaceability.py`:

```python
def test_irreplaceability_excludes_zero_target_features():
    """Score denominator must only count features with positive targets."""
    import pandas as pd
    from pymarxan.models.problem import ConservationProblem
    from pymarxan.analysis.irreplaceability import compute_irreplaceability

    pu = pd.DataFrame({"id": [1, 2], "cost": [1.0, 1.0], "status": [0, 0]})
    features = pd.DataFrame({
        "id": [1, 2],
        "name": ["has_target", "zero_target"],
        "target": [10.0, 0.0],
        "spf": [1.0, 1.0],
    })
    puvspr = pd.DataFrame({
        "species": [1, 1, 2, 2],
        "pu": [1, 2, 1, 2],
        "amount": [10.0, 5.0, 5.0, 5.0],
    })
    problem = ConservationProblem(
        planning_units=pu, features=features, pu_vs_features=puvspr,
    )
    scores = compute_irreplaceability(problem)
    # PU 1 is the sole provider of enough for feature 1 (10.0 >= target 10.0)
    # Removing PU 1: remaining = 5.0 < 10.0 => critical for feature 1
    # Only 1 positive-target feature, so score = 1/1 = 1.0
    assert scores[1] == 1.0, f"Expected 1.0, got {scores[1]}"
```

**Step 2: Run test to verify it fails**

Run: `/opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/analysis/test_irreplaceability.py::test_irreplaceability_excludes_zero_target_features -v`
Expected: FAIL — score = 1/2 = 0.5 instead of 1.0

**Step 3: Fix irreplaceability denominator**

In `src/pymarxan/analysis/irreplaceability.py`, replace lines 30 and 50:

Replace:
```python
    n_features = problem.n_features
```

With:
```python
    n_positive_target = sum(
        1 for _, r in problem.features.iterrows() if float(r.get("target", 0.0)) > 0
    )
```

Replace:
```python
        scores[pid] = critical_count / n_features if n_features > 0 else 0.0
```

With:
```python
        scores[pid] = critical_count / n_positive_target if n_positive_target > 0 else 0.0
```

**Step 4: Run tests**

Run: `/opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/analysis/test_irreplaceability.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add src/pymarxan/analysis/irreplaceability.py tests/pymarxan/analysis/test_irreplaceability.py
git commit -m "fix(irreplaceability): use positive-target feature count as denominator"
```

---

### Task 5: C5 — Heuristic solver self-boundary

**Files:**
- Modify: `src/pymarxan/solvers/heuristic.py:255-262`
- Test: `tests/pymarxan/solvers/test_heuristic.py`

**Step 1: Write the failing test**

Add to `tests/pymarxan/solvers/test_heuristic.py`:

```python
def test_heuristic_includes_self_boundary():
    """Heuristic solver must include self-boundary in cost calculation."""
    import pandas as pd
    import numpy as np
    from pymarxan.models.problem import ConservationProblem
    from pymarxan.solvers.heuristic import HeuristicSolver
    from pymarxan.solvers.base import SolverConfig

    pu = pd.DataFrame({"id": [1, 2, 3], "cost": [1.0, 1.0, 1.0], "status": [0, 0, 0]})
    features = pd.DataFrame({"id": [1], "name": ["f1"], "target": [1.0], "spf": [1.0]})
    puvspr = pd.DataFrame({"species": [1, 1, 1], "pu": [1, 2, 3], "amount": [1.0, 1.0, 1.0]})
    boundary = pd.DataFrame({
        "id1": [1, 2, 3, 1, 2],
        "id2": [1, 2, 3, 2, 3],
        "boundary": [10.0, 10.0, 10.0, 1.0, 1.0],
    })
    problem = ConservationProblem(
        planning_units=pu, features=features, pu_vs_features=puvspr,
        boundary=boundary, parameters={"BLM": 1.0},
    )
    solver = HeuristicSolver(heurtype=0)
    sols = solver.solve(problem, SolverConfig(num_solutions=1, seed=42))
    sol = sols[0]
    # Each selected PU should contribute 10.0 self-boundary
    assert sol.boundary >= 10.0, f"Expected self-boundary >= 10, got {sol.boundary}"
```

**Step 2: Run test to verify it fails**

Run: `/opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/solvers/test_heuristic.py::test_heuristic_includes_self_boundary -v`
Expected: FAIL — boundary = 0 or very small (only cross-boundary counted)

**Step 3: Fix heuristic boundary loop**

In `src/pymarxan/solvers/heuristic.py`, replace lines 255-262:

```python
        boundary_val = 0.0
        if problem.boundary is not None and blm > 0:
            for _, row in problem.boundary.iterrows():
                i = pu_id_to_idx.get(int(row["id1"]))
                j = pu_id_to_idx.get(int(row["id2"]))
                if i is not None and j is not None:
                    if selected[i] != selected[j]:
                        boundary_val += float(row["boundary"])
```

With:

```python
        boundary_val = 0.0
        if problem.boundary is not None and blm > 0:
            for _, row in problem.boundary.iterrows():
                id1 = int(row["id1"])
                id2 = int(row["id2"])
                bval = float(row["boundary"])
                if id1 == id2:
                    idx = pu_id_to_idx.get(id1)
                    if idx is not None and selected[idx]:
                        boundary_val += bval
                else:
                    i = pu_id_to_idx.get(id1)
                    j = pu_id_to_idx.get(id2)
                    if i is not None and j is not None:
                        if selected[i] != selected[j]:
                            boundary_val += bval
```

**Step 4: Run tests**

Run: `/opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/solvers/test_heuristic.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add src/pymarxan/solvers/heuristic.py tests/pymarxan/solvers/test_heuristic.py
git commit -m "fix(heuristic): include self-boundary in boundary cost calculation"
```

---

## Batch 2 — HIGH Fixes (7 Tasks)

### Task 6: H1 — app.py _sync_solver_params must call problem.set()

**Files:**
- Modify: `src/pymarxan_app/app.py:188-214`

**Step 1: Fix the sync effect**

In `src/pymarxan_app/app.py`, replace the `_sync_solver_params` effect (lines 188-214):

```python
    @reactive.effect
    def _sync_solver_params():
        """Sync UI solver config values into problem.parameters."""
        p = problem()
        cfg = solver_config()
        if p is None:
            return
        import copy
        updated = copy.deepcopy(p)
        st = cfg.get("solver_type", "mip")
        if st == "mip":
            updated.parameters["MIP_TIME_LIMIT"] = str(
                cfg.get("mip_time_limit", 300)
            )
            updated.parameters["MIP_GAP"] = str(
                cfg.get("mip_gap", 0.0)
            )
        elif st == "greedy":
            updated.parameters["HEURTYPE"] = str(
                cfg.get("heurtype", 2)
            )
        elif st == "iterative_improvement":
            updated.parameters["ITIMPTYPE"] = str(
                cfg.get("itimptype", 0)
            )
        elif st == "pipeline":
            updated.parameters["RUNMODE"] = str(
                cfg.get("runmode", 0)
            )
        problem.set(updated)
```

Note: `import copy` is already used in run_panel. Add it at top of function to be safe.

**Step 2: Run full test suite to verify no regressions**

Run: `/opt/micromamba/envs/shiny/bin/pytest tests/ -x -q`
Expected: ALL PASS

**Step 3: Commit**

```bash
git add src/pymarxan_app/app.py
git commit -m "fix(app): deepcopy + problem.set() in _sync_solver_params"
```

---

### Task 7: H2 — Hex grid tessellation fix

**Files:**
- Modify: `src/pymarxan/spatial/grid.py:88-107`
- Test: `tests/pymarxan/spatial/test_grid.py`

**Step 1: Write the failing test**

Add to `tests/pymarxan/spatial/test_grid.py`:

```python
def test_hex_grid_tessellates():
    """Hex cells must share edges with neighbors (no gaps)."""
    from pymarxan.spatial.grid import generate_planning_grid
    from shapely.ops import unary_union

    grid = generate_planning_grid(
        bounds=(0, 0, 10, 10), cell_size=2.0, grid_type="hexagonal",
    )
    assert len(grid) > 4, "Need enough hexes to test tessellation"

    # Union of all hex polygons should have no internal gaps
    # Check: total area of union ≈ sum of individual areas (no overlaps)
    # AND each hex touches at least one neighbor (shares edge, not just point)
    geoms = grid.geometry.values
    found_shared_edge = False
    for i in range(len(geoms)):
        for j in range(i + 1, len(geoms)):
            intersection = geoms[i].intersection(geoms[j])
            if intersection.length > 1e-10:
                found_shared_edge = True
                break
        if found_shared_edge:
            break
    assert found_shared_edge, "No hex pair shares an edge — tessellation is broken"
```

**Step 2: Run test to verify it fails**

Run: `/opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/spatial/test_grid.py::test_hex_grid_tessellates -v`
Expected: FAIL — current hex spacing creates gaps

**Step 3: Fix hex grid generation**

In `src/pymarxan/spatial/grid.py`, replace `_generate_hex_cells` (lines 88-107):

```python
def _generate_hex_cells(
    bounds: tuple[float, float, float, float],
    cell_size: float,
) -> list[Polygon]:
    minx, miny, maxx, maxy = bounds
    # Flat-top hexagon: width = cell_size, height = cell_size * sqrt(3)/2
    w = cell_size
    h = cell_size * math.sqrt(3) / 2
    # Flat-top hex grids tessellate in columns, not rows
    col_step = 3 * w / 4  # horizontal distance between column centers
    cells: list[Polygon] = []
    col = 0
    x = minx
    while x < maxx - 1e-10:
        y_offset = (h / 2) if col % 2 == 1 else 0.0
        y = miny + y_offset
        while y < maxy - 1e-10:
            cells.append(_flat_top_hex(x, y, cell_size))
            y += h
        x += col_step
        col += 1
    return cells
```

**Step 4: Run tests**

Run: `/opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/spatial/test_grid.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add src/pymarxan/spatial/grid.py tests/pymarxan/spatial/test_grid.py
git commit -m "fix(grid): use column-based layout for flat-top hex tessellation"
```

---

### Task 8: H3 — Zone boundary costs stored symmetrically

**Files:**
- Modify: `src/pymarxan/zones/cache.py:169-179`
- Test: `tests/pymarxan/zones/test_zone_cache.py`

**Step 1: Write the failing test**

Add to `tests/pymarxan/zones/test_zone_cache.py`:

```python
def test_zone_boundary_costs_symmetric(cache):
    """Zone boundary costs must be stored in both directions."""
    for (z1, z2), cost in list(cache.zone_boundary_costs.items()):
        reverse = cache.zone_boundary_costs.get((z2, z1), None)
        assert reverse is not None, f"Missing reverse for ({z1},{z2})"
        assert reverse == cost, f"Asymmetric: ({z1},{z2})={cost}, ({z2},{z1})={reverse}"
```

**Step 2: Run test — may pass since test data has both directions**

Run: `/opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/zones/test_zone_cache.py::test_zone_boundary_costs_symmetric -v`
Expected: PASS (test data already symmetric). The fix is defensive for data that only has one direction.

**Step 3: Fix zone boundary cost population**

In `src/pymarxan/zones/cache.py`, replace lines 178-179:

```python
                if zcol1 is not None and zcol2 is not None:
                    zbc[(zcol1, zcol2)] = cost
```

With:

```python
                if zcol1 is not None and zcol2 is not None:
                    zbc[(zcol1, zcol2)] = cost
                    zbc[(zcol2, zcol1)] = cost
```

**Step 4: Run tests**

Run: `/opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/zones/ -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add src/pymarxan/zones/cache.py tests/pymarxan/zones/test_zone_cache.py
git commit -m "fix(zones): store zone boundary costs symmetrically"
```

---

### Task 9: H4 — write_sum Shortfall column

**Files:**
- Modify: `src/pymarxan/solvers/base.py:21-22`
- Modify: `src/pymarxan/solvers/utils.py:167-200`
- Modify: `src/pymarxan/solvers/heuristic.py:279-288`
- Modify: `src/pymarxan/io/writers.py:244-245`
- Test: `tests/pymarxan/io/test_output_writers.py`

**Step 1: Write the failing test**

Add to `tests/pymarxan/io/test_output_writers.py`:

```python
def test_write_sum_shortfall_differs_from_penalty():
    """Shortfall column must be raw shortfall, not SPF-weighted penalty."""
    import numpy as np
    from pymarxan.solvers.base import Solution

    sol = Solution(
        selected=np.array([True, False, True]),
        cost=40.0,
        boundary=5.0,
        objective=95.0,
        targets_met={1: True, 2: False},
        penalty=50.0,   # SPF=10 * shortfall=5 for feature 2
        shortfall=5.0,   # raw shortfall
    )
    path = tmp_path / "sum.csv"
    write_sum([sol], path)
    df = pd.read_csv(path)
    assert df.iloc[0]["Shortfall"] == pytest.approx(5.0)
    assert df.iloc[0]["Penalty"] == pytest.approx(50.0)
```

Note: This test uses `tmp_path` from pytest. Add it as a method parameter.

**Step 2: Run test to verify it fails**

Run: `/opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/io/test_output_writers.py::test_write_sum_shortfall_differs_from_penalty -v`
Expected: FAIL — `Solution` has no `shortfall` field yet

**Step 3: Add shortfall field and fix**

In `src/pymarxan/solvers/base.py`, add after line 21 (`penalty: float = 0.0`):

```python
    shortfall: float = 0.0  # Total raw feature shortfall (sum of max(0, target - achieved))
```

In `src/pymarxan/solvers/utils.py`, in `build_solution()`, add shortfall computation after line 181 (`penalty = compute_penalty(...)`):

```python
    shortfalls = compute_feature_shortfalls(problem, selected, pu_index)
    total_shortfall = sum(shortfalls.values())
```

And update the Solution constructor at line 192 to include `shortfall=total_shortfall`:

```python
    return Solution(
        selected=selected.copy(),
        cost=total_cost,
        boundary=total_boundary,
        objective=objective,
        targets_met=targets_met,
        penalty=penalty,
        shortfall=total_shortfall,
        metadata=metadata or {},
    )
```

In `src/pymarxan/solvers/heuristic.py`, add shortfall computation before line 281 (the Solution constructor):

```python
        total_shortfall = sum(max(r, 0.0) for r in remaining.values())
```

And add `shortfall=total_shortfall` to the Solution constructor call.

In `src/pymarxan/io/writers.py`, replace line 245:

```python
        shortfall = penalty  # total shortfall approximated from penalty
```

With:

```python
        shortfall = sol.shortfall
```

**Step 4: Run tests**

Run: `/opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/io/test_output_writers.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add src/pymarxan/solvers/base.py src/pymarxan/solvers/utils.py \
  src/pymarxan/solvers/heuristic.py src/pymarxan/io/writers.py \
  tests/pymarxan/io/test_output_writers.py
git commit -m "fix(writers): separate shortfall from penalty in write_sum output"
```

---

### Task 10: H5 — import_features_from_vector CRS guard

**Files:**
- Modify: `src/pymarxan/spatial/importers.py:90`
- Test: `tests/pymarxan/spatial/test_importers.py`

**Step 1: Write the failing test**

Add to `tests/pymarxan/spatial/test_importers.py`:

```python
def test_import_features_handles_no_crs_on_pus():
    """import_features_from_vector must not crash when PU GDF has no CRS."""
    import geopandas as gpd
    import pandas as pd
    from shapely.geometry import box
    from pymarxan.spatial.importers import import_features_from_vector

    pu_gdf = gpd.GeoDataFrame(
        {"id": [1, 2]},
        geometry=[box(0, 0, 1, 1), box(1, 0, 2, 1)],
        # No CRS set
    )
    # Write a temp feature file
    feat_gdf = gpd.GeoDataFrame(
        {"value": [10.0]},
        geometry=[box(0, 0, 1.5, 1)],
        # No CRS set
    )
    tmp_path = "/tmp/test_no_crs_features.geojson"
    feat_gdf.to_file(tmp_path, driver="GeoJSON")

    result = import_features_from_vector(
        tmp_path, pu_gdf, feature_name="test", feature_id=1,
    )
    assert isinstance(result, pd.DataFrame)
```

**Step 2: Run test to verify it fails**

Run: `/opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/spatial/test_importers.py::test_import_features_handles_no_crs_on_pus -v`
Expected: FAIL — `to_crs(None)` crashes

**Step 3: Fix CRS guard**

In `src/pymarxan/spatial/importers.py`, replace line 90:

```python
    if features_gdf.crs != planning_units.crs and features_gdf.crs is not None:
```

With:

```python
    if (features_gdf.crs is not None
            and planning_units.crs is not None
            and features_gdf.crs != planning_units.crs):
```

**Step 4: Run tests**

Run: `/opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/spatial/test_importers.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add src/pymarxan/spatial/importers.py tests/pymarxan/spatial/test_importers.py
git commit -m "fix(importers): guard CRS reprojection when PU GDF has no CRS"
```

---

### Task 11: H6 — O(n²) cost surface geometry lookup

**Files:**
- Modify: `src/pymarxan/spatial/cost_surface.py:50-65`
- Test: `tests/pymarxan/spatial/test_cost_surface.py`

**Step 1: Fix the O(n²) lookup**

In `src/pymarxan/spatial/cost_surface.py`, replace lines 50-65:

```python
    new_costs: dict[int, float] = {}
    pu_area_by_id = dict(zip(
        planning_units["id"].values,
        planning_units.geometry.area,
    ))
    for pu_id, group in overlay.groupby("id"):
        pu_area = pu_area_by_id.get(pu_id, 0.0)
        if aggregation == "area_weighted_mean":
            weighted = (group[cost_column] * group["_intersection_area"]).sum()
            total_area = group["_intersection_area"].sum()
            if total_area > 0 and total_area >= pu_area * 0.01:
                new_costs[pu_id] = weighted / total_area
        elif aggregation == "sum":
            new_costs[pu_id] = group[cost_column].sum()
        elif aggregation == "max":
            new_costs[pu_id] = group[cost_column].max()

    for pu_id, cost in new_costs.items():
        result.loc[result["id"] == pu_id, "cost"] = cost
```

**Step 2: Run existing tests to verify no regression**

Run: `/opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/spatial/test_cost_surface.py -v`
Expected: ALL PASS

**Step 3: Commit**

```bash
git add src/pymarxan/spatial/cost_surface.py
git commit -m "perf(cost_surface): O(1) PU area lookup instead of O(n) per group"
```

---

### Task 12: H7 — scenarios.py TYPE_CHECKING imports

**Files:**
- Modify: `src/pymarxan/analysis/scenarios.py:1-12,114-118`

**Step 1: Fix forward references**

In `src/pymarxan/analysis/scenarios.py`, add TYPE_CHECKING imports after line 11 (`from pymarxan.solvers.base import Solution`):

```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pymarxan.models.problem import ConservationProblem
    from pymarxan.solvers.base import Solver, SolverConfig
```

Then remove the `# noqa: F821` comments from lines 114, 115, 118:

```python
        problem: ConservationProblem,
        solver: Solver,
        ...
        config: SolverConfig | None = None,
```

**Step 2: Run tests**

Run: `/opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/analysis/test_scenarios.py tests/pymarxan/analysis/test_scenario_overrides.py -v`
Expected: ALL PASS

**Step 3: Run lint**

Run: `/home/razinka/.local/bin/ruff check src/pymarxan/analysis/scenarios.py`
Expected: No errors

**Step 4: Commit**

```bash
git add src/pymarxan/analysis/scenarios.py
git commit -m "fix(scenarios): add TYPE_CHECKING imports for forward-referenced types"
```

---

## Batch 3 — Test Gaps (6 Tasks)

### Task 13: T1+T2 — Missing Shiny module tests

**Files:**
- Create: `tests/pymarxan_shiny/test_upload.py`
- Create: `tests/pymarxan_shiny/test_summary_table.py`
- Create: `tests/pymarxan_shiny/test_export.py`
- Create: `tests/pymarxan_shiny/test_zone_config.py`
- Create: `tests/pymarxan_shiny/test_blm_explorer.py`

**Step 1: Write tests for upload module**

```python
"""Tests for data_input/upload Shiny module."""
from pymarxan_shiny.modules.data_input.upload import upload_server, upload_ui


def test_upload_ui_callable():
    assert callable(upload_ui)


def test_upload_server_callable():
    assert callable(upload_server)
```

**Step 2: Write tests for summary_table module**

```python
"""Tests for results/summary_table Shiny module."""
from pymarxan_shiny.modules.results.summary_table import summary_table_server, summary_table_ui


def test_summary_table_ui_callable():
    assert callable(summary_table_ui)


def test_summary_table_server_callable():
    assert callable(summary_table_server)
```

**Step 3: Write tests for export, zone_config, blm_explorer**

Same pattern as above for each module.

**Step 4: Run tests**

Run: `/opt/micromamba/envs/shiny/bin/pytest tests/pymarxan_shiny/test_upload.py tests/pymarxan_shiny/test_summary_table.py tests/pymarxan_shiny/test_export.py tests/pymarxan_shiny/test_zone_config.py tests/pymarxan_shiny/test_blm_explorer.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add tests/pymarxan_shiny/test_upload.py tests/pymarxan_shiny/test_summary_table.py \
  tests/pymarxan_shiny/test_export.py tests/pymarxan_shiny/test_zone_config.py \
  tests/pymarxan_shiny/test_blm_explorer.py
git commit -m "test: add smoke tests for upload, summary_table, export, zone_config, blm_explorer"
```

---

### Task 14: T4 — Zone penalty with unmet targets

**Files:**
- Test: `tests/pymarxan/zones/test_objective.py` (create if missing, or add to existing)

**Step 1: Write the test**

```python
"""Test compute_zone_penalty with unmet targets."""
import numpy as np
import pytest
from pathlib import Path
from pymarxan.zones.objective import compute_zone_penalty
from pymarxan.zones.readers import load_zone_project

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "zones"


def test_zone_penalty_with_unmet_targets():
    """All PUs unassigned => all zone targets unmet => positive penalty."""
    problem = load_zone_project(DATA_DIR)
    assignment = np.zeros(problem.n_planning_units, dtype=int)
    penalty = compute_zone_penalty(problem, assignment)
    assert penalty > 0.0, "Unmet zone targets should produce positive penalty"


def test_zone_penalty_zero_when_all_met():
    """All PUs in zone 1 with enough amount => penalty should be small or zero."""
    problem = load_zone_project(DATA_DIR)
    assignment = np.ones(problem.n_planning_units, dtype=int)
    penalty = compute_zone_penalty(problem, assignment)
    # With all PUs in zone 1, check if targets are met
    # Zone 1 targets: feature1=10, feature2=8
    # PU amounts: f1=[10,8,6,5]=29, f2=[5,7,9,4]=25; contrib=1.0
    # held: f1=29, f2=25 => both exceed targets => penalty=0
    assert penalty == pytest.approx(0.0)
```

**Step 2: Run tests**

Run: `/opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/zones/test_objective.py -v`
Expected: ALL PASS (tests verify existing behavior is correct)

**Step 3: Commit**

```bash
git add tests/pymarxan/zones/test_objective.py
git commit -m "test: add zone penalty tests with unmet and met targets"
```

---

### Task 15: T5+T6+T7 — Strengthen weak assertion tests

**Files:**
- Modify: `tests/pymarxan/analysis/test_scenarios.py` (T5: overlap_matrix)
- Modify: `tests/pymarxan/io/test_output_writers.py` (T6: write_sum)
- Modify: `tests/pymarxan/analysis/test_irreplaceability.py` (T7: numerical verification)

**Step 1: Strengthen overlap_matrix test (T5)**

Add to `tests/pymarxan/analysis/test_scenarios.py`:

```python
def test_overlap_matrix_partial():
    """Jaccard index with partially overlapping selections."""
    ss = ScenarioSet()
    # A: PUs 0-4 selected (5 of 10)
    sol_a = _make_solution(30.0, 5)
    # B: PUs 0-2 selected, 3-9 not (3 of 10)
    selected_b = np.zeros(10, dtype=bool)
    selected_b[:3] = True
    sol_b = Solution(
        selected=selected_b, cost=20.0, boundary=5.0,
        objective=25.0, targets_met={1: True, 2: True},
    )
    ss.add("a", sol_a, {})
    ss.add("b", sol_b, {})
    matrix = ss.overlap_matrix()
    # Intersection: 3 PUs (0,1,2), Union: 5 PUs (0,1,2,3,4)
    expected_jaccard = 3.0 / 5.0
    assert matrix[0, 1] == pytest.approx(expected_jaccard)
    assert matrix[1, 0] == pytest.approx(expected_jaccard)
```

**Step 2: Strengthen irreplaceability test (T7)**

Add to `tests/pymarxan/analysis/test_irreplaceability.py`:

```python
def test_irreplaceability_numerical_values():
    """Verify exact irreplaceability scores for a known problem."""
    import pandas as pd
    from pymarxan.models.problem import ConservationProblem
    from pymarxan.analysis.irreplaceability import compute_irreplaceability

    pu = pd.DataFrame({"id": [1, 2, 3], "cost": [1.0, 1.0, 1.0], "status": [0, 0, 0]})
    features = pd.DataFrame({
        "id": [1, 2], "name": ["f1", "f2"],
        "target": [8.0, 6.0], "spf": [1.0, 1.0],
    })
    puvspr = pd.DataFrame({
        "species": [1, 1, 2, 2],
        "pu": [1, 2, 1, 3],
        "amount": [5.0, 5.0, 4.0, 4.0],
    })
    problem = ConservationProblem(
        planning_units=pu, features=features, pu_vs_features=puvspr,
    )
    scores = compute_irreplaceability(problem)
    # Feature 1 total=10, target=8. Remove PU1: remaining=5 < 8 => critical
    # Feature 2 total=8, target=6. Remove PU1: remaining=4 < 6 => critical
    # PU1 critical for 2/2 features => score = 1.0
    assert scores[1] == pytest.approx(1.0)
    # PU2: feature 1 only. Remove PU2: remaining=5 < 8 => critical for f1
    # PU2 not in feature 2. Score = 1/2 = 0.5
    assert scores[2] == pytest.approx(0.5)
    # PU3: feature 2 only. Remove PU3: remaining=4 < 6 => critical for f2
    # Score = 1/2 = 0.5
    assert scores[3] == pytest.approx(0.5)
```

**Step 3: Run tests**

Run: `/opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/analysis/test_scenarios.py tests/pymarxan/analysis/test_irreplaceability.py tests/pymarxan/io/test_output_writers.py -v`
Expected: ALL PASS

**Step 4: Commit**

```bash
git add tests/pymarxan/analysis/test_scenarios.py \
  tests/pymarxan/analysis/test_irreplaceability.py \
  tests/pymarxan/io/test_output_writers.py
git commit -m "test: strengthen overlap_matrix, irreplaceability, and write_sum assertions"
```

---

### Task 16: T8 — mvbest write/read roundtrip

**Files:**
- Modify: `tests/pymarxan/io/test_output_writers.py`

**Step 1: Write the roundtrip test**

Add to `tests/pymarxan/io/test_output_writers.py`:

```python
from pymarxan.io.readers import read_mvbest


class TestMvbestRoundtrip:
    def test_write_then_read_preserves_data(
        self, tmp_path, simple_problem, solution_all_selected,
    ):
        """write_mvbest -> read_mvbest should preserve all columns."""
        path = tmp_path / "roundtrip_mvbest.csv"
        write_mvbest(simple_problem, solution_all_selected, path)
        df = read_mvbest(path)
        assert len(df) == 2
        assert "Feature_ID" in df.columns
        assert "Target" in df.columns
        assert "Amount_Held" in df.columns
        assert "Shortfall" in df.columns

    def test_roundtrip_values(
        self, tmp_path, simple_problem, solution_all_selected,
    ):
        """Roundtrip preserves numeric values."""
        path = tmp_path / "roundtrip_mvbest.csv"
        write_mvbest(simple_problem, solution_all_selected, path)
        df = read_mvbest(path)
        # Feature 1: target=15, held=18 (10+8), shortfall=0
        f1 = df[df["Feature_ID"] == 1].iloc[0]
        assert f1["Target"] == pytest.approx(15.0)
        assert f1["Amount_Held"] == pytest.approx(18.0)
        assert f1["Shortfall"] == pytest.approx(0.0)
```

**Step 2: Run tests**

Run: `/opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/io/test_output_writers.py -v`
Expected: ALL PASS

**Step 3: Commit**

```bash
git add tests/pymarxan/io/test_output_writers.py
git commit -m "test: add mvbest write/read roundtrip test"
```

---

### Task 17: Full regression + lint

**Step 1: Run full test suite**

Run: `/opt/micromamba/envs/shiny/bin/pytest tests/ -x --tb=short`
Expected: ALL PASS

**Step 2: Run ruff lint**

Run: `/home/razinka/.local/bin/ruff check src/`
Expected: No errors

**Step 3: Check coverage**

Run: `/opt/micromamba/envs/shiny/bin/pytest tests/ --cov=src --cov-fail-under=75 -q`
Expected: PASS with coverage ≥ 75%

**Step 4: Commit any remaining fixes**

If lint or coverage issues found, fix and commit.
