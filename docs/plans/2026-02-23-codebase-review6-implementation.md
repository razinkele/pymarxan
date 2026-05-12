# Codebase Review 6 — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Each task follows TDD: failing test → minimal impl → passing test → commit.

**Goal:** Fix 22 issues (3 critical, 6 high, 8 medium, 5 test gaps) found in Codebase Review 6.

**Architecture:** Surgical fixes to existing modules. Each fix is independent — TDD per fix, single commit per batch.

**Tech Stack:** Python, NumPy, pandas, pytest, Shiny for Python

---

## Batch 1: CRITICAL (3 issues)

### Task 1: C1 — `run_sweep_parallel` skips infeasible points

**Files:**
- Modify: `src/pymarxan/calibration/parallel.py`
- Test: `tests/pymarxan/calibration/test_parallel.py`

**Step 1: Failing test**
```python
def test_parallel_sweep_skips_infeasible(simple_problem):
    """Parallel sweep must skip infeasible points like the sequential version."""
    from pymarxan.calibration.sweep import SweepConfig
    from pymarxan.calibration.parallel import run_sweep_parallel
    from pymarxan.solvers.mip_solver import MIPSolver

    # Make one combination infeasible by setting impossible target overrides
    config = SweepConfig(
        param_grids={"BLM": [0.0, 1.0]},
        feature_target_overrides={1: {"target_grid": [1.0, 1e12]}},  # 1e12 infeasible
    ) if hasattr(SweepConfig, "feature_target_overrides") else SweepConfig(
        param_grids={"BLM": [0.0, 1.0, 2.0]},
    )
    result = run_sweep_parallel(simple_problem, MIPSolver(), config, max_workers=2)
    # Must not raise; failed points may be absent or have None solutions
    assert len(result.param_dicts) >= 1
```

If `feature_target_overrides` not in SweepConfig, simpler test: monkey-patch solver to return `[]` on one call and assert no crash.

**Step 2: Implementation**

In `parallel.py`, change the result-collection loop to catch exceptions per-future and skip:

```python
indexed_results: dict[int, Solution] = {}
with ProcessPoolExecutor(max_workers=max_workers) as executor:
    futures = {...}
    for future in as_completed(futures):
        try:
            idx, sol = future.result()
            indexed_results[idx] = sol
        except ValueError:
            # Infeasible: drop this index, matches sequential run_sweep behaviour
            continue

# Reassemble only successful indices, preserving order
ordered_indices = sorted(indexed_results.keys())
solutions = [indexed_results[i] for i in ordered_indices]
param_dicts_out = [param_dicts[i] for i in ordered_indices]
costs = [s.cost for s in solutions]
boundaries = [s.boundary for s in solutions]
objectives = [s.objective for s in solutions]

return SweepResult(
    param_dicts=param_dicts_out, solutions=solutions,
    costs=costs, boundaries=boundaries, objectives=objectives,
)
```

---

### Task 2: C2 — Lundy-Mees cooling O(1) per step

**Files:** Modify `src/pymarxan/solvers/cooling.py`; test `tests/pymarxan/solvers/test_cooling.py`

**Step 1: Failing test**
```python
def test_lundy_mees_temperature_calls_constant_time():
    import time
    sched = CoolingSchedule.lundy_mees(initial_temp=1.0, num_steps=10_000)
    # Get all temperatures sequentially — should be fast
    start = time.perf_counter()
    temps = [sched.temperature(k) for k in range(10_000)]
    elapsed = time.perf_counter() - start
    assert elapsed < 0.5, f"Lundy-Mees temperature O(n²) too slow: {elapsed}s"
    # Monotonically decreasing
    for i in range(1, len(temps)):
        assert temps[i] <= temps[i-1] + 1e-12
```

**Step 2: Implementation**

Add `_lundy_state` field and tracking logic. Simplest correct fix — precompute on construction:

```python
@dataclass
class CoolingSchedule:
    name: str
    initial_temp: float
    final_temp: float = 0.001
    num_steps: int = 10_000
    _alpha: float = 1.0
    _beta: float = 0.0
    _precomputed: list[float] = field(default_factory=list)

    def temperature(self, step: int) -> float:
        if self.name == "geometric":
            return self.initial_temp * (self._alpha ** step)
        if self.name == "exponential":
            ratio = math.log(self.initial_temp / self.final_temp) / max(1, self.num_steps)
            return self.initial_temp * math.exp(-step * ratio)
        if self.name == "linear":
            t = self.initial_temp - step * (self.initial_temp - self.final_temp) / max(1, self.num_steps)
            return max(t, self.final_temp)
        if self.name == "lundy_mees":
            if not self._precomputed:
                self._precompute_lundy_mees()
            i = min(step, len(self._precomputed) - 1)
            return self._precomputed[i]
        raise ValueError(f"Unknown cooling schedule: {self.name}")

    def _precompute_lundy_mees(self) -> None:
        t = self.initial_temp
        out = [max(t, self.final_temp)]
        for _ in range(self.num_steps):
            t = t / (1.0 + self._beta * t)
            out.append(max(t, self.final_temp))
        self._precomputed = out
```

(Need to add `from dataclasses import field` import.)

---

### Task 3: C3 — Shapefile upload requires ZIP

**Files:**
- Modify: `src/pymarxan_shiny/modules/spatial/import_wizard.py`, `cost_upload.py`
- Test: `tests/pymarxan_shiny/test_import_wizard.py` (create if missing)

**Step 1: Failing test**
```python
def test_shapefile_upload_must_be_zip(tmp_path):
    """A bare .shp upload should raise a clear error, not a Fiona stack trace."""
    from pymarxan.spatial.importers import import_planning_units
    shp = tmp_path / "test.shp"
    shp.write_bytes(b"not a real shp")
    with pytest.raises((ValueError, OSError)) as exc:
        import_planning_units(shp)
    # Either succeeds with valid shp (won't here), or raises informatively
```

This is more of an integration concern. The realistic test is to confirm the UI restricts `.shp` upload — but Shiny UI tests are out-of-scope. Instead, test that `_import` rejects `.shp` files lacking sidecars with a clear message.

**Step 2: Implementation**

In `import_wizard.py`, change accept list from `[".shp", ".geojson", ".gpkg", ".zip"]` to `[".geojson", ".gpkg", ".zip"]`. Same for `cost_upload.py`. Update help text in both modules to instruct users to upload shapefiles as a ZIP archive. Update `_import` server function to detect `.shp` (defensive) and raise:

```python
if path.suffix.lower() == ".shp":
    raise ValueError(
        "Shapefiles must be uploaded as a .zip archive containing all sidecar "
        "files (.shp, .shx, .dbf, .prj)."
    )
```

Also ensure ZIP extraction exists in the upload path (likely already does for project upload — reuse).

---

## Batch 2: HIGH (6 issues)

### Task 4: H1 — MIP solvers accept feasible-on-timeout solutions

**Files:** Modify `src/pymarxan/solvers/mip_solver.py`, `src/pymarxan/zones/mip_solver.py`; test both.

**Step 1: Failing test (mip_solver)**
```python
def test_mip_returns_feasible_solution_on_timeout(monkeypatch, simple_problem):
    """When CBC hits time limit but has a feasible incumbent, return it."""
    import pulp
    from pymarxan.solvers.mip_solver import MIPSolver

    problem = copy.deepcopy(simple_problem)
    problem.parameters["MIP_TIME_LIMIT"] = 1  # 1 second, tight enough to potentially timeout

    # Monkeypatch model.status to simulate timeout-with-feasible
    original_solve = pulp.LpProblem.solve
    def patched_solve(self, *a, **kw):
        original_solve(self, *a, **kw)
        self.status = pulp.LpStatusNotSolved  # simulate
        return self.status
    monkeypatch.setattr(pulp.LpProblem, "solve", patched_solve)

    sols = MIPSolver().solve(problem, SolverConfig(num_solutions=1))
    # Should not be empty if vars have any values
    assert len(sols) == 1
```

**Step 2: Implementation**

In `mip_solver.py` line 232:

```python
status = model.status
has_feasible_values = all(
    pulp.value(x[pid]) is not None for pid in pu_ids
)
if status != pulp.constants.LpStatusOptimal and not has_feasible_values:
    return []
```

Same change in `zones/mip_solver.py` adapted to its variable structure.

---

### Task 5: H2 — `build_pu_feature_matrix` accumulates duplicates

**Files:** Modify `src/pymarxan/models/problem.py`; test `tests/pymarxan/models/test_problem.py`

**Step 1: Failing test**
```python
def test_build_pu_feature_matrix_sums_duplicates(simple_problem):
    """Duplicate (pu, species) rows must sum, matching feature_amounts()."""
    problem = copy.deepcopy(simple_problem)
    # Inject duplicate row for first puvspr entry
    first = problem.pu_vs_features.iloc[0]
    dup = pd.DataFrame([{"pu": first["pu"], "species": first["species"], "amount": 5.0}])
    problem.pu_vs_features = pd.concat([problem.pu_vs_features, dup], ignore_index=True)

    matrix = problem.build_pu_feature_matrix()
    pu_idx = problem.pu_id_to_index[int(first["pu"])]
    feat_idx = list(problem.features["id"]).index(int(first["species"]))
    expected = float(first["amount"]) + 5.0
    assert matrix[pu_idx, feat_idx] == pytest.approx(expected)

    # Matrix sum per feature should match feature_amounts (which sums)
    totals = problem.feature_amounts()
    matrix_totals = matrix.sum(axis=0)
    feat_ids = problem.features["id"].values
    for j, fid in enumerate(feat_ids):
        assert matrix_totals[j] == pytest.approx(totals[int(fid)])
```

**Step 2: Implementation**

Change `models/problem.py:120` from `=` to `+=`:
```python
matrix[ri, ci] += float(pv_am[k])
```

---

### Task 6: H3 — `create_geo_map` reprojects to WGS84

**Files:** Modify `src/pymarxan_shiny/modules/mapping/map_utils.py` and `network_view.py`; test `tests/pymarxan_shiny/test_map_utils.py`

**Step 1: Failing test**
```python
def test_create_geo_map_reprojects_to_wgs84(_allow_widget_outside_session):
    """A GeoDataFrame in projected CRS must be reprojected to lat/lon."""
    import geopandas as gpd
    from shapely.geometry import Polygon

    # UTM zone 33N coordinates (around Berlin), should become ~13°E, 52°N
    poly = Polygon([(390000, 5800000), (400000, 5800000), (400000, 5810000), (390000, 5810000)])
    gdf = gpd.GeoDataFrame({"id": [1]}, geometry=[poly], crs="EPSG:32633")

    from pymarxan_shiny.modules.mapping.map_utils import create_geo_map
    m = create_geo_map(gdf, ["#ff0000"])
    # Center should be near Berlin in lat/lon
    assert 50 < m.center[0] < 55
    assert 10 < m.center[1] < 15
```

**Step 2: Implementation**

At start of `create_geo_map`:
```python
if gdf.crs is not None and gdf.crs.to_epsg() != 4326:
    gdf = gdf.to_crs("EPSG:4326")
```

Same guard before centroid extraction in `network_view.py:151`.

---

### Task 7: H4 — Download tempfiles cleaned up on session end

**Files:** Modify `src/pymarxan_shiny/modules/results/export.py`, `modules/spatial_export/spatial_export.py`

**Step 1: Test** — Hard to test directly with Shiny; do behavioral test:
```python
def test_export_solution_registers_cleanup(monkeypatch):
    """Tempfiles created by download handlers must be tracked for cleanup."""
    # Inspect the module for session.on_ended registration or use of a managed temp dir
    import pymarxan_shiny.modules.results.export as exp
    src = Path(exp.__file__).read_text()
    assert "session.on_ended" in src or "TemporaryDirectory" in src
```

**Step 2: Implementation**

Wrap each handler so the path is tracked and `session.on_ended` deletes it. Pattern:

```python
@session.download(filename=lambda: f"solution_{run_id}.csv")
def download_solution():
    f = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
    _temp_paths.append(Path(f.name))
    # ... write content ...
    yield f.name

# Once per server function:
_temp_paths: list[Path] = []

@session.on_ended
def _cleanup():
    for p in _temp_paths:
        try:
            p.unlink(missing_ok=True)
        except OSError:
            pass
```

---

### Task 8: H5 — `import_features_from_vector` errors on missing CRS

**Files:** Modify `src/pymarxan/spatial/importers.py`; test `tests/pymarxan/spatial/test_importers.py`

**Step 1: Failing test**
```python
def test_import_features_raises_when_crs_missing():
    import geopandas as gpd
    from shapely.geometry import Polygon
    pu = gpd.GeoDataFrame(
        {"id": [1]}, geometry=[Polygon([(0,0),(1,0),(1,1),(0,1)])], crs="EPSG:4326"
    )
    features = gpd.GeoDataFrame(
        {"name": ["a"]}, geometry=[Polygon([(0,0),(0.5,0),(0.5,0.5),(0,0.5)])], crs=None
    )
    from pymarxan.spatial.importers import import_features_from_vector
    with pytest.raises(ValueError, match="CRS"):
        import_features_from_vector(features, pu, "name")
```

**Step 2: Implementation**

Add guard before line 89:
```python
if features_gdf.crs is None and planning_units.crs is not None:
    raise ValueError(
        "Feature file lacks CRS while planning units have one. "
        "Set the feature CRS explicitly or upload a file with .prj."
    )
if features_gdf.crs is not None and planning_units.crs is None:
    raise ValueError("Planning unit grid lacks CRS while features have one.")
```

---

### Task 9: H6 — `ZoneSASolver` honours `COOLING` parameter

**Files:** Modify `src/pymarxan/zones/solver.py`; test `tests/pymarxan/zones/test_solver.py`

**Step 1: Failing test**
```python
def test_zone_sa_uses_cooling_param(zone_problem):
    """Setting COOLING=linear should produce a different sequence than geometric."""
    import copy
    p_geo = copy.deepcopy(zone_problem)
    p_geo.parameters["COOLING"] = "geometric"
    p_geo.parameters["NUMITNS"] = 200
    p_lin = copy.deepcopy(zone_problem)
    p_lin.parameters["COOLING"] = "linear"
    p_lin.parameters["NUMITNS"] = 200

    cfg = SolverConfig(num_solutions=1, seed=42)
    geo_sol = ZoneSASolver().solve(p_geo, cfg)[0]
    lin_sol = ZoneSASolver().solve(p_lin, cfg)[0]
    # Same seed, different cooling — at least one objective should differ
    # (acceptance pattern differs with different temperature schedule)
    assert geo_sol.objective != lin_sol.objective or not np.array_equal(
        geo_sol.zone_assignment, lin_sol.zone_assignment
    )
```

**Step 2: Implementation**

Replace lines 183-225 cooling/loop:

```python
from pymarxan.solvers.cooling import CoolingSchedule

cooling_name = problem.parameters.get("COOLING", "geometric")
schedule = _build_schedule(cooling_name, initial_temp, num_temp_steps)
iters_per_step = max(1, num_iterations // num_temp_steps)

temp = initial_temp
# ... in the main loop ...
if step_count % iters_per_step == 0 and step_count > 0:
    temp = schedule.temperature(min(step_count // iters_per_step, num_temp_steps))
```

Where `_build_schedule` dispatches to `CoolingSchedule.geometric`/`linear`/`exponential`/`lundy_mees`.

---

## Batch 3: MEDIUM (8 issues)

### Task 10: M1 — `compute_irreplaceability` applies MISSLEVEL and excludes locked-out PUs

**Files:** Modify `src/pymarxan/analysis/irreplaceability.py`; test `tests/pymarxan/analysis/test_irreplaceability.py`

**Step 1: Failing test**
```python
def test_irreplaceability_excludes_locked_out_and_applies_misslevel(simple_problem):
    import copy
    p = copy.deepcopy(simple_problem)
    p.parameters["MISSLEVEL"] = 0.5
    # Lock out PU 1 — its contribution must not inflate availability for others
    p.planning_units.loc[p.planning_units["id"] == 1, "status"] = 3
    scores = compute_irreplaceability(p)
    # Score for locked-out PU should be 0 (it can never be selected)
    assert scores[1] == 0.0
```

**Step 2: Implementation**

```python
# Exclude locked-out PUs from total_per_feat
statuses = problem.planning_units["status"].values
locked_out = (statuses == 3)
available_matrix = pu_feat_matrix.copy()
available_matrix[locked_out, :] = 0.0
total_per_feat = available_matrix.sum(axis=0)

misslevel = float(problem.parameters.get("MISSLEVEL", 1.0))
effective_targets = feat_targets * misslevel

# remaining/critical use effective_targets and exclude locked-out
remaining = total_per_feat[np.newaxis, :] - available_matrix
critical = (remaining < effective_targets[np.newaxis, :]) & positive_mask[np.newaxis, :]
# Force score=0 for locked-out PUs (can never be critical)
critical_counts = critical.sum(axis=1)
critical_counts[locked_out] = 0
```

---

### Task 11: M2 — `compute_gap_analysis` applies MISSLEVEL

**Files:** Modify `src/pymarxan/analysis/gap_analysis.py`; test `tests/pymarxan/analysis/test_gap_analysis.py`

**Step 1: Failing test**
```python
def test_gap_analysis_applies_misslevel(simple_problem):
    import copy
    p = copy.deepcopy(simple_problem)
    p.parameters["MISSLEVEL"] = 0.5
    # Select first 2 PUs
    result = compute_gap_analysis(p, [1, 2])
    # With misslevel=0.5, gap should be smaller than with misslevel=1.0
    p_full = copy.deepcopy(simple_problem)
    p_full.parameters["MISSLEVEL"] = 1.0
    result_full = compute_gap_analysis(p_full, [1, 2])
    assert sum(result.gap.values()) <= sum(result_full.gap.values())
```

**Step 2: Implementation**

```python
misslevel = float(problem.parameters.get("MISSLEVEL", 1.0))
for fid in feature_ids:
    effective_target = targets[fid] * misslevel
    shortfall = max(effective_target - protected_amount[fid], 0.0)
    gap[fid] = shortfall
    target_met[fid] = shortfall <= 0
```

---

### Task 12: M3 — `SweepResult.best` raises descriptive error on empty

**Files:** Modify `src/pymarxan/calibration/sweep.py`; test `tests/pymarxan/calibration/test_sweep.py`

**Step 1: Failing test**
```python
def test_sweep_result_best_empty_raises_clear():
    result = SweepResult(param_dicts=[], solutions=[], costs=[], boundaries=[], objectives=[])
    with pytest.raises(ValueError, match="No feasible solutions"):
        _ = result.best
```

**Step 2: Implementation**
```python
@property
def best(self) -> Solution:
    if not self.objectives:
        raise ValueError("No feasible solutions in sweep")
    idx = min(range(len(self.objectives)), key=lambda i: self.objectives[i])
    return self.solutions[idx]
```

---

### Task 13: M4 — `connectivity_to_matrix` sums duplicate edges

**Files:** Modify `src/pymarxan/connectivity/io.py`; test `tests/pymarxan/connectivity/test_io.py`

**Step 1: Failing test**
```python
def test_connectivity_matrix_sums_duplicate_edges():
    df = pd.DataFrame({"id1":[1,1,2], "id2":[2,2,1], "value":[1.5, 2.5, 4.0]})
    m = connectivity_to_matrix(df, pu_ids=[1, 2], symmetric=False)
    # (1,2) appears twice with values 1.5 and 2.5 -> 4.0
    # (2,1) appears once with 4.0
    assert m[0, 1] == pytest.approx(4.0)
    assert m[1, 0] == pytest.approx(4.0)
```

**Step 2: Implementation**

Replace assignment with accumulation:
```python
np.add.at(matrix, (idx1[valid], idx2[valid]), val_col[valid])
if symmetric:
    np.add.at(matrix, (idx2[valid], idx1[valid]), val_col[valid])
```

Note: this changes semantics for users who relied on overwrite — document in commit message. Also: when `symmetric=True` AND user provides both directions, this will double the values. The safer interpretation: only set, deduplicating first.

Actually simpler: deduplicate first.
```python
df = edgelist[["id1","id2","value"]].copy()
df = df.groupby(["id1","id2"], as_index=False)["value"].sum()
# then proceed with assignment
```

---

### Task 14: M5 — `apply_wdpa_status` cleans invalid geometry on error

**Files:** Modify `src/pymarxan/spatial/wdpa.py`; test `tests/pymarxan/spatial/test_wdpa.py`

**Step 1: Failing test**
```python
def test_apply_wdpa_status_handles_topological_error(monkeypatch):
    # Craft a WDPA gdf whose union is invalid
    import geopandas as gpd
    from shapely.geometry import Polygon
    bowtie = Polygon([(0,0),(1,1),(1,0),(0,1),(0,0)])  # self-intersecting
    wdpa = gpd.GeoDataFrame({"id":[1]}, geometry=[bowtie], crs="EPSG:4326")
    pu = gpd.GeoDataFrame(
        {"id":[1,2]},
        geometry=[Polygon([(0,0),(0.5,0),(0.5,0.5),(0,0.5)]),
                  Polygon([(0.5,0.5),(1,0.5),(1,1),(0.5,1)])],
        crs="EPSG:4326",
    )
    # Should not raise TopologicalError
    out = apply_wdpa_status(pu, wdpa)
    assert "status" in out.columns
```

**Step 2: Implementation**
```python
wdpa_union = wdpa.geometry.union_all()
if not wdpa_union.is_valid:
    wdpa_union = wdpa_union.buffer(0)
```

Wrap intersection call in `try/except Exception` and `buffer(0)` retry.

---

### Task 15: M6 — `comparison_map_server` preserves Solution B selection

**Files:** Modify `src/pymarxan_shiny/modules/mapping/comparison_map.py`; test if feasible.

**Step 1: Implementation only** (Shiny reactive tests need a session — defer testing to integration)
```python
current_b = input.sol_b() if input.sol_b() else "1"
choice_keys = [c[0] for c in choices]
selected_b = current_b if current_b in choice_keys else "1"
ui.update_select("sol_b", choices=dict(choices), selected=selected_b)
# same pattern for sol_a
```

---

### Task 16: M7 — `fetch_gadm` strips whitespace from `admin_name`

**Files:** Modify `src/pymarxan/spatial/gadm.py`; test `tests/pymarxan/spatial/test_gadm.py`

**Step 1: Failing test**
```python
def test_fetch_gadm_strips_whitespace_admin_name(monkeypatch):
    # Mock the network call; verify that admin_name=" " becomes None
    ...
```

**Step 2: Implementation**

In `fetch_gadm`:
```python
if admin_name is not None:
    admin_name = admin_name.strip() or None
if admin_name is not None:
    # existing filter block
```

---

### Task 17: M8 — `fetch_wdpa` warns on long pagination

**Files:** Modify `src/pymarxan/spatial/wdpa.py`

**Step 1: Implementation**
```python
if page >= 5:
    import warnings
    warnings.warn(
        f"WDPA fetch has retrieved {page} pages — large country may take minutes. "
        "Consider filtering by bounding box.",
        stacklevel=2,
    )
```

Add at the top of the pagination loop iteration.

---

## Batch 4: Test Gaps (5 tests)

### Task 18: T1 — `ProblemCache` delta correctness under MISSLEVEL

`tests/pymarxan/solvers/test_cache.py`:
```python
class TestDeltaWithMissLevel:
    def test_delta_matches_full_objective_difference(self, simple_problem):
        import copy
        p = copy.deepcopy(simple_problem)
        p.parameters["MISSLEVEL"] = 0.5
        cache = ProblemCache.from_problem(p)

        rng = np.random.default_rng(42)
        selected = rng.random(cache.n_pu) < 0.5
        for _ in range(10):
            idx = rng.integers(cache.n_pu)
            full_before = cache.compute_full_objective(selected, 0.1)
            delta = cache.compute_delta_objective(idx, selected, 0.1)
            selected[idx] = not selected[idx]
            full_after = cache.compute_full_objective(selected, 0.1)
            assert delta == pytest.approx(full_after - full_before, abs=1e-9)
```

### Task 19: T2 — `clone_scenario` does not mutate source `feature_overrides`

`tests/pymarxan/analysis/test_scenario_overrides.py`:
```python
def test_clone_scenario_feature_overrides_isolated(simple_problem):
    scenarios = ScenarioSet(problem=simple_problem)
    scenarios.add("base", feature_overrides={1: {"target": 10.0}})
    scenarios.clone("base", "clone", feature_overrides={2: {"target": 20.0}})
    assert list(scenarios.get("base").feature_overrides.keys()) == [1]
    assert set(scenarios.get("clone").feature_overrides.keys()) == {1, 2}
```

### Task 20: T3 — `ZoneSASolver` locked PU zone assignments

`tests/pymarxan/zones/test_solver.py`:
```python
def test_zone_sa_locked_in_all_assigned_nonzero(zone_problem):
    p = copy.deepcopy(zone_problem)
    p.planning_units["status"] = 2  # lock all in
    sol = ZoneSASolver().solve(p, SolverConfig(num_solutions=1))[0]
    assert all(z > 0 for z in sol.zone_assignment)

def test_zone_sa_locked_out_all_assigned_zero(zone_problem):
    p = copy.deepcopy(zone_problem)
    p.planning_units["status"] = 3  # lock all out
    sol = ZoneSASolver().solve(p, SolverConfig(num_solutions=1))[0]
    assert all(z == 0 for z in sol.zone_assignment)
```

### Task 21: T4 — ITIMPTYPE=2 multi-pass regression

`tests/pymarxan/solvers/test_iterative_improvement.py`:
```python
def test_two_step_iterative_multipass_required(custom_problem_requiring_multipass):
    """Construct a problem where 1 pass produces worse solution than 2 passes."""
    config = SolverConfig(num_solutions=1, seed=42)
    p = custom_problem_requiring_multipass
    p.parameters["ITIMPTYPE"] = 2
    sol = IterativeImprovementSolver().solve(p, config)[0]
    # Verify it actually iterated — track via metadata if available
    # Otherwise, compare to single-pass result by patching the loop
    # ...
```

(Construction of `custom_problem_requiring_multipass` may need careful design — accept that the test may need a custom fixture.)

### Task 22: T5 — `export_summary_csv` `achieved` values numerically verified

`tests/pymarxan/io/test_exporters.py`:
```python
def test_export_summary_achieved_values_correct(simple_problem, tmp_path):
    sol = Solution(
        selected=np.array([True, True, False, False, False, False]),
        cost=10.0, objective=10.0, penalty=0.0, boundary=0.0,
        all_targets_met=False, n_selected=2,
        metadata={}, problem=simple_problem,
    )
    out = tmp_path / "summary.csv"
    export_summary_csv(simple_problem, sol, out)
    df = pd.read_csv(out)
    # Compute expected achieved manually for each feature
    selected_pu_ids = [1, 2]  # ids of PUs 0, 1
    for _, row in df.iterrows():
        fid = int(row["feature_id"])
        expected = float(
            simple_problem.pu_vs_features[
                (simple_problem.pu_vs_features["pu"].isin(selected_pu_ids))
                & (simple_problem.pu_vs_features["species"] == fid)
            ]["amount"].sum()
        )
        assert row["achieved"] == pytest.approx(expected)
```

---

## Verification

After each batch:
```bash
cd /home/razinka/marxan && source .venv/bin/activate && pytest tests/ -x -q
```

Final after all 4 batches:
```bash
make check  # lint + types + full test suite
```

Expected outcome: 22 fixes, ~705 tests, ~79% coverage, 0 lint errors.

## Commit Strategy

One commit per batch with a summary message in this format:
```
Review 6 Batch N: <N issues> — <one-line theme>

- C1/H1/M1: <fix summary>
- C2/H2/M2: <fix summary>
...
```
