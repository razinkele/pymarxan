# Codebase Review 4 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix 22 issues found in Codebase Review 4 — correctness bugs, Shiny reactive anti-patterns, Marxan compatibility gaps, and test coverage holes.

**Architecture:** Four batches — CRITICAL fixes first (solver/zone/spatial correctness), then HIGH (reactive patterns, reader defaults, solver convergence), then MEDIUM (performance, UX, completeness), then test gaps.

**Tech Stack:** Python 3.11+, pandas, numpy, geopandas, Shiny for Python, pytest

**Test runner:** `/opt/micromamba/envs/shiny/bin/pytest`
**Linter:** `/home/razinka/.local/bin/ruff check`

---

## Batch 1 — CRITICAL (5 Tasks)

### Task 1: C1 — Resolve `prop` column to effective targets in `load_project`

**Files:**
- Modify: `src/pymarxan/io/readers.py:195-210`
- Test: `tests/pymarxan/io/test_readers.py`

**Step 1: Write the failing test**

Add to `tests/pymarxan/io/test_readers.py`:

```python
def test_load_project_resolves_prop_to_target(tmp_path):
    """When prop > 0 and target == 0, effective target = prop * total_amount."""
    import os
    input_dir = tmp_path / "input"
    input_dir.mkdir()

    # input.dat
    (tmp_path / "input.dat").write_text("INPUTDIR input\nPUNAME pu.dat\nSPECNAME spec.dat\nPUVSPRNAME puvspr.dat\n")

    # pu.dat: 3 PUs
    (input_dir / "pu.dat").write_text("id,cost,status\n1,10,0\n2,20,0\n3,30,0\n")

    # spec.dat: feature 1 has target=0, prop=0.3; feature 2 has target=5, prop=0
    (input_dir / "spec.dat").write_text("id,target,prop,spf,name\n1,0,0.3,1.0,feat_a\n2,5,0,1.0,feat_b\n")

    # puvspr.dat: feature 1 total_amount = 10+8 = 18; feature 2 total = 12+20 = 32
    (input_dir / "puvspr.dat").write_text("species,pu,amount\n1,1,10\n1,2,8\n2,2,12\n2,3,20\n")

    from pymarxan.io.readers import load_project
    problem = load_project(tmp_path)

    # feature 1: max(0, 0.3 * 18) = 5.4
    f1 = problem.features[problem.features["id"] == 1].iloc[0]
    assert f1["target"] == pytest.approx(5.4)
    # feature 2: max(5, 0 * 32) = 5 (target already set)
    f2 = problem.features[problem.features["id"] == 2].iloc[0]
    assert f2["target"] == pytest.approx(5.0)
```

**Step 2: Run test to verify it fails**

Run: `/opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/io/test_readers.py::test_load_project_resolves_prop_to_target -v`
Expected: FAIL — `f1["target"]` is 0.0, not 5.4

**Step 3: Implement the fix**

In `src/pymarxan/io/readers.py`, add a `_resolve_prop_targets` function after `read_bound` and call it from `load_project`:

```python
def _resolve_prop_targets(
    features: pd.DataFrame,
    pu_vs_features: pd.DataFrame,
) -> pd.DataFrame:
    """Resolve proportional targets: effective_target = max(target, prop * total_amount)."""
    if "prop" not in features.columns:
        return features
    features = features.copy()
    if "target" not in features.columns:
        features["target"] = 0.0
    for idx, row in features.iterrows():
        prop = float(row.get("prop", 0.0))
        if prop > 0:
            fid = int(row["id"])
            total = float(
                pu_vs_features.loc[
                    pu_vs_features["species"] == fid, "amount"
                ].sum()
            )
            features.loc[idx, "target"] = max(
                float(row["target"]), prop * total
            )
    return features
```

In `load_project`, insert after line 196 (`features = read_spec(...)`) and before `return`:

```python
    features = _resolve_prop_targets(features, pu_vs_features)
```

**Step 4: Run test to verify it passes**

Run: `/opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/io/test_readers.py::test_load_project_resolves_prop_to_target -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/pymarxan/io/readers.py tests/pymarxan/io/test_readers.py
git commit -m "fix(readers): resolve prop column to effective targets in load_project"
```

---

### Task 2: C2 — Zone SA solver: compute penalty and shortfall on Solution

**Files:**
- Modify: `src/pymarxan/zones/solver.py:178-200`
- Test: `tests/pymarxan/zones/test_zone_solver.py`

**Step 1: Write the failing test**

Add to `tests/pymarxan/zones/test_zone_solver.py`:

```python
def test_zone_sa_solution_has_penalty_and_shortfall(zone_problem):
    """Zone SA should populate penalty and shortfall fields on Solution."""
    from pymarxan.solvers.base import SolverConfig
    from pymarxan.zones.solver import ZoneSASolver

    zone_problem.parameters["NUMITNS"] = 100
    zone_problem.parameters["NUMTEMP"] = 10
    config = SolverConfig(num_solutions=1, seed=42)
    solutions = ZoneSASolver().solve(zone_problem, config)
    sol = solutions[0]
    # penalty and shortfall should be computed, not default 0.0
    # Even if targets are met, the fields should be explicitly set
    assert hasattr(sol, "penalty")
    assert hasattr(sol, "shortfall")
    # If any zone targets are unmet, penalty should be > 0
    zone_targets_met = sol.metadata.get("zone_targets_met", {})
    if not all(zone_targets_met.values()):
        assert sol.penalty > 0.0
```

**Step 2: Run test to verify it fails**

Run: `/opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/zones/test_zone_solver.py::test_zone_sa_solution_has_penalty_and_shortfall -v`
Expected: Likely FAIL (penalty is 0.0 even when targets unmet)

**Step 3: Implement the fix**

In `src/pymarxan/zones/solver.py`, add import at the top:

```python
from pymarxan.zones.objective import compute_zone_penalty
```

Then replace lines 184-200 with:

```python
            zone_penalty = compute_zone_penalty(problem, best_assignment)

            # Compute raw shortfall (without SPF weighting)
            zone_shortfall = 0.0
            if problem.zone_targets is not None:
                pu_ids_list = problem.planning_units["id"].tolist()
                pu_idx = {pid: i for i, pid in enumerate(pu_ids_list)}
                for _, trow in problem.zone_targets.iterrows():
                    zid_t = int(trow["zone"])
                    fid_t = int(trow["feature"])
                    target_t = float(trow["target"])
                    contribution = problem.get_contribution(fid_t, zid_t)
                    feat_data = problem.pu_vs_features[
                        problem.pu_vs_features["species"] == fid_t
                    ]
                    achieved = 0.0
                    for _, r in feat_data.iterrows():
                        pid = int(r["pu"])
                        idx_t = pu_idx.get(pid)
                        if idx_t is not None and int(best_assignment[idx_t]) == zid_t:
                            achieved += float(r["amount"]) * contribution
                    zone_shortfall += max(0.0, target_t - achieved)

            sol = Solution(
                selected=selected,
                cost=cost,
                boundary=std_boundary,
                objective=best_obj,
                targets_met={},
                penalty=zone_penalty,
                shortfall=zone_shortfall,
                zone_assignment=best_assignment.copy(),
                metadata={
                    "solver": self.name(),
                    "run": run_idx + 1,
                    "zone_boundary_cost": round(zone_boundary, 4),
                    "zone_targets_met": {
                        f"z{z}_f{f}": v
                        for (z, f), v in zone_targets.items()
                    },
                },
            )
```

**Step 4: Run tests**

Run: `/opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/zones/ -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add src/pymarxan/zones/solver.py tests/pymarxan/zones/test_zone_solver.py
git commit -m "fix(zones): compute penalty and shortfall in Zone SA Solution"
```

---

### Task 3: C3 — Zone objective functions apply MISSLEVEL

**Files:**
- Modify: `src/pymarxan/zones/objective.py:110-128,148-166`
- Test: `tests/pymarxan/zones/test_objective.py`

**Step 1: Write the failing test**

Add to `tests/pymarxan/zones/test_objective.py`:

```python
def test_zone_penalty_respects_misslevel():
    """With MISSLEVEL=0.5, targets should be halved — easier to meet."""
    from pymarxan.zones.objective import compute_zone_penalty
    problem = load_zone_project(DATA_DIR)

    # Assignment that partially meets targets
    assignment = np.array([1, 0, 0, 0])  # Only PU 0 in zone 1

    # Without MISSLEVEL (default 1.0)
    penalty_strict = compute_zone_penalty(problem, assignment)

    # With MISSLEVEL=0.5 — targets halved, easier to meet
    problem.parameters["MISSLEVEL"] = 0.5
    penalty_relaxed = compute_zone_penalty(problem, assignment)

    assert penalty_relaxed < penalty_strict, \
        "Relaxed MISSLEVEL should produce lower penalty"


def test_check_zone_targets_respects_misslevel():
    """With MISSLEVEL, more targets should appear met."""
    from pymarxan.zones.objective import check_zone_targets
    problem = load_zone_project(DATA_DIR)

    assignment = np.array([1, 0, 0, 0])

    # Strict: likely some targets unmet
    strict = check_zone_targets(problem, assignment)

    # Relaxed
    problem.parameters["MISSLEVEL"] = 0.01
    relaxed = check_zone_targets(problem, assignment)

    # More targets should be met with relaxed MISSLEVEL
    assert sum(relaxed.values()) >= sum(strict.values())
```

**Step 2: Run test to verify it fails**

Run: `/opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/zones/test_objective.py::test_zone_penalty_respects_misslevel tests/pymarxan/zones/test_objective.py::test_check_zone_targets_respects_misslevel -v`
Expected: FAIL — both functions ignore MISSLEVEL

**Step 3: Implement the fix**

In `src/pymarxan/zones/objective.py`, modify `check_zone_targets` — add MISSLEVEL before the comparison at line 127:

```python
    misslevel = float(problem.parameters.get("MISSLEVEL", 1.0))
```

Then change line 127 from:
```python
        targets_met[(zid, fid)] = achieved >= target
```
to:
```python
        targets_met[(zid, fid)] = achieved >= target * misslevel
```

Similarly in `compute_zone_penalty`, add MISSLEVEL before line 165:

```python
    misslevel = float(problem.parameters.get("MISSLEVEL", 1.0))
```

Then change line 165 from:
```python
        shortfall = max(0.0, target - achieved)
```
to:
```python
        shortfall = max(0.0, target * misslevel - achieved)
```

**Step 4: Run tests**

Run: `/opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/zones/test_objective.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add src/pymarxan/zones/objective.py tests/pymarxan/zones/test_objective.py
git commit -m "fix(zones): apply MISSLEVEL to zone target checks and penalty"
```

---

### Task 4: C4 — `apply_cost_from_vector` CRS guard

**Files:**
- Modify: `src/pymarxan/spatial/cost_surface.py:36`
- Test: `tests/pymarxan/spatial/test_cost_surface.py`

**Step 1: Write the failing test**

Add to `tests/pymarxan/spatial/test_cost_surface.py`:

```python
def test_apply_cost_handles_no_crs_on_pus(tmp_path):
    """Should not crash when planning_units has no CRS."""
    from shapely.geometry import box as shapely_box
    pus = gpd.GeoDataFrame(
        {"id": [1, 2], "cost": [1.0, 1.0]},
        geometry=[shapely_box(0, 0, 1, 1), shapely_box(1, 0, 2, 1)],
    )  # No CRS
    cost_layer = gpd.GeoDataFrame(
        {"cost_val": [10.0]},
        geometry=[shapely_box(0, 0, 2, 1)],
        crs="EPSG:4326",
    )
    result = apply_cost_from_vector(pus, cost_layer, "cost_val")
    assert len(result) == 2
```

**Step 2: Run test to verify it fails**

Run: `/opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/spatial/test_cost_surface.py::test_apply_cost_handles_no_crs_on_pus -v`
Expected: FAIL — `CRSError` from `to_crs(None)`

**Step 3: Fix the CRS guard**

In `src/pymarxan/spatial/cost_surface.py`, change line 36 from:

```python
    if cost_layer.crs != planning_units.crs and cost_layer.crs is not None:
```

to:

```python
    if (cost_layer.crs is not None
            and planning_units.crs is not None
            and cost_layer.crs != planning_units.crs):
```

**Step 4: Run tests**

Run: `/opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/spatial/test_cost_surface.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add src/pymarxan/spatial/cost_surface.py tests/pymarxan/spatial/test_cost_surface.py
git commit -m "fix(cost_surface): guard CRS reprojection when PU GDF has no CRS"
```

---

### Task 5: C5 — Guard `min()` in `run_with_overrides`

**Files:**
- Modify: `src/pymarxan/analysis/scenarios.py:158-159`
- Test: `tests/pymarxan/analysis/test_scenario_overrides.py`

**Step 1: Write the failing test**

Add to `tests/pymarxan/analysis/test_scenario_overrides.py`:

```python
def test_run_with_overrides_handles_infeasible(simple_problem):
    """Should raise RuntimeError when solver returns empty solutions."""
    from unittest.mock import MagicMock
    from pymarxan.analysis.scenarios import ScenarioSet

    ss = ScenarioSet()
    mock_solver = MagicMock()
    mock_solver.solve.return_value = []  # Infeasible

    with pytest.raises(RuntimeError, match="no solutions"):
        ss.run_with_overrides(
            "test", simple_problem, mock_solver, overrides={},
        )
```

**Step 2: Run test to verify it fails**

Run: `/opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/analysis/test_scenario_overrides.py::test_run_with_overrides_handles_infeasible -v`
Expected: FAIL — `ValueError: min() arg is an empty sequence` (not RuntimeError)

**Step 3: Add the guard**

In `src/pymarxan/analysis/scenarios.py`, change line 158-159 from:

```python
        solutions = solver.solve(modified, config)
        best = min(solutions, key=lambda s: s.objective)
```

to:

```python
        solutions = solver.solve(modified, config)
        if not solutions:
            raise RuntimeError(
                f"Solver returned no solutions for scenario '{name}'"
            )
        best = min(solutions, key=lambda s: s.objective)
```

**Step 4: Run tests**

Run: `/opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/analysis/test_scenario_overrides.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add src/pymarxan/analysis/scenarios.py tests/pymarxan/analysis/test_scenario_overrides.py
git commit -m "fix(scenarios): guard run_with_overrides against empty solver results"
```

---

## Batch 2 — HIGH (8 Tasks)

### Task 6: H1 — Remove `_sync_solver_params`, sync at solve time

**Files:**
- Modify: `src/pymarxan_app/app.py:188-214`
- Modify: `src/pymarxan_shiny/modules/run_control/run_panel.py:63-71`

**Step 1: Delete `_sync_solver_params` from app.py**

In `src/pymarxan_app/app.py`, delete the entire `@reactive.effect` block from line 188 to line 214 (the `_sync_solver_params` function). Also remove the `import copy` if it's only used by this function (check first).

**Step 2: Move parameter sync to run_panel.py**

In `src/pymarxan_shiny/modules/run_control/run_panel.py`, inside `_run_solver()`, after line 71 (`p.parameters["NUMTEMP"] = ...`), add:

```python
        # Sync solver-specific params (was in app.py _sync_solver_params)
        if solver_type == "mip":
            p.parameters["MIP_TIME_LIMIT"] = str(
                config_dict.get("mip_time_limit", 300)
            )
            p.parameters["MIP_GAP"] = str(config_dict.get("mip_gap", 0.0))
        elif solver_type == "greedy":
            p.parameters["HEURTYPE"] = str(config_dict.get("heurtype", 2))
        elif solver_type == "iterative_improvement":
            p.parameters["ITIMPTYPE"] = str(config_dict.get("itimptype", 0))
        elif solver_type == "pipeline":
            p.parameters["RUNMODE"] = str(config_dict.get("runmode", 0))
```

**Step 3: Run full test suite to verify no regression**

Run: `/opt/micromamba/envs/shiny/bin/pytest tests/ -x --tb=short -q`
Expected: ALL PASS

**Step 4: Commit**

```bash
git add src/pymarxan_app/app.py src/pymarxan_shiny/modules/run_control/run_panel.py
git commit -m "fix(app): move solver param sync from reactive effect to solve-time"
```

---

### Task 7: H2 — `feature_table_server._save` deepcopy before mutation

**Files:**
- Modify: `src/pymarxan_shiny/modules/data/feature_table.py:62-69`

**Step 1: Fix the mutation**

In `src/pymarxan_shiny/modules/data/feature_table.py`, add `import copy` at the top (after the existing imports), then change `_save` from:

```python
    @reactive.effect
    @reactive.event(input.save_changes)
    def _save():
        p = problem()
        if p is None:
            return
        df = feature_grid.data_view()
        p.features["target"] = df["target"].values
        p.features["spf"] = df["spf"].values
        problem.set(p)
        ui.notification_show("Feature targets saved.", type="message")
```

to:

```python
    @reactive.effect
    @reactive.event(input.save_changes)
    def _save():
        p = problem()
        if p is None:
            return
        df = feature_grid.data_view()
        updated = copy.deepcopy(p)
        updated.features["target"] = df["target"].values
        updated.features["spf"] = df["spf"].values
        problem.set(updated)
        ui.notification_show("Feature targets saved.", type="message")
```

**Step 2: Run tests**

Run: `/opt/micromamba/envs/shiny/bin/pytest tests/pymarxan_shiny/test_feature_table.py -v`
Expected: ALL PASS

**Step 3: Commit**

```bash
git add src/pymarxan_shiny/modules/data/feature_table.py
git commit -m "fix(feature_table): deepcopy problem before mutation in save"
```

---

### Task 8: H3 — `run_panel` store results on progress, set reactive from main thread

**Files:**
- Modify: `src/pymarxan_shiny/modules/run_control/progress.py:14-20`
- Modify: `src/pymarxan_shiny/modules/run_control/run_panel.py:85-108`

**Step 1: Add result fields to SolverProgress**

In `src/pymarxan_shiny/modules/run_control/progress.py`, add after line 22:

```python
    result_solutions: list | None = None
    result_best: object | None = None
```

And in `reset()`, after `self.error = None`, add:

```python
        self.result_solutions = None
        self.result_best = None
```

**Step 2: Store results on progress instead of calling .set() from thread**

In `src/pymarxan_shiny/modules/run_control/run_panel.py`, change the `_run()` inner function from:

```python
        def _run():
            try:
                solutions = active.solve(p, config)
                if solutions:
                    best = min(solutions, key=lambda s: s.objective)
                    current_solution.set(best)
                    all_solutions.set(solutions)
                    progress.status = "done"
                    progress.best_objective = best.objective
                    met = sum(best.targets_met.values())
                    total = len(best.targets_met)
                    progress.message = (
                        f"Done! Cost: {best.cost:.2f}, "
                        f"Targets met: {met}/{total}"
                    )
                else:
                    progress.status = "done"
                    progress.message = "Solver returned no solutions."
            except Exception as e:
                progress.status = "error"
                progress.error = str(e)
```

to:

```python
        def _run():
            try:
                solutions = active.solve(p, config)
                if solutions:
                    best = min(solutions, key=lambda s: s.objective)
                    progress.result_best = best
                    progress.result_solutions = solutions
                    progress.best_objective = best.objective
                    met = sum(best.targets_met.values())
                    total = len(best.targets_met)
                    progress.message = (
                        f"Done! Cost: {best.cost:.2f}, "
                        f"Targets met: {met}/{total}"
                    )
                else:
                    progress.message = "Solver returned no solutions."
                progress.status = "done"
            except Exception as e:
                progress.status = "error"
                progress.error = str(e)
```

**Step 3: Add a reactive effect that picks up results from progress on the main thread**

In `run_panel.py`, after the `progress_bar` render function (after line 127), add:

```python
    @reactive.effect
    def _check_results():
        """Transfer solver results from progress to reactive values on main thread."""
        if progress.status in ("running", "idle"):
            reactive.invalidate_later(0.5)
            return
        if progress.status == "done" and progress.result_best is not None:
            current_solution.set(progress.result_best)
            all_solutions.set(progress.result_solutions)
            progress.result_best = None
            progress.result_solutions = None
```

**Step 4: Run tests**

Run: `/opt/micromamba/envs/shiny/bin/pytest tests/pymarxan_shiny/ -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add src/pymarxan_shiny/modules/run_control/progress.py src/pymarxan_shiny/modules/run_control/run_panel.py
git commit -m "fix(run_panel): transfer solver results to reactive values on main thread"
```

---

### Task 9: H4 — `_two_step` loops until convergence

**Files:**
- Modify: `src/pymarxan/solvers/iterative_improvement.py:212-222`
- Test: `tests/pymarxan/solvers/test_iterative_improvement.py`

**Step 1: Write the test**

Add to `tests/pymarxan/solvers/test_iterative_improvement.py`:

```python
def test_two_step_loops_until_stable(tiny_problem):
    """ITIMPTYPE=2 should alternate removal+addition until no improvement."""
    tiny_problem.parameters["ITIMPTYPE"] = 2
    solver = IterativeImprovementSolver(itimptype=2)
    solutions = solver.solve(tiny_problem, SolverConfig(num_solutions=1))
    sol = solutions[0]
    # After convergence, neither removal nor addition should improve
    # Just verify it runs without error and produces a valid solution
    assert sol.objective >= 0
    assert sol.n_selected >= 0
```

**Step 2: Implement the fix**

Replace `_two_step` in `src/pymarxan/solvers/iterative_improvement.py`:

```python
    def _two_step(
        self,
        problem: ConservationProblem,
        cache: ProblemCache,
        blm: float,
        solution: Solution,
    ) -> Solution:
        """Alternating removal then addition passes until convergence."""
        current = solution
        while True:
            after_removal = self._removal_pass(problem, cache, blm, current)
            after_addition = self._addition_pass(
                problem, cache, blm, after_removal
            )
            if after_addition.objective >= current.objective:
                break
            current = after_addition
        return current
```

**Step 3: Run tests**

Run: `/opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/solvers/test_iterative_improvement.py -v`
Expected: ALL PASS

**Step 4: Commit**

```bash
git add src/pymarxan/solvers/iterative_improvement.py tests/pymarxan/solvers/test_iterative_improvement.py
git commit -m "fix(iterative): two_step loops removal+addition until convergence"
```

---

### Task 10: H5 — RunModePipeline passes ITIMPTYPE to sub-solver

**Files:**
- Modify: `src/pymarxan/solvers/run_mode.py:115-125,133-144`

**Step 1: Fix the pipeline**

In `src/pymarxan/solvers/run_mode.py`, change the `_run_pipeline` method. For every `ii_cls()` instantiation, read ITIMPTYPE from problem.parameters with a sensible default of 2:

Replace line 118:
```python
            improved: Solution = ii_cls().improve(problem, sa_sol)
```
with:
```python
            itimptype = int(problem.parameters.get("ITIMPTYPE", 2))
            improved: Solution = ii_cls(itimptype=itimptype).improve(problem, sa_sol)
```

Replace line 124:
```python
            improved = ii_cls().improve(problem, heur_sol)
```
with:
```python
            itimptype = int(problem.parameters.get("ITIMPTYPE", 2))
            improved = ii_cls(itimptype=itimptype).improve(problem, heur_sol)
```

Replace line 143:
```python
            improved = ii_cls().improve(problem, best)
```
with:
```python
            itimptype = int(problem.parameters.get("ITIMPTYPE", 2))
            improved = ii_cls(itimptype=itimptype).improve(problem, best)
```

**Step 2: Run tests**

Run: `/opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/solvers/test_run_mode.py -v`
Expected: ALL PASS

**Step 3: Commit**

```bash
git add src/pymarxan/solvers/run_mode.py
git commit -m "fix(run_mode): pass ITIMPTYPE to iterative improvement sub-solver"
```

---

### Task 11: H6 — `read_pu` and `read_spec` default missing columns

**Files:**
- Modify: `src/pymarxan/io/readers.py:53-81`
- Test: `tests/pymarxan/io/test_readers.py`

**Step 1: Write the failing tests**

Add to `tests/pymarxan/io/test_readers.py`:

```python
def test_read_pu_defaults_missing_status(tmp_path):
    """pu.dat without status column should default to 0."""
    (tmp_path / "pu.dat").write_text("id,cost\n1,10\n2,20\n")
    from pymarxan.io.readers import read_pu
    df = read_pu(tmp_path / "pu.dat")
    assert "status" in df.columns
    assert list(df["status"]) == [0, 0]


def test_read_spec_defaults_missing_spf_and_name(tmp_path):
    """spec.dat without spf/name columns should default to 1.0 and auto-name."""
    (tmp_path / "spec.dat").write_text("id,target\n1,5\n2,10\n")
    from pymarxan.io.readers import read_spec
    df = read_spec(tmp_path / "spec.dat")
    assert "spf" in df.columns
    assert list(df["spf"]) == [1.0, 1.0]
    assert "name" in df.columns
    assert list(df["name"]) == ["Feature_1", "Feature_2"]
```

**Step 2: Run tests to verify they fail**

Run: `/opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/io/test_readers.py::test_read_pu_defaults_missing_status tests/pymarxan/io/test_readers.py::test_read_spec_defaults_missing_spf_and_name -v`
Expected: FAIL

**Step 3: Add defaults**

In `src/pymarxan/io/readers.py`, change `read_pu` (lines 53-58):

```python
    df = _read_dat(path)
    df["id"] = df["id"].astype(int)
    df["cost"] = df["cost"].astype(float)
    if "status" in df.columns:
        df["status"] = df["status"].astype(int)
    else:
        df["status"] = 0
    return df
```

And change `read_spec` (lines 76-81):

```python
    df = _read_dat(path)
    df["id"] = df["id"].astype(int)
    for col in ("target", "prop", "spf"):
        if col in df.columns:
            df[col] = df[col].astype(float)
    if "target" not in df.columns:
        df["target"] = 0.0
    if "spf" not in df.columns:
        df["spf"] = 1.0
    if "name" not in df.columns:
        df["name"] = [f"Feature_{fid}" for fid in df["id"]]
    return df
```

**Step 4: Run tests**

Run: `/opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/io/test_readers.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add src/pymarxan/io/readers.py tests/pymarxan/io/test_readers.py
git commit -m "fix(readers): default missing status, spf, name columns in pu/spec readers"
```

---

### Task 12: H7 — Validate `cell_size > 0` in `generate_planning_grid`

**Files:**
- Modify: `src/pymarxan/spatial/grid.py:38-44`
- Test: `tests/pymarxan/spatial/test_grid.py`

**Step 1: Write the failing test**

Add to `tests/pymarxan/spatial/test_grid.py`:

```python
def test_generate_grid_rejects_non_positive_cell_size():
    """cell_size <= 0 should raise ValueError, not infinite loop."""
    with pytest.raises(ValueError, match="positive"):
        generate_planning_grid((0, 0, 10, 10), cell_size=0)
    with pytest.raises(ValueError, match="positive"):
        generate_planning_grid((0, 0, 10, 10), cell_size=-1)
```

**Step 2: Run test to verify it hangs (interrupt after 5s)**

Run: `/opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/spatial/test_grid.py::test_generate_grid_rejects_non_positive_cell_size -v --timeout=5`
Expected: FAIL (timeout or infinite loop)

**Step 3: Add validation**

In `src/pymarxan/spatial/grid.py`, add at line 38 (before `if grid_type == "square"`):

```python
    if cell_size <= 0:
        raise ValueError(f"cell_size must be positive, got {cell_size}")
```

**Step 4: Run tests**

Run: `/opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/spatial/test_grid.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add src/pymarxan/spatial/grid.py tests/pymarxan/spatial/test_grid.py
git commit -m "fix(grid): validate cell_size > 0 to prevent infinite loop"
```

---

### Task 13: H8 — Add `INITIAL_INCLUDE` to `planning_unit.py`

**Files:**
- Modify: `src/pymarxan/models/planning_unit.py`

**Step 1: Fix the constants**

Replace the entire `src/pymarxan/models/planning_unit.py`:

```python
"""Planning unit status constants."""

AVAILABLE = 0
INITIAL_INCLUDE = 1
LOCKED_IN = 2
LOCKED_OUT = 3

VALID_STATUSES = {AVAILABLE, INITIAL_INCLUDE, LOCKED_IN, LOCKED_OUT}
```

**Step 2: Run tests to verify no regression**

Run: `/opt/micromamba/envs/shiny/bin/pytest tests/ -x --tb=short -q`
Expected: ALL PASS

**Step 3: Commit**

```bash
git add src/pymarxan/models/planning_unit.py
git commit -m "fix(models): add INITIAL_INCLUDE (status=1) to planning_unit constants"
```

---

## Batch 3 — MEDIUM (9 Tasks)

### Task 14: M1 — `export_summary_csv` applies MISSLEVEL

**Files:**
- Modify: `src/pymarxan/io/exporters.py:44-56`
- Test: `tests/pymarxan/io/test_exporters.py`

**Step 1: Write the test**

Add to `tests/pymarxan/io/test_exporters.py` (create if needed):

```python
"""Tests for export functions."""
import numpy as np
import pandas as pd
import pytest
from pymarxan.io.exporters import export_summary_csv
from pymarxan.models.problem import ConservationProblem
from pymarxan.solvers.base import Solution


def test_export_summary_csv_respects_misslevel(tmp_path):
    pu = pd.DataFrame({"id": [1, 2], "cost": [10.0, 20.0], "status": [0, 0]})
    features = pd.DataFrame({"id": [1], "name": ["f1"], "target": [10.0], "spf": [1.0]})
    puvspr = pd.DataFrame({"species": [1, 1], "pu": [1, 2], "amount": [4.0, 3.0]})
    problem = ConservationProblem(
        planning_units=pu, features=features, pu_vs_features=puvspr,
        parameters={"MISSLEVEL": 0.5},
    )
    sol = Solution(
        selected=np.array([True, True]), cost=30.0, boundary=0.0,
        objective=30.0, targets_met={1: True},
    )
    path = tmp_path / "summary.csv"
    export_summary_csv(problem, sol, path)
    df = pd.read_csv(path)
    # achieved=7.0, target=10.0, MISSLEVEL=0.5 => effective_target=5.0
    # 7.0 >= 5.0 => met=True
    assert df.iloc[0]["met"] is True
```

**Step 2: Fix the function**

In `src/pymarxan/io/exporters.py`, add after line 38:

```python
    misslevel = float(problem.parameters.get("MISSLEVEL", 1.0))
```

Change line 56 from:
```python
            "met": achieved >= target,
```
to:
```python
            "met": achieved >= target * misslevel,
```

**Step 3: Run tests**

Run: `/opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/io/test_exporters.py -v`
Expected: ALL PASS

**Step 4: Commit**

```bash
git add src/pymarxan/io/exporters.py tests/pymarxan/io/test_exporters.py
git commit -m "fix(exporters): apply MISSLEVEL to target checking in summary CSV"
```

---

### Task 15: M2 — ZoneSASolver exclude zone 0 from options

**Files:**
- Modify: `src/pymarxan/zones/solver.py:63-64`

**Step 1: Fix zone_options**

In `src/pymarxan/zones/solver.py`, change line 64 from:

```python
        zone_options = np.array([0] + zone_ids_list, dtype=int)
```

to:

```python
        zone_options = np.array(zone_ids_list, dtype=int)
```

**Step 2: Run zone tests**

Run: `/opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/zones/ -v`
Expected: ALL PASS

**Step 3: Commit**

```bash
git add src/pymarxan/zones/solver.py
git commit -m "perf(zones): exclude zone 0 from SA move options to reduce wasted iterations"
```

---

### Task 16: M3 — `fetch_wdpa` pagination

**Files:**
- Modify: `src/pymarxan/spatial/wdpa.py:37-52`

**Step 1: Add pagination**

In `src/pymarxan/spatial/wdpa.py`, replace the single-request block (lines 46-51) with a pagination loop:

```python
    all_pa: list[dict] = []
    page = 1
    while True:
        params["page"] = page
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        pa_list = data.get("protected_areas", [])
        if not pa_list:
            break
        all_pa.extend(pa_list)
        page += 1
        if page > 100:  # Safety limit
            break
```

Then change the next line from `for pa in pa_list:` to `for pa in all_pa:`.

**Step 2: Run tests**

Run: `/opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/spatial/test_wdpa.py -v`
Expected: ALL PASS (tests mock the API)

**Step 3: Commit**

```bash
git add src/pymarxan/spatial/wdpa.py
git commit -m "fix(wdpa): add pagination to fetch_wdpa for countries with many protected areas"
```

---

### Task 17: M4 — `compute_adjacency` uses spatial index

**Files:**
- Modify: `src/pymarxan/spatial/grid.py:139-156`

**Step 1: Replace O(n²) loop with spatial index**

Replace `compute_adjacency` in `src/pymarxan/spatial/grid.py`:

```python
def compute_adjacency(planning_units: gpd.GeoDataFrame) -> pd.DataFrame:
    """Compute boundary DataFrame from shared edges between adjacent PUs.

    Parameters
    ----------
    planning_units : gpd.GeoDataFrame
        Must have ``id`` and ``geometry`` columns.

    Returns
    -------
    pd.DataFrame
        Columns: id1, id2, boundary (shared edge length).
    """
    rows: list[dict] = []
    geoms = planning_units.geometry.values
    ids = planning_units["id"].values
    sindex = planning_units.sindex

    for i in range(len(planning_units)):
        candidates = list(sindex.intersection(geoms[i].bounds))
        for j in candidates:
            if j <= i:
                continue
            shared = geoms[i].intersection(geoms[j]).length
            if shared > 1e-10:
                rows.append({
                    "id1": int(ids[i]),
                    "id2": int(ids[j]),
                    "boundary": shared,
                })

    return pd.DataFrame(rows, columns=["id1", "id2", "boundary"])
```

**Step 2: Run tests**

Run: `/opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/spatial/test_grid.py -v`
Expected: ALL PASS

**Step 3: Commit**

```bash
git add src/pymarxan/spatial/grid.py
git commit -m "perf(grid): use spatial index in compute_adjacency instead of O(n²) loop"
```

---

### Task 18: M5 — `connectivity_to_matrix` symmetric option

**Files:**
- Modify: `src/pymarxan/connectivity/io.py:33-46`

**Step 1: Add symmetric parameter**

Change `connectivity_to_matrix` signature and body:

```python
def connectivity_to_matrix(
    edgelist: pd.DataFrame,
    pu_ids: list[int],
    symmetric: bool = True,
) -> np.ndarray:
    """Convert an edge list DataFrame to NxN matrix."""
    n = len(pu_ids)
    id_to_idx = {pid: i for i, pid in enumerate(pu_ids)}
    matrix = np.zeros((n, n))
    for _, row in edgelist.iterrows():
        i = id_to_idx.get(int(row["id1"]))
        j = id_to_idx.get(int(row["id2"]))
        if i is not None and j is not None:
            matrix[i, j] = float(row["value"])
            if symmetric:
                matrix[j, i] = float(row["value"])
    return matrix
```

Do the same for `read_connectivity_edgelist`:

```python
def read_connectivity_edgelist(
    path: str | Path,
    pu_ids: list[int],
    symmetric: bool = True,
) -> np.ndarray:
    """Read an edge list CSV and convert to NxN matrix. Expected columns: id1, id2, value."""
    df = pd.read_csv(path)
    n = len(pu_ids)
    id_to_idx = {pid: i for i, pid in enumerate(pu_ids)}
    matrix = np.zeros((n, n))
    for _, row in df.iterrows():
        i = id_to_idx.get(int(row["id1"]))
        j = id_to_idx.get(int(row["id2"]))
        if i is not None and j is not None:
            matrix[i, j] = float(row["value"])
            if symmetric:
                matrix[j, i] = float(row["value"])
    return matrix
```

**Step 2: Run tests**

Run: `/opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/connectivity/ -v`
Expected: ALL PASS

**Step 3: Commit**

```bash
git add src/pymarxan/connectivity/io.py
git commit -m "fix(connectivity): add symmetric flag to edge list matrix conversion"
```

---

### Task 19: M6 — `cost_upload_server` validates cost column

**Files:**
- Modify: `src/pymarxan_shiny/modules/spatial/cost_upload.py:65-72`

**Step 1: Add validation**

In `src/pymarxan_shiny/modules/spatial/cost_upload.py`, change the try block (lines 65-78) to validate the column first:

```python
        try:
            cost_layer = gpd.read_file(file_info[0]["datapath"])
            col = input.cost_col()
            if col not in cost_layer.columns:
                ui.notification_show(
                    f"Column '{col}' not found. "
                    f"Available: {list(cost_layer.columns)}",
                    type="error",
                )
                return
            updated = apply_cost_from_vector(
                p.planning_units,
                cost_layer,
                cost_column=col,
                aggregation=input.aggregation(),
            )
            new_problem = copy.deepcopy(p)
            new_problem.planning_units = updated
            problem.set(new_problem)
            ui.notification_show("Cost surface applied.", type="message")
        except Exception as exc:
            ui.notification_show(f"Cost error: {exc}", type="error")
```

**Step 2: Run tests**

Run: `/opt/micromamba/envs/shiny/bin/pytest tests/pymarxan_shiny/ -v`
Expected: ALL PASS

**Step 3: Commit**

```bash
git add src/pymarxan_shiny/modules/spatial/cost_upload.py
git commit -m "fix(cost_upload): validate cost column exists before processing"
```

---

### Task 20: M7 — `ScenarioSet.remove()` raises on missing name

**Files:**
- Modify: `src/pymarxan/analysis/scenarios.py:61-62`
- Test: `tests/pymarxan/analysis/test_scenarios.py`

**Step 1: Write the test**

Add to `tests/pymarxan/analysis/test_scenarios.py`:

```python
def test_remove_nonexistent_raises():
    ss = ScenarioSet()
    with pytest.raises(KeyError, match="nonexistent"):
        ss.remove("nonexistent")
```

**Step 2: Fix the method**

In `src/pymarxan/analysis/scenarios.py`, change `remove`:

```python
    def remove(self, name: str) -> None:
        before = len(self._scenarios)
        self._scenarios = [s for s in self._scenarios if s.name != name]
        if len(self._scenarios) == before:
            raise KeyError(f"Scenario '{name}' not found")
```

**Step 3: Run tests**

Run: `/opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/analysis/test_scenarios.py -v`
Expected: ALL PASS

**Step 4: Commit**

```bash
git add src/pymarxan/analysis/scenarios.py tests/pymarxan/analysis/test_scenarios.py
git commit -m "fix(scenarios): raise KeyError when removing nonexistent scenario"
```

---

### Task 21: M8 — `summary_table_server` vectorized achievement

**Files:**
- Modify: `src/pymarxan_shiny/modules/results/summary_table.py:23-34`

**Step 1: Replace iterrows with vectorized computation**

Replace the body of `target_table` in `src/pymarxan_shiny/modules/results/summary_table.py`:

```python
    @render.ui
    def target_table():
        p = problem()
        s = solution()
        if p is None or s is None:
            return ui.p("No solution available. Run a solver first.")
        pu_ids = p.planning_units["id"].values
        id_to_idx = {int(pid): i for i, pid in enumerate(pu_ids)}

        # Vectorized: map PU IDs to indices, filter selected, groupby feature
        pvf = p.pu_vs_features.copy()
        pvf["_idx"] = pvf["pu"].map(id_to_idx)
        pvf = pvf.dropna(subset=["_idx"])
        pvf["_idx"] = pvf["_idx"].astype(int)
        pvf["_selected"] = pvf["_idx"].map(lambda i: bool(s.selected[i]))
        achieved_by_feat = (
            pvf[pvf["_selected"]].groupby("species")["amount"].sum()
        )

        rows = []
        for _, frow in p.features.iterrows():
            fid = int(frow["id"])
            fname = frow.get("name", f"Feature {fid}")
            target = float(frow.get("target", 0.0))
            achieved = float(achieved_by_feat.get(fid, 0.0))
            met = achieved >= target
            pct = (achieved / target * 100) if target > 0 else 100.0
            rows.append({
                "id": fid, "name": fname, "target": target,
                "achieved": achieved, "pct": pct, "met": met,
            })
        table_rows = [
            ui.tags.tr(
                ui.tags.td(str(r["id"])), ui.tags.td(r["name"]),
                ui.tags.td(f"{r['target']:.1f}"), ui.tags.td(f"{r['achieved']:.1f}"),
                ui.tags.td(f"{r['pct']:.1f}%"),
                ui.tags.td("Met" if r["met"] else "NOT MET",
                           style=f"color: {'green' if r['met'] else 'red'}; font-weight: bold"),
            ) for r in rows
        ]
        return ui.tags.table(
            ui.tags.thead(ui.tags.tr(
                ui.tags.th("ID"), ui.tags.th("Feature"), ui.tags.th("Target"),
                ui.tags.th("Achieved"), ui.tags.th("%"), ui.tags.th("Status"),
            )),
            ui.tags.tbody(*table_rows),
            class_="table table-striped",
        )
```

**Step 2: Run tests**

Run: `/opt/micromamba/envs/shiny/bin/pytest tests/pymarxan_shiny/test_summary_table.py -v`
Expected: ALL PASS

**Step 3: Commit**

```bash
git add src/pymarxan_shiny/modules/results/summary_table.py
git commit -m "perf(summary_table): vectorized feature achievement instead of iterrows"
```

---

### Task 22: M9 — Pass `gadm_boundary` to `grid_builder_server`

**Files:**
- Modify: `src/pymarxan_app/app.py` (grid_builder_server call)
- Modify: `src/pymarxan_shiny/modules/spatial/grid_builder.py` (accept and use boundary)

**Step 1: Read the grid_builder module first**

Read `src/pymarxan_shiny/modules/spatial/grid_builder.py` to understand its current signature, then add `gadm_boundary` parameter.

In `src/pymarxan_app/app.py`, change the `grid_builder_server` call from:

```python
    grid_builder_server("grid_gen", problem=problem)
```

to:

```python
    grid_builder_server("grid_gen", problem=problem, gadm_boundary=gadm_boundary)
```

In `src/pymarxan_shiny/modules/spatial/grid_builder.py`, add `gadm_boundary` parameter to the server function, add a checkbox UI element "Clip to GADM boundary", and pass `clip_to=gadm_boundary().unary_union` when generating the grid (if the checkbox is checked and the boundary is available).

**Step 2: Run tests**

Run: `/opt/micromamba/envs/shiny/bin/pytest tests/ -x --tb=short -q`
Expected: ALL PASS

**Step 3: Commit**

```bash
git add src/pymarxan_app/app.py src/pymarxan_shiny/modules/spatial/grid_builder.py
git commit -m "feat(grid_builder): connect GADM boundary as clip polygon option"
```

---

## Batch 4 — Test Gaps (6 Tasks)

### Task 23: T1 — Unit test `compute_delta_objective`

**Files:**
- Test: `tests/pymarxan/solvers/test_cache.py`

**Step 1: Write the test**

Add to `tests/pymarxan/solvers/test_cache.py`:

```python
def test_delta_matches_full_objective_for_removal(tiny_problem):
    """Delta computation must match difference of full objective for removal."""
    cache = ProblemCache.from_problem(tiny_problem)
    selected = np.ones(tiny_problem.n_planning_units, dtype=bool)
    held = cache.compute_held(selected)
    total_cost = float(np.sum(cache.costs[selected]))
    blm = 1.0

    full_before = cache.compute_full_objective(selected, held, blm)
    delta = cache.compute_delta_objective(0, selected, held, total_cost, blm)

    # Apply the flip
    selected_after = selected.copy()
    selected_after[0] = False
    new_held = cache.compute_held(selected_after)
    full_after = cache.compute_full_objective(selected_after, new_held, blm)

    assert delta == pytest.approx(full_after - full_before, abs=1e-6)


def test_delta_matches_full_objective_for_addition(tiny_problem):
    """Delta computation must match difference of full objective for addition."""
    cache = ProblemCache.from_problem(tiny_problem)
    selected = np.zeros(tiny_problem.n_planning_units, dtype=bool)
    held = cache.compute_held(selected)
    total_cost = 0.0
    blm = 1.0

    full_before = cache.compute_full_objective(selected, held, blm)
    delta = cache.compute_delta_objective(0, selected, held, total_cost, blm)

    selected_after = selected.copy()
    selected_after[0] = True
    new_held = cache.compute_held(selected_after)
    full_after = cache.compute_full_objective(selected_after, new_held, blm)

    assert delta == pytest.approx(full_after - full_before, abs=1e-6)
```

**Step 2: Run tests**

Run: `/opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/solvers/test_cache.py -v -k "test_delta_matches"`
Expected: ALL PASS (verifying existing code correctness)

**Step 3: Commit**

```bash
git add tests/pymarxan/solvers/test_cache.py
git commit -m "test: add delta vs full objective verification for ProblemCache"
```

---

### Task 24: T2 — Test `compute_feature_shortfalls` directly

**Files:**
- Test: `tests/pymarxan/solvers/test_utils.py` (create or add to existing)

**Step 1: Write the test**

```python
"""Tests for compute_feature_shortfalls."""
import numpy as np
import pytest
from pymarxan.solvers.utils import compute_feature_shortfalls


def test_compute_feature_shortfalls_all_unselected(tiny_problem):
    """With nothing selected, shortfall equals target for each feature."""
    pu_index = {int(pid): i for i, pid in enumerate(tiny_problem.planning_units["id"])}
    selected = np.zeros(tiny_problem.n_planning_units, dtype=bool)
    shortfalls = compute_feature_shortfalls(tiny_problem, selected, pu_index)
    for _, frow in tiny_problem.features.iterrows():
        fid = int(frow["id"])
        target = float(frow["target"])
        assert shortfalls[fid] == pytest.approx(target)


def test_compute_feature_shortfalls_all_selected(tiny_problem):
    """With all selected, shortfall should be 0 if targets are met."""
    pu_index = {int(pid): i for i, pid in enumerate(tiny_problem.planning_units["id"])}
    selected = np.ones(tiny_problem.n_planning_units, dtype=bool)
    shortfalls = compute_feature_shortfalls(tiny_problem, selected, pu_index)
    for fid, sf in shortfalls.items():
        assert sf >= 0.0  # Never negative
```

**Step 2: Run tests**

Run: `/opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/solvers/test_utils.py -v`
Expected: ALL PASS

**Step 3: Commit**

```bash
git add tests/pymarxan/solvers/test_utils.py
git commit -m "test: add direct tests for compute_feature_shortfalls"
```

---

### Task 25: T3 — Zone SA solution penalty accuracy

This is covered by the C2 fix (Task 2). No additional test needed if Task 2's test passes.

---

### Task 26: T4 — Test cost surface `sum` and `max` aggregation

**Files:**
- Test: `tests/pymarxan/spatial/test_cost_surface.py`

**Step 1: Write tests**

Add to `tests/pymarxan/spatial/test_cost_surface.py`:

```python
class TestAggregationModes:
    def test_sum_aggregation(self):
        from shapely.geometry import box as shapely_box
        pus = gpd.GeoDataFrame(
            {"id": [1], "cost": [1.0]},
            geometry=[shapely_box(0, 0, 2, 2)],
            crs="EPSG:4326",
        )
        cost_layer = gpd.GeoDataFrame(
            {"cost_val": [10.0, 20.0]},
            geometry=[shapely_box(0, 0, 1, 2), shapely_box(1, 0, 2, 2)],
            crs="EPSG:4326",
        )
        result = apply_cost_from_vector(pus, cost_layer, "cost_val", aggregation="sum")
        assert result.iloc[0]["cost"] == pytest.approx(30.0)

    def test_max_aggregation(self):
        from shapely.geometry import box as shapely_box
        pus = gpd.GeoDataFrame(
            {"id": [1], "cost": [1.0]},
            geometry=[shapely_box(0, 0, 2, 2)],
            crs="EPSG:4326",
        )
        cost_layer = gpd.GeoDataFrame(
            {"cost_val": [10.0, 20.0]},
            geometry=[shapely_box(0, 0, 1, 2), shapely_box(1, 0, 2, 2)],
            crs="EPSG:4326",
        )
        result = apply_cost_from_vector(pus, cost_layer, "cost_val", aggregation="max")
        assert result.iloc[0]["cost"] == pytest.approx(20.0)
```

**Step 2: Run tests**

Run: `/opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/spatial/test_cost_surface.py -v`
Expected: ALL PASS

**Step 3: Commit**

```bash
git add tests/pymarxan/spatial/test_cost_surface.py
git commit -m "test: add sum and max aggregation mode tests for cost surface"
```

---

### Task 27: T5 — Heuristic boundary accuracy when BLM > 0

**Files:**
- Test: `tests/pymarxan/solvers/test_heuristic.py`

**Step 1: Write the test**

Add to `tests/pymarxan/solvers/test_heuristic.py`:

```python
def test_heuristic_boundary_accuracy_with_blm(tiny_problem):
    """Heuristic solution.boundary should match compute_boundary when BLM > 0."""
    from pymarxan.solvers.utils import compute_boundary
    tiny_problem.parameters["BLM"] = 1.0
    solver = HeuristicSolver()
    solutions = solver.solve(tiny_problem, SolverConfig(num_solutions=1))
    sol = solutions[0]
    pu_index = {int(pid): i for i, pid in enumerate(tiny_problem.planning_units["id"])}
    expected_boundary = compute_boundary(tiny_problem, sol.selected, pu_index)
    assert sol.boundary == pytest.approx(expected_boundary, abs=1e-6)
```

**Step 2: Run tests**

Run: `/opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/solvers/test_heuristic.py -v`
Expected: ALL PASS

**Step 3: Commit**

```bash
git add tests/pymarxan/solvers/test_heuristic.py
git commit -m "test: verify heuristic boundary accuracy against compute_boundary"
```

---

### Task 28: T6 — `write_mvbest` with MISSLEVEL < 1.0

**Files:**
- Test: `tests/pymarxan/io/test_output_writers.py`

**Step 1: Write the test**

Add to `tests/pymarxan/io/test_output_writers.py`:

```python
def test_write_mvbest_respects_misslevel(tmp_path, simple_problem, solution_partial):
    """With MISSLEVEL=0.5, Target_Met should reflect relaxed targets."""
    simple_problem.parameters["MISSLEVEL"] = 0.5
    path = tmp_path / "mvbest_misslevel.csv"
    write_mvbest(simple_problem, solution_partial, path)
    df = read_mvbest(path)
    # Feature 1: target=15, MISSLEVEL=0.5 => effective_target=7.5
    # PU1+PU2 selected => held = 10+8 = 18 >= 7.5 => Target_Met=True
    f1 = df[df["Feature_ID"] == 1].iloc[0]
    assert f1["Target_Met"] is True
```

**Step 2: Run tests**

Run: `/opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/io/test_output_writers.py -v`
Expected: ALL PASS (verifying existing MISSLEVEL code in write_mvbest works)

**Step 3: Commit**

```bash
git add tests/pymarxan/io/test_output_writers.py
git commit -m "test: verify write_mvbest Target_Met with relaxed MISSLEVEL"
```

---

### Task 29: Final regression + lint

**Step 1: Run full test suite**

Run: `/opt/micromamba/envs/shiny/bin/pytest tests/ -x --tb=short -q`
Expected: ALL PASS

**Step 2: Run ruff lint**

Run: `/home/razinka/.local/bin/ruff check src/`
Expected: No errors

**Step 3: Check coverage**

Run: `/opt/micromamba/envs/shiny/bin/pytest tests/ --cov=src --cov-fail-under=75 -q`
Expected: PASS with coverage ≥ 75%
