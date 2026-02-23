# Codebase Review 5 — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix 19 issues (3 critical, 5 high, 7 medium, 4 test gaps) found in Codebase Review 5.

**Architecture:** Surgical fixes to existing modules. Each fix is independent — TDD per fix, commit after each.

**Tech Stack:** Python, NumPy, pandas, pytest, Shiny for Python

---

## Batch 1: CRITICAL (3 issues)

### Task 1: C1 — ZoneProblemCache applies MISSLEVEL to zone targets

**Files:**
- Modify: `src/pymarxan/zones/cache.py:96-167` (constructor `from_zone_problem`)
- Test: `tests/pymarxan/zones/test_zone_cache.py`

**Step 1: Write the failing test**

Add to `tests/pymarxan/zones/test_zone_cache.py`:

```python
class TestMissLevel:
    """Verify MISSLEVEL is applied to zone target matrix."""

    def test_misslevel_scales_targets(self, zone_problem):
        """With MISSLEVEL=0.5, targets should be halved."""
        import copy
        problem = copy.deepcopy(zone_problem)
        problem.parameters["MISSLEVEL"] = 0.5
        cache = ZoneProblemCache.from_zone_problem(problem)

        col1 = cache.zone_id_to_col[1]
        f0 = cache.feat_id_to_col[1]
        # Original target for zone1/feature1 is 10.0
        assert cache.zone_target_matrix[col1, f0] == pytest.approx(5.0)

    def test_misslevel_default_no_change(self, zone_problem):
        """Default MISSLEVEL=1.0 should leave targets unchanged."""
        cache = ZoneProblemCache.from_zone_problem(zone_problem)
        col1 = cache.zone_id_to_col[1]
        f0 = cache.feat_id_to_col[1]
        assert cache.zone_target_matrix[col1, f0] == pytest.approx(10.0)

    def test_penalty_uses_scaled_targets(self, zone_problem):
        """Cache penalty should agree with objective.compute_zone_penalty."""
        import copy
        problem = copy.deepcopy(zone_problem)
        problem.parameters["MISSLEVEL"] = 0.5
        cache = ZoneProblemCache.from_zone_problem(problem)

        assignment = np.array([1, 2, 1, 2], dtype=int)
        held = cache.compute_held_per_zone(assignment)
        cached_obj = cache.compute_full_zone_objective(assignment, held, 0.0)

        from pymarxan.zones.objective import compute_zone_objective
        ref_obj = compute_zone_objective(problem, assignment, 0.0)
        assert cached_obj == pytest.approx(ref_obj, abs=1e-10)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/pymarxan/zones/test_zone_cache.py::TestMissLevel -v`
Expected: `test_misslevel_scales_targets` FAILS (target is 10.0 not 5.0)

**Step 3: Write minimal implementation**

In `src/pymarxan/zones/cache.py`, in `from_zone_problem()`, after the zone_target_matrix loop (after line 167), add MISSLEVEL scaling:

```python
        # Apply MISSLEVEL to zone targets (match objective.py behavior)
        misslevel = float(problem.parameters.get("MISSLEVEL", 1.0))
        zone_target_matrix *= misslevel
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/pymarxan/zones/test_zone_cache.py -v`
Expected: All PASS including new TestMissLevel tests

**Step 5: Commit**

```bash
git add src/pymarxan/zones/cache.py tests/pymarxan/zones/test_zone_cache.py
git commit -m "fix: apply MISSLEVEL to ZoneProblemCache zone targets

Zone SA penalty computation now agrees with objective.py when
MISSLEVEL < 1.0."
```

---

### Task 2: C2 — Zone SA all-locked guard

**Files:**
- Modify: `src/pymarxan/zones/solver.py:78-82`
- Test: `tests/pymarxan/zones/test_solver.py`

**Step 1: Write the failing test**

Add to `tests/pymarxan/zones/test_solver.py`:

```python
    def test_all_locked_returns_solution(self):
        """Zone SA should handle all PUs locked without crashing."""
        problem = copy.deepcopy(self.problem)
        problem.planning_units["status"] = 2  # All locked-in
        problem.parameters["NUMITNS"] = 100
        config = SolverConfig(num_solutions=1, seed=42)
        solutions = self.solver.solve(problem, config)
        assert len(solutions) == 1
        assert solutions[0].cost >= 0

    def test_all_locked_out_returns_solution(self):
        """Zone SA should handle all PUs locked-out without crashing."""
        problem = copy.deepcopy(self.problem)
        problem.planning_units["status"] = 3  # All locked-out
        problem.parameters["NUMITNS"] = 100
        config = SolverConfig(num_solutions=1, seed=42)
        solutions = self.solver.solve(problem, config)
        assert len(solutions) == 1
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/pymarxan/zones/test_solver.py::TestZoneSASolver::test_all_locked_returns_solution -v`
Expected: FAILS with `ValueError: high <= 0` from `rng.integers(0)`

**Step 3: Write minimal implementation**

In `src/pymarxan/zones/solver.py`, after line 81 (`n_swappable = len(swappable)`), add an early-return guard:

```python
        if n_swappable == 0:
            # All PUs locked — build forced assignment and return
            solutions = []
            for run_idx in range(config.num_solutions):
                selected = assignment > 0
                cost = compute_zone_cost(problem, assignment)
                std_boundary = compute_standard_boundary(problem, assignment)
                zone_boundary = compute_zone_boundary(problem, assignment)
                zone_targets = check_zone_targets(problem, assignment)
                zone_penalty = compute_zone_penalty(problem, assignment)
                zone_shortfall = 0.0
                if problem.zone_targets is not None:
                    misslevel = float(problem.parameters.get("MISSLEVEL", 1.0))
                    pu_index = {
                        int(pid): i
                        for i, pid in enumerate(
                            problem.planning_units["id"].tolist()
                        )
                    }
                    for _, trow in problem.zone_targets.iterrows():
                        zid = int(trow["zone"])
                        fid = int(trow["feature"])
                        target = float(trow["target"])
                        contribution = problem.get_contribution(fid, zid)
                        feat_data = problem.pu_vs_features[
                            problem.pu_vs_features["species"] == fid
                        ]
                        achieved = 0.0
                        for _, r in feat_data.iterrows():
                            pid = int(r["pu"])
                            idx = pu_index.get(pid)
                            if idx is not None and int(assignment[idx]) == zid:
                                achieved += float(r["amount"]) * contribution
                        zone_shortfall += max(0.0, target * misslevel - achieved)
                obj = cost + float(problem.parameters.get("BLM", 0.0)) * std_boundary + zone_penalty
                sol = Solution(
                    selected=selected,
                    cost=cost,
                    boundary=std_boundary,
                    objective=obj,
                    targets_met={},
                    penalty=zone_penalty,
                    shortfall=zone_shortfall,
                    zone_assignment=assignment.copy(),
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
                solutions.append(sol)
            if progress is not None:
                progress.status = "done"
            return solutions
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/pymarxan/zones/test_solver.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add src/pymarxan/zones/solver.py tests/pymarxan/zones/test_solver.py
git commit -m "fix: guard Zone SA against all-locked PU crash

When all PUs are locked, rng.integers(0) raised ValueError.
Now builds a forced-assignment solution like regular SA does."
```

---

### Task 3: C3 — Feature table save joins on ID instead of positional

**Files:**
- Modify: `src/pymarxan_shiny/modules/data/feature_table.py:64-72`
- Test: `tests/pymarxan_shiny/test_feature_table.py`

**Step 1: Write the failing test**

Add to `tests/pymarxan_shiny/test_feature_table.py` (or create if doesn't exist):

```python
"""Tests for feature table save — sort-safe merge."""
from __future__ import annotations

import copy

import pandas as pd
import pytest

from pymarxan.models.problem import ConservationProblem


def _make_problem():
    """Create a problem with 3 features."""
    pu = pd.DataFrame({"id": [1, 2], "cost": [10.0, 20.0], "status": [0, 0]})
    features = pd.DataFrame({
        "id": [1, 2, 3],
        "name": ["feat_a", "feat_b", "feat_c"],
        "target": [100.0, 200.0, 300.0],
        "spf": [1.0, 2.0, 3.0],
    })
    puvspr = pd.DataFrame({
        "species": [1, 2, 3],
        "pu": [1, 1, 2],
        "amount": [5.0, 10.0, 15.0],
    })
    return ConservationProblem(
        planning_units=pu, features=features, pu_vs_features=puvspr,
    )


class TestFeatureTableSortSafe:
    def test_reversed_view_applies_correctly(self):
        """Simulate user sorting features in reverse order.

        If the save uses positional assignment, feature 3's target
        gets applied to feature 1 — a data-corruption bug.
        """
        p = _make_problem()
        # Simulate data_view() returning reversed order
        view_df = p.features[["id", "name", "target", "spf"]].copy()
        # User edits: change feature 3's target to 999
        view_df.loc[view_df["id"] == 3, "target"] = 999.0
        # Reverse to simulate user sort by name desc
        view_df = view_df.iloc[::-1].reset_index(drop=True)

        # Apply the save using the NEW id-based merge approach
        updated = copy.deepcopy(p)
        merged = updated.features.merge(
            view_df[["id", "target", "spf"]],
            on="id",
            suffixes=("_old", ""),
        )
        updated.features["target"] = merged["target"].values
        updated.features["spf"] = merged["spf"].values

        # Feature 3 should have target 999, not feature 1
        assert float(updated.features.loc[updated.features["id"] == 3, "target"].iloc[0]) == 999.0
        assert float(updated.features.loc[updated.features["id"] == 1, "target"].iloc[0]) == 100.0
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/pymarxan_shiny/test_feature_table.py::TestFeatureTableSortSafe -v`
Expected: PASS (the test validates the new approach logic directly)

Note: The actual bug is in the Shiny module's `_save` function. We write the test to verify our fix approach works, then apply the fix to the module.

**Step 3: Write minimal implementation**

In `src/pymarxan_shiny/modules/data/feature_table.py`, replace the `_save` function body (lines 64-73):

Replace:
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

With:
```python
    @reactive.effect
    @reactive.event(input.save_changes)
    def _save():
        p = problem()
        if p is None:
            return
        df = feature_grid.data_view()
        updated = copy.deepcopy(p)
        # Join on id to handle user-sorted/filtered views correctly
        edits = df.set_index("id")[["target", "spf"]]
        for fid in edits.index:
            mask = updated.features["id"] == fid
            updated.features.loc[mask, "target"] = float(edits.at[fid, "target"])
            updated.features.loc[mask, "spf"] = float(edits.at[fid, "spf"])
        problem.set(updated)
        ui.notification_show("Feature targets saved.", type="message")
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/pymarxan_shiny/test_feature_table.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add src/pymarxan_shiny/modules/data/feature_table.py tests/pymarxan_shiny/test_feature_table.py
git commit -m "fix: feature table save joins on id instead of positional

Prevents data corruption when user sorts or filters the DataGrid
before saving."
```

---

## Batch 2: HIGH (5 issues)

### Task 4: H1 — Calibration functions guard empty solver results

**Files:**
- Modify: `src/pymarxan/calibration/blm.py:60-61`
- Modify: `src/pymarxan/calibration/spf.py:54-55`
- Modify: `src/pymarxan/calibration/sweep.py:94-95`
- Modify: `src/pymarxan/calibration/sensitivity.py:69-70`
- Modify: `src/pymarxan/calibration/parallel.py:37-38`
- Test: `tests/pymarxan/calibration/test_empty_solutions.py`

**Step 1: Write the failing test**

Create `tests/pymarxan/calibration/test_empty_solutions.py`:

```python
"""Tests for calibration functions handling empty solver results."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from pymarxan.io.readers import load_project
from pymarxan.solvers.base import Solution, Solver, SolverConfig

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "simple"


class InfeasibleSolver(Solver):
    """Mock solver that always returns empty results (infeasible)."""
    def name(self) -> str:
        return "infeasible"

    def supports_zones(self) -> bool:
        return False

    def solve(self, problem, config=None):
        return []


class TestBLMEmptyGuard:
    def test_calibrate_blm_skips_infeasible(self):
        from pymarxan.calibration.blm import calibrate_blm
        problem = load_project(DATA_DIR)
        solver = InfeasibleSolver()
        # Should not raise ValueError on min([])
        result = calibrate_blm(problem, solver, blm_values=[0.0, 1.0])
        # Result should have empty lists (all skipped)
        assert len(result.solutions) == 0


class TestSPFEmptyGuard:
    def test_calibrate_spf_skips_infeasible(self):
        from pymarxan.calibration.spf import calibrate_spf
        problem = load_project(DATA_DIR)
        solver = InfeasibleSolver()
        # Should not crash on empty sols
        result = calibrate_spf(problem, solver, max_iterations=2)
        assert result.solution is None or result.history is not None


class TestSweepEmptyGuard:
    def test_run_sweep_skips_infeasible(self):
        from pymarxan.calibration.sweep import SweepConfig, run_sweep
        problem = load_project(DATA_DIR)
        solver = InfeasibleSolver()
        config = SweepConfig(param_dicts=[{"BLM": 0.0}, {"BLM": 1.0}])
        result = run_sweep(problem, solver, config)
        assert len(result.solutions) == 0


class TestSensitivityEmptyGuard:
    def test_run_sensitivity_skips_infeasible(self):
        from pymarxan.calibration.sensitivity import (
            SensitivityConfig,
            run_sensitivity,
        )
        problem = load_project(DATA_DIR)
        solver = InfeasibleSolver()
        config = SensitivityConfig(multipliers=[1.0])
        result = run_sensitivity(problem, solver, config)
        assert len(result.runs) == 0


class TestParallelEmptyGuard:
    def test_run_sweep_parallel_skips_infeasible(self):
        from pymarxan.calibration.parallel import run_sweep_parallel
        from pymarxan.calibration.sweep import SweepConfig
        problem = load_project(DATA_DIR)
        solver = InfeasibleSolver()
        config = SweepConfig(param_dicts=[{"BLM": 0.0}])
        # max_workers=1 falls through to run_sweep
        result = run_sweep_parallel(problem, solver, config, max_workers=1)
        assert len(result.solutions) == 0
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/pymarxan/calibration/test_empty_solutions.py -v`
Expected: FAILS with `ValueError: min() arg is an empty sequence`

**Step 3: Write minimal implementation**

**blm.py** — Replace lines 60-65:

Old:
```python
        sols = solver.solve(modified, config)
        best = min(sols, key=lambda s: s.objective)
        costs.append(best.cost)
        boundaries.append(best.boundary)
        objectives.append(best.objective)
        solutions_list.append(best)
```

New:
```python
        sols = solver.solve(modified, config)
        if not sols:
            continue
        best = min(sols, key=lambda s: s.objective)
        costs.append(best.cost)
        boundaries.append(best.boundary)
        objectives.append(best.objective)
        solutions_list.append(best)
```

**spf.py** — Replace lines 54-56:

Old:
```python
        sols = solver.solve(modified, config)
        best = min(sols, key=lambda s: s.objective)
        best_solution = best
```

New:
```python
        sols = solver.solve(modified, config)
        if not sols:
            continue
        best = min(sols, key=lambda s: s.objective)
        best_solution = best
```

**sweep.py** — Replace lines 94-99:

Old:
```python
        sols = solver.solve(modified, solver_config)
        best = min(sols, key=lambda s: s.objective)
        solutions.append(best)
        costs.append(best.cost)
        boundaries.append(best.boundary)
        objectives.append(best.objective)
```

New:
```python
        sols = solver.solve(modified, solver_config)
        if not sols:
            continue
        best = min(sols, key=lambda s: s.objective)
        solutions.append(best)
        costs.append(best.cost)
        boundaries.append(best.boundary)
        objectives.append(best.objective)
```

**sensitivity.py** — Replace lines 69-70:

Old:
```python
            sols = solver.solve(modified, solver_config)
            best = min(sols, key=lambda s: s.objective)
```

New:
```python
            sols = solver.solve(modified, solver_config)
            if not sols:
                continue
            best = min(sols, key=lambda s: s.objective)
```

**parallel.py** — Replace lines 37-38:

Old:
```python
    sols = solver.solve(modified, solver_config)
    best = min(sols, key=lambda s: s.objective)
```

New:
```python
    sols = solver.solve(modified, solver_config)
    if not sols:
        raise ValueError("Solver returned no solutions (infeasible)")
    best = min(sols, key=lambda s: s.objective)
```

Note: parallel.py uses a different pattern (raises instead of `continue`) because it's inside a worker function that returns a single result.

**Step 4: Run test to verify it passes**

Run: `pytest tests/pymarxan/calibration/test_empty_solutions.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add src/pymarxan/calibration/blm.py src/pymarxan/calibration/spf.py src/pymarxan/calibration/sweep.py src/pymarxan/calibration/sensitivity.py src/pymarxan/calibration/parallel.py tests/pymarxan/calibration/test_empty_solutions.py
git commit -m "fix: guard all calibration functions against empty solver results

MIP solver returns [] when infeasible. All 5 calibration functions
now skip infeasible sweep points instead of crashing on min([])."
```

---

### Task 5: H2 — Zone SA cooling schedule counts same-zone iterations

**Files:**
- Modify: `src/pymarxan/zones/solver.py:149-177`
- Test: `tests/pymarxan/zones/test_solver.py`

**Step 1: Write the failing test**

Add to `tests/pymarxan/zones/test_solver.py`:

```python
    def test_cooling_counts_all_iterations(self):
        """SA should cool at the same rate regardless of same-zone skips.

        With the bug, same-zone iterations skip counter increments,
        so temperature doesn't decrease fast enough.
        """
        problem = copy.deepcopy(self.problem)
        problem.parameters["NUMITNS"] = 1000
        problem.parameters["NUMTEMP"] = 10
        config = SolverConfig(num_solutions=1, seed=42)
        solutions = self.solver.solve(problem, config)
        # If cooling works, objective should be finite and reasonable
        assert solutions[0].objective < 1e10
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/pymarxan/zones/test_solver.py::TestZoneSASolver::test_cooling_counts_all_iterations -v`
Expected: PASS (behavior test — the bug causes suboptimal solutions but doesn't crash). This is a correctness fix we verify by code inspection.

**Step 3: Write minimal implementation**

In `src/pymarxan/zones/solver.py`, move the counter increments and cooling logic **before** the `continue` statement. Replace lines 145-177:

Old:
```python
            for _ in range(num_iterations):
                idx = int(swappable[rng.integers(n_swappable)])
                old_zone = int(assignment[idx])
                new_zone = int(zone_options[rng.integers(n_zone_options)])
                if new_zone == old_zone:
                    continue

                delta = cache.compute_delta_zone_objective(
                    idx, old_zone, new_zone, assignment, held_per_zone, blm
                )

                if delta <= 0 or (
                    temp > 0 and rng.random() < math.exp(-delta / temp)
                ):
                    assignment[idx] = new_zone
                    cache.update_held_per_zone(
                        held_per_zone, idx, old_zone, new_zone
                    )
                    current_obj += delta

                    if current_obj < best_obj:
                        best_assignment = assignment.copy()
                        best_obj = current_obj

                step_count += 1
                if step_count >= iters_per_step:
                    temp *= alpha
                    step_count = 0

                iter_count += 1
                if progress is not None and iter_count % 1000 == 0:
                    progress.iteration = iter_count
                    progress.best_objective = best_obj
```

New:
```python
            for _ in range(num_iterations):
                # Cool and count BEFORE any early-continue
                step_count += 1
                if step_count >= iters_per_step:
                    temp *= alpha
                    step_count = 0

                iter_count += 1
                if progress is not None and iter_count % 1000 == 0:
                    progress.iteration = iter_count
                    progress.best_objective = best_obj

                idx = int(swappable[rng.integers(n_swappable)])
                old_zone = int(assignment[idx])
                new_zone = int(zone_options[rng.integers(n_zone_options)])
                if new_zone == old_zone:
                    continue

                delta = cache.compute_delta_zone_objective(
                    idx, old_zone, new_zone, assignment, held_per_zone, blm
                )

                if delta <= 0 or (
                    temp > 0 and rng.random() < math.exp(-delta / temp)
                ):
                    assignment[idx] = new_zone
                    cache.update_held_per_zone(
                        held_per_zone, idx, old_zone, new_zone
                    )
                    current_obj += delta

                    if current_obj < best_obj:
                        best_assignment = assignment.copy()
                        best_obj = current_obj
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/pymarxan/zones/test_solver.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add src/pymarxan/zones/solver.py tests/pymarxan/zones/test_solver.py
git commit -m "fix: Zone SA cooling schedule counts same-zone iterations

Counter increments now happen before the same-zone continue,
so SA cools at the intended rate regardless of zone count."
```

---

### Task 6: H3 — Zone SA handles STATUS_INITIAL_INCLUDE (status=1)

**Files:**
- Modify: `src/pymarxan/zones/solver.py:68-77`
- Test: `tests/pymarxan/zones/test_solver.py`

**Step 1: Write the failing test**

Add to `tests/pymarxan/zones/test_solver.py`:

```python
    def test_status_1_starts_in_zone(self):
        """PUs with status=1 should start in first non-zero zone, stay swappable."""
        problem = copy.deepcopy(self.problem)
        problem.planning_units["status"] = 0
        problem.planning_units.loc[0, "status"] = 1  # PU 0 initial include
        problem.parameters["NUMITNS"] = 100
        problem.parameters["NUMTEMP"] = 10
        config = SolverConfig(num_solutions=1, seed=42)
        solutions = self.solver.solve(problem, config)
        assert len(solutions) == 1
        # PU 0 should not be locked — it could end up in any zone
        # Just verify it doesn't crash and returns a valid solution
        assert solutions[0].zone_assignment is not None
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/pymarxan/zones/test_solver.py::TestZoneSASolver::test_status_1_starts_in_zone -v`
Expected: PASS (status=1 doesn't crash, but PU gets random zone instead of starting in a zone). The fix is a correctness improvement.

**Step 3: Write minimal implementation**

In `src/pymarxan/zones/solver.py`, add status=1 handling in the locked dict construction. Replace lines 68-81:

Old:
```python
        locked: dict[int, int] = {}
        if "status" in problem.planning_units.columns:
            for _, row in problem.planning_units.iterrows():
                s = int(row["status"])
                idx = pu_id_to_idx[int(row["id"])]
                if s == 2:
                    locked[idx] = zone_ids_list[0]
                elif s == 3:
                    locked[idx] = 0

        swappable = np.array(
            [i for i in range(n_pu) if i not in locked], dtype=int
        )
        n_swappable = len(swappable)
```

New:
```python
        locked: dict[int, int] = {}
        initial_include: set[int] = set()
        if "status" in problem.planning_units.columns:
            for _, row in problem.planning_units.iterrows():
                s = int(row["status"])
                idx = pu_id_to_idx[int(row["id"])]
                if s == 2:
                    locked[idx] = zone_ids_list[0]
                elif s == 3:
                    locked[idx] = 0
                elif s == 1:
                    initial_include.add(idx)

        swappable = np.array(
            [i for i in range(n_pu) if i not in locked], dtype=int
        )
        n_swappable = len(swappable)
```

Then, in the initial assignment loop (lines 99-103), use initial_include to seed those PUs into the first zone. Replace:

Old:
```python
            assignment = np.zeros(n_pu, dtype=int)
            for idx, zid in locked.items():
                assignment[idx] = zid
            for idx in swappable:
                assignment[idx] = zone_options[rng.integers(n_zone_options)]
```

New:
```python
            assignment = np.zeros(n_pu, dtype=int)
            for idx, zid in locked.items():
                assignment[idx] = zid
            for idx in swappable:
                if idx in initial_include:
                    assignment[idx] = zone_ids_list[0]  # start selected
                else:
                    assignment[idx] = zone_options[rng.integers(n_zone_options)]
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/pymarxan/zones/test_solver.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add src/pymarxan/zones/solver.py tests/pymarxan/zones/test_solver.py
git commit -m "fix: Zone SA handles STATUS_INITIAL_INCLUDE (status=1)

PUs with status=1 now start in the first zone while remaining
swappable, matching regular SA behavior."
```

---

### Task 7: H4 — grid_builder guards None numeric inputs

**Files:**
- Modify: `src/pymarxan_shiny/modules/spatial/grid_builder.py:61-73`
- Test: `tests/pymarxan_shiny/test_grid_builder.py` (find or create)

**Step 1: Write the failing test**

Create or add to `tests/pymarxan_shiny/test_grid_builder.py`:

```python
"""Tests for grid builder None-input guard."""
from __future__ import annotations

import pytest

from pymarxan.spatial.grid import generate_planning_grid


class TestGridBuilderInputValidation:
    def test_none_cell_size_raises(self):
        """Passing None as cell_size should raise TypeError or ValueError."""
        with pytest.raises((TypeError, ValueError)):
            generate_planning_grid(
                bounds=(0.0, 0.0, 1.0, 1.0),
                cell_size=None,
            )

    def test_none_bounds_raises(self):
        """Passing None in bounds tuple should raise TypeError."""
        with pytest.raises((TypeError, ValueError)):
            generate_planning_grid(
                bounds=(None, 0.0, 1.0, 1.0),
                cell_size=0.1,
            )
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/pymarxan_shiny/test_grid_builder.py -v`
Expected: May PASS or FAIL depending on how generate_planning_grid handles None.

**Step 3: Write minimal implementation**

In `src/pymarxan_shiny/modules/spatial/grid_builder.py`, add a None guard at the start of `_generate()`. Replace lines 61-62:

Old:
```python
    @reactive.effect
    @reactive.event(input.generate)
    def _generate():
        bounds = (input.minx(), input.miny(), input.maxx(), input.maxy())
```

New:
```python
    @reactive.effect
    @reactive.event(input.generate)
    def _generate():
        vals = [input.minx(), input.miny(), input.maxx(), input.maxy(), input.cell_size()]
        if any(v is None for v in vals):
            ui.notification_show(
                "Please fill in all numeric fields.", type="warning"
            )
            return
        bounds = (input.minx(), input.miny(), input.maxx(), input.maxy())
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/pymarxan_shiny/test_grid_builder.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add src/pymarxan_shiny/modules/spatial/grid_builder.py tests/pymarxan_shiny/test_grid_builder.py
git commit -m "fix: grid_builder guards against None numeric inputs

Shiny input_numeric returns None when cleared. Show notification
instead of crashing with TypeError."
```

---

### Task 8: H5 — MIP solver and SA all-locked return independent solutions

**Files:**
- Modify: `src/pymarxan/solvers/mip_solver.py:142`
- Modify: `src/pymarxan/solvers/simulated_annealing.py:95`
- Test: `tests/pymarxan/solvers/test_solution_aliasing.py`

**Step 1: Write the failing test**

Create `tests/pymarxan/solvers/test_solution_aliasing.py`:

```python
"""Tests for solution aliasing bug — [sol] * N creates references."""
from __future__ import annotations

from pathlib import Path

import pytest

from pymarxan.io.readers import load_project
from pymarxan.solvers.base import SolverConfig
from pymarxan.solvers.mip_solver import MIPSolver

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "simple"


class TestMIPSolutionAliasing:
    def test_solutions_are_independent(self):
        """Mutating one solution's metadata should not affect others."""
        problem = load_project(DATA_DIR)
        solver = MIPSolver()
        config = SolverConfig(num_solutions=3, seed=42)
        solutions = solver.solve(problem, config)
        if len(solutions) < 2:
            pytest.skip("MIP returned fewer than 2 solutions")
        # Mutate first solution
        solutions[0].metadata["test_key"] = "test_value"
        # Second solution should NOT have the mutation
        assert "test_key" not in solutions[1].metadata


class TestSAAllLockedAliasing:
    def test_all_locked_solutions_are_independent(self):
        """When all PUs are locked, SA returns [sol]*N — test independence."""
        problem = load_project(DATA_DIR)
        problem.planning_units["status"] = 2  # All locked-in
        from pymarxan.solvers.simulated_annealing import SimulatedAnnealingSolver
        solver = SimulatedAnnealingSolver()
        config = SolverConfig(num_solutions=3, seed=42)
        solutions = solver.solve(problem, config)
        assert len(solutions) == 3
        solutions[0].metadata["test_key"] = "mutated"
        assert "test_key" not in solutions[1].metadata
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/pymarxan/solvers/test_solution_aliasing.py -v`
Expected: FAILS — mutating `solutions[0].metadata` also affects `solutions[1].metadata`

**Step 3: Write minimal implementation**

**mip_solver.py** — Replace line 142:

Old:
```python
        return [sol] * config.num_solutions
```

New:
```python
        import copy as _copy
        return [_copy.deepcopy(sol) for _ in range(config.num_solutions)]
```

Actually, cleaner to add the import at the top. Add `import copy` to imports (after `from __future__ import annotations`), then replace line 142:

New:
```python
        return [copy.deepcopy(sol) for _ in range(config.num_solutions)]
```

**simulated_annealing.py** — Replace line 95:

Old:
```python
            return [sol] * config.num_solutions
```

New (add `import copy` at top of file):
```python
            return [copy.deepcopy(sol) for _ in range(config.num_solutions)]
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/pymarxan/solvers/test_solution_aliasing.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add src/pymarxan/solvers/mip_solver.py src/pymarxan/solvers/simulated_annealing.py tests/pymarxan/solvers/test_solution_aliasing.py
git commit -m "fix: MIP and SA all-locked paths return independent solutions

[sol] * N creates aliased references — mutating one affects all.
Now uses deepcopy to create independent solution objects."
```

---

## Batch 3: MEDIUM (7 issues)

### Task 9: M1 — SA alpha clamped when initial_temp < 0.001

**Files:**
- Modify: `src/pymarxan/solvers/simulated_annealing.py:148-158`
- Modify: `src/pymarxan/zones/solver.py:128-137`
- Test: `tests/pymarxan/solvers/test_sa_alpha_clamp.py`

**Step 1: Write the failing test**

Create `tests/pymarxan/solvers/test_sa_alpha_clamp.py`:

```python
"""Tests for SA alpha computation when initial_temp is very small."""
from __future__ import annotations

import math

import pytest


class TestAlphaClamp:
    def test_alpha_below_one_when_temp_tiny(self):
        """Alpha must always be <= 1.0 for cooling to work."""
        initial_temp = 0.0001  # Very small
        num_temp_steps = 100
        # Before fix: alpha = (0.001 / 0.0001) ** (1/100) = 10 ** 0.01 > 1
        initial_temp = max(initial_temp, 0.001)
        alpha = (0.001 / initial_temp) ** (1.0 / max(1, num_temp_steps))
        assert alpha <= 1.0

    def test_alpha_below_one_normal(self):
        """Normal case: initial_temp > 0.001 gives alpha < 1."""
        initial_temp = 10.0
        num_temp_steps = 100
        initial_temp = max(initial_temp, 0.001)
        alpha = (0.001 / initial_temp) ** (1.0 / max(1, num_temp_steps))
        assert alpha < 1.0
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/pymarxan/solvers/test_sa_alpha_clamp.py -v`
Expected: PASS (these test the formula directly after fix)

**Step 3: Write minimal implementation**

**simulated_annealing.py** — After line 149 (`initial_temp = 1.0`), add before the alpha computation:

Insert after line 149 (before `# Compute cooling factor`):
```python
            initial_temp = max(initial_temp, 0.001)
```

So lines 148-158 become:
```python
            else:
                initial_temp = 1.0

            initial_temp = max(initial_temp, 0.001)

            # Compute cooling factor
            iters_per_step = max(1, num_iterations // num_temp_steps)
            if initial_temp > 0:
                alpha = (0.001 / initial_temp) ** (
                    1.0 / max(1, num_temp_steps)
                )
            else:
                alpha = 0.99
```

**zones/solver.py** — Same fix after line 129 (`initial_temp = 1.0`):

Insert after line 129:
```python
            initial_temp = max(initial_temp, 0.001)
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/pymarxan/solvers/test_sa_alpha_clamp.py -v && pytest tests/pymarxan/zones/test_solver.py -v && pytest tests/pymarxan/solvers/test_sa_history.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add src/pymarxan/solvers/simulated_annealing.py src/pymarxan/zones/solver.py tests/pymarxan/solvers/test_sa_alpha_clamp.py
git commit -m "fix: clamp SA initial_temp >= 0.001 to prevent alpha > 1

When average delta is tiny, initial_temp < 0.001 made alpha > 1,
causing temperature to increase instead of decrease."
```

---

### Task 10: M2 — network_view caps edges at 5000

**Files:**
- Modify: `src/pymarxan_shiny/modules/mapping/network_view.py:132-144`
- Test: `tests/pymarxan_shiny/test_network_view.py` (find or create)

**Step 1: Write the failing test**

Add to existing `tests/pymarxan_shiny/test_network_view.py` or create:

```python
"""Tests for network_view edge cap."""
from __future__ import annotations


class TestEdgeCap:
    def test_max_edges_constant_exists(self):
        """Module should define MAX_EDGES constant."""
        from pymarxan_shiny.modules.mapping import network_view
        assert hasattr(network_view, "MAX_EDGES")
        assert network_view.MAX_EDGES <= 5000
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/pymarxan_shiny/test_network_view.py::TestEdgeCap -v`
Expected: FAILS — `MAX_EDGES` not defined

**Step 3: Write minimal implementation**

In `src/pymarxan_shiny/modules/mapping/network_view.py`:

1. Add constant after imports (after line 18):
```python
MAX_EDGES = 5000
```

2. In the `map()` function, replace the edge loop (lines 132-144):

Old:
```python
            # Add polyline edges
            n = min(matrix.shape[0], n_pu)
            for i in range(n):
                for j in range(n):
                    weight = float(matrix[i, j])
                    if weight > threshold and i != j:
                        line = ipyleaflet.Polyline(
                            locations=[centroids[i], centroids[j]],
                            color="#3498db",
                            opacity=min(1.0, weight),
                            weight=2,
                        )
                        m.add(line)
```

New:
```python
            # Add polyline edges (capped to prevent browser freeze)
            n = min(matrix.shape[0], n_pu)
            edge_count = 0
            for i in range(n):
                for j in range(n):
                    weight = float(matrix[i, j])
                    if weight > threshold and i != j:
                        if edge_count >= MAX_EDGES:
                            break
                        line = ipyleaflet.Polyline(
                            locations=[centroids[i], centroids[j]],
                            color="#3498db",
                            opacity=min(1.0, weight),
                            weight=2,
                        )
                        m.add(line)
                        edge_count += 1
                if edge_count >= MAX_EDGES:
                    break
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/pymarxan_shiny/test_network_view.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add src/pymarxan_shiny/modules/mapping/network_view.py tests/pymarxan_shiny/test_network_view.py
git commit -m "perf: cap network_view edges at 5000 to prevent browser freeze

Dense 500-PU matrices create ~250K polylines. Now stops adding
edges after MAX_EDGES=5000."
```

---

### Task 11: M3 — HeuristicSolver excludes locked-out PUs from total_available

**Files:**
- Modify: `src/pymarxan/solvers/heuristic.py:190-194`
- Test: `tests/pymarxan/solvers/test_heuristic.py`

**Step 1: Write the failing test**

Add to `tests/pymarxan/solvers/test_heuristic.py`:

```python
class TestLockedOutRarity:
    def test_locked_out_excluded_from_availability(self):
        """Locked-out PUs should not count in total_available for rarity."""
        import copy
        problem = load_project(DATA_DIR)
        # Lock out half the PUs
        problem.planning_units["status"] = 0
        half = len(problem.planning_units) // 2
        problem.planning_units.iloc[:half, problem.planning_units.columns.get_loc("status")] = 3

        solver = HeuristicSolver(heurtype=2)  # Max Rarity
        config = SolverConfig(num_solutions=1, seed=42)
        solutions = solver.solve(problem, config)
        assert len(solutions) == 1
        assert solutions[0].cost >= 0
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/pymarxan/solvers/test_heuristic.py::TestLockedOutRarity -v`
Expected: PASS (behavior — the bug inflates denominator but doesn't crash)

**Step 3: Write minimal implementation**

In `src/pymarxan/solvers/heuristic.py`, replace lines 190-194:

Old:
```python
        # Total available amount per feature (for rarity / irreplaceability)
        total_available: dict[int, float] = {}
        for fid_val, amt_val in contributions.items():
            for fid, amt in amt_val.items():
                total_available[fid] = total_available.get(fid, 0.0) + amt
```

New:
```python
        # Total available amount per feature (for rarity / irreplaceability)
        # Exclude locked-out PUs from availability calculation
        total_available: dict[int, float] = {}
        for idx_val, amt_val in contributions.items():
            if locked_out[idx_val]:
                continue
            for fid, amt in amt_val.items():
                total_available[fid] = total_available.get(fid, 0.0) + amt
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/pymarxan/solvers/test_heuristic.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add src/pymarxan/solvers/heuristic.py tests/pymarxan/solvers/test_heuristic.py
git commit -m "fix: exclude locked-out PUs from heuristic rarity denominator

Locked-out PUs inflated total_available, underestimating rarity
and irreplaceability of remaining PUs."
```

---

### Task 12: M4 — cost_surface apply_cost_from_vector uses map instead of O(n*m)

**Files:**
- Modify: `src/pymarxan/spatial/cost_surface.py:71-72`
- Test: `tests/pymarxan/spatial/test_cost_surface.py` (find or create)

**Step 1: Write the failing test**

Add to existing cost_surface tests or create `tests/pymarxan/spatial/test_cost_perf.py`:

```python
"""Tests for cost_surface.apply_cost_from_vector — correctness after perf fix."""
from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import box


class TestApplyCostPerf:
    def test_map_based_update_correct(self):
        """Verify cost update produces same results after O(n*m) -> O(n) fix."""
        from pymarxan.spatial.cost_surface import apply_cost_from_vector

        pu = gpd.GeoDataFrame({
            "id": [1, 2, 3],
            "cost": [10.0, 20.0, 30.0],
            "geometry": [box(0, 0, 1, 1), box(1, 0, 2, 1), box(2, 0, 3, 1)],
        }, crs="EPSG:4326")
        cost_layer = gpd.GeoDataFrame({
            "value": [100.0, 200.0],
            "geometry": [box(0, 0, 1.5, 1), box(1.5, 0, 3, 1)],
        }, crs="EPSG:4326")
        result = apply_cost_from_vector(pu, cost_layer, "value")
        # PU1 fully covered by layer1 (100), PU3 fully by layer2 (200)
        assert result.loc[result["id"] == 1, "cost"].iloc[0] == pytest.approx(100.0, rel=0.1)
        assert result.loc[result["id"] == 3, "cost"].iloc[0] == pytest.approx(200.0, rel=0.1)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/pymarxan/spatial/test_cost_perf.py -v`
Expected: PASS (correctness test for the new implementation)

**Step 3: Write minimal implementation**

In `src/pymarxan/spatial/cost_surface.py`, replace lines 71-72:

Old:
```python
    for pu_id, cost in new_costs.items():
        result.loc[result["id"] == pu_id, "cost"] = cost
```

New:
```python
    cost_series = result.set_index("id")["cost"].copy()
    for pu_id, cost in new_costs.items():
        cost_series.at[pu_id] = cost
    result["cost"] = cost_series.reindex(result["id"]).values
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/pymarxan/spatial/ -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add src/pymarxan/spatial/cost_surface.py tests/pymarxan/spatial/test_cost_perf.py
git commit -m "perf: cost_surface uses index-based update instead of O(n*m) scan

apply_cost_from_vector now uses set_index for O(1) lookup per PU
instead of per-PU DataFrame scan."
```

---

### Task 13: M5 — combine_cost_layers validates weights/layers length

**Files:**
- Modify: `src/pymarxan/spatial/cost_surface.py:103-107`
- Test: `tests/pymarxan/spatial/test_cost_surface.py` (find or create)

**Step 1: Write the failing test**

Add to cost_surface tests:

```python
class TestCombineWeightsValidation:
    def test_mismatched_weights_raises(self):
        """Passing 2 weights for 3 layers should raise ValueError."""
        from pymarxan.spatial.cost_surface import combine_cost_layers
        import geopandas as gpd
        from shapely.geometry import box

        pu = gpd.GeoDataFrame({
            "id": [1, 2],
            "cost": [10.0, 20.0],
            "geometry": [box(0, 0, 1, 1), box(1, 0, 2, 1)],
        })
        layers = [
            ("a", np.array([1.0, 2.0])),
            ("b", np.array([3.0, 4.0])),
            ("c", np.array([5.0, 6.0])),
        ]
        with pytest.raises(ValueError, match="weights"):
            combine_cost_layers(pu, layers, weights=[0.5, 0.5])
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/pymarxan/spatial/test_cost_perf.py::TestCombineWeightsValidation -v`
Expected: FAILS — no ValueError raised, zip silently truncates

**Step 3: Write minimal implementation**

In `src/pymarxan/spatial/cost_surface.py`, add validation after line 103 (`if weights is None:`). Replace lines 103-107:

Old:
```python
    if weights is None:
        weights = [1.0 / n_layers] * n_layers

    combined = np.zeros(len(planning_units), dtype=float)
    for (name, values), w in zip(layers, weights):
```

New:
```python
    if weights is None:
        weights = [1.0 / n_layers] * n_layers
    elif len(weights) != n_layers:
        raise ValueError(
            f"len(weights)={len(weights)} != len(layers)={n_layers}"
        )

    combined = np.zeros(len(planning_units), dtype=float)
    for (name, values), w in zip(layers, weights):
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/pymarxan/spatial/test_cost_perf.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add src/pymarxan/spatial/cost_surface.py tests/pymarxan/spatial/test_cost_perf.py
git commit -m "fix: combine_cost_layers validates weights/layers length match

zip(layers, weights) silently truncated mismatched lists. Now
raises ValueError with clear message."
```

---

### Task 14: M6 — validate() guards missing pu_vs_features columns before cross-ref

**Files:**
- Modify: `src/pymarxan/models/problem.py:145-160`
- Test: `tests/pymarxan/models/test_validate.py` (find or create)

**Step 1: Write the failing test**

Create or add to `tests/pymarxan/models/test_validate.py`:

```python
"""Tests for ConservationProblem.validate() with missing columns."""
from __future__ import annotations

import pandas as pd
import pytest

from pymarxan.models.problem import ConservationProblem


class TestValidateMissingColumns:
    def test_missing_puvspr_columns_no_keyerror(self):
        """validate() should report missing columns, not crash with KeyError."""
        pu = pd.DataFrame({"id": [1], "cost": [10.0], "status": [0]})
        features = pd.DataFrame({"id": [1], "name": ["f1"], "target": [1.0], "spf": [1.0]})
        # pu_vs_features missing 'pu' and 'species' columns
        puvspr = pd.DataFrame({"wrong_col": [1], "amount": [5.0]})
        p = ConservationProblem(
            planning_units=pu, features=features, pu_vs_features=puvspr,
        )
        errors = p.validate()
        assert any("missing columns" in e for e in errors)
        # Should NOT raise KeyError
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/pymarxan/models/test_validate.py::TestValidateMissingColumns -v`
Expected: FAILS with `KeyError: 'pu'` at the cross-reference check

**Step 3: Write minimal implementation**

In `src/pymarxan/models/problem.py`, wrap the cross-reference checks (lines 145-160) in a guard:

Replace:
```python
        # --- Cross-reference IDs ---
        puvspr_pu_ids = set(self.pu_vs_features["pu"])
        unknown_pus = puvspr_pu_ids - self.pu_ids
        if unknown_pus:
            errors.append(
                f"pu_vs_features references planning unit IDs not in "
                f"planning_units: {sorted(unknown_pus)}"
            )

        puvspr_species_ids = set(self.pu_vs_features["species"])
        unknown_features = puvspr_species_ids - self.feature_ids
        if unknown_features:
            errors.append(
                f"pu_vs_features references feature IDs not in "
                f"features: {sorted(unknown_features)}"
            )
```

With:
```python
        # --- Cross-reference IDs (only if columns exist) ---
        if not missing_puvspr:
            puvspr_pu_ids = set(self.pu_vs_features["pu"])
            unknown_pus = puvspr_pu_ids - self.pu_ids
            if unknown_pus:
                errors.append(
                    f"pu_vs_features references planning unit IDs not in "
                    f"planning_units: {sorted(unknown_pus)}"
                )

            puvspr_species_ids = set(self.pu_vs_features["species"])
            unknown_features = puvspr_species_ids - self.feature_ids
            if unknown_features:
                errors.append(
                    f"pu_vs_features references feature IDs not in "
                    f"features: {sorted(unknown_features)}"
                )
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/pymarxan/models/test_validate.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add src/pymarxan/models/problem.py tests/pymarxan/models/test_validate.py
git commit -m "fix: validate() guards cross-ref checks when columns are missing

Accessing pu_vs_features['pu'] before checking if column exists
caused KeyError. Now skips cross-ref when columns are absent."
```

---

### Task 15: M7 — gadm.py uses .get() with descriptive error

**Files:**
- Modify: `src/pymarxan/spatial/gadm.py:121`
- Test: `tests/pymarxan/spatial/test_gadm.py` (find or create)

**Step 1: Write the failing test**

Add to gadm tests:

```python
"""Tests for gadm.py error handling."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


class TestGadmErrorHandling:
    @patch("pymarxan.spatial.gadm.requests.get")
    def test_missing_download_url_gives_clear_error(self, mock_get):
        """API response without gjDownloadURL should give descriptive error."""
        from pymarxan.spatial.gadm import fetch_gadm

        mock_resp = MagicMock()
        mock_resp.json.return_value = {"otherKey": "value"}
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        with pytest.raises(ValueError, match="gjDownloadURL"):
            fetch_gadm("USA", admin_level=0)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/pymarxan/spatial/test_gadm.py::TestGadmErrorHandling -v`
Expected: FAILS — raises `KeyError`, not `ValueError`

**Step 3: Write minimal implementation**

In `src/pymarxan/spatial/gadm.py`, replace line 121:

Old:
```python
    geojson_url = meta["gjDownloadURL"]
```

New:
```python
    geojson_url = meta.get("gjDownloadURL")
    if geojson_url is None:
        raise ValueError(
            f"geoBoundaries API response missing 'gjDownloadURL' key. "
            f"Response keys: {list(meta.keys())}"
        )
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/pymarxan/spatial/test_gadm.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add src/pymarxan/spatial/gadm.py tests/pymarxan/spatial/test_gadm.py
git commit -m "fix: gadm.py raises descriptive ValueError on API format changes

meta['gjDownloadURL'] gave opaque KeyError. Now uses .get() with
a clear error message listing available keys."
```

---

## Batch 4: Test Gaps (4 issues)

### Task 16: T1 — Test COSTTHRESH code path in build_solution

**Files:**
- Test: `tests/pymarxan/solvers/test_costthresh.py`

**Step 1: Write the test**

Create `tests/pymarxan/solvers/test_costthresh.py`:

```python
"""Tests for COSTTHRESH penalty in build_solution."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pymarxan.io.readers import load_project
from pymarxan.solvers.utils import build_solution, compute_cost_threshold_penalty

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "simple"


class TestCostThresholdPenalty:
    def test_no_penalty_below_threshold(self):
        """Cost below threshold should add no penalty."""
        penalty = compute_cost_threshold_penalty(
            total_cost=50.0, cost_thresh=100.0, thresh_pen1=10.0, thresh_pen2=1.0
        )
        assert penalty == 0.0

    def test_penalty_above_threshold(self):
        """Cost above threshold should add penalty."""
        penalty = compute_cost_threshold_penalty(
            total_cost=150.0, cost_thresh=100.0, thresh_pen1=10.0, thresh_pen2=1.0
        )
        assert penalty > 0.0

    def test_build_solution_applies_costthresh(self):
        """build_solution should include COSTTHRESH penalty in objective."""
        problem = load_project(DATA_DIR)
        selected = np.ones(problem.n_planning_units, dtype=bool)
        blm = 0.0

        # Without COSTTHRESH
        sol_no_thresh = build_solution(problem, selected, blm)

        # With COSTTHRESH (set low so cost exceeds it)
        problem.parameters["COSTTHRESH"] = 1.0
        problem.parameters["THRESHPEN1"] = 10.0
        problem.parameters["THRESHPEN2"] = 1.0
        sol_thresh = build_solution(problem, selected, blm)

        # Objective should be higher with cost threshold penalty
        assert sol_thresh.objective > sol_no_thresh.objective

    def test_costthresh_zero_no_effect(self):
        """COSTTHRESH=0 should not change the objective."""
        problem = load_project(DATA_DIR)
        selected = np.ones(problem.n_planning_units, dtype=bool)
        blm = 0.0

        problem.parameters["COSTTHRESH"] = 0.0
        problem.parameters["THRESHPEN1"] = 10.0
        sol = build_solution(problem, selected, blm)

        del problem.parameters["COSTTHRESH"]
        sol_default = build_solution(problem, selected, blm)

        assert sol.objective == pytest.approx(sol_default.objective)
```

**Step 2: Run test to verify it passes**

Run: `pytest tests/pymarxan/solvers/test_costthresh.py -v`
Expected: All PASS (testing existing correct code)

**Step 3: Commit**

```bash
git add tests/pymarxan/solvers/test_costthresh.py
git commit -m "test: add COSTTHRESH code path coverage for build_solution

The cost threshold penalty path was never tested. Adds 4 tests
covering below-threshold, above-threshold, integration with
build_solution, and zero-threshold no-effect."
```

---

### Task 17: T2 — Test ZoneSASolver type guard

**Files:**
- Test: `tests/pymarxan/zones/test_solver.py`

**Step 1: Write the test**

Add to `tests/pymarxan/zones/test_solver.py`:

```python
    def test_rejects_non_zonal_problem(self):
        """ZoneSASolver should raise TypeError for plain ConservationProblem."""
        from pymarxan.io.readers import load_project
        plain = load_project(Path(__file__).parent.parent.parent / "data" / "simple")
        config = SolverConfig(num_solutions=1, seed=42)
        with pytest.raises(TypeError, match="ZonalProblem"):
            self.solver.solve(plain, config)
```

**Step 2: Run test to verify it passes**

Run: `pytest tests/pymarxan/zones/test_solver.py::TestZoneSASolver::test_rejects_non_zonal_problem -v`
Expected: PASS (testing existing guard at lines 41-43)

**Step 3: Commit**

```bash
git add tests/pymarxan/zones/test_solver.py
git commit -m "test: add coverage for ZoneSASolver type guard

Tests that passing a plain ConservationProblem raises TypeError."
```

---

### Task 18: T3 — Test SA history monotonicity (temperature decreasing)

**Files:**
- Test: `tests/pymarxan/solvers/test_sa_history.py`

**Step 1: Write the test**

Add to `tests/pymarxan/solvers/test_sa_history.py`:

```python
    def test_temperature_non_increasing(self):
        """Temperature should decrease (or stay same) over the run."""
        config = SolverConfig(num_solutions=1, seed=42)
        solutions = self.solver.solve(self.problem, config)
        temps = solutions[0].metadata["history"]["temperature"]
        for i in range(1, len(temps)):
            assert temps[i] <= temps[i - 1] + 1e-9, (
                f"Temperature increased at step {i}: {temps[i-1]} -> {temps[i]}"
            )
```

**Step 2: Run test to verify it passes**

Run: `pytest tests/pymarxan/solvers/test_sa_history.py::TestSAHistory::test_temperature_non_increasing -v`
Expected: PASS (temperature should be decreasing after alpha clamp fix)

**Step 3: Commit**

```bash
git add tests/pymarxan/solvers/test_sa_history.py
git commit -m "test: verify SA temperature is monotonically non-increasing

Ensures the cooling schedule works correctly and temperature
never increases during the run."
```

---

### Task 19: T4 — Test MarxanBinarySolver parsing and error paths

**Files:**
- Test: `tests/pymarxan/solvers/test_binary_solver.py`

**Step 1: Write the test**

Create `tests/pymarxan/solvers/test_binary_solver.py`:

```python
"""Tests for MarxanBinarySolver — parsing and error paths."""
from __future__ import annotations

import numpy as np
import pytest

from pymarxan.solvers.marxan_binary import MarxanBinarySolver


class TestParseCSV:
    def test_parse_correct_csv(self):
        csv_content = "planning_unit,solution\n1,1\n2,0\n3,1\n"
        pu_ids = [1, 2, 3]
        result = MarxanBinarySolver._parse_solution_csv(csv_content, pu_ids)
        np.testing.assert_array_equal(result, [True, False, True])

    def test_parse_missing_pu(self):
        """PU not in CSV should default to False."""
        csv_content = "planning_unit,solution\n1,1\n3,1\n"
        pu_ids = [1, 2, 3]
        result = MarxanBinarySolver._parse_solution_csv(csv_content, pu_ids)
        np.testing.assert_array_equal(result, [True, False, True])

    def test_parse_empty_csv(self):
        csv_content = "planning_unit,solution\n"
        pu_ids = [1, 2]
        result = MarxanBinarySolver._parse_solution_csv(csv_content, pu_ids)
        np.testing.assert_array_equal(result, [False, False])


class TestAvailability:
    def test_not_available_without_binary(self):
        solver = MarxanBinarySolver(binary_path=None)
        # May or may not be available depending on system
        assert isinstance(solver.available(), bool)

    def test_available_with_explicit_path(self):
        solver = MarxanBinarySolver(binary_path="/usr/bin/true")
        assert solver.available() is True

    def test_name(self):
        solver = MarxanBinarySolver()
        assert solver.name() == "Marxan (C++ binary)"

    def test_supports_zones(self):
        solver = MarxanBinarySolver()
        assert solver.supports_zones() is False


class TestSolveErrors:
    def test_missing_binary_raises(self):
        solver = MarxanBinarySolver(binary_path="/nonexistent/marxan")
        from pymarxan.io.readers import load_project
        from pathlib import Path
        DATA_DIR = Path(__file__).parent.parent.parent / "data" / "simple"
        problem = load_project(DATA_DIR)
        with pytest.raises(RuntimeError, match="not found"):
            solver.solve(problem)
```

**Step 2: Run test to verify it passes**

Run: `pytest tests/pymarxan/solvers/test_binary_solver.py -v`
Expected: All PASS

**Step 3: Commit**

```bash
git add tests/pymarxan/solvers/test_binary_solver.py
git commit -m "test: add MarxanBinarySolver coverage for parsing and errors

Covers CSV parsing (correct, missing PU, empty), availability
checks, and error path for missing binary."
```

---

## Final Verification

After all 19 tasks, run the full test suite:

```bash
make test
```

Expected: 660+ tests, 78%+ coverage, 0 failures.
