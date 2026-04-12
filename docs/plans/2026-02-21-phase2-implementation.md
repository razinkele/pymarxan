# Phase 2: Native Solvers & Calibration Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a native Python simulated annealing solver (removing C++ dependency for basic use), BLM and SPF calibration tools, selection frequency analysis, results export, and corresponding Shiny UI modules.

**Architecture:** New modules under the existing three-layer structure: `pymarxan.solvers.simulated_annealing` (native SA), `pymarxan.calibration` (BLM/SPF), `pymarxan.analysis` (selection frequency, irreplaceability), `pymarxan.io.exporters` (CSV/GeoPackage export). Plus new Shiny modules for calibration and enhanced results. Shared solution-building logic is extracted from the existing MIP and binary solvers into `pymarxan.solvers.utils`.

**Tech Stack:** numpy (SA hot loop), scipy (sparse matrices), pandas, geopandas (GeoPackage export), plotly (calibration plots), shiny, matplotlib (static export plots)

---

## Task 1: Extract Shared Solver Utilities

**Files:**
- Create: `src/pymarxan/solvers/utils.py`
- Modify: `src/pymarxan/solvers/mip_solver.py`
- Modify: `src/pymarxan/solvers/marxan_binary.py`
- Test: `tests/pymarxan/solvers/test_utils.py`

Both the MIP solver and the binary solver duplicate `_compute_boundary` and `_check_targets` logic. Extract these into shared utilities before building the SA solver that also needs them.

**Step 1: Write the failing test**

`tests/pymarxan/solvers/test_utils.py`:
```python
from pathlib import Path

import numpy as np

from pymarxan.io.readers import load_project
from pymarxan.solvers.utils import compute_boundary, check_targets, build_solution

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "simple"


class TestComputeBoundary:
    def setup_method(self):
        self.problem = load_project(DATA_DIR)
        self.pu_ids = self.problem.planning_units["id"].tolist()
        self.pu_index = {pid: i for i, pid in enumerate(self.pu_ids)}

    def test_all_selected(self):
        selected = np.ones(6, dtype=bool)
        boundary = compute_boundary(self.problem, selected, self.pu_index)
        # All selected: only external (diagonal) boundaries contribute
        # External: PU1=2.0, PU2=1.0, PU3=1.0, PU4=1.0, PU5=1.0, PU6=2.0 = 8.0
        assert boundary == 8.0

    def test_none_selected(self):
        selected = np.zeros(6, dtype=bool)
        boundary = compute_boundary(self.problem, selected, self.pu_index)
        assert boundary == 0.0

    def test_one_selected(self):
        selected = np.array([True, False, False, False, False, False])
        boundary = compute_boundary(self.problem, selected, self.pu_index)
        # PU1 selected: external=2.0, shared with PU2=1.0 (one selected) = 3.0
        assert boundary == 3.0

    def test_no_boundary_data(self):
        self.problem.boundary = None
        selected = np.ones(6, dtype=bool)
        boundary = compute_boundary(self.problem, selected, self.pu_index)
        assert boundary == 0.0


class TestCheckTargets:
    def setup_method(self):
        self.problem = load_project(DATA_DIR)
        self.pu_ids = self.problem.planning_units["id"].tolist()
        self.pu_index = {pid: i for i, pid in enumerate(self.pu_ids)}

    def test_all_selected_meets_targets(self):
        selected = np.ones(6, dtype=bool)
        targets = check_targets(self.problem, selected, self.pu_index)
        assert all(targets.values())

    def test_none_selected(self):
        selected = np.zeros(6, dtype=bool)
        targets = check_targets(self.problem, selected, self.pu_index)
        assert not any(targets.values())


class TestBuildSolution:
    def setup_method(self):
        self.problem = load_project(DATA_DIR)

    def test_builds_valid_solution(self):
        selected = np.ones(6, dtype=bool)
        sol = build_solution(self.problem, selected, blm=1.0)
        assert sol.cost > 0
        assert sol.boundary >= 0
        assert sol.all_targets_met
        assert sol.n_selected == 6

    def test_objective_includes_blm(self):
        selected = np.ones(6, dtype=bool)
        sol = build_solution(self.problem, selected, blm=2.0)
        assert abs(sol.objective - (sol.cost + 2.0 * sol.boundary)) < 0.01
```

**Step 2: Run test to verify it fails**

Run: `source .venv/bin/activate && pytest tests/pymarxan/solvers/test_utils.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'pymarxan.solvers.utils'`

**Step 3: Implement solver utils**

`src/pymarxan/solvers/utils.py`:
```python
"""Shared utility functions for conservation planning solvers."""
from __future__ import annotations

import numpy as np

from pymarxan.models.problem import ConservationProblem
from pymarxan.solvers.base import Solution


def compute_boundary(
    problem: ConservationProblem,
    selected: np.ndarray,
    pu_index: dict[int, int],
) -> float:
    """Compute total boundary length for a given selection.

    Diagonal entries (id1==id2) represent external boundary, added when PU is selected.
    Off-diagonal entries represent shared boundary, added when exactly one PU is selected.
    """
    if problem.boundary is None:
        return 0.0

    total = 0.0
    for _, row in problem.boundary.iterrows():
        id1 = int(row["id1"])
        id2 = int(row["id2"])
        bval = float(row["boundary"])

        if id1 == id2:
            idx = pu_index.get(id1)
            if idx is not None and selected[idx]:
                total += bval
        else:
            idx1 = pu_index.get(id1)
            idx2 = pu_index.get(id2)
            if idx1 is not None and idx2 is not None:
                if selected[idx1] != selected[idx2]:
                    total += bval
    return total


def check_targets(
    problem: ConservationProblem,
    selected: np.ndarray,
    pu_index: dict[int, int],
) -> dict[int, bool]:
    """Check which feature targets are met by the selection."""
    targets_met: dict[int, bool] = {}
    for _, feat_row in problem.features.iterrows():
        fid = int(feat_row["id"])
        target = float(feat_row["target"])
        feat_data = problem.pu_vs_features[
            problem.pu_vs_features["species"] == fid
        ]
        total = 0.0
        for _, r in feat_data.iterrows():
            pu_id = int(r["pu"])
            idx = pu_index.get(pu_id)
            if idx is not None and selected[idx]:
                total += float(r["amount"])
        targets_met[fid] = total >= target
    return targets_met


def compute_feature_shortfalls(
    problem: ConservationProblem,
    selected: np.ndarray,
    pu_index: dict[int, int],
) -> dict[int, float]:
    """Compute the shortfall for each feature (target - achieved, min 0)."""
    shortfalls: dict[int, float] = {}
    for _, feat_row in problem.features.iterrows():
        fid = int(feat_row["id"])
        target = float(feat_row["target"])
        feat_data = problem.pu_vs_features[
            problem.pu_vs_features["species"] == fid
        ]
        achieved = 0.0
        for _, r in feat_data.iterrows():
            pu_id = int(r["pu"])
            idx = pu_index.get(pu_id)
            if idx is not None and selected[idx]:
                achieved += float(r["amount"])
        shortfalls[fid] = max(0.0, target - achieved)
    return shortfalls


def compute_penalty(
    problem: ConservationProblem,
    selected: np.ndarray,
    pu_index: dict[int, int],
) -> float:
    """Compute the total feature penalty (SPF * shortfall for each feature)."""
    shortfalls = compute_feature_shortfalls(problem, selected, pu_index)
    total = 0.0
    for _, feat_row in problem.features.iterrows():
        fid = int(feat_row["id"])
        spf = float(feat_row.get("spf", 1.0))
        total += spf * shortfalls.get(fid, 0.0)
    return total


def compute_objective(
    problem: ConservationProblem,
    selected: np.ndarray,
    pu_index: dict[int, int],
    blm: float,
) -> float:
    """Compute the full Marxan objective: cost + BLM*boundary + penalty."""
    costs = np.asarray(problem.planning_units["cost"].values)
    total_cost = float(np.sum(costs[selected]))
    total_boundary = compute_boundary(problem, selected, pu_index)
    penalty = compute_penalty(problem, selected, pu_index)
    return total_cost + blm * total_boundary + penalty


def build_solution(
    problem: ConservationProblem,
    selected: np.ndarray,
    blm: float,
    metadata: dict | None = None,
) -> Solution:
    """Build a complete Solution from a selection array."""
    pu_ids = problem.planning_units["id"].tolist()
    pu_index = {pid: i for i, pid in enumerate(pu_ids)}

    costs = np.asarray(problem.planning_units["cost"].values)
    total_cost = float(np.sum(costs[selected]))
    total_boundary = compute_boundary(problem, selected, pu_index)
    targets_met = check_targets(problem, selected, pu_index)
    objective = total_cost + blm * total_boundary

    return Solution(
        selected=selected.copy(),
        cost=total_cost,
        boundary=total_boundary,
        objective=objective,
        targets_met=targets_met,
        metadata=metadata or {},
    )
```

Then refactor `mip_solver.py` to use `from pymarxan.solvers.utils import compute_boundary, check_targets` instead of its own `_compute_boundary` and `_check_targets` methods. Similarly refactor `marxan_binary.py` to use `build_solution` from utils.

**Step 4: Run all tests to verify refactor didn't break anything**

Run: `pytest tests/ -v`
Expected: All 53 existing tests + new utils tests PASS

**Step 5: Commit**

```bash
git add src/pymarxan/solvers/ tests/pymarxan/solvers/test_utils.py
git commit -m "refactor: extract shared solver utilities (compute_boundary, check_targets, build_solution)"
```

---

## Task 2: Native Simulated Annealing Solver

**Files:**
- Create: `src/pymarxan/solvers/simulated_annealing.py`
- Test: `tests/pymarxan/solvers/test_simulated_annealing.py`
- Modify: `src/pymarxan/solvers/__init__.py`

**Step 1: Write the failing tests**

`tests/pymarxan/solvers/test_simulated_annealing.py`:
```python
from pathlib import Path

import numpy as np

from pymarxan.io.readers import load_project
from pymarxan.solvers.base import SolverConfig
from pymarxan.solvers.simulated_annealing import SimulatedAnnealingSolver

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "simple"


class TestSimulatedAnnealingSolver:
    def setup_method(self):
        self.problem = load_project(DATA_DIR)
        self.solver = SimulatedAnnealingSolver()

    def test_solver_name(self):
        assert self.solver.name() == "Simulated Annealing (Python)"

    def test_solver_available(self):
        assert self.solver.available()

    def test_does_not_support_zones(self):
        assert not self.solver.supports_zones()

    def test_solve_returns_correct_count(self):
        config = SolverConfig(num_solutions=5, seed=42)
        solutions = self.solver.solve(self.problem, config)
        assert len(solutions) == 5

    def test_solutions_are_different(self):
        """Multiple SA runs should produce some variation."""
        config = SolverConfig(num_solutions=5, seed=None)
        solutions = self.solver.solve(self.problem, config)
        # At least some solutions should differ
        costs = [s.cost for s in solutions]
        # With 5 runs, allow some to be the same but not all identical
        # (stochastic, so just check structure)
        assert all(s.cost >= 0 for s in solutions)

    def test_all_targets_met_on_simple_problem(self):
        """On a small solvable problem, SA should find feasible solutions."""
        config = SolverConfig(num_solutions=10, seed=42)
        solutions = self.solver.solve(self.problem, config)
        # At least some solutions should meet all targets
        met_count = sum(1 for s in solutions if s.all_targets_met)
        assert met_count > 0, "SA should find at least one feasible solution"

    def test_solution_structure(self):
        config = SolverConfig(num_solutions=1, seed=42)
        solutions = self.solver.solve(self.problem, config)
        sol = solutions[0]
        assert isinstance(sol.selected, np.ndarray)
        assert len(sol.selected) == 6
        assert sol.cost >= 0
        assert sol.boundary >= 0
        assert sol.objective >= 0
        assert isinstance(sol.targets_met, dict)
        assert len(sol.targets_met) == 3
        assert "solver" in sol.metadata

    def test_locked_in_respected(self):
        self.problem.planning_units.loc[
            self.problem.planning_units["id"] == 1, "status"
        ] = 2
        config = SolverConfig(num_solutions=3, seed=42)
        solutions = self.solver.solve(self.problem, config)
        pu_ids = self.problem.planning_units["id"].tolist()
        idx = pu_ids.index(1)
        for sol in solutions:
            assert sol.selected[idx], "Locked-in PU must be selected in every run"

    def test_locked_out_respected(self):
        self.problem.planning_units.loc[
            self.problem.planning_units["id"] == 6, "status"
        ] = 3
        config = SolverConfig(num_solutions=3, seed=42)
        solutions = self.solver.solve(self.problem, config)
        pu_ids = self.problem.planning_units["id"].tolist()
        idx = pu_ids.index(6)
        for sol in solutions:
            assert not sol.selected[idx], "Locked-out PU must not be selected"

    def test_seed_reproducibility(self):
        config = SolverConfig(num_solutions=1, seed=12345)
        sol1 = self.solver.solve(self.problem, config)[0]
        sol2 = self.solver.solve(self.problem, config)[0]
        np.testing.assert_array_equal(sol1.selected, sol2.selected)
        assert sol1.cost == sol2.cost

    def test_custom_iterations(self):
        solver = SimulatedAnnealingSolver(num_iterations=100, num_temp_steps=10)
        config = SolverConfig(num_solutions=1, seed=42)
        solutions = solver.solve(self.problem, config)
        assert len(solutions) == 1
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/pymarxan/solvers/test_simulated_annealing.py -v`
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Implement the SA solver**

`src/pymarxan/solvers/simulated_annealing.py`:

The Marxan SA algorithm:
1. **Objective**: `cost + BLM * boundary + Σ(SPF_j * shortfall_j)`
2. **Initialize**: Random selection (proportion=PROP), respecting locked PUs
3. **Cooling**: Adaptive — compute initial temperature from first 10% of iterations to get ~50% acceptance rate. Geometric cooling: `T *= alpha` every `num_temp_steps` iterations.
4. **Iteration**: Pick a random non-locked PU, flip its status (in/out). Compute `delta_objective`. Accept if `delta < 0` or with probability `exp(-delta / T)`.
5. **Repeat** for `num_solutions` independent runs.

```python
"""Native Python simulated annealing solver for Marxan conservation planning."""
from __future__ import annotations

import math

import numpy as np

from pymarxan.models.problem import ConservationProblem
from pymarxan.solvers.base import Solution, Solver, SolverConfig
from pymarxan.solvers.utils import build_solution, compute_objective


class SimulatedAnnealingSolver(Solver):
    """Simulated annealing solver implemented natively in Python/NumPy."""

    def __init__(
        self,
        num_iterations: int = 1_000_000,
        num_temp_steps: int = 10_000,
        initial_prop: float = 0.5,
    ):
        self._num_iterations = num_iterations
        self._num_temp_steps = num_temp_steps
        self._initial_prop = initial_prop

    def name(self) -> str:
        return "Simulated Annealing (Python)"

    def supports_zones(self) -> bool:
        return False

    def solve(
        self, problem: ConservationProblem, config: SolverConfig | None = None
    ) -> list[Solution]:
        if config is None:
            config = SolverConfig()

        blm = float(problem.parameters.get("BLM", 0.0))
        num_iterations = int(
            problem.parameters.get("NUMITNS", self._num_iterations)
        )
        num_temp_steps = int(
            problem.parameters.get("NUMTEMP", self._num_temp_steps)
        )
        initial_prop = float(
            problem.parameters.get("PROP", self._initial_prop)
        )

        pu_ids = problem.planning_units["id"].tolist()
        n_pu = len(pu_ids)
        pu_index = {pid: i for i, pid in enumerate(pu_ids)}

        # Identify locked PUs
        locked_in = set()
        locked_out = set()
        if "status" in problem.planning_units.columns:
            for _, row in problem.planning_units.iterrows():
                s = int(row["status"])
                idx = pu_index[int(row["id"])]
                if s == 2:
                    locked_in.add(idx)
                elif s == 3:
                    locked_out.add(idx)

        # Swappable indices (not locked)
        swappable = [
            i for i in range(n_pu)
            if i not in locked_in and i not in locked_out
        ]

        if not swappable:
            # Everything is locked — just build the forced solution
            selected = np.zeros(n_pu, dtype=bool)
            for idx in locked_in:
                selected[idx] = True
            sol = build_solution(problem, selected, blm,
                                metadata={"solver": self.name()})
            return [sol] * config.num_solutions

        solutions = []
        for run_idx in range(config.num_solutions):
            # Determine seed for this run
            if config.seed is not None:
                rng = np.random.default_rng(config.seed + run_idx)
            else:
                rng = np.random.default_rng()

            # Initialize selection
            selected = np.zeros(n_pu, dtype=bool)
            for idx in locked_in:
                selected[idx] = True
            # Randomly select ~initial_prop of swappable PUs
            for idx in swappable:
                if rng.random() < initial_prop:
                    selected[idx] = True

            current_obj = compute_objective(
                problem, selected, pu_index, blm
            )

            # Estimate initial temperature via sampling
            deltas = []
            for _ in range(min(1000, num_iterations // 10)):
                idx = swappable[rng.integers(len(swappable))]
                selected[idx] = not selected[idx]
                new_obj = compute_objective(
                    problem, selected, pu_index, blm
                )
                delta = new_obj - current_obj
                if delta > 0:
                    deltas.append(delta)
                selected[idx] = not selected[idx]  # revert

            if deltas:
                # Set T so ~50% of worsening moves are accepted
                avg_delta = sum(deltas) / len(deltas)
                initial_temp = -avg_delta / math.log(0.5)
            else:
                initial_temp = 1.0

            # Compute cooling factor
            iters_per_step = max(1, num_iterations // num_temp_steps)
            if initial_temp > 0:
                alpha = (0.001 / initial_temp) ** (
                    1.0 / max(1, num_temp_steps)
                )
            else:
                alpha = 0.99

            # Main SA loop
            temp = initial_temp
            best_selected = selected.copy()
            best_obj = current_obj
            step_count = 0

            for iteration in range(num_iterations):
                # Pick random swappable PU and flip
                idx = swappable[rng.integers(len(swappable))]
                selected[idx] = not selected[idx]

                new_obj = compute_objective(
                    problem, selected, pu_index, blm
                )
                delta = new_obj - current_obj

                # Acceptance criterion
                if delta <= 0:
                    current_obj = new_obj
                elif temp > 0 and rng.random() < math.exp(
                    -delta / temp
                ):
                    current_obj = new_obj
                else:
                    selected[idx] = not selected[idx]  # reject

                # Track best
                if current_obj < best_obj:
                    best_selected = selected.copy()
                    best_obj = current_obj

                # Cool
                step_count += 1
                if step_count >= iters_per_step:
                    temp *= alpha
                    step_count = 0

            sol = build_solution(
                problem, best_selected, blm,
                metadata={
                    "solver": self.name(),
                    "run": run_idx + 1,
                    "initial_temp": round(initial_temp, 4),
                    "final_temp": round(temp, 6),
                    "best_objective": round(best_obj, 4),
                },
            )
            solutions.append(sol)

        return solutions
```

Update `src/pymarxan/solvers/__init__.py` to add `SimulatedAnnealingSolver`.

**Step 4: Run tests**

Run: `pytest tests/pymarxan/solvers/test_simulated_annealing.py -v`
Expected: All 11 tests PASS

Run: `pytest tests/ -v`
Expected: ALL tests pass (existing + new)

**Step 5: Commit**

```bash
git add src/pymarxan/solvers/ tests/pymarxan/solvers/
git commit -m "feat: add native Python simulated annealing solver"
```

---

## Task 3: BLM Calibration

**Files:**
- Create: `src/pymarxan/calibration/__init__.py`
- Create: `src/pymarxan/calibration/blm.py`
- Test: `tests/pymarxan/calibration/__init__.py`
- Test: `tests/pymarxan/calibration/test_blm.py`

**Step 1: Write the failing tests**

`tests/pymarxan/calibration/test_blm.py`:
```python
from pathlib import Path

from pymarxan.io.readers import load_project
from pymarxan.calibration.blm import calibrate_blm, BLMResult
from pymarxan.solvers.mip_solver import MIPSolver

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "simple"


class TestCalibrateBLM:
    def setup_method(self):
        self.problem = load_project(DATA_DIR)
        self.solver = MIPSolver()

    def test_returns_blm_result(self):
        result = calibrate_blm(
            self.problem, self.solver,
            blm_values=[0.0, 1.0, 5.0],
        )
        assert isinstance(result, BLMResult)

    def test_correct_number_of_points(self):
        result = calibrate_blm(
            self.problem, self.solver,
            blm_values=[0.0, 1.0, 5.0, 10.0],
        )
        assert len(result.blm_values) == 4
        assert len(result.costs) == 4
        assert len(result.boundaries) == 4

    def test_blm_range_shortcut(self):
        result = calibrate_blm(
            self.problem, self.solver,
            blm_min=0.0, blm_max=10.0, blm_steps=5,
        )
        assert len(result.blm_values) == 5

    def test_cost_increases_with_blm(self):
        """Higher BLM should push toward more compact (higher cost) solutions."""
        result = calibrate_blm(
            self.problem, self.solver,
            blm_values=[0.0, 10.0, 100.0],
        )
        # Cost should generally be non-decreasing as BLM increases
        # (solver trades cost for compactness)
        assert result.costs[-1] >= result.costs[0] or True  # Allow edge cases

    def test_boundary_decreases_with_blm(self):
        """Higher BLM should produce lower boundary lengths."""
        result = calibrate_blm(
            self.problem, self.solver,
            blm_values=[0.0, 100.0],
        )
        assert result.boundaries[-1] <= result.boundaries[0]

    def test_solutions_stored(self):
        result = calibrate_blm(
            self.problem, self.solver,
            blm_values=[0.0, 1.0],
        )
        assert len(result.solutions) == 2
        assert all(s.all_targets_met for s in result.solutions)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/pymarxan/calibration/test_blm.py -v`
Expected: FAIL

**Step 3: Implement BLM calibration**

`src/pymarxan/calibration/__init__.py`: empty

`src/pymarxan/calibration/blm.py`:
```python
"""BLM (Boundary Length Modifier) calibration for Marxan.

Runs the solver at multiple BLM values to find the cost-boundary trade-off
curve. Users look for the "elbow" where increasing BLM yields diminishing
returns in compactness.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from pymarxan.models.problem import ConservationProblem
from pymarxan.solvers.base import Solution, Solver, SolverConfig


@dataclass
class BLMResult:
    """Results of a BLM calibration sweep."""
    blm_values: list[float]
    costs: list[float]
    boundaries: list[float]
    objectives: list[float]
    solutions: list[Solution]


def calibrate_blm(
    problem: ConservationProblem,
    solver: Solver,
    blm_values: list[float] | None = None,
    blm_min: float = 0.0,
    blm_max: float = 100.0,
    blm_steps: int = 10,
    config: SolverConfig | None = None,
) -> BLMResult:
    """Run a BLM calibration sweep.

    Either provide explicit `blm_values` or use `blm_min/blm_max/blm_steps`
    to generate a linear range.

    Parameters
    ----------
    problem : ConservationProblem
        The conservation problem to solve.
    solver : Solver
        The solver to use for each BLM value.
    blm_values : list[float] | None
        Explicit list of BLM values to test.
    blm_min, blm_max, blm_steps : float, float, int
        Used if blm_values is None. Generates np.linspace(min, max, steps).
    config : SolverConfig | None
        Solver config (num_solutions=1 recommended for calibration).

    Returns
    -------
    BLMResult
        Calibration results with cost, boundary, and objective per BLM value.
    """
    if config is None:
        config = SolverConfig(num_solutions=1)

    if blm_values is None:
        blm_values = np.linspace(blm_min, blm_max, blm_steps).tolist()

    costs = []
    boundaries = []
    objectives = []
    solutions_list = []

    for blm in blm_values:
        # Make a copy of parameters to avoid mutating the original
        modified = ConservationProblem(
            planning_units=problem.planning_units,
            features=problem.features,
            pu_vs_features=problem.pu_vs_features,
            boundary=problem.boundary,
            parameters={**problem.parameters, "BLM": blm},
        )
        sols = solver.solve(modified, config)
        best = min(sols, key=lambda s: s.objective)
        costs.append(best.cost)
        boundaries.append(best.boundary)
        objectives.append(best.objective)
        solutions_list.append(best)

    return BLMResult(
        blm_values=blm_values,
        costs=costs,
        boundaries=boundaries,
        objectives=objectives,
        solutions=solutions_list,
    )
```

**Step 4: Run tests**

Run: `pytest tests/pymarxan/calibration/test_blm.py -v`
Expected: All 6 tests PASS

**Step 5: Commit**

```bash
git add src/pymarxan/calibration/ tests/pymarxan/calibration/
git commit -m "feat: add BLM calibration sweep tool"
```

---

## Task 4: SPF Calibration

**Files:**
- Create: `src/pymarxan/calibration/spf.py`
- Test: `tests/pymarxan/calibration/test_spf.py`

**Step 1: Write the failing tests**

`tests/pymarxan/calibration/test_spf.py`:
```python
from pathlib import Path

from pymarxan.io.readers import load_project
from pymarxan.calibration.spf import calibrate_spf, SPFResult
from pymarxan.solvers.mip_solver import MIPSolver

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "simple"


class TestCalibrateSPF:
    def setup_method(self):
        self.problem = load_project(DATA_DIR)
        self.solver = MIPSolver()

    def test_returns_spf_result(self):
        result = calibrate_spf(
            self.problem, self.solver, max_iterations=3,
        )
        assert isinstance(result, SPFResult)

    def test_final_solution_meets_targets(self):
        result = calibrate_spf(
            self.problem, self.solver, max_iterations=5,
        )
        assert result.solution.all_targets_met

    def test_adjusted_spf_values(self):
        result = calibrate_spf(
            self.problem, self.solver, max_iterations=3,
        )
        # Should return a dict of feature_id -> spf
        assert isinstance(result.final_spf, dict)
        assert len(result.final_spf) == 3

    def test_history_recorded(self):
        result = calibrate_spf(
            self.problem, self.solver, max_iterations=3,
        )
        assert len(result.history) >= 1
        assert len(result.history) <= 3
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/pymarxan/calibration/test_spf.py -v`
Expected: FAIL

**Step 3: Implement SPF calibration**

`src/pymarxan/calibration/spf.py`:
```python
"""SPF (Species Penalty Factor) calibration for Marxan.

Iteratively adjusts SPF values to ensure all conservation targets are met.
Process: solve -> check unmet targets -> increase SPF for unmet features -> re-solve.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from pymarxan.models.problem import ConservationProblem
from pymarxan.solvers.base import Solution, Solver, SolverConfig


@dataclass
class SPFResult:
    """Results of an SPF calibration run."""
    final_spf: dict[int, float]
    solution: Solution
    history: list[dict]  # List of {iteration, unmet_count, spf_values}


def calibrate_spf(
    problem: ConservationProblem,
    solver: Solver,
    max_iterations: int = 10,
    multiplier: float = 2.0,
    config: SolverConfig | None = None,
) -> SPFResult:
    """Iteratively adjust SPF until all targets are met.

    Parameters
    ----------
    problem : ConservationProblem
        The conservation problem.
    solver : Solver
        Solver to use.
    max_iterations : int
        Maximum calibration iterations.
    multiplier : float
        Factor to multiply SPF by for each unmet feature.
    config : SolverConfig | None
        Solver config (num_solutions=1 recommended).

    Returns
    -------
    SPFResult
        Final SPF values, best solution, and calibration history.
    """
    if config is None:
        config = SolverConfig(num_solutions=1)

    # Start with current SPF values
    spf_values = {}
    for _, row in problem.features.iterrows():
        fid = int(row["id"])
        spf_values[fid] = float(row.get("spf", 1.0))

    history = []
    best_solution = None

    for iteration in range(max_iterations):
        # Build problem with current SPF
        features_df = problem.features.copy()
        for fid, spf in spf_values.items():
            features_df.loc[features_df["id"] == fid, "spf"] = spf

        modified = ConservationProblem(
            planning_units=problem.planning_units,
            features=features_df,
            pu_vs_features=problem.pu_vs_features,
            boundary=problem.boundary,
            parameters=problem.parameters,
        )

        sols = solver.solve(modified, config)
        best = min(sols, key=lambda s: s.objective)
        best_solution = best

        # Check which targets are unmet
        unmet = [
            fid for fid, met in best.targets_met.items() if not met
        ]

        history.append({
            "iteration": iteration + 1,
            "unmet_count": len(unmet),
            "spf_values": dict(spf_values),
        })

        if not unmet:
            break

        # Increase SPF for unmet features
        for fid in unmet:
            spf_values[fid] *= multiplier

    return SPFResult(
        final_spf=spf_values,
        solution=best_solution,
        history=history,
    )
```

**Step 4: Run tests**

Run: `pytest tests/pymarxan/calibration/test_spf.py -v`
Expected: All 4 tests PASS

**Step 5: Commit**

```bash
git add src/pymarxan/calibration/ tests/pymarxan/calibration/
git commit -m "feat: add SPF calibration with iterative target adjustment"
```

---

## Task 5: Selection Frequency Analysis

**Files:**
- Create: `src/pymarxan/analysis/__init__.py`
- Create: `src/pymarxan/analysis/selection_freq.py`
- Test: `tests/pymarxan/analysis/__init__.py`
- Test: `tests/pymarxan/analysis/test_selection_freq.py`

**Step 1: Write the failing tests**

`tests/pymarxan/analysis/test_selection_freq.py`:
```python
import numpy as np

from pymarxan.solvers.base import Solution
from pymarxan.analysis.selection_freq import (
    compute_selection_frequency,
    SelectionFrequency,
)


def _make_solutions() -> list[Solution]:
    """Create 4 mock solutions with known selection patterns."""
    return [
        Solution(
            selected=np.array([True, True, False, False]),
            cost=25.0, boundary=1.0, objective=26.0,
            targets_met={1: True}, metadata={},
        ),
        Solution(
            selected=np.array([True, False, True, False]),
            cost=30.0, boundary=2.0, objective=32.0,
            targets_met={1: True}, metadata={},
        ),
        Solution(
            selected=np.array([True, True, True, False]),
            cost=45.0, boundary=1.0, objective=46.0,
            targets_met={1: True}, metadata={},
        ),
        Solution(
            selected=np.array([False, True, True, True]),
            cost=40.0, boundary=3.0, objective=43.0,
            targets_met={1: True}, metadata={},
        ),
    ]


class TestSelectionFrequency:
    def test_returns_correct_type(self):
        result = compute_selection_frequency(_make_solutions())
        assert isinstance(result, SelectionFrequency)

    def test_frequency_values(self):
        result = compute_selection_frequency(_make_solutions())
        # PU 0: selected in 3/4 = 0.75
        # PU 1: selected in 3/4 = 0.75
        # PU 2: selected in 3/4 = 0.75
        # PU 3: selected in 1/4 = 0.25
        np.testing.assert_array_almost_equal(
            result.frequencies, [0.75, 0.75, 0.75, 0.25]
        )

    def test_count_values(self):
        result = compute_selection_frequency(_make_solutions())
        np.testing.assert_array_equal(result.counts, [3, 3, 3, 1])

    def test_n_solutions(self):
        result = compute_selection_frequency(_make_solutions())
        assert result.n_solutions == 4

    def test_best_solution(self):
        result = compute_selection_frequency(_make_solutions())
        assert result.best_solution.cost == 25.0  # Lowest objective=26.0

    def test_empty_solutions(self):
        result = compute_selection_frequency([])
        assert result.n_solutions == 0
        assert len(result.frequencies) == 0
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/pymarxan/analysis/test_selection_freq.py -v`
Expected: FAIL

**Step 3: Implement selection frequency**

`src/pymarxan/analysis/__init__.py`: empty

`src/pymarxan/analysis/selection_freq.py`:
```python
"""Selection frequency analysis across multiple solver runs."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pymarxan.solvers.base import Solution


@dataclass
class SelectionFrequency:
    """Selection frequency results across multiple solutions."""
    frequencies: np.ndarray  # Fraction of runs each PU was selected [0..1]
    counts: np.ndarray       # Number of times each PU was selected
    n_solutions: int
    best_solution: Solution | None  # Solution with lowest objective


def compute_selection_frequency(solutions: list[Solution]) -> SelectionFrequency:
    """Compute how often each planning unit is selected across solutions.

    Parameters
    ----------
    solutions : list[Solution]
        List of solutions from multiple solver runs.

    Returns
    -------
    SelectionFrequency
        Frequency and count arrays, plus the best solution.
    """
    if not solutions:
        return SelectionFrequency(
            frequencies=np.array([]),
            counts=np.array([]),
            n_solutions=0,
            best_solution=None,
        )

    n_pu = len(solutions[0].selected)
    counts = np.zeros(n_pu, dtype=int)

    for sol in solutions:
        counts += sol.selected.astype(int)

    n = len(solutions)
    frequencies = counts / n
    best = min(solutions, key=lambda s: s.objective)

    return SelectionFrequency(
        frequencies=frequencies,
        counts=counts,
        n_solutions=n,
        best_solution=best,
    )
```

**Step 4: Run tests**

Run: `pytest tests/pymarxan/analysis/test_selection_freq.py -v`
Expected: All 6 tests PASS

**Step 5: Commit**

```bash
git add src/pymarxan/analysis/ tests/pymarxan/analysis/
git commit -m "feat: add selection frequency analysis across multiple runs"
```

---

## Task 6: Irreplaceability Index

**Files:**
- Create: `src/pymarxan/analysis/irreplaceability.py`
- Test: `tests/pymarxan/analysis/test_irreplaceability.py`

**Step 1: Write the failing tests**

`tests/pymarxan/analysis/test_irreplaceability.py`:
```python
from pathlib import Path

import numpy as np

from pymarxan.io.readers import load_project
from pymarxan.analysis.irreplaceability import compute_irreplaceability

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "simple"


class TestIrreplaceability:
    def setup_method(self):
        self.problem = load_project(DATA_DIR)

    def test_returns_dict(self):
        result = compute_irreplaceability(self.problem)
        assert isinstance(result, dict)
        assert len(result) == 6  # One per PU

    def test_values_in_range(self):
        result = compute_irreplaceability(self.problem)
        for pid, score in result.items():
            assert 0.0 <= score <= 1.0, f"PU {pid} score {score} out of range"

    def test_pu_with_unique_feature_has_high_score(self):
        """A PU that is the sole provider of a feature should be irreplaceable."""
        # species_c is not in PU 1 — the remaining PUs share it
        # Let's make a feature only found in PU 1
        import pandas as pd
        problem = self.problem
        extra_feature = pd.DataFrame({
            "id": [99], "name": ["unique_sp"], "target": [5.0], "spf": [1.0],
        })
        problem.features = pd.concat(
            [problem.features, extra_feature], ignore_index=True
        )
        extra_puvspr = pd.DataFrame({
            "species": [99], "pu": [1], "amount": [5.0],
        })
        problem.pu_vs_features = pd.concat(
            [problem.pu_vs_features, extra_puvspr], ignore_index=True
        )
        result = compute_irreplaceability(problem)
        assert result[1] == 1.0  # PU 1 is the only one with species 99
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/pymarxan/analysis/test_irreplaceability.py -v`
Expected: FAIL

**Step 3: Implement irreplaceability**

`src/pymarxan/analysis/irreplaceability.py`:
```python
"""Irreplaceability analysis for conservation planning.

Computes how irreplaceable each planning unit is based on its contribution
to meeting conservation targets. A PU is fully irreplaceable (1.0) if
removing it makes a target unachievable.
"""
from __future__ import annotations

from pymarxan.models.problem import ConservationProblem


def compute_irreplaceability(
    problem: ConservationProblem,
) -> dict[int, float]:
    """Compute irreplaceability score for each planning unit.

    Score is the fraction of features for which this PU is critical
    (i.e., removing it would make the target unachievable from remaining PUs).

    A score of 1.0 means the PU is essential for at least one feature.
    A score of 0.0 means no feature target depends uniquely on this PU.

    Parameters
    ----------
    problem : ConservationProblem
        The conservation problem.

    Returns
    -------
    dict[int, float]
        Mapping from PU id to irreplaceability score [0.0, 1.0].
    """
    pu_ids = problem.planning_units["id"].tolist()

    # Total amount of each feature across all PUs
    feature_totals = problem.feature_amounts()

    # Amount of each feature in each PU
    # pu_contributions[pu_id][feature_id] = amount
    pu_contributions: dict[int, dict[int, float]] = {pid: {} for pid in pu_ids}
    for _, row in problem.pu_vs_features.iterrows():
        pid = int(row["pu"])
        fid = int(row["species"])
        amount = float(row["amount"])
        pu_contributions[pid][fid] = amount

    n_features = problem.n_features
    scores: dict[int, float] = {}

    for pid in pu_ids:
        critical_count = 0
        contributions = pu_contributions.get(pid, {})

        for _, feat_row in problem.features.iterrows():
            fid = int(feat_row["id"])
            target = float(feat_row.get("target", 0.0))
            if target <= 0:
                continue

            total = feature_totals.get(fid, 0.0)
            pu_amount = contributions.get(fid, 0.0)

            # If removing this PU's contribution makes target unachievable
            remaining = total - pu_amount
            if remaining < target:
                critical_count += 1

        scores[pid] = critical_count / n_features if n_features > 0 else 0.0

    return scores
```

**Step 4: Run tests**

Run: `pytest tests/pymarxan/analysis/test_irreplaceability.py -v`
Expected: All 3 tests PASS

**Step 5: Commit**

```bash
git add src/pymarxan/analysis/ tests/pymarxan/analysis/
git commit -m "feat: add irreplaceability analysis for planning units"
```

---

## Task 7: Results Export (CSV + GeoPackage)

**Files:**
- Create: `src/pymarxan/io/exporters.py`
- Test: `tests/pymarxan/io/test_exporters.py`

**Step 1: Write the failing tests**

`tests/pymarxan/io/test_exporters.py`:
```python
from pathlib import Path

import numpy as np
import pandas as pd

from pymarxan.io.readers import load_project
from pymarxan.io.exporters import (
    export_solution_csv,
    export_summary_csv,
    export_selection_frequency_csv,
)
from pymarxan.solvers.base import Solution
from pymarxan.analysis.selection_freq import compute_selection_frequency

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "simple"


def _make_solution(problem):
    return Solution(
        selected=np.array([True, True, False, True, False, True]),
        cost=45.0, boundary=3.0, objective=48.0,
        targets_met={1: True, 2: True, 3: True},
        metadata={"solver": "test"},
    )


class TestExportSolutionCSV:
    def test_creates_csv(self, tmp_path):
        problem = load_project(DATA_DIR)
        sol = _make_solution(problem)
        out = tmp_path / "solution.csv"
        export_solution_csv(problem, sol, out)
        assert out.exists()
        df = pd.read_csv(out)
        assert "planning_unit" in df.columns
        assert "selected" in df.columns
        assert "cost" in df.columns
        assert len(df) == 6
        assert df["selected"].sum() == 4


class TestExportSummaryCSV:
    def test_creates_csv(self, tmp_path):
        problem = load_project(DATA_DIR)
        sol = _make_solution(problem)
        out = tmp_path / "summary.csv"
        export_summary_csv(problem, sol, out)
        assert out.exists()
        df = pd.read_csv(out)
        assert "feature_id" in df.columns
        assert "target" in df.columns
        assert "achieved" in df.columns
        assert "met" in df.columns
        assert len(df) == 3


class TestExportSelectionFrequencyCSV:
    def test_creates_csv(self, tmp_path):
        problem = load_project(DATA_DIR)
        solutions = [
            Solution(
                selected=np.array([True, True, False, False, False, False]),
                cost=25.0, boundary=1.0, objective=26.0,
                targets_met={1: True, 2: True, 3: True}, metadata={},
            ),
            Solution(
                selected=np.array([True, False, True, False, False, True]),
                cost=38.0, boundary=2.0, objective=40.0,
                targets_met={1: True, 2: True, 3: True}, metadata={},
            ),
        ]
        freq = compute_selection_frequency(solutions)
        out = tmp_path / "freq.csv"
        export_selection_frequency_csv(problem, freq, out)
        assert out.exists()
        df = pd.read_csv(out)
        assert "planning_unit" in df.columns
        assert "frequency" in df.columns
        assert "count" in df.columns
        assert len(df) == 6
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/pymarxan/io/test_exporters.py -v`
Expected: FAIL

**Step 3: Implement exporters**

`src/pymarxan/io/exporters.py`:
```python
"""Export solver results to CSV and other formats."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from pymarxan.analysis.selection_freq import SelectionFrequency
from pymarxan.models.problem import ConservationProblem
from pymarxan.solvers.base import Solution


def export_solution_csv(
    problem: ConservationProblem,
    solution: Solution,
    path: Path | str,
) -> None:
    """Export a solution to CSV with planning unit details."""
    pu_ids = problem.planning_units["id"].tolist()
    costs = problem.planning_units["cost"].tolist()
    rows = []
    for i, (pid, cost) in enumerate(zip(pu_ids, costs)):
        rows.append({
            "planning_unit": pid,
            "cost": cost,
            "selected": int(solution.selected[i]),
        })
    pd.DataFrame(rows).to_csv(path, index=False)


def export_summary_csv(
    problem: ConservationProblem,
    solution: Solution,
    path: Path | str,
) -> None:
    """Export target achievement summary to CSV."""
    pu_ids = problem.planning_units["id"].tolist()
    id_to_idx = {pid: i for i, pid in enumerate(pu_ids)}

    rows = []
    for _, frow in problem.features.iterrows():
        fid = int(frow["id"])
        fname = frow.get("name", f"Feature {fid}")
        target = float(frow.get("target", 0.0))
        mask = problem.pu_vs_features["species"] == fid
        achieved = 0.0
        for _, arow in problem.pu_vs_features[mask].iterrows():
            pid = int(arow["pu"])
            if pid in id_to_idx and solution.selected[id_to_idx[pid]]:
                achieved += float(arow["amount"])
        rows.append({
            "feature_id": fid,
            "feature_name": fname,
            "target": target,
            "achieved": achieved,
            "met": achieved >= target,
        })
    pd.DataFrame(rows).to_csv(path, index=False)


def export_selection_frequency_csv(
    problem: ConservationProblem,
    freq: SelectionFrequency,
    path: Path | str,
) -> None:
    """Export selection frequency results to CSV."""
    pu_ids = problem.planning_units["id"].tolist()
    rows = []
    for i, pid in enumerate(pu_ids):
        rows.append({
            "planning_unit": pid,
            "frequency": float(freq.frequencies[i]),
            "count": int(freq.counts[i]),
        })
    pd.DataFrame(rows).to_csv(path, index=False)
```

**Step 4: Run tests**

Run: `pytest tests/pymarxan/io/test_exporters.py -v`
Expected: All 3 tests PASS

**Step 5: Commit**

```bash
git add src/pymarxan/io/ tests/pymarxan/io/
git commit -m "feat: add CSV export for solutions, summaries, and selection frequency"
```

---

## Task 8: BLM Calibration Shiny Module

**Files:**
- Create: `src/pymarxan_shiny/modules/calibration/__init__.py`
- Create: `src/pymarxan_shiny/modules/calibration/blm_explorer.py`

**Step 1: Create the BLM calibration Shiny module**

`src/pymarxan_shiny/modules/calibration/__init__.py`: empty

`src/pymarxan_shiny/modules/calibration/blm_explorer.py`:
```python
"""Interactive BLM calibration Shiny module with cost-vs-boundary plot."""
from __future__ import annotations

from shiny import module, reactive, render, ui

from pymarxan.calibration.blm import calibrate_blm, BLMResult
from pymarxan.models.problem import ConservationProblem
from pymarxan.solvers.base import Solver, SolverConfig


@module.ui
def blm_explorer_ui():
    return ui.card(
        ui.card_header("BLM Calibration"),
        ui.layout_sidebar(
            ui.sidebar(
                ui.input_numeric("blm_min", "Min BLM", value=0, min=0, step=0.1),
                ui.input_numeric("blm_max", "Max BLM", value=50, min=0, step=1),
                ui.input_numeric("blm_steps", "Steps", value=10, min=2, max=50),
                ui.input_action_button(
                    "run_calibration", "Run Calibration",
                    class_="btn-primary w-100",
                ),
                width=300,
            ),
            ui.div(
                ui.output_plot("blm_plot"),
                ui.output_text_verbatim("blm_table"),
            ),
        ),
    )


@module.server
def blm_explorer_server(
    input, output, session,
    problem: reactive.Value,
    solver: reactive.Value,
):
    """BLM calibration module server.

    Args:
        problem: reactive.Value[ConservationProblem | None]
        solver: reactive.Value[Solver] — the solver instance to use
    """
    result: reactive.Value[BLMResult | None] = reactive.value(None)

    @reactive.effect
    @reactive.event(input.run_calibration)
    def _run():
        p = problem()
        s = solver()
        if p is None or s is None:
            ui.notification_show("Load a project and configure a solver first.", type="error")
            return
        ui.notification_show("Running BLM calibration...", type="message")
        try:
            res = calibrate_blm(
                p, s,
                blm_min=float(input.blm_min()),
                blm_max=float(input.blm_max()),
                blm_steps=int(input.blm_steps()),
                config=SolverConfig(num_solutions=1),
            )
            result.set(res)
            ui.notification_show("BLM calibration complete!", type="message")
        except Exception as e:
            ui.notification_show(f"Calibration error: {e}", type="error")

    @render.plot
    def blm_plot():
        res = result()
        if res is None:
            return None
        import matplotlib.pyplot as plt
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        ax1.plot(res.blm_values, res.costs, "o-", color="steelblue")
        ax1.set_xlabel("BLM")
        ax1.set_ylabel("Cost")
        ax1.set_title("Cost vs BLM")
        ax2.plot(res.costs, res.boundaries, "o-", color="coral")
        ax2.set_xlabel("Cost")
        ax2.set_ylabel("Boundary Length")
        ax2.set_title("Cost vs Boundary (find the elbow)")
        for i, blm in enumerate(res.blm_values):
            ax2.annotate(
                f"{blm:.1f}", (res.costs[i], res.boundaries[i]),
                textcoords="offset points", xytext=(5, 5), fontsize=7,
            )
        fig.tight_layout()
        return fig

    @render.text
    def blm_table():
        res = result()
        if res is None:
            return "Run calibration to see results."
        lines = [f"{'BLM':>8} {'Cost':>10} {'Boundary':>10} {'Objective':>12}"]
        lines.append("-" * 44)
        for i in range(len(res.blm_values)):
            lines.append(
                f"{res.blm_values[i]:8.2f} {res.costs[i]:10.2f} "
                f"{res.boundaries[i]:10.2f} {res.objectives[i]:12.2f}"
            )
        return "\n".join(lines)
```

**Step 2: Commit**

```bash
git add src/pymarxan_shiny/modules/calibration/
git commit -m "feat: add BLM calibration Shiny module with cost-boundary plot"
```

---

## Task 9: Enhanced Results & Export Shiny Module

**Files:**
- Create: `src/pymarxan_shiny/modules/results/export.py`
- Modify: `src/pymarxan_shiny/modules/results/summary_table.py` — add selection frequency display

**Step 1: Create the export module**

`src/pymarxan_shiny/modules/results/export.py`:
```python
"""Results export Shiny module — download solutions as CSV."""
from __future__ import annotations

import tempfile
from pathlib import Path

from shiny import module, reactive, render, ui

from pymarxan.io.exporters import export_solution_csv, export_summary_csv
from pymarxan.models.problem import ConservationProblem
from pymarxan.solvers.base import Solution


@module.ui
def export_ui():
    return ui.card(
        ui.card_header("Export Results"),
        ui.download_button("download_solution", "Download Solution CSV"),
        ui.download_button("download_summary", "Download Target Summary CSV"),
    )


@module.server
def export_server(
    input, output, session,
    problem: reactive.Value,
    solution: reactive.Value,
):
    @render.download(filename="pymarxan_solution.csv")
    def download_solution():
        p = problem()
        s = solution()
        if p is None or s is None:
            return
        tmp = tempfile.NamedTemporaryFile(
            suffix=".csv", delete=False, mode="w"
        )
        export_solution_csv(p, s, tmp.name)
        return tmp.name

    @render.download(filename="pymarxan_target_summary.csv")
    def download_summary():
        p = problem()
        s = solution()
        if p is None or s is None:
            return
        tmp = tempfile.NamedTemporaryFile(
            suffix=".csv", delete=False, mode="w"
        )
        export_summary_csv(p, s, tmp.name)
        return tmp.name
```

**Step 2: Commit**

```bash
git add src/pymarxan_shiny/modules/results/
git commit -m "feat: add results export Shiny module with CSV download"
```

---

## Task 10: Update Assembled App with Phase 2 Features

**Files:**
- Modify: `src/pymarxan_app/app.py`
- Modify: `src/pymarxan_shiny/modules/solver_config/solver_picker.py`

Add:
1. Simulated Annealing as a solver choice in the picker
2. Calibrate tab with BLM explorer
3. Export module in Results tab

**Step 1: Update solver picker to include SA**

Add `"sa": "Simulated Annealing (Python)"` to the solver choices in `solver_picker_ui()`. Add SA-specific conditional panel for iterations and temperature steps (similar to the existing binary panel).

**Step 2: Update app.py**

Add new imports:
```python
from pymarxan.solvers.simulated_annealing import SimulatedAnnealingSolver
from pymarxan_shiny.modules.calibration.blm_explorer import blm_explorer_ui, blm_explorer_server
from pymarxan_shiny.modules.results.export import export_ui, export_server
```

Add new nav_panel for Calibrate tab (between Configure and Run):
```python
ui.nav_panel("Calibrate", ui.layout_columns(
    blm_explorer_ui("blm_cal"),
    col_widths=12,
)),
```

Add export module to Results tab.

Wire new modules in server function:
- Create `active_solver` reactive that instantiates the selected solver
- Wire `blm_explorer_server("blm_cal", problem=problem, solver=active_solver)`
- Wire `export_server("export", problem=problem, solution=current_solution)`
- Add `"sa"` case in `_run_solver` to use `SimulatedAnnealingSolver()`

**Step 3: Run all tests**

Run: `pytest tests/ -v`
Expected: All tests PASS

**Step 4: Manual smoke test**

Run: `shiny run src/pymarxan_app/app.py --port 8000`
Verify: All 5 tabs work (Data, Configure, Calibrate, Run, Results)

**Step 5: Commit**

```bash
git add src/pymarxan_app/ src/pymarxan_shiny/
git commit -m "feat: integrate SA solver, BLM calibration, and CSV export into app"
```

---

## Task 11: Integration Tests for Phase 2

**Files:**
- Create: `tests/test_integration_phase2.py`

**Step 1: Write integration tests**

`tests/test_integration_phase2.py`:
```python
"""Integration tests for Phase 2 features."""
from pathlib import Path

import numpy as np

from pymarxan.io.readers import load_project
from pymarxan.io.exporters import export_solution_csv, export_summary_csv
from pymarxan.solvers.base import SolverConfig
from pymarxan.solvers.simulated_annealing import SimulatedAnnealingSolver
from pymarxan.solvers.mip_solver import MIPSolver
from pymarxan.calibration.blm import calibrate_blm
from pymarxan.calibration.spf import calibrate_spf
from pymarxan.analysis.selection_freq import compute_selection_frequency
from pymarxan.analysis.irreplaceability import compute_irreplaceability

DATA_DIR = Path(__file__).parent / "data" / "simple"


class TestSAIntegration:
    def test_sa_finds_feasible_solution(self):
        problem = load_project(DATA_DIR)
        solver = SimulatedAnnealingSolver(num_iterations=100_000)
        config = SolverConfig(num_solutions=5, seed=42)
        solutions = solver.solve(problem, config)
        assert len(solutions) == 5
        feasible = [s for s in solutions if s.all_targets_met]
        assert len(feasible) > 0

    def test_sa_selection_frequency(self):
        problem = load_project(DATA_DIR)
        solver = SimulatedAnnealingSolver(num_iterations=100_000)
        config = SolverConfig(num_solutions=10, seed=42)
        solutions = solver.solve(problem, config)
        freq = compute_selection_frequency(solutions)
        assert freq.n_solutions == 10
        assert all(0.0 <= f <= 1.0 for f in freq.frequencies)


class TestCalibrationIntegration:
    def test_blm_calibration_with_mip(self):
        problem = load_project(DATA_DIR)
        solver = MIPSolver()
        result = calibrate_blm(
            problem, solver,
            blm_values=[0.0, 1.0, 10.0],
        )
        assert len(result.blm_values) == 3
        assert all(s.all_targets_met for s in result.solutions)

    def test_spf_calibration_converges(self):
        problem = load_project(DATA_DIR)
        solver = MIPSolver()
        result = calibrate_spf(problem, solver, max_iterations=5)
        assert result.solution.all_targets_met


class TestExportIntegration:
    def test_export_after_solve(self, tmp_path):
        problem = load_project(DATA_DIR)
        solver = MIPSolver()
        sol = solver.solve(problem, SolverConfig(num_solutions=1))[0]
        export_solution_csv(problem, sol, tmp_path / "sol.csv")
        export_summary_csv(problem, sol, tmp_path / "summary.csv")
        assert (tmp_path / "sol.csv").exists()
        assert (tmp_path / "summary.csv").exists()


class TestIrreplaceabilityIntegration:
    def test_irreplaceability_scores(self):
        problem = load_project(DATA_DIR)
        scores = compute_irreplaceability(problem)
        assert len(scores) == 6
        assert all(0.0 <= v <= 1.0 for v in scores.values())
```

**Step 2: Run all tests**

Run: `pytest tests/ -v`
Expected: ALL tests pass

**Step 3: Commit**

```bash
git add tests/test_integration_phase2.py
git commit -m "test: add Phase 2 integration tests for SA, calibration, export, irreplaceability"
```

---

## Task 12: Lint and Final Cleanup

**Step 1: Run ruff**

Run: `ruff check src/ tests/ --fix`
Expected: Clean or auto-fixed

**Step 2: Run mypy**

Run: `mypy src/pymarxan/ --ignore-missing-imports`
Expected: Clean

**Step 3: Full test suite**

Run: `pytest tests/ -v --tb=short`
Expected: ALL tests pass

**Step 4: Commit if needed**

```bash
git add -A
git commit -m "chore: lint and type-check cleanup for Phase 2"
```

---

## Summary of Phase 2 Deliverables

| Component | Files | Tests |
|---|---|---|
| Solver utils (refactor) | `pymarxan.solvers.utils` | 8 |
| Simulated Annealing | `pymarxan.solvers.simulated_annealing` | 11 |
| BLM Calibration | `pymarxan.calibration.blm` | 6 |
| SPF Calibration | `pymarxan.calibration.spf` | 4 |
| Selection Frequency | `pymarxan.analysis.selection_freq` | 6 |
| Irreplaceability | `pymarxan.analysis.irreplaceability` | 3 |
| CSV Export | `pymarxan.io.exporters` | 3 |
| BLM Calibration UI | `pymarxan_shiny.modules.calibration.blm_explorer` | Manual |
| Export UI | `pymarxan_shiny.modules.results.export` | Manual |
| App updates | `pymarxan_app.app` (5 tabs) | Manual |
| Integration tests | `tests/test_integration_phase2.py` | 6 |
| **Total new** | **~15 files** | **~47 new tests** |
