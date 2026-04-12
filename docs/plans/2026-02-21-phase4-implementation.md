# Phase 4: Advanced Features Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add parameter sweep/sensitivity analysis, batch parallel execution, solver plugin registry, scenario comparison, and cloud deployment (Docker) to exceed current Marxan ecosystem capabilities.

**Architecture:** Generalise the existing BLM calibration sweep pattern into an arbitrary parameter sweep module that uses `concurrent.futures` for parallel execution. Add a solver registry for plugin discovery. Build Shiny modules for sweep configuration and scenario comparison. Package everything with a Dockerfile.

**Tech Stack:** Python 3.11+, NumPy, Pandas, concurrent.futures, Shiny for Python, Docker

---

### Task 1: Parameter Sweep Module

**Files:**
- Create: `src/pymarxan/calibration/sweep.py`
- Test: `tests/pymarxan/calibration/test_sweep.py`

**Context:** The existing `calibrate_blm()` in `src/pymarxan/calibration/blm.py` sweeps a single parameter (BLM) through a range and collects results. Phase 4 generalises this to sweep any combination of parameters (BLM, SPF multiplier, num_solutions, etc.) using a grid or list of parameter dicts.

**Step 1: Write the failing test**

```python
"""Tests for parameter sweep module."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pymarxan.calibration.sweep import SweepConfig, SweepResult, run_sweep
from pymarxan.models.problem import ConservationProblem
from pymarxan.solvers.base import Solution, Solver, SolverConfig


class _StubSolver(Solver):
    """Stub solver that returns a deterministic solution."""

    def solve(
        self, problem: ConservationProblem, config: SolverConfig | None = None
    ) -> list[Solution]:
        n = problem.n_planning_units
        blm = problem.parameters.get("BLM", 1.0)
        return [
            Solution(
                selected=np.ones(n, dtype=bool),
                cost=10.0 + blm,
                boundary=5.0,
                objective=10.0 + blm + blm * 5.0,
                targets_met={1: True},
                metadata={"blm": blm},
            )
        ]

    def name(self) -> str:
        return "stub"

    def supports_zones(self) -> bool:
        return False


@pytest.fixture()
def small_problem() -> ConservationProblem:
    pu = pd.DataFrame({"id": [1, 2], "cost": [10.0, 20.0], "status": [0, 0]})
    feat = pd.DataFrame({"id": [1], "name": ["f1"], "target": [5.0], "spf": [1.0]})
    puvspr = pd.DataFrame({"species": [1, 1], "pu": [1, 2], "amount": [3.0, 4.0]})
    return ConservationProblem(
        planning_units=pu, features=feat, pu_vs_features=puvspr,
    )


def test_sweep_with_explicit_param_dicts(small_problem: ConservationProblem):
    """Sweep over a list of explicit parameter dicts."""
    config = SweepConfig(
        param_dicts=[{"BLM": 0.0}, {"BLM": 1.0}, {"BLM": 10.0}],
    )
    result = run_sweep(small_problem, _StubSolver(), config)
    assert isinstance(result, SweepResult)
    assert len(result.solutions) == 3
    assert len(result.param_dicts) == 3
    assert result.objectives[0] < result.objectives[2]


def test_sweep_with_grid(small_problem: ConservationProblem):
    """Sweep over a grid of parameter combinations."""
    config = SweepConfig(
        param_grid={"BLM": [0.0, 1.0, 5.0]},
    )
    result = run_sweep(small_problem, _StubSolver(), config)
    assert len(result.solutions) == 3


def test_sweep_multi_param_grid(small_problem: ConservationProblem):
    """Grid with two parameters produces cartesian product."""
    config = SweepConfig(
        param_grid={"BLM": [0.0, 1.0], "NUMITNS": [100, 200]},
    )
    result = run_sweep(small_problem, _StubSolver(), config)
    assert len(result.solutions) == 4  # 2 x 2


def test_sweep_result_best(small_problem: ConservationProblem):
    """best property returns the solution with lowest objective."""
    config = SweepConfig(param_dicts=[{"BLM": 0.0}, {"BLM": 10.0}])
    result = run_sweep(small_problem, _StubSolver(), config)
    assert result.best.cost == pytest.approx(10.0)


def test_sweep_to_dataframe(small_problem: ConservationProblem):
    """to_dataframe returns a DataFrame with one row per sweep point."""
    config = SweepConfig(param_dicts=[{"BLM": 0.0}, {"BLM": 5.0}])
    result = run_sweep(small_problem, _StubSolver(), config)
    df = result.to_dataframe()
    assert len(df) == 2
    assert "cost" in df.columns
    assert "objective" in df.columns
    assert "BLM" in df.columns
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/pymarxan/calibration/test_sweep.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pymarxan.calibration.sweep'`

**Step 3: Write minimal implementation**

```python
"""Parameter sweep for conservation planning.

Generalises the BLM calibration pattern to sweep any combination of
problem parameters and collect results.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import pandas as pd

from pymarxan.models.problem import ConservationProblem
from pymarxan.solvers.base import Solution, Solver, SolverConfig


@dataclass
class SweepConfig:
    """Configuration for a parameter sweep.

    Provide either ``param_dicts`` (explicit list of parameter dicts)
    or ``param_grid`` (dict of param_name -> list of values, expanded
    into the cartesian product). If both are provided, ``param_dicts``
    takes precedence.
    """

    param_dicts: list[dict] | None = None
    param_grid: dict[str, list] | None = None
    solver_config: SolverConfig | None = None

    def expand(self) -> list[dict]:
        """Return the list of parameter dicts to sweep over."""
        if self.param_dicts is not None:
            return list(self.param_dicts)
        if self.param_grid is not None:
            keys = sorted(self.param_grid.keys())
            values = [self.param_grid[k] for k in keys]
            return [dict(zip(keys, combo)) for combo in itertools.product(*values)]
        return [{}]


@dataclass
class SweepResult:
    """Results of a parameter sweep."""

    param_dicts: list[dict]
    solutions: list[Solution]
    costs: list[float] = field(default_factory=list)
    boundaries: list[float] = field(default_factory=list)
    objectives: list[float] = field(default_factory=list)

    @property
    def best(self) -> Solution:
        """Return the solution with the lowest objective."""
        idx = min(range(len(self.objectives)), key=lambda i: self.objectives[i])
        return self.solutions[idx]

    def to_dataframe(self) -> pd.DataFrame:
        """Convert results to a DataFrame with one row per sweep point."""
        rows = []
        for i, params in enumerate(self.param_dicts):
            row = {**params}
            row["cost"] = self.costs[i]
            row["boundary"] = self.boundaries[i]
            row["objective"] = self.objectives[i]
            row["n_selected"] = self.solutions[i].n_selected
            row["all_targets_met"] = self.solutions[i].all_targets_met
            rows.append(row)
        return pd.DataFrame(rows)


def run_sweep(
    problem: ConservationProblem,
    solver: Solver,
    config: SweepConfig,
) -> SweepResult:
    """Run a parameter sweep over the given problem."""
    solver_config = config.solver_config or SolverConfig(num_solutions=1)
    param_dicts = config.expand()

    solutions: list[Solution] = []
    costs: list[float] = []
    boundaries: list[float] = []
    objectives: list[float] = []

    for params in param_dicts:
        modified = ConservationProblem(
            planning_units=problem.planning_units,
            features=problem.features,
            pu_vs_features=problem.pu_vs_features,
            boundary=problem.boundary,
            parameters={**problem.parameters, **params},
        )
        sols = solver.solve(modified, solver_config)
        best = min(sols, key=lambda s: s.objective)
        solutions.append(best)
        costs.append(best.cost)
        boundaries.append(best.boundary)
        objectives.append(best.objective)

    return SweepResult(
        param_dicts=param_dicts,
        solutions=solutions,
        costs=costs,
        boundaries=boundaries,
        objectives=objectives,
    )
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/pymarxan/calibration/test_sweep.py -v`
Expected: 5 PASSED

**Step 5: Commit**

```bash
git add src/pymarxan/calibration/sweep.py tests/pymarxan/calibration/test_sweep.py
git commit -m "feat: add parameter sweep module generalising BLM calibration"
```

---

### Task 2: Batch Parallel Execution

**Files:**
- Create: `src/pymarxan/calibration/parallel.py`
- Test: `tests/pymarxan/calibration/test_parallel.py`

**Context:** The sweep module runs sequentially. This task adds a `run_sweep_parallel()` function using `concurrent.futures.ProcessPoolExecutor` for parallel execution. Falls back to sequential if `max_workers=1`.

**Step 1: Write the failing test**

```python
"""Tests for parallel sweep execution."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pymarxan.calibration.parallel import run_sweep_parallel
from pymarxan.calibration.sweep import SweepConfig, SweepResult
from pymarxan.models.problem import ConservationProblem
from pymarxan.solvers.base import Solution, Solver, SolverConfig
from pymarxan.solvers.simulated_annealing import SimulatedAnnealingSolver


class _StubSolver(Solver):
    def solve(
        self, problem: ConservationProblem, config: SolverConfig | None = None
    ) -> list[Solution]:
        n = problem.n_planning_units
        blm = problem.parameters.get("BLM", 1.0)
        return [
            Solution(
                selected=np.ones(n, dtype=bool),
                cost=10.0 + blm,
                boundary=5.0,
                objective=10.0 + blm + blm * 5.0,
                targets_met={1: True},
            )
        ]

    def name(self) -> str:
        return "stub"

    def supports_zones(self) -> bool:
        return False


@pytest.fixture()
def small_problem() -> ConservationProblem:
    pu = pd.DataFrame({"id": [1, 2], "cost": [10.0, 20.0], "status": [0, 0]})
    feat = pd.DataFrame({"id": [1], "name": ["f1"], "target": [5.0], "spf": [1.0]})
    puvspr = pd.DataFrame({"species": [1, 1], "pu": [1, 2], "amount": [3.0, 4.0]})
    return ConservationProblem(
        planning_units=pu, features=feat, pu_vs_features=puvspr,
    )


def test_parallel_sweep_produces_same_results(
    small_problem: ConservationProblem,
):
    """Parallel sweep returns same number of results as sequential."""
    config = SweepConfig(param_dicts=[{"BLM": 0.0}, {"BLM": 1.0}, {"BLM": 5.0}])
    result = run_sweep_parallel(
        small_problem, _StubSolver(), config, max_workers=2,
    )
    assert isinstance(result, SweepResult)
    assert len(result.solutions) == 3


def test_parallel_single_worker(small_problem: ConservationProblem):
    """max_workers=1 runs sequentially without error."""
    config = SweepConfig(param_dicts=[{"BLM": 0.0}, {"BLM": 1.0}])
    result = run_sweep_parallel(
        small_problem, _StubSolver(), config, max_workers=1,
    )
    assert len(result.solutions) == 2


def test_parallel_preserves_order(small_problem: ConservationProblem):
    """Results are returned in the same order as param_dicts."""
    config = SweepConfig(param_dicts=[{"BLM": 0.0}, {"BLM": 5.0}, {"BLM": 10.0}])
    result = run_sweep_parallel(
        small_problem, _StubSolver(), config, max_workers=2,
    )
    assert result.objectives[0] < result.objectives[1] < result.objectives[2]
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/pymarxan/calibration/test_parallel.py -v`
Expected: FAIL with `ModuleNotFoundError`

**Step 3: Write minimal implementation**

```python
"""Parallel parameter sweep execution.

Uses concurrent.futures to run parameter sweep points in parallel.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed

from pymarxan.calibration.sweep import SweepConfig, SweepResult, run_sweep
from pymarxan.models.problem import ConservationProblem
from pymarxan.solvers.base import Solution, Solver, SolverConfig


def _solve_single(
    problem_data: dict,
    solver: Solver,
    params: dict,
    solver_config: SolverConfig,
    index: int,
) -> tuple[int, Solution]:
    """Solve a single sweep point. Returns (index, best_solution)."""
    import pandas as pd

    from pymarxan.models.problem import ConservationProblem

    modified = ConservationProblem(
        planning_units=pd.DataFrame(problem_data["planning_units"]),
        features=pd.DataFrame(problem_data["features"]),
        pu_vs_features=pd.DataFrame(problem_data["pu_vs_features"]),
        boundary=(
            pd.DataFrame(problem_data["boundary"])
            if problem_data["boundary"] is not None
            else None
        ),
        parameters={**problem_data["parameters"], **params},
    )
    sols = solver.solve(modified, solver_config)
    best = min(sols, key=lambda s: s.objective)
    return (index, best)


def run_sweep_parallel(
    problem: ConservationProblem,
    solver: Solver,
    config: SweepConfig,
    max_workers: int = 4,
) -> SweepResult:
    """Run a parameter sweep with parallel execution.

    Falls back to sequential ``run_sweep`` when ``max_workers=1``.
    """
    if max_workers <= 1:
        return run_sweep(problem, solver, config)

    solver_config = config.solver_config or SolverConfig(num_solutions=1)
    param_dicts = config.expand()

    # Serialise problem data for pickling across processes
    problem_data = {
        "planning_units": problem.planning_units.to_dict(orient="list"),
        "features": problem.features.to_dict(orient="list"),
        "pu_vs_features": problem.pu_vs_features.to_dict(orient="list"),
        "boundary": (
            problem.boundary.to_dict(orient="list")
            if problem.boundary is not None
            else None
        ),
        "parameters": dict(problem.parameters),
    }

    # Submit all jobs
    indexed_results: dict[int, Solution] = {}
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _solve_single, problem_data, solver, params, solver_config, i
            ): i
            for i, params in enumerate(param_dicts)
        }
        for future in as_completed(futures):
            idx, sol = future.result()
            indexed_results[idx] = sol

    # Reassemble in order
    solutions = [indexed_results[i] for i in range(len(param_dicts))]
    costs = [s.cost for s in solutions]
    boundaries = [s.boundary for s in solutions]
    objectives = [s.objective for s in solutions]

    return SweepResult(
        param_dicts=param_dicts,
        solutions=solutions,
        costs=costs,
        boundaries=boundaries,
        objectives=objectives,
    )
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/pymarxan/calibration/test_parallel.py -v`
Expected: 3 PASSED

**Step 5: Commit**

```bash
git add src/pymarxan/calibration/parallel.py tests/pymarxan/calibration/test_parallel.py
git commit -m "feat: add parallel parameter sweep execution with ProcessPoolExecutor"
```

---

### Task 3: Scenario Comparison Module

**Files:**
- Create: `src/pymarxan/analysis/scenarios.py`
- Test: `tests/pymarxan/analysis/test_scenarios.py`

**Context:** Users often run multiple solver configurations and need to compare results. This module stores named scenarios (a label + Solution + parameters) and provides comparison metrics.

**Step 1: Write the failing test**

```python
"""Tests for scenario comparison module."""
from __future__ import annotations

import numpy as np
import pytest

from pymarxan.analysis.scenarios import Scenario, ScenarioSet
from pymarxan.solvers.base import Solution


def _make_solution(cost: float, n_selected: int, n: int = 10) -> Solution:
    selected = np.zeros(n, dtype=bool)
    selected[:n_selected] = True
    return Solution(
        selected=selected,
        cost=cost,
        boundary=10.0,
        objective=cost + 10.0,
        targets_met={1: True, 2: cost < 50},
    )


def test_scenario_creation():
    sol = _make_solution(30.0, 5)
    s = Scenario(name="low-blm", solution=sol, parameters={"BLM": 0.1})
    assert s.name == "low-blm"
    assert s.solution.cost == 30.0


def test_scenario_set_add_and_list():
    ss = ScenarioSet()
    ss.add("a", _make_solution(30.0, 5), {"BLM": 0.1})
    ss.add("b", _make_solution(50.0, 3), {"BLM": 1.0})
    assert len(ss) == 2
    assert ss.names == ["a", "b"]


def test_scenario_set_compare_dataframe():
    ss = ScenarioSet()
    ss.add("a", _make_solution(30.0, 5), {"BLM": 0.1})
    ss.add("b", _make_solution(50.0, 3), {"BLM": 1.0})
    df = ss.compare()
    assert len(df) == 2
    assert "name" in df.columns
    assert "cost" in df.columns
    assert "n_selected" in df.columns
    assert "all_targets_met" in df.columns


def test_scenario_set_overlap():
    """Overlap is fraction of shared selected PUs."""
    ss = ScenarioSet()
    ss.add("a", _make_solution(30.0, 5), {})
    ss.add("b", _make_solution(30.0, 5), {})  # same selection
    matrix = ss.overlap_matrix()
    assert matrix.shape == (2, 2)
    assert matrix[0, 1] == pytest.approx(1.0)  # identical selections


def test_scenario_set_get():
    ss = ScenarioSet()
    ss.add("x", _make_solution(30.0, 5), {"BLM": 0.1})
    s = ss.get("x")
    assert s.name == "x"


def test_scenario_set_remove():
    ss = ScenarioSet()
    ss.add("a", _make_solution(30.0, 5), {})
    ss.remove("a")
    assert len(ss) == 0
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/pymarxan/analysis/test_scenarios.py -v`
Expected: FAIL with `ModuleNotFoundError`

**Step 3: Write minimal implementation**

```python
"""Scenario comparison for conservation planning.

Store named scenarios (label + Solution + parameters) and compare them.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from pymarxan.solvers.base import Solution


@dataclass
class Scenario:
    """A named solver result with its configuration."""

    name: str
    solution: Solution
    parameters: dict = field(default_factory=dict)


class ScenarioSet:
    """Collection of named scenarios for comparison."""

    def __init__(self) -> None:
        self._scenarios: list[Scenario] = []

    def __len__(self) -> int:
        return len(self._scenarios)

    @property
    def names(self) -> list[str]:
        return [s.name for s in self._scenarios]

    def add(
        self, name: str, solution: Solution, parameters: dict | None = None
    ) -> None:
        self._scenarios.append(
            Scenario(name=name, solution=solution, parameters=parameters or {})
        )

    def get(self, name: str) -> Scenario:
        for s in self._scenarios:
            if s.name == name:
                return s
        raise KeyError(f"Scenario '{name}' not found")

    def remove(self, name: str) -> None:
        self._scenarios = [s for s in self._scenarios if s.name != name]

    def compare(self) -> pd.DataFrame:
        """Return a DataFrame comparing all scenarios."""
        rows = []
        for s in self._scenarios:
            sol = s.solution
            rows.append({
                "name": s.name,
                "cost": sol.cost,
                "boundary": sol.boundary,
                "objective": sol.objective,
                "n_selected": sol.n_selected,
                "all_targets_met": sol.all_targets_met,
                **s.parameters,
            })
        return pd.DataFrame(rows)

    def overlap_matrix(self) -> np.ndarray:
        """Compute pairwise selection overlap (Jaccard index)."""
        n = len(self._scenarios)
        matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                si = self._scenarios[i].solution.selected
                sj = self._scenarios[j].solution.selected
                intersection = np.sum(si & sj)
                union = np.sum(si | sj)
                matrix[i, j] = intersection / union if union > 0 else 0.0
        return matrix
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/pymarxan/analysis/test_scenarios.py -v`
Expected: 6 PASSED

**Step 5: Commit**

```bash
git add src/pymarxan/analysis/scenarios.py tests/pymarxan/analysis/test_scenarios.py
git commit -m "feat: add scenario comparison module with overlap matrix"
```

---

### Task 4: Solver Plugin Registry

**Files:**
- Create: `src/pymarxan/solvers/registry.py`
- Test: `tests/pymarxan/solvers/test_registry.py`

**Context:** Currently solvers are hard-coded in the app. A registry allows registering and discovering solvers by name, including custom user-provided solvers. Built-in solvers are auto-registered.

**Step 1: Write the failing test**

```python
"""Tests for solver plugin registry."""
from __future__ import annotations

import numpy as np
import pytest

from pymarxan.models.problem import ConservationProblem
from pymarxan.solvers.base import Solution, Solver, SolverConfig
from pymarxan.solvers.registry import SolverRegistry


class _CustomSolver(Solver):
    def solve(
        self, problem: ConservationProblem, config: SolverConfig | None = None
    ) -> list[Solution]:
        n = problem.n_planning_units
        return [
            Solution(
                selected=np.ones(n, dtype=bool),
                cost=0.0, boundary=0.0, objective=0.0,
                targets_met={},
            )
        ]

    def name(self) -> str:
        return "custom"

    def supports_zones(self) -> bool:
        return False


def test_register_and_get():
    reg = SolverRegistry()
    reg.register("custom", _CustomSolver)
    solver = reg.create("custom")
    assert solver.name() == "custom"


def test_list_registered():
    reg = SolverRegistry()
    reg.register("custom", _CustomSolver)
    assert "custom" in reg.list_solvers()


def test_get_unknown_raises():
    reg = SolverRegistry()
    with pytest.raises(KeyError):
        reg.create("nonexistent")


def test_register_duplicate_raises():
    reg = SolverRegistry()
    reg.register("custom", _CustomSolver)
    with pytest.raises(ValueError):
        reg.register("custom", _CustomSolver)


def test_register_override():
    reg = SolverRegistry()
    reg.register("custom", _CustomSolver)
    reg.register("custom", _CustomSolver, override=True)
    assert "custom" in reg.list_solvers()


def test_default_registry_has_builtins():
    """The default registry includes built-in solvers."""
    from pymarxan.solvers.registry import get_default_registry

    reg = get_default_registry()
    names = reg.list_solvers()
    assert "mip" in names
    assert "sa" in names
    assert "zone_sa" in names


def test_available_solvers():
    """available_solvers filters by solver.available()."""
    reg = SolverRegistry()
    reg.register("custom", _CustomSolver)
    available = reg.available_solvers()
    assert "custom" in available
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/pymarxan/solvers/test_registry.py -v`
Expected: FAIL with `ModuleNotFoundError`

**Step 3: Write minimal implementation**

```python
"""Solver plugin registry for conservation planning.

Allows registration and discovery of solver implementations by name.
"""
from __future__ import annotations

from pymarxan.solvers.base import Solver


class SolverRegistry:
    """Registry for solver plugins."""

    def __init__(self) -> None:
        self._solvers: dict[str, type[Solver]] = {}

    def register(
        self,
        name: str,
        solver_class: type[Solver],
        override: bool = False,
    ) -> None:
        """Register a solver class under a given name."""
        if name in self._solvers and not override:
            raise ValueError(f"Solver '{name}' is already registered")
        self._solvers[name] = solver_class

    def create(self, name: str) -> Solver:
        """Create a solver instance by name."""
        if name not in self._solvers:
            raise KeyError(f"Unknown solver: '{name}'")
        return self._solvers[name]()

    def list_solvers(self) -> list[str]:
        """Return names of all registered solvers."""
        return sorted(self._solvers.keys())

    def available_solvers(self) -> list[str]:
        """Return names of solvers that are currently available."""
        result = []
        for name, cls in sorted(self._solvers.items()):
            try:
                instance = cls()
                if instance.available():
                    result.append(name)
            except Exception:
                pass
        return result


def get_default_registry() -> SolverRegistry:
    """Return a registry pre-loaded with built-in solvers."""
    from pymarxan.solvers.marxan_binary import MarxanBinarySolver
    from pymarxan.solvers.mip_solver import MIPSolver
    from pymarxan.solvers.simulated_annealing import SimulatedAnnealingSolver
    from pymarxan.zones.solver import ZoneSASolver

    reg = SolverRegistry()
    reg.register("mip", MIPSolver)
    reg.register("sa", SimulatedAnnealingSolver)
    reg.register("binary", MarxanBinarySolver)
    reg.register("zone_sa", ZoneSASolver)
    return reg
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/pymarxan/solvers/test_registry.py -v`
Expected: 7 PASSED

**Step 5: Commit**

```bash
git add src/pymarxan/solvers/registry.py tests/pymarxan/solvers/test_registry.py
git commit -m "feat: add solver plugin registry with auto-registered builtins"
```

---

### Task 5: Sweep Shiny Module

**Files:**
- Create: `src/pymarxan_shiny/modules/calibration/sweep_explorer.py`
- Create: `src/pymarxan_shiny/modules/calibration/__init__.py` (update if exists)
- Test: `tests/pymarxan_shiny/test_sweep_module.py`

**Context:** A Shiny module that lets users configure a parameter sweep (select parameters, ranges, number of steps), run it, and view results in a table. Reuses the sweep module from Task 1.

**Step 1: Write the failing test**

```python
"""Tests for sweep explorer Shiny module."""
from __future__ import annotations

import pytest

from pymarxan_shiny.modules.calibration.sweep_explorer import (
    sweep_explorer_server,
    sweep_explorer_ui,
)


def test_sweep_explorer_ui_returns_tag():
    """UI function returns a valid Shiny UI element."""
    ui_elem = sweep_explorer_ui("test_sweep")
    assert ui_elem is not None


def test_sweep_explorer_server_callable():
    """Server function is callable."""
    assert callable(sweep_explorer_server)
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/pymarxan_shiny/test_sweep_module.py -v`
Expected: FAIL with `ModuleNotFoundError`

**Step 3: Write minimal implementation**

```python
"""Sweep explorer Shiny module.

Lets users configure and run a parameter sweep, then view results.
"""
from __future__ import annotations

from shiny import Inputs, Outputs, Session, module, reactive, render, ui

from pymarxan.calibration.sweep import SweepConfig, SweepResult, run_sweep
from pymarxan.models.problem import ConservationProblem
from pymarxan.solvers.base import Solver


@module.ui
def sweep_explorer_ui():
    return ui.card(
        ui.card_header("Parameter Sweep"),
        ui.layout_sidebar(
            ui.sidebar(
                ui.input_select(
                    "sweep_param",
                    "Parameter to sweep",
                    choices=["BLM", "NUMITNS", "NUMTEMP"],
                    selected="BLM",
                ),
                ui.input_numeric("sweep_min", "Min value", value=0.0),
                ui.input_numeric("sweep_max", "Max value", value=100.0),
                ui.input_numeric("sweep_steps", "Number of steps", value=10),
                ui.input_action_button(
                    "run_sweep", "Run Sweep", class_="btn-primary w-100"
                ),
                width=280,
            ),
            ui.output_data_frame("sweep_results_table"),
        ),
    )


@module.server
def sweep_explorer_server(
    input: Inputs,
    output: Outputs,
    session: Session,
    problem: reactive.Value,
    solver: reactive.Calc,
):
    sweep_result: reactive.Value[SweepResult | None] = reactive.value(None)

    @reactive.effect
    @reactive.event(input.run_sweep)
    def _run():
        p = problem()
        if p is None:
            ui.notification_show("Load a project first!", type="error")
            return
        param_name = input.sweep_param()
        min_val = input.sweep_min()
        max_val = input.sweep_max()
        steps = int(input.sweep_steps())

        import numpy as np

        values = np.linspace(min_val, max_val, steps).tolist()
        config = SweepConfig(
            param_dicts=[{param_name: v} for v in values],
        )
        result = run_sweep(p, solver(), config)
        sweep_result.set(result)
        ui.notification_show(
            f"Sweep complete: {len(result.solutions)} points", type="message"
        )

    @render.data_frame
    def sweep_results_table():
        r = sweep_result()
        if r is None:
            return None
        return r.to_dataframe()
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/pymarxan_shiny/test_sweep_module.py -v`
Expected: 2 PASSED

**Step 5: Commit**

```bash
git add src/pymarxan_shiny/modules/calibration/sweep_explorer.py tests/pymarxan_shiny/test_sweep_module.py
git commit -m "feat: add sweep explorer Shiny module for parameter sweeps"
```

---

### Task 6: Scenario Comparison Shiny Module

**Files:**
- Create: `src/pymarxan_shiny/modules/results/scenario_compare.py`
- Test: `tests/pymarxan_shiny/test_scenario_module.py`

**Context:** A Shiny module that lets users name and save the current solution as a scenario, list saved scenarios, and view a comparison table. Uses the scenarios module from Task 3.

**Step 1: Write the failing test**

```python
"""Tests for scenario comparison Shiny module."""
from __future__ import annotations

from pymarxan_shiny.modules.results.scenario_compare import (
    scenario_compare_server,
    scenario_compare_ui,
)


def test_scenario_compare_ui_returns_tag():
    ui_elem = scenario_compare_ui("test_scenario")
    assert ui_elem is not None


def test_scenario_compare_server_callable():
    assert callable(scenario_compare_server)
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/pymarxan_shiny/test_scenario_module.py -v`
Expected: FAIL

**Step 3: Write minimal implementation**

```python
"""Scenario comparison Shiny module.

Lets users save solutions as named scenarios and compare them.
"""
from __future__ import annotations

from shiny import Inputs, Outputs, Session, module, reactive, render, ui

from pymarxan.analysis.scenarios import ScenarioSet
from pymarxan.solvers.base import Solution


@module.ui
def scenario_compare_ui():
    return ui.card(
        ui.card_header("Scenario Comparison"),
        ui.layout_sidebar(
            ui.sidebar(
                ui.input_text("scenario_name", "Scenario name", value=""),
                ui.input_action_button(
                    "save_scenario", "Save Current Solution",
                    class_="btn-primary w-100",
                ),
                ui.hr(),
                ui.output_text("scenario_count"),
                width=280,
            ),
            ui.output_data_frame("comparison_table"),
        ),
    )


@module.server
def scenario_compare_server(
    input: Inputs,
    output: Outputs,
    session: Session,
    solution: reactive.Value,
    solver_config: reactive.Value,
):
    scenarios: reactive.Value[ScenarioSet] = reactive.value(ScenarioSet())

    @reactive.effect
    @reactive.event(input.save_scenario)
    def _save():
        sol = solution()
        if sol is None:
            ui.notification_show("Run a solver first!", type="error")
            return
        name = input.scenario_name() or f"scenario-{len(scenarios()) + 1}"
        ss = scenarios()
        ss.add(name, sol, dict(solver_config()))
        scenarios.set(ss)
        ui.notification_show(f"Saved scenario: {name}", type="message")

    @render.text
    def scenario_count():
        return f"{len(scenarios())} scenarios saved"

    @render.data_frame
    def comparison_table():
        ss = scenarios()
        if len(ss) == 0:
            return None
        return ss.compare()
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/pymarxan_shiny/test_scenario_module.py -v`
Expected: 2 PASSED

**Step 5: Commit**

```bash
git add src/pymarxan_shiny/modules/results/scenario_compare.py tests/pymarxan_shiny/test_scenario_module.py
git commit -m "feat: add scenario comparison Shiny module"
```

---

### Task 7: Update App with Sweep and Scenarios Tabs

**Files:**
- Modify: `src/pymarxan_app/app.py`
- Test: manual verification (existing integration tests cover app import)

**Context:** Add the sweep explorer and scenario comparison modules to the app. Add a "Sweep" nav panel between "Calibrate" and "Zones", and add the scenario comparison to the "Results" tab.

**Step 1: Write the failing test**

```python
# In tests/test_integration_phase4.py
"""Integration tests for Phase 4 features."""
from __future__ import annotations

import numpy as np
import pandas as pd

from pymarxan.calibration.sweep import SweepConfig, run_sweep
from pymarxan.calibration.parallel import run_sweep_parallel
from pymarxan.analysis.scenarios import ScenarioSet
from pymarxan.models.problem import ConservationProblem
from pymarxan.solvers.base import Solution, SolverConfig
from pymarxan.solvers.mip_solver import MIPSolver
from pymarxan.solvers.registry import get_default_registry


def _small_problem() -> ConservationProblem:
    pu = pd.DataFrame({"id": [1, 2, 3], "cost": [10.0, 20.0, 15.0], "status": [0, 0, 0]})
    feat = pd.DataFrame({"id": [1], "name": ["f1"], "target": [5.0], "spf": [1.0]})
    puvspr = pd.DataFrame({
        "species": [1, 1, 1], "pu": [1, 2, 3], "amount": [3.0, 4.0, 2.0],
    })
    bnd = pd.DataFrame({
        "id1": [1, 2], "id2": [2, 3], "boundary": [1.0, 1.0],
    })
    return ConservationProblem(
        planning_units=pu, features=feat,
        pu_vs_features=puvspr, boundary=bnd,
        parameters={"BLM": 1.0},
    )


def test_sweep_with_mip():
    """End-to-end: sweep BLM with MIP solver."""
    p = _small_problem()
    config = SweepConfig(param_grid={"BLM": [0.0, 1.0, 5.0]})
    result = run_sweep(p, MIPSolver(), config)
    assert len(result.solutions) == 3
    assert all(s.all_targets_met for s in result.solutions)


def test_parallel_sweep_with_mip():
    """End-to-end: parallel sweep with MIP solver."""
    p = _small_problem()
    config = SweepConfig(param_grid={"BLM": [0.0, 1.0]})
    result = run_sweep_parallel(p, MIPSolver(), config, max_workers=2)
    assert len(result.solutions) == 2


def test_scenario_workflow():
    """End-to-end: save scenarios and compare."""
    p = _small_problem()
    solver = MIPSolver()
    ss = ScenarioSet()

    for blm in [0.0, 1.0, 5.0]:
        p.parameters["BLM"] = blm
        sols = solver.solve(p, SolverConfig(num_solutions=1))
        ss.add(f"blm-{blm}", sols[0], {"BLM": blm})

    df = ss.compare()
    assert len(df) == 3
    assert "BLM" in df.columns


def test_registry_create_and_solve():
    """Registry creates working solver instances."""
    reg = get_default_registry()
    solver = reg.create("mip")
    p = _small_problem()
    sols = solver.solve(p, SolverConfig(num_solutions=1))
    assert len(sols) == 1


def test_app_imports():
    """App module imports successfully with new modules."""
    import pymarxan_app.app  # noqa: F401
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_integration_phase4.py -v`
Expected: FAIL (app import may fail with new module references)

**Step 3: Update app.py**

Add the sweep explorer and scenario compare modules to the app. Add imports for the new modules, a "Sweep" tab, and update the Results tab to include scenario comparison.

The key changes to `src/pymarxan_app/app.py`:
1. Import `sweep_explorer_ui, sweep_explorer_server`
2. Import `scenario_compare_ui, scenario_compare_server`
3. Add `ui.nav_panel("Sweep", ...)` between Calibrate and Zones
4. Add `scenario_compare_ui("scenarios")` to the Results tab
5. Wire up server modules

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_integration_phase4.py -v`
Expected: 5 PASSED

**Step 5: Commit**

```bash
git add src/pymarxan_app/app.py tests/test_integration_phase4.py
git commit -m "feat: add sweep and scenario tabs to app, phase 4 integration tests"
```

---

### Task 8: Dockerfile for Deployment

**Files:**
- Create: `Dockerfile`
- Create: `docker-compose.yml`
- Test: `docker build --no-cache -t pymarxan-app .` (manual/CI)

**Context:** A simple Dockerfile that packages the Shiny app for deployment. Uses a Python 3.11 slim image, installs dependencies, copies source, and runs `shiny run`.

**Step 1: Write the Dockerfile**

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for numpy/scipy
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc g++ && \
    rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY pyproject.toml .
COPY src/ src/

RUN pip install --no-cache-dir -e ".[all]" && \
    apt-get purge -y gcc g++ && \
    apt-get autoremove -y

EXPOSE 8000

CMD ["shiny", "run", "src/pymarxan_app/app.py", "--host", "0.0.0.0", "--port", "8000"]
```

**Step 2: Write docker-compose.yml**

```yaml
version: "3.8"

services:
  pymarxan:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./data:/app/data
    environment:
      - SHINY_LOG_LEVEL=info
```

**Step 3: Commit**

```bash
git add Dockerfile docker-compose.yml
git commit -m "feat: add Dockerfile and docker-compose for cloud deployment"
```

---

### Task 9: Lint and Cleanup

**Files:**
- All new files from Tasks 1-8

**Step 1: Run ruff**

Run: `ruff check src/ tests/ --fix`
Fix any remaining issues.

**Step 2: Run mypy**

Run: `mypy src/pymarxan/ --ignore-missing-imports`
Fix any type errors.

**Step 3: Run full test suite**

Run: `python -m pytest tests/ -v --tb=short`
Expected: All tests pass (127 existing + ~25 new ≈ 152+ tests)

**Step 4: Commit any fixes**

```bash
git add -u
git commit -m "chore: fix lint and type issues from phase 4"
```

---

## Summary

| Task | Module | Tests | Key concept |
|------|--------|-------|-------------|
| 1 | `calibration/sweep.py` | 5 | Generic parameter sweep with grid expansion |
| 2 | `calibration/parallel.py` | 3 | ProcessPoolExecutor for parallel sweeps |
| 3 | `analysis/scenarios.py` | 6 | Named scenarios with overlap matrix |
| 4 | `solvers/registry.py` | 7 | Plugin registry with auto-registered builtins |
| 5 | `calibration/sweep_explorer.py` (Shiny) | 2 | Sweep configuration and results UI |
| 6 | `results/scenario_compare.py` (Shiny) | 2 | Save/compare scenarios UI |
| 7 | App update + integration tests | 5 | Wire everything together |
| 8 | Dockerfile + docker-compose | — | Cloud deployment |
| 9 | Lint + cleanup | — | Code quality |

**Total new tests:** ~30
**Total estimated tests after Phase 4:** ~157+

**Parallelisable pairs:**
- Tasks 1 + 4 (sweep module + registry — no dependencies)
- Tasks 3 + 2 (scenarios + parallel — no dependencies)
- Tasks 5 + 6 (Shiny modules — independent)
