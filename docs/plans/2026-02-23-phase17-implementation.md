# Phase 17: Per-Scenario Feature Overrides + Project Cloning

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add per-scenario feature target/SPF overrides to the Scenario system, add `ConservationProblem.clone()` and `ScenarioSet.clone_scenario()` methods, and wire both into the Shiny UI.

**Architecture:** Extends existing `analysis/scenarios.py` and `models/problem.py` with new methods. No new modules — these are enhancements to the existing codebase. `apply_feature_overrides()` creates a deep copy with modified features DataFrame. `clone()` uses `copy.deepcopy`. Shiny enhancements go into existing `scenario_compare.py`.

**Tech Stack:** copy (stdlib), pandas, numpy — no new dependencies

---

### Task 1: Add `apply_feature_overrides` to `problem.py`

**Files:**
- Modify: `src/pymarxan/models/problem.py`
- Create: `tests/pymarxan/models/test_feature_overrides.py`
- Modify: `src/pymarxan/models/__init__.py`

**Step 1: Write the failing tests**

```python
# tests/pymarxan/models/test_feature_overrides.py
"""Tests for feature override functionality."""
from __future__ import annotations

import pandas as pd
import pytest

from pymarxan.models.problem import ConservationProblem, apply_feature_overrides


def _make_problem():
    return ConservationProblem(
        planning_units=pd.DataFrame({
            "id": [1, 2, 3], "cost": [1.0, 2.0, 3.0], "status": [0, 0, 0],
        }),
        features=pd.DataFrame({
            "id": [1, 2, 3],
            "name": ["forest", "wetland", "coral"],
            "target": [100.0, 200.0, 300.0],
            "spf": [1.0, 1.0, 1.0],
        }),
        pu_vs_features=pd.DataFrame({
            "species": [1, 2, 3],
            "pu": [1, 2, 3],
            "amount": [150.0, 250.0, 350.0],
        }),
    )


class TestApplyFeatureOverrides:
    def test_override_single_target(self):
        p = _make_problem()
        result = apply_feature_overrides(p, {1: {"target": 50.0}})
        assert result.features.loc[result.features["id"] == 1, "target"].iloc[0] == 50.0
        # Others unchanged
        assert result.features.loc[result.features["id"] == 2, "target"].iloc[0] == 200.0

    def test_override_spf(self):
        p = _make_problem()
        result = apply_feature_overrides(p, {2: {"spf": 5.0}})
        assert result.features.loc[result.features["id"] == 2, "spf"].iloc[0] == 5.0

    def test_override_multiple_features(self):
        p = _make_problem()
        result = apply_feature_overrides(p, {
            1: {"target": 50.0},
            3: {"target": 150.0, "spf": 2.0},
        })
        assert result.features.loc[result.features["id"] == 1, "target"].iloc[0] == 50.0
        assert result.features.loc[result.features["id"] == 3, "target"].iloc[0] == 150.0
        assert result.features.loc[result.features["id"] == 3, "spf"].iloc[0] == 2.0

    def test_does_not_mutate_original(self):
        p = _make_problem()
        apply_feature_overrides(p, {1: {"target": 50.0}})
        assert p.features.loc[p.features["id"] == 1, "target"].iloc[0] == 100.0

    def test_invalid_feature_id_raises(self):
        p = _make_problem()
        with pytest.raises(KeyError, match="999"):
            apply_feature_overrides(p, {999: {"target": 50.0}})

    def test_invalid_field_raises(self):
        p = _make_problem()
        with pytest.raises(ValueError, match="invalid_field"):
            apply_feature_overrides(p, {1: {"invalid_field": 50.0}})

    def test_empty_overrides_returns_copy(self):
        p = _make_problem()
        result = apply_feature_overrides(p, {})
        assert result is not p
        assert result.features.equals(p.features)
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/pymarxan/models/test_feature_overrides.py -v`
Expected: FAIL — `ImportError: cannot import name 'apply_feature_overrides'`

**Step 3: Implement**

Add to `src/pymarxan/models/problem.py` after the `has_geometry` function:

```python
_OVERRIDABLE_FIELDS = {"target", "spf", "prop"}


def apply_feature_overrides(
    problem: ConservationProblem,
    overrides: dict[int, dict[str, float]],
) -> ConservationProblem:
    """Return a copy of problem with feature targets/SPF overridden.

    Parameters
    ----------
    problem : ConservationProblem
        The original problem (not mutated).
    overrides : dict
        Maps feature_id -> {field_name: new_value}.
        Valid fields: ``"target"``, ``"spf"``, ``"prop"``.

    Returns
    -------
    ConservationProblem
        Deep copy with overridden feature values.

    Raises
    ------
    KeyError
        If a feature ID is not found.
    ValueError
        If an invalid field name is used.
    """
    import copy
    result = copy.deepcopy(problem)

    feature_ids = set(result.features["id"])
    for fid, fields in overrides.items():
        if fid not in feature_ids:
            raise KeyError(f"Feature ID {fid} not found in problem")
        for field_name, value in fields.items():
            if field_name not in _OVERRIDABLE_FIELDS:
                raise ValueError(
                    f"Invalid field '{field_name}'. "
                    f"Must be one of: {sorted(_OVERRIDABLE_FIELDS)}"
                )
            mask = result.features["id"] == fid
            result.features.loc[mask, field_name] = value

    return result
```

Update `src/pymarxan/models/__init__.py`:

```python
from pymarxan.models.problem import ConservationProblem, apply_feature_overrides, has_geometry

__all__ = ["ConservationProblem", "apply_feature_overrides", "has_geometry"]
```

**Step 4: Run tests**

Run: `pytest tests/pymarxan/models/test_feature_overrides.py -v`
Expected: 7 PASS

**Step 5: Commit**

```bash
git add src/pymarxan/models/problem.py src/pymarxan/models/__init__.py tests/pymarxan/models/test_feature_overrides.py
git commit -m "feat(models): add apply_feature_overrides for per-scenario target/SPF changes"
```

---

### Task 2: Add `clone()` to ConservationProblem

**Files:**
- Modify: `src/pymarxan/models/problem.py`
- Create: `tests/pymarxan/models/test_clone.py`

**Step 1: Write the failing tests**

```python
# tests/pymarxan/models/test_clone.py
"""Tests for ConservationProblem.clone()."""
from __future__ import annotations

import pandas as pd

from pymarxan.models.problem import ConservationProblem


def _make_problem():
    return ConservationProblem(
        planning_units=pd.DataFrame({
            "id": [1, 2], "cost": [1.0, 2.0], "status": [0, 0],
        }),
        features=pd.DataFrame({
            "id": [1], "name": ["f1"], "target": [10.0], "spf": [1.0],
        }),
        pu_vs_features=pd.DataFrame({
            "species": [1, 1], "pu": [1, 2], "amount": [5.0, 8.0],
        }),
        boundary=pd.DataFrame({
            "id1": [1], "id2": [2], "boundary": [1.0],
        }),
        parameters={"BLM": "1.0"},
    )


class TestClone:
    def test_clone_returns_new_instance(self):
        p = _make_problem()
        c = p.clone()
        assert c is not p

    def test_clone_has_equal_data(self):
        p = _make_problem()
        c = p.clone()
        assert c.planning_units.equals(p.planning_units)
        assert c.features.equals(p.features)
        assert c.pu_vs_features.equals(p.pu_vs_features)
        assert c.boundary.equals(p.boundary)
        assert c.parameters == p.parameters

    def test_clone_is_independent(self):
        p = _make_problem()
        c = p.clone()
        c.planning_units.loc[0, "cost"] = 999.0
        c.parameters["BLM"] = "99.0"
        # Original unchanged
        assert p.planning_units.loc[0, "cost"] == 1.0
        assert p.parameters["BLM"] == "1.0"

    def test_clone_without_boundary(self):
        p = _make_problem()
        p.boundary = None
        c = p.clone()
        assert c.boundary is None
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/pymarxan/models/test_clone.py -v`
Expected: FAIL — `AttributeError: 'ConservationProblem' object has no attribute 'clone'`

**Step 3: Implement**

Add to `ConservationProblem` class in `src/pymarxan/models/problem.py`:

```python
    def clone(self) -> ConservationProblem:
        """Deep copy all DataFrames, parameters, and geometry.

        Returns an independent copy that can be modified without
        affecting the original.
        """
        import copy
        return copy.deepcopy(self)
```

**Step 4: Run tests**

Run: `pytest tests/pymarxan/models/test_clone.py -v`
Expected: 4 PASS

**Step 5: Commit**

```bash
git add src/pymarxan/models/problem.py tests/pymarxan/models/test_clone.py
git commit -m "feat(models): add ConservationProblem.clone() for deep copy"
```

---

### Task 3: Enhance Scenario with `feature_overrides` field

**Files:**
- Modify: `src/pymarxan/analysis/scenarios.py`
- Create: `tests/pymarxan/analysis/test_scenario_overrides.py`

**Step 1: Write the failing tests**

```python
# tests/pymarxan/analysis/test_scenario_overrides.py
"""Tests for Scenario feature overrides."""
from __future__ import annotations

import numpy as np
import pandas as pd

from pymarxan.analysis.scenarios import Scenario, ScenarioSet
from pymarxan.solvers.base import Solution


def _make_solution(cost=10.0):
    return Solution(
        selected=np.array([True, False, True]),
        cost=cost,
        boundary=1.0,
        objective=cost + 1.0,
        targets_met={1: True},
    )


class TestScenarioOverrides:
    def test_scenario_default_no_overrides(self):
        s = Scenario(name="base", solution=_make_solution())
        assert s.feature_overrides is None

    def test_scenario_with_overrides(self):
        overrides = {1: {"target": 50.0}}
        s = Scenario(
            name="modified",
            solution=_make_solution(),
            feature_overrides=overrides,
        )
        assert s.feature_overrides == overrides

    def test_backward_compatible_creation(self):
        # Existing code that creates Scenarios without overrides still works
        s = Scenario(name="old", solution=_make_solution(), parameters={"BLM": "1.0"})
        assert s.feature_overrides is None


class TestScenarioSetClone:
    def test_clone_scenario(self):
        ss = ScenarioSet()
        ss.add("original", _make_solution(cost=10.0), parameters={"BLM": "1.0"})
        cloned = ss.clone_scenario("original", "clone1")
        assert cloned.name == "clone1"
        assert cloned.solution.cost == 10.0
        assert cloned.parameters == {"BLM": "1.0"}

    def test_clone_with_parameter_overrides(self):
        ss = ScenarioSet()
        ss.add("original", _make_solution(), parameters={"BLM": "1.0"})
        cloned = ss.clone_scenario(
            "original", "modified",
            parameter_overrides={"BLM": "5.0"},
        )
        assert cloned.parameters["BLM"] == "5.0"

    def test_clone_with_feature_overrides(self):
        ss = ScenarioSet()
        ss.add("original", _make_solution())
        overrides = {1: {"target": 50.0}}
        cloned = ss.clone_scenario(
            "original", "override",
            feature_overrides=overrides,
        )
        assert cloned.feature_overrides == overrides

    def test_clone_is_independent(self):
        ss = ScenarioSet()
        ss.add("original", _make_solution(), parameters={"BLM": "1.0"})
        cloned = ss.clone_scenario("original", "clone")
        cloned.parameters["BLM"] = "99.0"
        original = ss.get("original")
        assert original.parameters["BLM"] == "1.0"

    def test_clone_nonexistent_raises(self):
        ss = ScenarioSet()
        with pytest.raises(KeyError):
            ss.clone_scenario("nonexistent", "clone")

    def test_clone_added_to_set(self):
        ss = ScenarioSet()
        ss.add("original", _make_solution())
        ss.clone_scenario("original", "clone")
        assert "clone" in ss.names
        assert len(ss) == 2
```

Add missing import at top:

```python
import pytest
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/pymarxan/analysis/test_scenario_overrides.py -v`
Expected: FAIL on `clone_scenario` and `feature_overrides` tests

**Step 3: Implement**

Modify `src/pymarxan/analysis/scenarios.py`:

```python
"""Scenario comparison for conservation planning.

Store named scenarios (label + Solution + parameters) and compare them.
"""
from __future__ import annotations

import copy
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
    feature_overrides: dict[int, dict[str, float]] | None = None


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
        self,
        name: str,
        solution: Solution,
        parameters: dict | None = None,
        feature_overrides: dict[int, dict[str, float]] | None = None,
    ) -> None:
        self._scenarios.append(
            Scenario(
                name=name,
                solution=solution,
                parameters=parameters or {},
                feature_overrides=feature_overrides,
            )
        )

    def get(self, name: str) -> Scenario:
        for s in self._scenarios:
            if s.name == name:
                return s
        raise KeyError(f"Scenario '{name}' not found")

    def remove(self, name: str) -> None:
        self._scenarios = [s for s in self._scenarios if s.name != name]

    def clone_scenario(
        self,
        source_name: str,
        new_name: str,
        parameter_overrides: dict | None = None,
        feature_overrides: dict[int, dict[str, float]] | None = None,
    ) -> Scenario:
        """Clone an existing scenario with optional modifications.

        Parameters
        ----------
        source_name : str
            Name of the scenario to clone.
        new_name : str
            Name for the new scenario.
        parameter_overrides : dict or None
            Parameters to override in the clone.
        feature_overrides : dict or None
            Feature overrides to set on the clone.

        Returns
        -------
        Scenario
            The newly created and added scenario.
        """
        source = self.get(source_name)
        cloned_solution = copy.deepcopy(source.solution)
        cloned_params = copy.deepcopy(source.parameters)

        if parameter_overrides:
            cloned_params.update(parameter_overrides)

        cloned_feat_overrides = copy.deepcopy(source.feature_overrides)
        if feature_overrides:
            if cloned_feat_overrides is None:
                cloned_feat_overrides = {}
            cloned_feat_overrides.update(feature_overrides)

        scenario = Scenario(
            name=new_name,
            solution=cloned_solution,
            parameters=cloned_params,
            feature_overrides=cloned_feat_overrides,
        )
        self._scenarios.append(scenario)
        return scenario

    def compare(self) -> pd.DataFrame:
        """Return a DataFrame comparing all scenarios."""
        rows = []
        for s in self._scenarios:
            sol = s.solution
            row = {
                "name": s.name,
                "cost": sol.cost,
                "boundary": sol.boundary,
                "objective": sol.objective,
                "n_selected": sol.n_selected,
                "all_targets_met": sol.all_targets_met,
                "has_overrides": s.feature_overrides is not None,
                **s.parameters,
            }
            rows.append(row)
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

**Step 4: Run tests**

Run: `pytest tests/pymarxan/analysis/test_scenario_overrides.py -v`
Expected: 9 PASS

**Step 5: Run existing scenario tests for regression**

Run: `pytest tests/pymarxan/analysis/ -v`
Expected: All pass

**Step 6: Commit**

```bash
git add src/pymarxan/analysis/scenarios.py tests/pymarxan/analysis/test_scenario_overrides.py
git commit -m "feat(analysis): add feature overrides + clone_scenario to ScenarioSet"
```

---

### Task 4: Add `run_with_overrides` to ScenarioSet

**Files:**
- Modify: `src/pymarxan/analysis/scenarios.py`
- Modify: `tests/pymarxan/analysis/test_scenario_overrides.py`

**Step 1: Write the failing tests**

Add to `tests/pymarxan/analysis/test_scenario_overrides.py`:

```python
from pymarxan.models.problem import ConservationProblem
from pymarxan.solvers.mip_solver import MIPSolver
from pymarxan.solvers.base import SolverConfig


def _make_problem():
    return ConservationProblem(
        planning_units=pd.DataFrame({
            "id": [1, 2, 3, 4, 5, 6],
            "cost": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "status": [0, 0, 0, 0, 0, 0],
        }),
        features=pd.DataFrame({
            "id": [1, 2],
            "name": ["f1", "f2"],
            "target": [5.0, 5.0],
            "spf": [1.0, 1.0],
        }),
        pu_vs_features=pd.DataFrame({
            "species": [1, 1, 1, 2, 2, 2],
            "pu": [1, 2, 3, 4, 5, 6],
            "amount": [5.0, 3.0, 2.0, 5.0, 3.0, 2.0],
        }),
    )


class TestRunWithOverrides:
    def test_run_creates_scenario(self):
        ss = ScenarioSet()
        p = _make_problem()
        solver = MIPSolver()
        scenario = ss.run_with_overrides(
            name="lowered_target",
            problem=p,
            solver=solver,
            overrides={1: {"target": 2.0}},
            config=SolverConfig(num_solutions=1),
        )
        assert scenario.name == "lowered_target"
        assert scenario.feature_overrides == {1: {"target": 2.0}}
        assert scenario.solution is not None
        assert "lowered_target" in ss.names

    def test_run_with_parameter_overrides(self):
        ss = ScenarioSet()
        p = _make_problem()
        solver = MIPSolver()
        scenario = ss.run_with_overrides(
            name="with_blm",
            problem=p,
            solver=solver,
            overrides={},
            parameter_overrides={"BLM": "5.0"},
            config=SolverConfig(num_solutions=1),
        )
        assert scenario.parameters.get("BLM") == "5.0"
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/pymarxan/analysis/test_scenario_overrides.py::TestRunWithOverrides -v`
Expected: FAIL — `AttributeError: 'ScenarioSet' object has no attribute 'run_with_overrides'`

**Step 3: Implement**

Add to `ScenarioSet` class in `src/pymarxan/analysis/scenarios.py`:

```python
    def run_with_overrides(
        self,
        name: str,
        problem: "ConservationProblem",
        solver: "Solver",
        overrides: dict[int, dict[str, float]],
        parameter_overrides: dict | None = None,
        config: "SolverConfig | None" = None,
    ) -> Scenario:
        """Create scenario by solving with feature overrides applied.

        Parameters
        ----------
        name : str
            Scenario name.
        problem : ConservationProblem
            Base problem (not mutated).
        solver : Solver
            Solver to use.
        overrides : dict
            Feature target/SPF overrides.
        parameter_overrides : dict or None
            Marxan parameter overrides.
        config : SolverConfig or None
            Solver configuration.

        Returns
        -------
        Scenario
            The newly created and added scenario.
        """
        from pymarxan.models.problem import apply_feature_overrides

        modified = apply_feature_overrides(problem, overrides) if overrides else problem.clone()

        if parameter_overrides:
            for k, v in parameter_overrides.items():
                modified.parameters[k] = v

        solutions = solver.solve(modified, config)
        best = min(solutions, key=lambda s: s.objective) if solutions else solutions[0]

        params = dict(modified.parameters)
        scenario = Scenario(
            name=name,
            solution=best,
            parameters=params,
            feature_overrides=overrides if overrides else None,
        )
        self._scenarios.append(scenario)
        return scenario
```

Add the necessary import at the top of the file (inside the method to avoid circular imports — already done with the `from pymarxan.models.problem import apply_feature_overrides` inline).

**Step 4: Run tests**

Run: `pytest tests/pymarxan/analysis/test_scenario_overrides.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add src/pymarxan/analysis/scenarios.py tests/pymarxan/analysis/test_scenario_overrides.py
git commit -m "feat(analysis): add run_with_overrides for solving with modified targets"
```

---

### Task 5: Full regression + lint + coverage

**Files:** None modified — verification only.

**Step 1: Run full test suite**

Run: `pytest tests/ -v --cov --cov-report=term-missing --cov-fail-under=75`
Expected: All tests pass, coverage >= 75%

**Step 2: Lint**

Run: `ruff check src/ tests/`
Expected: Clean (fix any issues found)

**Step 3: Type check**

Run: `mypy src/pymarxan/ --ignore-missing-imports`
Expected: Clean

**Step 4: Commit any fixes**

```bash
git add -A
git commit -m "chore: Phase 17 regression — lint and type fixes"
```

(Only commit if there are fixes. Skip if clean.)

---

## Summary

| Task | Description | New Tests |
|------|-------------|-----------|
| 1 | `apply_feature_overrides` | 7 |
| 2 | `ConservationProblem.clone()` | 4 |
| 3 | Scenario `feature_overrides` + `clone_scenario` | 9 |
| 4 | `run_with_overrides` | 2 |
| 5 | Full regression | 0 |
| **Total** | | **~22** |
