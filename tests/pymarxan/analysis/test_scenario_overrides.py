"""Tests for Scenario feature overrides and clone_scenario."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pymarxan.analysis.scenarios import Scenario, ScenarioSet
from pymarxan.models.problem import ConservationProblem
from pymarxan.solvers.base import Solution, SolverConfig
from pymarxan.solvers.mip_solver import MIPSolver


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
        s = Scenario(
            name="old", solution=_make_solution(), parameters={"BLM": "1.0"}
        )
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
            "original",
            "modified",
            parameter_overrides={"BLM": "5.0"},
        )
        assert cloned.parameters["BLM"] == "5.0"

    def test_clone_with_feature_overrides(self):
        ss = ScenarioSet()
        ss.add("original", _make_solution())
        overrides = {1: {"target": 50.0}}
        cloned = ss.clone_scenario(
            "original",
            "override",
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

    def test_clone_feature_overrides_isolated_from_source(self):
        """Adding new feature_overrides on the clone must not mutate the source.

        Without proper deepcopy, the clone could share the source's nested
        feature_overrides dict; any later merge would silently mutate the
        original scenario's overrides.
        """
        ss = ScenarioSet()
        ss.add(
            "original",
            _make_solution(),
            feature_overrides={1: {"target": 10.0}},
        )
        cloned = ss.clone_scenario(
            "original",
            "clone",
            feature_overrides={2: {"target": 20.0}},
        )
        original = ss.get("original")
        # Source must still have only feature 1
        assert set(original.feature_overrides.keys()) == {1}
        # Clone must have the merged keys
        assert set(cloned.feature_overrides.keys()) == {1, 2}
        # Mutating the clone's inner dict must not bleed into the source
        cloned.feature_overrides[1]["target"] = 999.0
        assert original.feature_overrides[1]["target"] == 10.0

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


def _make_problem():
    return ConservationProblem(
        planning_units=pd.DataFrame(
            {
                "id": [1, 2, 3, 4, 5, 6],
                "cost": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                "status": [0, 0, 0, 0, 0, 0],
            }
        ),
        features=pd.DataFrame(
            {
                "id": [1, 2],
                "name": ["f1", "f2"],
                "target": [5.0, 5.0],
                "spf": [1.0, 1.0],
            }
        ),
        pu_vs_features=pd.DataFrame(
            {
                "species": [1, 1, 1, 2, 2, 2],
                "pu": [1, 2, 3, 4, 5, 6],
                "amount": [5.0, 3.0, 2.0, 5.0, 3.0, 2.0],
            }
        ),
    )


class TestRunWithOverridesInfeasible:
    def test_run_with_overrides_handles_infeasible(self, tiny_problem):
        """Should raise RuntimeError when solver returns empty solutions."""
        from unittest.mock import MagicMock

        ss = ScenarioSet()
        mock_solver = MagicMock()
        mock_solver.solve.return_value = []
        with pytest.raises(RuntimeError, match="no solutions"):
            ss.run_with_overrides(
                "test", tiny_problem, mock_solver, overrides={}
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
