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
