"""Tests for sensitivity analysis module."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pymarxan.calibration.sensitivity import (
    SensitivityConfig,
    SensitivityResult,
    run_sensitivity,
)
from pymarxan.models.problem import ConservationProblem
from pymarxan.solvers.base import Solution, Solver, SolverConfig


class _StubSolver(Solver):
    def solve(
        self, problem: ConservationProblem, config: SolverConfig | None = None
    ) -> list[Solution]:
        n = problem.n_planning_units
        total_target = sum(
            float(r["target"]) for _, r in problem.features.iterrows()
        )
        return [
            Solution(
                selected=np.ones(n, dtype=bool),
                cost=total_target * 2,
                boundary=5.0,
                objective=total_target * 2 + 5.0,
                targets_met={
                    int(r["id"]): True for _, r in problem.features.iterrows()
                },
            )
        ]

    def name(self) -> str:
        return "stub"

    def supports_zones(self) -> bool:
        return False


@pytest.fixture()
def problem() -> ConservationProblem:
    pu = pd.DataFrame({"id": [1, 2, 3], "cost": [10.0, 20.0, 15.0], "status": [0, 0, 0]})
    feat = pd.DataFrame({
        "id": [1, 2],
        "name": ["bird", "mammal"],
        "target": [5.0, 8.0],
        "spf": [1.0, 1.0],
    })
    puvspr = pd.DataFrame({
        "species": [1, 1, 2, 2],
        "pu": [1, 2, 2, 3],
        "amount": [3.0, 4.0, 5.0, 6.0],
    })
    return ConservationProblem(
        planning_units=pu, features=feat, pu_vs_features=puvspr,
    )


def test_sensitivity_default_multipliers(problem: ConservationProblem):
    config = SensitivityConfig()
    result = run_sensitivity(problem, _StubSolver(), config)
    assert isinstance(result, SensitivityResult)
    assert len(result.runs) == 10  # 2 features x 5 multipliers


def test_sensitivity_custom_multipliers(problem: ConservationProblem):
    config = SensitivityConfig(multipliers=[0.5, 1.0, 1.5])
    result = run_sensitivity(problem, _StubSolver(), config)
    assert len(result.runs) == 6


def test_sensitivity_single_feature(problem: ConservationProblem):
    config = SensitivityConfig(feature_ids=[1], multipliers=[0.5, 1.0, 2.0])
    result = run_sensitivity(problem, _StubSolver(), config)
    assert len(result.runs) == 3


def test_sensitivity_to_dataframe(problem: ConservationProblem):
    config = SensitivityConfig(multipliers=[0.8, 1.0, 1.2])
    result = run_sensitivity(problem, _StubSolver(), config)
    df = result.to_dataframe()
    assert "feature_id" in df.columns
    assert "multiplier" in df.columns
    assert "cost" in df.columns
    assert "objective" in df.columns
    assert len(df) == 6


def test_sensitivity_baseline_at_1(problem: ConservationProblem):
    config = SensitivityConfig(multipliers=[1.0])
    result = run_sensitivity(problem, _StubSolver(), config)
    for run in result.runs:
        assert run["multiplier"] == 1.0
