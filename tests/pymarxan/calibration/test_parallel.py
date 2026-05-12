"""Tests for parallel sweep execution."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pymarxan.calibration.parallel import run_sweep_parallel
from pymarxan.calibration.sweep import SweepConfig, SweepResult
from pymarxan.models.problem import ConservationProblem
from pymarxan.solvers.base import Solution, Solver, SolverConfig


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


class _InfeasibleAtZeroSolver(Solver):
    """Returns no solutions when BLM=0, otherwise a stub solution."""

    def solve(
        self, problem: ConservationProblem, config: SolverConfig | None = None
    ) -> list[Solution]:
        blm = problem.parameters.get("BLM", 1.0)
        if blm == 0.0:
            return []
        n = problem.n_planning_units
        return [
            Solution(
                selected=np.ones(n, dtype=bool),
                cost=10.0 + blm,
                boundary=5.0,
                objective=10.0 + blm,
                targets_met={1: True},
            )
        ]

    def name(self) -> str:
        return "infeasible_at_zero"

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


def test_parallel_sweep_skips_infeasible(small_problem: ConservationProblem):
    """Parallel sweep must skip infeasible points instead of crashing.

    The sequential ``run_sweep`` skips points where the solver returns ``[]``
    (Review 5 fix). The parallel version must mirror that behaviour so a single
    infeasible point does not lose all the completed parallel results.
    """
    config = SweepConfig(
        param_dicts=[{"BLM": 0.0}, {"BLM": 1.0}, {"BLM": 5.0}],
    )
    result = run_sweep_parallel(
        small_problem, _InfeasibleAtZeroSolver(), config, max_workers=2,
    )
    # Infeasible point dropped; the two feasible ones survive
    assert len(result.solutions) == 2
    assert len(result.param_dicts) == 2
    # Surviving points should be BLM=1.0 and BLM=5.0
    blms = sorted(p["BLM"] for p in result.param_dicts)
    assert blms == [1.0, 5.0]
