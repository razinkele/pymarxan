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


# --- v0.47.0: workers receive the whole problem (spec §3.1, round 4 I-2) ----


class _TaggedProblem(ConservationProblem):
    """A subclass, so the test sees whether the workers keep the problem's type."""


class _RecordingSolver(Solver):
    """Records what the worker received in the solution metadata."""

    def solve(
        self, problem: ConservationProblem, config: SolverConfig | None = None
    ) -> list[Solution]:
        n = problem.n_planning_units
        return [
            Solution(
                selected=np.ones(n, dtype=bool),
                cost=1.0,
                boundary=0.0,
                objective=1.0,
                targets_met={1: True},
                metadata={
                    "type": type(problem).__name__,
                    "has_probability": problem.probability is not None,
                    "has_connectivity": problem.connectivity is not None,
                },
            )
        ]

    def name(self) -> str:
        return "recording"

    def supports_zones(self) -> bool:
        return False


def _mode1_problem() -> ConservationProblem:
    """PROBMODE 1 (risk premium) from a prob.dat table: the premium is in the objective."""
    pu = pd.DataFrame({"id": [1, 2, 3], "cost": [10.0, 12.0, 8.0], "status": [0, 0, 0]})
    feat = pd.DataFrame({"id": [1], "name": ["f1"], "target": [5.0], "spf": [100.0]})
    puvspr = pd.DataFrame({"species": [1, 1, 1], "pu": [1, 2, 3], "amount": [3.0, 4.0, 5.0]})
    prob = pd.DataFrame({"pu": [1, 2, 3], "probability": [0.5, 0.1, 0.9]})
    return ConservationProblem(
        planning_units=pu, features=feat, pu_vs_features=puvspr, probability=prob,
        parameters={"PROBMODE": 1, "PROBABILITYWEIGHTING": 2.0},
    )


def test_parallel_sweep_keeps_probability_table():
    """A prob.dat-driven objective agrees between run_sweep and run_sweep_parallel.

    Before v0.47.0 the workers rebuilt the problem from five fields and dropped
    ``problem.probability``, so the PROBMODE 1 premium vanished in the workers.
    """
    from pymarxan.calibration.sweep import run_sweep
    from pymarxan.solvers.heuristic import HeuristicSolver

    problem = _mode1_problem()
    config = SweepConfig(
        param_dicts=[{"BLM": 0.0}, {"BLM": 1.0}],
        # Seeded: the greedy heuristic breaks ties with random noise.
        solver_config=SolverConfig(num_solutions=1, seed=42),
    )
    seq = run_sweep(problem, HeuristicSolver(), config)
    par = run_sweep_parallel(problem, HeuristicSolver(), config, max_workers=2)
    assert par.objectives == pytest.approx(seq.objectives, rel=1e-12)
    assert [s.objective for s in par.solutions] == pytest.approx(
        [s.objective for s in seq.solutions], rel=1e-12,
    )


def test_parallel_sweep_keeps_subclass_and_optional_tables():
    """The workers see the problem's subclass, probability and connectivity tables."""
    base = _mode1_problem()
    problem = _TaggedProblem(
        planning_units=base.planning_units,
        features=base.features,
        pu_vs_features=base.pu_vs_features,
        probability=base.probability,
        connectivity=pd.DataFrame({"id1": [1], "id2": [2], "value": [1.0]}),
        parameters=dict(base.parameters),
    )
    config = SweepConfig(param_dicts=[{"BLM": 0.0}, {"BLM": 1.0}])
    result = run_sweep_parallel(problem, _RecordingSolver(), config, max_workers=2)
    assert len(result.solutions) == 2
    for sol in result.solutions:
        assert sol.metadata == {
            "type": "_TaggedProblem",
            "has_probability": True,
            "has_connectivity": True,
        }


def test_parallel_sweep_applies_point_parameters_on_whole_problem(
    small_problem: ConservationProblem,
):
    """Each point's parameters still override the problem's, and the caller's
    problem is left untouched."""
    small_problem.parameters["BLM"] = 99.0
    config = SweepConfig(param_dicts=[{"BLM": 0.0}, {"BLM": 5.0}])
    result = run_sweep_parallel(small_problem, _StubSolver(), config, max_workers=2)
    assert result.costs == [10.0, 15.0]
    assert small_problem.parameters["BLM"] == 99.0
