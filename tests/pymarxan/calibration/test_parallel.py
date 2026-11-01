"""Tests for parallel sweep execution."""
from __future__ import annotations

import os
import time
from pathlib import Path

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
    assert result.boundaries == [5.0, 5.0]
    assert small_problem.parameters["BLM"] == 99.0


def test_parallel_sweep_raises_what_run_sweep_raises():
    """A worker's error reaches the caller, as in run_sweep; only a point with no
    solutions is skipped.

    Every ValueError used to count as an infeasible point. With the whole problem in the
    workers (v0.47.0), a NaN in prob.dat, which run_sweep reports, silently dropped the
    sweep points that read it (v0.47.0 plan review).
    """
    from pymarxan.calibration.sweep import run_sweep
    from pymarxan.solvers.heuristic import HeuristicSolver

    base = _mode1_problem()
    prob = base.probability.copy()
    prob["probability"] = [0.5, np.nan, 0.9]
    problem = base.copy_with(probability=prob)
    config = SweepConfig(
        param_dicts=[{"PROBABILITYWEIGHTING": 0.0}, {"PROBABILITYWEIGHTING": 2.0}],
        solver_config=SolverConfig(num_solutions=1, seed=42),
    )
    with pytest.raises(ValueError, match="non-finite"):
        run_sweep(problem, HeuristicSolver(), config)
    with pytest.raises(ValueError, match="non-finite"):
        run_sweep_parallel(problem, HeuristicSolver(), config, max_workers=2)


# --- v0.47.0 review: the sweep runs in workers and fails fast, in sweep order -------


class _PidSolver(_StubSolver):
    """Records the process each point ran in."""

    def solve(
        self, problem: ConservationProblem, config: SolverConfig | None = None
    ) -> list[Solution]:
        sols = super().solve(problem, config)
        sols[0].metadata = {"pid": os.getpid()}
        return sols


def test_parallel_sweep_runs_points_in_worker_processes(
    small_problem: ConservationProblem,
):
    """The points are solved in worker processes, so the pickling tests above test
    pickling: a sequential fallback or a thread pool would pass them all."""
    config = SweepConfig(param_dicts=[{"BLM": 0.0}, {"BLM": 1.0}])
    result = run_sweep_parallel(small_problem, _PidSolver(), config, max_workers=2)
    assert len(result.solutions) == 2
    assert all(s.metadata["pid"] != os.getpid() for s in result.solutions)


class _FailFirstSolver(_StubSolver):
    """Point BLM=0 raises at once; every other point leaves a marker file in
    MARKER_DIR, then takes a second."""

    def solve(
        self, problem: ConservationProblem, config: SolverConfig | None = None
    ) -> list[Solution]:
        blm = problem.parameters["BLM"]
        if blm == 0.0:
            raise ValueError("bad sweep point")
        (Path(problem.parameters["MARKER_DIR"]) / f"{blm:g}").touch()
        time.sleep(1.0)
        return super().solve(problem, config)


def test_parallel_sweep_cancels_queued_points_after_an_error(
    small_problem: ConservationProblem, tmp_path: Path,
):
    """After a worker's error the points still queued are cancelled, not solved and
    thrown away. The points already running, or already handed to a worker, still
    finish, so a few markers appear, never all 19."""
    config = SweepConfig(
        param_dicts=[{"BLM": float(b), "MARKER_DIR": str(tmp_path)} for b in range(20)],
    )
    with pytest.raises(ValueError, match="bad sweep point"):
        run_sweep_parallel(small_problem, _FailFirstSolver(), config, max_workers=2)
    assert len(list(tmp_path.iterdir())) < 19


class _TwoErrorsSolver(_StubSolver):
    """Point BLM=1 (index 1) raises TypeError at once; point BLM=0 (index 0) waits until
    index 1 has failed, then raises ValueError; other points succeed. No timing: the
    marker file orders the two errors."""

    def solve(
        self, problem: ConservationProblem, config: SolverConfig | None = None
    ) -> list[Solution]:
        blm = problem.parameters["BLM"]
        marker = Path(problem.parameters["MARKER_DIR"]) / "index1_failed"
        if blm == 1.0:
            marker.touch()
            raise TypeError("index 1 failed")
        if blm == 0.0:
            deadline = time.monotonic() + 60.0
            while not marker.exists() and time.monotonic() < deadline:
                time.sleep(0.05)
            time.sleep(0.5)
            raise ValueError("index 0 failed")
        return super().solve(problem, config)


def test_parallel_sweep_raises_the_lowest_index_error(
    small_problem: ConservationProblem, tmp_path: Path,
):
    """When several points fail, the caller gets the lowest-index point's error, as from
    run_sweep, not the error that happened to finish first."""
    config = SweepConfig(
        param_dicts=[{"BLM": b, "MARKER_DIR": str(tmp_path)} for b in (0.0, 1.0, 5.0)],
    )
    with pytest.raises(ValueError, match="index 0"):
        run_sweep_parallel(small_problem, _TwoErrorsSolver(), config, max_workers=2)
