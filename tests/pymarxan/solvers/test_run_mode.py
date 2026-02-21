"""Tests for RunModePipeline (RUNMODE 0-6)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pymarxan.io.readers import load_project
from pymarxan.solvers.base import Solution, SolverConfig
from pymarxan.solvers.run_mode import RunModePipeline

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "simple"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def problem():
    """Load the simple test problem with fast SA settings."""
    prob = load_project(DATA_DIR)
    prob.parameters["NUMITNS"] = 1_000
    prob.parameters["NUMTEMP"] = 100
    prob.parameters["BLM"] = 0.0
    prob.features["spf"] = 100.0
    # Remove any RUNMODE from input.dat so constructor value is used by default
    prob.parameters.pop("RUNMODE", None)
    return prob


# ---------------------------------------------------------------------------
# Solver interface
# ---------------------------------------------------------------------------


class TestSolverInterface:
    def test_name(self):
        solver = RunModePipeline(runmode=0)
        assert solver.name() == "run_mode_pipeline"

    def test_supports_zones(self):
        solver = RunModePipeline(runmode=0)
        assert solver.supports_zones() is False


# ---------------------------------------------------------------------------
# Invalid runmode
# ---------------------------------------------------------------------------


class TestInvalidRunmode:
    def test_negative_raises(self):
        with pytest.raises(ValueError, match="runmode"):
            RunModePipeline(runmode=-1)

    def test_too_high_raises(self):
        with pytest.raises(ValueError, match="runmode"):
            RunModePipeline(runmode=7)

    def test_non_integer_string_raises(self):
        with pytest.raises((ValueError, TypeError)):
            RunModePipeline(runmode="abc")


# ---------------------------------------------------------------------------
# Parametrized: each runmode 0-6 produces at least one solution
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("runmode", [0, 1, 2, 3, 4, 5, 6])
def test_runmode_produces_solution(problem, runmode):
    """Each RUNMODE should produce at least one valid solution."""
    solver = RunModePipeline(runmode=runmode)
    config = SolverConfig(num_solutions=1, seed=42)
    solutions = solver.solve(problem, config)
    assert len(solutions) == 1
    sol = solutions[0]
    assert isinstance(sol, Solution)
    assert isinstance(sol.selected, np.ndarray)
    assert len(sol.selected) == problem.n_planning_units
    assert sol.cost >= 0
    assert sol.objective >= 0


# ---------------------------------------------------------------------------
# RUNMODE 0: SA only -- metadata should indicate SA
# ---------------------------------------------------------------------------


class TestRunmode0:
    def test_sa_metadata(self, problem):
        solver = RunModePipeline(runmode=0)
        config = SolverConfig(num_solutions=1, seed=42)
        solutions = solver.solve(problem, config)
        sol = solutions[0]
        assert "solver" in sol.metadata
        assert "Simulated Annealing" in sol.metadata["solver"]

    def test_multiple_solutions(self, problem):
        solver = RunModePipeline(runmode=0)
        config = SolverConfig(num_solutions=3, seed=42)
        solutions = solver.solve(problem, config)
        assert len(solutions) == 3


# ---------------------------------------------------------------------------
# RUNMODE 1: Heuristic only -- metadata should indicate greedy
# ---------------------------------------------------------------------------


class TestRunmode1:
    def test_greedy_metadata(self, problem):
        solver = RunModePipeline(runmode=1)
        config = SolverConfig(num_solutions=1, seed=42)
        solutions = solver.solve(problem, config)
        sol = solutions[0]
        assert "solver" in sol.metadata
        assert sol.metadata["solver"] == "greedy"


# ---------------------------------------------------------------------------
# RUNMODE from problem.parameters
# ---------------------------------------------------------------------------


class TestRunmodeFromParameters:
    def test_parameter_overrides_constructor(self, problem):
        """RUNMODE in problem.parameters should override the constructor."""
        problem.parameters["RUNMODE"] = 1
        solver = RunModePipeline(runmode=0)  # constructor says SA
        config = SolverConfig(num_solutions=1, seed=42)
        solutions = solver.solve(problem, config)
        sol = solutions[0]
        # Should have used heuristic (mode 1) not SA (mode 0)
        assert sol.metadata["solver"] == "greedy"


# ---------------------------------------------------------------------------
# RUNMODE 5: Full pipeline -- heuristic then SA then iterative improvement
# ---------------------------------------------------------------------------


class TestRunmode5:
    def test_full_pipeline_valid_solution(self, problem):
        solver = RunModePipeline(runmode=5)
        config = SolverConfig(num_solutions=1, seed=42)
        solutions = solver.solve(problem, config)
        sol = solutions[0]
        assert isinstance(sol, Solution)
        assert sol.objective > 0
        assert sol.cost >= 0

    def test_full_pipeline_positive_objective(self, problem):
        solver = RunModePipeline(runmode=5)
        config = SolverConfig(num_solutions=2, seed=42)
        solutions = solver.solve(problem, config)
        for sol in solutions:
            assert sol.objective > 0


# ---------------------------------------------------------------------------
# RUNMODE 6: Iterative improvement only (from all-selected)
# ---------------------------------------------------------------------------


class TestRunmode6:
    def test_iterative_only_produces_solution(self, problem):
        solver = RunModePipeline(runmode=6)
        config = SolverConfig(num_solutions=1, seed=42)
        solutions = solver.solve(problem, config)
        assert len(solutions) == 1
        sol = solutions[0]
        assert isinstance(sol, Solution)
        assert sol.cost >= 0


# ---------------------------------------------------------------------------
# RUNMODE 2: SA then iterative improvement
# ---------------------------------------------------------------------------


class TestRunmode2:
    def test_sa_then_improvement(self, problem):
        """RUNMODE 2 should produce a solution with iterative improvement applied."""
        problem.parameters["ITIMPTYPE"] = 1  # enable removal pass
        solver = RunModePipeline(runmode=2)
        config = SolverConfig(num_solutions=1, seed=42)
        solutions = solver.solve(problem, config)
        assert len(solutions) == 1
        sol = solutions[0]
        assert sol.objective > 0


# ---------------------------------------------------------------------------
# RUNMODE 3: Heuristic then iterative improvement
# ---------------------------------------------------------------------------


class TestRunmode3:
    def test_heuristic_then_improvement(self, problem):
        """RUNMODE 3 should apply iterative improvement to heuristic result."""
        problem.parameters["ITIMPTYPE"] = 1
        solver = RunModePipeline(runmode=3)
        config = SolverConfig(num_solutions=1, seed=42)
        solutions = solver.solve(problem, config)
        assert len(solutions) == 1
        sol = solutions[0]
        assert sol.objective > 0


# ---------------------------------------------------------------------------
# RUNMODE 4: Heuristic then SA (pick best)
# ---------------------------------------------------------------------------


class TestRunmode4:
    def test_picks_best_of_heuristic_and_sa(self, problem):
        """RUNMODE 4 should return the best of heuristic and SA."""
        solver = RunModePipeline(runmode=4)
        config = SolverConfig(num_solutions=1, seed=42)
        solutions = solver.solve(problem, config)
        assert len(solutions) == 1
        sol = solutions[0]
        assert sol.objective > 0

    def test_multiple_solutions(self, problem):
        solver = RunModePipeline(runmode=4)
        config = SolverConfig(num_solutions=3, seed=42)
        solutions = solver.solve(problem, config)
        assert len(solutions) == 3
