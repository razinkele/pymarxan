"""Tests for solution aliasing bug — [sol] * N creates references."""
from __future__ import annotations

import copy
from pathlib import Path

import pytest

from pymarxan.io.readers import load_project
from pymarxan.solvers.base import SolverConfig
from pymarxan.solvers.mip_solver import MIPSolver

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "simple"


class TestMIPSolutionAliasing:
    def test_solutions_are_independent(self):
        """Mutating one solution's metadata should not affect others."""
        problem = load_project(DATA_DIR)
        solver = MIPSolver()
        config = SolverConfig(num_solutions=3, seed=42)
        solutions = solver.solve(problem, config)
        if len(solutions) < 2:
            pytest.skip("MIP returned fewer than 2 solutions")
        solutions[0].metadata["test_key"] = "test_value"
        assert "test_key" not in solutions[1].metadata


class TestSAAllLockedAliasing:
    def test_all_locked_solutions_are_independent(self):
        """When all PUs are locked, SA returns [sol]*N — test independence."""
        problem = load_project(DATA_DIR)
        problem.planning_units["status"] = 2  # All locked-in
        from pymarxan.solvers.simulated_annealing import SimulatedAnnealingSolver
        solver = SimulatedAnnealingSolver()
        config = SolverConfig(num_solutions=3, seed=42)
        solutions = solver.solve(problem, config)
        assert len(solutions) == 3
        solutions[0].metadata["test_key"] = "mutated"
        assert "test_key" not in solutions[1].metadata
