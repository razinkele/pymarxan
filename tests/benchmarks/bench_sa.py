"""Simulated annealing performance benchmarks."""
from __future__ import annotations

import time

from pymarxan.solvers.base import SolverConfig
from pymarxan.solvers.simulated_annealing import SimulatedAnnealingSolver
from tests.benchmarks.conftest import make_problem


class TestSAPerformance:
    """Performance benchmarks for the SA solver at various problem sizes."""

    def test_small_100pu_10feat(self) -> None:
        """100 PU, 10 features, 10K iterations -- must complete in <2s."""
        problem = make_problem(n_pu=100, n_feat=10, density=0.3, seed=42)
        problem.parameters["NUMITNS"] = 10_000
        problem.parameters["NUMTEMP"] = 100

        solver = SimulatedAnnealingSolver()
        config = SolverConfig(num_solutions=1, seed=42)

        start = time.perf_counter()
        solutions = solver.solve(problem, config)
        elapsed = time.perf_counter() - start

        assert len(solutions) == 1
        assert solutions[0].cost > 0
        assert elapsed < 2.0, f"Small SA took {elapsed:.2f}s, expected <2s"

    def test_medium_1k_pu_50feat(self) -> None:
        """1K PU, 50 features, 100K iterations -- must complete in <15s."""
        problem = make_problem(n_pu=1000, n_feat=50, density=0.3, seed=42)
        problem.parameters["NUMITNS"] = 100_000
        problem.parameters["NUMTEMP"] = 1000

        solver = SimulatedAnnealingSolver()
        config = SolverConfig(num_solutions=1, seed=42)

        start = time.perf_counter()
        solutions = solver.solve(problem, config)
        elapsed = time.perf_counter() - start

        assert len(solutions) == 1
        assert solutions[0].cost > 0
        assert elapsed < 15.0, f"Medium SA took {elapsed:.2f}s, expected <15s"

    def test_large_5k_pu_100feat(self) -> None:
        """5K PU, 100 features, 100K iterations -- must complete in <60s."""
        problem = make_problem(n_pu=5000, n_feat=100, density=0.3, seed=42)
        problem.parameters["NUMITNS"] = 100_000
        problem.parameters["NUMTEMP"] = 1000

        solver = SimulatedAnnealingSolver()
        config = SolverConfig(num_solutions=1, seed=42)

        start = time.perf_counter()
        solutions = solver.solve(problem, config)
        elapsed = time.perf_counter() - start

        assert len(solutions) == 1
        assert solutions[0].cost > 0
        assert elapsed < 60.0, f"Large SA took {elapsed:.2f}s, expected <60s"
