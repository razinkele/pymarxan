"""Zone SA performance benchmarks."""
from __future__ import annotations

import time

from pymarxan.solvers.base import SolverConfig
from pymarxan.zones.solver import ZoneSASolver
from tests.benchmarks.conftest import make_zone_problem


class TestZoneSAPerformance:
    """Performance benchmarks for the Zone SA solver at various problem sizes."""

    def test_small_100pu_3zones(self) -> None:
        """100 PU, 10 features, 3 zones, 10K iterations -- must complete in <3s."""
        problem = make_zone_problem(
            n_pu=100, n_feat=10, n_zones=3, density=0.3, seed=42,
        )
        problem.parameters["NUMITNS"] = 10_000
        problem.parameters["NUMTEMP"] = 100

        solver = ZoneSASolver()
        config = SolverConfig(num_solutions=1, seed=42)

        start = time.perf_counter()
        solutions = solver.solve(problem, config)
        elapsed = time.perf_counter() - start

        assert len(solutions) == 1
        assert solutions[0].cost > 0
        assert elapsed < 3.0, f"Small Zone SA took {elapsed:.2f}s, expected <3s"

    def test_medium_1k_pu_3zones(self) -> None:
        """1K PU, 50 features, 3 zones, 100K iterations -- must complete in <30s."""
        problem = make_zone_problem(
            n_pu=1000, n_feat=50, n_zones=3, density=0.3, seed=42,
        )
        problem.parameters["NUMITNS"] = 100_000
        problem.parameters["NUMTEMP"] = 1000

        solver = ZoneSASolver()
        config = SolverConfig(num_solutions=1, seed=42)

        start = time.perf_counter()
        solutions = solver.solve(problem, config)
        elapsed = time.perf_counter() - start

        assert len(solutions) == 1
        assert solutions[0].cost > 0
        assert elapsed < 30.0, f"Medium Zone SA took {elapsed:.2f}s, expected <30s"
