"""Tests for SA solver iteration history recording."""
from __future__ import annotations

from pathlib import Path

from pymarxan.io.readers import load_project
from pymarxan.solvers.base import SolverConfig
from pymarxan.solvers.simulated_annealing import SimulatedAnnealingSolver

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "simple"


class TestSAHistory:
    def setup_method(self):
        self.problem = load_project(DATA_DIR)
        self.problem.parameters["NUMITNS"] = 10_000
        self.problem.parameters["NUMTEMP"] = 100
        self.solver = SimulatedAnnealingSolver()

    def test_history_present_in_metadata(self):
        config = SolverConfig(num_solutions=1, seed=42)
        solutions = self.solver.solve(self.problem, config)
        assert "history" in solutions[0].metadata
        history = solutions[0].metadata["history"]
        assert "iteration" in history
        assert "objective" in history
        assert "best_objective" in history
        assert "temperature" in history

    def test_history_has_correct_length(self):
        config = SolverConfig(num_solutions=1, seed=42)
        solutions = self.solver.solve(self.problem, config)
        history = solutions[0].metadata["history"]
        n = len(history["iteration"])
        # 10K iterations sampled every 1000 → ~10 points, plus initial
        assert 5 <= n <= 20
        assert len(history["objective"]) == n
        assert len(history["best_objective"]) == n
        assert len(history["temperature"]) == n

    def test_history_iterations_monotonic(self):
        config = SolverConfig(num_solutions=1, seed=42)
        solutions = self.solver.solve(self.problem, config)
        iters = solutions[0].metadata["history"]["iteration"]
        for i in range(1, len(iters)):
            assert iters[i] > iters[i - 1]

    def test_best_objective_non_increasing(self):
        config = SolverConfig(num_solutions=1, seed=42)
        solutions = self.solver.solve(self.problem, config)
        best = solutions[0].metadata["history"]["best_objective"]
        for i in range(1, len(best)):
            assert best[i] <= best[i - 1] + 1e-9

    def test_history_per_run(self):
        """Each run gets its own history."""
        config = SolverConfig(num_solutions=3, seed=42)
        solutions = self.solver.solve(self.problem, config)
        for sol in solutions:
            assert "history" in sol.metadata
            assert len(sol.metadata["history"]["iteration"]) > 0

    def test_existing_tests_still_pass(self):
        """Verify basic solve still works with history recording."""
        config = SolverConfig(num_solutions=2, seed=42)
        solutions = self.solver.solve(self.problem, config)
        assert len(solutions) == 2
        for sol in solutions:
            assert sol.cost >= 0
            assert sol.objective >= 0
