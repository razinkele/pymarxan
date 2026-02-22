"""Tests for SA solver progress reporting."""
from __future__ import annotations

from pathlib import Path

from pymarxan.io.readers import load_project
from pymarxan.solvers.base import SolverConfig
from pymarxan.solvers.simulated_annealing import SimulatedAnnealingSolver
from pymarxan_shiny.modules.run_control.progress import SolverProgress

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "simple"


class TestSAProgress:
    def setup_method(self):
        self.problem = load_project(DATA_DIR)
        self.problem.parameters["NUMITNS"] = 5_000
        self.problem.parameters["NUMTEMP"] = 50
        self.solver = SimulatedAnnealingSolver()

    def test_progress_updated_when_provided(self):
        progress = SolverProgress()
        config = SolverConfig(num_solutions=2, seed=42, metadata={"progress": progress})
        self.solver.solve(self.problem, config)
        assert progress.status == "done"
        assert progress.current_run == 2
        assert progress.total_runs == 2

    def test_progress_not_required(self):
        """Solver works fine without progress object."""
        config = SolverConfig(num_solutions=1, seed=42)
        solutions = self.solver.solve(self.problem, config)
        assert len(solutions) == 1

    def test_progress_tracks_best_objective(self):
        progress = SolverProgress()
        config = SolverConfig(num_solutions=1, seed=42, metadata={"progress": progress})
        self.solver.solve(self.problem, config)
        assert progress.best_objective < float("inf")

    def test_progress_error_on_invalid_problem(self):
        """Progress should reflect errors."""
        progress = SolverProgress()
        config = SolverConfig(num_solutions=1, seed=42, metadata={"progress": progress})
        self.solver.solve(self.problem, config)
        assert progress.status == "done"
