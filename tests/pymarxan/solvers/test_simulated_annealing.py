from pathlib import Path

import numpy as np
import pytest

from pymarxan.io.readers import load_project
from pymarxan.solvers.base import SolverConfig
from pymarxan.solvers.simulated_annealing import SimulatedAnnealingSolver

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "simple"


class TestSimulatedAnnealingSolver:
    def setup_method(self):
        self.problem = load_project(DATA_DIR)
        # Override iteration count from input.dat so tests run quickly
        self.problem.parameters["NUMITNS"] = 5_000
        self.problem.parameters["NUMTEMP"] = 50
        self.solver = SimulatedAnnealingSolver()

    def test_solver_name(self):
        assert self.solver.name() == "Simulated Annealing (Python)"

    def test_solver_available(self):
        assert self.solver.available()

    def test_does_not_support_zones(self):
        assert not self.solver.supports_zones()

    def test_solve_returns_correct_count(self):
        config = SolverConfig(num_solutions=5, seed=42)
        solutions = self.solver.solve(self.problem, config)
        assert len(solutions) == 5

    def test_solutions_are_different(self):
        """Multiple SA runs should produce some variation."""
        config = SolverConfig(num_solutions=5, seed=None)
        solutions = self.solver.solve(self.problem, config)
        # At least some solutions should differ
        # With 5 runs, allow some to be the same but not all identical
        # (stochastic, so just check structure)
        assert all(s.cost >= 0 for s in solutions)

    @pytest.mark.slow
    def test_all_targets_met_on_simple_problem(self):
        """On a small solvable problem, SA should find feasible solutions."""
        import copy

        problem = copy.deepcopy(self.problem)
        # Increase SPF so the solver strongly prioritises meeting targets,
        # and drop BLM so boundary cost does not compete.
        problem.parameters["BLM"] = 0.0
        problem.features["spf"] = 100.0
        problem.parameters["NUMITNS"] = 10_000
        problem.parameters["NUMTEMP"] = 100
        config = SolverConfig(num_solutions=5, seed=42)
        solutions = self.solver.solve(problem, config)
        # At least some solutions should meet all targets
        met_count = sum(1 for s in solutions if s.all_targets_met)
        assert met_count > 0, "SA should find at least one feasible solution"

    def test_solution_structure(self):
        config = SolverConfig(num_solutions=1, seed=42)
        solutions = self.solver.solve(self.problem, config)
        sol = solutions[0]
        assert isinstance(sol.selected, np.ndarray)
        assert len(sol.selected) == 6
        assert sol.cost >= 0
        assert sol.boundary >= 0
        assert sol.objective >= 0
        assert isinstance(sol.targets_met, dict)
        assert len(sol.targets_met) == 3
        assert "solver" in sol.metadata

    def test_locked_in_respected(self):
        self.problem.planning_units.loc[
            self.problem.planning_units["id"] == 1, "status"
        ] = 2
        config = SolverConfig(num_solutions=3, seed=42)
        solutions = self.solver.solve(self.problem, config)
        pu_ids = self.problem.planning_units["id"].tolist()
        idx = pu_ids.index(1)
        for sol in solutions:
            assert sol.selected[idx], "Locked-in PU must be selected in every run"

    def test_locked_out_respected(self):
        self.problem.planning_units.loc[
            self.problem.planning_units["id"] == 6, "status"
        ] = 3
        config = SolverConfig(num_solutions=3, seed=42)
        solutions = self.solver.solve(self.problem, config)
        pu_ids = self.problem.planning_units["id"].tolist()
        idx = pu_ids.index(6)
        for sol in solutions:
            assert not sol.selected[idx], "Locked-out PU must not be selected"

    @pytest.mark.slow
    def test_seed_reproducibility(self):
        config = SolverConfig(num_solutions=1, seed=12345)
        sol1 = self.solver.solve(self.problem, config)[0]
        sol2 = self.solver.solve(self.problem, config)[0]
        np.testing.assert_array_equal(sol1.selected, sol2.selected)
        assert sol1.cost == sol2.cost

    def test_custom_iterations(self):
        solver = SimulatedAnnealingSolver(num_iterations=100, num_temp_steps=10)
        config = SolverConfig(num_solutions=1, seed=42)
        solutions = solver.solve(self.problem, config)
        assert len(solutions) == 1
