import copy
from pathlib import Path

import numpy as np
import pytest

from pymarxan.solvers.base import SolverConfig
from pymarxan.zones.readers import load_zone_project
from pymarxan.zones.solver import ZoneSASolver

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "zones"


class TestZoneSASolver:
    def setup_method(self):
        self.problem = load_zone_project(DATA_DIR)
        self.problem.parameters["NUMITNS"] = 5_000
        self.problem.parameters["NUMTEMP"] = 50
        self.solver = ZoneSASolver()

    def test_solver_name(self):
        assert self.solver.name() == "Zone SA (Python)"

    def test_supports_zones(self):
        assert self.solver.supports_zones()

    def test_solve_returns_solutions(self):
        config = SolverConfig(num_solutions=3, seed=42)
        solutions = self.solver.solve(self.problem, config)
        assert len(solutions) == 3

    def test_zone_assignment_present(self):
        config = SolverConfig(num_solutions=1, seed=42)
        sol = self.solver.solve(self.problem, config)[0]
        assert sol.zone_assignment is not None
        assert len(sol.zone_assignment) == 4

    def test_zone_values_valid(self):
        config = SolverConfig(num_solutions=1, seed=42)
        sol = self.solver.solve(self.problem, config)[0]
        valid_zones = {0, 1, 2}
        for z in sol.zone_assignment:
            assert int(z) in valid_zones

    @pytest.mark.slow
    def test_cost_nonnegative(self):
        config = SolverConfig(num_solutions=3, seed=42)
        for sol in self.solver.solve(self.problem, config):
            assert sol.cost >= 0
            assert sol.objective >= 0

    @pytest.mark.slow
    def test_seed_reproducibility(self):
        config = SolverConfig(num_solutions=1, seed=99)
        sol1 = self.solver.solve(self.problem, config)[0]
        sol2 = self.solver.solve(self.problem, config)[0]
        np.testing.assert_array_equal(sol1.zone_assignment, sol2.zone_assignment)

    def test_zone_sa_solution_has_penalty_and_shortfall(self):
        """Zone SA should populate penalty and shortfall fields on Solution."""
        self.problem.parameters["NUMITNS"] = 100
        self.problem.parameters["NUMTEMP"] = 10
        config = SolverConfig(num_solutions=1, seed=42)
        solutions = self.solver.solve(self.problem, config)
        sol = solutions[0]
        assert hasattr(sol, "penalty")
        assert hasattr(sol, "shortfall")
        zone_targets_met = sol.metadata.get("zone_targets_met", {})
        if not all(zone_targets_met.values()):
            assert sol.penalty > 0.0

    def test_all_locked_in_returns_solution(self):
        """Zone SA should handle all PUs locked-in without crashing."""
        problem = copy.deepcopy(self.problem)
        problem.planning_units["status"] = 2  # All locked-in
        problem.parameters["NUMITNS"] = 100
        config = SolverConfig(num_solutions=1, seed=42)
        solutions = self.solver.solve(problem, config)
        assert len(solutions) == 1
        assert solutions[0].cost >= 0

    def test_all_locked_out_returns_solution(self):
        """Zone SA should handle all PUs locked-out without crashing."""
        problem = copy.deepcopy(self.problem)
        problem.planning_units["status"] = 3  # All locked-out
        problem.parameters["NUMITNS"] = 100
        config = SolverConfig(num_solutions=1, seed=42)
        solutions = self.solver.solve(problem, config)
        assert len(solutions) == 1

    def test_cooling_counts_all_iterations(self):
        """SA should cool at the same rate regardless of same-zone skips."""
        problem = copy.deepcopy(self.problem)
        problem.parameters["NUMITNS"] = 1000
        problem.parameters["NUMTEMP"] = 10
        config = SolverConfig(num_solutions=1, seed=42)
        solutions = self.solver.solve(problem, config)
        assert solutions[0].objective < 1e10

    def test_status_1_starts_in_zone(self):
        """PUs with status=1 should start in first non-zero zone, stay swappable."""
        problem = copy.deepcopy(self.problem)
        problem.planning_units["status"] = 0
        problem.planning_units.loc[0, "status"] = 1
        problem.parameters["NUMITNS"] = 100
        problem.parameters["NUMTEMP"] = 10
        config = SolverConfig(num_solutions=1, seed=42)
        solutions = self.solver.solve(problem, config)
        assert len(solutions) == 1
        assert solutions[0].zone_assignment is not None

    def test_rejects_non_zonal_problem(self):
        """ZoneSASolver should raise TypeError for plain ConservationProblem."""
        from pymarxan.io.readers import load_project
        plain = load_project(Path(__file__).parent.parent.parent / "data" / "simple")
        config = SolverConfig(num_solutions=1, seed=42)
        with pytest.raises(TypeError, match="ZonalProblem"):
            self.solver.solve(plain, config)

    @pytest.mark.slow
    def test_finds_feasible_on_simple_problem(self):
        problem = copy.deepcopy(self.problem)
        problem.parameters["BLM"] = 0.0
        problem.features["spf"] = 100.0
        problem.parameters["NUMITNS"] = 10_000
        problem.parameters["NUMTEMP"] = 100
        config = SolverConfig(num_solutions=5, seed=42)
        solutions = self.solver.solve(problem, config)
        assigned = sum(1 for s in solutions if s.zone_assignment.any())
        assert assigned > 0
