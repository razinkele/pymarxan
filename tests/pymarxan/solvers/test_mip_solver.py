from pathlib import Path

import numpy as np

from pymarxan.io.readers import load_project
from pymarxan.solvers.base import SolverConfig
from pymarxan.solvers.mip_solver import MIPSolver

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "simple"

class TestMIPSolver:
    def setup_method(self):
        self.problem = load_project(DATA_DIR)
        self.solver = MIPSolver()

    def test_solver_name(self):
        assert self.solver.name() == "MIP (PuLP)"

    def test_solver_available(self):
        assert self.solver.available()

    def test_does_not_support_zones(self):
        assert not self.solver.supports_zones()

    def test_solve_returns_solutions(self):
        config = SolverConfig(num_solutions=1)
        solutions = self.solver.solve(self.problem, config)
        assert len(solutions) == 1
        sol = solutions[0]
        assert isinstance(sol.selected, np.ndarray)
        assert len(sol.selected) == 6
        assert sol.cost > 0

    def test_all_targets_met(self):
        config = SolverConfig(num_solutions=1)
        solutions = self.solver.solve(self.problem, config)
        sol = solutions[0]
        assert sol.all_targets_met, f"Unmet targets: {sol.targets_met}"

    def test_solution_cost_is_optimal_or_near(self):
        config = SolverConfig(num_solutions=1)
        solutions = self.solver.solve(self.problem, config)
        sol = solutions[0]
        total_cost = self.problem.planning_units["cost"].sum()
        assert sol.cost < total_cost

    def test_locked_in_units_selected(self):
        self.problem.planning_units.loc[self.problem.planning_units["id"] == 1, "status"] = 2
        config = SolverConfig(num_solutions=1)
        solutions = self.solver.solve(self.problem, config)
        sol = solutions[0]
        pu_ids = self.problem.planning_units["id"].tolist()
        idx = pu_ids.index(1)
        assert sol.selected[idx], "Locked-in PU 1 should be selected"

    def test_locked_out_units_not_selected(self):
        self.problem.planning_units.loc[self.problem.planning_units["id"] == 6, "status"] = 3
        config = SolverConfig(num_solutions=1)
        solutions = self.solver.solve(self.problem, config)
        sol = solutions[0]
        pu_ids = self.problem.planning_units["id"].tolist()
        idx = pu_ids.index(6)
        assert not sol.selected[idx], "Locked-out PU 6 should not be selected"

    def test_blm_zero_ignores_boundary(self):
        self.problem.parameters["BLM"] = 0.0
        config = SolverConfig(num_solutions=1)
        solutions = self.solver.solve(self.problem, config)
        sol = solutions[0]
        assert sol.all_targets_met
        assert abs(sol.objective - sol.cost) < 0.01
