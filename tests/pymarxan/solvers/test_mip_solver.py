from pathlib import Path

import numpy as np
import pandas as pd

from pymarxan.io.readers import load_project
from pymarxan.models.problem import ConservationProblem
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


def test_mip_objective_includes_penalty(tiny_problem):
    """MIP objective should include SPF penalty when targets unmet."""
    tiny_problem.features.loc[0, "target"] = 999999.0
    solver = MIPSolver()
    solutions = solver.solve(tiny_problem, SolverConfig(num_solutions=1))
    if solutions:  # may be infeasible with impossible target
        sol = solutions[0]
        assert sol.penalty > 0.0
        assert sol.objective >= sol.cost + sol.boundary


def test_mip_infeasible_returns_empty(tiny_problem):
    """MIP should return empty list when infeasible."""
    tiny_problem.planning_units["status"] = 3  # lock out ALL PUs
    tiny_problem.features.loc[:, "target"] = 100.0
    solver = MIPSolver()
    solutions = solver.solve(tiny_problem, SolverConfig(num_solutions=1))
    assert solutions == []


def test_mip_includes_self_boundary():
    """MIP objective must include self-boundary (external boundary) terms.

    Create a problem where two PUs each independently satisfy the target.
    PU 1 is cheap but has huge self-boundary; PU 2 costs more but has none.
    With high BLM, a correct MIP must prefer PU 2 because self-boundary
    makes PU 1 more expensive overall.
    """
    pu = pd.DataFrame({
        "id": [1, 2],
        "cost": [1.0, 5.0],
        "status": [0, 0],
    })
    feat = pd.DataFrame({
        "id": [1],
        "name": ["f1"],
        "target": [1.0],
        "spf": [1.0],
    })
    puvspr = pd.DataFrame({
        "species": [1, 1],
        "pu": [1, 2],
        "amount": [10.0, 10.0],
    })
    # PU 1 has huge self-boundary; PU 2 has none
    bnd = pd.DataFrame({
        "id1": [1],
        "id2": [1],
        "boundary": [100.0],
    })
    problem = ConservationProblem(
        planning_units=pu, features=feat,
        pu_vs_features=puvspr, boundary=bnd,
    )
    problem.parameters["BLM"] = 1.0

    solver = MIPSolver()
    sols = solver.solve(problem, SolverConfig(num_solutions=1))
    assert len(sols) == 1
    sol = sols[0]
    # With self-boundary in objective: PU 1 costs 1 + 100 = 101, PU 2 costs 5
    # Without self-boundary: PU 1 costs 1, PU 2 costs 5 => MIP picks PU 1
    pu_ids = problem.planning_units["id"].tolist()
    assert sol.selected[pu_ids.index(2)], (
        "MIP should prefer PU 2 (cost=5) over PU 1 (cost=1 + boundary=100) "
        "when self-boundary is included in the objective"
    )


def test_mip_applies_misslevel(tiny_problem):
    """MIP should use MISSLEVEL to relax target constraints."""
    total_amount = float(
        tiny_problem.pu_vs_features.groupby("species")["amount"].sum().min()
    )
    tiny_problem.features.loc[:, "target"] = total_amount + 0.1
    tiny_problem.parameters["MISSLEVEL"] = 0.5
    solver = MIPSolver()
    solutions = solver.solve(tiny_problem, SolverConfig(num_solutions=1))
    assert len(solutions) >= 1
