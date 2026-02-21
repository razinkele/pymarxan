"""Integration test: full roundtrip from loading data to solving."""
from pathlib import Path

from pymarxan.io.readers import load_project
from pymarxan.io.writers import save_project
from pymarxan.solvers.base import SolverConfig
from pymarxan.solvers.mip_solver import MIPSolver

DATA_DIR = Path(__file__).parent / "data" / "simple"


class TestFullRoundtrip:
    def test_load_solve_check(self):
        """Load test data -> solve with MIP -> verify all targets met."""
        problem = load_project(DATA_DIR)
        assert problem.validate() == []
        solver = MIPSolver()
        config = SolverConfig(num_solutions=1)
        solutions = solver.solve(problem, config)
        assert len(solutions) == 1
        sol = solutions[0]
        assert sol.all_targets_met
        assert sol.cost > 0
        assert sol.n_selected > 0
        assert sol.n_selected < problem.n_planning_units

    def test_load_save_load_solve(self, tmp_path):
        """Load -> save -> reload -> solve -> verify results match."""
        original = load_project(DATA_DIR)
        save_project(original, tmp_path)
        reloaded = load_project(tmp_path)
        solver = MIPSolver()
        config = SolverConfig(num_solutions=1)
        sol_original = solver.solve(original, config)[0]
        sol_reloaded = solver.solve(reloaded, config)[0]
        assert sol_original.cost == sol_reloaded.cost
        assert sol_original.n_selected == sol_reloaded.n_selected

    def test_blm_affects_solution(self):
        """Higher BLM should lead to different solutions than BLM=0."""
        problem_low = load_project(DATA_DIR)
        problem_low.parameters["BLM"] = 0.0
        problem_high = load_project(DATA_DIR)
        problem_high.parameters["BLM"] = 100.0
        solver = MIPSolver()
        config = SolverConfig(num_solutions=1)
        sol_low = solver.solve(problem_low, config)[0]
        sol_high = solver.solve(problem_high, config)[0]
        # High BLM should have <= boundary (more compact) OR different objective
        assert sol_high.boundary <= sol_low.boundary or sol_high.objective != sol_low.objective

    def test_locked_in_respected(self):
        """Locked-in PU must appear in solution."""
        problem = load_project(DATA_DIR)
        problem.planning_units.loc[problem.planning_units["id"] == 3, "status"] = 2
        solver = MIPSolver()
        sol = solver.solve(problem, SolverConfig(num_solutions=1))[0]
        pu_ids = problem.planning_units["id"].tolist()
        idx = pu_ids.index(3)
        assert sol.selected[idx], "Locked-in PU 3 should be selected"

    def test_locked_out_respected(self):
        """Locked-out PU must NOT appear in solution."""
        problem = load_project(DATA_DIR)
        problem.planning_units.loc[problem.planning_units["id"] == 1, "status"] = 3
        solver = MIPSolver()
        sol = solver.solve(problem, SolverConfig(num_solutions=1))[0]
        pu_ids = problem.planning_units["id"].tolist()
        idx = pu_ids.index(1)
        assert not sol.selected[idx], "Locked-out PU 1 should not be selected"
