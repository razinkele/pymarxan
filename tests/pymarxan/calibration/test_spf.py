from pathlib import Path

from pymarxan.io.readers import load_project
from pymarxan.calibration.spf import calibrate_spf, SPFResult
from pymarxan.solvers.mip_solver import MIPSolver

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "simple"


class TestCalibrateSPF:
    def setup_method(self):
        self.problem = load_project(DATA_DIR)
        self.solver = MIPSolver()

    def test_returns_spf_result(self):
        result = calibrate_spf(
            self.problem, self.solver, max_iterations=3,
        )
        assert isinstance(result, SPFResult)

    def test_final_solution_meets_targets(self):
        result = calibrate_spf(
            self.problem, self.solver, max_iterations=5,
        )
        assert result.solution.all_targets_met

    def test_adjusted_spf_values(self):
        result = calibrate_spf(
            self.problem, self.solver, max_iterations=3,
        )
        assert isinstance(result.final_spf, dict)
        assert len(result.final_spf) == 3

    def test_history_recorded(self):
        result = calibrate_spf(
            self.problem, self.solver, max_iterations=3,
        )
        assert len(result.history) >= 1
        assert len(result.history) <= 3
