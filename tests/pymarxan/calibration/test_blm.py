from pathlib import Path

from pymarxan.calibration.blm import BLMResult, calibrate_blm
from pymarxan.io.readers import load_project
from pymarxan.solvers.mip_solver import MIPSolver

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "simple"


class TestCalibrateBLM:
    def setup_method(self):
        self.problem = load_project(DATA_DIR)
        self.solver = MIPSolver()

    def test_returns_blm_result(self):
        result = calibrate_blm(
            self.problem, self.solver,
            blm_values=[0.0, 1.0, 5.0],
        )
        assert isinstance(result, BLMResult)

    def test_correct_number_of_points(self):
        result = calibrate_blm(
            self.problem, self.solver,
            blm_values=[0.0, 1.0, 5.0, 10.0],
        )
        assert len(result.blm_values) == 4
        assert len(result.costs) == 4
        assert len(result.boundaries) == 4

    def test_blm_range_shortcut(self):
        result = calibrate_blm(
            self.problem, self.solver,
            blm_min=0.0, blm_max=10.0, blm_steps=5,
        )
        assert len(result.blm_values) == 5

    def test_cost_increases_with_blm(self):
        result = calibrate_blm(
            self.problem, self.solver,
            blm_values=[0.0, 10.0, 100.0],
        )
        assert result.costs[-1] >= result.costs[0]

    def test_boundary_decreases_with_blm(self):
        result = calibrate_blm(
            self.problem, self.solver,
            blm_values=[0.0, 100.0],
        )
        assert result.boundaries[-1] <= result.boundaries[0]

    def test_solutions_stored(self):
        result = calibrate_blm(
            self.problem, self.solver,
            blm_values=[0.0, 1.0],
        )
        assert len(result.solutions) == 2
        assert all(s.all_targets_met for s in result.solutions)
