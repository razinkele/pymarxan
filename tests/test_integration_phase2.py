"""Integration tests for Phase 2 features."""
from pathlib import Path

import pytest

from pymarxan.analysis.irreplaceability import compute_irreplaceability
from pymarxan.analysis.selection_freq import compute_selection_frequency
from pymarxan.calibration.blm import calibrate_blm
from pymarxan.calibration.spf import calibrate_spf
from pymarxan.io.exporters import export_solution_csv, export_summary_csv
from pymarxan.io.readers import load_project
from pymarxan.solvers.base import SolverConfig
from pymarxan.solvers.mip_solver import MIPSolver
from pymarxan.solvers.simulated_annealing import SimulatedAnnealingSolver

DATA_DIR = Path(__file__).parent / "data" / "simple"


class TestSAIntegration:
    @pytest.mark.integration
    @pytest.mark.slow
    def test_sa_finds_feasible_solution(self):
        problem = load_project(DATA_DIR)
        # Override the high iteration count from input.dat so the test
        # finishes quickly while still giving SA enough room to converge.
        problem.parameters["NUMITNS"] = 10_000
        problem.parameters["NUMTEMP"] = 100
        problem.parameters["BLM"] = 0.0
        problem.features["spf"] = 100.0
        solver = SimulatedAnnealingSolver()
        config = SolverConfig(num_solutions=5, seed=42)
        solutions = solver.solve(problem, config)
        assert len(solutions) == 5
        feasible = [s for s in solutions if s.all_targets_met]
        assert len(feasible) > 0

    @pytest.mark.integration
    @pytest.mark.slow
    def test_sa_selection_frequency(self):
        problem = load_project(DATA_DIR)
        problem.parameters["NUMITNS"] = 10_000
        problem.parameters["NUMTEMP"] = 100
        problem.parameters["BLM"] = 0.0
        problem.features["spf"] = 100.0
        solver = SimulatedAnnealingSolver()
        config = SolverConfig(num_solutions=10, seed=42)
        solutions = solver.solve(problem, config)
        freq = compute_selection_frequency(solutions)
        assert freq.n_solutions == 10
        assert all(0.0 <= f <= 1.0 for f in freq.frequencies)


class TestCalibrationIntegration:
    @pytest.mark.integration
    def test_blm_calibration_with_mip(self):
        problem = load_project(DATA_DIR)
        solver = MIPSolver()
        result = calibrate_blm(
            problem, solver,
            blm_values=[0.0, 1.0, 10.0],
        )
        assert len(result.blm_values) == 3
        assert all(s.all_targets_met for s in result.solutions)

    @pytest.mark.integration
    def test_spf_calibration_converges(self):
        problem = load_project(DATA_DIR)
        solver = MIPSolver()
        result = calibrate_spf(problem, solver, max_iterations=5)
        assert result.solution.all_targets_met


class TestExportIntegration:
    @pytest.mark.integration
    def test_export_after_solve(self, tmp_path):
        problem = load_project(DATA_DIR)
        solver = MIPSolver()
        sol = solver.solve(problem, SolverConfig(num_solutions=1))[0]
        export_solution_csv(problem, sol, tmp_path / "sol.csv")
        export_summary_csv(problem, sol, tmp_path / "summary.csv")
        assert (tmp_path / "sol.csv").exists()
        assert (tmp_path / "summary.csv").exists()


class TestIrreplaceabilityIntegration:
    @pytest.mark.integration
    def test_irreplaceability_scores(self):
        problem = load_project(DATA_DIR)
        scores = compute_irreplaceability(problem)
        assert len(scores) == 6
        assert all(0.0 <= v <= 1.0 for v in scores.values())
