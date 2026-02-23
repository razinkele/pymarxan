"""Tests for calibration functions handling empty solver results."""
from __future__ import annotations

from pathlib import Path

import pytest

from pymarxan.io.readers import load_project
from pymarxan.solvers.base import Solution, Solver, SolverConfig

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "simple"


class InfeasibleSolver(Solver):
    """Mock solver that always returns empty results (infeasible)."""
    def name(self) -> str:
        return "infeasible"

    def supports_zones(self) -> bool:
        return False

    def solve(self, problem, config=None):
        return []


class TestBLMEmptyGuard:
    def test_calibrate_blm_skips_infeasible(self):
        from pymarxan.calibration.blm import calibrate_blm
        problem = load_project(DATA_DIR)
        solver = InfeasibleSolver()
        result = calibrate_blm(problem, solver, blm_values=[0.0, 1.0])
        assert len(result.solutions) == 0


class TestSPFEmptyGuard:
    def test_calibrate_spf_skips_infeasible(self):
        from pymarxan.calibration.spf import calibrate_spf
        problem = load_project(DATA_DIR)
        solver = InfeasibleSolver()
        result = calibrate_spf(problem, solver, max_iterations=2)
        assert result.solution is None or result.history is not None


class TestSweepEmptyGuard:
    def test_run_sweep_skips_infeasible(self):
        from pymarxan.calibration.sweep import SweepConfig, run_sweep
        problem = load_project(DATA_DIR)
        solver = InfeasibleSolver()
        config = SweepConfig(param_dicts=[{"BLM": 0.0}, {"BLM": 1.0}])
        result = run_sweep(problem, solver, config)
        assert len(result.solutions) == 0


class TestSensitivityEmptyGuard:
    def test_run_sensitivity_skips_infeasible(self):
        from pymarxan.calibration.sensitivity import SensitivityConfig, run_sensitivity
        problem = load_project(DATA_DIR)
        solver = InfeasibleSolver()
        config = SensitivityConfig(multipliers=[1.0])
        result = run_sensitivity(problem, solver, config)
        assert len(result.runs) == 0


class TestParallelEmptyGuard:
    def test_run_sweep_parallel_skips_infeasible(self):
        from pymarxan.calibration.parallel import run_sweep_parallel
        from pymarxan.calibration.sweep import SweepConfig
        problem = load_project(DATA_DIR)
        solver = InfeasibleSolver()
        config = SweepConfig(param_dicts=[{"BLM": 0.0}])
        result = run_sweep_parallel(problem, solver, config, max_workers=1)
        assert len(result.solutions) == 0
