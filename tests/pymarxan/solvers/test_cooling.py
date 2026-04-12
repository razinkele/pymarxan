"""Tests for SA cooling schedule variants."""
from __future__ import annotations

import copy
from pathlib import Path

import pytest

from pymarxan.io.readers import load_project
from pymarxan.solvers.base import SolverConfig
from pymarxan.solvers.cooling import CoolingSchedule
from pymarxan.solvers.simulated_annealing import SimulatedAnnealingSolver

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "simple"


class TestGeometricSchedule:
    def test_initial_temperature(self):
        s = CoolingSchedule.geometric(initial_temp=100.0, num_steps=1000)
        assert s.temperature(0) == pytest.approx(100.0)

    def test_final_temperature(self):
        s = CoolingSchedule.geometric(initial_temp=100.0, final_temp=0.01, num_steps=1000)
        assert s.temperature(1000) == pytest.approx(0.01, rel=1e-3)

    def test_monotone_decreasing(self):
        s = CoolingSchedule.geometric(initial_temp=100.0, num_steps=100)
        temps = [s.temperature(i) for i in range(101)]
        for i in range(len(temps) - 1):
            assert temps[i] >= temps[i + 1]


class TestExponentialSchedule:
    def test_initial_temperature(self):
        s = CoolingSchedule.exponential(initial_temp=50.0, num_steps=500)
        assert s.temperature(0) == pytest.approx(50.0)

    def test_final_temperature(self):
        s = CoolingSchedule.exponential(initial_temp=50.0, final_temp=0.01, num_steps=500)
        assert s.temperature(500) == pytest.approx(0.01, rel=1e-3)

    def test_monotone_decreasing(self):
        s = CoolingSchedule.exponential(initial_temp=50.0, num_steps=100)
        temps = [s.temperature(i) for i in range(101)]
        for i in range(len(temps) - 1):
            assert temps[i] >= temps[i + 1]


class TestLinearSchedule:
    def test_initial_temperature(self):
        s = CoolingSchedule.linear(initial_temp=100.0, num_steps=1000)
        assert s.temperature(0) == pytest.approx(100.0)

    def test_final_temperature(self):
        s = CoolingSchedule.linear(initial_temp=100.0, final_temp=0.01, num_steps=1000)
        assert s.temperature(1000) == pytest.approx(0.01, rel=1e-3)

    def test_clamped_below_final(self):
        s = CoolingSchedule.linear(initial_temp=100.0, final_temp=1.0, num_steps=100)
        # Past num_steps, should not go below final_temp
        assert s.temperature(200) == pytest.approx(1.0)

    def test_monotone_decreasing(self):
        s = CoolingSchedule.linear(initial_temp=100.0, num_steps=100)
        temps = [s.temperature(i) for i in range(101)]
        for i in range(len(temps) - 1):
            assert temps[i] >= temps[i + 1]


class TestLundyMeesSchedule:
    def test_initial_temperature(self):
        s = CoolingSchedule.lundy_mees(initial_temp=100.0, num_steps=1000)
        assert s.temperature(0) == pytest.approx(100.0)

    def test_final_temperature(self):
        s = CoolingSchedule.lundy_mees(initial_temp=100.0, final_temp=0.01, num_steps=1000)
        assert s.temperature(1000) == pytest.approx(0.01, rel=1e-2)

    def test_monotone_decreasing(self):
        s = CoolingSchedule.lundy_mees(initial_temp=100.0, num_steps=100)
        temps = [s.temperature(i) for i in range(101)]
        for i in range(len(temps) - 1):
            assert temps[i] >= temps[i + 1]

    def test_convergence_reciprocal(self):
        """1/T(k) should increase linearly with step for Lundy-Mees."""
        s = CoolingSchedule.lundy_mees(initial_temp=10.0, final_temp=0.01, num_steps=500)
        inv_t0 = 1.0 / s.temperature(0)
        inv_t250 = 1.0 / s.temperature(250)
        inv_t500 = 1.0 / s.temperature(500)
        # Check approximate linearity: gap should be roughly equal
        gap1 = inv_t250 - inv_t0
        gap2 = inv_t500 - inv_t250
        assert gap1 == pytest.approx(gap2, rel=1e-2)


class TestUnknownSchedule:
    def test_unknown_name_raises(self):
        s = CoolingSchedule(name="unknown", initial_temp=1.0)
        with pytest.raises(ValueError, match="Unknown cooling schedule"):
            s.temperature(0)


class TestSAWithStartTemp:
    def setup_method(self):
        self.problem = load_project(DATA_DIR)
        self.problem.parameters["NUMITNS"] = 5_000
        self.problem.parameters["NUMTEMP"] = 50

    def test_starttemp_used(self):
        problem = copy.deepcopy(self.problem)
        problem.parameters["STARTTEMP"] = 500.0
        solver = SimulatedAnnealingSolver()
        config = SolverConfig(num_solutions=1, seed=42)
        solutions = solver.solve(problem, config)
        assert len(solutions) == 1
        assert solutions[0].metadata["initial_temp"] == pytest.approx(500.0)

    @pytest.mark.slow
    def test_starttemp_reproducible(self):
        problem = copy.deepcopy(self.problem)
        problem.parameters["STARTTEMP"] = 200.0
        solver = SimulatedAnnealingSolver()
        config = SolverConfig(num_solutions=1, seed=99)
        s1 = solver.solve(problem, config)[0]
        s2 = solver.solve(problem, config)[0]
        assert s1.cost == s2.cost


class TestSACoolingSchedules:
    def setup_method(self):
        self.problem = load_project(DATA_DIR)
        self.problem.parameters["NUMITNS"] = 5_000
        self.problem.parameters["NUMTEMP"] = 50

    @pytest.mark.parametrize("schedule", ["geometric", "exponential", "linear", "lundy_mees"])
    def test_schedule_produces_solution(self, schedule):
        problem = copy.deepcopy(self.problem)
        problem.parameters["COOLING"] = schedule
        solver = SimulatedAnnealingSolver()
        config = SolverConfig(num_solutions=1, seed=42)
        solutions = solver.solve(problem, config)
        assert len(solutions) == 1
        assert solutions[0].cost >= 0

    def test_invalid_schedule_raises(self):
        problem = copy.deepcopy(self.problem)
        problem.parameters["COOLING"] = "invalid_schedule"
        solver = SimulatedAnnealingSolver()
        config = SolverConfig(num_solutions=1, seed=42)
        with pytest.raises(ValueError, match="Unknown COOLING schedule"):
            solver.solve(problem, config)

    @pytest.mark.slow
    @pytest.mark.parametrize("schedule", ["geometric", "exponential", "linear", "lundy_mees"])
    def test_schedule_with_starttemp(self, schedule):
        problem = copy.deepcopy(self.problem)
        problem.parameters["COOLING"] = schedule
        problem.parameters["STARTTEMP"] = 300.0
        problem.parameters["NUMITNS"] = 10_000
        problem.parameters["NUMTEMP"] = 100
        solver = SimulatedAnnealingSolver()
        config = SolverConfig(num_solutions=2, seed=42)
        solutions = solver.solve(problem, config)
        assert len(solutions) == 2
        for sol in solutions:
            assert sol.metadata["initial_temp"] == pytest.approx(300.0)
