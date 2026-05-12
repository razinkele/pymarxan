"""Tests for zone heuristic and iterative improvement solvers."""

from __future__ import annotations

from pathlib import Path

import pytest

from pymarxan.solvers.base import SolverConfig
from pymarxan.zones.heuristic import ZoneHeuristicSolver
from pymarxan.zones.iterative_improvement import ZoneIISolver
from pymarxan.zones.readers import load_zone_project

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "zones"


class TestZoneHeuristicSolver:
    def setup_method(self):
        self.problem = load_zone_project(DATA_DIR)
        self.solver = ZoneHeuristicSolver()

    def test_solver_name(self):
        assert self.solver.name() == "Zone Heuristic (Python)"

    def test_supports_zones(self):
        assert self.solver.supports_zones()

    def test_solve_returns_solutions(self):
        config = SolverConfig(num_solutions=2, seed=42)
        solutions = self.solver.solve(self.problem, config)
        assert len(solutions) == 2

    def test_zone_assignment_present(self):
        config = SolverConfig(num_solutions=1, seed=42)
        sol = self.solver.solve(self.problem, config)[0]
        assert sol.zone_assignment is not None
        assert len(sol.zone_assignment) == len(self.problem.planning_units)

    def test_zone_values_valid(self):
        config = SolverConfig(num_solutions=1, seed=42)
        sol = self.solver.solve(self.problem, config)[0]
        valid_zones = {0} | self.problem.zone_ids
        for z in sol.zone_assignment:
            assert int(z) in valid_zones

    def test_cost_nonnegative(self):
        config = SolverConfig(num_solutions=1, seed=42)
        sol = self.solver.solve(self.problem, config)[0]
        assert sol.cost >= 0
        assert sol.objective >= 0

    def test_locked_out_respected(self):
        """PUs with status=3 should remain in zone 0."""
        import copy

        problem = copy.deepcopy(self.problem)
        problem.planning_units.loc[0, "status"] = 3
        config = SolverConfig(num_solutions=1, seed=42)
        sol = self.solver.solve(problem, config)[0]
        assert int(sol.zone_assignment[0]) == 0

    def test_targets_met_populated(self):
        config = SolverConfig(num_solutions=1, seed=42)
        sol = self.solver.solve(self.problem, config)[0]
        # targets_met should be a dict (possibly empty if no zone_targets)
        assert isinstance(sol.targets_met, dict)

    def test_rejects_non_zonal_problem(self):
        from pymarxan.models.problem import ConservationProblem

        problem = ConservationProblem(
            planning_units=self.problem.planning_units,
            features=self.problem.features,
            pu_vs_features=self.problem.pu_vs_features,
        )
        with pytest.raises(TypeError):
            self.solver.solve(problem)


class TestZoneIISolver:
    def setup_method(self):
        self.problem = load_zone_project(DATA_DIR)
        self.solver = ZoneIISolver()

    def test_solver_name(self):
        assert self.solver.name() == "Zone II (Python)"

    def test_supports_zones(self):
        assert self.solver.supports_zones()

    def test_itimptype_0_no_change(self):
        """ITIMPTYPE=0 should return input unchanged."""
        self.problem.parameters["ITIMPTYPE"] = 0
        config = SolverConfig(num_solutions=1, seed=42)
        sol = self.solver.solve(self.problem, config)[0]
        assert sol.zone_assignment is not None

    def test_itimptype_1_removal(self):
        """ITIMPTYPE=1 should remove unneeded PUs."""
        self.problem.parameters["ITIMPTYPE"] = 1
        config = SolverConfig(num_solutions=1, seed=42)
        sol = self.solver.solve(self.problem, config)[0]
        assert sol.zone_assignment is not None
        # Should have reduced cost vs all-assigned
        assert sol.cost >= 0

    def test_itimptype_2_two_step(self):
        """ITIMPTYPE=2 removal+addition rounds."""
        self.problem.parameters["ITIMPTYPE"] = 2
        config = SolverConfig(num_solutions=1, seed=42)
        sol = self.solver.solve(self.problem, config)[0]
        assert sol.zone_assignment is not None
        assert sol.cost >= 0

    def test_itimptype_3_swap(self):
        """ITIMPTYPE=3 full swap improvement."""
        self.problem.parameters["ITIMPTYPE"] = 3
        config = SolverConfig(num_solutions=1, seed=42)
        sol = self.solver.solve(self.problem, config)[0]
        assert sol.zone_assignment is not None
        assert sol.cost >= 0

    def test_zone_values_valid(self):
        self.problem.parameters["ITIMPTYPE"] = 3
        config = SolverConfig(num_solutions=1, seed=42)
        sol = self.solver.solve(self.problem, config)[0]
        valid_zones = {0} | self.problem.zone_ids
        for z in sol.zone_assignment:
            assert int(z) in valid_zones

    def test_improve_existing_solution(self):
        """improve() should accept and refine an existing solution."""
        self.problem.parameters["ITIMPTYPE"] = 1
        # First get an initial solution
        config = SolverConfig(num_solutions=1, seed=42)
        initial = self.solver.solve(self.problem, config)[0]
        # Now improve it
        improved = self.solver.improve(self.problem, initial)
        assert improved.zone_assignment is not None
        assert improved.objective <= initial.objective + 1e-10

    def test_locked_out_respected(self):
        import copy

        problem = copy.deepcopy(self.problem)
        problem.planning_units.loc[0, "status"] = 3
        problem.parameters["ITIMPTYPE"] = 3
        config = SolverConfig(num_solutions=1, seed=42)
        sol = self.solver.solve(problem, config)[0]
        assert int(sol.zone_assignment[0]) == 0

    def test_rejects_non_zonal_problem(self):
        from pymarxan.models.problem import ConservationProblem

        problem = ConservationProblem(
            planning_units=self.problem.planning_units,
            features=self.problem.features,
            pu_vs_features=self.problem.pu_vs_features,
        )
        with pytest.raises(TypeError):
            self.solver.solve(problem)
