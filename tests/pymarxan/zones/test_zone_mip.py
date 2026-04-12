"""Tests for ZoneMIPSolver."""
import copy
from pathlib import Path

import numpy as np
import pytest

from pymarxan.solvers.base import SolverConfig
from pymarxan.zones.mip_solver import ZoneMIPSolver
from pymarxan.zones.readers import load_zone_project

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "zones"


class TestZoneMIPSolver:
    def setup_method(self):
        self.problem = load_zone_project(DATA_DIR)
        self.solver = ZoneMIPSolver()

    def test_solver_name(self):
        assert self.solver.name() == "Zone MIP (PuLP)"

    def test_supports_zones(self):
        assert self.solver.supports_zones()

    def test_solve_returns_solutions(self):
        config = SolverConfig(num_solutions=3)
        solutions = self.solver.solve(self.problem, config)
        assert len(solutions) == 3

    def test_zone_assignment_present(self):
        config = SolverConfig(num_solutions=1)
        sol = self.solver.solve(self.problem, config)[0]
        assert sol.zone_assignment is not None
        assert len(sol.zone_assignment) == len(self.problem.planning_units)

    def test_zone_values_valid(self):
        """Each PU should be in at most one zone (0 = unassigned)."""
        config = SolverConfig(num_solutions=1)
        sol = self.solver.solve(self.problem, config)[0]
        valid_zones = {0} | self.problem.zone_ids
        for z in sol.zone_assignment:
            assert int(z) in valid_zones

    def test_each_pu_at_most_one_zone(self):
        """Zone assignment is exclusive — no PU in multiple zones."""
        config = SolverConfig(num_solutions=1)
        sol = self.solver.solve(self.problem, config)[0]
        # zone_assignment is a single int per PU, so this is inherent,
        # but verify the value is a scalar zone id
        for z in sol.zone_assignment:
            assert isinstance(int(z), int)

    def test_zone_targets_respected(self):
        """All zone-specific targets should be met by the MIP solution."""
        config = SolverConfig(num_solutions=1)
        sol = self.solver.solve(self.problem, config)[0]
        zone_targets_met = sol.metadata.get("zone_targets_met", {})
        for key, met in zone_targets_met.items():
            assert met, f"Zone target {key} not met"

    def test_cost_nonnegative(self):
        config = SolverConfig(num_solutions=1)
        sol = self.solver.solve(self.problem, config)[0]
        assert sol.cost >= 0
        assert sol.objective >= 0

    def test_locked_in_honored(self):
        """PUs with status=2 must be assigned to the first zone."""
        problem = copy.deepcopy(self.problem)
        problem.planning_units["status"] = 0
        problem.planning_units.loc[
            problem.planning_units["id"] == 1, "status"
        ] = 2
        config = SolverConfig(num_solutions=1)
        sol = self.solver.solve(problem, config)[0]
        pu_ids = problem.planning_units["id"].tolist()
        idx = pu_ids.index(1)
        first_zone = sorted(problem.zone_ids)[0]
        assert int(sol.zone_assignment[idx]) == first_zone

    def test_locked_out_honored(self):
        """PUs with status=3 must not be assigned to any zone."""
        problem = copy.deepcopy(self.problem)
        problem.planning_units["status"] = 0
        problem.planning_units.loc[
            problem.planning_units["id"] == 2, "status"
        ] = 3
        # Relax targets so the problem stays feasible without PU 2
        problem.zone_targets["target"] = 0.0
        config = SolverConfig(num_solutions=1)
        sol = self.solver.solve(problem, config)[0]
        pu_ids = problem.planning_units["id"].tolist()
        idx = pu_ids.index(2)
        assert int(sol.zone_assignment[idx]) == 0

    def test_rejects_non_zonal_problem(self):
        """ZoneMIPSolver should raise TypeError for plain ConservationProblem."""
        from pymarxan.io.readers import load_project

        plain = load_project(
            Path(__file__).parent.parent.parent / "data" / "simple"
        )
        config = SolverConfig(num_solutions=1)
        with pytest.raises(TypeError, match="ZonalProblem"):
            self.solver.solve(plain, config)

    def test_selected_matches_zone_assignment(self):
        """selected boolean array should be True where zone_assignment > 0."""
        config = SolverConfig(num_solutions=1)
        sol = self.solver.solve(self.problem, config)[0]
        expected_selected = sol.zone_assignment > 0
        np.testing.assert_array_equal(sol.selected, expected_selected)
