"""Tests for ZoneMIPSolver."""
import copy
from pathlib import Path

import numpy as np
import pandas as pd
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


def test_mip_objective_matches_reported_objective_with_non_unit_lengths():
    """The MIP's own objective equals Solution.objective when lengths != 1 (issue #8)."""
    problem = load_zone_project(DATA_DIR)
    bnd = problem.boundary.copy()
    lengths = {(1, 2): 2.0, (2, 3): 0.5, (3, 4): 3.0}
    for (a, b), length in lengths.items():
        bnd.loc[(bnd["id1"] == a) & (bnd["id2"] == b), "boundary"] = length
    # A negative zone boundary cost rewards cut/uncut edges, so the optimum has some.
    zbc = problem.zone_boundary_costs.copy()
    zbc.loc[zbc["zone1"] != zbc["zone2"], "cost"] = -50.0
    problem = problem.copy_with(boundary=bnd, zone_boundary_costs=zbc)
    sol = ZoneMIPSolver().solve(problem, SolverConfig(num_solutions=1))[0]
    zones = sol.zone_assignment
    assert (zones[1:] != zones[:-1]).any() and (zones > 0).all()
    assert sol.objective == pytest.approx(sol.metadata["mip_objective_value"], abs=1e-6)


@pytest.mark.parametrize("rows", [
    {"zone1": [1], "zone2": [2], "cost": [-50.0]},
    {"zone1": [2], "zone2": [1], "cost": [-50.0]},
])
def test_one_direction_zone_boundary_row_matches_both_directions(rows):
    """A single (z1, z2) row means the same as listing both orders (MarZone)."""
    base = load_zone_project(DATA_DIR)
    both = base.copy_with(zone_boundary_costs=pd.DataFrame(
        {"zone1": [1, 2], "zone2": [2, 1], "cost": [-50.0, -50.0]}))
    one = base.copy_with(zone_boundary_costs=pd.DataFrame(rows))
    config = SolverConfig(num_solutions=1)
    sol_both = ZoneMIPSolver().solve(both, config)[0]
    sol_one = ZoneMIPSolver().solve(one, config)[0]
    assert sol_one.metadata["mip_objective_value"] == pytest.approx(
        sol_both.metadata["mip_objective_value"], abs=1e-6)
    assert sol_one.objective == pytest.approx(sol_one.metadata["mip_objective_value"], abs=1e-6)


@pytest.mark.parametrize("blm", [0.0, 1.0, 2.0])
@pytest.mark.parametrize("same_zone_cost", [-1000.0, 40.0])
def test_same_zone_boundary_costs_enter_the_mip(same_zone_cost, blm):
    """zbc(z, z) rows are charged inside zone z by the MIP too (#10).

    Exact optimum by enumerating every assignment that meets the zone and overall
    targets, which the MIP enforces as hard constraints.
    """
    import itertools

    from pymarxan.zones.objective import (
        check_overall_targets,
        check_zone_targets,
        compute_zone_objective,
    )

    base = load_zone_project(DATA_DIR)
    problem = base.copy_with(zone_boundary_costs=pd.DataFrame(
        {"zone1": [1, 2], "zone2": [2, 2], "cost": [50.0, same_zone_cost]}),
        parameters={**base.parameters, "BLM": blm})
    best = min(
        compute_zone_objective(problem, np.array(a), blm)
        for a in itertools.product((0, 1, 2), repeat=4)
        if all(check_zone_targets(problem, np.array(a)).values())
        and all(check_overall_targets(problem, np.array(a)).values())
    )
    sol = ZoneMIPSolver().solve(problem, SolverConfig(num_solutions=1))[0]
    assert sol.metadata["mip_objective_value"] == pytest.approx(best, abs=1e-6)
    assert sol.objective == pytest.approx(best, abs=1e-6)
