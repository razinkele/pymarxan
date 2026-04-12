"""Tests for zone connectivity penalty support."""
import copy

import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from pymarxan.solvers.base import SolverConfig
from pymarxan.zones.cache import ZoneProblemCache
from pymarxan.zones.objective import compute_zone_connectivity, compute_zone_objective
from pymarxan.zones.readers import load_zone_project
from pymarxan.zones.solver import ZoneSASolver

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "zones"


@pytest.fixture
def base_problem():
    return load_zone_project(DATA_DIR)


def _make_connectivity(problem, pairs, value=1.0):
    """Build a connectivity DataFrame from index pairs."""
    pu_ids = problem.planning_units["id"].values
    rows = []
    for i, j in pairs:
        rows.append({"id1": int(pu_ids[i]), "id2": int(pu_ids[j]), "value": value})
    return pd.DataFrame(rows)


class TestComputeZoneConnectivity:
    def test_no_connectivity_data(self, base_problem):
        """No connectivity data → 0.0 penalty."""
        problem = copy.deepcopy(base_problem)
        problem.connectivity = None
        assignment = np.array([1, 1, 0, 0])
        assert compute_zone_connectivity(problem, assignment) == 0.0

    def test_zero_weight(self, base_problem):
        """Zero CONNECTIVITY_WEIGHT → 0.0 penalty."""
        problem = copy.deepcopy(base_problem)
        problem.connectivity = _make_connectivity(problem, [(0, 1)])
        problem.parameters["CONNECTIVITY_WEIGHT"] = 0.0
        assignment = np.array([1, 1, 0, 0])
        assert compute_zone_connectivity(problem, assignment) == 0.0

    def test_same_zone_bonus(self, base_problem):
        """Two PUs in same zone with connection get bonus (negative)."""
        problem = copy.deepcopy(base_problem)
        problem.connectivity = _make_connectivity(problem, [(0, 1)], value=5.0)
        problem.parameters["CONNECTIVITY_WEIGHT"] = 1.0
        # Both in zone 1
        assignment = np.array([1, 1, 0, 0])
        result = compute_zone_connectivity(problem, assignment)
        assert result == -5.0

    def test_different_zone_no_bonus(self, base_problem):
        """Two PUs in different zones don't get bonus."""
        problem = copy.deepcopy(base_problem)
        problem.connectivity = _make_connectivity(problem, [(0, 1)], value=5.0)
        problem.parameters["CONNECTIVITY_WEIGHT"] = 1.0
        # Different zones
        assignment = np.array([1, 2, 0, 0])
        result = compute_zone_connectivity(problem, assignment)
        assert result == 0.0

    def test_unassigned_no_bonus(self, base_problem):
        """PU in zone 0 doesn't get connectivity bonus."""
        problem = copy.deepcopy(base_problem)
        problem.connectivity = _make_connectivity(problem, [(0, 1)], value=5.0)
        problem.parameters["CONNECTIVITY_WEIGHT"] = 1.0
        assignment = np.array([0, 0, 1, 1])
        result = compute_zone_connectivity(problem, assignment)
        assert result == 0.0

    def test_weight_scales(self, base_problem):
        """CONNECTIVITY_WEIGHT scales the penalty."""
        problem = copy.deepcopy(base_problem)
        problem.connectivity = _make_connectivity(problem, [(0, 1)], value=5.0)
        problem.parameters["CONNECTIVITY_WEIGHT"] = 2.0
        assignment = np.array([1, 1, 0, 0])
        result = compute_zone_connectivity(problem, assignment)
        assert result == -10.0


class TestZoneObjectiveWithConnectivity:
    def test_objective_includes_connectivity(self, base_problem):
        """Objective with connectivity differs from without."""
        problem = copy.deepcopy(base_problem)
        problem.connectivity = _make_connectivity(problem, [(0, 1)], value=10.0)
        problem.parameters["CONNECTIVITY_WEIGHT"] = 1.0
        assignment = np.array([1, 1, 0, 0])

        obj_with = compute_zone_objective(problem, assignment, blm=0.0)

        problem_no_conn = copy.deepcopy(base_problem)
        problem_no_conn.connectivity = None
        obj_without = compute_zone_objective(problem_no_conn, assignment, blm=0.0)

        assert obj_with < obj_without


class TestZoneCacheConnectivity:
    def test_full_objective_matches(self, base_problem):
        """Cache full objective matches objective.py with connectivity."""
        problem = copy.deepcopy(base_problem)
        problem.connectivity = _make_connectivity(problem, [(0, 1), (1, 2)], value=3.0)
        problem.parameters["CONNECTIVITY_WEIGHT"] = 1.0

        cache = ZoneProblemCache.from_zone_problem(problem)
        assignment = np.array([1, 1, 2, 0])
        held = cache.compute_held_per_zone(assignment)
        cache_obj = cache.compute_full_zone_objective(assignment, held, blm=0.0)
        direct_obj = compute_zone_objective(problem, assignment, blm=0.0)
        np.testing.assert_allclose(cache_obj, direct_obj, atol=1e-10)

    def test_delta_matches_full_recompute(self, base_problem):
        """Delta computation matches full objective difference."""
        problem = copy.deepcopy(base_problem)
        problem.connectivity = _make_connectivity(
            problem, [(0, 1), (1, 2), (2, 3)], value=2.0
        )
        problem.parameters["CONNECTIVITY_WEIGHT"] = 1.0

        cache = ZoneProblemCache.from_zone_problem(problem)
        assignment = np.array([1, 1, 2, 2])
        held = cache.compute_held_per_zone(assignment)
        obj_before = cache.compute_full_zone_objective(assignment, held, blm=0.0)

        # Try changing PU 1 from zone 1 to zone 2
        idx, old_zone, new_zone = 1, 1, 2
        delta = cache.compute_delta_zone_objective(
            idx, old_zone, new_zone, assignment, held, blm=0.0
        )

        # Apply change and recompute
        new_assignment = assignment.copy()
        new_assignment[idx] = new_zone
        new_held = cache.compute_held_per_zone(new_assignment)
        obj_after = cache.compute_full_zone_objective(new_assignment, new_held, blm=0.0)

        np.testing.assert_allclose(delta, obj_after - obj_before, atol=1e-10)

    def test_no_connectivity_cache_works(self, base_problem):
        """Cache without connectivity still works normally."""
        problem = copy.deepcopy(base_problem)
        problem.connectivity = None
        cache = ZoneProblemCache.from_zone_problem(problem)
        assignment = np.array([1, 2, 1, 0])
        held = cache.compute_held_per_zone(assignment)
        obj = cache.compute_full_zone_objective(assignment, held, blm=0.0)
        assert obj >= 0


class TestZoneSolverConnectivity:
    def test_solver_with_connectivity(self, base_problem):
        """Solver runs without error when connectivity is present."""
        problem = copy.deepcopy(base_problem)
        problem.connectivity = _make_connectivity(
            problem, [(0, 1), (1, 2), (2, 3)], value=5.0
        )
        problem.parameters["CONNECTIVITY_WEIGHT"] = 1.0
        problem.parameters["NUMITNS"] = 2_000
        problem.parameters["NUMTEMP"] = 20

        solver = ZoneSASolver()
        config = SolverConfig(num_solutions=1, seed=42)
        solutions = solver.solve(problem, config)
        assert len(solutions) == 1
        assert solutions[0].zone_assignment is not None

    def test_connectivity_prefers_same_zone(self, base_problem):
        """Strong connectivity bonus encourages grouping PUs in same zone."""
        problem = copy.deepcopy(base_problem)
        # All PUs connected to each other with strong bonus
        pairs = [(i, j) for i in range(4) for j in range(i + 1, 4)]
        problem.connectivity = _make_connectivity(problem, pairs, value=100.0)
        problem.parameters["CONNECTIVITY_WEIGHT"] = 10.0
        problem.parameters["BLM"] = 0.0
        problem.features["spf"] = 0.0  # disable penalty to isolate connectivity
        problem.parameters["NUMITNS"] = 10_000
        problem.parameters["NUMTEMP"] = 100

        solver = ZoneSASolver()
        config = SolverConfig(num_solutions=5, seed=42)
        solutions = solver.solve(problem, config)

        # At least one solution should have all assigned PUs in the same zone
        found_grouped = False
        for sol in solutions:
            za = sol.zone_assignment
            assigned = za[za > 0]
            if len(assigned) > 0 and len(set(assigned)) == 1:
                found_grouped = True
                break
        assert found_grouped, "Strong connectivity should group PUs in same zone"

    def test_no_connectivity_solver_works(self, base_problem):
        """Solver works normally without connectivity data."""
        problem = copy.deepcopy(base_problem)
        problem.connectivity = None
        problem.parameters["NUMITNS"] = 1_000
        problem.parameters["NUMTEMP"] = 10

        solver = ZoneSASolver()
        config = SolverConfig(num_solutions=1, seed=42)
        solutions = solver.solve(problem, config)
        assert len(solutions) == 1
