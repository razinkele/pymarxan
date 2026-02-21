"""Integration tests for Phase 3 (zones and connectivity)."""
from pathlib import Path

import numpy as np

from pymarxan.connectivity.features import add_connectivity_features
from pymarxan.connectivity.metrics import (
    compute_betweenness_centrality,
    compute_in_degree,
)
from pymarxan.io.readers import load_project
from pymarxan.solvers.base import SolverConfig
from pymarxan.solvers.mip_solver import MIPSolver
from pymarxan.zones.objective import (
    compute_zone_cost,
    compute_zone_objective,
)
from pymarxan.zones.readers import load_zone_project
from pymarxan.zones.solver import ZoneSASolver

ZONE_DIR = Path(__file__).parent / "data" / "zones"
SIMPLE_DIR = Path(__file__).parent / "data" / "simple"


class TestZoneIntegration:
    def test_load_and_solve_zones(self):
        problem = load_zone_project(ZONE_DIR)
        problem.parameters["NUMITNS"] = 5_000
        problem.parameters["NUMTEMP"] = 50
        solver = ZoneSASolver()
        config = SolverConfig(num_solutions=3, seed=42)
        solutions = solver.solve(problem, config)
        assert len(solutions) == 3
        for sol in solutions:
            assert sol.zone_assignment is not None
            assert len(sol.zone_assignment) == 4

    def test_zone_objective_components(self):
        problem = load_zone_project(ZONE_DIR)
        assignment = np.array([1, 2, 1, 2])
        cost = compute_zone_cost(problem, assignment)
        assert cost > 0
        obj = compute_zone_objective(problem, assignment, blm=1.0)
        assert obj >= cost


class TestConnectivityIntegration:
    def test_metrics_to_features_with_solver(self):
        problem = load_project(SIMPLE_DIR)
        pu_ids = problem.planning_units["id"].tolist()
        n = len(pu_ids)

        # Create a simple connectivity matrix
        matrix = np.zeros((n, n))
        for i in range(n - 1):
            matrix[i, i + 1] = 0.5
            matrix[i + 1, i] = 0.3

        in_deg = compute_in_degree(matrix)
        bc = compute_betweenness_centrality(matrix)

        enhanced = add_connectivity_features(
            problem,
            metrics={"in_degree": in_deg, "betweenness": bc},
            targets={"in_degree": 0.5, "betweenness": 0.1},
        )

        assert enhanced.n_features == problem.n_features + 2

        solver = MIPSolver()
        config = SolverConfig(num_solutions=1)
        solutions = solver.solve(enhanced, config)
        assert len(solutions) == 1
        assert solutions[0].all_targets_met
