"""Tests for connectivity support in the MIP solver."""
import pandas as pd

from pymarxan.models.problem import ConservationProblem
from pymarxan.solvers.base import SolverConfig
from pymarxan.solvers.mip_solver import MIPSolver


def _make_problem(connectivity=None, params=None):
    """3 PUs, 1 feature. PU costs: 1=1, 2=10, 3=1. Target requires any one PU."""
    pu = pd.DataFrame({
        "id": [1, 2, 3],
        "cost": [1.0, 10.0, 1.0],
        "status": [0, 0, 0],
    })
    feat = pd.DataFrame({
        "id": [1],
        "name": ["f1"],
        "target": [1.0],
        "spf": [1.0],
    })
    puvspr = pd.DataFrame({
        "species": [1, 1, 1],
        "pu": [1, 2, 3],
        "amount": [5.0, 5.0, 5.0],
    })
    problem = ConservationProblem(
        planning_units=pu, features=feat, pu_vs_features=puvspr,
        connectivity=connectivity,
    )
    if params:
        problem.parameters.update(params)
    return problem


def _solve(problem):
    solver = MIPSolver()
    sols = solver.solve(problem, SolverConfig(num_solutions=1))
    assert len(sols) == 1
    return sols[0]


def _selected_ids(problem, sol):
    ids = problem.planning_units["id"].tolist()
    return {ids[i] for i in range(len(ids)) if sol.selected[i]}


class TestSymmetricConnectivity:
    def test_prefers_connected_pair(self):
        """With strong symmetric connectivity between PU 1 and PU 2,
        the solver should select both despite PU 2 being expensive."""
        conn = pd.DataFrame({
            "id1": [1, 2],
            "id2": [2, 3],
            "value": [100.0, 0.1],
        })
        problem = _make_problem(
            connectivity=conn,
            params={"CONNECTIVITY_WEIGHT": 1.0},
        )
        sol = _solve(problem)
        selected = _selected_ids(problem, sol)
        # Connectivity bonus for (1,2) = 100 outweighs PU 2 cost of 10
        assert 1 in selected and 2 in selected, (
            f"Expected PUs 1 & 2 selected for connectivity bonus, got {selected}"
        )


class TestAsymmetricConnectivity:
    def test_avoids_source_without_sink(self):
        """Asymmetric: edge 1->2 means selecting PU 1 without PU 2 is penalized.
        PU 3 is cheap and sufficient, so solver should prefer PU 3 alone."""
        conn = pd.DataFrame({
            "id1": [1],
            "id2": [2],
            "value": [50.0],
        })
        problem = _make_problem(
            connectivity=conn,
            params={"CONNECTIVITY_WEIGHT": 1.0, "ASYMMETRIC_CONNECTIVITY": 1},
        )
        sol = _solve(problem)
        selected = _selected_ids(problem, sol)
        # PU 1 alone costs 1 + penalty 50 = 51; PU 3 alone costs 1
        assert 1 not in selected or 2 in selected, (
            f"PU 1 selected without PU 2 despite asymmetric penalty, got {selected}"
        )


class TestZeroWeight:
    def test_zero_weight_no_effect(self):
        """CONNECTIVITY_WEIGHT=0 should ignore connectivity data entirely."""
        conn = pd.DataFrame({
            "id1": [1],
            "id2": [2],
            "value": [1000.0],
        })
        problem_with = _make_problem(
            connectivity=conn,
            params={"CONNECTIVITY_WEIGHT": 0.0},
        )
        problem_without = _make_problem()
        sol_with = _solve(problem_with)
        sol_without = _solve(problem_without)
        assert _selected_ids(problem_with, sol_with) == _selected_ids(
            problem_without, sol_without
        )


class TestNoConnectivity:
    def test_no_connectivity_data(self):
        """No connectivity DataFrame at all — solver should work normally."""
        problem = _make_problem(params={"CONNECTIVITY_WEIGHT": 1.0})
        sol = _solve(problem)
        assert sol.all_targets_met
        # Should pick cheapest PU (1 or 3, cost=1)
        selected = _selected_ids(problem, sol)
        assert selected & {1, 3}, f"Expected cheap PU selected, got {selected}"
