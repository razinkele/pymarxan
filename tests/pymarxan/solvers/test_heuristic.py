"""Tests for greedy heuristic solver."""
from __future__ import annotations

import pandas as pd
import pytest

from pymarxan.models.problem import ConservationProblem
from pymarxan.solvers.base import SolverConfig
from pymarxan.solvers.heuristic import HeuristicSolver


@pytest.fixture()
def simple_problem() -> ConservationProblem:
    pu = pd.DataFrame({
        "id": [1, 2, 3, 4],
        "cost": [10.0, 20.0, 15.0, 25.0],
        "status": [0, 0, 0, 0],
    })
    feat = pd.DataFrame({
        "id": [1, 2],
        "name": ["f1", "f2"],
        "target": [5.0, 4.0],
        "spf": [1.0, 1.0],
    })
    puvspr = pd.DataFrame({
        "species": [1, 1, 2, 2],
        "pu": [1, 2, 3, 4],
        "amount": [3.0, 4.0, 5.0, 3.0],
    })
    bnd = pd.DataFrame({"id1": [1, 2], "id2": [2, 3], "boundary": [1.0, 1.0]})
    return ConservationProblem(
        planning_units=pu, features=feat,
        pu_vs_features=puvspr, boundary=bnd,
        parameters={"BLM": 0.0},
    )


def test_heuristic_solver_returns_solution(simple_problem):
    solver = HeuristicSolver()
    sols = solver.solve(simple_problem)
    assert len(sols) == 1
    assert sols[0].selected.dtype == bool


def test_heuristic_meets_targets(simple_problem):
    solver = HeuristicSolver()
    sols = solver.solve(simple_problem)
    assert sols[0].all_targets_met


def test_heuristic_prefers_cheap_units(simple_problem):
    solver = HeuristicSolver()
    sols = solver.solve(simple_problem)
    selected_ids = set(simple_problem.planning_units.loc[sols[0].selected, "id"])
    assert 4 not in selected_ids


def test_heuristic_name():
    assert HeuristicSolver().name() == "greedy"


def test_heuristic_supports_zones():
    assert HeuristicSolver().supports_zones() is False


def test_heuristic_locked_in(simple_problem):
    simple_problem.planning_units.loc[
        simple_problem.planning_units["id"] == 4, "status"
    ] = 2
    solver = HeuristicSolver()
    sols = solver.solve(simple_problem)
    idx = simple_problem.planning_units.index[
        simple_problem.planning_units["id"] == 4
    ][0]
    assert sols[0].selected[idx]


def test_heuristic_locked_out(simple_problem):
    simple_problem.planning_units.loc[
        simple_problem.planning_units["id"] == 1, "status"
    ] = 3
    solver = HeuristicSolver()
    sols = solver.solve(simple_problem)
    idx = simple_problem.planning_units.index[
        simple_problem.planning_units["id"] == 1
    ][0]
    assert not sols[0].selected[idx]


def test_heuristic_multiple_solutions(simple_problem):
    solver = HeuristicSolver()
    config = SolverConfig(num_solutions=3, seed=42)
    sols = solver.solve(simple_problem, config)
    assert len(sols) == 3
