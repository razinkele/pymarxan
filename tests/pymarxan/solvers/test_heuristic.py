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


def test_heuristic_applies_misslevel(simple_problem):
    """Greedy solver should use MISSLEVEL to relax targets."""
    # With MISSLEVEL=0.0, targets become target*0.0=0 — all targets already met,
    # so no PUs should be selected by the greedy loop (0 selected).
    simple_problem.parameters["MISSLEVEL"] = 0.0
    solver = HeuristicSolver(heurtype=0)
    solutions = solver.solve(simple_problem, SolverConfig(num_solutions=1, seed=42))
    sol_relaxed = solutions[0]

    # With MISSLEVEL=0, effective targets are 0, so no PUs needed
    assert sol_relaxed.n_selected == 0, (
        f"Expected 0 PUs selected with MISSLEVEL=0.0, got {sol_relaxed.n_selected}"
    )


def test_heuristic_status1_starts_selected():
    """PUs with status=1 should start selected and contribute to targets.

    We create a problem where the expensive PU (status=1) fully covers
    the target. The cheap PU also covers it. Without status=1 handling,
    the greedy solver (heurtype=1 = cheapest first) picks the cheap PU
    and never selects the expensive one. With proper status=1, the
    expensive PU starts selected, satisfying the target immediately,
    and the cheap PU is unnecessary.
    """
    pu = pd.DataFrame({
        "id": [1, 2],
        "cost": [1000.0, 1.0],
        "status": [1, 0],  # PU 1 is expensive but initial-include
    })
    feat = pd.DataFrame({
        "id": [1],
        "name": ["f1"],
        "target": [5.0],
        "spf": [1.0],
    })
    puvspr = pd.DataFrame({
        "species": [1, 1],
        "pu": [1, 2],
        "amount": [10.0, 10.0],  # Both cover the target fully
    })
    problem = ConservationProblem(
        planning_units=pu, features=feat,
        pu_vs_features=puvspr, boundary=None,
        parameters={"BLM": 0.0},
    )
    # heurtype=1 (cheapest first) — without status=1 handling, only PU 2 is selected
    solver = HeuristicSolver(heurtype=1)
    solutions = solver.solve(problem, SolverConfig(num_solutions=1, seed=42))
    sol = solutions[0]
    # PU 1 (status=1) must be selected even though it's expensive
    assert sol.selected[0], "PU with status=1 should be selected"


def test_heuristic_includes_self_boundary():
    """Heuristic solver must include self-boundary in cost calculation."""
    import numpy as np

    pu = pd.DataFrame({"id": [1, 2, 3], "cost": [1.0, 1.0, 1.0], "status": [0, 0, 0]})
    features = pd.DataFrame({"id": [1], "name": ["f1"], "target": [1.0], "spf": [1.0]})
    puvspr = pd.DataFrame({"species": [1, 1, 1], "pu": [1, 2, 3], "amount": [1.0, 1.0, 1.0]})
    boundary = pd.DataFrame({
        "id1": [1, 2, 3, 1, 2],
        "id2": [1, 2, 3, 2, 3],
        "boundary": [10.0, 10.0, 10.0, 1.0, 1.0],
    })
    problem = ConservationProblem(
        planning_units=pu, features=features, pu_vs_features=puvspr,
        boundary=boundary, parameters={"BLM": 1.0},
    )
    solver = HeuristicSolver(heurtype=0)
    sols = solver.solve(problem, SolverConfig(num_solutions=1, seed=42))
    sol = sols[0]
    assert sol.boundary >= 10.0, f"Expected self-boundary >= 10, got {sol.boundary}"


def test_heuristic_solution_has_penalty(simple_problem):
    """Greedy solution should have a correctly computed penalty field."""
    # Lock out all PUs so targets can't be met — penalty must be > 0
    simple_problem.planning_units["status"] = 3
    solver = HeuristicSolver(heurtype=0)
    solutions = solver.solve(simple_problem, SolverConfig(num_solutions=1, seed=42))
    sol = solutions[0]
    assert hasattr(sol, "penalty")
    assert isinstance(sol.penalty, float)
    # With all PUs locked out, no targets met, so penalty should be positive
    assert sol.penalty > 0.0, f"Expected positive penalty, got {sol.penalty}"
