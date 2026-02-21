"""Tests for all 8 Marxan HEURTYPE scoring modes (0-7)."""
from __future__ import annotations

import pandas as pd
import pytest

from pymarxan.models.problem import ConservationProblem
from pymarxan.solvers.base import SolverConfig
from pymarxan.solvers.heuristic import HeuristicSolver


@pytest.fixture()
def simple_problem() -> ConservationProblem:
    """A small problem with 4 PUs and 2 features."""
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


@pytest.fixture()
def richness_problem() -> ConservationProblem:
    """Problem where PU 1 contributes to 3 features, PU 2 to 1 feature.

    PU 1: cost=100, contributes to features 1,2,3
    PU 2: cost=10, contributes to feature 1 only
    PU 3: cost=10, contributes to feature 2 only
    PU 4: cost=10, contributes to feature 3 only

    Richness (heurtype=0) should pick PU 1 first (covers 3 features).
    """
    pu = pd.DataFrame({
        "id": [1, 2, 3, 4],
        "cost": [100.0, 10.0, 10.0, 10.0],
        "status": [0, 0, 0, 0],
    })
    feat = pd.DataFrame({
        "id": [1, 2, 3],
        "name": ["f1", "f2", "f3"],
        "target": [5.0, 5.0, 5.0],
        "spf": [1.0, 1.0, 1.0],
    })
    puvspr = pd.DataFrame({
        "species": [1, 2, 3, 1, 2, 3],
        "pu": [1, 1, 1, 2, 3, 4],
        "amount": [6.0, 6.0, 6.0, 6.0, 6.0, 6.0],
    })
    return ConservationProblem(
        planning_units=pu, features=feat,
        pu_vs_features=puvspr, boundary=None,
        parameters={"BLM": 0.0},
    )


@pytest.fixture()
def cheapest_problem() -> ConservationProblem:
    """Problem where all PUs contribute equally but have different costs.

    PU 1: cost=1, feature 1 amount=10
    PU 2: cost=100, feature 1 amount=10

    Greedy cheapest (heurtype=1) should prefer PU 1.
    """
    pu = pd.DataFrame({
        "id": [1, 2],
        "cost": [1.0, 100.0],
        "status": [0, 0],
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
        "amount": [10.0, 10.0],
    })
    return ConservationProblem(
        planning_units=pu, features=feat,
        pu_vs_features=puvspr, boundary=None,
        parameters={"BLM": 0.0},
    )


# ---------------------------------------------------------------
# Parametrized tests: valid solution for each heurtype
# ---------------------------------------------------------------


@pytest.mark.parametrize("heurtype", [0, 1, 2, 3, 4, 5, 6, 7])
def test_heurtype_returns_valid_solution(simple_problem, heurtype):
    """Each heurtype 0-7 should return a valid solution."""
    solver = HeuristicSolver(heurtype=heurtype)
    sols = solver.solve(simple_problem, SolverConfig(num_solutions=1, seed=42))
    assert len(sols) == 1
    assert sols[0].selected.dtype == bool
    assert sols[0].cost >= 0.0


@pytest.mark.parametrize("heurtype", [0, 1, 2, 3, 4, 5, 6, 7])
def test_heurtype_selects_some_pus(simple_problem, heurtype):
    """Each heurtype should select at least some planning units."""
    solver = HeuristicSolver(heurtype=heurtype)
    sols = solver.solve(simple_problem, SolverConfig(num_solutions=1, seed=42))
    assert sols[0].n_selected > 0


# ---------------------------------------------------------------
# Specific heurtype behavior tests
# ---------------------------------------------------------------


def test_heurtype_0_richness_prefers_multi_feature_pus(richness_problem):
    """Richness mode should prefer PUs contributing to more features.

    PU 1 contributes to 3 features and alone meets all targets,
    so richness should select it (despite being expensive).
    """
    solver = HeuristicSolver(heurtype=0)
    sols = solver.solve(richness_problem, SolverConfig(num_solutions=1, seed=42))
    selected_ids = set(
        richness_problem.planning_units.loc[sols[0].selected, "id"]
    )
    # PU 1 covers all 3 features; richness picks it first
    assert 1 in selected_ids


def test_heurtype_1_greedy_cheapest_prefers_low_cost(cheapest_problem):
    """Greedy cheapest mode should prefer PUs with lower cost.

    PU 1 costs 1 and meets the target. PU 2 costs 100.
    Heurtype 1 should select PU 1, not PU 2.
    """
    solver = HeuristicSolver(heurtype=1)
    sols = solver.solve(cheapest_problem, SolverConfig(num_solutions=1, seed=42))
    selected_ids = set(
        cheapest_problem.planning_units.loc[sols[0].selected, "id"]
    )
    assert 1 in selected_ids
    # PU 2 not needed since PU 1 meets the target
    assert 2 not in selected_ids


# ---------------------------------------------------------------
# Default heurtype
# ---------------------------------------------------------------


def test_default_heurtype_is_2():
    """Default heurtype should be 2 (Max Rarity), matching Marxan default."""
    solver = HeuristicSolver()
    assert solver.heurtype == 2


# ---------------------------------------------------------------
# Reading HEURTYPE from problem.parameters
# ---------------------------------------------------------------


def test_heurtype_from_problem_parameters(simple_problem):
    """HEURTYPE in problem.parameters should override constructor default."""
    simple_problem.parameters["HEURTYPE"] = 0
    solver = HeuristicSolver()  # default heurtype=2
    sols = solver.solve(simple_problem, SolverConfig(num_solutions=1, seed=42))
    # The solver should have used heurtype=0 due to problem.parameters
    assert sols[0].metadata.get("heurtype") == 0


def test_heurtype_from_problem_parameters_explicit_override(simple_problem):
    """Explicit constructor heurtype should be overridden by problem.parameters."""
    simple_problem.parameters["HEURTYPE"] = 5
    solver = HeuristicSolver(heurtype=3)
    sols = solver.solve(simple_problem, SolverConfig(num_solutions=1, seed=42))
    assert sols[0].metadata.get("heurtype") == 5


# ---------------------------------------------------------------
# Invalid heurtype
# ---------------------------------------------------------------


def test_invalid_heurtype_raises_valueerror():
    """heurtype outside 0-7 should raise ValueError."""
    with pytest.raises(ValueError, match="heurtype"):
        HeuristicSolver(heurtype=-1)


def test_invalid_heurtype_8_raises_valueerror():
    """heurtype=8 should raise ValueError."""
    with pytest.raises(ValueError, match="heurtype"):
        HeuristicSolver(heurtype=8)


def test_invalid_heurtype_string_raises():
    """Non-integer heurtype should raise an error."""
    with pytest.raises((ValueError, TypeError)):
        HeuristicSolver(heurtype="abc")  # type: ignore[arg-type]
