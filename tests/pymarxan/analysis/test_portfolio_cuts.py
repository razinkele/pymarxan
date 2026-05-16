"""Phase 25: solver-agnostic no-good-cut portfolio generation.

Generate K diverse high-quality solutions by iteratively solving and
adding "no-good cut" constraints (problem rejects the previous
selection vector outright). Solver-agnostic — works on any
``MIPSolver``-compatible backend.
"""
from __future__ import annotations

import pandas as pd
import pytest

from pymarxan.analysis.portfolio_cuts import generate_portfolio_cuts
from pymarxan.models.problem import ConservationProblem
from pymarxan.solvers.base import SolverConfig
from pymarxan.solvers.mip_solver import MIPSolver


def _diverse_problem():
    """6 PUs, 1 feature; multiple combinations satisfy the target so the
    portfolio has room to generate distinct solutions."""
    pu = pd.DataFrame({
        "id": [1, 2, 3, 4, 5, 6],
        "cost": [1.0] * 6,
        "status": [0] * 6,
    })
    features = pd.DataFrame({
        "id": [1], "name": ["a"], "target": [2.0], "spf": [1.0],
    })
    puvspr = pd.DataFrame({
        "species": [1] * 6,
        "pu":      [1, 2, 3, 4, 5, 6],
        "amount":  [1.0] * 6,
    })
    return ConservationProblem(
        planning_units=pu, features=features, pu_vs_features=puvspr,
    )


def test_portfolio_cuts_returns_k_solutions():
    p = _diverse_problem()
    sols = generate_portfolio_cuts(
        p, solver=MIPSolver(), k=3, config=SolverConfig(num_solutions=1),
    )
    assert len(sols) == 3


def test_portfolio_cuts_all_distinct_selections():
    """Every returned solution has a distinct selection vector — that's
    the entire point of no-good cuts."""
    p = _diverse_problem()
    sols = generate_portfolio_cuts(
        p, solver=MIPSolver(), k=4, config=SolverConfig(num_solutions=1),
    )
    seen = set()
    for s in sols:
        key = tuple(bool(b) for b in s.selected)
        assert key not in seen
        seen.add(key)


def test_portfolio_cuts_all_solutions_feasible():
    """Each generated solution still meets the original target."""
    p = _diverse_problem()
    sols = generate_portfolio_cuts(
        p, solver=MIPSolver(), k=3, config=SolverConfig(num_solutions=1),
    )
    for s in sols:
        assert all(s.targets_met.values())


def test_portfolio_cuts_stops_when_no_more_feasible():
    """If K is larger than the number of distinct feasible solutions,
    return what we found rather than crashing."""
    # Trivial problem: target 1, 2 PUs each supplying 1. There are exactly
    # 3 feasible binary selections (PU1, PU2, PU1+PU2). Ask for 10 → get
    # whatever the MIP finds before infeasibility, in this case 3 (or
    # however many distinct ones the no-good cuts allow).
    pu = pd.DataFrame({
        "id": [1, 2], "cost": [1.0, 1.0], "status": [0, 0],
    })
    features = pd.DataFrame({
        "id": [1], "name": ["a"], "target": [1.0], "spf": [1.0],
    })
    puvspr = pd.DataFrame({
        "species": [1, 1], "pu": [1, 2], "amount": [1.0, 1.0],
    })
    p = ConservationProblem(
        planning_units=pu, features=features, pu_vs_features=puvspr,
    )
    sols = generate_portfolio_cuts(
        p, solver=MIPSolver(), k=10, config=SolverConfig(num_solutions=1),
    )
    # Should be at least 1, at most 3.
    assert 1 <= len(sols) <= 3


def test_portfolio_cuts_objectives_non_decreasing():
    """The MIP gives the OPTIMUM first; each no-good cut forces a
    sub-optimal solution next. Therefore objectives are weakly
    non-decreasing in iteration order."""
    p = _diverse_problem()
    sols = generate_portfolio_cuts(
        p, solver=MIPSolver(), k=3, config=SolverConfig(num_solutions=1),
    )
    objs = [s.objective for s in sols]
    for prev, cur in zip(objs, objs[1:]):
        assert cur >= prev - 1e-9, (
            f"portfolio cuts must produce non-decreasing objectives; "
            f"got {objs}"
        )


def test_portfolio_cuts_invalid_k_rejected():
    """k must be a positive int."""
    p = _diverse_problem()
    with pytest.raises(ValueError, match="k"):
        generate_portfolio_cuts(
            p, solver=MIPSolver(), k=0, config=SolverConfig(num_solutions=1),
        )
    with pytest.raises(ValueError, match="k"):
        generate_portfolio_cuts(
            p, solver=MIPSolver(), k=-1, config=SolverConfig(num_solutions=1),
        )


def test_portfolio_cuts_metadata_records_iteration():
    """Each solution's metadata records which portfolio iteration it
    came from, so downstream consumers know the rank."""
    p = _diverse_problem()
    sols = generate_portfolio_cuts(
        p, solver=MIPSolver(), k=3, config=SolverConfig(num_solutions=1),
    )
    for i, s in enumerate(sols):
        assert s.metadata.get("portfolio_iteration") == i
