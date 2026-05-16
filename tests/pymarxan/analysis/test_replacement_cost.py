"""Tests for replacement-cost importance (Phase 22).

For each PU in a baseline solution, lock it out, re-solve, and compute
``optimum_locked_out - optimum_baseline``. PUs whose absence drives the
optimum up by a lot are critical (high replacement cost); PUs whose
absence costs nothing are interchangeable (low replacement cost).

Reference: Ferrier et al. (2000); Cabeza & Moilanen (2006).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pymarxan.analysis.replacement_cost import compute_replacement_cost
from pymarxan.models.problem import ConservationProblem


def _toy_problem():
    """4 PUs × 1 feature; target tight enough that one PU's removal
    forces a strictly more-expensive solution."""
    pu = pd.DataFrame({
        "id": [1, 2, 3, 4],
        "cost": [1.0, 2.0, 2.0, 2.0],
        "status": [0, 0, 0, 0],
    })
    features = pd.DataFrame({
        "id": [1], "name": ["a"], "target": [3.0], "spf": [1.0],
    })
    # PU 1 (cheap, contributes 3) is single-handedly sufficient; the
    # optimum is just {PU 1} at cost 1. Locking out PU 1 forces selection
    # of {PU 2, PU 3, PU 4} at cost 6 (or any 3 of them). Cost gap = 5.
    puvspr = pd.DataFrame({
        "species": [1, 1, 1, 1],
        "pu":      [1, 2, 3, 4],
        "amount":  [3.0, 1.0, 1.0, 1.0],
    })
    return ConservationProblem(
        planning_units=pu, features=features, pu_vs_features=puvspr,
    )


def test_replacement_returns_dict_with_one_entry_per_pu():
    p = _toy_problem()
    scores = compute_replacement_cost(p)
    assert set(scores.keys()) == {1, 2, 3, 4}


def test_replacement_higher_for_irreplaceable_pu():
    """PU 1 is the cheapest way to meet the target — locking it out
    forces a far more expensive replacement. Its replacement cost
    should exceed that of any other PU."""
    p = _toy_problem()
    scores = compute_replacement_cost(p)
    # PU 1's replacement cost: optimum without PU 1 - optimum with PU 1
    # Without lock-out: cost 1. With PU 1 locked out: need 3 from
    # {PU 2, 3, 4}, cheapest is {2, 3, 4} totalling 6. Δ = 5.
    assert scores[1] == pytest.approx(5.0, abs=1e-6)
    # The other PUs aren't in the optimal solution (or their absence is
    # easily replaceable), so they get 0.
    assert scores[2] == 0.0
    assert scores[3] == 0.0
    assert scores[4] == 0.0


def test_replacement_only_evaluates_pus_in_baseline():
    """By default, only PUs selected in the baseline solution get their
    replacement cost computed — others are 0. This keeps the runtime
    bounded by ``n_selected`` MIP re-solves, not ``n_pu``."""
    p = _toy_problem()
    scores = compute_replacement_cost(p)
    # Of 4 PUs, only PU 1 is in the optimum → only it gets a non-zero
    # score even though all four PUs are unlocked.
    assert sum(1 for v in scores.values() if v > 0) == 1


def test_replacement_returns_floats():
    p = _toy_problem()
    scores = compute_replacement_cost(p)
    assert all(isinstance(v, float) for v in scores.values())
    assert all(np.isfinite(v) for v in scores.values())


def test_replacement_handles_infeasible_lockout():
    """If locking out a PU makes the problem infeasible (no replacement
    exists), the replacement cost is reported as ``+inf`` — never crash."""
    pu = pd.DataFrame({
        "id": [1, 2],
        "cost": [1.0, 1.0],
        "status": [0, 0],
    })
    features = pd.DataFrame({
        "id": [1], "name": ["a"], "target": [10.0], "spf": [1.0],
    })
    # PU 1 supplies the whole target; locking it out leaves only PU 2
    # which can't meet the target → MIP infeasible.
    puvspr = pd.DataFrame({
        "species": [1, 1], "pu": [1, 2], "amount": [10.0, 1.0],
    })
    p = ConservationProblem(
        planning_units=pu, features=features, pu_vs_features=puvspr,
    )
    scores = compute_replacement_cost(p)
    assert scores[1] == float("inf")
