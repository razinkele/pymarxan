"""Tests for Jung et al. (2021) rank importance (Phase 22).

Sequential-removal ranking: start with a solution, repeatedly remove the
PU whose removal least increases the objective, record the order. PUs
removed later are more important — they are harder to replace.

Reference: Jung et al. (2021) *Methods in Ecology and Evolution* 12(5):
869-877. https://doi.org/10.1111/2041-210X.13578
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from pymarxan.analysis.rank_importance import compute_rank_importance
from pymarxan.models.problem import ConservationProblem
from pymarxan.solvers.base import Solution


def _toy_problem():
    """Target is set so PU 1's contribution is irreplaceable: total
    available is 13, target is 11, slack is 2 — removing PU 1 (which
    supplies 10) creates a shortfall of 8 that the SPF penalty makes
    expensive, while removing PU 2/3/4 (each supplying 1) leaves a
    surplus of 1 → no shortfall."""
    pu = pd.DataFrame({
        "id": [1, 2, 3, 4],
        "cost": [1.0, 1.0, 1.0, 1.0],
        "status": [0, 0, 0, 0],
    })
    features = pd.DataFrame({
        "id": [1], "name": ["a"], "target": [11.0], "spf": [10.0],
    })
    puvspr = pd.DataFrame({
        "species": [1, 1, 1, 1],
        "pu":      [1, 2, 3, 4],
        "amount":  [10.0, 1.0, 1.0, 1.0],
    })
    return ConservationProblem(
        planning_units=pu, features=features, pu_vs_features=puvspr,
    )


def _solution_of(selected: list[bool]) -> Solution:
    """Minimal Solution stub — rank_importance only reads .selected."""
    return Solution(
        selected=np.array(selected, dtype=bool),
        cost=0.0,
        boundary=0.0,
        objective=0.0,
        targets_met={1: True},
    )


def test_rank_returns_dict_with_one_entry_per_selected_pu():
    p = _toy_problem()
    sol = _solution_of([True, True, True, True])
    ranks = compute_rank_importance(p, sol)
    assert set(ranks.keys()) == {1, 2, 3, 4}


def test_rank_unselected_pus_score_zero():
    """PUs not in the reserve aren't ranked — they get score 0."""
    p = _toy_problem()
    sol = _solution_of([True, True, False, False])
    ranks = compute_rank_importance(p, sol)
    assert ranks[3] == 0.0
    assert ranks[4] == 0.0
    # Selected PUs get integer ranks 1..n (higher = more important).
    assert ranks[1] > 0
    assert ranks[2] > 0


def test_rank_richest_pu_gets_highest_score():
    """PU 1 supplies the bulk of the target; it should be removed last
    (highest rank). Removing PU 1 first would leave us with 3 units
    spread across PU 2-4 (3 total). The target is 3 — exactly met.
    Removing PU 2 first leaves 12 units — still met. So PU 1 IS
    removable but only after lower-amount PUs are gone."""
    p = _toy_problem()
    sol = _solution_of([True, True, True, True])
    ranks = compute_rank_importance(p, sol)
    # PU 1 has the most amount; it should be last out → rank == 4.
    assert ranks[1] == max(ranks.values())
    # PU 2, 3, 4 are interchangeable; their ranks tie at the bottom of
    # the selected set.


def test_rank_returns_floats():
    p = _toy_problem()
    sol = _solution_of([True, True, True, True])
    ranks = compute_rank_importance(p, sol)
    assert all(isinstance(v, float) for v in ranks.values())


def test_rank_handles_empty_selection():
    """No PUs selected → all-zero ranks (graceful)."""
    p = _toy_problem()
    sol = _solution_of([False, False, False, False])
    ranks = compute_rank_importance(p, sol)
    assert all(v == 0.0 for v in ranks.values())
