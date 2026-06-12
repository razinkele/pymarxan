"""Tests for area-based (30x30 / GBF Target 3) representation reporting.

Unlike gap analysis (which scores existing protection, status==2, against
each feature's optimisation target), this reports how much of each
feature a *solution* represents, against a uniform policy threshold such
as the Kunming-Montreal "protect 30%" goal.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pymarxan.analysis.representation import (
    RepresentationResult,
    compute_representation,
)
from pymarxan.models.problem import ConservationProblem
from pymarxan.solvers.base import Solution


def _problem() -> ConservationProblem:
    planning_units = pd.DataFrame(
        {"id": [1, 2, 3, 4], "cost": [1.0, 1.0, 1.0, 1.0], "status": [0, 0, 0, 0]}
    )
    features = pd.DataFrame(
        {
            "id": [1, 2],
            "name": ["sp_a", "sp_b"],
            "target": [1.0, 1.0],
            "spf": [1.0, 1.0],
        }
    )
    # sp_a: 10 on each of PU1-4 (total 40); sp_b: 5 on PU1, PU2 (total 10)
    pu_vs_features = pd.DataFrame(
        {
            "species": [1, 1, 1, 1, 2, 2],
            "pu": [1, 2, 3, 4, 1, 2],
            "amount": [10.0, 10.0, 10.0, 10.0, 5.0, 5.0],
        }
    )
    return ConservationProblem(planning_units, features, pu_vs_features)


def _solution(selected: list[bool]) -> Solution:
    return Solution(
        selected=np.array(selected, dtype=bool),
        cost=0.0,
        boundary=0.0,
        objective=0.0,
        targets_met={},
    )


def test_total_amount_uses_all_units():
    problem = _problem()
    sol = _solution([True, False, False, False])
    result = compute_representation(problem, sol)
    assert result.total_amount == {1: 40.0, 2: 10.0}


def test_represented_amount_uses_only_selected():
    problem = _problem()
    sol = _solution([True, True, False, False])  # PU1, PU2
    result = compute_representation(problem, sol)
    # sp_a: 10+10 = 20 ; sp_b: 5+5 = 10
    assert result.represented_amount == {1: 20.0, 2: 10.0}
    assert result.pct_represented[1] == pytest.approx(50.0)
    assert result.pct_represented[2] == pytest.approx(100.0)


def test_threshold_pass_and_fail():
    problem = _problem()
    sol = _solution([True, False, False, False])  # only PU1
    result = compute_representation(problem, sol, threshold=0.30)
    # sp_a: 10/40 = 25% < 30% → fail ; sp_b: 5/10 = 50% → meets
    assert result.meets_threshold == {1: False, 2: True}
    assert result.n_features_meeting == 1
    assert result.fraction_features_meeting == pytest.approx(0.5)


def test_threshold_is_inclusive_at_exactly_30_percent():
    problem = _problem()
    # Need sp_a at exactly 30%: 12/40. Not reachable with 10-unit PUs, so
    # use the default to confirm >= semantics on sp_b (50% >= 30%).
    sol = _solution([True, True, False, False])
    result = compute_representation(problem, sol, threshold=0.50)
    # sp_a 20/40 = 50% which is >= 0.50 → meets (inclusive)
    assert result.meets_threshold[1] is True


def test_all_features_meeting_when_everything_selected():
    problem = _problem()
    sol = _solution([True, True, True, True])
    result = compute_representation(problem, sol, threshold=0.30)
    assert result.n_features_meeting == 2
    assert result.fraction_features_meeting == pytest.approx(1.0)


def test_custom_threshold():
    problem = _problem()
    sol = _solution([True, True, False, False])  # sp_a 50%, sp_b 100%
    result = compute_representation(problem, sol, threshold=0.60)
    # sp_a 50% < 60% → fail ; sp_b 100% → meets
    assert result.meets_threshold == {1: False, 2: True}
    assert result.threshold == pytest.approx(0.60)


def test_invalid_threshold_raises():
    problem = _problem()
    sol = _solution([True, True, True, True])
    with pytest.raises(ValueError, match="threshold"):
        compute_representation(problem, sol, threshold=1.5)


def test_to_dataframe_shape():
    problem = _problem()
    sol = _solution([True, True, False, False])
    df = compute_representation(problem, sol).to_dataframe()
    assert set(df.columns) == {
        "feature_id",
        "feature_name",
        "total_amount",
        "represented_amount",
        "pct_represented",
        "meets_threshold",
    }
    assert len(df) == 2


def test_result_is_dataclass():
    problem = _problem()
    sol = _solution([True, True, True, True])
    result = compute_representation(problem, sol)
    assert isinstance(result, RepresentationResult)
