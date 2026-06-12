"""Tests for distributional-equity analysis of a conservation solution.

Equity asks: are the benefits (or burdens) of a reserve shared evenly
across social/spatial groups, or concentrated? See Gopalakrishna et al.
(2024), PNAS, https://doi.org/10.1073/pnas.2402970121.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pymarxan.analysis.equity import EquityResult, compute_equity
from pymarxan.models.problem import ConservationProblem
from pymarxan.solvers.base import Solution


def _problem() -> ConservationProblem:
    planning_units = pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5, 6],
            "cost": [10.0, 15.0, 20.0, 12.0, 18.0, 8.0],
            "status": [0, 0, 0, 0, 0, 0],
        }
    )
    features = pd.DataFrame(
        {"id": [1], "name": ["sp"], "target": [1.0], "spf": [1.0]}
    )
    pu_vs_features = pd.DataFrame(
        {"species": [1, 1], "pu": [1, 2], "amount": [1.0, 1.0]}
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


# Groups: A = {1, 3, 5}, B = {2, 4, 6}
GROUPS = {1: "A", 2: "B", 3: "A", 4: "B", 5: "A", 6: "B"}


def test_equal_distribution_has_zero_gini():
    # Select one A (PU1, cost 10) and one B (PU6, cost... 8) — make equal:
    # select PU1 (A, 10) and PU4 (B, 12) is not equal; use count value.
    problem = _problem()
    sol = _solution([True, True, False, False, False, False])  # PU1=A, PU2=B
    result = compute_equity(problem, sol, GROUPS, value="count")
    # One PU per group → perfectly equal counts → Gini 0.
    assert result.group_values == {"A": 1.0, "B": 1.0}
    assert result.gini == pytest.approx(0.0)
    assert result.group_shares["A"] == pytest.approx(0.5)
    assert result.group_shares["B"] == pytest.approx(0.5)


def test_fully_unequal_distribution_two_groups():
    problem = _problem()
    # Select only A-group PUs → all value to A, none to B.
    sol = _solution([True, False, True, False, True, False])
    result = compute_equity(problem, sol, GROUPS, value="count")
    assert result.group_values == {"A": 3.0, "B": 0.0}
    # Gini for [3, 0] over two groups = (n-1)/n = 0.5.
    assert result.gini == pytest.approx(0.5)
    assert result.max_share == pytest.approx(1.0)
    assert result.min_share == pytest.approx(0.0)


def test_cost_value_aggregates_selected_pu_costs_per_group():
    problem = _problem()
    sol = _solution([True, True, True, False, False, False])  # PU1,2,3
    result = compute_equity(problem, sol, GROUPS, value="cost")
    # A: PU1(10) + PU3(20) = 30 ; B: PU2(15) = 15
    assert result.group_values == {"A": 30.0, "B": 15.0}
    assert result.group_shares["A"] == pytest.approx(30.0 / 45.0)
    # Gini of [30, 15] = 30 / (2*2*45) = 0.16667.
    assert result.gini == pytest.approx(30.0 / 180.0)


def test_custom_value_mapping():
    problem = _problem()
    sol = _solution([True, True, False, False, False, False])
    values = {1: 5.0, 2: 5.0}
    result = compute_equity(problem, sol, GROUPS, value=values)
    assert result.group_values == {"A": 5.0, "B": 5.0}
    assert result.gini == pytest.approx(0.0)


def test_only_selected_units_count():
    problem = _problem()
    # Nothing selected → all group values zero, shares zero, gini zero.
    sol = _solution([False] * 6)
    result = compute_equity(problem, sol, GROUPS, value="cost")
    assert result.group_values == {"A": 0.0, "B": 0.0}
    assert result.gini == pytest.approx(0.0)
    assert result.group_shares == {"A": 0.0, "B": 0.0}


def test_three_groups_single_group_gets_all():
    problem = _problem()
    groups = {1: "A", 2: "B", 3: "C", 4: "A", 5: "B", 6: "C"}
    sol = _solution([True, False, False, False, False, False])  # only PU1=A
    result = compute_equity(problem, sol, groups, value="count")
    assert result.group_values == {"A": 1.0, "B": 0.0, "C": 0.0}
    # Gini for [1,0,0] over three groups = (n-1)/n = 0.6667.
    assert result.gini == pytest.approx(2.0 / 3.0)


def test_unknown_value_keyword_raises():
    problem = _problem()
    sol = _solution([True] * 6)
    with pytest.raises(ValueError, match="value"):
        compute_equity(problem, sol, GROUPS, value="bogus")


def test_to_dataframe_shape():
    problem = _problem()
    sol = _solution([True, True, True, False, False, False])
    result = compute_equity(problem, sol, GROUPS, value="cost")
    df = result.to_dataframe()
    assert set(df.columns) == {"group", "value", "share"}
    assert len(df) == 2
    assert set(df["group"]) == {"A", "B"}


def test_result_is_dataclass_instance():
    problem = _problem()
    sol = _solution([True, True, False, False, False, False])
    result = compute_equity(problem, sol, GROUPS)
    assert isinstance(result, EquityResult)
