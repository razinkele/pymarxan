"""Costello–Polasky informed-myopic dynamic reserve scheduling."""
from __future__ import annotations

import numpy as np
import pytest

from pymarxan.temporal.dynamic import dynamic_reserve_greedy


def test_informed_beats_naive_on_value_at_risk():
    """The Costello–Polasky insight: protect by value × loss-risk, not value
    alone. A valuable-but-safe site vs a slightly-less-valuable-but-at-risk
    site, budget for one, one period — informed protects the at-risk site and
    retains more expected value than naive (protect-highest-value)."""
    values = np.array([10.0, 8.0])
    loss_prob = np.array([0.01, 0.9])  # site A safe, site B at risk
    informed = dynamic_reserve_greedy(values, loss_prob, budgets=[1.0])
    naive = dynamic_reserve_greedy(values, loss_prob, budgets=[1.0], prioritize="value")
    assert informed.protected == {1}          # protect the at-risk site B
    assert naive.protected == {0}             # protect the high-value site A
    assert informed.expected_value > naive.expected_value


def test_budget_protects_everything_retains_full_value():
    values = np.array([5.0, 3.0, 7.0])
    p = 0.5
    sol = dynamic_reserve_greedy(values, p, budgets=[3.0])  # 3 sites, cost 1 each
    assert sol.protected == {0, 1, 2}
    assert sol.expected_value == pytest.approx(15.0)  # all protected at t=0


def test_zero_budget_leaves_everything_exposed():
    values = np.array([4.0, 6.0])
    p = np.array([0.25, 0.5])
    sol = dynamic_reserve_greedy(values, p, budgets=[0.0])
    assert sol.protected == set()
    # one period of exposure: Σ v·(1-p)^1
    assert sol.expected_value == pytest.approx(4.0 * 0.75 + 6.0 * 0.5)


def test_multi_period_schedule_length_and_split():
    values = np.array([9.0, 8.0, 1.0, 1.0])
    loss_prob = np.array([0.9, 0.9, 0.1, 0.1])
    sol = dynamic_reserve_greedy(values, loss_prob, budgets=[1.0, 1.0])
    assert len(sol.schedule) == 2
    # the two high-value at-risk sites get protected first (period 0, then 1)
    assert sol.protected.issuperset({0})
    assert sum(len(s) for s in sol.schedule) == len(sol.protected)


def test_costs_respected():
    values = np.array([10.0, 10.0])
    p = 0.5
    costs = np.array([3.0, 1.0])
    sol = dynamic_reserve_greedy(values, p, budgets=[1.0], costs=costs)
    assert sol.protected == {1}  # only the cost-1 site is affordable


def test_rejects_bad_loss_prob():
    with pytest.raises(ValueError, match="loss_prob|probabilit"):
        dynamic_reserve_greedy(np.array([1.0]), np.array([1.5]), budgets=[1.0])


def test_rejects_unknown_prioritize():
    with pytest.raises(ValueError, match="prioritize"):
        dynamic_reserve_greedy(np.array([1.0]), 0.1, budgets=[1.0], prioritize="x")
