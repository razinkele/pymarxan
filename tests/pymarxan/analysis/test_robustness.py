"""Tests for multi-scenario robustness / minimax-regret selection."""
from __future__ import annotations

import numpy as np
import pytest

from pymarxan.analysis.robustness import RegretResult, minimax_regret


def test_regret_matrix_subtracts_best_in_each_scenario():
    # rows = plans, cols = scenarios; entries are costs (lower better).
    # scenario 0 best = 10 (plan 0); scenario 1 best = 5 (plan 1).
    cost = np.array([[10.0, 20.0], [15.0, 5.0]])
    result = minimax_regret(cost)
    # regret = cost - column min
    assert result.regret_matrix.tolist() == [[0.0, 15.0], [5.0, 0.0]]


def test_minimax_regret_picks_plan_with_smallest_max_regret():
    cost = np.array([[10.0, 20.0], [15.0, 5.0]])
    result = minimax_regret(cost, plan_labels=["A", "B"])
    # plan A max regret = 15 ; plan B max regret = 5 -> choose B
    assert result.max_regret.tolist() == [15.0, 5.0]
    assert result.minimax_regret_plan == "B"


def test_minimax_cost_plan_picks_best_worst_case_cost():
    # plan A worst-case 20, plan B worst-case 15 -> robust choice B
    cost = np.array([[10.0, 20.0], [15.0, 15.0]])
    result = minimax_regret(cost, plan_labels=["A", "B"])
    assert result.minimax_cost_plan == "B"


def test_default_labels_are_indices():
    cost = np.array([[1.0, 2.0], [2.0, 1.0]])
    result = minimax_regret(cost)
    assert result.plan_labels == [0, 1]
    assert result.scenario_labels == [0, 1]


def test_result_is_dataclass():
    result = minimax_regret(np.array([[1.0]]))
    assert isinstance(result, RegretResult)
