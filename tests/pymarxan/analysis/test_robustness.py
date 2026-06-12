"""Tests for multi-scenario robustness / minimax-regret selection."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pymarxan.analysis.robustness import (
    RegretResult,
    evaluate_plans_across_scenarios,
    minimax_regret,
)
from pymarxan.models.problem import ConservationProblem
from pymarxan.solvers.base import Solution


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


def _problem(costs: list[float]) -> ConservationProblem:
    planning_units = pd.DataFrame(
        {"id": [1, 2], "cost": costs, "status": [0, 0]}
    )
    features = pd.DataFrame(
        {"id": [1], "name": ["sp"], "target": [1.0], "spf": [1.0]}
    )
    pu_vs_features = pd.DataFrame(
        {"species": [1, 1], "pu": [1, 2], "amount": [1.0, 1.0]}
    )
    return ConservationProblem(planning_units, features, pu_vs_features)


def _sol(selected: list[bool]) -> Solution:
    return Solution(
        selected=np.array(selected, dtype=bool),
        cost=0.0,
        boundary=0.0,
        objective=0.0,
        targets_met={},
    )


def test_evaluate_plans_builds_objective_matrix():
    # Two scenarios differ only in PU costs.
    scen_a = _problem([10.0, 20.0])  # PU1 cheap
    scen_b = _problem([20.0, 10.0])  # PU2 cheap
    plan_pick1 = _sol([True, False])  # selects PU1
    plan_pick2 = _sol([False, True])  # selects PU2

    matrix, plans, scens = evaluate_plans_across_scenarios(
        problems={"A": scen_a, "B": scen_b},
        solutions={"pick1": plan_pick1, "pick2": plan_pick2},
        blm=0.0,
    )
    # rows align to solutions order, cols to problems order.
    assert plans == ["pick1", "pick2"]
    assert scens == ["A", "B"]
    # pick1 (PU1) costs 10 under A, 20 under B ; pick2 (PU2) costs 20 / 10.
    assert matrix[plans.index("pick1"), scens.index("A")] == pytest.approx(10.0)
    assert matrix[plans.index("pick1"), scens.index("B")] == pytest.approx(20.0)
    assert matrix[plans.index("pick2"), scens.index("B")] == pytest.approx(10.0)
    assert matrix[plans.index("pick2"), scens.index("A")] == pytest.approx(20.0)


def test_to_dataframe_shape():
    result = minimax_regret(
        np.array([[10.0, 20.0], [15.0, 5.0]]), plan_labels=["A", "B"]
    )
    df = result.to_dataframe()
    assert set(df.columns) == {"plan", "max_regret", "worst_case_cost"}
    assert len(df) == 2


def test_non_2d_cost_matrix_raises():
    with pytest.raises(ValueError, match="2-D"):
        minimax_regret(np.array([1.0, 2.0, 3.0]))


def test_empty_cost_matrix_raises():
    with pytest.raises(ValueError, match="empty"):
        minimax_regret(np.zeros((0, 3)))


def test_evaluate_plans_mismatched_pu_count_raises():
    # A plan sized for a 2-PU problem cannot be evaluated under a 3-PU one.
    two_pu = _problem([10.0, 20.0])
    three_pu = ConservationProblem(
        pd.DataFrame({"id": [1, 2, 3], "cost": [1.0, 1.0, 1.0], "status": [0, 0, 0]}),
        pd.DataFrame({"id": [1], "name": ["sp"], "target": [1.0], "spf": [1.0]}),
        pd.DataFrame({"species": [1], "pu": [1], "amount": [1.0]}),
    )
    plan = _sol([True, False])  # 2 selections
    with pytest.raises(ValueError, match="planning-unit set"):
        evaluate_plans_across_scenarios(
            problems={"ok": two_pu, "bad": three_pu},
            solutions={"plan": plan},
        )
