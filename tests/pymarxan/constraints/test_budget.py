"""Tests for budget_constraint convenience wrapper."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from pymarxan.constraints.budget import budget_constraint
from pymarxan.constraints.linear import LinearConstraint
from pymarxan.models.problem import ConservationProblem
from pymarxan.solvers.cache import ProblemCache


def _make_problem() -> ConservationProblem:
    planning_units = pd.DataFrame({
        "id": [1, 2, 3, 4],
        "cost": [10.0, 20.0, 30.0, 40.0],
        "status": [0, 0, 0, 0],
    })
    features = pd.DataFrame({
        "id": [1],
        "name": ["sp_a"],
        "target": [10.0],
        "spf": [1.0],
    })
    pu_vs_features = pd.DataFrame({
        "species": [1, 1, 1, 1],
        "pu": [1, 2, 3, 4],
        "amount": [5.0, 5.0, 5.0, 5.0],
    })
    return ConservationProblem(
        planning_units=planning_units,
        features=features,
        pu_vs_features=pu_vs_features,
        parameters={"BLM": 0.0},
    )


class TestBudgetConstraintCreation:
    def test_returns_linear_constraint(self):
        problem = _make_problem()
        c = budget_constraint(problem, max_budget=50.0)
        assert isinstance(c, LinearConstraint)

    def test_sense_is_leq(self):
        problem = _make_problem()
        c = budget_constraint(problem, max_budget=50.0)
        assert c.sense == "<="

    def test_rhs_matches_max_budget(self):
        problem = _make_problem()
        c = budget_constraint(problem, max_budget=42.0)
        assert c.rhs == 42.0

    def test_label_is_budget(self):
        problem = _make_problem()
        c = budget_constraint(problem, max_budget=50.0)
        assert c.name() == "Budget"

    def test_hard_default_true(self):
        problem = _make_problem()
        c = budget_constraint(problem, max_budget=50.0)
        assert c.hard is True

    def test_soft_mode(self):
        problem = _make_problem()
        c = budget_constraint(problem, max_budget=50.0, hard=False)
        assert c.hard is False

    def test_custom_penalty_weight(self):
        problem = _make_problem()
        c = budget_constraint(
            problem, max_budget=50.0, penalty_weight=500.0
        )
        assert c.penalty_weight == 500.0


class TestBudgetCoefficients:
    def test_coefficients_match_costs(self):
        problem = _make_problem()
        c = budget_constraint(problem, max_budget=100.0)
        assert c.coefficients == {1: 10.0, 2: 20.0, 3: 30.0, 4: 40.0}

    def test_coefficients_keys_are_pu_ids(self):
        problem = _make_problem()
        c = budget_constraint(problem, max_budget=100.0)
        assert set(c.coefficients.keys()) == {1, 2, 3, 4}


class TestBudgetEvaluation:
    def test_satisfied_within_budget(self):
        problem = _make_problem()
        c = budget_constraint(problem, max_budget=50.0)
        selected = np.array([True, True, False, False])  # cost=30
        result = c.evaluate(problem, selected)
        assert result.satisfied
        assert result.violation == 0.0

    def test_violated_over_budget(self):
        problem = _make_problem()
        c = budget_constraint(problem, max_budget=25.0)
        selected = np.array([True, True, False, False])  # cost=30 > 25
        result = c.evaluate(problem, selected)
        assert not result.satisfied
        assert abs(result.violation - 5.0) < 1e-10


class TestBudgetHardMode:
    def test_hard_rejects_over_budget_move(self):
        problem = _make_problem()
        cache = ProblemCache.from_problem(problem)
        c = budget_constraint(problem, max_budget=35.0)
        selected = np.array([True, True, False, False])  # cost=30
        state = c.init_state(problem, selected, cache)
        # Adding PU 3 (idx=2, cost=30) → total=60 > 35
        delta = c.compute_delta(2, selected, state, cache)
        assert math.isinf(delta) and delta > 0

    def test_hard_allows_within_budget_move(self):
        problem = _make_problem()
        cache = ProblemCache.from_problem(problem)
        c = budget_constraint(problem, max_budget=100.0)
        selected = np.array([True, False, False, False])  # cost=10
        state = c.init_state(problem, selected, cache)
        # Adding PU 2 (idx=1, cost=20) → total=30 <= 100
        delta = c.compute_delta(1, selected, state, cache)
        assert delta == 0.0


class TestBudgetPenalty:
    def test_penalty_zero_when_satisfied(self):
        problem = _make_problem()
        c = budget_constraint(
            problem, max_budget=50.0, hard=False, penalty_weight=100.0
        )
        selected = np.array([True, True, False, False])  # cost=30
        assert c.penalty(problem, selected) == 0.0

    def test_penalty_matches_overrun_times_weight(self):
        problem = _make_problem()
        c = budget_constraint(
            problem, max_budget=25.0, hard=False, penalty_weight=100.0
        )
        selected = np.array([True, True, False, False])  # cost=30, v=5
        assert abs(c.penalty(problem, selected) - 500.0) < 1e-10


class TestBudgetDeltaConsistency:
    def test_delta_matches_full_recompute_all_flips(self):
        """Incremental delta matches full penalty diff for every flip."""
        problem = _make_problem()
        cache = ProblemCache.from_problem(problem)
        c = budget_constraint(
            problem, max_budget=35.0, hard=False, penalty_weight=100.0
        )
        for flip_idx in range(4):
            selected = np.array([True, True, False, False])
            state = c.init_state(problem, selected, cache)
            pen_before = c.penalty(problem, selected)
            delta = c.compute_delta(flip_idx, selected, state, cache)
            selected[flip_idx] = not selected[flip_idx]
            pen_after = c.penalty(problem, selected)
            assert abs(delta - (pen_after - pen_before)) < 1e-10, (
                f"flip_idx={flip_idx}: delta={delta}, "
                f"actual={pen_after - pen_before}"
            )
