"""Tests for LinearConstraint."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

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


class TestLinearConstraintCreation:
    def test_valid_senses(self):
        for sense in ("<=", ">=", "=="):
            c = LinearConstraint(
                coefficients={1: 1.0}, sense=sense, rhs=10.0
            )
            assert c.sense == sense

    def test_invalid_sense(self):
        with pytest.raises(ValueError, match="sense must be"):
            LinearConstraint(coefficients={1: 1.0}, sense="<", rhs=10.0)

    def test_name(self):
        c = LinearConstraint(
            coefficients={1: 1.0}, sense="<=", rhs=10.0, label="Budget"
        )
        assert c.name() == "Budget"


class TestLinearConstraintEvaluate:
    def test_leq_satisfied(self):
        problem = _make_problem()
        # coefficients = cost, so total = 10+20 = 30
        c = LinearConstraint(
            coefficients={1: 10.0, 2: 20.0, 3: 30.0, 4: 40.0},
            sense="<=",
            rhs=50.0,
        )
        selected = np.array([True, True, False, False])
        result = c.evaluate(problem, selected)
        assert result.satisfied
        assert result.violation == 0.0

    def test_leq_violated(self):
        problem = _make_problem()
        c = LinearConstraint(
            coefficients={1: 10.0, 2: 20.0, 3: 30.0, 4: 40.0},
            sense="<=",
            rhs=25.0,
        )
        selected = np.array([True, True, False, False])  # LHS=30 > 25
        result = c.evaluate(problem, selected)
        assert not result.satisfied
        assert abs(result.violation - 5.0) < 1e-10

    def test_geq_satisfied(self):
        problem = _make_problem()
        c = LinearConstraint(
            coefficients={1: 10.0, 2: 20.0, 3: 30.0},
            sense=">=",
            rhs=25.0,
        )
        selected = np.array([True, True, False, False])  # LHS=30 >= 25
        result = c.evaluate(problem, selected)
        assert result.satisfied

    def test_geq_violated(self):
        problem = _make_problem()
        c = LinearConstraint(
            coefficients={1: 10.0, 2: 20.0, 3: 30.0},
            sense=">=",
            rhs=50.0,
        )
        selected = np.array([True, True, False, False])  # LHS=30 < 50
        result = c.evaluate(problem, selected)
        assert not result.satisfied
        assert abs(result.violation - 20.0) < 1e-10

    def test_eq_satisfied(self):
        problem = _make_problem()
        c = LinearConstraint(
            coefficients={1: 10.0, 2: 20.0},
            sense="==",
            rhs=30.0,
        )
        selected = np.array([True, True, False, False])
        result = c.evaluate(problem, selected)
        assert result.satisfied

    def test_eq_violated(self):
        problem = _make_problem()
        c = LinearConstraint(
            coefficients={1: 10.0, 2: 20.0},
            sense="==",
            rhs=25.0,
        )
        selected = np.array([True, True, False, False])  # LHS=30 != 25
        result = c.evaluate(problem, selected)
        assert not result.satisfied
        assert abs(result.violation - 5.0) < 1e-10


class TestLinearConstraintPenalty:
    def test_penalty_satisfied(self):
        problem = _make_problem()
        c = LinearConstraint(
            coefficients={1: 10.0, 2: 20.0},
            sense="<=",
            rhs=50.0,
            penalty_weight=100.0,
        )
        selected = np.array([True, True, False, False])
        assert c.penalty(problem, selected) == 0.0

    def test_penalty_violated(self):
        problem = _make_problem()
        c = LinearConstraint(
            coefficients={1: 10.0, 2: 20.0},
            sense="<=",
            rhs=25.0,
            penalty_weight=100.0,
        )
        selected = np.array([True, True, False, False])  # violation=5
        assert abs(c.penalty(problem, selected) - 500.0) < 1e-10


class TestLinearConstraintIncremental:
    def test_init_state(self):
        problem = _make_problem()
        cache = ProblemCache.from_problem(problem)
        c = LinearConstraint(
            coefficients={1: 10.0, 2: 20.0, 3: 30.0, 4: 40.0},
            sense="<=",
            rhs=50.0,
        )
        selected = np.array([True, True, False, False])
        state = c.init_state(problem, selected, cache)
        assert abs(state["lhs"] - 30.0) < 1e-10

    def test_delta_soft_no_change(self):
        """Adding a PU with zero coefficient produces zero delta."""
        problem = _make_problem()
        cache = ProblemCache.from_problem(problem)
        c = LinearConstraint(
            coefficients={1: 10.0},  # only PU 1 has coefficient
            sense="<=",
            rhs=50.0,
            penalty_weight=100.0,
        )
        selected = np.array([True, False, False, False])
        state = c.init_state(problem, selected, cache)
        # Flip PU 2 (idx=1) — not in coefficients
        delta = c.compute_delta(1, selected, state, cache)
        assert delta == 0.0

    def test_delta_matches_full_penalty_all_flips(self):
        """Incremental delta matches difference in full penalty for every flip."""
        problem = _make_problem()
        cache = ProblemCache.from_problem(problem)
        c = LinearConstraint(
            coefficients={1: 10.0, 2: 20.0, 3: 30.0, 4: 40.0},
            sense="<=",
            rhs=35.0,
            penalty_weight=100.0,
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

    def test_update_state_add(self):
        problem = _make_problem()
        cache = ProblemCache.from_problem(problem)
        c = LinearConstraint(
            coefficients={1: 10.0, 2: 20.0, 3: 30.0},
            sense="<=",
            rhs=50.0,
        )
        selected = np.array([True, False, False, False])
        state = c.init_state(problem, selected, cache)
        assert abs(state["lhs"] - 10.0) < 1e-10
        # Accept flip: add PU 2 (idx=1)
        selected[1] = True
        c.update_state(1, selected, state, cache)
        assert abs(state["lhs"] - 30.0) < 1e-10

    def test_update_state_remove(self):
        problem = _make_problem()
        cache = ProblemCache.from_problem(problem)
        c = LinearConstraint(
            coefficients={1: 10.0, 2: 20.0, 3: 30.0},
            sense="<=",
            rhs=50.0,
        )
        selected = np.array([True, True, False, False])
        state = c.init_state(problem, selected, cache)
        # Remove PU 1 (idx=0)
        selected[0] = False
        c.update_state(0, selected, state, cache)
        assert abs(state["lhs"] - 20.0) < 1e-10


class TestLinearConstraintHard:
    def test_hard_rejects_infeasible_move(self):
        """Hard constraint returns +inf for feasible→infeasible."""
        problem = _make_problem()
        cache = ProblemCache.from_problem(problem)
        c = LinearConstraint(
            coefficients={1: 10.0, 2: 20.0, 3: 30.0, 4: 40.0},
            sense="<=",
            rhs=35.0,
            hard=True,
        )
        selected = np.array([True, True, False, False])  # LHS=30, feasible
        state = c.init_state(problem, selected, cache)
        # Adding PU 3 (idx=2, coeff=30) → LHS=60 > 35
        delta = c.compute_delta(2, selected, state, cache)
        assert math.isinf(delta) and delta > 0

    def test_hard_allows_feasible_move(self):
        """Hard constraint returns 0 for feasible→feasible."""
        problem = _make_problem()
        cache = ProblemCache.from_problem(problem)
        c = LinearConstraint(
            coefficients={1: 10.0, 2: 20.0, 3: 30.0, 4: 40.0},
            sense="<=",
            rhs=50.0,
            hard=True,
        )
        selected = np.array([True, False, False, False])  # LHS=10
        state = c.init_state(problem, selected, cache)
        # Adding PU 2 (idx=1, coeff=20) → LHS=30 <= 50
        delta = c.compute_delta(1, selected, state, cache)
        assert delta == 0.0

    def test_hard_allows_repair_move(self):
        """Hard constraint allows violated→less-violated (negative delta)."""
        problem = _make_problem()
        cache = ProblemCache.from_problem(problem)
        c = LinearConstraint(
            coefficients={1: 10.0, 2: 20.0, 3: 30.0, 4: 40.0},
            sense="<=",
            rhs=25.0,
            hard=True,
        )
        # All selected: LHS=100, violation=75
        selected = np.array([True, True, True, True])
        state = c.init_state(problem, selected, cache)
        # Removing PU 4 (idx=3, coeff=40) → LHS=60, violation=35 (less)
        delta = c.compute_delta(3, selected, state, cache)
        assert delta < 0  # repair move is strongly incentivized

    def test_hard_blocks_worsening_from_infeasible(self):
        """Hard constraint returns +inf for violated→more-violated."""
        problem = _make_problem()
        cache = ProblemCache.from_problem(problem)
        c = LinearConstraint(
            coefficients={1: 10.0, 2: 20.0, 3: 30.0, 4: 40.0},
            sense="<=",
            rhs=25.0,
            hard=True,
        )
        selected = np.array([True, True, False, False])  # LHS=30, v=5
        state = c.init_state(problem, selected, cache)
        # Adding PU 3 (idx=2, coeff=30) → LHS=60, v=35 (worse)
        delta = c.compute_delta(2, selected, state, cache)
        assert math.isinf(delta) and delta > 0

    def test_hard_geq_blocks_removal(self):
        """Hard >= constraint blocks removal that causes violation."""
        problem = _make_problem()
        cache = ProblemCache.from_problem(problem)
        c = LinearConstraint(
            coefficients={1: 10.0, 2: 20.0},
            sense=">=",
            rhs=25.0,
            hard=True,
        )
        selected = np.array([True, True, False, False])  # LHS=30 >= 25
        state = c.init_state(problem, selected, cache)
        # Removing PU 2 (idx=1, coeff=20) → LHS=10 < 25 (violation)
        delta = c.compute_delta(1, selected, state, cache)
        assert math.isinf(delta) and delta > 0


class TestLinearConstraintMIP:
    def test_apply_to_mip_leq(self):
        """Smoke test MIP integration."""
        pulp = pytest.importorskip("pulp")
        problem = _make_problem()
        model = pulp.LpProblem("test", pulp.LpMinimize)
        x = {
            pid: pulp.LpVariable(f"x_{pid}", cat="Binary")
            for pid in [1, 2, 3, 4]
        }
        c = LinearConstraint(
            coefficients={1: 10.0, 2: 20.0, 3: 30.0},
            sense="<=",
            rhs=40.0,
            label="BudgetLimit",
        )
        c.apply_to_mip(problem, model, x)
        assert "BudgetLimit" in [c.name for c in model.constraints.values()]

    def test_apply_to_mip_geq(self):
        pulp = pytest.importorskip("pulp")
        problem = _make_problem()
        model = pulp.LpProblem("test", pulp.LpMinimize)
        x = {
            pid: pulp.LpVariable(f"x_{pid}", cat="Binary")
            for pid in [1, 2, 3, 4]
        }
        c = LinearConstraint(
            coefficients={1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0},
            sense=">=",
            rhs=2.0,
            label="MinPUs",
        )
        c.apply_to_mip(problem, model, x)
        assert "MinPUs" in [c.name for c in model.constraints.values()]

    def test_apply_to_mip_eq(self):
        pulp = pytest.importorskip("pulp")
        problem = _make_problem()
        model = pulp.LpProblem("test", pulp.LpMinimize)
        x = {
            pid: pulp.LpVariable(f"x_{pid}", cat="Binary")
            for pid in [1, 2, 3, 4]
        }
        c = LinearConstraint(
            coefficients={1: 1.0, 2: 1.0},
            sense="==",
            rhs=1.0,
            label="ExactlyOne",
        )
        c.apply_to_mip(problem, model, x)
        assert "ExactlyOne" in [c.name for c in model.constraints.values()]
