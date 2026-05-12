"""Tests for constraint framework base classes."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pymarxan.constraints.base import (
    Constraint,
    ConstraintResult,
    IncrementalConstraint,
    IncrementalZonalConstraint,
    ZonalConstraint,
)
from pymarxan.models.problem import ConservationProblem
from pymarxan.solvers.cache import ProblemCache


def _make_simple_problem() -> ConservationProblem:
    planning_units = pd.DataFrame({
        "id": [1, 2, 3],
        "cost": [10.0, 15.0, 20.0],
        "status": [0, 0, 0],
    })
    features = pd.DataFrame({
        "id": [1, 2],
        "name": ["sp_a", "sp_b"],
        "target": [20.0, 10.0],
        "spf": [1.0, 1.0],
    })
    pu_vs_features = pd.DataFrame({
        "species": [1, 1, 1, 2, 2],
        "pu": [1, 2, 3, 1, 3],
        "amount": [10.0, 15.0, 5.0, 8.0, 12.0],
    })
    boundary = pd.DataFrame({
        "id1": [1, 2],
        "id2": [2, 3],
        "boundary": [1.0, 1.0],
    })
    return ConservationProblem(
        planning_units=planning_units,
        features=features,
        pu_vs_features=pu_vs_features,
        boundary=boundary,
        parameters={"BLM": 1.0},
    )


# --- Concrete test implementations ---


class MockConstraint(Constraint):
    """MIP-only constraint for testing."""

    def name(self) -> str:
        return "MockConstraint"

    def evaluate(self, problem, selected) -> ConstraintResult:
        # Violated if more than 2 PUs selected
        n_selected = int(np.sum(selected))
        satisfied = n_selected <= 2
        violation = max(0, n_selected - 2)
        return ConstraintResult(
            satisfied=satisfied,
            violation=float(violation),
            description=f"{n_selected} PUs selected (max 2)",
        )

    def penalty(self, problem, selected) -> float:
        result = self.evaluate(problem, selected)
        return 1000.0 * result.violation


class MockIncrementalConstraint(IncrementalConstraint):
    """Incremental constraint that limits total cost."""

    def __init__(self, max_cost: float = 30.0, penalty_weight: float = 100.0):
        self.max_cost = max_cost
        self.penalty_weight = penalty_weight

    def name(self) -> str:
        return "MockCostLimit"

    def evaluate(self, problem, selected) -> ConstraintResult:
        costs = np.asarray(problem.planning_units["cost"].values)
        total = float(np.sum(costs[selected]))
        satisfied = total <= self.max_cost
        violation = max(0.0, total - self.max_cost)
        return ConstraintResult(
            satisfied=satisfied,
            violation=violation,
            description=f"Cost {total:.1f} vs limit {self.max_cost}",
        )

    def penalty(self, problem, selected) -> float:
        result = self.evaluate(problem, selected)
        return self.penalty_weight * result.violation

    def init_state(self, problem, selected, cache) -> dict:
        total_cost = float(np.sum(cache.costs[selected]))
        return {"total_cost": total_cost}

    def compute_delta(self, idx, selected, state, cache) -> float:
        sign = 1.0 if not selected[idx] else -1.0
        new_cost = state["total_cost"] + sign * cache.costs[idx]
        old_violation = max(0.0, state["total_cost"] - self.max_cost)
        new_violation = max(0.0, new_cost - self.max_cost)
        return self.penalty_weight * (new_violation - old_violation)

    def update_state(self, idx, selected, state, cache) -> None:
        # Called AFTER selected[idx] is mutated
        sign = 1.0 if selected[idx] else -1.0
        state["total_cost"] += sign * cache.costs[idx]


class MockZonalConstraint(ZonalConstraint):
    """Zonal constraint for testing."""

    def evaluate_zonal(self, problem, assignment) -> ConstraintResult:
        # All PUs must be assigned to some zone (no zone 0)
        n_unassigned = int(np.sum(assignment == 0))
        return ConstraintResult(
            satisfied=n_unassigned == 0,
            violation=float(n_unassigned),
            description=f"{n_unassigned} unassigned PUs",
        )

    def penalty_zonal(self, problem, assignment) -> float:
        result = self.evaluate_zonal(problem, assignment)
        return 500.0 * result.violation


class MockIncrementalZonalConstraint(IncrementalZonalConstraint):
    """Incremental zonal constraint for testing."""

    def evaluate_zonal(self, problem, assignment) -> ConstraintResult:
        n_unassigned = int(np.sum(assignment == 0))
        return ConstraintResult(
            satisfied=n_unassigned == 0,
            violation=float(n_unassigned),
            description=f"{n_unassigned} unassigned",
        )

    def penalty_zonal(self, problem, assignment) -> float:
        return 500.0 * self.evaluate_zonal(problem, assignment).violation

    def init_zone_state(self, problem, assignment, zone_cache) -> dict:
        return {"n_unassigned": int(np.sum(assignment == 0))}

    def compute_zone_delta(self, idx, old_zone, new_zone, assignment,
                           state, zone_cache) -> float:
        old_n = state["n_unassigned"]
        # Moving to zone 0 adds an unassigned; moving from zone 0 removes one
        delta_n = 0
        if old_zone == 0 and new_zone != 0:
            delta_n = -1
        elif old_zone != 0 and new_zone == 0:
            delta_n = 1
        return 500.0 * delta_n

    def update_zone_state(self, idx, old_zone, new_zone, assignment,
                          state, zone_cache) -> None:
        if old_zone == 0 and new_zone != 0:
            state["n_unassigned"] -= 1
        elif old_zone != 0 and new_zone == 0:
            state["n_unassigned"] += 1


# --- Tests ---


class TestConstraintResult:
    def test_satisfied(self):
        r = ConstraintResult(satisfied=True, violation=0.0, description="ok")
        assert r.satisfied
        assert r.violation == 0.0

    def test_violated(self):
        r = ConstraintResult(satisfied=False, violation=2.5, description="bad")
        assert not r.satisfied
        assert r.violation == 2.5


class TestConstraintABC:
    def test_evaluate(self):
        c = MockConstraint()
        problem = _make_simple_problem()
        selected = np.array([True, False, False])
        result = c.evaluate(problem, selected)
        assert result.satisfied
        assert result.violation == 0.0

    def test_evaluate_violated(self):
        c = MockConstraint()
        problem = _make_simple_problem()
        selected = np.array([True, True, True])
        result = c.evaluate(problem, selected)
        assert not result.satisfied
        assert result.violation == 1.0

    def test_penalty_satisfied(self):
        c = MockConstraint()
        problem = _make_simple_problem()
        selected = np.array([True, True, False])
        assert c.penalty(problem, selected) == 0.0

    def test_penalty_violated(self):
        c = MockConstraint()
        problem = _make_simple_problem()
        selected = np.array([True, True, True])
        assert c.penalty(problem, selected) == 1000.0

    def test_apply_to_mip_not_implemented(self):
        c = MockConstraint()
        with pytest.raises(NotImplementedError, match="MockConstraint"):
            c.apply_to_mip(None, None, None)  # type: ignore[arg-type]

    def test_name(self):
        c = MockConstraint()
        assert c.name() == "MockConstraint"


class TestIncrementalConstraint:
    def test_init_state(self):
        c = MockIncrementalConstraint(max_cost=30.0)
        problem = _make_simple_problem()
        cache = ProblemCache.from_problem(problem)
        selected = np.array([True, True, False])  # cost = 10 + 15 = 25
        state = c.init_state(problem, selected, cache)
        assert abs(state["total_cost"] - 25.0) < 1e-10

    def test_compute_delta_no_violation(self):
        c = MockIncrementalConstraint(max_cost=50.0)
        problem = _make_simple_problem()
        cache = ProblemCache.from_problem(problem)
        selected = np.array([True, False, False])  # cost = 10
        state = c.init_state(problem, selected, cache)
        # Adding PU 1 (idx=1, cost=15) → total=25, still under 50
        delta = c.compute_delta(1, selected, state, cache)
        assert delta == 0.0

    def test_compute_delta_into_violation(self):
        c = MockIncrementalConstraint(max_cost=20.0, penalty_weight=100.0)
        problem = _make_simple_problem()
        cache = ProblemCache.from_problem(problem)
        selected = np.array([True, True, False])  # cost = 25, already over
        state = c.init_state(problem, selected, cache)
        # Adding PU 2 (idx=2, cost=20) → total=45, violation goes 5→25
        delta = c.compute_delta(2, selected, state, cache)
        assert delta == 100.0 * (25.0 - 5.0)

    def test_update_state_add(self):
        c = MockIncrementalConstraint(max_cost=30.0)
        problem = _make_simple_problem()
        cache = ProblemCache.from_problem(problem)
        selected = np.array([True, False, False])
        state = c.init_state(problem, selected, cache)
        assert abs(state["total_cost"] - 10.0) < 1e-10
        # Simulate accepting a flip: add PU 1 (idx=1)
        selected[1] = True
        c.update_state(1, selected, state, cache)
        assert abs(state["total_cost"] - 25.0) < 1e-10

    def test_update_state_remove(self):
        c = MockIncrementalConstraint(max_cost=30.0)
        problem = _make_simple_problem()
        cache = ProblemCache.from_problem(problem)
        selected = np.array([True, True, False])
        state = c.init_state(problem, selected, cache)
        assert abs(state["total_cost"] - 25.0) < 1e-10
        # Simulate accepting a flip: remove PU 0 (idx=0)
        selected[0] = False
        c.update_state(0, selected, state, cache)
        assert abs(state["total_cost"] - 15.0) < 1e-10

    def test_delta_matches_full_penalty_change(self):
        """Incremental delta must match difference in full penalty."""
        c = MockIncrementalConstraint(max_cost=25.0, penalty_weight=100.0)
        problem = _make_simple_problem()
        cache = ProblemCache.from_problem(problem)

        for flip_idx in range(3):
            selected = np.array([True, True, False])
            state = c.init_state(problem, selected, cache)
            pen_before = c.penalty(problem, selected)
            delta = c.compute_delta(flip_idx, selected, state, cache)

            # Apply flip
            selected[flip_idx] = not selected[flip_idx]
            pen_after = c.penalty(problem, selected)

            assert abs(delta - (pen_after - pen_before)) < 1e-10, (
                f"Delta mismatch for flip_idx={flip_idx}: "
                f"delta={delta}, actual={pen_after - pen_before}"
            )


class TestZonalConstraint:
    def test_evaluate_zonal_satisfied(self):
        c = MockZonalConstraint()
        assignment = np.array([1, 2, 1])
        result = c.evaluate_zonal(None, assignment)  # type: ignore[arg-type]
        assert result.satisfied

    def test_evaluate_zonal_violated(self):
        c = MockZonalConstraint()
        assignment = np.array([0, 2, 1])
        result = c.evaluate_zonal(None, assignment)  # type: ignore[arg-type]
        assert not result.satisfied
        assert result.violation == 1.0

    def test_penalty_zonal(self):
        c = MockZonalConstraint()
        assignment = np.array([0, 0, 1])
        assert c.penalty_zonal(None, assignment) == 1000.0  # type: ignore[arg-type]

    def test_apply_to_zone_mip_not_implemented(self):
        c = MockZonalConstraint()
        with pytest.raises(NotImplementedError):
            c.apply_to_zone_mip(None, None, None)  # type: ignore[arg-type]


class TestIncrementalZonalConstraint:
    def test_init_zone_state(self):
        c = MockIncrementalZonalConstraint()
        assignment = np.array([0, 1, 2])
        state = c.init_zone_state(None, assignment, None)  # type: ignore[arg-type]
        assert state["n_unassigned"] == 1

    def test_compute_zone_delta_assign(self):
        c = MockIncrementalZonalConstraint()
        assignment = np.array([0, 1, 2])
        state = {"n_unassigned": 1}
        # Assign PU 0 from zone 0 to zone 1
        delta = c.compute_zone_delta(0, 0, 1, assignment, state, None)  # type: ignore[arg-type]
        assert delta == -500.0  # one fewer unassigned

    def test_compute_zone_delta_unassign(self):
        c = MockIncrementalZonalConstraint()
        assignment = np.array([1, 1, 2])
        state = {"n_unassigned": 0}
        # Unassign PU 0 from zone 1 to zone 0
        delta = c.compute_zone_delta(0, 1, 0, assignment, state, None)  # type: ignore[arg-type]
        assert delta == 500.0  # one more unassigned

    def test_update_zone_state(self):
        c = MockIncrementalZonalConstraint()
        assignment = np.array([0, 1, 2])
        state = {"n_unassigned": 1}
        # Accept move: PU 0 goes from zone 0 to zone 1
        assignment[0] = 1
        c.update_zone_state(0, 0, 1, assignment, state, None)  # type: ignore[arg-type]
        assert state["n_unassigned"] == 0

    def test_zone_delta_matches_full_penalty(self):
        """Incremental zone delta must match full penalty difference."""
        c = MockIncrementalZonalConstraint()
        assignment = np.array([0, 1, 2])
        state = c.init_zone_state(None, assignment, None)  # type: ignore[arg-type]
        pen_before = c.penalty_zonal(None, assignment)  # type: ignore[arg-type]
        delta = c.compute_zone_delta(0, 0, 1, assignment, state, None)  # type: ignore[arg-type]

        assignment[0] = 1
        c.update_zone_state(0, 0, 1, assignment, state, None)  # type: ignore[arg-type]
        pen_after = c.penalty_zonal(None, assignment)  # type: ignore[arg-type]

        assert abs(delta - (pen_after - pen_before)) < 1e-10
