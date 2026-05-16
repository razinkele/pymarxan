"""Phase 25: Cohon Pareto filter on a BLM calibration sweep.

Given a ``BLMResult``, drop dominated points so users see only the
Pareto-optimal cost–boundary trade-offs.

A point ``i`` is dominated by point ``j`` iff ``cost_j ≤ cost_i AND
boundary_j ≤ boundary_i AND (cost_j < cost_i OR boundary_j < boundary_i)``.
"""
from __future__ import annotations

import numpy as np

from pymarxan.calibration.blm import BLMResult
from pymarxan.calibration.pareto import pareto_frontier
from pymarxan.solvers.base import Solution


def _make_blm_result(points: list[tuple[float, float, float]]) -> BLMResult:
    """Build a BLMResult from (blm, cost, boundary) triples. Objective
    is ``cost + blm * boundary`` for the test; solutions are stubs."""
    blms, costs, bnds = zip(*points)
    objectives = [c + b * x for x, c, b in points]
    sols = [
        Solution(
            selected=np.zeros(1, dtype=bool),
            cost=c, boundary=b, objective=obj,
            targets_met={1: True},
        )
        for (_, c, b), obj in zip(points, objectives)
    ]
    return BLMResult(
        blm_values=list(blms),
        costs=list(costs),
        boundaries=list(bnds),
        objectives=objectives,
        solutions=sols,
    )


def test_pareto_filter_removes_dominated_points():
    """Point (5, 5) is dominated by (3, 3) — both cheaper and more
    compact."""
    res = _make_blm_result([
        (0.0, 3.0, 3.0),
        (0.5, 5.0, 5.0),   # dominated
        (1.0, 7.0, 2.0),
    ])
    pareto = pareto_frontier(res)
    assert len(pareto.costs) == 2
    assert 3.0 in pareto.costs and 7.0 in pareto.costs
    assert 5.0 not in pareto.costs


def test_pareto_filter_preserves_all_when_already_pareto():
    """Strictly trade-off points all survive: (1, 10), (5, 5), (10, 1)."""
    res = _make_blm_result([
        (0.0, 1.0, 10.0),
        (0.5, 5.0, 5.0),
        (1.0, 10.0, 1.0),
    ])
    pareto = pareto_frontier(res)
    assert len(pareto.costs) == 3


def test_pareto_filter_drops_strictly_inferior_duplicates():
    """Two points at (5, 5) — neither dominates the other strictly, but
    a third at (3, 3) dominates both. Filter keeps only the dominating
    one."""
    res = _make_blm_result([
        (0.0, 5.0, 5.0),
        (0.5, 5.0, 5.0),
        (1.0, 3.0, 3.0),
    ])
    pareto = pareto_frontier(res)
    assert len(pareto.costs) == 1
    assert pareto.costs[0] == 3.0


def test_pareto_filter_preserves_blm_value_pairing():
    """BLM values stay paired with their (cost, boundary) on the
    surviving points."""
    res = _make_blm_result([
        (0.0, 3.0, 3.0),
        (0.5, 7.0, 7.0),   # dominated
        (1.0, 7.0, 2.0),
    ])
    pareto = pareto_frontier(res)
    # The two surviving points have blms 0.0 and 1.0; 0.5 dropped.
    assert sorted(pareto.blm_values) == [0.0, 1.0]


def test_pareto_filter_empty_input_returns_empty():
    """A BLMResult with no points returns the same shape."""
    res = BLMResult(
        blm_values=[], costs=[], boundaries=[], objectives=[], solutions=[],
    )
    pareto = pareto_frontier(res)
    assert pareto.blm_values == []
    assert pareto.costs == []


def test_pareto_filter_returns_new_object():
    """Filter doesn't mutate the input — useful for repeated analysis."""
    res = _make_blm_result([
        (0.0, 3.0, 3.0),
        (0.5, 5.0, 5.0),
    ])
    before = list(res.costs)
    pareto_frontier(res)
    assert res.costs == before
