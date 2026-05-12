"""Tests for MinNeighborConstraint."""
from __future__ import annotations

import numpy as np
import pandas as pd

from pymarxan.constraints.neighbor import MinNeighborConstraint
from pymarxan.models.problem import ConservationProblem
from pymarxan.solvers.cache import ProblemCache


def _make_grid_problem() -> ConservationProblem:
    """4-PU grid: 1-2-3-4 with boundary between adjacent pairs.

    Layout (linear chain):  1 — 2 — 3 — 4
    Adjacency: 1↔2, 2↔3, 3↔4
    """
    planning_units = pd.DataFrame({
        "id": [1, 2, 3, 4],
        "cost": [1.0, 1.0, 1.0, 1.0],
        "status": [0, 0, 0, 0],
    })
    features = pd.DataFrame({
        "id": [1],
        "name": ["sp_a"],
        "target": [1.0],
        "spf": [1.0],
    })
    pu_vs_features = pd.DataFrame({
        "species": [1, 1, 1, 1],
        "pu": [1, 2, 3, 4],
        "amount": [1.0, 1.0, 1.0, 1.0],
    })
    boundary = pd.DataFrame({
        "id1": [1, 2, 3],
        "id2": [2, 3, 4],
        "boundary": [1.0, 1.0, 1.0],
    })
    return ConservationProblem(
        planning_units=planning_units,
        features=features,
        pu_vs_features=pu_vs_features,
        boundary=boundary,
        parameters={"BLM": 0.0},
    )


def _make_no_boundary_problem() -> ConservationProblem:
    """4-PU problem with no boundary data (no neighbors)."""
    planning_units = pd.DataFrame({
        "id": [1, 2, 3, 4],
        "cost": [1.0, 1.0, 1.0, 1.0],
        "status": [0, 0, 0, 0],
    })
    features = pd.DataFrame({
        "id": [1],
        "name": ["sp_a"],
        "target": [1.0],
        "spf": [1.0],
    })
    pu_vs_features = pd.DataFrame({
        "species": [1, 1, 1, 1],
        "pu": [1, 2, 3, 4],
        "amount": [1.0, 1.0, 1.0, 1.0],
    })
    return ConservationProblem(
        planning_units=planning_units,
        features=features,
        pu_vs_features=pu_vs_features,
        parameters={"BLM": 0.0},
    )


class TestEvaluate:
    def test_all_neighbors_satisfied(self):
        """All selected PUs have enough neighbors → satisfied."""
        problem = _make_grid_problem()
        # Select PUs 2,3 (idx 1,2): each has 1 selected neighbor
        c = MinNeighborConstraint(min_neighbors=1)
        selected = np.array([False, True, True, False])
        result = c.evaluate(problem, selected)
        assert result.satisfied
        assert result.violation == 0.0

    def test_isolated_pu_violated(self):
        """Single isolated PU has no neighbors → violated."""
        problem = _make_grid_problem()
        c = MinNeighborConstraint(min_neighbors=1)
        # Select only PU 1 (idx 0): neighbor is PU 2 (not selected)
        selected = np.array([True, False, False, False])
        result = c.evaluate(problem, selected)
        assert not result.satisfied
        assert result.violation == 1.0
        assert "1 PUs" in result.description

    def test_multiple_violations(self):
        """Two isolated PUs → deficit = 2."""
        problem = _make_grid_problem()
        c = MinNeighborConstraint(min_neighbors=1)
        # Select PUs 1 and 4 (idx 0, 3): neither has a selected neighbor
        selected = np.array([True, False, False, True])
        result = c.evaluate(problem, selected)
        assert not result.satisfied
        assert result.violation == 2.0

    def test_higher_min_neighbors(self):
        """min_neighbors=2: endpoint PUs only have 1 neighbor each."""
        problem = _make_grid_problem()
        c = MinNeighborConstraint(min_neighbors=2)
        # Select all: PU 1 has 1 neighbor, PU 4 has 1 neighbor
        # PU 2 has 2, PU 3 has 2 → deficit for PU 1 and PU 4
        selected = np.array([True, True, True, True])
        result = c.evaluate(problem, selected)
        assert not result.satisfied
        # PU 1: deficit=1, PU 4: deficit=1
        assert result.violation == 2.0


class TestPenalty:
    def test_penalty_proportional_to_deficit(self):
        problem = _make_grid_problem()
        c = MinNeighborConstraint(min_neighbors=1, penalty_weight=50.0)
        # Two isolated PUs → deficit=2
        selected = np.array([True, False, False, True])
        pen = c.penalty(problem, selected)
        assert abs(pen - 100.0) < 1e-10

    def test_penalty_zero_when_satisfied(self):
        problem = _make_grid_problem()
        c = MinNeighborConstraint(min_neighbors=1, penalty_weight=100.0)
        selected = np.array([False, True, True, False])
        assert c.penalty(problem, selected) == 0.0


class TestInitState:
    def test_neighbor_counts(self):
        """init_state computes correct neighbor counts."""
        problem = _make_grid_problem()
        cache = ProblemCache.from_problem(problem)
        c = MinNeighborConstraint(min_neighbors=1)
        # Select PUs 1,2,3 (idx 0,1,2)
        selected = np.array([True, True, True, False])
        state = c.init_state(problem, selected, cache)
        nc = state["neighbor_count"]
        # PU 0 neighbors: [1] → 1 selected neighbor contributing to PU 0
        assert nc[0] == 1  # PU 2 is selected neighbor of PU 1
        assert nc[1] == 2  # PU 1 and PU 3 selected
        assert nc[2] == 1  # PU 2 selected
        assert nc[3] == 1  # PU 3 selected (contributes to PU 4)

    def test_none_selected(self):
        problem = _make_grid_problem()
        cache = ProblemCache.from_problem(problem)
        c = MinNeighborConstraint(min_neighbors=1)
        selected = np.array([False, False, False, False])
        state = c.init_state(problem, selected, cache)
        np.testing.assert_array_equal(
            state["neighbor_count"], [0, 0, 0, 0]
        )


class TestDelta:
    def test_delta_matches_full_penalty_all_flips(self):
        """Exhaustive: delta matches full penalty diff for every flip."""
        problem = _make_grid_problem()
        cache = ProblemCache.from_problem(problem)
        c = MinNeighborConstraint(min_neighbors=1, penalty_weight=100.0)

        for init_mask in range(16):  # all 2^4 starting states
            selected = np.array(
                [(init_mask >> i) & 1 == 1 for i in range(4)]
            )
            for flip_idx in range(4):
                sel = selected.copy()
                state = c.init_state(problem, sel, cache)
                pen_before = c.penalty(problem, sel)
                delta = c.compute_delta(flip_idx, sel, state, cache)
                sel[flip_idx] = not sel[flip_idx]
                pen_after = c.penalty(problem, sel)
                expected = pen_after - pen_before
                assert abs(delta - expected) < 1e-10, (
                    f"mask={init_mask:04b} flip={flip_idx}: "
                    f"delta={delta}, expected={expected}"
                )

    def test_delta_with_min_neighbors_2(self):
        """Delta correct with higher min_neighbors threshold."""
        problem = _make_grid_problem()
        cache = ProblemCache.from_problem(problem)
        c = MinNeighborConstraint(min_neighbors=2, penalty_weight=10.0)

        for init_mask in range(16):
            selected = np.array(
                [(init_mask >> i) & 1 == 1 for i in range(4)]
            )
            for flip_idx in range(4):
                sel = selected.copy()
                state = c.init_state(problem, sel, cache)
                pen_before = c.penalty(problem, sel)
                delta = c.compute_delta(flip_idx, sel, state, cache)
                sel[flip_idx] = not sel[flip_idx]
                pen_after = c.penalty(problem, sel)
                expected = pen_after - pen_before
                assert abs(delta - expected) < 1e-10, (
                    f"mask={init_mask:04b} flip={flip_idx}: "
                    f"delta={delta}, expected={expected}"
                )


class TestUpdateState:
    def test_update_after_add(self):
        problem = _make_grid_problem()
        cache = ProblemCache.from_problem(problem)
        c = MinNeighborConstraint(min_neighbors=1)
        selected = np.array([True, False, False, False])
        state = c.init_state(problem, selected, cache)
        # Add PU 2 (idx 1)
        selected[1] = True
        c.update_state(1, selected, cache=cache, state=state)
        nc = state["neighbor_count"]
        # PU 0 now has 1 selected neighbor (PU 1)
        assert nc[0] == 1
        # PU 2 (idx 2) now has 1 selected neighbor (PU 1)
        assert nc[2] == 1

    def test_update_after_remove(self):
        problem = _make_grid_problem()
        cache = ProblemCache.from_problem(problem)
        c = MinNeighborConstraint(min_neighbors=1)
        selected = np.array([True, True, True, False])
        state = c.init_state(problem, selected, cache)
        # Remove PU 2 (idx 1)
        selected[1] = False
        c.update_state(1, selected, cache=cache, state=state)
        nc = state["neighbor_count"]
        # PU 0: lost PU 1 as neighbor → 0
        assert nc[0] == 0
        # PU 2: lost PU 1, still has PU 0 (not selected) → 1 from PU 2
        # Wait — PU 2 (idx=2) neighbors are PU 1 (idx=1) and PU 3 (idx=3)
        # PU 1 removed, PU 3 not selected → nc[2] should be 0
        assert nc[2] == 0

    def test_update_sequence_consistency(self):
        """Sequential flips keep state consistent with init_state."""
        problem = _make_grid_problem()
        cache = ProblemCache.from_problem(problem)
        c = MinNeighborConstraint(min_neighbors=1)
        selected = np.array([False, False, False, False])
        state = c.init_state(problem, selected, cache)

        # Add PUs one by one: 0, 1, 2, 3
        for idx in range(4):
            selected[idx] = True
            c.update_state(idx, selected, cache=cache, state=state)

        # Should match fresh init_state with all selected
        fresh = c.init_state(
            problem, np.array([True, True, True, True]), cache
        )
        np.testing.assert_array_equal(
            state["neighbor_count"], fresh["neighbor_count"]
        )


class TestNoBoundary:
    def test_no_neighbors_all_violated(self):
        """No boundary data → every selected PU violates min_neighbors."""
        problem = _make_no_boundary_problem()
        c = MinNeighborConstraint(min_neighbors=1, penalty_weight=10.0)
        selected = np.array([True, True, False, False])
        result = c.evaluate(problem, selected)
        assert not result.satisfied
        assert result.violation == 2.0
        assert c.penalty(problem, selected) == 20.0

    def test_no_neighbors_delta(self):
        """Delta correct when no neighbors exist."""
        problem = _make_no_boundary_problem()
        cache = ProblemCache.from_problem(problem)
        c = MinNeighborConstraint(min_neighbors=1, penalty_weight=10.0)

        for init_mask in range(16):
            selected = np.array(
                [(init_mask >> i) & 1 == 1 for i in range(4)]
            )
            for flip_idx in range(4):
                sel = selected.copy()
                state = c.init_state(problem, sel, cache)
                pen_before = c.penalty(problem, sel)
                delta = c.compute_delta(flip_idx, sel, state, cache)
                sel[flip_idx] = not sel[flip_idx]
                pen_after = c.penalty(problem, sel)
                expected = pen_after - pen_before
                assert abs(delta - expected) < 1e-10


class TestName:
    def test_name_format(self):
        c = MinNeighborConstraint(min_neighbors=3)
        assert c.name() == "MinNeighbor(3)"
