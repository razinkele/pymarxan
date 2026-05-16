"""ClumpState incremental delta — Phase 19 Batch 3 Task 8.

The bedrock contract:

    For ANY sequence of flips, ClumpState.delta_penalty(...) +
    ClumpState.apply_flip(...) produces a `held_effective` and
    accumulated penalty identical (within 1e-9) to a fresh
    `compute_clump_penalty_from_scratch(problem, selected)` call.

This is the same delta-matches-full property the Phase 18 cache test
pinned for PROBMODE 3, applied to the clumping path. If this test
passes, the SA inner loop has the right answer; if it fails, no other
clumping behaviour matters because the solver objective is wrong.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pymarxan.models.problem import ConservationProblem
from pymarxan.solvers.cache import ProblemCache
from pymarxan.solvers.clumping import (
    ClumpState,
    compute_clump_penalty_from_scratch,
)


def _line_graph_problem(
    n_pu: int = 8,
    target2: float = 25.0,
    clumptype: int = 0,
) -> ConservationProblem:
    """A linear-chain PU graph (1-2-3-...-n) with one type-4 feature.

    Each PU contributes 10 of the feature; target = 0.4·n·10 (so the
    target is reachable but TARGET2 matters).
    """
    pu = pd.DataFrame({
        "id": list(range(1, n_pu + 1)),
        "cost": [10.0] * n_pu,
        "status": [0] * n_pu,
    })
    features = pd.DataFrame({
        "id": [1],
        "name": ["sp"],
        "target": [round(0.4 * n_pu * 10, 1)],
        "spf": [1.0],
        "target2": [target2],
        "clumptype": [clumptype],
    })
    puvspr = pd.DataFrame({
        "species": [1] * n_pu,
        "pu": list(range(1, n_pu + 1)),
        "amount": [10.0] * n_pu,
    })
    boundary = pd.DataFrame({
        "id1": list(range(1, n_pu)),
        "id2": list(range(2, n_pu + 1)),
        "boundary": [1.0] * (n_pu - 1),
    })
    return ConservationProblem(
        planning_units=pu, features=features, pu_vs_features=puvspr,
        boundary=boundary,
    )


# --- from_selection initial state matches from-scratch ------------------


class TestClumpStateInit:

    def test_empty_selection(self):
        problem = _line_graph_problem(n_pu=5)
        cache = ProblemCache.from_problem(problem)
        selected = np.zeros(5, dtype=bool)
        state = ClumpState.from_selection(cache, selected)
        held = state.held_effective()
        scratch_held, _ = compute_clump_penalty_from_scratch(problem, selected)
        np.testing.assert_allclose(held, scratch_held, atol=1e-9)

    def test_full_selection(self):
        problem = _line_graph_problem(n_pu=5)
        cache = ProblemCache.from_problem(problem)
        selected = np.ones(5, dtype=bool)
        state = ClumpState.from_selection(cache, selected)
        held = state.held_effective()
        scratch_held, _ = compute_clump_penalty_from_scratch(problem, selected)
        np.testing.assert_allclose(held, scratch_held, atol=1e-9)

    def test_partial_selection_two_clumps(self):
        problem = _line_graph_problem(n_pu=6)
        cache = ProblemCache.from_problem(problem)
        # [T, T, F, F, T, T] → two 2-PU clumps of occ=20 each
        selected = np.array([True, True, False, False, True, True])
        state = ClumpState.from_selection(cache, selected)
        held = state.held_effective()
        scratch_held, _ = compute_clump_penalty_from_scratch(problem, selected)
        np.testing.assert_allclose(held, scratch_held, atol=1e-9)


# --- delta_penalty matches penalty_after - penalty_before ---------------


class TestClumpStateDelta:
    """delta_penalty must equal the difference of from-scratch penalties.

    The function is allowed to mutate internal scratch buffers but must
    leave the canonical state (comp_of, comp_occ, held_eff) unchanged
    so that subsequent calls without apply_flip stay consistent.
    """

    def test_delta_add_to_empty(self):
        problem = _line_graph_problem(n_pu=5)
        cache = ProblemCache.from_problem(problem)
        selected = np.zeros(5, dtype=bool)
        state = ClumpState.from_selection(cache, selected)

        # Compute scratch before & after for the same flip
        _, before = compute_clump_penalty_from_scratch(problem, selected)
        selected_after = selected.copy()
        selected_after[2] = True
        _, after = compute_clump_penalty_from_scratch(problem, selected_after)
        delta = state.delta_penalty(cache, idx=2, adding=True)
        assert delta == pytest.approx(after - before, abs=1e-9)

    def test_delta_remove_from_full(self):
        problem = _line_graph_problem(n_pu=5)
        cache = ProblemCache.from_problem(problem)
        selected = np.ones(5, dtype=bool)
        state = ClumpState.from_selection(cache, selected)

        _, before = compute_clump_penalty_from_scratch(problem, selected)
        selected_after = selected.copy()
        selected_after[2] = False
        _, after = compute_clump_penalty_from_scratch(problem, selected_after)
        delta = state.delta_penalty(cache, idx=2, adding=False)
        assert delta == pytest.approx(after - before, abs=1e-9)


# --- The bedrock test: delta-matches-full over a random flip sequence ---


class TestClumpStateBedrock:
    """For a long sequence of flips, sequenced delta_penalty + apply_flip
    must produce the same held_eff and accumulated penalty as a fresh
    from-scratch evaluation at every step.

    This is THE test for Phase 19 — analogous to Phase 18's PROBMODE 3
    cache delta-correctness test."""

    @pytest.mark.parametrize("clumptype", [0, 1, 2])
    def test_bedrock_50_flips(self, clumptype):
        problem = _line_graph_problem(n_pu=10, target2=25.0, clumptype=clumptype)
        cache = ProblemCache.from_problem(problem)
        rng = np.random.default_rng(clumptype * 7 + 1)
        selected = rng.random(cache.n_pu) > 0.5
        # Avoid all-true / all-false starting state for a more interesting walk
        if not selected.any():
            selected[0] = True
        state = ClumpState.from_selection(cache, selected)

        for step in range(50):
            idx = int(rng.integers(cache.n_pu))
            adding = not selected[idx]

            # Predicted delta via incremental path
            delta = state.delta_penalty(cache, idx, adding)

            # Reference: fresh from-scratch before & after
            _, ref_before = compute_clump_penalty_from_scratch(problem, selected)
            selected_after = selected.copy()
            selected_after[idx] = adding
            _, ref_after = compute_clump_penalty_from_scratch(
                problem, selected_after,
            )

            assert delta == pytest.approx(
                ref_after - ref_before, abs=1e-9,
            ), f"step {step}: incremental delta diverged from scratch"

            # Commit the flip and verify held_eff still matches
            state.apply_flip(cache, idx, adding)
            selected = selected_after
            scratch_held, _ = compute_clump_penalty_from_scratch(
                problem, selected,
            )
            np.testing.assert_allclose(
                state.held_effective(), scratch_held, atol=1e-9,
                err_msg=f"step {step}: held_effective drifted from scratch",
            )

    def test_bedrock_clumptype_0_two_features(self):
        """Multi-feature problem stresses the per-feature bookkeeping."""
        n_pu = 6
        pu = pd.DataFrame({
            "id": list(range(1, n_pu + 1)),
            "cost": [10.0] * n_pu,
            "status": [0] * n_pu,
        })
        features = pd.DataFrame({
            "id": [1, 2],
            "name": ["a", "b"],
            "target": [30.0, 20.0],
            "spf": [1.0, 2.0],
            "target2": [15.0, 10.0],
            "clumptype": [0, 0],
        })
        puvspr = pd.DataFrame({
            "species": [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2],
            "pu":      [1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6],
            "amount":  [10.0]*6 + [5.0]*6,
        })
        boundary = pd.DataFrame({
            "id1": list(range(1, n_pu)),
            "id2": list(range(2, n_pu + 1)),
            "boundary": [1.0] * (n_pu - 1),
        })
        problem = ConservationProblem(
            planning_units=pu, features=features, pu_vs_features=puvspr,
            boundary=boundary,
        )
        cache = ProblemCache.from_problem(problem)
        rng = np.random.default_rng(42)
        selected = rng.random(n_pu) > 0.5
        if not selected.any():
            selected[0] = True
        state = ClumpState.from_selection(cache, selected)

        for _ in range(30):
            idx = int(rng.integers(n_pu))
            adding = not selected[idx]
            delta = state.delta_penalty(cache, idx, adding)
            _, ref_before = compute_clump_penalty_from_scratch(problem, selected)
            selected_after = selected.copy()
            selected_after[idx] = adding
            _, ref_after = compute_clump_penalty_from_scratch(
                problem, selected_after,
            )
            assert delta == pytest.approx(ref_after - ref_before, abs=1e-9)
            state.apply_flip(cache, idx, adding)
            selected = selected_after


# --- Edge cases ---------------------------------------------------------


class TestClumpStateEdges:

    def test_non_clumping_problem_held_eff_is_raw_sum(self):
        """When no feature has target2 > 0, held_effective is just the
        raw selected amount per feature — same as `compute_held`."""
        problem = _line_graph_problem(n_pu=5, target2=0.0)
        cache = ProblemCache.from_problem(problem)
        selected = np.array([True, False, True, False, True])
        state = ClumpState.from_selection(cache, selected)
        held = state.held_effective()
        raw = cache.compute_held(selected)
        np.testing.assert_allclose(held, raw, atol=1e-9)

    def test_isolated_pu_flips_create_singleton_clump(self):
        """Adding a PU that has no selected neighbours creates a singleton.
        Under CLUMPTYPE 0 with target2 > singleton amount → no contribution."""
        problem = _line_graph_problem(n_pu=5, target2=15.0, clumptype=0)
        cache = ProblemCache.from_problem(problem)
        selected = np.zeros(5, dtype=bool)
        state = ClumpState.from_selection(cache, selected)
        # Add PU 0 alone → singleton occ=10 < target2=15 → CLUMPTYPE 0 contributes 0
        state.apply_flip(cache, idx=0, adding=True)
        held = state.held_effective()
        assert held[0] == 0.0
