"""ProblemCache PROB2D / PROBMODE 3 bookkeeping.

Phase 18 Batch 3 Task 7. Verifies the cache:

- Precomputes expected_matrix, var_matrix, feat_ptarget from the new
  optional columns (with sensible defaults when columns absent).
- compute_full_objective adds the PROBMODE 3 chance-constraint penalty.
- compute_delta_objective tracks variance and produces deltas that match
  the full-recompute difference (delta correctness — the bedrock SA
  contract that has to hold for the inner loop to be valid).
"""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest

from pymarxan.models.problem import ConservationProblem
from pymarxan.solvers.cache import ProblemCache


def _make_probmode3_problem(num_pus: int = 6, seed: int = 0) -> ConservationProblem:
    rng = np.random.default_rng(seed)
    pu = pd.DataFrame({
        "id": list(range(1, num_pus + 1)),
        "cost": list(rng.uniform(5.0, 20.0, num_pus)),
        "status": [0] * num_pus,
    })
    features = pd.DataFrame({
        "id": [1, 2],
        "name": ["f1", "f2"],
        "target": [12.0, 8.0],
        "spf": [1.0, 2.0],
        "ptarget": [0.95, 0.80],
    })
    rows = []
    for fid in (1, 2):
        for pu_id in range(1, num_pus + 1):
            rows.append({
                "species": fid,
                "pu": pu_id,
                "amount": float(rng.uniform(1.0, 5.0)),
                "prob": float(rng.uniform(0.0, 0.3)),
            })
    puvspr = pd.DataFrame(rows)
    return ConservationProblem(
        planning_units=pu, features=features, pu_vs_features=puvspr,
        parameters={"PROBMODE": 3, "PROBABILITYWEIGHTING": 1.0},
    )


class TestProbmode3CacheFields:
    """The cache exposes the PROB2D state needed by all PROBMODE 3 paths."""

    def test_probmode_stored(self):
        problem = _make_probmode3_problem()
        cache = ProblemCache.from_problem(problem)
        assert cache.probmode == 3

    def test_prob_weight_stored(self):
        problem = _make_probmode3_problem()
        problem.parameters["PROBABILITYWEIGHTING"] = 2.5
        cache = ProblemCache.from_problem(problem)
        assert cache.prob_weight == 2.5

    def test_expected_matrix_uses_amount_times_one_minus_prob(self):
        problem = _make_probmode3_problem(num_pus=2, seed=42)
        cache = ProblemCache.from_problem(problem)
        # For each (pu, feat): expected = amount * (1 - prob)
        puvspr = problem.pu_vs_features
        for _, row in puvspr.iterrows():
            pu_idx = cache.pu_id_to_idx[int(row["pu"])]
            feat_idx = cache.feat_id_to_col[int(row["species"])]
            amount = float(row["amount"])
            prob = float(row["prob"])
            assert cache.expected_matrix[pu_idx, feat_idx] == pytest.approx(
                amount * (1.0 - prob)
            )

    def test_var_matrix_is_bernoulli_form(self):
        problem = _make_probmode3_problem(num_pus=2, seed=42)
        cache = ProblemCache.from_problem(problem)
        # var = amount^2 * prob * (1 - prob)
        puvspr = problem.pu_vs_features
        for _, row in puvspr.iterrows():
            pu_idx = cache.pu_id_to_idx[int(row["pu"])]
            feat_idx = cache.feat_id_to_col[int(row["species"])]
            amount = float(row["amount"])
            prob = float(row["prob"])
            expected = amount ** 2 * prob * (1.0 - prob)
            assert cache.var_matrix[pu_idx, feat_idx] == pytest.approx(expected)

    def test_feat_ptarget_stored(self):
        problem = _make_probmode3_problem()
        cache = ProblemCache.from_problem(problem)
        np.testing.assert_array_almost_equal(
            cache.feat_ptarget, np.array([0.95, 0.80]),
        )

    def test_missing_prob_column_defaults_to_zero(self):
        """A problem without a `prob` column gets variance=0 throughout."""
        problem = _make_probmode3_problem()
        problem = problem.copy_with(
            pu_vs_features=problem.pu_vs_features.drop(columns=["prob"]),
        )
        cache = ProblemCache.from_problem(problem)
        np.testing.assert_array_equal(
            cache.var_matrix, np.zeros_like(cache.var_matrix),
        )
        # expected_matrix collapses to amount (since 1-0 = 1)
        np.testing.assert_array_almost_equal(
            cache.expected_matrix, cache.pu_feat_matrix,
        )


class TestProbmode3FullObjective:
    """compute_full_objective adds the PROBMODE 3 penalty on top of
    the deterministic objective."""

    def test_full_objective_includes_probability_penalty(self):
        problem = _make_probmode3_problem(num_pus=4, seed=7)
        cache = ProblemCache.from_problem(problem)
        rng = np.random.default_rng(7)
        selected = rng.random(cache.n_pu) > 0.5
        # Ensure something is selected (otherwise penalty path is degenerate)
        if not selected.any():
            selected[0] = True
        held = cache.compute_held(selected)

        # Same problem under PROBMODE 0 (no probability penalty)
        problem_det = copy.deepcopy(problem)
        problem_det.parameters["PROBMODE"] = 0
        cache_det = ProblemCache.from_problem(problem_det)
        held_det = cache_det.compute_held(selected)

        obj_3 = cache.compute_full_objective(selected, held, blm=0.0)
        obj_0 = cache_det.compute_full_objective(selected, held_det, blm=0.0)
        # PROBMODE 3 must add a non-negative term
        assert obj_3 >= obj_0 - 1e-9

    def test_full_objective_disabled_ptarget_no_penalty(self):
        """When every feature has ptarget=-1, PROBMODE 3 contributes 0."""
        problem = _make_probmode3_problem(num_pus=4, seed=7)
        features = problem.features.copy()
        features["ptarget"] = -1.0
        problem = problem.copy_with(features=features)
        cache = ProblemCache.from_problem(problem)
        selected = np.array([True, True, False, False])
        held = cache.compute_held(selected)

        problem_det = copy.deepcopy(problem)
        problem_det.parameters["PROBMODE"] = 0
        cache_det = ProblemCache.from_problem(problem_det)

        obj_3 = cache.compute_full_objective(selected, held, blm=0.0)
        obj_0 = cache_det.compute_full_objective(selected, held, blm=0.0)
        assert obj_3 == pytest.approx(obj_0, abs=1e-9)


class TestProbmode3Delta:
    """compute_delta_objective produces deltas that match the full-recompute
    difference — this is the bedrock SA inner-loop contract."""

    def test_delta_matches_full_under_probmode3(self):
        problem = _make_probmode3_problem(num_pus=8, seed=42)
        cache = ProblemCache.from_problem(problem)
        rng = np.random.default_rng(123)
        selected = rng.random(cache.n_pu) > 0.5
        if not selected.any():
            selected[0] = True
        held = cache.compute_held(selected)
        total_cost = float(np.sum(cache.costs[selected]))
        blm = 1.5

        for _ in range(10):
            flip_idx = int(rng.integers(cache.n_pu))
            full_before = cache.compute_full_objective(selected, held, blm)
            delta = cache.compute_delta_objective(
                flip_idx, selected, held, total_cost, blm,
            )
            # Apply the flip
            sign = -1.0 if selected[flip_idx] else 1.0
            held = held + sign * cache.pu_feat_matrix[flip_idx]
            total_cost += sign * cache.costs[flip_idx]
            selected = selected.copy()
            selected[flip_idx] = not selected[flip_idx]
            full_after = cache.compute_full_objective(selected, held, blm)
            assert delta == pytest.approx(full_after - full_before, abs=1e-7)
