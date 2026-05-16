"""Tests for compute_probability_penalty PROBMODE 0/1/2/3 dispatch.

Phase 18 Batch 2 Task 6: ``compute_probability_penalty`` in
``solvers/utils.py`` keeps its signature but routes internally based on
``problem.parameters["PROBMODE"]``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pymarxan.models.problem import ConservationProblem
from pymarxan.solvers.utils import compute_probability_penalty


def _problem_with_prob1d(prob_mode: int) -> ConservationProblem:
    """Build a small problem with PROB1D (per-PU probability)."""
    pu = pd.DataFrame({
        "id": [1, 2, 3], "cost": [10.0, 20.0, 30.0], "status": [0, 0, 0],
    })
    features = pd.DataFrame({
        "id": [1], "name": ["a"], "target": [5.0], "spf": [1.0],
    })
    puvspr = pd.DataFrame({
        "species": [1, 1], "pu": [1, 2], "amount": [3.0, 4.0],
    })
    probability = pd.DataFrame({
        "pu": [1, 2, 3], "probability": [0.1, 0.2, 0.3],
    })
    return ConservationProblem(
        planning_units=pu, features=features, pu_vs_features=puvspr,
        probability=probability,
        parameters={"PROBMODE": prob_mode, "PROBABILITYWEIGHTING": 1.0},
    )


def _problem_with_prob2d(prob_mode: int) -> ConservationProblem:
    """Build a small problem with PROB2D (per-cell `prob` on puvspr)
    and feature ptarget."""
    pu = pd.DataFrame({
        "id": [1, 2], "cost": [10.0, 20.0], "status": [0, 0],
    })
    features = pd.DataFrame({
        "id": [1], "name": ["a"], "target": [10.0], "spf": [2.0],
        "ptarget": [0.95],
    })
    puvspr = pd.DataFrame({
        "species": [1, 1], "pu": [1, 2], "amount": [8.0, 8.0],
        "prob": [0.1, 0.1],
    })
    return ConservationProblem(
        planning_units=pu, features=features, pu_vs_features=puvspr,
        parameters={"PROBMODE": prob_mode, "PROBABILITYWEIGHTING": 1.0},
    )


def test_probmode_0_returns_zero():
    """PROBMODE 0 = no probability term."""
    problem = _problem_with_prob1d(prob_mode=0)
    selected = np.array([True, True, False])
    pu_index = {1: 0, 2: 1, 3: 2}
    assert compute_probability_penalty(problem, selected, pu_index) == 0.0


def test_probmode_1_returns_risk_premium():
    """PROBMODE 1 unchanged: γ · Σ prob_i · cost_i · x_i."""
    problem = _problem_with_prob1d(prob_mode=1)
    selected = np.array([True, True, False])
    pu_index = {1: 0, 2: 1, 3: 2}
    # PU1: 0.1 * 10 = 1.0; PU2: 0.2 * 20 = 4.0; weight 1.0; total = 5.0
    expected = 1.0 * 10.0 * 0.1 + 1.0 * 20.0 * 0.2
    assert compute_probability_penalty(problem, selected, pu_index) == pytest.approx(expected)


def test_probmode_2_returns_zero():
    """PROBMODE 2 handles persistence-adjusted amounts upstream (in the
    feature matrix), so compute_probability_penalty contributes 0."""
    problem = _problem_with_prob1d(prob_mode=2)
    selected = np.array([True, True, False])
    pu_index = {1: 0, 2: 1, 3: 2}
    assert compute_probability_penalty(problem, selected, pu_index) == 0.0


def test_probmode_3_with_no_probability_data_returns_zero():
    """PROBMODE 3 but no `prob` column on puvspr -> deterministic case;
    every feature gets the zero-variance sentinel; P ≈ 1 ≥ ptarget so
    no penalty."""
    problem = _problem_with_prob2d(prob_mode=3)
    # Drop the prob column to simulate "deterministic" data
    problem = problem.copy_with(
        pu_vs_features=problem.pu_vs_features.drop(columns=["prob"]),
    )
    selected = np.array([True, True])
    pu_index = {1: 0, 2: 1}
    assert compute_probability_penalty(problem, selected, pu_index) == 0.0


def test_probmode_3_disabled_ptarget_returns_zero():
    """PROBMODE 3 with all features ptarget=-1 -> no probability penalty."""
    problem = _problem_with_prob2d(prob_mode=3)
    features = problem.features.copy()
    features["ptarget"] = -1.0
    problem = problem.copy_with(features=features)
    selected = np.array([True, True])
    pu_index = {1: 0, 2: 1}
    assert compute_probability_penalty(problem, selected, pu_index) == 0.0


def test_probmode_3_computes_zscore_penalty():
    """PROBMODE 3 returns the Marxan-faithful Z-score penalty when
    variance is non-zero and ptarget is active."""
    from scipy.stats import norm
    problem = _problem_with_prob2d(prob_mode=3)
    # Both PUs selected. amount=8 each, prob=0.1 each.
    # E[T] = sum(amount * (1-prob)) = 8*0.9 + 8*0.9 = 14.4
    # Var[T] = sum(amount² * prob * (1-prob)) = 64*0.1*0.9 + 64*0.1*0.9 = 11.52
    # target=10, so Z = (10 - 14.4) / sqrt(11.52) ≈ -1.2964
    # P = norm.sf(Z) ≈ 0.9025; ptarget=0.95; penalty = 2.0 * (0.95 - 0.9025) / 0.95
    selected = np.array([True, True])
    pu_index = {1: 0, 2: 1}

    e = 8.0 * 0.9 + 8.0 * 0.9
    v = 64.0 * 0.09 + 64.0 * 0.09
    z = (10.0 - e) / v ** 0.5
    p = norm.sf(z)
    expected = 2.0 * max(0.0, (0.95 - p) / 0.95) * 1.0  # weight = 1.0

    assert compute_probability_penalty(problem, selected, pu_index) == pytest.approx(
        expected, abs=1e-9
    )


def test_probmode_3_respects_probability_weighting():
    """PROBABILITYWEIGHTING scales the PROBMODE 3 penalty too — not just mode 1."""
    problem = _problem_with_prob2d(prob_mode=3)
    selected = np.array([True, True])
    pu_index = {1: 0, 2: 1}
    base = compute_probability_penalty(problem, selected, pu_index)

    problem_scaled = problem.copy_with(
        parameters={**problem.parameters, "PROBABILITYWEIGHTING": 3.0},
    )
    scaled = compute_probability_penalty(problem_scaled, selected, pu_index)
    assert scaled == pytest.approx(3.0 * base, abs=1e-9)


def test_probmode_3_partial_selection():
    """Z-score depends on which PUs are selected, not just how many."""
    problem = _problem_with_prob2d(prob_mode=3)
    pu_index = {1: 0, 2: 1}

    one = compute_probability_penalty(
        problem, np.array([True, False]), pu_index,
    )
    both = compute_probability_penalty(
        problem, np.array([True, True]), pu_index,
    )
    # Selecting more PUs reduces shortfall -> reduces penalty
    assert one >= both
