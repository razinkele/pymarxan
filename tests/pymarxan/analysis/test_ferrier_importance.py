"""Tests for Ferrier importance scores (Phase 22).

Ferrier et al. (2000) — *Biological Conservation* 93(3): 303-325 — defines
per-PU importance as the SPF-weighted sum of the PU's contribution to
meeting each feature target. The metric is a closed-form score; no
re-solving needed.

Formula per PU ``i`` (Phase 22 implementation):
    ferrier_i = Σ_j SPF_j · min(amount_ij, target_j) / target_j

When ``target_j == 0`` the feature is skipped. Scores are non-negative
and unbounded above (proportional to SPF magnitudes).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pymarxan.analysis.ferrier_importance import compute_ferrier_importance
from pymarxan.models.problem import ConservationProblem


def _toy_problem():
    """4 PUs × 2 features; even amount distribution for predictable scores."""
    pu = pd.DataFrame({
        "id": [1, 2, 3, 4], "cost": [1.0] * 4, "status": [0] * 4,
    })
    features = pd.DataFrame({
        "id": [1, 2],
        "name": ["a", "b"],
        "target": [4.0, 8.0],
        "spf": [1.0, 2.0],
    })
    puvspr = pd.DataFrame({
        "species": [1, 1, 1, 1, 2, 2, 2, 2],
        "pu":      [1, 2, 3, 4, 1, 2, 3, 4],
        "amount":  [1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0],
    })
    return ConservationProblem(
        planning_units=pu, features=features, pu_vs_features=puvspr,
    )


def test_ferrier_returns_dict_keyed_by_pu_id():
    p = _toy_problem()
    scores = compute_ferrier_importance(p)
    assert isinstance(scores, dict)
    assert set(scores.keys()) == {1, 2, 3, 4}


def test_ferrier_score_is_spf_weighted_amount_over_target():
    """Hand computed:
    PU 1: feature_a contrib = 1·(1/4) = 0.25; feature_b contrib = 2·(2/8) = 0.5
          → total 0.75. Same for all four PUs by symmetry."""
    p = _toy_problem()
    scores = compute_ferrier_importance(p)
    assert scores[1] == pytest.approx(0.75, abs=1e-10)
    assert scores[2] == pytest.approx(0.75, abs=1e-10)
    assert scores[3] == pytest.approx(0.75, abs=1e-10)
    assert scores[4] == pytest.approx(0.75, abs=1e-10)


def test_ferrier_clamps_amount_at_target():
    """A single PU that supplies ≥ feature target gets credit clamped at 1·SPF.
    Otherwise a small reserve with a single very-rich PU would absorb all
    the importance, leaving the rest at zero."""
    pu = pd.DataFrame({
        "id": [1, 2], "cost": [1.0, 1.0], "status": [0, 0],
    })
    features = pd.DataFrame({
        "id": [1], "name": ["a"], "target": [5.0], "spf": [1.0],
    })
    # PU 1 supplies 100, PU 2 supplies 1. Without clamping, PU 1 would
    # score 100/5 = 20; with clamping it gets 1·SPF = 1.
    puvspr = pd.DataFrame({
        "species": [1, 1], "pu": [1, 2], "amount": [100.0, 1.0],
    })
    p = ConservationProblem(
        planning_units=pu, features=features, pu_vs_features=puvspr,
    )
    scores = compute_ferrier_importance(p)
    assert scores[1] == pytest.approx(1.0, abs=1e-10)
    assert scores[2] == pytest.approx(0.2, abs=1e-10)  # 1/5


def test_ferrier_skips_zero_target_features():
    """Features with target=0 don't contribute to anyone's score."""
    pu = pd.DataFrame({
        "id": [1], "cost": [1.0], "status": [0],
    })
    features = pd.DataFrame({
        "id": [1], "name": ["a"], "target": [0.0], "spf": [1.0],
    })
    puvspr = pd.DataFrame({
        "species": [1], "pu": [1], "amount": [10.0],
    })
    p = ConservationProblem(
        planning_units=pu, features=features, pu_vs_features=puvspr,
    )
    scores = compute_ferrier_importance(p)
    assert scores[1] == 0.0


def test_ferrier_locked_out_pus_score_zero():
    """Status==3 PUs can never be in a reserve, so their importance is 0."""
    p = _toy_problem()
    p.planning_units.loc[p.planning_units["id"] == 2, "status"] = 3
    scores = compute_ferrier_importance(p)
    assert scores[2] == 0.0
    # The other PUs keep their normal score.
    assert scores[1] == pytest.approx(0.75, abs=1e-10)


def test_ferrier_returns_floats():
    """All returned values must be plain Python floats so downstream
    dict-based exports / JSON serialisation just work."""
    p = _toy_problem()
    scores = compute_ferrier_importance(p)
    assert all(isinstance(v, float) for v in scores.values())
    assert all(np.isfinite(v) for v in scores.values())
