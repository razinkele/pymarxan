"""Climate-refugia scoring — composes velocity (stability) + connectivity."""
from __future__ import annotations

import numpy as np
import pytest

from pymarxan.connectivity.refugia import refugia_score


def test_low_velocity_scores_higher():
    """Velocity-only: slower climate change → better refugium."""
    velocity = np.array([[1.0, 5.0], [10.0, 2.0]])
    score = refugia_score(velocity)
    assert score[0, 0] > score[1, 0]  # 1.0 km/yr beats 10.0 km/yr
    assert ((score >= 0) & (score <= 1)).all()


def test_connectivity_breaks_ties():
    """Two equally-stable cells: the better-connected one scores higher."""
    velocity = np.array([[1.0, 1.0], [1.0, 1.0]])      # uniform stability
    connectivity = np.array([[10.0, 1.0], [1.0, 1.0]])  # (0,0) best connected
    score = refugia_score(velocity, connectivity)
    assert score[0, 0] > score[0, 1]


def test_infinite_velocity_is_worst():
    """Flat-climate cells (inf velocity) are treated as the worst refugia."""
    velocity = np.array([[1.0, np.inf], [2.0, 3.0]])
    score = refugia_score(velocity)
    assert score[0, 1] == pytest.approx(score.min())
    assert np.isfinite(score).all()


def test_connectivity_weight_zero_ignores_connectivity():
    velocity = np.array([[1.0, 5.0], [10.0, 2.0]])
    conn = np.array([[1.0, 9.0], [9.0, 1.0]])
    s_vonly = refugia_score(velocity)
    s_weighted = refugia_score(
        velocity, conn, velocity_weight=1.0, connectivity_weight=0.0
    )
    assert np.allclose(s_vonly, s_weighted)


def test_geometric_method_zeroes_disconnected_cells():
    velocity = np.array([[1.0, 1.0], [1.0, 1.0]])
    conn = np.array([[5.0, 1.0], [1.0, 1.0]])  # min-maxed → (1,1)/(0,1)/(1,0)=0
    score = refugia_score(velocity, conn, method="geometric")
    assert score[0, 1] == pytest.approx(0.0)
    assert score[0, 0] > 0.0


def test_score_in_unit_range():
    rng = np.random.default_rng(1)
    velocity = rng.random((6, 6)) * 20
    conn = rng.random((6, 6))
    score = refugia_score(velocity, conn)
    assert ((score >= 0) & (score <= 1)).all()


def test_shape_mismatch_raises():
    with pytest.raises(ValueError, match="shape"):
        refugia_score(np.ones((3, 3)), np.ones((2, 2)))


def test_rejects_unknown_method():
    with pytest.raises(ValueError, match="method"):
        refugia_score(np.ones((3, 3)), method="bogus")
