"""Tests for dispersal-kernel distribution smoothing."""
from __future__ import annotations

import numpy as np
import pytest

from pymarxan.connectivity.smoothing import (
    distance_matrix_from_points,
    smooth_distribution,
)


def test_distance_matrix_is_symmetric_with_zero_diagonal():
    coords = np.array([[0.0, 0.0], [3.0, 4.0]])
    d = distance_matrix_from_points(coords)
    assert d.shape == (2, 2)
    assert d[0, 0] == pytest.approx(0.0)
    assert d[0, 1] == pytest.approx(5.0)  # 3-4-5 triangle
    assert d[1, 0] == pytest.approx(5.0)


def test_large_alpha_recovers_original_amounts():
    # With a very steep kernel, off-diagonal weights vanish and smoothing
    # is (near) the identity.
    amounts = np.array([5.0, 0.0, 0.0])
    distances = np.array(
        [[0.0, 1.0, 2.0], [1.0, 0.0, 1.0], [2.0, 1.0, 0.0]]
    )
    smoothed = smooth_distribution(amounts, distances, alpha=50.0)
    assert smoothed[0] == pytest.approx(5.0, abs=1e-6)
    assert smoothed[1] == pytest.approx(0.0, abs=1e-6)


def test_smoothing_is_mass_conserving_when_normalized():
    amounts = np.array([10.0, 0.0, 0.0])
    distances = np.array(
        [[0.0, 1.0, 2.0], [1.0, 0.0, 1.0], [2.0, 1.0, 0.0]]
    )
    smoothed = smooth_distribution(amounts, distances, alpha=0.5, normalize=True)
    assert smoothed.sum() == pytest.approx(10.0)
    # Mass spreads to neighbours, so the source loses some.
    assert smoothed[0] < 10.0
    assert smoothed[1] > 0.0


def test_unnormalized_accumulates_mass():
    amounts = np.array([1.0, 1.0])
    distances = np.array([[0.0, 1.0], [1.0, 0.0]])
    smoothed = smooth_distribution(amounts, distances, alpha=0.5, normalize=False)
    # raw K @ amounts with K_ij = exp(-0.5*d): each entry 1 + exp(-0.5) > 1
    assert smoothed[0] == pytest.approx(1.0 + np.exp(-0.5))


def test_alpha_must_be_positive():
    amounts = np.array([1.0])
    distances = np.array([[0.0]])
    with pytest.raises(ValueError, match="alpha"):
        smooth_distribution(amounts, distances, alpha=0.0)
