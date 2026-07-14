"""Tests for the Zonation SmoothingSpec (Phase C)."""
from __future__ import annotations

import numpy as np
import pytest

from pymarxan.connectivity.smoothing import distance_matrix_from_points
from pymarxan.zonation.smoothing import SmoothingSpec


def test_point_mass_spreads_monotonically():
    spec = SmoothingSpec(alpha=1.0, coords=np.array([[0.0], [1.0], [2.0]]))
    q = np.array([[10.0], [0.0], [0.0]])  # feature peaked on PU1
    sm = spec.apply(q)[:, 0]
    assert sm[0] > sm[1] > sm[2] > 0  # decays with distance from the peak
    assert sm.sum() == pytest.approx(10.0)  # total conserved


def test_total_conserved_per_feature():
    spec = SmoothingSpec(alpha=0.5, coords=np.array([[0.0], [1.0], [2.0], [3.0]]))
    q = np.array([[3.0, 1.0], [0.0, 2.0], [5.0, 0.0], [0.0, 4.0]])
    sm = spec.apply(q)
    assert sm[:, 0].sum() == pytest.approx(q[:, 0].sum())
    assert sm[:, 1].sum() == pytest.approx(q[:, 1].sum())


def test_distances_matches_coords():
    coords = np.array([[0.0], [1.0], [2.0]])
    q = np.array([[10.0], [0.0], [0.0]])
    d = distance_matrix_from_points(coords)
    by_coords = SmoothingSpec(alpha=1.0, coords=coords).apply(q)
    by_dist = SmoothingSpec(alpha=1.0, distances=d).apply(q)
    assert np.allclose(by_coords, by_dist)


def test_alpha_must_be_positive():
    with pytest.raises(ValueError, match="alpha"):
        SmoothingSpec(alpha=0.0, coords=np.array([[0.0]]))


def test_requires_exactly_one_of_coords_or_distances():
    with pytest.raises(ValueError, match="exactly one"):
        SmoothingSpec(alpha=1.0)  # neither
    with pytest.raises(ValueError, match="exactly one"):
        SmoothingSpec(
            alpha=1.0, coords=np.array([[0.0]]), distances=np.array([[0.0]])
        )  # both


def test_1d_coords_rejected_at_construction():
    with pytest.raises(ValueError, match="2-D"):
        SmoothingSpec(alpha=1.0, coords=np.array([0.0, 1.0, 2.0]))  # 1-D


def test_resolve_distances_validates_shape():
    spec = SmoothingSpec(alpha=1.0, distances=np.zeros((2, 2)))
    with pytest.raises(ValueError, match="distances must be"):
        spec.resolve_distances(3)  # wrong distances shape
    spec2 = SmoothingSpec(alpha=1.0, coords=np.array([[0.0], [1.0]]))
    with pytest.raises(ValueError, match="coords must have"):
        spec2.resolve_distances(3)  # 2-D coords, wrong row count


def test_single_pu_smooths_to_itself():
    spec = SmoothingSpec(alpha=1.0, coords=np.array([[0.0]]))
    assert spec.apply(np.array([[7.0]]))[0, 0] == pytest.approx(7.0)  # kernel [[1]]
