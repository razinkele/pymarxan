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
