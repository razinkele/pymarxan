"""Omniscape-style omnidirectional current-flow connectivity."""
from __future__ import annotations

import numpy as np
import pytest

from pymarxan.connectivity.omniscape import omniscape


def test_uniform_resistance_normalizes_to_one():
    """On a uniform-resistance landscape the cumulative current equals the
    flat-resistance null, so normalized current is ~1 everywhere — the key
    Omniscape invariant."""
    res = np.ones((7, 7))
    out = omniscape(res, radius=3)
    interior = out.normalized_current[1:-1, 1:-1]
    assert np.allclose(interior, 1.0, atol=0.15)


def test_corridor_is_a_pinch_point():
    """Two low-resistance regions split by a high-resistance wall with a single
    low-resistance gap: current funnels through the gap, so its normalized
    current exceeds both 1 and a neighbouring wall cell."""
    res = np.full((5, 5), 50.0)
    res[:, 0:2] = 1.0          # left region
    res[:, 3:5] = 1.0          # right region
    res[2, 2] = 1.0            # the gap in the wall (col 2)
    out = omniscape(res, radius=6)
    gap = out.normalized_current[2, 2]
    wall = out.normalized_current[0, 2]
    assert gap > 1.0
    assert gap > wall


def test_cumulative_current_non_negative_and_finite():
    rng = np.random.default_rng(0)
    res = 1.0 + rng.random((6, 6)) * 5.0
    out = omniscape(res, radius=3)
    assert out.cumulative_current.shape == res.shape
    assert np.all(out.cumulative_current >= 0)
    assert np.all(np.isfinite(out.cumulative_current))
    assert np.all(np.isfinite(out.flow_potential))


def test_rejects_non_2d():
    with pytest.raises(ValueError, match="2-?D|2 dimensions|two"):
        omniscape(np.ones(5), radius=2)


def test_rejects_non_positive_resistance():
    res = np.ones((4, 4))
    res[0, 0] = 0.0
    with pytest.raises(ValueError, match="positive"):
        omniscape(res, radius=2)


def test_rejects_bad_radius():
    with pytest.raises(ValueError, match="radius"):
        omniscape(np.ones((4, 4)), radius=0)
