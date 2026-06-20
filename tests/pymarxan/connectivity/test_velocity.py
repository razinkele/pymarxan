"""Climate velocity (Burrows et al. 2014) — spatial gradient + velocity raster.

Climate velocity = local temporal climate trend / spatial climate gradient
(the speed a species must move to track its niche). The spatial gradient uses
the Horn (1981) 3×3 method (as in Burrows 2014 / the VoCC package).
"""
from __future__ import annotations

import numpy as np
import pytest

from pymarxan.connectivity.velocity import climate_velocity, spatial_gradient


def _interior(a):
    """Drop the 1-cell border (edge cells use replicate-padding)."""
    return a[1:-1, 1:-1]


# --- spatial_gradient --------------------------------------------------


def test_west_east_gradient_magnitude_is_unity():
    # climate increases by 1 per cell west→east; gradient magnitude == 1
    climate = np.tile(np.arange(5.0), (5, 1))  # climate[i, j] = j
    mag, _ = spatial_gradient(climate, cell_size=1.0)
    assert np.allclose(_interior(mag), 1.0)


def test_flat_climate_has_zero_gradient():
    mag, _ = spatial_gradient(np.full((5, 5), 7.0), cell_size=1.0)
    assert np.allclose(mag, 0.0)


def test_diagonal_gradient_magnitude():
    # climate[i, j] = i + j → gradient magnitude sqrt(2)
    ii, jj = np.mgrid[0:5, 0:5]
    mag, _ = spatial_gradient((ii + jj).astype(float), cell_size=1.0)
    assert np.allclose(_interior(mag), np.sqrt(2.0))


def test_cell_size_scales_gradient_inversely():
    climate = np.tile(np.arange(5.0), (5, 1))
    mag1, _ = spatial_gradient(climate, cell_size=1.0)
    mag2, _ = spatial_gradient(climate, cell_size=2.0)
    assert np.allclose(_interior(mag2), _interior(mag1) / 2.0)


def test_rejects_non_2d():
    with pytest.raises(ValueError, match="2D"):
        spatial_gradient(np.arange(5.0), cell_size=1.0)


def test_rejects_bad_cell_size():
    with pytest.raises(ValueError, match="cell_size"):
        spatial_gradient(np.zeros((3, 3)), cell_size=0.0)


# --- climate_velocity --------------------------------------------------


def test_velocity_is_trend_over_gradient():
    climate = np.tile(np.arange(5.0), (5, 1))  # gradient 1 per cell
    vel = climate_velocity(climate, temporal_trend=2.0, cell_size=1.0)
    assert np.allclose(_interior(vel), 2.0)  # (2 °C/yr) / (1 °C/cell) = 2 cell/yr


def test_velocity_uses_distance_units():
    climate = np.tile(np.arange(5.0), (5, 1))
    # cell_size 2 → gradient 0.5 °C/dist → velocity 2/0.5 = 4 dist/yr
    vel = climate_velocity(climate, temporal_trend=2.0, cell_size=2.0)
    assert np.allclose(_interior(vel), 4.0)


def test_flat_climate_gives_infinite_velocity():
    vel = climate_velocity(np.full((5, 5), 3.0), temporal_trend=1.0, cell_size=1.0)
    assert np.isinf(vel).all()


def test_max_velocity_caps_infinities():
    vel = climate_velocity(
        np.full((5, 5), 3.0), temporal_trend=1.0, cell_size=1.0, max_velocity=100.0
    )
    assert np.allclose(vel, 100.0)


def test_per_cell_trend_array():
    climate = np.tile(np.arange(5.0), (5, 1))  # gradient 1
    trend = np.full((5, 5), 3.0)
    trend[2, 2] = 6.0
    vel = climate_velocity(climate, temporal_trend=trend, cell_size=1.0)
    assert vel[2, 2] == pytest.approx(6.0)
    assert vel[2, 1] == pytest.approx(3.0)


def test_velocity_magnitude_is_non_negative_for_cooling():
    # negative (cooling) trend still yields a non-negative speed magnitude
    climate = np.tile(np.arange(5.0), (5, 1))
    vel = climate_velocity(climate, temporal_trend=-2.0, cell_size=1.0)
    assert np.allclose(_interior(vel), 2.0)
    assert (vel >= 0).all()
