"""Tests for ``pymarxan.solvers.separation.get_pu_coordinates``.

Phase 20 Task 5: PU coordinate resolution with three-tier fallback:
(1) GeoDataFrame ``geometry.centroid``,
(2) ``xloc`` / ``yloc`` columns on planning_units,
(3) raise ``PUCoordinatesUnavailableError``.

Round-2 H3 NaN guard: empty or invalid geometries / NaN xloc/yloc → reject.
Round-3 M8: custom exception subclass lets ``build_solution`` catch only
the coordinates-unavailable case, not all ValueErrors.
"""
from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Polygon

from pymarxan.models.problem import ConservationProblem
from pymarxan.solvers.separation import (
    PUCoordinatesUnavailableError,
    get_pu_coordinates,
)


def _make_problem(planning_units: pd.DataFrame) -> ConservationProblem:
    """Minimal valid problem with the given planning_units."""
    n = len(planning_units)
    features = pd.DataFrame({
        "id": [1], "name": ["a"], "target": [1.0], "spf": [1.0],
    })
    puvspr = pd.DataFrame({
        "species": [1] * n,
        "pu": planning_units["id"].tolist(),
        "amount": [1.0] * n,
    })
    return ConservationProblem(
        planning_units=planning_units, features=features, pu_vs_features=puvspr,
    )


def test_get_pu_coordinates_from_geometry():
    """Tier 1: GeoDataFrame planning_units → centroid extraction."""
    geoms = [Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
             Polygon([(2, 2), (3, 2), (3, 3), (2, 3)])]
    pu = gpd.GeoDataFrame(
        {"id": [1, 2], "cost": [1.0, 1.0], "status": [0, 0]},
        geometry=geoms, crs="EPSG:3857",
    )
    p = _make_problem(pu)
    coords = get_pu_coordinates(p)
    assert coords.shape == (2, 2)
    np.testing.assert_array_almost_equal(coords, [[0.5, 0.5], [2.5, 2.5]])


def test_get_pu_coordinates_from_xloc_yloc():
    """Tier 2: plain DataFrame with xloc/yloc columns."""
    pu = pd.DataFrame({
        "id": [1, 2, 3], "cost": [1.0] * 3, "status": [0] * 3,
        "xloc": [10.0, 20.0, 30.0],
        "yloc": [100.0, 200.0, 300.0],
    })
    p = _make_problem(pu)
    coords = get_pu_coordinates(p)
    assert coords.shape == (3, 2)
    np.testing.assert_array_almost_equal(coords[:, 0], [10.0, 20.0, 30.0])
    np.testing.assert_array_almost_equal(coords[:, 1], [100.0, 200.0, 300.0])


def test_get_pu_coordinates_raises_when_unavailable():
    """Tier 3: no geometry, no xloc/yloc → PUCoordinatesUnavailableError."""
    pu = pd.DataFrame({"id": [1, 2], "cost": [1.0, 1.0], "status": [0, 0]})
    p = _make_problem(pu)
    with pytest.raises(PUCoordinatesUnavailableError):
        get_pu_coordinates(p)


def test_get_pu_coordinates_rejects_empty_geometry():
    """Round-2 H3: empty geometry produces NaN centroid; reject loudly."""
    geoms = [Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
             Polygon()]  # empty
    pu = gpd.GeoDataFrame(
        {"id": [1, 2], "cost": [1.0, 1.0], "status": [0, 0]},
        geometry=geoms, crs="EPSG:3857",
    )
    p = _make_problem(pu)
    with pytest.raises(PUCoordinatesUnavailableError, match="empty"):
        get_pu_coordinates(p)


def test_get_pu_coordinates_rejects_nan_xloc():
    """Round-2 H3: NaN xloc / yloc silently rejects every comparison; guard."""
    pu = pd.DataFrame({
        "id": [1, 2], "cost": [1.0, 1.0], "status": [0, 0],
        "xloc": [10.0, np.nan], "yloc": [100.0, 200.0],
    })
    p = _make_problem(pu)
    with pytest.raises(PUCoordinatesUnavailableError, match="NaN"):
        get_pu_coordinates(p)


def test_pu_coordinates_unavailable_error_is_valueerror():
    """Round-3 M8: build_solution must be able to catch a *specific* class.
    Subclassing ValueError keeps backward compat with broad except blocks."""
    assert issubclass(PUCoordinatesUnavailableError, ValueError)
