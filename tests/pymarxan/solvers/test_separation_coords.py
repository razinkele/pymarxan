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

import warnings

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Polygon

from pymarxan.models.grid import GridGeometry
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


# --- S4a: grid-cell-centroid fallback (raster PUs) -----------------------------

def _grid_problem(grid, sep=False):
    """A grid ConservationProblem (no geometry/xloc); optionally sep-active."""
    n = grid.n_pu
    pu = pd.DataFrame({"id": np.arange(1, n + 1), "cost": [1.0] * n, "status": [0] * n})
    feat = pd.DataFrame({"id": [1], "name": ["a"], "target": [1.0], "spf": [1.0]})
    if sep:
        feat["sepnum"] = [2]
        feat["sepdistance"] = [1.0]
    pvf = pd.DataFrame({"species": [1] * n, "pu": pu["id"].tolist(), "amount": [1.0] * n})
    return ConservationProblem(pu, feat, pvf, grid=grid)


def test_get_pu_coordinates_from_grid():
    """Tier 3 (new): no geometry, no xloc/yloc, but a grid → cell_centroids()."""
    grid = GridGeometry(0.0, 2.0, 1.0, 1.0, np.ones((2, 2), dtype=bool))  # 4 PUs
    coords = get_pu_coordinates(_grid_problem(grid))
    np.testing.assert_array_almost_equal(coords, grid.cell_centroids())


def test_geometry_beats_grid():
    grid = GridGeometry(0.0, 1.0, 1.0, 1.0, np.ones((1, 2), dtype=bool))  # 2 PUs
    geoms = [Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
             Polygon([(2, 2), (3, 2), (3, 3), (2, 3)])]
    pu = gpd.GeoDataFrame({"id": [1, 2], "cost": [1.0, 1.0], "status": [0, 0]},
                          geometry=geoms, crs="EPSG:3857")
    p = _make_problem(pu).copy_with(grid=grid)
    coords = get_pu_coordinates(p)
    np.testing.assert_array_almost_equal(coords, [[0.5, 0.5], [2.5, 2.5]])  # geometry, not grid


def test_xloc_beats_grid():
    grid = GridGeometry(0.0, 1.0, 1.0, 1.0, np.ones((1, 2), dtype=bool))
    pu = pd.DataFrame({"id": [1, 2], "cost": [1.0, 1.0], "status": [0, 0],
                       "xloc": [7.0, 8.0], "yloc": [70.0, 80.0]})
    p = _make_problem(pu).copy_with(grid=grid)
    coords = get_pu_coordinates(p)
    np.testing.assert_array_almost_equal(coords[:, 0], [7.0, 8.0])


def test_cache_builds_separation_on_grid():
    from pymarxan.solvers.cache import ProblemCache

    grid = GridGeometry(0.0, 3.0, 1.0, 1.0, np.ones((3, 3), dtype=bool), crs="EPSG:3035")
    cache = ProblemCache.from_problem(_grid_problem(grid, sep=True))  # must not raise
    assert cache.separation_active
    np.testing.assert_array_almost_equal(cache.pu_coords, grid.cell_centroids())


def test_geographic_grid_separation_warns():
    from pymarxan.solvers.cache import ProblemCache

    grid = GridGeometry(0.0, 3.0, 1.0, 1.0, np.ones((3, 3), dtype=bool), crs="EPSG:4326")
    with pytest.warns(UserWarning, match="geographic CRS"):
        ProblemCache.from_problem(_grid_problem(grid, sep=True))


def test_projected_grid_separation_no_geo_warning():
    from pymarxan.solvers.cache import ProblemCache

    grid = GridGeometry(0.0, 3.0, 1.0, 1.0, np.ones((3, 3), dtype=bool), crs="EPSG:3035")
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        ProblemCache.from_problem(_grid_problem(grid, sep=True))
    assert not any("geographic" in str(w.message) for w in rec)
