"""Tests for shared map helper."""
from __future__ import annotations

import geopandas as gpd
import pytest
from shapely.geometry import box

from pymarxan.models.geometry import generate_grid
from pymarxan_shiny.modules.mapping.map_utils import create_geo_map, create_grid_map


@pytest.fixture(autouse=True)
def _allow_widget_outside_session():
    """Temporarily remove shinywidgets session check so we can test map creation."""
    from ipywidgets import Widget

    original = Widget._widget_construction_callback
    Widget._widget_construction_callback = None
    yield
    Widget._widget_construction_callback = original


def test_create_grid_map_returns_map():
    """create_grid_map returns an ipyleaflet Map."""
    import ipyleaflet

    grid = generate_grid(4)
    colors = ["#ff0000", "#00ff00", "#0000ff", "#ffff00"]
    m = create_grid_map(grid, colors)
    assert isinstance(m, ipyleaflet.Map)


def test_create_grid_map_layer_count():
    """Map has one Rectangle per grid cell (plus base TileLayer)."""
    import ipyleaflet

    grid = generate_grid(6)
    colors = ["#aaaaaa"] * 6
    m = create_grid_map(grid, colors)
    rectangles = [
        layer for layer in m.layers
        if isinstance(layer, ipyleaflet.Rectangle)
    ]
    assert len(rectangles) == 6


def test_create_grid_map_auto_center():
    """Map auto-centers on grid midpoint when center not provided."""
    grid = generate_grid(4, origin=(10.0, 20.0), cell_size=0.01)
    colors = ["#000000"] * 4
    m = create_grid_map(grid, colors)
    # Grid spans from (10.0, 20.0) to (10.02, 20.02)
    # Midpoint should be approximately (10.01, 20.01)
    assert abs(m.center[0] - 10.01) < 0.01
    assert abs(m.center[1] - 20.01) < 0.01


def test_create_grid_map_custom_center():
    """Map uses provided center when given."""
    grid = generate_grid(4)
    colors = ["#000000"] * 4
    m = create_grid_map(grid, colors, center=(50.0, 10.0))
    assert list(m.center) == [50.0, 10.0]


def test_create_grid_map_empty_grid():
    """Empty grid produces Map with no Rectangle layers."""
    import ipyleaflet

    m = create_grid_map([], [])
    rectangles = [
        layer for layer in m.layers
        if isinstance(layer, ipyleaflet.Rectangle)
    ]
    assert len(rectangles) == 0


class TestCreateGeoMap:
    """Tests for create_geo_map — renders GeoDataFrame polygons."""

    def test_creates_map_from_geodataframe(self):
        gdf = gpd.GeoDataFrame(
            {"id": [1, 2, 3]},
            geometry=[box(0, 0, 1, 1), box(1, 0, 2, 1), box(0, 1, 1, 2)],
            crs="EPSG:4326",
        )
        colors = ["#ff0000", "#00ff00", "#0000ff"]
        m = create_geo_map(gdf, colors)
        import ipyleaflet

        assert isinstance(m, ipyleaflet.Map)

    def test_map_has_correct_number_of_geojson_layers(self):
        import ipyleaflet

        gdf = gpd.GeoDataFrame(
            {"id": [1, 2]},
            geometry=[box(0, 0, 1, 1), box(1, 0, 2, 1)],
            crs="EPSG:4326",
        )
        colors = ["#ff0000", "#00ff00"]
        m = create_geo_map(gdf, colors)
        geo_layers = [
            layer for layer in m.layers if isinstance(layer, ipyleaflet.GeoJSON)
        ]
        assert len(geo_layers) == 2

    def test_plain_dataframe_raises_clear_error(self):
        """A non-spatial (Marxan-format) project has a plain DataFrame with no
        geometry; create_geo_map must raise a clear ValueError instead of an
        opaque ``AttributeError: 'DataFrame' object has no attribute 'crs'``."""
        import pandas as pd

        df = pd.DataFrame({"id": [1, 2], "cost": [1.0, 2.0]})
        with pytest.raises(ValueError, match="geometry|GeoDataFrame"):
            create_geo_map(df, ["#ff0000", "#00ff00"])

    def test_reprojects_projected_crs_to_wgs84(self):
        """Projected CRS (e.g. UTM, metres) must be reprojected to lat/lon.

        ipyleaflet renders all coordinates as geographic. Without
        reprojection, a UTM polygon's metre-valued coordinates end up as
        absurd lat/lon (off the coast of Africa or beyond), making the map
        unusable for any imported or generated non-geographic data.
        """
        # UTM zone 33N polygon around Berlin (approx 13°E, 52.5°N)
        berlin_utm = box(390_000, 5_800_000, 400_000, 5_810_000)
        gdf = gpd.GeoDataFrame(
            {"id": [1]}, geometry=[berlin_utm], crs="EPSG:32633",
        )
        m = create_geo_map(gdf, ["#ff0000"])
        # Center should land near Berlin in geographic coordinates
        assert 50 < m.center[0] < 55, f"lat {m.center[0]} not near Berlin"
        assert 10 < m.center[1] < 15, f"lon {m.center[1]} not near Berlin"


# --- S4b: grid-aware map helpers ----------------------------------------------

import numpy as np  # noqa: E402

from pymarxan.models.problem import has_geometry, has_grid  # noqa: E402
from pymarxan.spatial.raster import from_arrays  # noqa: E402
from pymarxan_shiny.modules.mapping.map_utils import (  # noqa: E402
    _grid_bounds_latlon,
    build_pu_map,
    pu_centroids_latlon,
    too_large_for_map,
)


def _grid_problem(nrows=3, ncols=3, crs="EPSG:3035", x_min=4_321_000.0, y_max=3_210_000.0):
    return from_arrays(
        {1: np.ones((nrows, ncols))},
        x_min=x_min, y_max=y_max, cell_width=100.0, cell_height=100.0, crs=crs,
        include_boundary=False,
    )


def test_has_grid_predicate():
    p = _grid_problem()
    assert has_grid(p) and not has_geometry(p)
    import pandas as pd

    from pymarxan.models.problem import ConservationProblem
    plain = ConservationProblem(
        pd.DataFrame({"id": [1], "cost": [1.0], "status": [0]}),
        pd.DataFrame({"id": [1], "name": ["a"], "target": [1.0], "spf": [1.0]}),
        pd.DataFrame({"species": [1], "pu": [1], "amount": [1.0]}),
    )
    assert not has_grid(plain)


def test_too_large_for_map_none_safe():
    assert too_large_for_map(None) is False  # no project loaded -> no crash
    p = _grid_problem()
    assert too_large_for_map(p, max_cells=2) is True
    assert too_large_for_map(p, max_cells=100) is False


def test_grid_bounds_latlon_reprojects():
    p = _grid_problem()  # EPSG:3035, realistic European coords
    bounds = _grid_bounds_latlon(p.grid)
    assert len(bounds) == p.grid.n_pu
    for (s, w), (n, e) in bounds:
        assert s < n and w < e
        assert -90 <= s <= 90 and -180 <= w <= 180


def test_grid_bounds_latlon_none_crs_passthrough():
    p = _grid_problem(crs=None)
    bounds = _grid_bounds_latlon(p.grid)
    assert len(bounds) == p.grid.n_pu
    minx, miny, maxx, maxy = p.grid.cell_bounds()[0]
    assert bounds[0] == ((miny, minx), (maxy, maxx))


def test_build_pu_map_grid_branch():
    from ipyleaflet import Rectangle
    p = _grid_problem()
    m = build_pu_map(p, ["#ff0000"] * p.grid.n_pu)
    assert m is not None
    assert sum(isinstance(layer, Rectangle) for layer in m.layers) == p.grid.n_pu


def test_build_pu_map_cap():
    p = _grid_problem()  # 9 cells
    assert build_pu_map(p, ["#f00"] * 9, max_cells=2) is None
    assert build_pu_map(p, ["#f00"] * 9, max_cells=100) is not None


def test_build_pu_map_vector_branch():
    import pandas as pd
    from ipyleaflet import GeoJSON

    from pymarxan.models.problem import ConservationProblem
    pu = gpd.GeoDataFrame(
        {"id": [1, 2], "cost": [1.0, 1.0], "status": [0, 0]},
        geometry=[box(0, 0, 1, 1), box(1, 0, 2, 1)], crs="EPSG:4326",
    )
    p = ConservationProblem(
        pu, pd.DataFrame({"id": [1], "name": ["a"], "target": [1.0], "spf": [1.0]}),
        pd.DataFrame({"species": [1, 1], "pu": [1, 2], "amount": [1.0, 1.0]}),
    )
    m = build_pu_map(p, ["#f00", "#0f0"])
    assert m is not None and any(isinstance(x, GeoJSON) for x in m.layers)


def test_pu_centroids_latlon_grid():
    p = _grid_problem()
    cents = pu_centroids_latlon(p)
    assert len(cents) == p.grid.n_pu
    for lat, lon in cents:
        assert -90 <= lat <= 90 and -180 <= lon <= 180
