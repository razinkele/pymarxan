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
