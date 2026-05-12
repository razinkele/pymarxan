"""Shared map helper for building ipyleaflet maps from grid data."""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import geopandas as gpd

try:
    import ipyleaflet

    _HAS_IPYLEAFLET = True
except ImportError:
    _HAS_IPYLEAFLET = False


def create_grid_map(
    grid: list[tuple[tuple[float, float], tuple[float, float]]],
    colors: list[str],
    center: tuple[float, float] | None = None,
    zoom: int = 12,
) -> ipyleaflet.Map:
    """Create ipyleaflet Map with colored Rectangle layers.

    Parameters
    ----------
    grid : list of ((south, west), (north, east)) tuples
        Bounding boxes for each planning unit.
    colors : list of hex color strings
        One color per grid cell (same length as grid).
    center : optional (lat, lon) tuple
        Map center. Auto-computed from grid midpoint if not provided.
    zoom : int, default 12
        Initial zoom level.

    Returns
    -------
    ipyleaflet.Map with one Rectangle per planning unit.

    Raises
    ------
    ImportError
        If ipyleaflet is not installed.
    """
    if not _HAS_IPYLEAFLET:
        raise ImportError("ipyleaflet is required for create_grid_map")

    if center is None and grid:
        all_lats = [b[0][0] for b in grid] + [b[1][0] for b in grid]
        all_lons = [b[0][1] for b in grid] + [b[1][1] for b in grid]
        center = (
            (min(all_lats) + max(all_lats)) / 2,
            (min(all_lons) + max(all_lons)) / 2,
        )
    elif center is None:
        center = (0.0, 0.0)

    m = ipyleaflet.Map(center=center, zoom=zoom)

    for bounds, color in zip(grid, colors):
        rect = ipyleaflet.Rectangle(
            bounds=bounds,
            color=color,
            fill_color=color,
            fill_opacity=0.7,
            weight=1,
        )
        m.add(rect)

    return m


def create_geo_map(
    gdf: gpd.GeoDataFrame,
    colors: list[str],
    center: tuple[float, float] | None = None,
    zoom: int = 10,
) -> ipyleaflet.Map:
    """Create ipyleaflet Map with GeoJSON layers from a GeoDataFrame.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        Must have a geometry column with Polygon geometries.
    colors : list[str]
        One hex color string per row in gdf.
    center : optional (lat, lon)
        Map center. Auto-computed from gdf bounds if not provided.
    zoom : int
        Initial zoom level.
    """
    if not _HAS_IPYLEAFLET:
        raise ImportError("ipyleaflet is required for create_geo_map")

    # ipyleaflet expects geographic coordinates (lat/lon). Projected CRSs
    # (UTM, equal-area, etc.) would otherwise render polygons at absurd
    # locations because their coordinates are in metres rather than degrees.
    if gdf.crs is not None and gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs("EPSG:4326")

    if center is None:
        b = gdf.total_bounds  # [minx, miny, maxx, maxy]
        center = ((b[1] + b[3]) / 2, (b[0] + b[2]) / 2)

    m = ipyleaflet.Map(center=center, zoom=zoom)

    for idx, (_, row) in enumerate(gdf.iterrows()):
        color = colors[idx] if idx < len(colors) else "#bcc9d1"
        geo_json = ipyleaflet.GeoJSON(
            data={
                "type": "Feature",
                "geometry": row.geometry.__geo_interface__,
                "properties": {},
            },
            style={
                "color": color,
                "fillColor": color,
                "fillOpacity": 0.7,
                "weight": 1,
            },
        )
        m.add(geo_json)

    return m
