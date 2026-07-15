"""Shared map helper for building ipyleaflet maps from grid data."""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from pymarxan.models.problem import has_geometry, has_grid

if TYPE_CHECKING:
    import geopandas as gpd

try:
    import ipyleaflet

    _HAS_IPYLEAFLET = True
except ImportError:
    _HAS_IPYLEAFLET = False

_MAX_MAP_CELLS = 5000


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

    # A GeoDataFrame always exposes a `.crs` attribute (possibly None); a plain
    # pandas DataFrame does not. A non-spatial (classic Marxan-format) project
    # has no geometry, so callers must gate on `has_geometry(problem)` and fall
    # back to a synthetic grid. Fail loudly here rather than with an opaque
    # ``AttributeError: 'DataFrame' object has no attribute 'crs'``.
    if not hasattr(gdf, "crs"):
        raise ValueError(
            "create_geo_map requires a GeoDataFrame with geometry; got a plain "
            "DataFrame. The problem has no spatial geometry — gate on "
            "has_geometry(problem) and use create_grid_map for a synthetic grid."
        )

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


def compute_centroids(
    grid: list[tuple[tuple[float, float], tuple[float, float]]],
) -> list[tuple[float, float]]:
    """Center (lat, lon) of each bounding box. (Relocated from network_view.)"""
    return [((s + n) / 2, (w + e) / 2) for (s, w), (n, e) in grid]


def too_large_for_map(problem, max_cells: int = _MAX_MAP_CELLS) -> bool:
    """True when a grid problem has too many cells to render as rectangles.

    None-safe (``problem`` may be None when no project is loaded).
    """
    return (
        problem is not None
        and has_grid(problem)
        and problem.n_planning_units > max_cells
    )


def _latlon_transformer(crs):
    """A pyproj Transformer to EPSG:4326, or None (use raw coords) for None / already-4326 /
    unparseable CRS."""
    if crs is None:
        return None
    try:
        from pyproj import CRS, Transformer

        if CRS.from_user_input(crs).to_epsg() == 4326:
            return None
        return Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    except Exception:
        return None


def _grid_bounds_latlon(grid) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """``grid.cell_bounds()`` -> ``[((south, west), (north, east))]`` in EPSG:4326."""
    arr = np.asarray(grid.cell_bounds(), dtype=float)  # (n, 4): minx, miny, maxx, maxy
    minx, miny, maxx, maxy = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]
    tr = _latlon_transformer(grid.crs)
    if tr is not None:
        west, south = tr.transform(minx, miny)  # always_xy -> (lon, lat)
        east, north = tr.transform(maxx, maxy)
    else:
        west, south, east, north = minx, miny, maxx, maxy
    return [
        ((float(s), float(w)), (float(n), float(e)))
        for s, w, n, e in zip(south, west, north, east)
    ]


def build_pu_map(problem, colors: list[str], *, max_cells: int = _MAX_MAP_CELLS):
    """Base ipyleaflet map for a problem's PUs: vector geometry, raster grid, or synthetic
    grid. Returns None when ipyleaflet is missing, or a grid exceeds ``max_cells``.

    ``colors`` is one entry per PU in ``planning_units`` order — which, for a grid problem,
    coincides with ``cell_bounds()``/``cell_centroids()`` PU (row-major) order, so cells,
    centroids, and colors line up. (A grid source that didn't preserve that order would
    mis-color silently.)
    """
    if not _HAS_IPYLEAFLET:
        return None
    if has_geometry(problem):
        return create_geo_map(problem.planning_units, colors)
    if has_grid(problem):
        if problem.n_planning_units > max_cells:
            return None
        return create_grid_map(_grid_bounds_latlon(problem.grid), colors)
    from pymarxan.models.geometry import generate_grid

    return create_grid_map(generate_grid(problem.n_planning_units), colors)


def pu_centroids_latlon(problem) -> list[tuple[float, float]]:
    """(lat, lon) centroid per PU (for network overlays); same 3-way dispatch as build_pu_map."""
    if has_geometry(problem):
        pus = problem.planning_units
        if pus.crs is not None and pus.crs.to_epsg() != 4326:
            pus = pus.to_crs("EPSG:4326")
        return [(g.centroid.y, g.centroid.x) for g in pus.geometry]
    if has_grid(problem):
        cents = np.asarray(problem.grid.cell_centroids(), dtype=float)  # (n, 2): x, y
        x, y = cents[:, 0], cents[:, 1]
        tr = _latlon_transformer(problem.grid.crs)
        lon, lat = (tr.transform(x, y) if tr is not None else (x, y))
        return [(float(la), float(lo)) for la, lo in zip(lat, lon)]
    from pymarxan.models.geometry import generate_grid

    return compute_centroids(generate_grid(problem.n_planning_units))
