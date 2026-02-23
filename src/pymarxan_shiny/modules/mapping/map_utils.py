"""Shared map helper for building ipyleaflet maps from grid data."""
from __future__ import annotations

import ipyleaflet


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
    """
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
