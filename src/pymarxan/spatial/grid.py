"""Planning unit grid generation."""
from __future__ import annotations

import math

import geopandas as gpd
import pandas as pd
from shapely.geometry import Polygon, box
from shapely.geometry.base import BaseGeometry


def generate_planning_grid(
    bounds: tuple[float, float, float, float],
    cell_size: float,
    grid_type: str = "square",
    crs: str = "EPSG:4326",
    clip_to: BaseGeometry | None = None,
) -> gpd.GeoDataFrame:
    """Generate a planning unit grid as a GeoDataFrame.

    Parameters
    ----------
    bounds : tuple
        (minx, miny, maxx, maxy) bounding box.
    cell_size : float
        Width/height of each cell in CRS units.
    grid_type : str
        ``"square"`` or ``"hexagonal"``.
    crs : str
        Coordinate reference system (default EPSG:4326).
    clip_to : BaseGeometry or None
        Optional polygon to clip the grid to.

    Returns
    -------
    gpd.GeoDataFrame
        Columns: id (int), cost (float), status (int), geometry (Polygon).
    """
    if grid_type == "square":
        cells = _generate_square_cells(bounds, cell_size)
    elif grid_type == "hexagonal":
        cells = _generate_hex_cells(bounds, cell_size)
    else:
        raise ValueError(f"Unknown grid_type: {grid_type!r}. Use 'square' or 'hexagonal'.")

    if not cells:
        return gpd.GeoDataFrame(
            {"id": pd.array([], dtype="int64"), "cost": pd.array([], dtype="float64"), "status": pd.array([], dtype="int64")},
            geometry=[],
            crs=crs,
        )

    if clip_to is not None:
        cells = [c for c in cells if c.centroid.within(clip_to)]

    n = len(cells)
    return gpd.GeoDataFrame(
        {
            "id": list(range(1, n + 1)),
            "cost": [1.0] * n,
            "status": [0] * n,
        },
        geometry=cells,
        crs=crs,
    )


def _generate_square_cells(
    bounds: tuple[float, float, float, float],
    cell_size: float,
) -> list[Polygon]:
    minx, miny, maxx, maxy = bounds
    cells: list[Polygon] = []
    y = miny
    while y < maxy - 1e-10:
        x = minx
        while x < maxx - 1e-10:
            cells.append(box(x, y, x + cell_size, y + cell_size))
            x += cell_size
        y += cell_size
    return cells


def _generate_hex_cells(
    bounds: tuple[float, float, float, float],
    cell_size: float,
) -> list[Polygon]:
    minx, miny, maxx, maxy = bounds
    # Flat-top hexagon: width = cell_size, height = cell_size * sqrt(3)/2
    w = cell_size
    h = cell_size * math.sqrt(3) / 2
    cells: list[Polygon] = []
    row = 0
    y = miny
    while y < maxy - 1e-10:
        x_offset = (w / 2) if row % 2 == 1 else 0.0
        x = minx + x_offset
        while x < maxx - 1e-10:
            cells.append(_flat_top_hex(x, y, cell_size))
            x += w
        y += h
        row += 1
    return cells


def _flat_top_hex(cx: float, cy: float, size: float) -> Polygon:
    """Create a flat-top hexagon centered at (cx, cy)."""
    half = size / 2
    h = size * math.sqrt(3) / 4
    return Polygon([
        (cx - half, cy),
        (cx - half / 2, cy + h),
        (cx + half / 2, cy + h),
        (cx + half, cy),
        (cx + half / 2, cy - h),
        (cx - half / 2, cy - h),
    ])


def compute_adjacency(planning_units: gpd.GeoDataFrame) -> pd.DataFrame:
    """Compute boundary DataFrame from shared edges between adjacent PUs.

    Parameters
    ----------
    planning_units : gpd.GeoDataFrame
        Must have ``id`` and ``geometry`` columns.

    Returns
    -------
    pd.DataFrame
        Columns: id1, id2, boundary (shared edge length).
    """
    rows: list[dict] = []
    geoms = planning_units.geometry.values
    ids = planning_units["id"].values

    for i in range(len(planning_units)):
        for j in range(i + 1, len(planning_units)):
            if geoms[i].touches(geoms[j]) or (
                geoms[i].intersection(geoms[j]).length > 1e-10
            ):
                shared = geoms[i].intersection(geoms[j]).length
                if shared > 1e-10:
                    rows.append({
                        "id1": int(ids[i]),
                        "id2": int(ids[j]),
                        "boundary": shared,
                    })

    return pd.DataFrame(rows, columns=["id1", "id2", "boundary"])
