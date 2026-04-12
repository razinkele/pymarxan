"""Boundary generation from planning unit geometry."""
from __future__ import annotations

import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString, MultiLineString


def compute_boundary(
    planning_units: gpd.GeoDataFrame,
    tolerance: float = 0.0,
) -> pd.DataFrame:
    """Compute boundary lengths between adjacent planning units.

    Returns a DataFrame with columns ``id1``, ``id2``, ``boundary``.
    For pairs where ``id1 != id2``, the boundary is the shared edge length.
    For self-boundary rows (``id1 == id2``), the boundary is the perimeter
    minus the sum of shared edges.

    Algorithm
    ---------
    1. Use the spatial index (STRtree) for candidate neighbor pairs.
    2. Compute intersection geometry between candidates.
    3. Filter to LineString/MultiLineString (shared edges).
    4. Measure length with optional snapping tolerance.

    Parameters
    ----------
    planning_units : gpd.GeoDataFrame
        Must have ``id`` and ``geometry`` columns.
    tolerance : float
        Snapping tolerance for geometry simplification before intersection.
        Use 0.0 (default) for no snapping.

    Returns
    -------
    pd.DataFrame
        Columns: ``id1``, ``id2``, ``boundary``.
    """
    geoms = planning_units.geometry.values
    ids = planning_units["id"].values.astype(int)
    sindex = planning_units.sindex

    if tolerance > 0:
        geoms = [g.simplify(tolerance, preserve_topology=True) for g in geoms]

    # Track shared edge lengths per PU for self-boundary computation
    shared_lengths: dict[int, float] = {int(pid): 0.0 for pid in ids}
    rows: list[dict] = []

    for i in range(len(planning_units)):
        candidates = list(sindex.intersection(geoms[i].bounds))
        for j in candidates:
            if j <= i:
                continue
            intersection = geoms[i].intersection(geoms[j])
            if intersection.is_empty:
                continue
            # Only count linear intersections (shared edges)
            if isinstance(intersection, (LineString, MultiLineString)):
                length = intersection.length
            elif hasattr(intersection, "geoms"):
                # GeometryCollection — extract linear parts
                length = sum(
                    g.length
                    for g in intersection.geoms
                    if isinstance(g, (LineString, MultiLineString))
                )
            else:
                continue

            if length > 1e-10:
                id_i, id_j = int(ids[i]), int(ids[j])
                rows.append({"id1": id_i, "id2": id_j, "boundary": length})
                shared_lengths[id_i] += length
                shared_lengths[id_j] += length

    # Self-boundary: perimeter - sum of shared edges
    for i in range(len(planning_units)):
        pid = int(ids[i])
        perimeter = geoms[i].length
        self_boundary = perimeter - shared_lengths[pid]
        if self_boundary > 1e-10:
            rows.append({"id1": pid, "id2": pid, "boundary": self_boundary})

    return pd.DataFrame(rows, columns=["id1", "id2", "boundary"])
