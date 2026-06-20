"""River-network ingest + barrier snapping (Phase D).

``from_hydrorivers`` builds a ``RiverNetwork`` straight from a HydroRIVERS /
NHDPlus-style GeoDataFrame using the downstream-pointer field (``NEXT_DOWN``);
the segment geometry is retained so barriers can later be snapped to it.
``snap_barriers`` assigns each barrier point to its nearest segment (the
barrier then sits at that segment's downstream end, per the model convention).

Pass a single complete basin to ``from_hydrorivers``: ``RiverNetwork`` requires
exactly one outlet, which is the reach whose ``NEXT_DOWN`` is 0 / NA (or points
outside the provided set). See design §9.
"""
from __future__ import annotations

import warnings

import geopandas as gpd
import pandas as pd

from pymarxan.rivers.network import RiverNetwork


def _empty_barriers() -> pd.DataFrame:
    return pd.DataFrame(
        {"id": [], "segment": [], "pass_up": [], "pass_down": []}
    )


def from_hydrorivers(
    gdf: gpd.GeoDataFrame,
    *,
    id_col: str = "HYRIV_ID",
    next_down: str = "NEXT_DOWN",
    length: str = "LENGTH_KM",
) -> RiverNetwork:
    """Build a (barrier-free) ``RiverNetwork`` from a river-reach GeoDataFrame.

    Parameters
    ----------
    gdf
        One row per river reach, with a unique id column, a downstream-pointer
        column (0 / NA at the outlet), a length column, and LineString geometry.
    id_col, next_down, length
        Column names (HydroRIVERS defaults).
    """
    for col in (id_col, next_down, length):
        if col not in gdf.columns:
            raise ValueError(f"column {col!r} not found in GeoDataFrame")
    segments = gdf.rename(
        columns={id_col: "id", next_down: "down_id", length: "length"}
    )
    return RiverNetwork(segments=segments, barriers=_empty_barriers())


def snap_barriers(
    network: RiverNetwork,
    barriers_gdf: gpd.GeoDataFrame,
    *,
    tolerance: float | None = None,
    id_col: str = "id",
    passability: str = "pass_up",
) -> RiverNetwork:
    """Snap barrier points to their nearest segment, returning a new network.

    Each barrier is assigned the id of the nearest segment (within ``tolerance``
    in the segments' CRS units, if given); barriers with no segment inside the
    tolerance are dropped with a warning. ``pass_up`` and any ``pass_down`` /
    ``removal_cost`` / ``status`` columns on ``barriers_gdf`` are carried over.
    """
    seg = network.segments
    if not isinstance(seg, gpd.GeoDataFrame) or seg.geometry.isna().all():
        raise ValueError(
            "network segments have no geometry; build the network via "
            "from_hydrorivers (or pass a GeoDataFrame of segments) before snapping"
        )

    pts = barriers_gdf
    if seg.crs is not None and pts.crs is not None and seg.crs != pts.crs:
        warnings.warn(
            f"CRS mismatch in snap_barriers: segments {seg.crs}, barriers "
            f"{pts.crs}. Reprojecting barriers to the segments' CRS.",
            UserWarning,
            stacklevel=2,
        )
        pts = pts.to_crs(seg.crs)
    elif (seg.crs is None) != (pts.crs is None):
        warnings.warn(
            "CRS only set on one of segments/barriers in snap_barriers; "
            "nearest-segment results may be meaningless. Set both CRSes.",
            UserWarning,
            stacklevel=2,
        )

    seg_geom = seg[["id", "geometry"]].rename(columns={"id": "_seg_id"})
    joined = gpd.sjoin_nearest(
        pts, seg_geom, how="left", max_distance=tolerance, distance_col="_dist"
    )
    # sjoin_nearest can emit duplicate rows for equidistant ties — keep one.
    joined = joined[~joined.index.duplicated(keep="first")]

    dropped = int(joined["_seg_id"].isna().sum())
    if dropped:
        warnings.warn(
            f"{dropped} barrier(s) had no segment within tolerance and were "
            "dropped from snap_barriers.",
            UserWarning,
            stacklevel=2,
        )
    kept = joined[joined["_seg_id"].notna()]

    out: dict[str, object] = {
        "id": kept[id_col].astype(int).to_numpy(),
        "segment": kept["_seg_id"].astype(int).to_numpy(),
        "pass_up": kept[passability].astype(float).to_numpy(),
    }
    out["pass_down"] = (
        kept["pass_down"].astype(float).to_numpy()
        if "pass_down" in barriers_gdf.columns
        else out["pass_up"]
    )
    for opt in ("removal_cost", "status"):
        if opt in barriers_gdf.columns:
            out[opt] = kept[opt].to_numpy()

    return RiverNetwork(segments=seg, barriers=pd.DataFrame(out))
