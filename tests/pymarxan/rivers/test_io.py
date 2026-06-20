"""Phase D — river-network ingest (HydroRIVERS) + barrier snapping."""
from __future__ import annotations

import warnings

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import LineString, Point

from pymarxan.rivers import (
    RiverNetwork,
    dci_diadromous,
    from_hydrorivers,
    snap_barriers,
)

pytestmark = pytest.mark.spatial


def _chain_gdf(crs="EPSG:3857") -> gpd.GeoDataFrame:
    """3-reach chain along the x-axis. S1 (0->1) is the outlet (NEXT_DOWN 0);
    S2 (1->2) drains to S1; S3 (2->3) drains to S2."""
    return gpd.GeoDataFrame(
        {
            "HYRIV_ID": [1, 2, 3],
            "NEXT_DOWN": [0, 1, 2],
            "LENGTH_KM": [10.0, 10.0, 10.0],
            "geometry": [
                LineString([(0, 0), (1, 0)]),
                LineString([(1, 0), (2, 0)]),
                LineString([(2, 0), (3, 0)]),
            ],
        },
        crs=crs,
    )


def _barrier_points(crs="EPSG:3857") -> gpd.GeoDataFrame:
    """Two barrier points near the midpoints of S2 and S3."""
    return gpd.GeoDataFrame(
        {
            "id": [10, 11],
            "pass_up": [0.5, 0.5],
            "geometry": [Point(1.5, 0.0), Point(2.5, 0.0)],
        },
        crs=crs,
    )


# --- from_hydrorivers --------------------------------------------------


def test_from_hydrorivers_builds_chain():
    net = from_hydrorivers(_chain_gdf())
    assert net.n_segments == 3
    assert net.outlet == 1          # NEXT_DOWN 0 → outlet
    assert net.n_barriers == 0


def test_from_hydrorivers_retains_geometry_for_snapping():
    net = from_hydrorivers(_chain_gdf())
    assert isinstance(net.segments, gpd.GeoDataFrame)
    assert net.segments.geometry.notna().all()


def test_from_hydrorivers_custom_columns():
    gdf = _chain_gdf().rename(
        columns={"HYRIV_ID": "rid", "NEXT_DOWN": "to", "LENGTH_KM": "len_km"}
    )
    net = from_hydrorivers(gdf, id_col="rid", next_down="to", length="len_km")
    assert net.n_segments == 3
    assert net.outlet == 1


def test_from_hydrorivers_missing_column_errors():
    gdf = _chain_gdf().drop(columns=["NEXT_DOWN"])
    with pytest.raises(ValueError, match="NEXT_DOWN"):
        from_hydrorivers(gdf)


# --- snap_barriers -----------------------------------------------------


def test_snap_barriers_assigns_nearest_segment():
    net = snap_barriers(from_hydrorivers(_chain_gdf()), _barrier_points())
    assert net.n_barriers == 2
    seg_of = dict(zip(net.barriers["id"], net.barriers["segment"]))
    assert seg_of[10] == 2          # point at x=1.5 → segment S2
    assert seg_of[11] == 3          # point at x=2.5 → segment S3


def test_snap_barriers_end_to_end_dci():
    # snapped chain == the Phase A hand fixture (B on S2 & S3, p=0.5) → DCId 58.333
    net = snap_barriers(from_hydrorivers(_chain_gdf()), _barrier_points())
    assert dci_diadromous(net) == pytest.approx(58.3333333, abs=1e-4)


def test_snap_barriers_drops_beyond_tolerance():
    far = gpd.GeoDataFrame(
        {"id": [10], "pass_up": [0.5], "geometry": [Point(100.0, 100.0)]},
        crs="EPSG:3857",
    )
    with pytest.warns(UserWarning, match="tolerance"):
        net = snap_barriers(from_hydrorivers(_chain_gdf()), far, tolerance=1.0)
    assert net.n_barriers == 0


def test_snap_barriers_warns_on_one_sided_crs():
    pts = _barrier_points(crs=None)
    with pytest.warns(UserWarning, match="CRS"):
        snap_barriers(from_hydrorivers(_chain_gdf()), pts)


def test_snap_barriers_requires_geometry():
    # a RiverNetwork built from a plain DataFrame has no geometry
    plain = RiverNetwork(
        segments=pd.DataFrame(
            {"id": [1, 2], "length": [10.0, 10.0], "down_id": [-1, 1]}
        ),
        barriers=pd.DataFrame(
            {"id": [], "segment": [], "pass_up": [], "pass_down": []}
        ),
    )
    with pytest.raises(ValueError, match="geometry"):
        snap_barriers(plain, _barrier_points())


def test_snap_barriers_carries_optional_columns():
    pts = _barrier_points()
    pts["removal_cost"] = [2.0, 3.0]
    pts["status"] = [0, 0]
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # no spurious warnings on the clean path
        net = snap_barriers(from_hydrorivers(_chain_gdf()), pts)
    assert "removal_cost" in net.barriers.columns
    cost_of = dict(zip(net.barriers["id"], net.barriers["removal_cost"]))
    assert cost_of[10] == 2.0
