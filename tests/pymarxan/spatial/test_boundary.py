"""Tests for boundary generation from PU geometry."""
from __future__ import annotations

import geopandas as gpd
import pytest
from shapely.geometry import box

from pymarxan.spatial.boundary import compute_boundary


def _make_2x2_grid():
    """2x2 grid of 1x1 unit squares."""
    return gpd.GeoDataFrame(
        {"id": [1, 2, 3, 4]},
        geometry=[
            box(0, 0, 1, 1),  # bottom-left
            box(1, 0, 2, 1),  # bottom-right
            box(0, 1, 1, 2),  # top-left
            box(1, 1, 2, 2),  # top-right
        ],
        crs="EPSG:4326",
    )


@pytest.mark.spatial
class TestComputeBoundary:
    def test_shared_edges_2x2(self):
        pus = _make_2x2_grid()
        result = compute_boundary(pus)

        # Shared edges: 1-2 (bottom), 3-4 (top), 1-3 (left), 2-4 (right)
        shared = result[result["id1"] != result["id2"]]
        assert len(shared) == 4

        # Each shared edge should be length 1.0
        for _, row in shared.iterrows():
            assert row["boundary"] == pytest.approx(1.0, abs=0.01)

    def test_self_boundary_2x2(self):
        pus = _make_2x2_grid()
        result = compute_boundary(pus)

        self_rows = result[result["id1"] == result["id2"]]
        assert len(self_rows) == 4

        for _, row in self_rows.iterrows():
            # Perimeter = 4, each corner PU shares 2 edges of length 1
            # Self-boundary = 4 - 2 = 2
            assert row["boundary"] == pytest.approx(2.0, abs=0.01)

    def test_single_pu(self):
        pus = gpd.GeoDataFrame(
            {"id": [1]}, geometry=[box(0, 0, 1, 1)], crs="EPSG:4326"
        )
        result = compute_boundary(pus)
        # Only self-boundary = full perimeter = 4
        assert len(result) == 1
        assert result.iloc[0]["id1"] == 1
        assert result.iloc[0]["id2"] == 1
        assert result.iloc[0]["boundary"] == pytest.approx(4.0, abs=0.01)

    def test_linear_strip(self):
        """Three PUs in a row: 1-2-3."""
        pus = gpd.GeoDataFrame(
            {"id": [1, 2, 3]},
            geometry=[box(0, 0, 1, 1), box(1, 0, 2, 1), box(2, 0, 3, 1)],
            crs="EPSG:4326",
        )
        result = compute_boundary(pus)

        shared = result[result["id1"] != result["id2"]]
        assert len(shared) == 2  # 1-2 and 2-3

        # PU 1: perimeter=4, 1 shared edge => self=3
        self_1 = result[(result["id1"] == 1) & (result["id2"] == 1)]
        assert self_1.iloc[0]["boundary"] == pytest.approx(3.0, abs=0.01)

        # PU 2: perimeter=4, 2 shared edges => self=2
        self_2 = result[(result["id1"] == 2) & (result["id2"] == 2)]
        assert self_2.iloc[0]["boundary"] == pytest.approx(2.0, abs=0.01)

    def test_columns(self):
        pus = _make_2x2_grid()
        result = compute_boundary(pus)
        assert list(result.columns) == ["id1", "id2", "boundary"]

    def test_with_tolerance(self):
        pus = _make_2x2_grid()
        result = compute_boundary(pus, tolerance=0.01)
        shared = result[result["id1"] != result["id2"]]
        # Should still find 4 shared edges
        assert len(shared) == 4
