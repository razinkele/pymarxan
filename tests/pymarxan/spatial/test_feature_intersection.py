"""Tests for feature intersection."""
from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import from_bounds
from shapely.geometry import box

from pymarxan.spatial.feature_intersection import (
    intersect_raster_features,
    intersect_vector_features,
)


def _make_pus():
    """2x2 grid of 1x1 squares."""
    return gpd.GeoDataFrame(
        {"id": [1, 2, 3, 4]},
        geometry=[
            box(0, 0, 1, 1),
            box(1, 0, 2, 1),
            box(0, 1, 1, 2),
            box(1, 1, 2, 2),
        ],
        crs="EPSG:4326",
    )


def _write_test_raster(path: Path, data: np.ndarray, bounds=(0, 0, 2, 2)):
    height, width = data.shape
    transform = from_bounds(*bounds, width, height)
    with rasterio.open(
        str(path), "w", driver="GTiff",
        height=height, width=width, count=1,
        dtype=data.dtype, crs="EPSG:4326", transform=transform,
        nodata=-9999,
    ) as dst:
        dst.write(data, 1)


@pytest.mark.spatial
class TestIntersectVectorFeatures:
    def test_full_overlap_area(self):
        pus = _make_pus()
        feature = gpd.GeoDataFrame(
            geometry=[box(0, 0, 2, 2)], crs="EPSG:4326"
        )
        result = intersect_vector_features(pus, {1: feature}, method="area")
        assert len(result) == 4
        assert set(result["species"]) == {1}
        for _, row in result.iterrows():
            assert row["amount"] == pytest.approx(1.0, abs=0.01)

    def test_partial_overlap_area(self):
        pus = _make_pus()
        # Feature covers only left half
        feature = gpd.GeoDataFrame(
            geometry=[box(0, 0, 1, 2)], crs="EPSG:4326"
        )
        result = intersect_vector_features(pus, {1: feature}, method="area")
        covered = set(result["pu"])
        assert 1 in covered
        assert 3 in covered
        assert 2 not in covered

    def test_binary_method(self):
        pus = _make_pus()
        feature = gpd.GeoDataFrame(
            geometry=[box(0, 0, 1.5, 1.5)], crs="EPSG:4326"
        )
        result = intersect_vector_features(pus, {1: feature}, method="binary")
        for _, row in result.iterrows():
            assert row["amount"] == 1.0

    def test_proportion_method(self):
        pus = _make_pus()
        # Feature covers bottom-left quadrant half
        feature = gpd.GeoDataFrame(
            geometry=[box(0, 0, 0.5, 1)], crs="EPSG:4326"
        )
        result = intersect_vector_features(pus, {1: feature}, method="proportion")
        pu1 = result[result["pu"] == 1]
        assert len(pu1) == 1
        assert pu1.iloc[0]["amount"] == pytest.approx(0.5, abs=0.01)

    def test_multiple_features(self):
        pus = _make_pus()
        f1 = gpd.GeoDataFrame(geometry=[box(0, 0, 1, 1)], crs="EPSG:4326")
        f2 = gpd.GeoDataFrame(geometry=[box(1, 1, 2, 2)], crs="EPSG:4326")
        result = intersect_vector_features(pus, {10: f1, 20: f2}, method="binary")
        assert set(result["species"]) == {10, 20}

    def test_no_overlap(self):
        pus = _make_pus()
        feature = gpd.GeoDataFrame(
            geometry=[box(5, 5, 6, 6)], crs="EPSG:4326"
        )
        result = intersect_vector_features(pus, {1: feature}, method="area")
        assert len(result) == 0

    def test_invalid_method(self):
        pus = _make_pus()
        feature = gpd.GeoDataFrame(geometry=[box(0, 0, 1, 1)], crs="EPSG:4326")
        with pytest.raises(ValueError, match="Unknown method"):
            intersect_vector_features(pus, {1: feature}, method="bad")


@pytest.mark.spatial
class TestIntersectRasterFeatures:
    def test_uniform_raster(self, tmp_path):
        pus = _make_pus()
        data = np.full((10, 10), 3.0, dtype=np.float32)
        raster_path = tmp_path / "feat.tif"
        _write_test_raster(raster_path, data)

        result = intersect_raster_features(pus, {1: raster_path}, aggregation="sum")
        assert len(result) == 4
        for _, row in result.iterrows():
            assert row["amount"] > 0

    def test_multiple_rasters(self, tmp_path):
        pus = _make_pus()
        d1 = np.full((4, 4), 1.0, dtype=np.float32)
        d2 = np.full((4, 4), 2.0, dtype=np.float32)
        r1 = tmp_path / "f1.tif"
        r2 = tmp_path / "f2.tif"
        _write_test_raster(r1, d1)
        _write_test_raster(r2, d2)

        result = intersect_raster_features(pus, {1: r1, 2: r2}, aggregation="sum")
        assert set(result["species"]) == {1, 2}

    def test_invalid_aggregation(self, tmp_path):
        pus = _make_pus()
        data = np.ones((4, 4), dtype=np.float32)
        raster_path = tmp_path / "test.tif"
        _write_test_raster(raster_path, data)

        with pytest.raises(ValueError, match="Unknown aggregation"):
            intersect_raster_features(pus, {1: raster_path}, aggregation="bad")
