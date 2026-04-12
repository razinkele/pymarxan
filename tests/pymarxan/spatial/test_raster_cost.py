"""Tests for raster cost surface import."""
from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_bounds
from shapely.geometry import box

from pymarxan.spatial.cost_surface import apply_cost_from_raster


def _make_pus():
    """2x2 grid of 1x1 squares covering [0,2] x [0,2]."""
    return gpd.GeoDataFrame(
        {"id": [1, 2, 3, 4], "cost": [1.0, 1.0, 1.0, 1.0], "status": [0, 0, 0, 0]},
        geometry=[
            box(0, 0, 1, 1),
            box(1, 0, 2, 1),
            box(0, 1, 1, 2),
            box(1, 1, 2, 2),
        ],
        crs="EPSG:4326",
    )


def _write_test_raster(path: Path, data: np.ndarray, bounds=(0, 0, 2, 2), nodata=-9999):
    """Write a small GeoTIFF for testing."""
    height, width = data.shape
    transform = from_bounds(*bounds, width, height)
    with rasterio.open(
        str(path),
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype=data.dtype,
        crs="EPSG:4326",
        transform=transform,
        nodata=nodata,
    ) as dst:
        dst.write(data, 1)


@pytest.mark.spatial
class TestApplyCostFromRaster:
    def test_uniform_raster(self, tmp_path):
        pus = _make_pus()
        data = np.full((10, 10), 5.0, dtype=np.float32)
        raster_path = tmp_path / "uniform.tif"
        _write_test_raster(raster_path, data)

        result = apply_cost_from_raster(pus, raster_path, aggregation="mean")
        np.testing.assert_allclose(result["cost"].values, 5.0, atol=0.01)

    def test_gradient_raster_mean(self, tmp_path):
        pus = _make_pus()
        # Left half = 10, right half = 20
        data = np.zeros((10, 10), dtype=np.float32)
        data[:, :5] = 10.0
        data[:, 5:] = 20.0
        raster_path = tmp_path / "gradient.tif"
        _write_test_raster(raster_path, data)

        result = apply_cost_from_raster(pus, raster_path, aggregation="mean")
        # PU 1 (left-bottom) should be ~10, PU 2 (right-bottom) ~20
        assert result.loc[result["id"] == 1, "cost"].iloc[0] == pytest.approx(10.0, abs=1.0)
        assert result.loc[result["id"] == 2, "cost"].iloc[0] == pytest.approx(20.0, abs=1.0)

    def test_sum_aggregation(self, tmp_path):
        pus = _make_pus()
        data = np.ones((4, 4), dtype=np.float32)
        raster_path = tmp_path / "ones.tif"
        _write_test_raster(raster_path, data)

        result = apply_cost_from_raster(pus, raster_path, aggregation="sum")
        # Each PU covers 1/4 of 4x4 raster => 4 pixels each
        for cost in result["cost"].values:
            assert cost == pytest.approx(4.0, abs=0.01)

    def test_nodata_excluded(self, tmp_path):
        pus = _make_pus()
        data = np.full((10, 10), 100.0, dtype=np.float32)
        # Set bottom-left quadrant to nodata
        data[5:, :5] = -9999.0
        raster_path = tmp_path / "nodata.tif"
        _write_test_raster(raster_path, data, nodata=-9999)

        result = apply_cost_from_raster(pus, raster_path, aggregation="mean")
        # PU 2 (right-bottom) should still be 100
        assert result.loc[result["id"] == 2, "cost"].iloc[0] == pytest.approx(100.0, abs=1.0)

    def test_does_not_mutate_input(self, tmp_path):
        pus = _make_pus()
        data = np.full((4, 4), 50.0, dtype=np.float32)
        raster_path = tmp_path / "test.tif"
        _write_test_raster(raster_path, data)

        apply_cost_from_raster(pus, raster_path)
        assert all(pus["cost"] == 1.0)

    def test_invalid_aggregation(self, tmp_path):
        pus = _make_pus()
        data = np.ones((4, 4), dtype=np.float32)
        raster_path = tmp_path / "test.tif"
        _write_test_raster(raster_path, data)

        with pytest.raises(ValueError, match="Unknown aggregation"):
            apply_cost_from_raster(pus, raster_path, aggregation="invalid")
