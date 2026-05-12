"""Tests for spatial export of solutions."""
from __future__ import annotations

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import box

from pymarxan.io.spatial_export import export_frequency_spatial, export_solution_spatial
from pymarxan.solvers.base import Solution


def _make_pus():
    return gpd.GeoDataFrame(
        {"id": [1, 2, 3, 4], "cost": [1.0, 2.0, 3.0, 4.0], "status": [0, 0, 0, 0]},
        geometry=[
            box(0, 0, 1, 1),
            box(1, 0, 2, 1),
            box(0, 1, 1, 2),
            box(1, 1, 2, 2),
        ],
        crs="EPSG:4326",
    )


def _make_solution(selected_mask):
    return Solution(
        selected=np.array(selected_mask, dtype=bool),
        cost=sum(c for c, s in zip([1, 2, 3, 4], selected_mask) if s),
        boundary=0.0,
        objective=0.0,
        targets_met={},
    )


@pytest.mark.spatial
class TestExportSolutionSpatial:
    def test_gpkg_roundtrip(self, tmp_path):
        pus = _make_pus()
        sol = _make_solution([True, False, True, False])
        out = tmp_path / "solution.gpkg"
        export_solution_spatial(pus, sol, out)

        result = gpd.read_file(out)
        assert "selected" in result.columns
        assert len(result) == 4
        assert list(result["selected"]) == [1, 0, 1, 0]

    def test_shapefile_roundtrip(self, tmp_path):
        pus = _make_pus()
        sol = _make_solution([False, True, False, True])
        out = tmp_path / "solution.shp"
        export_solution_spatial(pus, sol, out, driver="ESRI Shapefile")

        result = gpd.read_file(out)
        assert len(result) == 4
        assert list(result["selected"]) == [0, 1, 0, 1]

    def test_does_not_mutate_input(self, tmp_path):
        pus = _make_pus()
        sol = _make_solution([True, True, True, True])
        out = tmp_path / "solution.gpkg"
        export_solution_spatial(pus, sol, out)
        assert "selected" not in pus.columns


@pytest.mark.spatial
class TestExportFrequencySpatial:
    def test_frequency_calculation(self, tmp_path):
        pus = _make_pus()
        sols = [
            _make_solution([True, False, True, False]),
            _make_solution([True, True, False, False]),
            _make_solution([True, False, False, True]),
        ]
        out = tmp_path / "freq.gpkg"
        export_frequency_spatial(pus, sols, out)

        result = gpd.read_file(out)
        assert "frequency" in result.columns
        assert "count" in result.columns
        # PU 1 selected 3/3 times
        assert result.loc[result["id"] == 1, "frequency"].iloc[0] == pytest.approx(1.0)
        # PU 2 selected 1/3 times
        assert result.loc[result["id"] == 2, "frequency"].iloc[0] == pytest.approx(1 / 3, abs=0.01)
        # PU 3 selected 1/3 times
        assert result.loc[result["id"] == 3, "count"].iloc[0] == 1

    def test_empty_solutions(self, tmp_path):
        pus = _make_pus()
        out = tmp_path / "empty.gpkg"
        export_frequency_spatial(pus, [], out)

        result = gpd.read_file(out)
        assert all(result["frequency"] == 0.0)
        assert all(result["count"] == 0)

    def test_roundtrip_preserves_geometry(self, tmp_path):
        pus = _make_pus()
        sol = _make_solution([True, True, True, True])
        out = tmp_path / "geom.gpkg"
        export_frequency_spatial(pus, [sol], out)

        result = gpd.read_file(out)
        assert result.crs is not None
        for orig, read in zip(pus.geometry, result.geometry):
            assert orig.equals_exact(read, tolerance=1e-6)
