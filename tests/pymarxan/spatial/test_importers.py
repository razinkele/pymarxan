"""Tests for GIS file importers."""
from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import box

from pymarxan.spatial.importers import import_features_from_vector, import_planning_units

SPATIAL_DATA = Path(__file__).parent.parent.parent / "data" / "spatial"


class TestImportPlanningUnits:
    def test_import_geojson_with_column_mapping(self):
        gdf = import_planning_units(
            SPATIAL_DATA / "test_pus.geojson",
            id_column="pu_id",
            cost_column="pu_cost",
            status_column="lock",
        )
        assert isinstance(gdf, gpd.GeoDataFrame)
        assert len(gdf) == 3
        assert set(gdf.columns) >= {"id", "cost", "status", "geometry"}
        assert gdf["id"].tolist() == [1, 2, 3]
        assert gdf["cost"].tolist() == [5.0, 3.0, 7.0]
        assert gdf["status"].tolist() == [0, 2, 0]

    def test_import_defaults_missing_cost(self):
        gdf_src = gpd.GeoDataFrame(
            {"myid": [1, 2]},
            geometry=[box(0, 0, 1, 1), box(1, 0, 2, 1)],
            crs="EPSG:4326",
        )
        path = SPATIAL_DATA / "_temp_no_cost.geojson"
        gdf_src.to_file(path, driver="GeoJSON")
        try:
            gdf = import_planning_units(path, id_column="myid", cost_column="cost")
            assert all(gdf["cost"] == 1.0)
        finally:
            path.unlink(missing_ok=True)

    def test_import_defaults_missing_status(self):
        gdf_src = gpd.GeoDataFrame(
            {"id": [1, 2], "cost": [1.0, 2.0]},
            geometry=[box(0, 0, 1, 1), box(1, 0, 2, 1)],
            crs="EPSG:4326",
        )
        path = SPATIAL_DATA / "_temp_no_status.geojson"
        gdf_src.to_file(path, driver="GeoJSON")
        try:
            gdf = import_planning_units(path, status_column=None)
            assert all(gdf["status"] == 0)
        finally:
            path.unlink(missing_ok=True)

    def test_import_invalid_id_column_raises(self):
        with pytest.raises(ValueError, match="not found"):
            import_planning_units(
                SPATIAL_DATA / "test_pus.geojson",
                id_column="nonexistent",
            )

    def test_import_preserves_crs(self):
        gdf = import_planning_units(
            SPATIAL_DATA / "test_pus.geojson",
            id_column="pu_id",
            cost_column="pu_cost",
        )
        assert gdf.crs is not None
