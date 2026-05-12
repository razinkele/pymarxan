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

    def test_import_bare_shp_raises_clear_error(self, tmp_path):
        """A standalone .shp file (no .shx/.dbf sidecars) should error helpfully.

        Shiny's ``input_file`` delivers only the .shp without sidecars, so users
        must upload shapefiles as a ZIP archive. The error must point this out
        instead of surfacing a raw Fiona stack trace.
        """
        fake_shp = tmp_path / "test.shp"
        fake_shp.write_bytes(b"not a real shapefile")
        with pytest.raises(ValueError, match="zip|sidecar|\\.shx"):
            import_planning_units(fake_shp)

    def test_import_zipped_shapefile_works(self, tmp_path):
        """A ZIP containing a full shapefile bundle should be importable."""
        import zipfile

        # Write a real shapefile to disk
        gdf = gpd.GeoDataFrame(
            {"id": [1, 2], "cost": [1.0, 2.0], "status": [0, 0]},
            geometry=[box(0, 0, 1, 1), box(1, 0, 2, 1)],
            crs="EPSG:4326",
        )
        shp_dir = tmp_path / "shp_bundle"
        shp_dir.mkdir()
        gdf.to_file(shp_dir / "pus.shp", driver="ESRI Shapefile")

        # Zip the bundle (.shp + .shx + .dbf + .prj)
        zip_path = tmp_path / "pus.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            for f in shp_dir.iterdir():
                zf.write(f, arcname=f.name)

        result = import_planning_units(zip_path)
        assert len(result) == 2
        assert result["id"].tolist() == [1, 2]


class TestImportFeaturesFromVector:
    def _make_pus(self):
        return gpd.GeoDataFrame(
            {"id": [1, 2, 3]},
            geometry=[box(0, 0, 1, 1), box(1, 0, 2, 1), box(0, 1, 1, 2)],
            crs="EPSG:4326",
        )

    def test_import_with_area_overlap(self):
        pus = self._make_pus()
        df = import_features_from_vector(
            SPATIAL_DATA / "test_features.geojson",
            pus,
            feature_name="forest",
            feature_id=1,
        )
        assert isinstance(df, pd.DataFrame)
        assert set(df.columns) == {"species", "pu", "amount"}
        assert all(df["species"] == 1)
        assert all(df["amount"] > 0)

    def test_import_with_amount_column(self):
        pus = self._make_pus()
        df = import_features_from_vector(
            SPATIAL_DATA / "test_features.geojson",
            pus,
            feature_name="forest",
            feature_id=1,
            amount_column="area_ha",
        )
        assert all(df["amount"] > 0)

    def test_no_overlap_returns_empty(self):
        pus = gpd.GeoDataFrame(
            {"id": [1]},
            geometry=[box(100, 100, 101, 101)],
            crs="EPSG:4326",
        )
        df = import_features_from_vector(
            SPATIAL_DATA / "test_features.geojson",
            pus,
            feature_name="forest",
            feature_id=1,
        )
        assert len(df) == 0

    def test_import_features_handles_no_crs_on_pus(self, tmp_path):
        """import_features_from_vector must not crash when PU GDF has no CRS."""
        pu_gdf = gpd.GeoDataFrame(
            {"id": [1, 2]},
            geometry=[box(0, 0, 1, 1), box(1, 0, 2, 1)],
        )
        feat_gdf = gpd.GeoDataFrame(
            {"value": [10.0]},
            geometry=[box(0, 0, 1.5, 1)],
        )
        feat_path = tmp_path / "test_feat.geojson"
        feat_gdf.to_file(feat_path, driver="GeoJSON")

        result = import_features_from_vector(
            feat_path, pu_gdf, feature_name="test", feature_id=1,
        )
        assert isinstance(result, pd.DataFrame)
