"""Tests for has_geometry detection."""
from __future__ import annotations

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point, box

from pymarxan.models.problem import ConservationProblem, has_geometry


def _make_features():
    return pd.DataFrame({
        "id": [1], "name": ["f1"], "target": [10.0], "spf": [1.0],
    })


def _make_puvspr():
    return pd.DataFrame({"species": [1], "pu": [1], "amount": [20.0]})


class TestHasGeometry:
    def test_plain_dataframe_no_geometry(self):
        p = ConservationProblem(
            planning_units=pd.DataFrame({
                "id": [1, 2], "cost": [1.0, 2.0], "status": [0, 0],
            }),
            features=_make_features(),
            pu_vs_features=_make_puvspr(),
        )
        assert has_geometry(p) is False

    def test_geodataframe_with_geometry(self):
        gdf = gpd.GeoDataFrame(
            {"id": [1, 2], "cost": [1.0, 2.0], "status": [0, 0]},
            geometry=[box(0, 0, 1, 1), box(1, 0, 2, 1)],
            crs="EPSG:4326",
        )
        p = ConservationProblem(
            planning_units=gdf,
            features=_make_features(),
            pu_vs_features=_make_puvspr(),
        )
        assert has_geometry(p) is True

    def test_geodataframe_empty_geometry(self):
        gdf = gpd.GeoDataFrame(
            {"id": [1], "cost": [1.0], "status": [0]},
            geometry=[Point()],  # empty geometry
            crs="EPSG:4326",
        )
        p = ConservationProblem(
            planning_units=gdf,
            features=_make_features(),
            pu_vs_features=_make_puvspr(),
        )
        assert has_geometry(p) is False
