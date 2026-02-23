"""Tests for cost_surface performance fix -- correctness after O(n)->O(1) update."""
from __future__ import annotations

import numpy as np
import geopandas as gpd
import pytest
from shapely.geometry import box


class TestApplyCostPerf:
    def test_map_based_update_correct(self):
        """Verify cost update produces correct results after perf fix."""
        from pymarxan.spatial.cost_surface import apply_cost_from_vector

        pu = gpd.GeoDataFrame({
            "id": [1, 2, 3],
            "cost": [10.0, 20.0, 30.0],
            "geometry": [box(0, 0, 1, 1), box(1, 0, 2, 1), box(2, 0, 3, 1)],
        }, crs="EPSG:4326")
        cost_layer = gpd.GeoDataFrame({
            "value": [100.0, 200.0],
            "geometry": [box(0, 0, 1.5, 1), box(1.5, 0, 3, 1)],
        }, crs="EPSG:4326")
        result = apply_cost_from_vector(pu, cost_layer, "value")
        assert result.loc[result["id"] == 1, "cost"].iloc[0] == pytest.approx(100.0, rel=0.1)
        assert result.loc[result["id"] == 3, "cost"].iloc[0] == pytest.approx(200.0, rel=0.1)
