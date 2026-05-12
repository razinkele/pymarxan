"""Tests for cost_surface performance fix -- correctness after O(n)->O(1) update."""
from __future__ import annotations

import geopandas as gpd
import numpy as np
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


class TestCombineWeightsValidation:
    def test_mismatched_weights_raises(self):
        """Passing 2 weights for 3 layers should raise ValueError."""
        from pymarxan.spatial.cost_surface import combine_cost_layers

        pu = gpd.GeoDataFrame({
            "id": [1, 2],
            "cost": [10.0, 20.0],
            "geometry": [box(0, 0, 1, 1), box(1, 0, 2, 1)],
        })
        layers = [
            ("a", np.array([1.0, 2.0])),
            ("b", np.array([3.0, 4.0])),
            ("c", np.array([5.0, 6.0])),
        ]
        with pytest.raises(ValueError, match="weights"):
            combine_cost_layers(pu, layers, weights=[0.5, 0.5])
