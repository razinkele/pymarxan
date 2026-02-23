"""Tests for cost surface processing."""
from __future__ import annotations

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import box

from pymarxan.spatial.cost_surface import apply_cost_from_vector, combine_cost_layers


def _make_pus():
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


class TestApplyCostFromVector:
    def test_full_coverage_assigns_cost(self):
        pus = _make_pus()
        cost_layer = gpd.GeoDataFrame(
            {"cost_val": [10.0]},
            geometry=[box(0, 0, 2, 2)],
            crs="EPSG:4326",
        )
        result = apply_cost_from_vector(pus, cost_layer, cost_column="cost_val")
        assert all(result["cost"] == 10.0)

    def test_partial_coverage_area_weighted(self):
        pus = _make_pus()
        # Cost layer covers left half only
        cost_layer = gpd.GeoDataFrame(
            {"cost_val": [20.0]},
            geometry=[box(0, 0, 1, 2)],
            crs="EPSG:4326",
        )
        result = apply_cost_from_vector(
            pus,
            cost_layer,
            cost_column="cost_val",
            aggregation="area_weighted_mean",
        )
        # PU 1 and 3 (fully covered) should have cost 20
        assert result.loc[result["id"] == 1, "cost"].iloc[0] == pytest.approx(
            20.0, abs=0.1
        )
        # PU 2 and 4 (not covered) should keep original cost
        assert result.loc[result["id"] == 2, "cost"].iloc[0] == pytest.approx(
            1.0, abs=0.1
        )

    def test_does_not_mutate_input(self):
        pus = _make_pus()
        cost_layer = gpd.GeoDataFrame(
            {"cost_val": [10.0]},
            geometry=[box(0, 0, 2, 2)],
            crs="EPSG:4326",
        )
        apply_cost_from_vector(pus, cost_layer, cost_column="cost_val")
        assert all(pus["cost"] == 1.0)


class TestCombineCostLayers:
    def test_equal_weight_combination(self):
        pus = _make_pus()
        layer1 = np.array([10.0, 20.0, 30.0, 40.0])
        layer2 = np.array([40.0, 30.0, 20.0, 10.0])
        result = combine_cost_layers(
            pus,
            layers=[("layer1", layer1), ("layer2", layer2)],
        )
        # Normalized: both sum to ~1.0 per element with equal weights
        costs = result["cost"].values
        assert np.allclose(costs, costs[0], atol=0.01)

    def test_weighted_combination(self):
        pus = _make_pus()
        layer1 = np.array([0.0, 0.0, 0.0, 0.0])
        layer2 = np.array([10.0, 10.0, 10.0, 10.0])
        result = combine_cost_layers(
            pus,
            layers=[("zero", layer1), ("ten", layer2)],
            weights=[0.0, 1.0],
        )
        # Only layer2 matters, all equal -> all same cost
        assert len(set(result["cost"].round(4).tolist())) == 1

    def test_single_layer(self):
        pus = _make_pus()
        layer = np.array([5.0, 10.0, 15.0, 20.0])
        result = combine_cost_layers(pus, layers=[("costs", layer)])
        # Single layer normalized to [0, 0.33, 0.67, 1.0]
        assert result["cost"].iloc[0] < result["cost"].iloc[-1]
