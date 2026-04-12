"""Tests for connectivity decay functions."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pymarxan.connectivity.decay import (
    apply_decay,
    inverse_power,
    negative_exponential,
    threshold,
)


class TestNegativeExponential:
    """Tests for negative_exponential decay function."""

    def test_at_zero_distance(self):
        """At distance=0, connectivity should be 1.0."""
        result = negative_exponential(np.array([0.0]), alpha=0.5)
        assert np.isclose(result[0], 1.0)

    def test_at_zero_distance_array(self):
        """Multiple zero distances should all give 1.0."""
        result = negative_exponential(np.array([0.0, 0.0, 0.0]), alpha=1.0)
        assert np.allclose(result, 1.0)

    def test_at_large_distance(self):
        """At large distance, connectivity should approach 0."""
        result = negative_exponential(np.array([100.0]), alpha=0.5)
        assert result[0] < 1e-10

    def test_monotone_decreasing(self):
        """Connectivity should decrease monotonically with distance."""
        distances = np.array([0.0, 1.0, 2.0, 5.0, 10.0])
        result = negative_exponential(distances, alpha=0.5)
        assert np.all(np.diff(result) < 0)

    def test_alpha_parameter_effect(self):
        """Larger alpha should decay faster."""
        distance = np.array([5.0])
        result_alpha_1 = negative_exponential(distance, alpha=0.1)
        result_alpha_2 = negative_exponential(distance, alpha=1.0)
        assert result_alpha_2[0] < result_alpha_1[0]

    def test_invalid_alpha_zero(self):
        """alpha=0 should raise ValueError."""
        with pytest.raises(ValueError, match="alpha must be positive"):
            negative_exponential(np.array([1.0]), alpha=0.0)

    def test_invalid_alpha_negative(self):
        """alpha<0 should raise ValueError."""
        with pytest.raises(ValueError, match="alpha must be positive"):
            negative_exponential(np.array([1.0]), alpha=-0.5)

    def test_formula_accuracy(self):
        """Test formula: exp(-alpha * distance)."""
        distances = np.array([1.0, 2.0, 3.0])
        alpha = 0.5
        result = negative_exponential(distances, alpha=alpha)
        expected = np.exp(-alpha * distances)
        assert np.allclose(result, expected)


class TestInversePower:
    """Tests for inverse_power decay function."""

    def test_at_zero_distance(self):
        """At distance=0, connectivity should be 1.0."""
        result = inverse_power(np.array([0.0]), beta=2.0)
        assert np.isclose(result[0], 1.0)

    def test_at_zero_distance_array(self):
        """Multiple zero distances should all give 1.0."""
        result = inverse_power(np.array([0.0, 0.0]), beta=1.0)
        assert np.allclose(result, 1.0)

    def test_at_large_distance(self):
        """At large distance, connectivity should approach 0."""
        result = inverse_power(np.array([1000.0]), beta=2.0)
        assert result[0] < 1e-5

    def test_monotone_decreasing(self):
        """Connectivity should decrease monotonically with distance."""
        distances = np.array([0.0, 1.0, 2.0, 5.0, 10.0])
        result = inverse_power(distances, beta=2.0)
        assert np.all(np.diff(result) < 0)

    def test_beta_parameter_effect(self):
        """Larger beta should decay faster (steeper curve)."""
        distance = np.array([2.0])
        result_beta_1 = inverse_power(distance, beta=1.0)
        result_beta_2 = inverse_power(distance, beta=2.0)
        assert result_beta_2[0] < result_beta_1[0]

    def test_invalid_beta_zero(self):
        """beta=0 should raise ValueError."""
        with pytest.raises(ValueError, match="beta must be positive"):
            inverse_power(np.array([1.0]), beta=0.0)

    def test_invalid_beta_negative(self):
        """beta<0 should raise ValueError."""
        with pytest.raises(ValueError, match="beta must be positive"):
            inverse_power(np.array([1.0]), beta=-1.0)

    def test_formula_accuracy(self):
        """Test formula: 1 / (1 + distance^beta)."""
        distances = np.array([1.0, 2.0, 3.0])
        beta = 1.5
        result = inverse_power(distances, beta=beta)
        expected = 1.0 / (1.0 + np.power(distances, beta))
        assert np.allclose(result, expected)


class TestThreshold:
    """Tests for threshold decay function."""

    def test_within_threshold(self):
        """Distance <= max_distance should give 1.0."""
        distances = np.array([0.5, 1.0, 2.0])
        result = threshold(distances, max_distance=3.0)
        assert np.allclose(result, 1.0)

    def test_exactly_at_threshold(self):
        """Distance exactly at max_distance should give 1.0."""
        result = threshold(np.array([5.0]), max_distance=5.0)
        assert np.isclose(result[0], 1.0)

    def test_beyond_threshold(self):
        """Distance > max_distance should give 0.0."""
        distances = np.array([5.1, 10.0, 100.0])
        result = threshold(distances, max_distance=5.0)
        assert np.allclose(result, 0.0)

    def test_mixed_distances(self):
        """Mixed distances should give correct binary output."""
        distances = np.array([1.0, 2.5, 5.0, 6.0, 10.0])
        result = threshold(distances, max_distance=5.0)
        expected = np.array([1.0, 1.0, 1.0, 0.0, 0.0])
        assert np.allclose(result, expected)

    def test_at_zero_distance(self):
        """Distance=0 should always be within threshold."""
        result = threshold(np.array([0.0]), max_distance=5.0)
        assert np.isclose(result[0], 1.0)

    def test_invalid_max_distance_zero(self):
        """max_distance=0 should raise ValueError."""
        with pytest.raises(ValueError, match="max_distance must be positive"):
            threshold(np.array([1.0]), max_distance=0.0)

    def test_invalid_max_distance_negative(self):
        """max_distance<0 should raise ValueError."""
        with pytest.raises(ValueError, match="max_distance must be positive"):
            threshold(np.array([1.0]), max_distance=-5.0)


class TestApplyDecay:
    """Tests for apply_decay function."""

    def test_exponential_decay(self):
        """Test apply_decay with exponential function."""
        edges = pd.DataFrame({
            "id1": [1, 2, 3],
            "id2": [2, 3, 4],
            "distance": [0.0, 1.0, 2.0],
        })
        result = apply_decay(edges, "exponential", alpha=1.0)

        assert list(result.columns) == ["id1", "id2", "value"]
        assert len(result) == 3
        assert np.isclose(result.loc[0, "value"], 1.0)
        assert np.isclose(result.loc[1, "value"], np.exp(-1.0))
        assert np.isclose(result.loc[2, "value"], np.exp(-2.0))

    def test_power_decay(self):
        """Test apply_decay with inverse power function."""
        edges = pd.DataFrame({
            "id1": [1, 2],
            "id2": [2, 3],
            "distance": [0.0, 1.0],
        })
        result = apply_decay(edges, "power", beta=2.0)

        assert np.isclose(result.loc[0, "value"], 1.0)
        assert np.isclose(result.loc[1, "value"], 1.0 / (1.0 + 1.0 ** 2.0))

    def test_threshold_decay(self):
        """Test apply_decay with threshold function."""
        edges = pd.DataFrame({
            "id1": [1, 2, 3],
            "id2": [2, 3, 4],
            "distance": [1.0, 2.5, 5.0],
        })
        result = apply_decay(edges, "threshold", max_distance=3.0)

        expected = np.array([1.0, 1.0, 0.0])
        assert np.allclose(result["value"].values, expected)

    def test_missing_id1_column(self):
        """Missing id1 column should raise ValueError."""
        edges = pd.DataFrame({
            "id2": [1, 2],
            "distance": [1.0, 2.0],
        })
        with pytest.raises(ValueError, match="edges missing columns"):
            apply_decay(edges, "exponential", alpha=1.0)

    def test_missing_id2_column(self):
        """Missing id2 column should raise ValueError."""
        edges = pd.DataFrame({
            "id1": [1, 2],
            "distance": [1.0, 2.0],
        })
        with pytest.raises(ValueError, match="edges missing columns"):
            apply_decay(edges, "exponential", alpha=1.0)

    def test_missing_distance_column(self):
        """Missing distance column should raise ValueError."""
        edges = pd.DataFrame({
            "id1": [1, 2],
            "id2": [2, 3],
        })
        with pytest.raises(ValueError, match="edges missing columns"):
            apply_decay(edges, "exponential", alpha=1.0)

    def test_multiple_missing_columns(self):
        """Multiple missing columns should raise ValueError."""
        edges = pd.DataFrame({
            "id1": [1, 2],
        })
        with pytest.raises(ValueError, match="edges missing columns"):
            apply_decay(edges, "exponential", alpha=1.0)

    def test_unknown_decay_type(self):
        """Unknown decay_type should raise ValueError."""
        edges = pd.DataFrame({
            "id1": [1],
            "id2": [2],
            "distance": [1.0],
        })
        with pytest.raises(ValueError, match="Unknown decay_type"):
            apply_decay(edges, "unknown_type", alpha=1.0)

    def test_default_alpha_parameter(self):
        """Test apply_decay uses default alpha=1.0 when not provided."""
        edges = pd.DataFrame({
            "id1": [1],
            "id2": [2],
            "distance": [1.0],
        })
        result = apply_decay(edges, "exponential")
        expected = np.exp(-1.0)
        assert np.isclose(result.loc[0, "value"], expected)

    def test_default_beta_parameter(self):
        """Test apply_decay uses default beta=1.0 when not provided."""
        edges = pd.DataFrame({
            "id1": [1],
            "id2": [2],
            "distance": [1.0],
        })
        result = apply_decay(edges, "power")
        expected = 1.0 / (1.0 + 1.0)
        assert np.isclose(result.loc[0, "value"], expected)

    def test_default_max_distance_parameter(self):
        """Test apply_decay uses default max_distance=1.0 when not provided."""
        edges = pd.DataFrame({
            "id1": [1, 2],
            "id2": [2, 3],
            "distance": [0.5, 1.5],
        })
        result = apply_decay(edges, "threshold")
        expected = np.array([1.0, 0.0])
        assert np.allclose(result["value"].values, expected)

    def test_preserves_id_values(self):
        """Test that id1 and id2 values are preserved exactly."""
        edges = pd.DataFrame({
            "id1": [100, 200, 300],
            "id2": [101, 201, 301],
            "distance": [1.0, 2.0, 3.0],
        })
        result = apply_decay(edges, "exponential", alpha=0.5)

        assert list(result["id1"].values) == [100, 200, 300]
        assert list(result["id2"].values) == [101, 201, 301]

    def test_large_edge_list(self):
        """Test with larger edge list (simulating 50k+ PU scenario)."""
        n_edges = 10000
        edges = pd.DataFrame({
            "id1": np.arange(n_edges),
            "id2": np.arange(1, n_edges + 1),
            "distance": np.random.uniform(0, 100, n_edges),
        })
        result = apply_decay(edges, "exponential", alpha=0.1)

        assert len(result) == n_edges
        assert list(result.columns) == ["id1", "id2", "value"]
        assert np.all((result["value"] >= 0) & (result["value"] <= 1))

    def test_output_values_in_valid_range(self):
        """All output values should be in [0, 1]."""
        edges = pd.DataFrame({
            "id1": [1, 2, 3, 4, 5],
            "id2": [2, 3, 4, 5, 6],
            "distance": [0.1, 1.0, 5.0, 10.0, 100.0],
        })
        for decay_type in ["exponential", "power", "threshold"]:
            result = apply_decay(edges, decay_type, alpha=1.0, beta=1.0,
                                 max_distance=5.0)
            assert np.all((result["value"] >= 0) & (result["value"] <= 1))

    def test_integer_distance_column(self):
        """Test that integer distance columns are handled correctly."""
        edges = pd.DataFrame({
            "id1": [1, 2],
            "id2": [2, 3],
            "distance": [1, 5],  # integers, not floats
        })
        result = apply_decay(edges, "threshold", max_distance=3)

        expected = np.array([1.0, 0.0])
        assert np.allclose(result["value"].values, expected)

    def test_empty_dataframe(self):
        """Test with empty edge list."""
        edges = pd.DataFrame({
            "id1": [],
            "id2": [],
            "distance": [],
        })
        result = apply_decay(edges, "exponential", alpha=1.0)

        assert len(result) == 0
        assert list(result.columns) == ["id1", "id2", "value"]
