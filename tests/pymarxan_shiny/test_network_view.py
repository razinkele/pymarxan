"""Tests for network view Shiny module."""
from __future__ import annotations

from pymarxan_shiny.modules.mapping.network_view import (
    compute_centroids,
    metric_color,
    network_view_server,
    network_view_ui,
)


def test_network_view_ui_returns_tag():
    ui_elem = network_view_ui("test_nv")
    assert ui_elem is not None


def test_network_view_server_callable():
    assert callable(network_view_server)


def test_metric_color_gradient():
    """metric_color maps 0-1 to yellow-to-purple gradient."""
    low = metric_color(0.0)
    high = metric_color(1.0)
    assert isinstance(low, str) and low.startswith("#")
    assert isinstance(high, str) and high.startswith("#")
    assert low != high


def test_compute_centroids():
    """compute_centroids returns center of each bounding box."""
    from pymarxan.models.geometry import generate_grid

    grid = generate_grid(4, origin=(0.0, 0.0), cell_size=0.01)
    centroids = compute_centroids(grid)
    assert len(centroids) == 4
    # First cell (0,0)-(0.01,0.01) => centroid (0.005, 0.005)
    assert abs(centroids[0][0] - 0.005) < 1e-10
    assert abs(centroids[0][1] - 0.005) < 1e-10


def test_compute_centroids_empty():
    """Empty grid returns empty list."""
    assert compute_centroids([]) == []
