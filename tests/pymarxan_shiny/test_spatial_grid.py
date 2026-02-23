"""Tests for spatial grid Shiny module."""
from __future__ import annotations

from pymarxan_shiny.modules.mapping.spatial_grid import (
    cost_color,
    spatial_grid_server,
    spatial_grid_ui,
    status_color,
)


def test_spatial_grid_ui_returns_tag():
    ui_elem = spatial_grid_ui("test_grid")
    assert ui_elem is not None


def test_spatial_grid_server_callable():
    assert callable(spatial_grid_server)


def test_cost_color_gradient():
    """cost_color returns hex strings on a yellow-to-red gradient."""
    low = cost_color(0.0)
    high = cost_color(1.0)
    assert isinstance(low, str) and low.startswith("#")
    assert isinstance(high, str) and high.startswith("#")
    assert low != high


def test_status_color_mapping():
    """status_color returns categorical colors for known status values."""
    assert status_color(0) != status_color(2)  # available vs locked-in
    assert status_color(3) != status_color(0)  # locked-out vs available
    # Unknown status should not crash
    assert isinstance(status_color(99), str)


def test_spatial_grid_ui_has_output_widget():
    """UI should contain an output_widget for the map."""
    html = str(spatial_grid_ui("test_sg"))
    assert "ipywidget" in html.lower()
