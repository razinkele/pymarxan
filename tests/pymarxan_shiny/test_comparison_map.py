"""Tests for comparison map Shiny module."""
from __future__ import annotations

from pymarxan_shiny.modules.mapping.comparison_map import (
    comparison_color,
    comparison_map_server,
    comparison_map_ui,
)


def test_comparison_map_ui_returns_tag():
    ui_elem = comparison_map_ui("test_cmp")
    assert ui_elem is not None


def test_comparison_map_server_callable():
    assert callable(comparison_map_server)


def test_comparison_color_both():
    """Both selected -> green."""
    assert comparison_color(True, True) == "#2ecc71"


def test_comparison_color_a_only():
    """A only -> blue."""
    assert comparison_color(True, False) == "#3498db"


def test_comparison_color_b_only():
    """B only -> orange."""
    assert comparison_color(False, True) == "#e67e22"


def test_comparison_color_neither():
    """Neither -> gray."""
    assert comparison_color(False, False) == "#bdc3c7"


def test_comparison_map_ui_has_output_widget():
    """UI should contain an output_widget for the map."""
    html = str(comparison_map_ui("test_cm"))
    assert "ipywidget" in html.lower()
