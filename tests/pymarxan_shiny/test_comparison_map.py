"""Tests for comparison map Shiny module."""
from __future__ import annotations

from pymarxan_shiny.modules.mapping.comparison_map import (
    comparison_color,
    comparison_map_server,
    comparison_map_ui,
)
from pymarxan_shiny.modules.mapping.ocean_palette import (
    CMP_A_ONLY,
    CMP_B_ONLY,
    CMP_BOTH,
    CMP_NEITHER,
)


def test_comparison_map_ui_returns_tag():
    ui_elem = comparison_map_ui("test_cmp")
    assert ui_elem is not None


def test_comparison_map_server_callable():
    assert callable(comparison_map_server)


def test_comparison_color_both():
    """Both selected -> teal."""
    assert comparison_color(True, True) == CMP_BOTH


def test_comparison_color_a_only():
    """A only -> ocean-blue."""
    assert comparison_color(True, False) == CMP_A_ONLY


def test_comparison_color_b_only():
    """B only -> coral."""
    assert comparison_color(False, True) == CMP_B_ONLY


def test_comparison_color_neither():
    """Neither -> steel-gray."""
    assert comparison_color(False, False) == CMP_NEITHER


def test_comparison_map_ui_has_output_widget():
    """UI should contain an output_widget for the map."""
    html = str(comparison_map_ui("test_cm"))
    assert "ipywidget" in html.lower()
