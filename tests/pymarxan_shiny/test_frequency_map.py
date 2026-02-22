"""Tests for frequency map Shiny module."""
from __future__ import annotations

from pymarxan_shiny.modules.mapping.frequency_map import (
    frequency_color,
    frequency_map_server,
    frequency_map_ui,
)


def test_frequency_map_ui_returns_tag():
    ui_elem = frequency_map_ui("test_freq")
    assert ui_elem is not None


def test_frequency_map_server_callable():
    assert callable(frequency_map_server)


def test_frequency_color_gradient():
    """frequency_color maps 0-1 to white-to-blue gradient."""
    white = frequency_color(0.0)
    blue = frequency_color(1.0)
    assert isinstance(white, str) and white.startswith("#")
    assert isinstance(blue, str) and blue.startswith("#")
    assert white != blue


def test_frequency_color_midpoint():
    """Midpoint value produces intermediate color."""
    mid = frequency_color(0.5)
    assert isinstance(mid, str) and mid.startswith("#")
