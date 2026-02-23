"""Tests for solution map Shiny module (ipyleaflet upgrade)."""
from __future__ import annotations

from pymarxan_shiny.modules.mapping.solution_map import (
    solution_map_server,
    solution_map_ui,
)


def test_solution_map_ui_returns_tag():
    ui_elem = solution_map_ui("test_sol")
    assert ui_elem is not None


def test_solution_map_server_callable():
    assert callable(solution_map_server)


def test_solution_map_ui_has_output_widget():
    """UI should contain an output_widget for the map."""
    html = str(solution_map_ui("test_sol"))
    assert "ipywidget" in html.lower()
