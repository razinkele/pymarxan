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


def test_module_has_no_dead_map_code():
    """Verify _pu_color helper was removed with the dead Map code."""
    import pymarxan_shiny.modules.mapping.solution_map as mod

    assert not hasattr(mod, "_pu_color")
