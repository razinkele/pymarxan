"""Smoke test: the rivers Shiny panel imports and exposes ui/server."""
from __future__ import annotations


def test_module_exposes_ui_and_server():
    from pymarxan_shiny.modules.rivers import (
        rivers_panel_server,
        rivers_panel_ui,
    )

    assert callable(rivers_panel_ui)
    assert callable(rivers_panel_server)
