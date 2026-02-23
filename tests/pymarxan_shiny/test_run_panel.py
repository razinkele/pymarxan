"""Tests for run panel Shiny module."""
from __future__ import annotations

from pymarxan_shiny.modules.run_control.run_panel import (
    run_panel_server,
    run_panel_ui,
)


def test_run_panel_ui_returns_tag():
    ui_elem = run_panel_ui("test_run")
    assert ui_elem is not None


def test_run_panel_server_callable():
    assert callable(run_panel_server)


def test_run_panel_module_is_callable():
    from pymarxan_shiny.modules.run_control.run_panel import run_panel_server, run_panel_ui
    assert callable(run_panel_server)
    assert callable(run_panel_ui)
