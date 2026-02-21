"""Tests for sweep explorer Shiny module."""
from __future__ import annotations

from pymarxan_shiny.modules.calibration.sweep_explorer import (
    sweep_explorer_server,
    sweep_explorer_ui,
)


def test_sweep_explorer_ui_returns_tag():
    """UI function returns a valid Shiny UI element."""
    ui_elem = sweep_explorer_ui("test_sweep")
    assert ui_elem is not None


def test_sweep_explorer_server_callable():
    """Server function is callable."""
    assert callable(sweep_explorer_server)
