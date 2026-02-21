"""Tests for target achievement Shiny module."""
from __future__ import annotations

from pymarxan_shiny.modules.results.target_met import (
    target_met_server,
    target_met_ui,
)


def test_target_met_ui_returns_tag():
    ui_elem = target_met_ui("test_targets")
    assert ui_elem is not None


def test_target_met_server_callable():
    assert callable(target_met_server)
