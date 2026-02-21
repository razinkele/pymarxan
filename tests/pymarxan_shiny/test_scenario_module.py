"""Tests for scenario comparison Shiny module."""
from __future__ import annotations

from pymarxan_shiny.modules.results.scenario_compare import (
    scenario_compare_server,
    scenario_compare_ui,
)


def test_scenario_compare_ui_returns_tag():
    ui_elem = scenario_compare_ui("test_scenario")
    assert ui_elem is not None


def test_scenario_compare_server_callable():
    assert callable(scenario_compare_server)
