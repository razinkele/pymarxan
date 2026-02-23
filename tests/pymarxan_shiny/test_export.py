"""Tests for results/export Shiny module."""
from pymarxan_shiny.modules.results.export import export_server, export_ui


def test_export_ui_callable():
    assert callable(export_ui)


def test_export_server_callable():
    assert callable(export_server)
