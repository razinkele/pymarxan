"""Tests for results/summary_table Shiny module."""
from pymarxan_shiny.modules.results.summary_table import summary_table_server, summary_table_ui


def test_summary_table_ui_callable():
    assert callable(summary_table_ui)


def test_summary_table_server_callable():
    assert callable(summary_table_server)
