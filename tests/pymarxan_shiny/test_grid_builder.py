"""Tests for grid builder Shiny module."""
from __future__ import annotations

from pymarxan_shiny.modules.spatial.grid_builder import grid_builder_server, grid_builder_ui


def test_grid_builder_ui_callable():
    assert callable(grid_builder_ui)


def test_grid_builder_server_callable():
    assert callable(grid_builder_server)
