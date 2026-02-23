"""Tests for WDPA overlay Shiny module."""
from pymarxan_shiny.modules.spatial.wdpa_overlay import wdpa_overlay_server, wdpa_overlay_ui


def test_wdpa_overlay_ui_callable():
    assert callable(wdpa_overlay_ui)


def test_wdpa_overlay_server_callable():
    assert callable(wdpa_overlay_server)
