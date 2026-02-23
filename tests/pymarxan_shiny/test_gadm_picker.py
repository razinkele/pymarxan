"""Tests for GADM picker Shiny module."""
from pymarxan_shiny.modules.spatial.gadm_picker import gadm_picker_server, gadm_picker_ui


def test_gadm_picker_ui_callable():
    assert callable(gadm_picker_ui)


def test_gadm_picker_server_callable():
    assert callable(gadm_picker_server)
