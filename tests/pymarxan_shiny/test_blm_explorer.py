"""Tests for calibration/blm_explorer Shiny module."""
from pymarxan_shiny.modules.calibration.blm_explorer import blm_explorer_server, blm_explorer_ui


def test_blm_explorer_ui_callable():
    assert callable(blm_explorer_ui)


def test_blm_explorer_server_callable():
    assert callable(blm_explorer_server)
