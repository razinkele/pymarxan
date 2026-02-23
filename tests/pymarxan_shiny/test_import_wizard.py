"""Tests for import wizard Shiny module."""
from pymarxan_shiny.modules.spatial.import_wizard import (
    import_wizard_server,
    import_wizard_ui,
)


def test_import_wizard_ui_callable():
    assert callable(import_wizard_ui)


def test_import_wizard_server_callable():
    assert callable(import_wizard_server)
