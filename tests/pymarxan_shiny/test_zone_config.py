"""Tests for zones/zone_config Shiny module."""
from pymarxan_shiny.modules.zones.zone_config import zone_config_server, zone_config_ui


def test_zone_config_ui_callable():
    assert callable(zone_config_ui)


def test_zone_config_server_callable():
    assert callable(zone_config_server)
