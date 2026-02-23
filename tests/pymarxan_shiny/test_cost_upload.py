"""Tests for cost upload Shiny module."""
from pymarxan_shiny.modules.spatial.cost_upload import (
    cost_upload_server,
    cost_upload_ui,
)


def test_cost_upload_ui_callable():
    assert callable(cost_upload_ui)


def test_cost_upload_server_callable():
    assert callable(cost_upload_server)
