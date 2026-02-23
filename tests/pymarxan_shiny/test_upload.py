"""Tests for data_input/upload Shiny module."""
from pymarxan_shiny.modules.data_input.upload import upload_server, upload_ui


def test_upload_ui_callable():
    assert callable(upload_ui)


def test_upload_server_callable():
    assert callable(upload_server)
