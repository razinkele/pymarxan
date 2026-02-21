"""Tests for SPF explorer Shiny module."""
from __future__ import annotations

from pymarxan_shiny.modules.calibration.spf_explorer import (
    spf_explorer_server,
    spf_explorer_ui,
)


def test_spf_explorer_ui_returns_tag():
    ui_elem = spf_explorer_ui("test_spf")
    assert ui_elem is not None


def test_spf_explorer_server_callable():
    assert callable(spf_explorer_server)
