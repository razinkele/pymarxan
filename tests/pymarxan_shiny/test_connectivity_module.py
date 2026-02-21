"""Tests for connectivity metrics visualization Shiny module."""
from __future__ import annotations

from pymarxan_shiny.modules.connectivity.metrics_viz import (
    metrics_viz_server,
    metrics_viz_ui,
)


def test_metrics_viz_ui_returns_tag():
    ui_elem = metrics_viz_ui("test_conn")
    assert ui_elem is not None


def test_metrics_viz_server_callable():
    assert callable(metrics_viz_server)
