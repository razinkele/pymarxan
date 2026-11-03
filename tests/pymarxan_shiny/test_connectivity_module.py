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


def test_connectivity_weight_sets_a_new_problem_not_the_same_object():
    """reactive.Value.set ignores the object it already holds, so editing the problem in
    place and setting it again notified nothing (found in the v0.48.0 review)."""
    from pathlib import Path

    import pymarxan_shiny.modules.connectivity.connectivity_config as cc

    src = Path(cc.__file__).read_text()
    assert 'p.parameters["CONNECTIVITY_WEIGHT"] =' not in src
    assert "copy_with(" in src
