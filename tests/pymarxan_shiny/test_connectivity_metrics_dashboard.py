"""File-based smoke tests for the connectivity metrics dashboard module.

The dashboard surfaces all seven connectivity metrics shipped by
Phase 24 (in/out-degree, betweenness, eigenvector, PageRank, donor,
recipient) in a single Shiny module. Tests verify the right imports
and strings appear so the module's wiring stays in lockstep with the
``pymarxan.connectivity.metrics`` API.
"""
from __future__ import annotations

from pathlib import Path

import pymarxan_shiny.modules.connectivity.metrics_dashboard as dash_mod


def test_dashboard_imports_all_seven_metrics():
    """Module source references every Phase 24 metric function by name."""
    src = Path(dash_mod.__file__).read_text()
    assert "compute_in_degree" in src
    assert "compute_out_degree" in src
    assert "compute_betweenness_centrality" in src
    assert "compute_eigenvector_centrality" in src
    assert "compute_pagerank_centrality" in src
    assert "compute_donors" in src
    assert "compute_recipients" in src


def test_dashboard_exports_ui_and_server():
    """Standard Shiny module shape: ``{module}_ui`` and
    ``{module}_server`` decorated functions."""
    assert hasattr(dash_mod, "connectivity_metrics_dashboard_ui")
    assert hasattr(dash_mod, "connectivity_metrics_dashboard_server")


def test_dashboard_uses_connectivity_to_matrix_for_edgelist_input():
    """Problem.connectivity is an edge list — module must convert it to
    a matrix via the shared utility before computing metrics."""
    src = Path(dash_mod.__file__).read_text()
    assert "connectivity_to_matrix" in src


def test_dashboard_renders_data_frame_output():
    """The dashboard surfaces metrics as a sortable table (matches the
    target_met / feature_table pattern in pymarxan_shiny)."""
    src = Path(dash_mod.__file__).read_text()
    assert "@render.data_frame" in src
    assert "output_data_frame" in src


def test_dashboard_handles_problem_without_connectivity():
    """When problem.connectivity is None the module degrades to a
    placeholder rather than crashing — tested by source-grep that
    a None-check exists (Shiny modules aren't easy to exercise without
    a session, so we pin the guard at the source level)."""
    src = Path(dash_mod.__file__).read_text()
    assert "connectivity is None" in src or "connectivity is not None" in src
