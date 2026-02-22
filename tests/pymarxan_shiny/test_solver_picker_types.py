"""Tests for greedy, iterative_improvement, pipeline solver types."""
from __future__ import annotations

from pymarxan_shiny.modules.solver_config.solver_picker import (
    solver_picker_ui,
)


def test_picker_has_greedy():
    ui_elem = solver_picker_ui("test_sp")
    html = str(ui_elem)
    assert "greedy" in html


def test_picker_has_iterative_improvement():
    ui_elem = solver_picker_ui("test_sp")
    html = str(ui_elem)
    assert "iterative_improvement" in html


def test_picker_has_pipeline():
    ui_elem = solver_picker_ui("test_sp")
    html = str(ui_elem)
    assert "pipeline" in html


def test_picker_has_heurtype_selector():
    ui_elem = solver_picker_ui("test_sp")
    html = str(ui_elem)
    assert "heurtype" in html


def test_picker_has_itimptype_selector():
    ui_elem = solver_picker_ui("test_sp")
    html = str(ui_elem)
    assert "itimptype" in html


def test_picker_has_runmode_selector():
    ui_elem = solver_picker_ui("test_sp")
    html = str(ui_elem)
    assert "runmode" in html
