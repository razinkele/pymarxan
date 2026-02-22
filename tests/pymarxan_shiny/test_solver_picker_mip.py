"""Tests for MIP solver panel in solver picker Shiny module."""
from __future__ import annotations

from pymarxan_shiny.modules.solver_config.solver_picker import (
    solver_picker_ui,
)


def test_solver_picker_ui_returns_tag():
    ui_elem = solver_picker_ui("test_sp")
    assert ui_elem is not None


def test_solver_picker_has_mip_panel():
    ui_elem = solver_picker_ui("test_sp")
    html = str(ui_elem)
    assert "mip_time_limit" in html


def test_solver_picker_has_mip_gap():
    ui_elem = solver_picker_ui("test_sp")
    html = str(ui_elem)
    assert "mip_gap" in html


def test_solver_picker_has_mip_verbose():
    ui_elem = solver_picker_ui("test_sp")
    html = str(ui_elem)
    assert "mip_verbose" in html
