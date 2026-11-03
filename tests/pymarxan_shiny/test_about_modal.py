"""Tests for the About modal: version, authorship and the comparison table."""
from __future__ import annotations

from pymarxan_shiny.modules.help.about import COMPARISON_ROWS, about_modal


def _html(version: str = "9.9.9") -> str:
    return str(about_modal(version))


def test_shows_version():
    assert "Version 9.9.9" in _html("9.9.9")


def test_shows_authorship():
    html = _html()
    assert "Artūras Razinkovas-Baziukas" in html
    assert "Marine Research Institute" in html
    assert "Klaipėda University" in html


def test_comparison_table_has_three_tools():
    html = _html()
    assert "<table" in html
    for tool in ("Marxan", "prioritizr", "pymarxan"):
        assert f"<th>{tool}</th>" in html


def test_every_comparison_row_is_rendered_with_four_cells():
    html = _html()
    for row in COMPARISON_ROWS:
        assert len(row) == 4
        for cell in row:
            assert cell in html
