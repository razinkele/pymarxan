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


def test_acknowledges_marxan_and_prioritizr_with_citations():
    html = _html()
    assert "Acknowledgements" in html
    # Marxan: Ball, Possingham & Watts (2009); Marxan with Zones: Watts et al. (2009)
    assert "Ball, I. R., Possingham, H. P., &amp; Watts, M. (2009)" in html
    assert "https://doi.org/10.1016/j.envsoft.2009.06.005" in html
    # prioritizr: Hanson et al. (2024)
    assert "Hanson, J. O., Schuster, R., Strimas-Mackey, M., et al. (2024)" in html
    assert "https://doi.org/10.1111/cobi.14376" in html
    assert "not affiliated with" in html


def test_comparison_rows_match_the_manuals_side_by_side_table():
    """The About dialog mirrors docs/USER_MANUAL.md "Side by side"; keep them in step."""
    from pathlib import Path

    manual = (Path(__file__).resolve().parents[2] / "docs" / "USER_MANUAL.md").read_text()
    table = manual[manual.index("### Side by side"):manual.index("### Which tool when")]
    for row in COMPARISON_ROWS:
        assert "| " + " | ".join(row) + " |" in table, row
