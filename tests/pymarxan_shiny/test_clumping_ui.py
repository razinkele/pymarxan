"""Phase 19 Batch 4 — Shiny UI surface tests.

File-based smoke tests (Review 6 H4 / Phase 18 pattern): verify the
right strings appear in the right modules. Pinning these makes the
plan's UI integration points discoverable in code search and prevents
silent regressions when the surrounding files are refactored.
"""
from __future__ import annotations

from pathlib import Path

import pymarxan_shiny.modules.data.feature_table as feature_table_mod
import pymarxan_shiny.modules.help.help_content as help_content_mod
import pymarxan_shiny.modules.results.target_met as target_met_mod


def test_feature_table_offers_target2_and_clumptype_columns():
    """feature_table.py exposes the TARGET2 / CLUMPTYPE editing surface
    described in the design doc — three sites: _COLUMN_ORDER, the
    feature_grid render selection, and the validate_feature_edit
    whitelist with the int-only check for clumptype."""
    src = Path(feature_table_mod.__file__).read_text()
    assert '"target2"' in src
    assert '"clumptype"' in src
    # _COLUMN_ORDER lists the full schema
    assert "_COLUMN_ORDER" in src
    # Editable-column allow-lists distinguish float vs int
    assert "_EDITABLE_FLOAT_COLUMNS" in src
    assert "_EDITABLE_INT_COLUMNS" in src
    # CLUMPTYPE validation must reject anything outside {0,1,2}
    assert "(0, 1, 2)" in src


def test_validate_feature_edit_rejects_invalid_clumptype():
    """The validator returns None for clumptype outside {0,1,2}."""
    from pymarxan_shiny.modules.data.feature_table import validate_feature_edit
    assert validate_feature_edit("clumptype", "0") == 0
    assert validate_feature_edit("clumptype", "1") == 1
    assert validate_feature_edit("clumptype", "2") == 2
    assert validate_feature_edit("clumptype", "3") is None
    assert validate_feature_edit("clumptype", "-1") is None
    assert validate_feature_edit("clumptype", "abc") is None


def test_validate_feature_edit_accepts_target2_float():
    """target2 is a non-negative float (0 disables clumping)."""
    from pymarxan_shiny.modules.data.feature_table import validate_feature_edit
    assert validate_feature_edit("target2", "0") == 0.0
    assert validate_feature_edit("target2", "25.5") == 25.5
    assert validate_feature_edit("target2", "-1") is None


def test_target_met_shows_clump_short_column_when_active():
    """target_met.py adds clump_short / target2 / clumptype columns when
    any feature has target2 > 0 AND the Solution carries a clump
    evaluation. Pinned via source-text presence."""
    src = Path(target_met_mod.__file__).read_text()
    assert "clump_shortfalls" in src
    assert "target2" in src
    assert "clumptype" in src
    assert "clump_short" in src
    assert "has_clumping" in src


def test_help_content_documents_target2_clumping():
    """help_content explains the TARGET2 / CLUMPTYPE columns + cites the
    Marxan source-of-truth and the foundational papers."""
    src = Path(help_content_mod.__file__).read_text()
    assert "TARGET2" in src
    assert "type-4" in src or "type-4 species" in src
    # CLUMPTYPE semantics documented per Marxan source
    assert "PartialPen4" in src
    # Citations
    assert "Ball" in src and "2009" in src
    assert "Metcalfe" in src and "2015" in src
