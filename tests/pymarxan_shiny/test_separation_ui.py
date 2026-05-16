"""Phase 20 Batch 4 — Shiny UI surface tests.

File-based smoke tests (Phase 19's `test_clumping_ui.py` pattern): verify
the right strings appear in the right modules. Pins the plan's UI
integration points in code search and prevents silent regressions when
the surrounding files are refactored.
"""
from __future__ import annotations

from pathlib import Path

import pymarxan_shiny.modules.data.feature_table as feature_table_mod
import pymarxan_shiny.modules.help.help_content as help_content_mod
import pymarxan_shiny.modules.results.target_met as target_met_mod


def test_feature_table_offers_separation_columns():
    """feature_table.py exposes the sepdistance / sepnum editing surface.

    Phase 20 touch points: _COLUMN_ORDER lists both, _EDITABLE_FLOAT/_INT
    cover them with the round-3 H9 split-validator (clumptype keeps its
    {0,1,2} rule; sepnum gets a >= 0 rule)."""
    src = Path(feature_table_mod.__file__).read_text()
    assert '"sepdistance"' in src
    assert '"sepnum"' in src
    # _COLUMN_ORDER lists the full schema
    assert "_COLUMN_ORDER" in src
    # Per-column int validator override exists (round-3 H9)
    assert "_INT_VALIDATORS" in src
    # The shared optional-columns tuple appears with sepdistance + sepnum
    assert '("ptarget", "target2", "clumptype", "sepdistance", "sepnum")' in src


def test_validate_feature_edit_accepts_sepnum_above_two():
    """``sepnum = 3, 5, 100`` are all valid — round-3 H9 split fix
    prevents clumptype's {0,1,2} rule from rejecting them."""
    from pymarxan_shiny.modules.data.feature_table import validate_feature_edit
    assert validate_feature_edit("sepnum", "0") == 0
    assert validate_feature_edit("sepnum", "1") == 1
    assert validate_feature_edit("sepnum", "2") == 2
    assert validate_feature_edit("sepnum", "3") == 3
    assert validate_feature_edit("sepnum", "100") == 100
    assert validate_feature_edit("sepnum", "-1") is None
    assert validate_feature_edit("sepnum", "abc") is None


def test_validate_feature_edit_accepts_sepdistance_float():
    """sepdistance is a non-negative float (0 disables separation)."""
    from pymarxan_shiny.modules.data.feature_table import validate_feature_edit
    assert validate_feature_edit("sepdistance", "0") == 0.0
    assert validate_feature_edit("sepdistance", "1250.5") == 1250.5
    assert validate_feature_edit("sepdistance", "-1") is None


def test_validate_feature_edit_keeps_clumptype_strict():
    """Round-3 H9 anti-regression: clumptype's {0,1,2} rule must NOT
    bleed into sepnum, and sepnum's >= 0 rule must NOT bleed into clumptype."""
    from pymarxan_shiny.modules.data.feature_table import validate_feature_edit
    assert validate_feature_edit("clumptype", "0") == 0
    assert validate_feature_edit("clumptype", "1") == 1
    assert validate_feature_edit("clumptype", "2") == 2
    assert validate_feature_edit("clumptype", "3") is None
    assert validate_feature_edit("clumptype", "-1") is None


def test_target_met_shows_sep_short_column_when_active():
    """target_met.py adds sep_short / sepdistance / sepnum columns when
    any feature is sep-active AND the Solution carries a sep evaluation."""
    src = Path(target_met_mod.__file__).read_text()
    assert "sep_shortfalls" in src
    assert "sepdistance" in src
    assert "sepnum" in src
    assert "sep_short" in src
    assert "has_separation" in src


def test_help_content_documents_separation():
    """help_content explains the SEPDISTANCE / SEPNUM columns + cites the
    Marxan source-of-truth and the foundational papers."""
    src = Path(help_content_mod.__file__).read_text()
    assert "SEPDISTANCE" in src
    assert "SEPNUM" in src
    # Marxan source-of-truth pointers
    assert "computeSepPenalty" in src
    assert "CountSeparation2" in src
    # Hyperbolic curve documented
    assert "hyperbolic" in src
    # Round-3 L3 unit-mismatch note
    assert "Unit mismatch" in src
    # Citations
    assert "Watts" in src
