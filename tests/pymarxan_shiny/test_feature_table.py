"""Tests for feature table editor Shiny module."""
from __future__ import annotations

from pymarxan_shiny.modules.data.feature_table import (
    feature_table_server,
    feature_table_ui,
    validate_feature_edit,
)


def test_feature_table_ui_returns_tag():
    ui_elem = feature_table_ui("test_ft")
    assert ui_elem is not None


def test_feature_table_server_callable():
    assert callable(feature_table_server)


def test_validate_target_positive():
    """Target edits must be non-negative floats."""
    assert validate_feature_edit("target", "10.5") == 10.5
    assert validate_feature_edit("target", "0") == 0.0


def test_validate_target_negative_rejected():
    """Negative target values are rejected (returns None)."""
    assert validate_feature_edit("target", "-5") is None


def test_validate_spf_positive():
    """SPF edits must be non-negative floats."""
    assert validate_feature_edit("spf", "1.5") == 1.5


def test_validate_spf_negative_rejected():
    """Negative SPF values are rejected."""
    assert validate_feature_edit("spf", "-0.1") is None


def test_validate_non_numeric_rejected():
    """Non-numeric values are rejected."""
    assert validate_feature_edit("target", "abc") is None


def test_validate_readonly_column():
    """Edits to id or name columns are rejected."""
    assert validate_feature_edit("id", "999") is None
    assert validate_feature_edit("name", "new_name") is None
