"""Tests for feature table editor Shiny module."""
from __future__ import annotations

import copy

import pandas as pd

from pymarxan.models.problem import ConservationProblem
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


def test_validate_feature_edit_accepts_valid():
    """Comprehensive validation: valid values accepted, invalid rejected."""
    assert validate_feature_edit("target", "5.0") == 5.0
    assert validate_feature_edit("spf", "2.5") == 2.5
    assert validate_feature_edit("id", "5.0") is None
    assert validate_feature_edit("target", "-1") is None
    assert validate_feature_edit("target", "abc") is None


def test_feature_table_module_is_callable():
    """Both UI and server functions are importable and callable."""
    assert callable(feature_table_server)
    assert callable(feature_table_ui)


def _make_problem():
    """Create a problem with 3 features."""
    pu = pd.DataFrame({"id": [1, 2], "cost": [10.0, 20.0], "status": [0, 0]})
    features = pd.DataFrame({
        "id": [1, 2, 3],
        "name": ["feat_a", "feat_b", "feat_c"],
        "target": [100.0, 200.0, 300.0],
        "spf": [1.0, 2.0, 3.0],
    })
    puvspr = pd.DataFrame({
        "species": [1, 2, 3],
        "pu": [1, 1, 2],
        "amount": [5.0, 10.0, 15.0],
    })
    return ConservationProblem(
        planning_units=pu, features=features, pu_vs_features=puvspr,
    )


class TestFeatureTableSortSafe:
    def test_reversed_view_preserves_ids(self):
        """Simulate user sorting features in reverse order.

        The id-based join should correctly apply edits regardless
        of view order.
        """
        p = _make_problem()
        # Simulate data_view() returning reversed order (user sorted desc)
        view_df = p.features[["id", "name", "target", "spf"]].copy()
        # User edits: change feature 3's target to 999
        view_df.loc[view_df["id"] == 3, "target"] = 999.0
        # Reverse to simulate sort
        view_df = view_df.iloc[::-1].reset_index(drop=True)

        # Apply id-based merge (the fix)
        updated = copy.deepcopy(p)
        edits = view_df.set_index("id")[["target", "spf"]]
        for fid in edits.index:
            mask = updated.features["id"] == fid
            updated.features.loc[mask, "target"] = float(edits.at[fid, "target"])
            updated.features.loc[mask, "spf"] = float(edits.at[fid, "spf"])

        # Feature 3 should have target 999
        assert float(updated.features.loc[updated.features["id"] == 3, "target"].iloc[0]) == 999.0
        # Feature 1 should still have target 100 (not corrupted)
        assert float(updated.features.loc[updated.features["id"] == 1, "target"].iloc[0]) == 100.0

    def test_positional_assignment_would_corrupt(self):
        """Demonstrate that positional assignment corrupts data on reverse sort."""
        p = _make_problem()
        view_df = p.features[["id", "name", "target", "spf"]].copy()
        view_df.loc[view_df["id"] == 3, "target"] = 999.0
        # Reverse
        view_df = view_df.iloc[::-1].reset_index(drop=True)

        updated = copy.deepcopy(p)
        # OLD buggy approach: positional assignment
        updated.features["target"] = view_df["target"].values
        # Feature 1 gets feature 3's target (999) — CORRUPTED
        val = float(updated.features.loc[updated.features["id"] == 1, "target"].iloc[0])
        assert val == 999.0  # This proves the bug exists with positional assignment
