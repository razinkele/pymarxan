"""Tests for feature override functionality."""
from __future__ import annotations

import pandas as pd
import pytest

from pymarxan.models.problem import ConservationProblem, apply_feature_overrides


def _make_problem():
    return ConservationProblem(
        planning_units=pd.DataFrame(
            {"id": [1, 2, 3], "cost": [1.0, 2.0, 3.0], "status": [0, 0, 0]}
        ),
        features=pd.DataFrame(
            {
                "id": [1, 2, 3],
                "name": ["forest", "wetland", "coral"],
                "target": [100.0, 200.0, 300.0],
                "spf": [1.0, 1.0, 1.0],
            }
        ),
        pu_vs_features=pd.DataFrame(
            {
                "species": [1, 2, 3],
                "pu": [1, 2, 3],
                "amount": [150.0, 250.0, 350.0],
            }
        ),
    )


class TestApplyFeatureOverrides:
    def test_override_single_target(self):
        p = _make_problem()
        result = apply_feature_overrides(p, {1: {"target": 50.0}})
        assert result.features.loc[result.features["id"] == 1, "target"].iloc[0] == 50.0
        assert result.features.loc[result.features["id"] == 2, "target"].iloc[0] == 200.0

    def test_override_spf(self):
        p = _make_problem()
        result = apply_feature_overrides(p, {2: {"spf": 5.0}})
        assert result.features.loc[result.features["id"] == 2, "spf"].iloc[0] == 5.0

    def test_override_multiple_features(self):
        p = _make_problem()
        result = apply_feature_overrides(
            p,
            {
                1: {"target": 50.0},
                3: {"target": 150.0, "spf": 2.0},
            },
        )
        assert result.features.loc[result.features["id"] == 1, "target"].iloc[0] == 50.0
        assert result.features.loc[result.features["id"] == 3, "target"].iloc[0] == 150.0
        assert result.features.loc[result.features["id"] == 3, "spf"].iloc[0] == 2.0

    def test_does_not_mutate_original(self):
        p = _make_problem()
        apply_feature_overrides(p, {1: {"target": 50.0}})
        assert p.features.loc[p.features["id"] == 1, "target"].iloc[0] == 100.0

    def test_invalid_feature_id_raises(self):
        p = _make_problem()
        with pytest.raises(KeyError, match="999"):
            apply_feature_overrides(p, {999: {"target": 50.0}})

    def test_invalid_field_raises(self):
        p = _make_problem()
        with pytest.raises(ValueError, match="invalid_field"):
            apply_feature_overrides(p, {1: {"invalid_field": 50.0}})

    def test_empty_overrides_returns_copy(self):
        p = _make_problem()
        result = apply_feature_overrides(p, {})
        assert result is not p
        assert result.features.equals(p.features)
