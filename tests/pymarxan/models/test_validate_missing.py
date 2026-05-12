"""Tests for ConservationProblem.validate() with missing columns."""
from __future__ import annotations

import pandas as pd

from pymarxan.models.problem import ConservationProblem


class TestValidateMissingColumns:
    def test_missing_puvspr_columns_no_keyerror(self):
        """validate() should report missing columns, not crash with KeyError."""
        pu = pd.DataFrame({"id": [1], "cost": [10.0], "status": [0]})
        features = pd.DataFrame({"id": [1], "name": ["f1"], "target": [1.0], "spf": [1.0]})
        puvspr = pd.DataFrame({"wrong_col": [1], "amount": [5.0]})
        p = ConservationProblem(
            planning_units=pu, features=features, pu_vs_features=puvspr,
        )
        errors = p.validate()
        assert any("missing columns" in e for e in errors)
