"""Tests for ConservationProblem.clone()."""
from __future__ import annotations

import pandas as pd

from pymarxan.models.problem import ConservationProblem


def _make_problem():
    return ConservationProblem(
        planning_units=pd.DataFrame(
            {"id": [1, 2], "cost": [1.0, 2.0], "status": [0, 0]}
        ),
        features=pd.DataFrame(
            {"id": [1], "name": ["f1"], "target": [10.0], "spf": [1.0]}
        ),
        pu_vs_features=pd.DataFrame(
            {"species": [1, 1], "pu": [1, 2], "amount": [5.0, 8.0]}
        ),
        boundary=pd.DataFrame(
            {"id1": [1], "id2": [2], "boundary": [1.0]}
        ),
        parameters={"BLM": "1.0"},
    )


class TestClone:
    def test_clone_returns_new_instance(self):
        p = _make_problem()
        c = p.clone()
        assert c is not p

    def test_clone_has_equal_data(self):
        p = _make_problem()
        c = p.clone()
        assert c.planning_units.equals(p.planning_units)
        assert c.features.equals(p.features)
        assert c.pu_vs_features.equals(p.pu_vs_features)
        assert c.boundary.equals(p.boundary)
        assert c.parameters == p.parameters

    def test_clone_is_independent(self):
        p = _make_problem()
        c = p.clone()
        c.planning_units.loc[0, "cost"] = 999.0
        c.parameters["BLM"] = "99.0"
        assert p.planning_units.loc[0, "cost"] == 1.0
        assert p.parameters["BLM"] == "1.0"

    def test_clone_without_boundary(self):
        p = _make_problem()
        p.boundary = None
        c = p.clone()
        assert c.boundary is None
