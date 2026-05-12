"""Tests for FeatureContiguityConstraint."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pymarxan.constraints.feature_contiguity import FeatureContiguityConstraint
from pymarxan.models.problem import ConservationProblem

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _make_problem(
    n_pu: int = 6,
    features: pd.DataFrame | None = None,
    pu_vs_features: pd.DataFrame | None = None,
    boundary: pd.DataFrame | None = None,
) -> ConservationProblem:
    """Flexible helper for building test problems."""
    planning_units = pd.DataFrame({
        "id": list(range(1, n_pu + 1)),
        "cost": [1.0] * n_pu,
        "status": [0] * n_pu,
    })
    if features is None:
        features = pd.DataFrame({
            "id": [1],
            "name": ["sp_a"],
            "target": [1.0],
            "spf": [1.0],
        })
    if pu_vs_features is None:
        pu_vs_features = pd.DataFrame({
            "species": [1] * n_pu,
            "pu": list(range(1, n_pu + 1)),
            "amount": [1.0] * n_pu,
        })
    if boundary is None:
        # Default: line graph 1-2-3-...-n
        boundary = pd.DataFrame({
            "id1": list(range(1, n_pu)),
            "id2": list(range(2, n_pu + 1)),
            "boundary": [1.0] * (n_pu - 1),
        })
    return ConservationProblem(
        planning_units=planning_units,
        features=features,
        pu_vs_features=pu_vs_features,
        boundary=boundary,
        parameters={"BLM": 0.0},
    )


def _two_feature_problem() -> ConservationProblem:
    """6-PU line graph with 2 features.

    Feature 1: contributes in PUs 1,2,3
    Feature 2: contributes in PUs 4,5,6
    """
    features = pd.DataFrame({
        "id": [1, 2],
        "name": ["sp_a", "sp_b"],
        "target": [1.0, 1.0],
        "spf": [1.0, 1.0],
    })
    pu_vs_features = pd.DataFrame({
        "species": [1, 1, 1, 2, 2, 2],
        "pu": [1, 2, 3, 4, 5, 6],
        "amount": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    })
    return _make_problem(n_pu=6, features=features, pu_vs_features=pu_vs_features)


# ------------------------------------------------------------------
# evaluate
# ------------------------------------------------------------------


class TestFeatureContiguityEvaluate:
    def test_all_feature_pus_connected(self):
        """All PUs for a feature are adjacent → satisfied."""
        problem = _make_problem(n_pu=4)
        c = FeatureContiguityConstraint()
        selected = np.array([True, True, True, True])
        result = c.evaluate(problem, selected)
        assert result.satisfied
        assert result.violation == 0.0

    def test_feature_pus_split_two_groups(self):
        """Feature PUs split into 2 disconnected groups → violation = 1."""
        problem = _make_problem(n_pu=4)
        c = FeatureContiguityConstraint()
        # Select PU 1 and PU 4 only — not adjacent in line graph
        selected = np.array([True, False, False, True])
        result = c.evaluate(problem, selected)
        assert not result.satisfied
        assert result.violation == 1.0

    def test_multiple_features_partial_violation(self):
        """Two features: one connected, one not → violation from one only."""
        problem = _two_feature_problem()
        c = FeatureContiguityConstraint()
        # Select PUs 1,2,3 (feature 1 contiguous) and PUs 4,6 (feature 2 split)
        selected = np.array([True, True, True, True, False, True])
        result = c.evaluate(problem, selected)
        assert not result.satisfied
        assert result.violation == 1.0
        assert "feature 2" in result.description

    def test_feature_ids_filter(self):
        """Only check specified feature IDs."""
        problem = _two_feature_problem()
        # Only enforce for feature 1 — skip feature 2
        c = FeatureContiguityConstraint(feature_ids=[1])
        # Feature 1 PUs (1,2,3) all selected and connected
        # Feature 2 PUs (4,6) selected but split — should be ignored
        selected = np.array([True, True, True, True, False, True])
        result = c.evaluate(problem, selected)
        assert result.satisfied
        assert result.violation == 0.0

    def test_no_pus_selected(self):
        """No PUs selected → vacuously satisfied."""
        problem = _make_problem(n_pu=4)
        c = FeatureContiguityConstraint()
        selected = np.array([False, False, False, False])
        result = c.evaluate(problem, selected)
        assert result.satisfied
        assert result.violation == 0.0

    def test_no_contributing_pus_selected(self):
        """Feature present but no contributing PUs selected → satisfied."""
        features = pd.DataFrame({
            "id": [1, 2],
            "name": ["sp_a", "sp_b"],
            "target": [1.0, 1.0],
            "spf": [1.0, 1.0],
        })
        # Feature 2 only in PUs 3,4
        pu_vs_features = pd.DataFrame({
            "species": [1, 1, 1, 1, 2, 2],
            "pu": [1, 2, 3, 4, 3, 4],
            "amount": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        })
        problem = _make_problem(
            n_pu=4, features=features, pu_vs_features=pu_vs_features
        )
        c = FeatureContiguityConstraint()
        # Only PUs 1 and 2 selected — feature 2 has no selected contributing PUs
        selected = np.array([True, True, False, False])
        result = c.evaluate(problem, selected)
        assert result.satisfied
        assert result.violation == 0.0


# ------------------------------------------------------------------
# penalty
# ------------------------------------------------------------------


class TestFeatureContiguityPenalty:
    def test_penalty_satisfied(self):
        problem = _make_problem(n_pu=4)
        c = FeatureContiguityConstraint(penalty_weight=500.0)
        selected = np.array([True, True, True, True])
        assert c.penalty(problem, selected) == 0.0

    def test_penalty_violated(self):
        problem = _make_problem(n_pu=4)
        c = FeatureContiguityConstraint(penalty_weight=500.0)
        selected = np.array([True, False, False, True])
        assert c.penalty(problem, selected) == 500.0


# ------------------------------------------------------------------
# MIP formulation
# ------------------------------------------------------------------


class TestFeatureContiguityMIP:
    def test_mip_enforces_feature_contiguity(self):
        """MIP must connect PUs contributing to the same feature."""
        pulp = pytest.importorskip("pulp")

        problem = _make_problem(n_pu=4)
        model = pulp.LpProblem("fc_test", pulp.LpMinimize)
        x = {
            pid: pulp.LpVariable(f"x_{pid}", cat="Binary")
            for pid in [1, 2, 3, 4]
        }

        model += pulp.lpSum(x[pid] for pid in [1, 2, 3, 4])

        # Force PU 1 and PU 4 selected
        model += x[1] >= 1, "force_1"
        model += x[4] >= 1, "force_4"

        c = FeatureContiguityConstraint()
        c.apply_to_mip(problem, model, x)

        status = model.solve(pulp.PULP_CBC_CMD(msg=0))
        assert status == pulp.constants.LpStatusOptimal

        selected_pus = {pid for pid in [1, 2, 3, 4] if x[pid].varValue > 0.5}
        assert 1 in selected_pus
        assert 4 in selected_pus
        # All 4 PUs must be selected to form connected path
        assert len(selected_pus) == 4


# ------------------------------------------------------------------
# Misc
# ------------------------------------------------------------------


class TestFeatureContiguityMisc:
    def test_name(self):
        assert FeatureContiguityConstraint().name() == "FeatureContiguityConstraint"
