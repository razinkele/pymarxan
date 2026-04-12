"""Tests for ContiguityConstraint."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pymarxan.constraints.contiguity import (
    ContiguityConstraint,
    count_connected_components,
)
from pymarxan.models.problem import ConservationProblem


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _make_line_problem(n: int = 4) -> ConservationProblem:
    """Create a line graph problem: 1-2-3-...-n."""
    planning_units = pd.DataFrame({
        "id": list(range(1, n + 1)),
        "cost": [1.0] * n,
        "status": [0] * n,
    })
    features = pd.DataFrame({
        "id": [1],
        "name": ["sp_a"],
        "target": [1.0],
        "spf": [1.0],
    })
    pu_vs_features = pd.DataFrame({
        "species": [1] * n,
        "pu": list(range(1, n + 1)),
        "amount": [1.0] * n,
    })
    boundary = pd.DataFrame({
        "id1": list(range(1, n)),
        "id2": list(range(2, n + 1)),
        "boundary": [1.0] * (n - 1),
    })
    return ConservationProblem(
        planning_units=planning_units,
        features=features,
        pu_vs_features=pu_vs_features,
        boundary=boundary,
        parameters={"BLM": 0.0},
    )


def _make_zero_target_problem() -> ConservationProblem:
    """Problem where all feature targets are zero."""
    planning_units = pd.DataFrame({
        "id": [1, 2, 3, 4],
        "cost": [1.0, 1.0, 1.0, 1.0],
        "status": [0, 0, 0, 0],
    })
    features = pd.DataFrame({
        "id": [1],
        "name": ["sp_a"],
        "target": [0.0],
        "spf": [1.0],
    })
    pu_vs_features = pd.DataFrame({
        "species": [1, 1, 1, 1],
        "pu": [1, 2, 3, 4],
        "amount": [1.0, 1.0, 1.0, 1.0],
    })
    boundary = pd.DataFrame({
        "id1": [1, 2, 3],
        "id2": [2, 3, 4],
        "boundary": [1.0, 1.0, 1.0],
    })
    return ConservationProblem(
        planning_units=planning_units,
        features=features,
        pu_vs_features=pu_vs_features,
        boundary=boundary,
        parameters={"BLM": 0.0},
    )


# ------------------------------------------------------------------
# count_connected_components helper
# ------------------------------------------------------------------

class TestCountConnectedComponents:
    def test_empty(self):
        assert count_connected_components(set(), {}) == 0

    def test_single_node(self):
        assert count_connected_components({1}, {}) == 1

    def test_connected(self):
        adj = {1: {2}, 2: {1, 3}, 3: {2}}
        assert count_connected_components({1, 2, 3}, adj) == 1

    def test_two_components(self):
        adj = {1: {2}, 2: {1}, 3: {4}, 4: {3}}
        assert count_connected_components({1, 2, 3, 4}, adj) == 2

    def test_subset_connected(self):
        adj = {1: {2}, 2: {1, 3}, 3: {2, 4}, 4: {3}}
        assert count_connected_components({1, 2}, adj) == 1

    def test_subset_disconnected(self):
        adj = {1: {2}, 2: {1, 3}, 3: {2, 4}, 4: {3}}
        # Select only PUs 1 and 4 — not connected
        assert count_connected_components({1, 4}, adj) == 2


# ------------------------------------------------------------------
# ContiguityConstraint.evaluate
# ------------------------------------------------------------------

class TestContiguityEvaluate:
    def test_single_connected_component(self):
        """All adjacent PUs selected → satisfied."""
        problem = _make_line_problem(4)
        c = ContiguityConstraint()
        selected = np.array([True, True, True, True])
        result = c.evaluate(problem, selected)
        assert result.satisfied
        assert result.violation == 0.0

    def test_two_disconnected_groups(self):
        """PUs 1 and 4 selected (not connected) → violated."""
        problem = _make_line_problem(4)
        c = ContiguityConstraint()
        selected = np.array([True, False, False, True])
        result = c.evaluate(problem, selected)
        assert not result.satisfied
        assert result.violation == 1.0

    def test_no_pus_selected(self):
        """No PUs selected → vacuously satisfied."""
        problem = _make_line_problem(4)
        c = ContiguityConstraint()
        selected = np.array([False, False, False, False])
        result = c.evaluate(problem, selected)
        assert result.satisfied
        assert result.violation == 0.0

    def test_single_pu_selected(self):
        """Single PU selected → satisfied."""
        problem = _make_line_problem(4)
        c = ContiguityConstraint()
        selected = np.array([False, True, False, False])
        result = c.evaluate(problem, selected)
        assert result.satisfied
        assert result.violation == 0.0


# ------------------------------------------------------------------
# ContiguityConstraint.penalty
# ------------------------------------------------------------------

class TestContiguityPenalty:
    def test_penalty_satisfied(self):
        problem = _make_line_problem(4)
        c = ContiguityConstraint(penalty_weight=500.0)
        selected = np.array([True, True, True, True])
        assert c.penalty(problem, selected) == 0.0

    def test_penalty_violated(self):
        problem = _make_line_problem(4)
        c = ContiguityConstraint(penalty_weight=500.0)
        # PUs 1 and 4 → 2 components → violation=1
        selected = np.array([True, False, False, True])
        assert c.penalty(problem, selected) == 500.0


# ------------------------------------------------------------------
# MIP formulation
# ------------------------------------------------------------------

class TestContiguityMIP:
    def test_mip_produces_connected_result(self):
        """4-PU line, features on PU 1 and 4 → MIP must connect them."""
        pulp = pytest.importorskip("pulp")

        planning_units = pd.DataFrame({
            "id": [1, 2, 3, 4],
            "cost": [1.0, 1.0, 1.0, 1.0],
            "status": [0, 0, 0, 0],
        })
        features = pd.DataFrame({
            "id": [1, 2],
            "name": ["sp_a", "sp_b"],
            "target": [1.0, 1.0],
            "spf": [1.0, 1.0],
        })
        # Feature 1 only in PU 1, feature 2 only in PU 4
        pu_vs_features = pd.DataFrame({
            "species": [1, 2],
            "pu": [1, 4],
            "amount": [1.0, 1.0],
        })
        boundary = pd.DataFrame({
            "id1": [1, 2, 3],
            "id2": [2, 3, 4],
            "boundary": [1.0, 1.0, 1.0],
        })
        problem = ConservationProblem(
            planning_units=planning_units,
            features=features,
            pu_vs_features=pu_vs_features,
            boundary=boundary,
            parameters={"BLM": 0.0},
        )

        model = pulp.LpProblem("contiguity_test", pulp.LpMinimize)
        x = {
            pid: pulp.LpVariable(f"x_{pid}", cat="Binary")
            for pid in [1, 2, 3, 4]
        }

        # Objective: minimize cost
        model += pulp.lpSum(x[pid] * 1.0 for pid in [1, 2, 3, 4])

        # Feature coverage: PU 1 must be selected (feature 1), PU 4 must be selected (feature 2)
        model += x[1] >= 1, "feat1_target"
        model += x[4] >= 1, "feat2_target"

        # Apply contiguity constraint
        c = ContiguityConstraint()
        c.apply_to_mip(problem, model, x)

        status = model.solve(pulp.PULP_CBC_CMD(msg=0))
        assert status == pulp.constants.LpStatusOptimal

        selected_pus = {pid for pid in [1, 2, 3, 4] if x[pid].varValue > 0.5}
        # Must include PU 1 and 4, and connecting PUs 2 and 3
        assert 1 in selected_pus
        assert 4 in selected_pus
        # Verify contiguity
        adj = {1: {2}, 2: {1, 3}, 3: {2, 4}, 4: {3}}
        assert count_connected_components(selected_pus, adj) == 1

    def test_mip_skips_zero_targets(self):
        """MIP with all zero targets → contiguity constraints not added."""
        pulp = pytest.importorskip("pulp")

        problem = _make_zero_target_problem()
        model = pulp.LpProblem("skip_test", pulp.LpMinimize)
        x = {
            pid: pulp.LpVariable(f"x_{pid}", cat="Binary")
            for pid in [1, 2, 3, 4]
        }
        model += pulp.lpSum(x[pid] for pid in [1, 2, 3, 4])

        c = ContiguityConstraint()
        c.apply_to_mip(problem, model, x)

        # No contiguity constraints should have been added
        contiguity_constraints = [
            name for name in model.constraints if "contiguity" in name
        ]
        assert len(contiguity_constraints) == 0


# ------------------------------------------------------------------
# Misc
# ------------------------------------------------------------------

class TestContiguityMisc:
    def test_name(self):
        assert ContiguityConstraint().name() == "ContiguityConstraint"
