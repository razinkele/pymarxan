"""Tests for MISSLEVEL and COSTTHRESH parameter support in utils."""
from __future__ import annotations

import numpy as np
import pandas as pd

from pymarxan.models.problem import ConservationProblem
from pymarxan.solvers.utils import (
    check_targets,
    compute_cost_threshold_penalty,
    compute_objective,
)


def _make_problem(
    parameters: dict | None = None,
) -> tuple[ConservationProblem, np.ndarray, dict[int, int]]:
    """Build a minimal problem with one feature.

    Planning units 1-3, cost 10 each.
    Feature 1 target=10, amounts: PU1=3, PU2=3, PU3=3  (total if all selected=9).
    Selecting all 3 PUs yields amount=9 against target=10.
    """
    pu = pd.DataFrame({"id": [1, 2, 3], "cost": [10.0, 10.0, 10.0], "status": [0, 0, 0]})
    features = pd.DataFrame({"id": [1], "name": ["feat_a"], "target": [10.0], "spf": [1.0]})
    puvspr = pd.DataFrame(
        {
            "species": [1, 1, 1],
            "pu": [1, 2, 3],
            "amount": [3.0, 3.0, 3.0],
        }
    )
    problem = ConservationProblem(
        planning_units=pu,
        features=features,
        pu_vs_features=puvspr,
        boundary=None,
        parameters=parameters or {},
    )
    selected = np.ones(3, dtype=bool)  # all selected -> total amount = 9
    pu_index = {1: 0, 2: 1, 3: 2}
    return problem, selected, pu_index


# ---------- MISSLEVEL ----------


class TestMisslevelDefault:
    def test_misslevel_default(self):
        """MISSLEVEL=1.0 (default): 9 out of 10 target -> not met."""
        problem, selected, pu_index = _make_problem()
        targets = check_targets(problem, selected, pu_index)
        assert targets[1] is False


class TestMisslevelRelaxed:
    def test_misslevel_relaxed(self):
        """MISSLEVEL=0.9: 9 out of 10 target -> met (9 >= 10*0.9=9.0)."""
        problem, selected, pu_index = _make_problem({"MISSLEVEL": "0.9"})
        targets = check_targets(problem, selected, pu_index)
        assert targets[1] is True


class TestMisslevelStrict:
    def test_misslevel_strict(self):
        """MISSLEVEL=0.95: 9 out of 10 target -> not met (9 < 10*0.95=9.5)."""
        problem, selected, pu_index = _make_problem({"MISSLEVEL": "0.95"})
        targets = check_targets(problem, selected, pu_index)
        assert targets[1] is False


# ---------- COSTTHRESH ----------


class TestCostThresholdNoPenalty:
    def test_cost_threshold_no_penalty(self):
        """Cost below threshold -> 0.0."""
        result = compute_cost_threshold_penalty(
            total_cost=40.0,
            cost_thresh=50.0,
            thresh_pen1=10.0,
            thresh_pen2=2.0,
        )
        assert result == 0.0


class TestCostThresholdWithPenalty:
    def test_cost_threshold_with_penalty(self):
        """cost=60, thresh=50, pen1=10, pen2=2 -> 10 + 2*(60-50) = 30.0."""
        result = compute_cost_threshold_penalty(
            total_cost=60.0,
            cost_thresh=50.0,
            thresh_pen1=10.0,
            thresh_pen2=2.0,
        )
        assert result == 30.0


class TestCostThresholdInObjective:
    def test_cost_threshold_in_objective(self):
        """COSTTHRESH affects compute_objective result.

        With 3 PUs at cost=10 each, total_cost=30.
        No boundary, feature shortfall = max(0, 10-9)=1, spf=1 -> penalty=1.
        Without COSTTHRESH: obj = 30 + 0 + 1 = 31.
        With COSTTHRESH=20, THRESHPEN1=5, THRESHPEN2=1:
            cost_thresh_penalty = 5 + 1*(30-20) = 15
            obj = 31 + 15 = 46.
        """
        # Without COSTTHRESH
        problem_no_thresh, selected, pu_index = _make_problem()
        obj_no = compute_objective(problem_no_thresh, selected, pu_index, blm=0.0)
        assert obj_no == 31.0

        # With COSTTHRESH
        problem_with_thresh, selected, pu_index = _make_problem(
            {"COSTTHRESH": "20.0", "THRESHPEN1": "5.0", "THRESHPEN2": "1.0"}
        )
        obj_with = compute_objective(problem_with_thresh, selected, pu_index, blm=0.0)
        assert obj_with == 46.0
