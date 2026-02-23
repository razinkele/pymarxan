"""Tests for COSTTHRESH penalty in build_solution."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pymarxan.io.readers import load_project
from pymarxan.solvers.utils import build_solution, compute_cost_threshold_penalty

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "simple"


class TestCostThresholdPenalty:
    def test_no_penalty_below_threshold(self):
        """Cost below threshold should add no penalty."""
        penalty = compute_cost_threshold_penalty(
            total_cost=50.0, cost_thresh=100.0, thresh_pen1=10.0, thresh_pen2=1.0
        )
        assert penalty == 0.0

    def test_penalty_above_threshold(self):
        """Cost above threshold should add penalty."""
        penalty = compute_cost_threshold_penalty(
            total_cost=150.0, cost_thresh=100.0, thresh_pen1=10.0, thresh_pen2=1.0
        )
        assert penalty > 0.0

    def test_build_solution_applies_costthresh(self):
        """build_solution should include COSTTHRESH penalty in objective."""
        problem = load_project(DATA_DIR)
        selected = np.ones(problem.n_planning_units, dtype=bool)
        blm = 0.0

        sol_no_thresh = build_solution(problem, selected, blm)

        problem.parameters["COSTTHRESH"] = 1.0
        problem.parameters["THRESHPEN1"] = 10.0
        problem.parameters["THRESHPEN2"] = 1.0
        sol_thresh = build_solution(problem, selected, blm)

        assert sol_thresh.objective > sol_no_thresh.objective

    def test_costthresh_zero_no_effect(self):
        """COSTTHRESH=0 should not change the objective."""
        problem = load_project(DATA_DIR)
        selected = np.ones(problem.n_planning_units, dtype=bool)
        blm = 0.0

        problem.parameters["COSTTHRESH"] = 0.0
        problem.parameters["THRESHPEN1"] = 10.0
        sol = build_solution(problem, selected, blm)

        del problem.parameters["COSTTHRESH"]
        sol_default = build_solution(problem, selected, blm)

        assert sol.objective == pytest.approx(sol_default.objective)
