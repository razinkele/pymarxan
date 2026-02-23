from pathlib import Path

import numpy as np

import pytest

from pymarxan.io.readers import load_project
from pymarxan.solvers.utils import (
    build_solution,
    check_targets,
    compute_boundary,
    compute_feature_shortfalls,
)

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "simple"


class TestComputeBoundary:
    def setup_method(self):
        self.problem = load_project(DATA_DIR)
        self.pu_ids = self.problem.planning_units["id"].tolist()
        self.pu_index = {pid: i for i, pid in enumerate(self.pu_ids)}

    def test_all_selected(self):
        selected = np.ones(6, dtype=bool)
        boundary = compute_boundary(self.problem, selected, self.pu_index)
        # All selected: only external (diagonal) boundaries contribute
        # External: PU1=2.0, PU2=1.0, PU3=1.0, PU4=1.0, PU5=1.0, PU6=2.0 = 8.0
        assert boundary == 8.0

    def test_none_selected(self):
        selected = np.zeros(6, dtype=bool)
        boundary = compute_boundary(self.problem, selected, self.pu_index)
        assert boundary == 0.0

    def test_one_selected(self):
        selected = np.array([True, False, False, False, False, False])
        boundary = compute_boundary(self.problem, selected, self.pu_index)
        # PU1 selected: external=2.0, shared with PU2=1.0 (one selected) = 3.0
        assert boundary == 3.0

    def test_no_boundary_data(self):
        self.problem.boundary = None
        selected = np.ones(6, dtype=bool)
        boundary = compute_boundary(self.problem, selected, self.pu_index)
        assert boundary == 0.0


class TestCheckTargets:
    def setup_method(self):
        self.problem = load_project(DATA_DIR)
        self.pu_ids = self.problem.planning_units["id"].tolist()
        self.pu_index = {pid: i for i, pid in enumerate(self.pu_ids)}

    def test_all_selected_meets_targets(self):
        selected = np.ones(6, dtype=bool)
        targets = check_targets(self.problem, selected, self.pu_index)
        assert all(targets.values())

    def test_none_selected(self):
        selected = np.zeros(6, dtype=bool)
        targets = check_targets(self.problem, selected, self.pu_index)
        assert not any(targets.values())


class TestBuildSolution:
    def setup_method(self):
        self.problem = load_project(DATA_DIR)

    def test_builds_valid_solution(self):
        selected = np.ones(6, dtype=bool)
        sol = build_solution(self.problem, selected, blm=1.0)
        assert sol.cost > 0
        assert sol.boundary >= 0
        assert sol.all_targets_met
        assert sol.n_selected == 6

    def test_objective_includes_blm(self):
        selected = np.ones(6, dtype=bool)
        sol = build_solution(self.problem, selected, blm=2.0)
        # All targets met => penalty == 0 => objective = cost + blm*boundary
        assert abs(sol.objective - (sol.cost + 2.0 * sol.boundary)) < 0.01

    def test_objective_includes_penalty_when_targets_unmet(self):
        """build_solution must include SPF penalty in objective."""
        selected = np.zeros(6, dtype=bool)  # Nothing selected => all targets unmet
        sol = build_solution(self.problem, selected, blm=0.0)
        # With blm=0 and nothing selected: cost=0, boundary=0
        # So objective should equal the penalty (which must be > 0)
        assert sol.cost == 0.0
        assert sol.objective > 0.0, "Objective must include SPF penalty"


class TestComputeFeatureShortfalls:
    def test_all_unselected_shortfall_equals_target(self, tiny_problem):
        """With nothing selected, shortfall equals target for each feature."""
        pu_index = {
            int(pid): i
            for i, pid in enumerate(tiny_problem.planning_units["id"])
        }
        selected = np.zeros(tiny_problem.n_planning_units, dtype=bool)
        shortfalls = compute_feature_shortfalls(tiny_problem, selected, pu_index)
        for _, frow in tiny_problem.features.iterrows():
            fid = int(frow["id"])
            target = float(frow["target"])
            assert shortfalls[fid] == pytest.approx(target)

    def test_all_selected_shortfall_non_negative(self, tiny_problem):
        """With all selected, shortfall should be 0 if targets are met."""
        pu_index = {
            int(pid): i
            for i, pid in enumerate(tiny_problem.planning_units["id"])
        }
        selected = np.ones(tiny_problem.n_planning_units, dtype=bool)
        shortfalls = compute_feature_shortfalls(tiny_problem, selected, pu_index)
        for fid, sf in shortfalls.items():
            assert sf >= 0.0  # Never negative


def test_build_solution_has_penalty_field(tiny_problem):
    """build_solution should populate a penalty field on the Solution."""
    # Select no PUs — targets are unmet, so penalty > 0
    selected = np.zeros(tiny_problem.n_planning_units, dtype=bool)
    sol = build_solution(tiny_problem, selected, blm=0.0)
    assert hasattr(sol, "penalty")
    assert sol.penalty > 0.0

    # Select all PUs — targets should be met, penalty == 0
    all_selected = np.ones(tiny_problem.n_planning_units, dtype=bool)
    sol2 = build_solution(tiny_problem, all_selected, blm=0.0)
    assert sol2.penalty == 0.0
