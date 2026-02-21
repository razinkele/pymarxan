from pathlib import Path

import numpy as np

from pymarxan.io.readers import load_project
from pymarxan.solvers.utils import build_solution, check_targets, compute_boundary

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
        assert abs(sol.objective - (sol.cost + 2.0 * sol.boundary)) < 0.01
