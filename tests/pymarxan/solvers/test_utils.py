from pathlib import Path

import numpy as np
import pytest

from pymarxan.io.readers import load_project
from pymarxan.solvers.utils import (
    build_solution,
    check_targets,
    compute_boundary,
    compute_feature_shortfalls,
    compute_objective_terms,
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


class TestComputeObjectiveTerms:
    def setup_method(self):
        self.problem = load_project(DATA_DIR)
        self.pu_ids = self.problem.planning_units["id"].tolist()
        self.pu_index = {pid: i for i, pid in enumerate(self.pu_ids)}

    def test_terms_sum_to_objective(self):
        selected = np.ones(6, dtype=bool)
        terms = compute_objective_terms(self.problem, selected, self.pu_index, blm=1.0)
        non_obj = sum(v for k, v in terms.items() if k != "objective")
        assert terms["objective"] == pytest.approx(non_obj)

    def test_terms_match_build_solution(self):
        selected = np.ones(6, dtype=bool)
        sol = build_solution(self.problem, selected, blm=1.5)
        terms = compute_objective_terms(self.problem, selected, self.pu_index, blm=1.5)
        assert sol.objective == pytest.approx(terms["objective"])

    def test_terms_keys(self):
        selected = np.zeros(6, dtype=bool)
        terms = compute_objective_terms(self.problem, selected, self.pu_index, blm=0.0)
        assert "base" in terms
        assert "boundary" in terms
        assert "penalty" in terms
        assert "cost_threshold" in terms
        assert "probability" in terms
        assert "connectivity" in terms
        assert "objective" in terms

    def test_cost_threshold_term(self):
        """When COSTTHRESH is set, the cost_threshold term should be nonzero."""
        self.problem.parameters["COSTTHRESH"] = 1.0
        self.problem.parameters["THRESHPEN1"] = 10.0
        self.problem.parameters["THRESHPEN2"] = 5.0
        selected = np.ones(6, dtype=bool)
        terms = compute_objective_terms(self.problem, selected, self.pu_index, blm=0.0)
        assert terms["cost_threshold"] > 0.0
        assert terms["objective"] == pytest.approx(
            terms["base"] + terms["boundary"] + terms["penalty"]
            + terms["cost_threshold"] + terms["probability"]
            + terms["connectivity"]
        )


class TestProbabilityObjectiveTerm:
    """Test probability risk premium in objective."""

    def setup_method(self):
        import pandas as pd

        self.pu = pd.DataFrame({
            "id": [1, 2, 3],
            "cost": [10.0, 20.0, 30.0],
            "status": [0, 0, 0],
        })
        self.features = pd.DataFrame({
            "id": [1],
            "name": ["sp_a"],
            "target": [10.0],
            "spf": [1.0],
        })
        self.puvspr = pd.DataFrame({
            "species": [1, 1, 1],
            "pu": [1, 2, 3],
            "amount": [5.0, 5.0, 5.0],
        })
        self.prob = pd.DataFrame({
            "pu": [1, 2, 3],
            "probability": [0.1, 0.5, 0.9],
        })

    def _make_problem(self, **kw):
        from pymarxan.models.problem import ConservationProblem

        return ConservationProblem(
            planning_units=self.pu,
            features=self.features,
            pu_vs_features=self.puvspr,
            parameters=kw.pop("parameters", {"BLM": 0.0}),
            **kw,
        )

    def test_no_probability_data(self):
        problem = self._make_problem()
        pu_index = {1: 0, 2: 1, 3: 2}
        selected = np.array([True, True, False])
        terms = compute_objective_terms(problem, selected, pu_index, blm=0.0)
        assert terms["probability"] == 0.0

    def test_mode1_risk_premium(self):
        problem = self._make_problem(
            probability=self.prob,
            parameters={"BLM": 0.0, "PROBMODE": 1, "PROBABILITYWEIGHTING": 2.0},
        )
        pu_index = {1: 0, 2: 1, 3: 2}
        selected = np.array([True, True, False])
        terms = compute_objective_terms(problem, selected, pu_index, blm=0.0)
        # Risk premium = 2.0 * (0.1*10 + 0.5*20) = 2.0 * 11.0 = 22.0
        assert terms["probability"] == pytest.approx(22.0)

    def test_mode2_no_penalty_term(self):
        """Mode 2 modifies pu_feat_matrix, not an explicit penalty."""
        problem = self._make_problem(
            probability=self.prob,
            parameters={"BLM": 0.0, "PROBMODE": 2},
        )
        pu_index = {1: 0, 2: 1, 3: 2}
        selected = np.array([True, True, False])
        terms = compute_objective_terms(problem, selected, pu_index, blm=0.0)
        assert terms["probability"] == 0.0

    def test_zero_weight_no_penalty(self):
        problem = self._make_problem(
            probability=self.prob,
            parameters={"BLM": 0.0, "PROBABILITYWEIGHTING": 0.0},
        )
        pu_index = {1: 0, 2: 1, 3: 2}
        selected = np.array([True, True, True])
        terms = compute_objective_terms(problem, selected, pu_index, blm=0.0)
        assert terms["probability"] == 0.0

    def test_probability_included_in_objective(self):
        problem = self._make_problem(
            probability=self.prob,
            parameters={"BLM": 0.0, "PROBABILITYWEIGHTING": 1.0},
        )
        pu_index = {1: 0, 2: 1, 3: 2}
        selected = np.array([True, True, True])
        terms = compute_objective_terms(problem, selected, pu_index, blm=0.0)
        non_obj = sum(v for k, v in terms.items() if k != "objective")
        assert terms["objective"] == pytest.approx(non_obj)
        assert terms["probability"] > 0.0

    def test_terms_has_probability_key(self):
        problem = self._make_problem()
        pu_index = {1: 0, 2: 1, 3: 2}
        selected = np.array([True, False, False])
        terms = compute_objective_terms(problem, selected, pu_index, blm=0.0)
        assert "probability" in terms


class TestConnectivityObjectiveTerm:
    """Test connectivity penalty/bonus in objective."""

    def setup_method(self):
        import pandas as pd

        self.pu = pd.DataFrame({
            "id": [1, 2, 3, 4],
            "cost": [10.0, 20.0, 30.0, 40.0],
            "status": [0, 0, 0, 0],
        })
        self.features = pd.DataFrame({
            "id": [1],
            "name": ["sp_a"],
            "target": [5.0],
            "spf": [1.0],
        })
        self.puvspr = pd.DataFrame({
            "species": [1, 1, 1, 1],
            "pu": [1, 2, 3, 4],
            "amount": [5.0, 5.0, 5.0, 5.0],
        })
        # Connectivity: 1-2 (value 3.0), 2-3 (value 2.0), 3-4 (value 1.0)
        self.conn = pd.DataFrame({
            "id1": [1, 2, 3],
            "id2": [2, 3, 4],
            "value": [3.0, 2.0, 1.0],
        })

    def _make_problem(self, **kw):
        from pymarxan.models.problem import ConservationProblem

        return ConservationProblem(
            planning_units=self.pu,
            features=self.features,
            pu_vs_features=self.puvspr,
            parameters=kw.pop("parameters", {"BLM": 0.0}),
            **kw,
        )

    def test_no_connectivity_data(self):
        problem = self._make_problem()
        pu_index = {1: 0, 2: 1, 3: 2, 4: 3}
        selected = np.array([True, True, False, False])
        terms = compute_objective_terms(problem, selected, pu_index, blm=0.0)
        assert terms["connectivity"] == 0.0

    def test_zero_weight_no_penalty(self):
        problem = self._make_problem(
            connectivity=self.conn,
            parameters={"BLM": 0.0, "CONNECTIVITY_WEIGHT": 0.0},
        )
        pu_index = {1: 0, 2: 1, 3: 2, 4: 3}
        selected = np.array([True, True, False, False])
        terms = compute_objective_terms(problem, selected, pu_index, blm=0.0)
        assert terms["connectivity"] == 0.0

    def test_both_selected_bonus(self):
        """Both PUs in a connected pair selected → negative (bonus)."""
        problem = self._make_problem(
            connectivity=self.conn,
            parameters={"BLM": 0.0, "CONNECTIVITY_WEIGHT": 1.0},
        )
        pu_index = {1: 0, 2: 1, 3: 2, 4: 3}
        # Select PU 1 and 2 (connected with value 3.0)
        # Edge 1-2: both selected → -3.0
        # Edge 2-3: only PU 2 selected → +2.0
        # Edge 3-4: neither selected → 0
        # Total = 1.0 * (-3.0 + 2.0) = -1.0
        selected = np.array([True, True, False, False])
        terms = compute_objective_terms(problem, selected, pu_index, blm=0.0)
        assert terms["connectivity"] == pytest.approx(-1.0)

    def test_one_selected_penalty(self):
        """Only one PU in a connected pair selected → positive (penalty)."""
        problem = self._make_problem(
            connectivity=self.conn,
            parameters={"BLM": 0.0, "CONNECTIVITY_WEIGHT": 1.0},
        )
        pu_index = {1: 0, 2: 1, 3: 2, 4: 3}
        # Select only PU 2
        # Edge 1-2: only PU 2 → +3.0
        # Edge 2-3: only PU 2 → +2.0
        # Edge 3-4: neither → 0
        # Total = 1.0 * (3.0 + 2.0) = 5.0
        selected = np.array([False, True, False, False])
        terms = compute_objective_terms(problem, selected, pu_index, blm=0.0)
        assert terms["connectivity"] == pytest.approx(5.0)

    def test_all_selected_full_bonus(self):
        """All PUs selected → maximum connectivity bonus."""
        problem = self._make_problem(
            connectivity=self.conn,
            parameters={"BLM": 0.0, "CONNECTIVITY_WEIGHT": 2.0},
        )
        pu_index = {1: 0, 2: 1, 3: 2, 4: 3}
        # All edges have both selected: -(3+2+1) = -6.0
        # Total = 2.0 * -6.0 = -12.0
        selected = np.array([True, True, True, True])
        terms = compute_objective_terms(problem, selected, pu_index, blm=0.0)
        assert terms["connectivity"] == pytest.approx(-12.0)

    def test_none_selected_zero(self):
        """No PUs selected → zero connectivity contribution."""
        problem = self._make_problem(
            connectivity=self.conn,
            parameters={"BLM": 0.0, "CONNECTIVITY_WEIGHT": 1.0},
        )
        pu_index = {1: 0, 2: 1, 3: 2, 4: 3}
        selected = np.array([False, False, False, False])
        terms = compute_objective_terms(problem, selected, pu_index, blm=0.0)
        assert terms["connectivity"] == 0.0

    def test_connectivity_included_in_objective(self):
        problem = self._make_problem(
            connectivity=self.conn,
            parameters={"BLM": 0.0, "CONNECTIVITY_WEIGHT": 1.5},
        )
        pu_index = {1: 0, 2: 1, 3: 2, 4: 3}
        selected = np.array([True, True, True, False])
        terms = compute_objective_terms(problem, selected, pu_index, blm=0.0)
        non_obj = sum(v for k, v in terms.items() if k != "objective")
        assert terms["objective"] == pytest.approx(non_obj)
        assert terms["connectivity"] != 0.0

    def test_terms_has_connectivity_key(self):
        problem = self._make_problem()
        pu_index = {1: 0, 2: 1, 3: 2, 4: 3}
        selected = np.array([True, False, False, False])
        terms = compute_objective_terms(problem, selected, pu_index, blm=0.0)
        assert "connectivity" in terms
