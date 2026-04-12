import pandas as pd

from pymarxan.models.problem import ConservationProblem


def _make_simple_problem() -> ConservationProblem:
    planning_units = pd.DataFrame({
        "id": [1, 2, 3],
        "cost": [10.0, 15.0, 20.0],
        "status": [0, 0, 0],
    })
    features = pd.DataFrame({
        "id": [1, 2],
        "name": ["sp_a", "sp_b"],
        "target": [20.0, 10.0],
        "spf": [1.0, 1.0],
    })
    pu_vs_features = pd.DataFrame({
        "species": [1, 1, 1, 2, 2],
        "pu": [1, 2, 3, 1, 3],
        "amount": [10.0, 15.0, 5.0, 8.0, 12.0],
    })
    boundary = pd.DataFrame({
        "id1": [1, 2],
        "id2": [2, 3],
        "boundary": [1.0, 1.0],
    })
    return ConservationProblem(
        planning_units=planning_units,
        features=features,
        pu_vs_features=pu_vs_features,
        boundary=boundary,
        parameters={"BLM": 1.0},
    )


class TestConservationProblem:
    def test_create_problem(self):
        problem = _make_simple_problem()
        assert problem.n_planning_units == 3
        assert problem.n_features == 2

    def test_validate_valid_problem(self):
        problem = _make_simple_problem()
        errors = problem.validate()
        assert errors == []

    def test_validate_missing_pu_in_puvspr(self):
        problem = _make_simple_problem()
        extra = pd.DataFrame({"species": [1], "pu": [99], "amount": [5.0]})
        problem.pu_vs_features = pd.concat([problem.pu_vs_features, extra], ignore_index=True)
        errors = problem.validate()
        assert any("planning unit" in e.lower() for e in errors)

    def test_validate_missing_feature_in_puvspr(self):
        problem = _make_simple_problem()
        extra = pd.DataFrame({"species": [99], "pu": [1], "amount": [5.0]})
        problem.pu_vs_features = pd.concat([problem.pu_vs_features, extra], ignore_index=True)
        errors = problem.validate()
        assert any("feature" in e.lower() for e in errors)

    def test_feature_amounts_per_pu(self):
        problem = _make_simple_problem()
        amounts = problem.feature_amounts()
        assert amounts[1] == 30.0
        assert amounts[2] == 20.0

    def test_targets_achievable(self):
        problem = _make_simple_problem()
        assert problem.targets_achievable()

    def test_targets_not_achievable(self):
        problem = _make_simple_problem()
        problem.features.loc[problem.features["id"] == 1, "target"] = 999.0
        assert not problem.targets_achievable()

    def test_summary_returns_string(self):
        problem = _make_simple_problem()
        s = problem.summary()
        assert "3 planning units" in s
        assert "2 features" in s

    def test_no_boundary(self):
        problem = _make_simple_problem()
        problem.boundary = None
        errors = problem.validate()
        assert errors == []

    def test_copy_with_preserves_fields(self):
        problem = _make_simple_problem()
        copied = problem.copy_with(parameters={"BLM": 2.0})
        assert copied.parameters == {"BLM": 2.0}
        # Original unchanged
        assert problem.parameters == {"BLM": 1.0}
        # Other fields are shared (shallow copy)
        assert copied.planning_units is problem.planning_units
        assert copied.features is problem.features
        assert copied.pu_vs_features is problem.pu_vs_features
        assert copied.boundary is problem.boundary

    def test_copy_with_replace_dataframe(self):
        problem = _make_simple_problem()
        new_features = problem.features.copy()
        new_features.loc[new_features["id"] == 1, "target"] = 999.0
        copied = problem.copy_with(features=new_features)
        assert copied.features.loc[copied.features["id"] == 1, "target"].iloc[0] == 999.0
        # Original unchanged
        assert problem.features.loc[problem.features["id"] == 1, "target"].iloc[0] == 20.0

    def test_copy_with_no_overrides(self):
        problem = _make_simple_problem()
        copied = problem.copy_with()
        assert copied.n_planning_units == problem.n_planning_units
        assert copied is not problem
