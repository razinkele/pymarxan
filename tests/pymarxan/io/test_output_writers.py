"""Tests for Marxan output file writers (mvbest, ssoln, sum)."""

import numpy as np
import pandas as pd
import pytest

from pymarxan.io.writers import write_mvbest, write_ssoln, write_sum
from pymarxan.models.problem import ConservationProblem
from pymarxan.solvers.base import Solution


@pytest.fixture()
def simple_problem():
    """A small problem with 3 PUs and 2 features."""
    pu = pd.DataFrame({"id": [1, 2, 3], "cost": [10.0, 20.0, 30.0], "status": [0, 0, 0]})
    features = pd.DataFrame({
        "id": [1, 2],
        "name": ["feat_a", "feat_b"],
        "target": [15.0, 25.0],
        "spf": [1.0, 1.0],
    })
    puvspr = pd.DataFrame({
        "species": [1, 1, 2, 2],
        "pu": [1, 2, 2, 3],
        "amount": [10.0, 8.0, 12.0, 20.0],
    })
    return ConservationProblem(
        planning_units=pu,
        features=features,
        pu_vs_features=puvspr,
    )


@pytest.fixture()
def solution_all_selected():
    """Solution where all 3 PUs are selected."""
    return Solution(
        selected=np.array([True, True, True]),
        cost=60.0,
        boundary=5.0,
        objective=70.0,
        targets_met={1: True, 2: True},
        penalty=5.0,
    )


@pytest.fixture()
def solution_partial():
    """Solution where only PU 1 and PU 2 are selected."""
    return Solution(
        selected=np.array([True, True, False]),
        cost=30.0,
        boundary=3.0,
        objective=40.0,
        targets_met={1: True, 2: False},
    )


class TestWriteMvbest:
    def test_correct_columns(self, tmp_path, simple_problem, solution_all_selected):
        path = tmp_path / "mvbest.csv"
        write_mvbest(simple_problem, solution_all_selected, path)
        df = pd.read_csv(path)
        expected_cols = [
            "Feature_ID", "Feature_Name", "Target",
            "Amount_Held", "Target_Met", "Shortfall",
        ]
        assert list(df.columns) == expected_cols

    def test_two_feature_rows(self, tmp_path, simple_problem, solution_all_selected):
        path = tmp_path / "mvbest.csv"
        write_mvbest(simple_problem, solution_all_selected, path)
        df = pd.read_csv(path)
        assert len(df) == 2

    def test_amount_held_values(self, tmp_path, simple_problem, solution_all_selected):
        """When all PUs selected: feat_a gets 10+8=18, feat_b gets 12+20=32."""
        path = tmp_path / "mvbest.csv"
        write_mvbest(simple_problem, solution_all_selected, path)
        df = pd.read_csv(path)
        feat_a = df[df["Feature_ID"] == 1].iloc[0]
        assert feat_a["Amount_Held"] == pytest.approx(18.0)
        assert feat_a["Target_Met"] is True or feat_a["Target_Met"] == 1
        assert feat_a["Shortfall"] == pytest.approx(0.0)

    def test_shortfall_when_target_not_met(self, tmp_path, simple_problem, solution_partial):
        """PU 1+2 selected: feat_b gets 12 vs target 25 => shortfall=13."""
        path = tmp_path / "mvbest.csv"
        write_mvbest(simple_problem, solution_partial, path)
        df = pd.read_csv(path)
        feat_b = df[df["Feature_ID"] == 2].iloc[0]
        assert feat_b["Amount_Held"] == pytest.approx(12.0)
        assert feat_b["Shortfall"] == pytest.approx(13.0)


class TestWriteSsoln:
    def test_correct_columns(self, tmp_path, simple_problem, solution_all_selected):
        path = tmp_path / "ssoln.csv"
        write_ssoln(simple_problem, [solution_all_selected], path)
        df = pd.read_csv(path)
        assert list(df.columns) == ["Planning_Unit", "Number"]

    def test_selection_counts(
        self, tmp_path, simple_problem, solution_all_selected, solution_partial,
    ):
        """PU 1 selected in both => 2, PU 2 selected in both => 2, PU 3 only in first => 1."""
        path = tmp_path / "ssoln.csv"
        write_ssoln(simple_problem, [solution_all_selected, solution_partial], path)
        df = pd.read_csv(path)
        counts = dict(zip(df["Planning_Unit"], df["Number"]))
        assert counts[1] == 2
        assert counts[2] == 2
        assert counts[3] == 1

    def test_empty_solutions_writes_zeros(self, tmp_path, simple_problem):
        path = tmp_path / "ssoln.csv"
        write_ssoln(simple_problem, [], path)
        df = pd.read_csv(path)
        assert len(df) == 3
        assert all(df["Number"] == 0)


class TestWriteSum:
    def test_correct_columns(self, tmp_path, solution_all_selected):
        path = tmp_path / "sum.csv"
        write_sum([solution_all_selected], path)
        df = pd.read_csv(path)
        expected_cols = [
            "Run", "Score", "Cost", "Planning_Units",
            "Boundary", "Penalty", "Shortfall",
        ]
        assert list(df.columns) == expected_cols

    def test_per_run_rows(self, tmp_path, solution_all_selected, solution_partial):
        path = tmp_path / "sum.csv"
        write_sum([solution_all_selected, solution_partial], path)
        df = pd.read_csv(path)
        assert len(df) == 2
        assert list(df["Run"]) == [1, 2]

    def test_values(self, tmp_path, solution_all_selected):
        """objective=70, cost=60, boundary=5 => penalty=70-60-5=5."""
        path = tmp_path / "sum.csv"
        write_sum([solution_all_selected], path)
        df = pd.read_csv(path)
        row = df.iloc[0]
        assert row["Score"] == pytest.approx(70.0)
        assert row["Cost"] == pytest.approx(60.0)
        assert row["Planning_Units"] == 3
        assert row["Boundary"] == pytest.approx(5.0)
        assert row["Penalty"] == pytest.approx(5.0)

    def test_penalty_clamped_to_zero(self, tmp_path):
        """When objective < cost + boundary, penalty should be 0."""
        sol = Solution(
            selected=np.array([True, False]),
            cost=10.0,
            boundary=5.0,
            objective=12.0,  # 12 - 10 - 5 = -3, clamped to 0
            targets_met={1: True},
        )
        path = tmp_path / "sum.csv"
        write_sum([sol], path)
        df = pd.read_csv(path)
        assert df.iloc[0]["Penalty"] == pytest.approx(0.0)

    def test_write_sum_shortfall_differs_from_penalty(self, tmp_path):
        """Shortfall column must be raw shortfall, not SPF-weighted penalty."""
        sol = Solution(
            selected=np.array([True, False, True]),
            cost=40.0, boundary=5.0, objective=95.0,
            targets_met={1: True, 2: False},
            penalty=50.0, shortfall=5.0,
        )
        path = tmp_path / "sum.csv"
        write_sum([sol], path)
        df = pd.read_csv(path)
        assert df.iloc[0]["Shortfall"] == pytest.approx(5.0)
        assert df.iloc[0]["Penalty"] == pytest.approx(50.0)
