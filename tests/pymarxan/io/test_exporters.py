from pathlib import Path

import numpy as np
import pandas as pd

from pymarxan.analysis.selection_freq import compute_selection_frequency
from pymarxan.io.exporters import (
    export_selection_frequency_csv,
    export_solution_csv,
    export_summary_csv,
)
from pymarxan.io.readers import load_project
from pymarxan.models.problem import ConservationProblem
from pymarxan.solvers.base import Solution

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "simple"


def _make_solution(problem):
    return Solution(
        selected=np.array([True, True, False, True, False, True]),
        cost=45.0, boundary=3.0, objective=48.0,
        targets_met={1: True, 2: True, 3: True},
        metadata={"solver": "test"},
    )


class TestExportSolutionCSV:
    def test_creates_csv(self, tmp_path):
        problem = load_project(DATA_DIR)
        sol = _make_solution(problem)
        out = tmp_path / "solution.csv"
        export_solution_csv(problem, sol, out)
        assert out.exists()
        df = pd.read_csv(out)
        assert "planning_unit" in df.columns
        assert "selected" in df.columns
        assert "cost" in df.columns
        assert len(df) == 6
        assert df["selected"].sum() == 4


class TestExportSummaryCSV:
    def test_creates_csv(self, tmp_path):
        problem = load_project(DATA_DIR)
        sol = _make_solution(problem)
        out = tmp_path / "summary.csv"
        export_summary_csv(problem, sol, out)
        assert out.exists()
        df = pd.read_csv(out)
        assert "feature_id" in df.columns
        assert "target" in df.columns
        assert "achieved" in df.columns
        assert "met" in df.columns
        assert len(df) == 3


    def test_respects_misslevel(self, tmp_path):
        pu = pd.DataFrame({"id": [1, 2], "cost": [10.0, 20.0], "status": [0, 0]})
        features = pd.DataFrame({"id": [1], "name": ["f1"], "target": [10.0], "spf": [1.0]})
        puvspr = pd.DataFrame({"species": [1, 1], "pu": [1, 2], "amount": [4.0, 3.0]})
        problem = ConservationProblem(
            planning_units=pu, features=features, pu_vs_features=puvspr,
            parameters={"MISSLEVEL": 0.5},
        )
        sol = Solution(
            selected=np.array([True, True]), cost=30.0, boundary=0.0,
            objective=30.0, targets_met={1: True},
        )
        path = tmp_path / "summary.csv"
        export_summary_csv(problem, sol, path)
        df = pd.read_csv(path)
        # achieved=7.0, target=10.0, MISSLEVEL=0.5 => effective_target=5.0 => met=True
        assert df.iloc[0]["met"] is True or df.iloc[0]["met"] == True

    def test_met_false_without_misslevel(self, tmp_path):
        """Without MISSLEVEL (default 1.0), 7.0 < 10.0 => met=False."""
        pu = pd.DataFrame({"id": [1, 2], "cost": [10.0, 20.0], "status": [0, 0]})
        features = pd.DataFrame({"id": [1], "name": ["f1"], "target": [10.0], "spf": [1.0]})
        puvspr = pd.DataFrame({"species": [1, 1], "pu": [1, 2], "amount": [4.0, 3.0]})
        problem = ConservationProblem(
            planning_units=pu, features=features, pu_vs_features=puvspr,
        )
        sol = Solution(
            selected=np.array([True, True]), cost=30.0, boundary=0.0,
            objective=30.0, targets_met={1: False},
        )
        path = tmp_path / "summary.csv"
        export_summary_csv(problem, sol, path)
        df = pd.read_csv(path)
        # achieved=7.0, target=10.0, no MISSLEVEL => met=False
        assert df.iloc[0]["met"] == False

    def test_achieved_values_match_manual_sum(self, tmp_path):
        """Verify ``achieved`` is the actual sum of amounts for selected PUs.

        Existing tests only checked that the column exists and that ``met``
        is correct. A regression in the achievement-summing logic could
        keep ``met`` correct while emitting wrong ``achieved`` numbers.
        """
        pu = pd.DataFrame({
            "id": [1, 2, 3], "cost": [1.0, 1.0, 1.0], "status": [0, 0, 0],
        })
        features = pd.DataFrame({
            "id": [10, 20],
            "name": ["a", "b"],
            "target": [100.0, 50.0],
            "spf": [1.0, 1.0],
        })
        # Carefully chosen amounts so each PU contributes unambiguously
        puvspr = pd.DataFrame({
            "species": [10, 10, 10, 20, 20, 20],
            "pu":      [ 1,  2,  3,  1,  2,  3],
            "amount":  [3.0, 5.0, 7.0, 2.0, 4.0, 6.0],
        })
        problem = ConservationProblem(
            planning_units=pu, features=features, pu_vs_features=puvspr,
        )
        # Select PUs 1 and 3 (skip PU 2): feature 10 -> 3+7=10, feature 20 -> 2+6=8
        sol = Solution(
            selected=np.array([True, False, True]),
            cost=2.0, boundary=0.0, objective=2.0,
            targets_met={10: False, 20: False},
        )
        out = tmp_path / "summary.csv"
        export_summary_csv(problem, sol, out)
        df = pd.read_csv(out)
        df = df.set_index("feature_id")
        assert df.loc[10, "achieved"] == 10.0
        assert df.loc[20, "achieved"] == 8.0
        # Sanity-check met flags vs achieved (raw targets, MISSLEVEL=1.0)
        assert not df.loc[10, "met"]  # 10 < 100
        assert not df.loc[20, "met"]  # 8 < 50


class TestExportSelectionFrequencyCSV:
    def test_creates_csv(self, tmp_path):
        problem = load_project(DATA_DIR)
        solutions = [
            Solution(
                selected=np.array([True, True, False, False, False, False]),
                cost=25.0, boundary=1.0, objective=26.0,
                targets_met={1: True, 2: True, 3: True}, metadata={},
            ),
            Solution(
                selected=np.array([True, False, True, False, False, True]),
                cost=38.0, boundary=2.0, objective=40.0,
                targets_met={1: True, 2: True, 3: True}, metadata={},
            ),
        ]
        freq = compute_selection_frequency(solutions)
        out = tmp_path / "freq.csv"
        export_selection_frequency_csv(problem, freq, out)
        assert out.exists()
        df = pd.read_csv(out)
        assert "planning_unit" in df.columns
        assert "frequency" in df.columns
        assert "count" in df.columns
        assert len(df) == 6
