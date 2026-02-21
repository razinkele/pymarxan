from pathlib import Path

import numpy as np
import pandas as pd

from pymarxan.io.readers import load_project
from pymarxan.io.exporters import (
    export_solution_csv,
    export_summary_csv,
    export_selection_frequency_csv,
)
from pymarxan.solvers.base import Solution
from pymarxan.analysis.selection_freq import compute_selection_frequency

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
