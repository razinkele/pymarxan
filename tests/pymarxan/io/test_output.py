"""Tests for classic Marxan output file controller."""

from pathlib import Path

import numpy as np
import pandas as pd

from pymarxan.io.output import OutputController, _should_save
from pymarxan.io.readers import load_project
from pymarxan.solvers.base import Solution
from pymarxan.solvers.utils import build_solution

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "simple"


def _make_solution(n_pu: int, all_selected: bool = True) -> Solution:
    selected = (
        np.ones(n_pu, dtype=bool)
        if all_selected
        else np.zeros(n_pu, dtype=bool)
    )
    return Solution(
        selected=selected,
        cost=100.0 if all_selected else 0.0,
        boundary=10.0,
        objective=110.0 if all_selected else 50.0,
        targets_met={1: all_selected, 2: all_selected},
        penalty=0.0 if all_selected else 50.0,
        shortfall=0.0 if all_selected else 25.0,
    )


class TestShouldSave:
    def test_save_always(self):
        assert _should_save(3, True)
        assert _should_save(3, False)

    def test_save_never(self):
        assert not _should_save(0, True)
        assert not _should_save(0, False)

    def test_save_if_met(self):
        assert _should_save(1, True)
        assert not _should_save(1, False)

    def test_save_if_unmet(self):
        assert not _should_save(2, True)
        assert _should_save(2, False)


class TestOutputControllerParams:
    def test_defaults_save_everything(self):
        ctrl = OutputController()
        assert ctrl.should_save_run()
        assert ctrl.should_save_best()
        assert ctrl.should_save_summary()
        assert ctrl.should_save_scenario()
        assert ctrl.should_save_targmet()
        assert ctrl.should_save_solution_matrix()

    def test_save_never_params(self):
        params = {
            "SAVERUN": 0, "SAVEBEST": 0, "SAVESUMM": 0,
            "SAVESCEN": 0, "SAVETARGMET": 0,
            "SAVESOLUTIONSMATRIX": 0,
        }
        ctrl = OutputController(params=params)
        assert not ctrl.should_save_run()
        assert not ctrl.should_save_best()
        assert not ctrl.should_save_summary()
        assert not ctrl.should_save_scenario()
        assert not ctrl.should_save_targmet()
        assert not ctrl.should_save_solution_matrix()

    def test_scenname_from_params(self):
        ctrl = OutputController(params={"SCENNAME": "test"})
        assert ctrl.scenname == "test"

    def test_scenname_default(self):
        ctrl = OutputController()
        assert ctrl.scenname == "output"


class TestWriteOutputs:
    def setup_method(self):
        self.problem = load_project(DATA_DIR)
        n = self.problem.n_planning_units
        self.solutions = [
            build_solution(self.problem, np.ones(n, dtype=bool), blm=1.0),
            build_solution(self.problem, np.ones(n, dtype=bool), blm=1.0),
        ]

    def test_write_all_outputs(self, tmp_path):
        ctrl = OutputController(params=self.problem.parameters)
        ctrl.write_outputs(self.problem, self.solutions, tmp_path)

        sn = ctrl.scenname
        assert (tmp_path / f"{sn}_sum.csv").exists()
        assert (tmp_path / f"{sn}_best.csv").exists()
        assert (tmp_path / f"{sn}_mvbest.csv").exists()
        assert (tmp_path / f"{sn}_ssoln.csv").exists()
        assert (tmp_path / f"{sn}_r001.csv").exists()
        assert (tmp_path / f"{sn}_r002.csv").exists()
        assert (tmp_path / f"{sn}_targmet.csv").exists()
        assert (tmp_path / f"{sn}_solutionsmatrix.csv").exists()
        assert (tmp_path / f"{sn}_sen.csv").exists()

    def test_write_no_outputs(self, tmp_path):
        params = {
            "SAVERUN": 0, "SAVEBEST": 0, "SAVESUMM": 0,
            "SAVESCEN": 0, "SAVETARGMET": 0,
            "SAVESOLUTIONSMATRIX": 0,
        }
        ctrl = OutputController(params=params)
        ctrl.write_outputs(self.problem, self.solutions, tmp_path)
        # Only directory created, no files
        csv_files = list(tmp_path.glob("*.csv"))
        assert len(csv_files) == 0

    def test_empty_solutions(self, tmp_path):
        ctrl = OutputController()
        ctrl.write_outputs(self.problem, [], tmp_path)
        csv_files = list(tmp_path.glob("*.csv"))
        assert len(csv_files) == 0

    def test_best_file_content(self, tmp_path):
        ctrl = OutputController()
        ctrl.write_outputs(self.problem, self.solutions, tmp_path)
        best_df = pd.read_csv(tmp_path / "output_best.csv")
        assert "Planning_Unit" in best_df.columns
        assert "Solution" in best_df.columns
        assert len(best_df) == self.problem.n_planning_units

    def test_sum_file_content(self, tmp_path):
        ctrl = OutputController()
        ctrl.write_outputs(self.problem, self.solutions, tmp_path)
        sum_df = pd.read_csv(tmp_path / "output_sum.csv")
        assert len(sum_df) == 2
        assert "Score" in sum_df.columns

    def test_solution_matrix_content(self, tmp_path):
        ctrl = OutputController()
        ctrl.write_outputs(self.problem, self.solutions, tmp_path)
        mat_df = pd.read_csv(
            tmp_path / "output_solutionsmatrix.csv",
        )
        assert len(mat_df) == self.problem.n_planning_units
        assert "S1" in mat_df.columns
        assert "S2" in mat_df.columns

    def test_scenario_file_content(self, tmp_path):
        ctrl = OutputController()
        ctrl.write_outputs(self.problem, self.solutions, tmp_path)
        sen_df = pd.read_csv(tmp_path / "output_sen.csv")
        assert "Best_Score" in sen_df.columns
        assert len(sen_df) == 1

    def test_targmet_file_content(self, tmp_path):
        ctrl = OutputController()
        ctrl.write_outputs(self.problem, self.solutions, tmp_path)
        tm_df = pd.read_csv(tmp_path / "output_targmet.csv")
        n_feat = self.problem.n_features
        assert len(tm_df) == 2 * n_feat  # 2 runs * n features

    def test_run_file_content(self, tmp_path):
        ctrl = OutputController()
        ctrl.write_outputs(self.problem, self.solutions, tmp_path)
        r1_df = pd.read_csv(tmp_path / "output_r001.csv")
        assert "Planning_Unit" in r1_df.columns
        assert "Solution" in r1_df.columns
        assert len(r1_df) == self.problem.n_planning_units
