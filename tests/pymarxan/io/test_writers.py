from pathlib import Path

import pandas as pd

from pymarxan.io.readers import load_project, read_input_dat, read_pu, read_spec
from pymarxan.io.writers import (
    save_project,
    write_input_dat,
    write_pu,
    write_spec,
)

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "simple"

class TestWritePu:
    def test_roundtrip(self, tmp_path):
        original = pd.DataFrame({"id": [1, 2, 3], "cost": [10.0, 15.0, 20.0], "status": [0, 0, 2]})
        out_path = tmp_path / "pu.dat"
        write_pu(original, out_path)
        loaded = read_pu(out_path)
        pd.testing.assert_frame_equal(original, loaded)

class TestWriteSpec:
    def test_roundtrip(self, tmp_path):
        original = pd.DataFrame({
            "id": [1, 2], "name": ["sp_a", "sp_b"],
            "target": [20.0, 10.0], "spf": [1.0, 1.0],
        })
        out_path = tmp_path / "spec.dat"
        write_spec(original, out_path)
        loaded = read_spec(out_path)
        pd.testing.assert_frame_equal(original, loaded)

class TestWriteInputDat:
    def test_roundtrip(self, tmp_path):
        params = {"BLM": 1.5, "NUMREPS": 10, "INPUTDIR": "input", "PUNAME": "pu.dat"}
        out_path = tmp_path / "input.dat"
        write_input_dat(params, out_path)
        loaded = read_input_dat(out_path)
        assert loaded["BLM"] == 1.5
        assert loaded["NUMREPS"] == 10
        assert loaded["INPUTDIR"] == "input"

class TestSaveProject:
    def test_roundtrip(self, tmp_path):
        original = load_project(DATA_DIR)
        save_project(original, tmp_path)
        reloaded = load_project(tmp_path)
        assert reloaded.n_planning_units == original.n_planning_units
        assert reloaded.n_features == original.n_features
        assert len(reloaded.pu_vs_features) == len(original.pu_vs_features)
        assert reloaded.parameters["BLM"] == original.parameters["BLM"]


def test_write_mvbest_applies_misslevel(tmp_path):
    """write_mvbest should use MISSLEVEL for Target_Met column."""
    import numpy as np

    from pymarxan.io.writers import write_mvbest
    from pymarxan.solvers.utils import build_solution

    problem = load_project(DATA_DIR)
    # MISSLEVEL=0.0 means all targets are effectively 0 -> all met
    problem.parameters["MISSLEVEL"] = 0.0
    selected = np.zeros(problem.n_planning_units, dtype=bool)
    sol = build_solution(problem, selected, blm=0.0)

    path = tmp_path / "mvbest.csv"
    write_mvbest(problem, sol, path)
    df = pd.read_csv(path)
    assert all(df["Target_Met"]), "With MISSLEVEL=0.0, all targets should be met"


def test_write_sum_uses_penalty_field(tmp_path):
    """write_sum should use sol.penalty directly."""
    import numpy as np

    from pymarxan.io.writers import write_sum
    from pymarxan.solvers.base import Solution

    sol = Solution(
        selected=np.array([True, False]),
        cost=10.0,
        boundary=5.0,
        objective=25.0,  # 10 + 2*5 + 5 (penalty=5)
        targets_met={1: False},
        penalty=5.0,
    )
    path = tmp_path / "sum.csv"
    write_sum([sol], path)
    df = pd.read_csv(path)
    assert df["Penalty"].iloc[0] == 5.0


def test_io_exports_writers():
    """io module should export write_mvbest, write_ssoln, write_sum."""
    from pymarxan import io

    assert hasattr(io, "write_mvbest")
    assert hasattr(io, "write_ssoln")
    assert hasattr(io, "write_sum")
