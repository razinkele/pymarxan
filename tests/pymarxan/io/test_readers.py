from pathlib import Path

import pandas as pd
import pytest

from pymarxan.io.readers import (
    load_project,
    read_bound,
    read_input_dat,
    read_pu,
    read_puvspr,
    read_spec,
)
from pymarxan.models.problem import ConservationProblem

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "simple"

class TestReadPu:
    def test_reads_csv(self):
        df = read_pu(DATA_DIR / "input" / "pu.dat")
        assert isinstance(df, pd.DataFrame)
        assert "id" in df.columns
        assert "cost" in df.columns
        assert len(df) == 6

    def test_id_column_is_int(self):
        df = read_pu(DATA_DIR / "input" / "pu.dat")
        assert df["id"].dtype in ("int64", "int32")

    def test_cost_column_is_float(self):
        df = read_pu(DATA_DIR / "input" / "pu.dat")
        assert df["cost"].dtype == "float64"

class TestReadPuDefaults:
    def test_read_pu_defaults_missing_status(self, tmp_path):
        """pu.dat without status column should default to 0."""
        (tmp_path / "pu.dat").write_text("id,cost\n1,10\n2,20\n")
        df = read_pu(tmp_path / "pu.dat")
        assert "status" in df.columns
        assert list(df["status"]) == [0, 0]

    def test_read_pu_double_tab_separated(self, tmp_path):
        """Some real Marxan exports pad columns with repeated tabs/whitespace
        (e.g. ``id\\t\\tcost\\t\\tstatus``). The reader must collapse the runs
        rather than create empty ``Unnamed`` columns."""
        (tmp_path / "pu.dat").write_text(
            "id\t\tcost\t\tstatus\n1\t\t5\t\t0\n2\t\t3\t\t2\n"
        )
        df = read_pu(tmp_path / "pu.dat")
        assert not any(str(c).startswith("Unnamed") for c in df.columns)
        assert list(df["id"]) == [1, 2]
        assert list(df["cost"]) == [5.0, 3.0]
        assert list(df["status"]) == [0, 2]

    def test_read_pu_space_separated(self, tmp_path):
        """Whitespace-padded columns are also collapsed."""
        (tmp_path / "pu.dat").write_text("id  cost  status\n1  5  0\n2  3  2\n")
        df = read_pu(tmp_path / "pu.dat")
        assert not any(str(c).startswith("Unnamed") for c in df.columns)
        assert list(df["cost"]) == [5.0, 3.0]


class TestReadSpec:
    def test_reads_csv(self):
        df = read_spec(DATA_DIR / "input" / "spec.dat")
        assert len(df) == 3
        assert "id" in df.columns
        assert "name" in df.columns

    def test_has_target(self):
        df = read_spec(DATA_DIR / "input" / "spec.dat")
        assert "target" in df.columns
        assert df["target"].iloc[0] == 30.0

class TestReadSpecDefaults:
    def test_read_spec_defaults_missing_spf_and_name(self, tmp_path):
        """spec.dat without spf/name columns should default to 1.0 and auto-name."""
        (tmp_path / "spec.dat").write_text("id,target\n1,5\n2,10\n")
        df = read_spec(tmp_path / "spec.dat")
        assert "spf" in df.columns
        assert list(df["spf"]) == [1.0, 1.0]
        assert "name" in df.columns
        assert list(df["name"]) == ["Feature_1", "Feature_2"]


class TestReadPuvspr:
    def test_reads_csv(self):
        df = read_puvspr(DATA_DIR / "input" / "puvspr.dat")
        assert len(df) == 17
        assert set(df.columns) >= {"species", "pu", "amount"}

    def test_species_are_ints(self):
        df = read_puvspr(DATA_DIR / "input" / "puvspr.dat")
        assert df["species"].dtype in ("int64", "int32")

class TestReadBound:
    def test_reads_csv(self):
        df = read_bound(DATA_DIR / "input" / "bound.dat")
        assert set(df.columns) >= {"id1", "id2", "boundary"}
        assert len(df) == 11

    def test_accepts_bound_column_alias(self, tmp_path):
        """Some Marxan exports name the boundary column ``bound`` rather than
        ``boundary``; both must work."""
        (tmp_path / "bound.dat").write_text("id1,id2,bound\n1,2,1.0\n2,3,0.5\n")
        df = read_bound(tmp_path / "bound.dat")
        assert "boundary" in df.columns
        assert df["boundary"].tolist() == [1.0, 0.5]

class TestReadInputDat:
    def test_reads_parameters(self):
        params = read_input_dat(DATA_DIR / "input.dat")
        assert isinstance(params, dict)
        assert params["BLM"] == 1.0
        assert params["NUMREPS"] == 10
        assert params["INPUTDIR"] == "input"
        assert params["PUNAME"] == "pu.dat"

    def test_numeric_conversion(self):
        params = read_input_dat(DATA_DIR / "input.dat")
        assert isinstance(params["BLM"], float)
        assert isinstance(params["NUMREPS"], int)
        assert isinstance(params["RANDSEED"], int)

class TestLoadProject:
    def test_loads_full_project(self):
        problem = load_project(DATA_DIR)
        assert isinstance(problem, ConservationProblem)
        assert problem.n_planning_units == 6
        assert problem.n_features == 3
        assert len(problem.pu_vs_features) == 17
        assert problem.boundary is not None
        assert len(problem.boundary) == 11
        assert problem.parameters["BLM"] == 1.0

    def test_loaded_problem_validates(self):
        problem = load_project(DATA_DIR)
        errors = problem.validate()
        assert errors == [], f"Validation errors: {errors}"


def test_load_project_resolves_prop_to_target(tmp_path):
    """When prop > 0 and target == 0, effective target = prop * total_amount."""
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (tmp_path / "input.dat").write_text(
        "INPUTDIR input\nPUNAME pu.dat\nSPECNAME spec.dat\nPUVSPRNAME puvspr.dat\n"
    )
    (input_dir / "pu.dat").write_text("id,cost,status\n1,10,0\n2,20,0\n3,30,0\n")
    (input_dir / "spec.dat").write_text(
        "id,target,prop,spf,name\n1,0,0.3,1.0,feat_a\n2,5,0,1.0,feat_b\n"
    )
    (input_dir / "puvspr.dat").write_text(
        "species,pu,amount\n1,1,10\n1,2,8\n2,2,12\n2,3,20\n"
    )
    problem = load_project(tmp_path)
    f1 = problem.features[problem.features["id"] == 1].iloc[0]
    assert f1["target"] == pytest.approx(5.4)  # 0.3 * (10 + 8) = 5.4
    f2 = problem.features[problem.features["id"] == 2].iloc[0]
    assert f2["target"] == pytest.approx(5.0)  # prop=0 so target stays 5.0
