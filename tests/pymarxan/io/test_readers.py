from pathlib import Path
import pandas as pd
from pymarxan.io.readers import read_pu, read_spec, read_puvspr, read_bound, read_input_dat, load_project
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
