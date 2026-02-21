from pathlib import Path
import pandas as pd
from pymarxan.io.readers import load_project, read_input_dat, read_pu, read_spec
from pymarxan.io.writers import write_pu, write_spec, write_puvspr, write_bound, write_input_dat, save_project

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
        original = pd.DataFrame({"id": [1, 2], "name": ["sp_a", "sp_b"], "target": [20.0, 10.0], "spf": [1.0, 1.0]})
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
