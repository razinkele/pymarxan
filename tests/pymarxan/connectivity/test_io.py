import pandas as pd

from pymarxan.connectivity.io import (
    connectivity_to_matrix,
    read_connectivity_edgelist,
    read_connectivity_matrix,
)


class TestReadEdgelist:
    def test_reads_edgelist(self, tmp_path):
        p = tmp_path / "edges.csv"
        p.write_text("id1,id2,value\n1,2,0.5\n2,3,0.8\n3,1,0.3\n")
        pu_ids = [1, 2, 3]
        m = read_connectivity_edgelist(p, pu_ids)
        assert m.shape == (3, 3)
        assert m[0, 1] == 0.5
        assert m[1, 2] == 0.8
        assert m[2, 0] == 0.3

    def test_missing_edges_are_zero(self, tmp_path):
        p = tmp_path / "edges.csv"
        p.write_text("id1,id2,value\n1,2,0.5\n")
        pu_ids = [1, 2, 3]
        m = read_connectivity_edgelist(p, pu_ids)
        assert m[0, 2] == 0.0
        assert m[2, 1] == 0.0


class TestReadMatrix:
    def test_reads_csv_matrix(self, tmp_path):
        p = tmp_path / "matrix.csv"
        p.write_text(",1,2,3\n1,0,0.5,0\n2,0.1,0,0.8\n3,0,0.3,0\n")
        m = read_connectivity_matrix(p)
        assert m.shape == (3, 3)
        assert m[0, 1] == 0.5
        assert m[1, 2] == 0.8


class TestConnectivityToMatrix:
    def test_edgelist_df(self):
        df = pd.DataFrame({
            "id1": [1, 2], "id2": [2, 3], "value": [0.5, 0.8],
        })
        m = connectivity_to_matrix(df, pu_ids=[1, 2, 3])
        assert m.shape == (3, 3)
        assert m[0, 1] == 0.5
