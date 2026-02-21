import pandas as pd

from pymarxan.io.readers import read_mvbest, read_ssoln, read_sum


class TestReadMvbest:
    def test_reads_two_rows(self, tmp_path):
        csv = tmp_path / "out_mvbest.csv"
        csv.write_text(
            "Feature_ID,Target,Amount_Held,Shortfall,Target_Met\n"
            "1,100.0,120.5,0.0,True\n"
            "2,50.0,30.0,20.0,False\n"
        )
        df = read_mvbest(csv)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2

    def test_feature_id_is_int(self, tmp_path):
        csv = tmp_path / "out_mvbest.csv"
        csv.write_text(
            "Feature_ID,Target,Amount_Held,Shortfall,Target_Met\n"
            "1,100.0,120.5,0.0,True\n"
            "2,50.0,30.0,20.0,False\n"
        )
        df = read_mvbest(csv)
        assert df["Feature_ID"].dtype in ("int64", "int32")

    def test_target_met_is_bool(self, tmp_path):
        csv = tmp_path / "out_mvbest.csv"
        csv.write_text(
            "Feature_ID,Target,Amount_Held,Shortfall,Target_Met\n"
            "1,100.0,120.5,0.0,True\n"
            "2,50.0,30.0,20.0,False\n"
        )
        df = read_mvbest(csv)
        assert df["Target_Met"].dtype == "bool"

    def test_float_columns(self, tmp_path):
        csv = tmp_path / "out_mvbest.csv"
        csv.write_text(
            "Feature_ID,Target,Amount_Held,Shortfall,Target_Met\n"
            "1,100.0,120.5,0.0,True\n"
            "2,50.0,30.0,20.0,False\n"
        )
        df = read_mvbest(csv)
        assert df["Target"].dtype == "float64"
        assert df["Amount_Held"].dtype == "float64"
        assert df["Shortfall"].dtype == "float64"


class TestReadSsoln:
    def test_reads_three_rows(self, tmp_path):
        csv = tmp_path / "out_ssoln.csv"
        csv.write_text(
            "Planning_Unit,Number\n"
            "1,5\n"
            "2,3\n"
            "3,10\n"
        )
        df = read_ssoln(csv)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3

    def test_planning_unit_is_int(self, tmp_path):
        csv = tmp_path / "out_ssoln.csv"
        csv.write_text(
            "Planning_Unit,Number\n"
            "1,5\n"
            "2,3\n"
            "3,10\n"
        )
        df = read_ssoln(csv)
        assert df["Planning_Unit"].dtype in ("int64", "int32")

    def test_number_is_int(self, tmp_path):
        csv = tmp_path / "out_ssoln.csv"
        csv.write_text(
            "Planning_Unit,Number\n"
            "1,5\n"
            "2,3\n"
            "3,10\n"
        )
        df = read_ssoln(csv)
        assert df["Number"].dtype in ("int64", "int32")


class TestReadSum:
    def test_reads_two_rows(self, tmp_path):
        csv = tmp_path / "out_sum.csv"
        csv.write_text(
            "Run,Score,Cost,Planning_Units,Boundary,Penalty,Shortfall\n"
            "1,250.5,200.0,10,30.5,15.0,5.0\n"
            "2,300.0,280.0,12,40.0,10.0,0.0\n"
        )
        df = read_sum(csv)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2

    def test_run_is_int(self, tmp_path):
        csv = tmp_path / "out_sum.csv"
        csv.write_text(
            "Run,Score,Cost,Planning_Units,Boundary,Penalty,Shortfall\n"
            "1,250.5,200.0,10,30.5,15.0,5.0\n"
            "2,300.0,280.0,12,40.0,10.0,0.0\n"
        )
        df = read_sum(csv)
        assert df["Run"].dtype in ("int64", "int32")
        assert df["Planning_Units"].dtype in ("int64", "int32")

    def test_score_column_exists(self, tmp_path):
        csv = tmp_path / "out_sum.csv"
        csv.write_text(
            "Run,Score,Cost,Planning_Units,Boundary,Penalty,Shortfall\n"
            "1,250.5,200.0,10,30.5,15.0,5.0\n"
            "2,300.0,280.0,12,40.0,10.0,0.0\n"
        )
        df = read_sum(csv)
        assert "Score" in df.columns

    def test_float_columns(self, tmp_path):
        csv = tmp_path / "out_sum.csv"
        csv.write_text(
            "Run,Score,Cost,Planning_Units,Boundary,Penalty,Shortfall\n"
            "1,250.5,200.0,10,30.5,15.0,5.0\n"
            "2,300.0,280.0,12,40.0,10.0,0.0\n"
        )
        df = read_sum(csv)
        for col in ("Score", "Cost", "Boundary", "Penalty", "Shortfall"):
            assert df[col].dtype == "float64"
