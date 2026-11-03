"""Probability 1D I/O (v0.48.0, spec §3.2): spec ``ptarget1d`` / ``ptarget2d``.

Marxan 3 (``input.c``) reads the spec columns ``ptarget1d`` and ``ptarget2d`` (default
−1, disabled). In memory the 2D target stays ``features["ptarget"]`` (pymarxan's Phase 18
name) and the 1D target is ``features["ptarget1d"]``. ``read_spec`` folds ``ptarget2d``
into ``ptarget`` and drops it; ``write_spec`` writes the Marxan-native names only.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from pymarxan.io.readers import load_project, read_input_dat, read_spec
from pymarxan.io.writers import save_project, write_input_dat, write_spec
from pymarxan.models.problem import (
    _OVERRIDABLE_FIELDS,
    ConservationProblem,
    apply_feature_overrides,
)


def _write(tmp_path, text: str):
    path = tmp_path / "spec.dat"
    path.write_text(text)
    return path


def _header(path) -> list[str]:
    return path.read_text().splitlines()[0].split(",")


class TestReadSpec:
    def test_ptarget2d_is_read_into_ptarget_and_dropped(self, tmp_path):
        path = _write(tmp_path, "id,name,target,spf,ptarget2d\n1,a,5,1,0.9\n2,b,3,1,-1\n")
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # a whitelisted column: no "unrecognised" warning
            df = read_spec(path)
        assert "ptarget2d" not in df.columns
        np.testing.assert_array_equal(df["ptarget"].to_numpy(), [0.9, -1.0])

    def test_ptarget1d_is_read_as_float(self, tmp_path):
        path = _write(tmp_path, "id,name,target,spf,ptarget1d\n1,a,5,1,1\n2,b,3,1,-1\n")
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            df = read_spec(path)
        assert df["ptarget1d"].dtype == np.float64
        np.testing.assert_array_equal(df["ptarget1d"].to_numpy(), [1.0, -1.0])

    def test_missing_ptarget1d_defaults_to_disabled(self, tmp_path):
        df = read_spec(_write(tmp_path, "id,name,target,spf\n1,a,5,1\n"))
        assert df["ptarget1d"].tolist() == [-1.0]
        assert df["ptarget"].tolist() == [-1.0]

    def test_ptarget_and_ptarget2d_agreeing_are_merged(self, tmp_path):
        path = _write(tmp_path, "id,name,target,spf,ptarget,ptarget2d\n1,a,5,1,0.8,0.8\n")
        df = read_spec(path)
        assert "ptarget2d" not in df.columns
        assert df["ptarget"].tolist() == [0.8]

    def test_ptarget_and_ptarget2d_disagreeing_raise(self, tmp_path):
        path = _write(
            tmp_path, "id,name,target,spf,ptarget,ptarget2d\n1,a,5,1,0.8,0.8\n2,b,5,1,0.5,0.6\n",
        )
        with pytest.raises(ValueError, match="ptarget2d"):
            read_spec(path)

    def test_blank_ptarget1d_counts_as_disabled(self, tmp_path):
        path = _write(tmp_path, "id,name,target,spf,ptarget1d\n1,a,5,1,0.7\n2,b,3,1,\n")
        df = read_spec(path)
        assert df["ptarget1d"].tolist() == [0.7, -1.0]


class TestWriteSpec:
    def _features(self, **cols) -> pd.DataFrame:
        base = {"id": [1, 2], "name": ["a", "b"], "target": [5.0, 3.0], "spf": [1.0, 1.0]}
        base.update(cols)
        return pd.DataFrame(base)

    def test_writes_ptarget2d_never_ptarget(self, tmp_path):
        path = tmp_path / "spec.dat"
        write_spec(self._features(ptarget=[0.9, -1.0]), path)
        header = _header(path)
        assert "ptarget2d" in header
        assert "ptarget" not in header

    def test_ptarget2d_takes_ptargets_position(self, tmp_path):
        path = tmp_path / "spec.dat"
        df = self._features(ptarget=[0.9, -1.0])
        df["prop"] = [0.1, 0.2]  # a column after ptarget
        write_spec(df, path)
        assert _header(path) == ["id", "name", "target", "spf", "ptarget2d", "prop"]

    def test_writes_ptarget1d(self, tmp_path):
        path = tmp_path / "spec.dat"
        write_spec(self._features(ptarget1d=[0.95, -1.0]), path)
        assert "ptarget1d" in _header(path)

    def test_all_disabled_columns_are_dropped(self, tmp_path):
        path = tmp_path / "spec.dat"
        write_spec(self._features(ptarget=[-1.0, -1.0], ptarget1d=[-1.0, -1.0]), path)
        header = _header(path)
        assert not any(h.startswith("ptarget") for h in header)

    def test_nan_ptarget1d_counts_as_disabled(self, tmp_path):
        path = tmp_path / "spec.dat"
        write_spec(self._features(ptarget1d=[0.8, np.nan]), path)
        assert read_spec(path)["ptarget1d"].tolist() == [0.8, -1.0]
        write_spec(self._features(ptarget1d=[np.nan, np.nan]), path)
        assert "ptarget1d" not in _header(path)

    def test_does_not_mutate_the_frame(self, tmp_path):
        df = self._features(ptarget=[0.9, -1.0], ptarget1d=[0.5, np.nan])
        before = df.copy()
        write_spec(df, tmp_path / "spec.dat")
        pd.testing.assert_frame_equal(df, before)


def _problem(**feature_cols) -> ConservationProblem:
    features = {"id": [1, 2], "name": ["a", "b"], "target": [5.0, 3.0], "spf": [1.0, 1.0]}
    features.update(feature_cols)
    return ConservationProblem(
        planning_units=pd.DataFrame({
            "id": [1, 2, 3], "cost": [1.0, 2.0, 3.0], "status": [0, 0, 0],
            "prob": [0.1, 0.2, 0.0],
        }),
        features=pd.DataFrame(features),
        pu_vs_features=pd.DataFrame({
            "species": [1, 1, 2], "pu": [1, 2, 3], "amount": [4.0, 3.0, 5.0],
        }),
        parameters={"PROBMODE": 4},
    )


class TestRoundTrip:
    def test_legacy_ptarget_round_trips_as_ptarget2d(self, tmp_path):
        save_project(_problem(ptarget=[0.9, -1.0]), tmp_path)
        assert "ptarget2d" in _header(tmp_path / "input" / "spec.dat")
        loaded = load_project(tmp_path)
        assert loaded.features["ptarget"].tolist() == [0.9, -1.0]
        assert "ptarget2d" not in loaded.features.columns

    def test_ptarget1d_and_pu_prob_round_trip(self, tmp_path):
        save_project(_problem(ptarget1d=[0.95, -1.0]), tmp_path)
        loaded = load_project(tmp_path)
        assert loaded.features["ptarget1d"].tolist() == [0.95, -1.0]
        np.testing.assert_array_equal(loaded.planning_units["prob"], [0.1, 0.2, 0.0])

    def test_in_memory_ptarget_set_after_loading_is_honoured(self, tmp_path):
        save_project(_problem(ptarget=[0.9, -1.0]), tmp_path)
        loaded = load_project(tmp_path)
        loaded.features["ptarget"] = [0.5, 0.6]
        out = tmp_path / "again"
        save_project(loaded, out)
        assert load_project(out).features["ptarget"].tolist() == [0.5, 0.6]

    def test_probzerovar_round_trips(self, tmp_path):
        write_input_dat({"PROBZEROVAR": 2, "PROBMODE": 4}, tmp_path / "input.dat")
        params = read_input_dat(tmp_path / "input.dat")
        assert params["PROBZEROVAR"] == 2
        assert params["PROBMODE"] == 4

    def test_puvspr_all_one_prob_round_trips(self, tmp_path):
        problem = _problem()
        problem.pu_vs_features["prob"] = 1.0
        save_project(problem, tmp_path)
        loaded = load_project(tmp_path)
        assert loaded.pu_vs_features["prob"].tolist() == [1.0, 1.0, 1.0]


def test_ptarget1d_is_overridable():
    assert "ptarget1d" in _OVERRIDABLE_FIELDS
    out = apply_feature_overrides(_problem(ptarget1d=[0.9, -1.0]), {2: {"ptarget1d": 0.5}})
    assert out.features["ptarget1d"].tolist() == [0.9, 0.5]
