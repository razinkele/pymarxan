"""Tests for Marxan-compatible probability extensions to puvspr.dat and spec.dat.

Phase 18 Batch 1: optional `prob` column on puvspr.dat (Marxan PROB2D wire
format — per-cell Bernoulli probability) and optional `ptarget` column on
spec.dat (Marxan PTARGET2D, default -1 = disabled).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from pymarxan.io.readers import load_project, read_puvspr, read_spec
from pymarxan.io.writers import save_project, write_puvspr, write_spec
from pymarxan.models.problem import ConservationProblem
from pymarxan.solvers.utils import _active_chance_terms

# --- Task 1: puvspr `prob` column round-trip -----------------------------


def test_puvspr_round_trip_with_prob(tmp_path):
    """An explicit `prob` column round-trips through write → read."""
    df = pd.DataFrame({
        "species": [1, 1, 2],
        "pu": [1, 2, 3],
        "amount": [10.0, 8.0, 5.0],
        "prob": [0.10, 0.05, 0.0],
    })
    out = tmp_path / "puvspr.dat"
    write_puvspr(df, out)

    read = read_puvspr(out)
    assert "prob" in read.columns
    np.testing.assert_array_almost_equal(read["prob"].values, [0.10, 0.05, 0.0])


def test_puvspr_all_zero_prob_is_written(tmp_path):
    """An all-zero `prob` column is written, not dropped.

    Rewritten (v0.47.0): the old test pinned a bug. The writer dropped an
    all-zero column while the readers treat a missing `prob` as presence 1.0,
    so presence 0 round-tripped as presence 1 (spec §3.2, round 1 M3).
    """
    df = pd.DataFrame({
        "species": [1], "pu": [1], "amount": [10.0], "prob": [0.0],
    })
    out = tmp_path / "puvspr.dat"
    write_puvspr(df, out)

    assert "prob" in out.read_text().splitlines()[0]


def test_puvspr_all_zero_prob_round_trips(tmp_path):
    """Presence 0 stays presence 0 through write → read."""
    df = pd.DataFrame({
        "species": [1, 1], "pu": [1, 2], "amount": [10.0, 8.0], "prob": [0.0, 0.0],
    })
    out = tmp_path / "puvspr.dat"
    write_puvspr(df, out)

    read = read_puvspr(out)
    assert "prob" in read.columns
    np.testing.assert_array_equal(read["prob"].values, [0.0, 0.0])


def test_puvspr_all_one_prob_round_trips(tmp_path):
    """An all-1 `prob` column is kept: its header switches Marxan's 2D term on.

    Under the Z = 4 zero-variance rule an all-1 column is not a no-op
    (spec §3.2, final review I-7), so dropping it would change the problem.
    """
    df = pd.DataFrame({
        "species": [1, 2], "pu": [1, 1], "amount": [10.0, 8.0], "prob": [1.0, 1.0],
    })
    out = tmp_path / "puvspr.dat"
    write_puvspr(df, out)

    read = read_puvspr(out)
    assert "prob" in read.columns
    np.testing.assert_array_equal(read["prob"].values, [1.0, 1.0])


def test_puvspr_legacy_file_loads_without_prob(tmp_path):
    """Reading a 3-column puvspr.dat doesn't add a `prob` column."""
    legacy = "species,pu,amount\n1,1,10.0\n1,2,8.0\n"
    out = tmp_path / "puvspr.dat"
    out.write_text(legacy)

    read = read_puvspr(out)
    assert "prob" not in read.columns
    assert list(read["amount"].values) == [10.0, 8.0]


def test_puvspr_frame_without_prob_is_written_without_prob(tmp_path):
    """The writer never adds a ``prob`` column: its presence alone switches the 2D chance
    term on, so a legacy PROBMODE 3 project must stay legacy across save → load.

    v0.47.0 plan review: every writer test wrote a frame that already had a ``prob``
    column, so a writer that added one (``prob = 1.0``, or a reindex to Marxan's four
    columns leaving it empty) passed them all, and after save_project / load_project a
    3-column PROBMODE 3 project scored a chance penalty it never had.
    """
    df = pd.DataFrame({"species": [1, 1, 2], "pu": [1, 2, 2], "amount": [4.0, 3.0, 2.0]})
    out = tmp_path / "puvspr.dat"
    write_puvspr(df, out)
    assert out.read_text().splitlines()[0] == "species,pu,amount"

    pu = pd.DataFrame({"id": [1, 2], "cost": [1.0, 2.0], "status": [0, 0]})
    feat = pd.DataFrame({"id": [1, 2], "name": ["a", "b"], "target": [5.0, 1.0],
                         "spf": [1.0, 1.0], "ptarget": [0.9, 0.9]})
    problem = ConservationProblem(
        planning_units=pu, features=feat, pu_vs_features=df,
        parameters={"PROBMODE": 3, "PROBABILITYWEIGHTING": 10.0},
    )
    save_project(problem, tmp_path / "project")
    puvspr = tmp_path / "project" / "input" / "puvspr.dat"
    assert puvspr.read_text().splitlines()[0] == "species,pu,amount"
    loaded = load_project(tmp_path / "project")
    assert "prob" not in loaded.pu_vs_features.columns
    assert _active_chance_terms(loaded) == frozenset()


# --- Task 2: spec `ptarget` column round-trip ----------------------------


def test_spec_round_trip_with_ptarget(tmp_path):
    """An explicit `ptarget` column round-trips. -1 marks 'disabled'."""
    df = pd.DataFrame({
        "id": [1, 2],
        "name": ["a", "b"],
        "target": [10.0, 20.0],
        "spf": [1.0, 2.0],
        "ptarget": [0.95, -1.0],
    })
    out = tmp_path / "spec.dat"
    write_spec(df, out)

    read = read_spec(out)
    assert "ptarget" in read.columns
    np.testing.assert_array_almost_equal(read["ptarget"].values, [0.95, -1.0])


def test_spec_omits_ptarget_when_all_disabled(tmp_path):
    """Writer omits ptarget when every value is -1 (the disabled sentinel)."""
    df = pd.DataFrame({
        "id": [1], "name": ["a"], "target": [10.0], "spf": [1.0],
        "ptarget": [-1.0],
    })
    out = tmp_path / "spec.dat"
    write_spec(df, out)

    content = out.read_text()
    assert "ptarget" not in content


def test_spec_default_ptarget_when_column_absent(tmp_path):
    """A legacy spec.dat without ptarget gets ptarget=-1 filled in by the reader.

    Lets downstream solver code unconditionally read `features["ptarget"]`
    without checking for column presence.
    """
    legacy = "id,name,target,spf\n1,a,10.0,1.0\n"
    out = tmp_path / "spec.dat"
    out.write_text(legacy)

    df = read_spec(out)
    assert "ptarget" in df.columns
    assert df["ptarget"].iloc[0] == -1.0
