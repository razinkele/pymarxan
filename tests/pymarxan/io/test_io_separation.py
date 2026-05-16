"""Tests for Marxan-compatible separation-distance columns on spec.dat.

Phase 20 Batch 1: optional ``sepdistance`` (float, default 0.0) and ``sepnum``
(int, default 1) columns on spec.dat. A feature is separation-active iff
``sepdistance > 0 AND sepnum > 1`` (matches Marxan's ``CountSeparation2``
short-circuit + "sepnum<=1 disabled" convention).

Also pins the round-3 H10 cross-phase unrecognised-column whitelist warning
on ``read_spec``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pymarxan.io.readers import read_spec
from pymarxan.io.writers import write_spec

# --- Task 1: sepdistance column round-trip ------------------------------


def test_spec_round_trip_with_sepdistance(tmp_path):
    """An explicit ``sepdistance`` column round-trips through write → read."""
    df = pd.DataFrame({
        "id": [1, 2, 3],
        "name": ["a", "b", "c"],
        "target": [10.0, 20.0, 30.0],
        "spf": [1.0, 2.0, 3.0],
        "sepdistance": [5000.0, 0.0, 1250.5],
    })
    out = tmp_path / "spec.dat"
    write_spec(df, out)

    read = read_spec(out)
    assert "sepdistance" in read.columns
    np.testing.assert_array_almost_equal(
        read["sepdistance"].values, [5000.0, 0.0, 1250.5],
    )


def test_spec_omits_sepdistance_when_all_zero(tmp_path):
    """Writer omits sepdistance when every value is 0 (the disabled default)."""
    df = pd.DataFrame({
        "id": [1], "name": ["a"], "target": [10.0], "spf": [1.0],
        "sepdistance": [0.0],
    })
    out = tmp_path / "spec.dat"
    write_spec(df, out)

    content = out.read_text()
    assert "sepdistance" not in content, "writer must omit all-zero sepdistance"


def test_spec_default_sepdistance_when_column_absent(tmp_path):
    """Reading a legacy spec.dat without sepdistance fills 0.0."""
    legacy = "id,name,target,spf\n1,a,10.0,1.0\n"
    out = tmp_path / "spec.dat"
    out.write_text(legacy)

    df = read_spec(out)
    assert "sepdistance" in df.columns
    assert df["sepdistance"].iloc[0] == 0.0


def test_spec_rejects_negative_sepdistance(tmp_path):
    """Round-3 H6: negative sepdistance values are silently sign-stripped by
    the squared-distance comparison. Catch at read time instead."""
    bad = "id,name,target,spf,sepdistance\n1,a,10.0,1.0,-5.0\n"
    out = tmp_path / "spec.dat"
    out.write_text(bad)

    with pytest.raises(ValueError, match="sepdistance"):
        read_spec(out)


# --- Task 2: sepnum column round-trip -----------------------------------


def test_spec_round_trip_with_sepnum(tmp_path):
    """An explicit ``sepnum`` column round-trips. Values stay int."""
    df = pd.DataFrame({
        "id": [1, 2, 3],
        "name": ["a", "b", "c"],
        "target": [10.0, 20.0, 30.0],
        "spf": [1.0, 2.0, 3.0],
        "sepdistance": [5000.0, 0.0, 1000.0],
        "sepnum": [3, 1, 5],
    })
    out = tmp_path / "spec.dat"
    write_spec(df, out)

    read = read_spec(out)
    assert "sepnum" in read.columns
    np.testing.assert_array_equal(read["sepnum"].values, [3, 1, 5])


def test_spec_omits_sepnum_when_all_disabled(tmp_path):
    """Writer omits sepnum when every value is ≤ 1 (Marxan's disabled sentinel)."""
    df = pd.DataFrame({
        "id": [1, 2], "name": ["a", "b"], "target": [10.0, 20.0], "spf": [1.0, 1.0],
        "sepnum": [1, 1],
    })
    out = tmp_path / "spec.dat"
    write_spec(df, out)

    content = out.read_text()
    assert "sepnum" not in content, "writer must omit all-disabled sepnum"


def test_spec_default_sepnum_when_column_absent(tmp_path):
    """Reading a legacy spec.dat without sepnum fills 1 (Marxan disabled sentinel)."""
    legacy = "id,name,target,spf\n1,a,10.0,1.0\n"
    out = tmp_path / "spec.dat"
    out.write_text(legacy)

    df = read_spec(out)
    assert "sepnum" in df.columns
    assert int(df["sepnum"].iloc[0]) == 1


def test_spec_rejects_negative_sepnum(tmp_path):
    """Negative sepnum is meaningless and silently truncates to wrong behaviour."""
    bad = "id,name,target,spf,sepnum\n1,a,10.0,1.0,-1\n"
    out = tmp_path / "spec.dat"
    out.write_text(bad)

    with pytest.raises(ValueError, match="sepnum"):
        read_spec(out)


def test_spec_rejects_non_integer_sepnum(tmp_path):
    """Round-2 M6: ``2.7`` silently truncating to ``2`` is a hidden user bug.
    Round-3 plan extends this to all decimal-bearing inputs."""
    bad = "id,name,target,spf,sepnum\n1,a,10.0,1.0,2.7\n"
    out = tmp_path / "spec.dat"
    out.write_text(bad)

    with pytest.raises(ValueError, match="sepnum"):
        read_spec(out)


# --- Combined column round-trip -----------------------------------------


def test_spec_round_trip_all_phase18_19_20_columns(tmp_path):
    """All three Marxan-classic constraint columns coexist on the same wire."""
    df = pd.DataFrame({
        "id": [1, 2],
        "name": ["a", "b"],
        "target": [10.0, 20.0],
        "spf": [1.0, 2.0],
        "ptarget": [0.95, -1.0],
        "target2": [5.0, 0.0],
        "clumptype": [1, 0],
        "sepdistance": [5000.0, 0.0],
        "sepnum": [3, 1],
    })
    out = tmp_path / "spec.dat"
    write_spec(df, out)

    read = read_spec(out)
    np.testing.assert_array_almost_equal(read["ptarget"].values, [0.95, -1.0])
    np.testing.assert_array_almost_equal(read["target2"].values, [5.0, 0.0])
    np.testing.assert_array_equal(read["clumptype"].values, [1, 0])
    np.testing.assert_array_almost_equal(read["sepdistance"].values, [5000.0, 0.0])
    np.testing.assert_array_equal(read["sepnum"].values, [3, 1])


# --- Task 0: cross-phase unrecognised-column whitelist warning ----------


def test_read_spec_warns_on_unrecognised_column(tmp_path, recwarn):
    """Round-3 H10: typo'd column names like ``sepnnum`` are silently ignored
    today. ``read_spec`` should warn on unknown columns so users notice."""
    typo = "id,name,target,spf,sepnnum\n1,a,10.0,1.0,3\n"
    out = tmp_path / "spec.dat"
    out.write_text(typo)

    with pytest.warns(UserWarning, match="unrecognised"):
        read_spec(out)


def test_read_spec_silent_on_known_columns(tmp_path):
    """No warning when every column is on the Phase 18+19+20 whitelist."""
    import warnings

    legacy = (
        "id,name,target,prop,spf,ptarget,target2,clumptype,sepdistance,sepnum\n"
        "1,a,10.0,0.0,1.0,-1,0,0,0.0,1\n"
    )
    out = tmp_path / "spec.dat"
    out.write_text(legacy)

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        read_spec(out)


# --- Task 3: validate() smoke -------------------------------------------


def test_validate_accepts_separation_columns(tmp_path):
    """``ConservationProblem.validate()`` returns no errors for a problem
    whose ``features`` carry the new ``sepdistance`` / ``sepnum`` columns.

    Phase 20 doesn't add new required columns; the existing validate()
    just needs to not reject the new optional ones."""
    from pymarxan.models.problem import ConservationProblem

    features = pd.DataFrame({
        "id": [1, 2],
        "name": ["a", "b"],
        "target": [10.0, 20.0],
        "spf": [1.0, 2.0],
        "sepdistance": [5000.0, 0.0],
        "sepnum": [3, 1],
    })
    planning_units = pd.DataFrame({
        "id": [1, 2], "cost": [1.0, 1.0], "status": [0, 0],
    })
    puvspr = pd.DataFrame({
        "species": [1, 2], "pu": [1, 2], "amount": [10.0, 20.0],
    })

    problem = ConservationProblem(
        planning_units=planning_units,
        features=features,
        pu_vs_features=puvspr,
    )
    errors = problem.validate()
    assert errors == [], f"validate() unexpectedly rejected: {errors}"
