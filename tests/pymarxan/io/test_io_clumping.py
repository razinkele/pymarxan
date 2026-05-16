"""Tests for Marxan-compatible clumping columns on spec.dat.

Phase 19 Batch 1: optional `target2` and `clumptype` columns on spec.dat
(Marxan TARGET2 / CLUMPTYPE wire format). `target2 <= 0` disables clumping
for the feature; `clumptype` is 0/1/2 (binary / amount-÷-2 / quadratic per
``clumping.cpp::PartialPen4``).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pymarxan.io.readers import read_spec
from pymarxan.io.writers import write_spec

# --- Task 1: target2 column round-trip ----------------------------------


def test_spec_round_trip_with_target2(tmp_path):
    """An explicit `target2` column round-trips through write → read."""
    df = pd.DataFrame({
        "id": [1, 2, 3],
        "name": ["a", "b", "c"],
        "target": [10.0, 20.0, 30.0],
        "spf": [1.0, 2.0, 3.0],
        "target2": [5.0, 0.0, 12.5],
    })
    out = tmp_path / "spec.dat"
    write_spec(df, out)

    read = read_spec(out)
    assert "target2" in read.columns
    np.testing.assert_array_almost_equal(
        read["target2"].values, [5.0, 0.0, 12.5],
    )


def test_spec_omits_target2_when_all_zero(tmp_path):
    """Writer omits target2 when every value is 0 (the disabled default).

    Keeps legacy non-clumping projects byte-identical on round-trip.
    """
    df = pd.DataFrame({
        "id": [1], "name": ["a"], "target": [10.0], "spf": [1.0],
        "target2": [0.0],
    })
    out = tmp_path / "spec.dat"
    write_spec(df, out)

    content = out.read_text()
    assert "target2" not in content, "writer must omit all-zero target2 column"


def test_spec_default_target2_when_column_absent(tmp_path):
    """Reading a legacy spec.dat without target2 fills 0.0.

    Lets downstream solver code unconditionally read `features["target2"]`
    without checking column presence.
    """
    legacy = "id,name,target,spf\n1,a,10.0,1.0\n"
    out = tmp_path / "spec.dat"
    out.write_text(legacy)

    df = read_spec(out)
    assert "target2" in df.columns
    assert df["target2"].iloc[0] == 0.0


# --- Task 2: clumptype column round-trip --------------------------------


def test_spec_round_trip_with_clumptype(tmp_path):
    """An explicit `clumptype` column round-trips. Values stay int 0/1/2."""
    df = pd.DataFrame({
        "id": [1, 2, 3],
        "name": ["a", "b", "c"],
        "target": [10.0, 20.0, 30.0],
        "spf": [1.0, 2.0, 3.0],
        "target2": [5.0, 0.0, 12.5],
        "clumptype": [0, 1, 2],
    })
    out = tmp_path / "spec.dat"
    write_spec(df, out)

    read = read_spec(out)
    assert "clumptype" in read.columns
    np.testing.assert_array_equal(read["clumptype"].values, [0, 1, 2])


def test_spec_omits_clumptype_when_all_zero(tmp_path):
    """Writer omits clumptype when every value is 0 (the binary default)."""
    df = pd.DataFrame({
        "id": [1], "name": ["a"], "target": [10.0], "spf": [1.0],
        "clumptype": [0],
    })
    out = tmp_path / "spec.dat"
    write_spec(df, out)

    content = out.read_text()
    assert "clumptype" not in content


def test_spec_default_clumptype_when_column_absent(tmp_path):
    """Reading a legacy spec.dat without clumptype fills 0 (binary)."""
    legacy = "id,name,target,spf\n1,a,10.0,1.0\n"
    out = tmp_path / "spec.dat"
    out.write_text(legacy)

    df = read_spec(out)
    assert "clumptype" in df.columns
    assert int(df["clumptype"].iloc[0]) == 0


def test_spec_rejects_invalid_clumptype(tmp_path):
    """Reading a spec with clumptype outside {0,1,2} raises ValueError.

    Catches user errors at I/O time rather than letting them propagate to
    the SA inner loop where the wrong code path is harder to diagnose.
    """
    bad = "id,name,target,spf,clumptype\n1,a,10.0,1.0,5\n"
    out = tmp_path / "spec.dat"
    out.write_text(bad)

    with pytest.raises(ValueError, match="clumptype"):
        read_spec(out)


# --- Combined target2 + clumptype + ptarget round-trip ------------------


def test_spec_round_trip_all_phase18_19_columns(tmp_path):
    """Phase 18 (ptarget) + Phase 19 (target2, clumptype) coexist."""
    df = pd.DataFrame({
        "id": [1, 2],
        "name": ["a", "b"],
        "target": [10.0, 20.0],
        "spf": [1.0, 2.0],
        "ptarget": [0.95, -1.0],
        "target2": [5.0, 0.0],
        "clumptype": [1, 0],
    })
    out = tmp_path / "spec.dat"
    write_spec(df, out)

    read = read_spec(out)
    np.testing.assert_array_almost_equal(read["ptarget"].values, [0.95, -1.0])
    np.testing.assert_array_almost_equal(read["target2"].values, [5.0, 0.0])
    np.testing.assert_array_equal(read["clumptype"].values, [1, 0])
