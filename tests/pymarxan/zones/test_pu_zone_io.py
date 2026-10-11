"""pulock.dat / puzone.dat I/O (#6 spec §1; MarZone pu.hpp:708-790)."""
from __future__ import annotations

import shutil
import warnings
from pathlib import Path

import pandas as pd
import pytest

from pymarxan.zones import (
    load_zone_project,
    read_pu_lock,
    read_pu_zone,
    write_pu_lock,
    write_pu_zone,
)

ZONES_DIR = Path(__file__).parent.parent.parent / "data" / "zones"
MARZONE_PULOCK = b"puid, zoneid\r\n2,1\r\n5,2\r\n30,1"


def _frame(rows):
    return pd.DataFrame(rows, columns=["pu", "zone"])


def test_reads_marzone_fixture_bytes(tmp_path):
    f = tmp_path / "pulock.dat"
    f.write_bytes(MARZONE_PULOCK)
    pd.testing.assert_frame_equal(read_pu_lock(f), _frame([(2, 1), (5, 2), (30, 1)]))


def test_bom_and_crlf(tmp_path):
    f = tmp_path / "puzone.dat"
    f.write_bytes(b"\xef\xbb\xbfpuid,zoneid\r\n1,1\r\n1,2\r\n")
    pd.testing.assert_frame_equal(read_pu_zone(f), _frame([(1, 1), (1, 2)]))


def test_bom_headerless(tmp_path):
    # BOM on a DATA line: without utf-8-sig the first row looks like a header and is lost
    f = tmp_path / "puzone.dat"
    f.write_bytes(b"\xef\xbb\xbf1,1\r\n1,2\r\n")
    pd.testing.assert_frame_equal(read_pu_zone(f), _frame([(1, 1), (1, 2)]))


@pytest.mark.parametrize(
    "text", ["1\t2\n3\t1\n", "1 2\n3  1\n", "1\t2\n3 1\n"], ids=["tab", "space", "mixed"]
)
def test_headerless_tab_and_space(tmp_path, text):
    f = tmp_path / "pulock.dat"
    f.write_text(text)
    pd.testing.assert_frame_equal(read_pu_lock(f), _frame([(1, 2), (3, 1)]))


def test_header_only_is_empty(tmp_path):
    f = tmp_path / "pulock.dat"
    f.write_text("puid,zoneid\n")
    out = read_pu_lock(f)
    assert list(out.columns) == ["pu", "zone"] and len(out) == 0


@pytest.mark.parametrize("content", [
    b"", b"\n", b"\xef\xbb\xbf\r\n", b"puid,zoneid\n1,\n", b"puid,zoneid\n1,2.5\n",
    b"puid,zoneid\n1,2\n3 1\n",   # comma file, later whitespace line -> rejected
    b"1 2\n3\n",                    # one value on a line -> 'needs two values'
])
def test_bad_files_raise(tmp_path, content):
    f = tmp_path / "pulock.dat"
    f.write_bytes(content)
    with pytest.raises(ValueError, match="pulock.dat"):
        read_pu_lock(f)


def test_writers_round_trip(tmp_path):
    df = _frame([(1, 2), (4, 1)])
    write_pu_lock(df, tmp_path / "a.dat")
    write_pu_zone(df, tmp_path / "b.dat")
    assert (tmp_path / "a.dat").read_text().splitlines()[0] == "puid,zoneid"
    pd.testing.assert_frame_equal(read_pu_lock(tmp_path / "a.dat"), df)
    pd.testing.assert_frame_equal(read_pu_zone(tmp_path / "b.dat"), df)


def _project(tmp_path, extra_lines: list[str]) -> Path:
    proj = tmp_path / "proj"
    shutil.copytree(ZONES_DIR, proj)
    with open(proj / "input.dat", "a") as fh:
        fh.write("\n" + "\n".join(extra_lines) + "\n")
    return proj


def _load(proj):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return load_zone_project(proj)


def test_loader_reads_only_named_files(tmp_path):
    proj = _project(tmp_path, [])
    (proj / "input" / "pulock.dat").write_text("puid,zoneid\n1,2\n")
    assert _load(proj).pu_locks is None  # not named in input.dat -> not loaded (MarZone)
    proj2 = _project(tmp_path / "b", ["PULOCKNAME pulock.dat", "PUZONENAME puzone.dat"])
    (proj2 / "input" / "pulock.dat").write_text("puid,zoneid\n1,2\n")
    (proj2 / "input" / "puzone.dat").write_text("puid,zoneid\n2,1\n2,2\n")
    p = _load(proj2)
    assert p.pu_lock_lookup() == {1: 2}
    assert p.pu_zone_lookup() == {2: frozenset({1, 2})}


def test_loader_drops_unknown_pus_from_marzone_file(tmp_path):
    proj = _project(tmp_path, ["PULOCKNAME pulock.dat"])
    (proj / "input" / "pulock.dat").write_bytes(MARZONE_PULOCK)
    assert _load(proj).pu_lock_lookup() == {2: 1}  # PUs 5 and 30 are not in pu.dat


def test_loader_named_empty_file_raises(tmp_path):
    proj = _project(tmp_path, ["PUZONENAME puzone.dat"])
    (proj / "input" / "puzone.dat").write_bytes(b"")
    with pytest.raises(ValueError, match="puzone.dat"):
        _load(proj)


def test_blank_middle_cell_raises(tmp_path):
    f = tmp_path / "pulock.dat"
    f.write_text("puid,zoneid\n1,,2\n")
    with pytest.raises(ValueError, match="pulock.dat"):
        read_pu_lock(f)


def test_loader_named_missing_file_raises(tmp_path):
    proj = _project(tmp_path, ["PULOCKNAME nope.dat"])
    with pytest.raises(FileNotFoundError, match="nope.dat"):
        _load(proj)
