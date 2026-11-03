"""save_zone_project is the inverse of load_zone_project (v0.49.0 R5, spec §3.8)."""
from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd
import pytest

from pymarxan.solvers.base import SolverConfig
from pymarxan.zones import ZonalProblem, ZoneMIPSolver, load_zone_project, save_zone_project

ZONES_DIR = Path(__file__).parent.parent.parent / "data" / "zones"
pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


def _mip(problem: ZonalProblem) -> tuple[list[int], float]:
    p = problem.copy_with(parameters={**problem.parameters, "MIP_GAP": 0.0})
    sol = ZoneMIPSolver(mip_backend="cbc").solve(p, SolverConfig(num_solutions=1))[0]
    return sol.zone_assignment.tolist(), round(sol.cost, 9)


def _same(a: pd.DataFrame | None, b: pd.DataFrame | None, keys: list[str]) -> None:
    assert (a is None) == (b is None)
    if a is not None and b is not None:
        pd.testing.assert_frame_equal(
            a.sort_values(keys).reset_index(drop=True)[sorted(a.columns)],
            b.sort_values(keys).reset_index(drop=True)[sorted(b.columns)],
            check_dtype=False,
        )


def _load(d: Path) -> ZonalProblem:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return load_zone_project(d)


def test_marzone_fixture_round_trips(tmp_path: Path) -> None:
    original = _load(ZONES_DIR)
    save_zone_project(original, tmp_path)
    again = _load(tmp_path)
    assert again.validate() == []
    _same(original.zones, again.zones, ["id"])
    _same(original.zone_costs, again.zone_costs, ["pu", "zone"])
    _same(original.zone_contributions, again.zone_contributions, ["feature", "zone"])
    _same(original.zone_targets, again.zone_targets, ["zone", "feature"])
    _same(original.zone_boundary_costs, again.zone_boundary_costs, ["zone1", "zone2"])
    assert _mip(again) == _mip(original)


def test_locks_and_restrictions_are_named_and_round_trip(tmp_path: Path) -> None:
    p = _load(ZONES_DIR).copy_with(
        pu_locks=pd.DataFrame({"pu": [1], "zone": [2]}),
        pu_zones=pd.DataFrame({"pu": [2, 2], "zone": [1, 2]}),
    )
    save_zone_project(p, tmp_path)
    text = (tmp_path / "input.dat").read_text()
    assert "PULOCKNAME pulock.dat" in text and "PUZONENAME puzone.dat" in text
    again = _load(tmp_path)
    assert again.validate() == []
    _same(p.pu_locks, again.pu_locks, ["pu"])
    _same(p.pu_zones, again.pu_zones, ["pu", "zone"])
    assert _mip(again) == _mip(p)


def test_saving_without_a_table_removes_its_stale_file(tmp_path: Path) -> None:
    locked = _load(ZONES_DIR).copy_with(pu_locks=pd.DataFrame({"pu": [1], "zone": [2]}))
    save_zone_project(locked, tmp_path)
    plain = locked.copy_with(pu_locks=None, zone_contributions=None)
    save_zone_project(plain, tmp_path)
    again = _load(tmp_path)
    assert again.pu_locks is None and again.zone_contributions is None
    assert not (tmp_path / "input" / "pulock.dat").exists()
    assert "PULOCKNAME" not in (tmp_path / "input.dat").read_text()
