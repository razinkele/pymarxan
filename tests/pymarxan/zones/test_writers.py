"""Tests for zone writers – CSV output and roundtrip with readers."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from pymarxan.solvers.base import Solution
from pymarxan.zones.model import ZonalProblem
from pymarxan.zones.readers import (
    read_zone_boundary_costs,
    read_zone_contributions,
    read_zone_costs,
    read_zone_targets,
    read_zones,
)
from pymarxan.zones.writers import (
    write_zone_boundary_costs,
    write_zone_contributions,
    write_zone_costs,
    write_zone_solution,
    write_zone_summary,
    write_zone_targets,
    write_zones,
)

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "zones"
INPUT_DIR = DATA_DIR / "input"


# ── helpers ────────────────────────────────────────────────────────────


def _make_zones_df() -> pd.DataFrame:
    return pd.DataFrame({"id": [1, 2], "name": ["core", "buffer"]})


def _make_zone_costs_df() -> pd.DataFrame:
    return pd.DataFrame({
        "pu": [1, 1, 2, 2],
        "zone": [1, 2, 1, 2],
        "cost": [100.0, 50.0, 200.0, 80.0],
    })


def _make_zone_contributions_df() -> pd.DataFrame:
    return pd.DataFrame({
        "feature": [1, 1, 2, 2],
        "zone": [1, 2, 1, 2],
        "contribution": [1.0, 0.5, 0.8, 0.3],
    })


def _make_zone_targets_df() -> pd.DataFrame:
    return pd.DataFrame({
        "zone": [1, 1, 2, 2],
        "feature": [1, 2, 1, 2],
        "target": [50.0, 30.0, 20.0, 10.0],
    })


def _make_zone_boundary_costs_df() -> pd.DataFrame:
    return pd.DataFrame({
        "zone1": [1, 1, 2, 2],
        "zone2": [1, 2, 1, 2],
        "cost": [1.0, 0.5, 0.5, 1.0],
    })


def _make_solution(
    selected: list[bool],
    zone_assignment: list[int] | None = None,
) -> Solution:
    sel = np.array(selected, dtype=bool)
    za = (
        np.array(zone_assignment, dtype=int)
        if zone_assignment is not None
        else None
    )
    return Solution(
        selected=sel,
        cost=100.0,
        boundary=10.0,
        objective=120.0,
        targets_met={1: True, 2: False},
        penalty=5.0,
        shortfall=2.0,
        zone_assignment=za,
    )


def _make_zonal_problem() -> ZonalProblem:
    pu = pd.DataFrame({"id": [1, 2], "cost": [100, 200], "status": [0, 0]})
    features = pd.DataFrame({
        "id": [1, 2],
        "name": ["feat_a", "feat_b"],
        "target": [50.0, 30.0],
        "spf": [1.0, 1.0],
    })
    puvspr = pd.DataFrame({
        "species": [1, 1, 2, 2],
        "pu": [1, 2, 1, 2],
        "amount": [30.0, 40.0, 20.0, 25.0],
    })
    zones = _make_zones_df()
    zone_costs = pd.DataFrame({
        "pu": [1, 1, 2, 2],
        "zone": [1, 2, 1, 2],
        "cost": [100.0, 50.0, 200.0, 80.0],
    })
    zone_contributions = _make_zone_contributions_df()

    return ZonalProblem(
        planning_units=pu,
        features=features,
        pu_vs_features=puvspr,
        boundary=None,
        parameters={},
        zones=zones,
        zone_costs=zone_costs,
        zone_contributions=zone_contributions,
        zone_targets=None,
        zone_boundary_costs=None,
    )


# ── write + CSV validity tests ────────────────────────────────────────


class TestWriteZones:
    def test_produces_valid_csv(self, tmp_path: Path) -> None:
        df = _make_zones_df()
        out = tmp_path / "zones.dat"
        write_zones(df, out)
        result = pd.read_csv(out)
        assert list(result.columns) == ["id", "name"]
        assert len(result) == 2

    def test_roundtrip(self, tmp_path: Path) -> None:
        original = read_zones(INPUT_DIR / "zones.dat")
        out = tmp_path / "zones.dat"
        write_zones(original, out)
        roundtripped = read_zones(out)
        pd.testing.assert_frame_equal(original, roundtripped)


class TestWriteZoneCosts:
    def test_produces_valid_csv(self, tmp_path: Path) -> None:
        df = _make_zone_costs_df()
        out = tmp_path / "zonecost.dat"
        write_zone_costs(df, out)
        result = pd.read_csv(out)
        assert set(result.columns) >= {"pu", "zone", "cost"}
        assert len(result) == 4

    def test_roundtrip(self, tmp_path: Path) -> None:
        original = read_zone_costs(INPUT_DIR / "zonecost.dat")
        out = tmp_path / "zonecost.dat"
        write_zone_costs(original, out)
        roundtripped = read_zone_costs(out)
        pd.testing.assert_frame_equal(original, roundtripped)


class TestWriteZoneContributions:
    def test_produces_valid_csv(self, tmp_path: Path) -> None:
        df = _make_zone_contributions_df()
        out = tmp_path / "zonecontrib.dat"
        write_zone_contributions(df, out)
        result = pd.read_csv(out)
        assert set(result.columns) >= {"feature", "zone", "contribution"}
        assert len(result) == 4

    def test_roundtrip(self, tmp_path: Path) -> None:
        original = read_zone_contributions(
            INPUT_DIR / "zonecontrib.dat"
        )
        out = tmp_path / "zonecontrib.dat"
        write_zone_contributions(original, out)
        roundtripped = read_zone_contributions(out)
        pd.testing.assert_frame_equal(original, roundtripped)


class TestWriteZoneTargets:
    def test_produces_valid_csv(self, tmp_path: Path) -> None:
        df = _make_zone_targets_df()
        out = tmp_path / "zonetarget.dat"
        write_zone_targets(df, out)
        result = pd.read_csv(out)
        assert set(result.columns) >= {"zone", "feature", "target"}
        assert len(result) == 4

    def test_roundtrip(self, tmp_path: Path) -> None:
        original = read_zone_targets(INPUT_DIR / "zonetarget.dat")
        out = tmp_path / "zonetarget.dat"
        write_zone_targets(original, out)
        roundtripped = read_zone_targets(out)
        pd.testing.assert_frame_equal(original, roundtripped)


class TestWriteZoneBoundaryCosts:
    def test_produces_valid_csv(self, tmp_path: Path) -> None:
        df = _make_zone_boundary_costs_df()
        out = tmp_path / "zoneboundcost.dat"
        write_zone_boundary_costs(df, out)
        result = pd.read_csv(out)
        assert set(result.columns) >= {"zone1", "zone2", "cost"}
        assert len(result) == 4

    def test_roundtrip(self, tmp_path: Path) -> None:
        original = read_zone_boundary_costs(
            INPUT_DIR / "zoneboundcost.dat"
        )
        out = tmp_path / "zoneboundcost.dat"
        write_zone_boundary_costs(original, out)
        roundtripped = read_zone_boundary_costs(out)
        pd.testing.assert_frame_equal(original, roundtripped)


class TestWriteZoneSolution:
    def test_with_zone_assignment(self, tmp_path: Path) -> None:
        sol = _make_solution(
            selected=[True, False, True],
            zone_assignment=[1, 0, 2],
        )
        out = tmp_path / "zone_soln.csv"
        write_zone_solution(sol, out)
        result = pd.read_csv(out)
        assert list(result.columns) == ["planning_unit", "zone"]
        assert result["planning_unit"].tolist() == [1, 2, 3]
        assert result["zone"].tolist() == [1, 0, 2]

    def test_without_zone_assignment(self, tmp_path: Path) -> None:
        sol = _make_solution(selected=[True, False, True])
        out = tmp_path / "zone_soln.csv"
        write_zone_solution(sol, out)
        result = pd.read_csv(out)
        assert result["zone"].tolist() == [1, 0, 1]

    def test_zone_zero_means_unassigned(self, tmp_path: Path) -> None:
        sol = _make_solution(
            selected=[False, False],
            zone_assignment=[0, 0],
        )
        out = tmp_path / "zone_soln.csv"
        write_zone_solution(sol, out)
        result = pd.read_csv(out)
        assert (result["zone"] == 0).all()


class TestWriteZoneSummary:
    def test_summary_structure(self, tmp_path: Path) -> None:
        problem = _make_zonal_problem()
        sol1 = _make_solution(
            selected=[True, True],
            zone_assignment=[1, 2],
        )
        sol2 = _make_solution(
            selected=[True, False],
            zone_assignment=[1, 0],
        )
        out = tmp_path / "zone_summary.csv"
        write_zone_summary(problem, [sol1, sol2], out)
        result = pd.read_csv(out)
        assert set(result.columns) == {
            "zone", "feature", "target", "times_met", "total_runs",
        }
        # 2 zones × 2 features = 4 rows
        assert len(result) == 4
        assert (result["total_runs"] == 2).all()

    def test_summary_times_met(self, tmp_path: Path) -> None:
        problem = _make_zonal_problem()
        # Both PUs in zone 1: achieved = (30+40)*1.0 = 70 >= 50 → met
        sol = _make_solution(
            selected=[True, True],
            zone_assignment=[1, 1],
        )
        out = tmp_path / "zone_summary.csv"
        write_zone_summary(problem, [sol], out)
        result = pd.read_csv(out)
        z1_f1 = result[
            (result["zone"] == 1) & (result["feature"] == 1)
        ]
        assert z1_f1["times_met"].iloc[0] == 1

    def test_summary_multiple_solutions(self, tmp_path: Path) -> None:
        problem = _make_zonal_problem()
        # sol1: PU1→zone1, PU2→zone1
        #   zone1 feat1: (30+40)*1.0=70 >=50 → met
        sol1 = _make_solution(
            selected=[True, True],
            zone_assignment=[1, 1],
        )
        # sol2: PU1→zone2, PU2→zone2
        #   zone1 feat1: nothing → 0 < 50 → not met
        sol2 = _make_solution(
            selected=[True, True],
            zone_assignment=[2, 2],
        )
        out = tmp_path / "zone_summary.csv"
        write_zone_summary(problem, [sol1, sol2], out)
        result = pd.read_csv(out)
        z1_f1 = result[
            (result["zone"] == 1) & (result["feature"] == 1)
        ]
        assert z1_f1["times_met"].iloc[0] == 1
        assert z1_f1["total_runs"].iloc[0] == 2
