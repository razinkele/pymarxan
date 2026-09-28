"""Tests for zone writers – CSV output and roundtrip with readers."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pymarxan.solvers.base import Solution
from pymarxan.zones.readers import (
    load_zone_project,
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
    """Rows equal the objective module's numbers (review H5: old writer matched neither tier)."""

    def _problem_and_solutions(self):
        problem = load_zone_project(DATA_DIR)
        sols = [
            _make_solution([True, True, True, True], [1, 1, 2, 1]),
            _make_solution([True, True, True, True], [2, 2, 2, 2]),
        ]
        return problem, sols

    def test_columns_and_row_groups(self, tmp_path: Path) -> None:
        problem, sols = self._problem_and_solutions()
        out = tmp_path / "zone_summary.csv"
        write_zone_summary(problem, sols, out)
        df = pd.read_csv(out)
        assert list(df.columns) == [
            "tier", "zone", "feature", "target", "mean_achieved", "times_met", "total_runs",
        ]
        assert (df["tier"] == "overall").sum() == 2          # one per feature
        assert (df["tier"] == "zone").sum() == 4             # one per listed zone target
        assert (df["total_runs"] == 2).all()
        assert (df.loc[df["tier"] == "overall", "zone"] == 0).all()

    def test_overall_rows_match_objective_module(self, tmp_path: Path) -> None:
        from pymarxan.zones.objective import check_overall_targets, compute_overall_achieved
        problem, sols = self._problem_and_solutions()
        out = tmp_path / "zone_summary.csv"
        write_zone_summary(problem, sols, out)
        df = pd.read_csv(out)
        for fid in (1, 2):
            row = df[(df["tier"] == "overall") & (df["feature"] == fid)].iloc[0]
            achieved = [compute_overall_achieved(problem, s.zone_assignment)[fid] for s in sols]
            met = [check_overall_targets(problem, s.zone_assignment)[fid] for s in sols]
            assert row["mean_achieved"] == pytest.approx(sum(achieved) / 2)
            assert row["times_met"] == sum(met)
            assert row["target"] == float(problem.features.set_index("id").loc[fid, "target"])

    def test_zone_rows_match_objective_module(self, tmp_path: Path) -> None:
        from pymarxan.zones.objective import _compute_zone_achieved, check_zone_targets
        problem, sols = self._problem_and_solutions()
        out = tmp_path / "zone_summary.csv"
        write_zone_summary(problem, sols, out)
        df = pd.read_csv(out)
        for zid, fid, target in problem.zone_targets[["zone", "feature", "target"]].itertuples(
            index=False,
        ):
            row = df[(df["tier"] == "zone") & (df["zone"] == zid) & (df["feature"] == fid)].iloc[0]
            achieved = [
                _compute_zone_achieved(problem, s.zone_assignment).get((zid, fid), 0.0)
                for s in sols
            ]
            met = [check_zone_targets(problem, s.zone_assignment)[(zid, fid)] for s in sols]
            assert row["mean_achieved"] == pytest.approx(sum(achieved) / 2)
            assert row["times_met"] == sum(met)
            assert row["target"] == float(target)

    def test_zone_rows_are_raw_by_default(self, tmp_path: Path) -> None:
        problem, sols = self._problem_and_solutions()
        out = tmp_path / "zone_summary.csv"
        write_zone_summary(problem, sols, out)
        df = pd.read_csv(out)
        row = df[(df["tier"] == "zone") & (df["zone"] == 2) & (df["feature"] == 1)].iloc[0]
        # run 1: PU3 raw 6; run 2: all four raw 29 -> mean 17.5 (weighted would be 8.75)
        assert row["mean_achieved"] == pytest.approx(17.5)

    def test_no_zone_targets_gives_overall_rows_only(self, tmp_path: Path) -> None:
        problem, sols = self._problem_and_solutions()
        problem = problem.copy_with(zone_targets=None)
        out = tmp_path / "zone_summary.csv"
        write_zone_summary(problem, sols, out)
        df = pd.read_csv(out)
        assert set(df["tier"]) == {"overall"}

    def test_selected_only_solution_counts_as_first_zone(self, tmp_path: Path) -> None:
        problem, _ = self._problem_and_solutions()
        sol = _make_solution([True, True, False, False], None)
        out = tmp_path / "zone_summary.csv"
        write_zone_summary(problem, [sol], out)
        df = pd.read_csv(out)
        row = df[(df["tier"] == "zone") & (df["zone"] == 1) & (df["feature"] == 1)].iloc[0]
        assert row["mean_achieved"] == pytest.approx(18.0)   # PU1 10 + PU2 8, raw
