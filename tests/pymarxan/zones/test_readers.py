import warnings
from pathlib import Path

import pandas as pd
import pytest

from pymarxan.io.readers import read_input_dat
from pymarxan.io.writers import save_project, write_input_dat
from pymarxan.zones.model import ZonalProblem
from pymarxan.zones.readers import (
    load_zone_project,
    read_zone_boundary_costs,
    read_zone_contributions,
    read_zone_costs,
    read_zone_targets,
    read_zones,
    resolve_zone_target_types,
)
from pymarxan.zones.writers import (
    write_zone_boundary_costs,
    write_zone_contributions,
    write_zone_costs,
    write_zone_targets,
    write_zones,
)

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "zones"
INPUT_DIR = DATA_DIR / "input"


class TestReadZones:
    def test_reads_zones(self):
        df = read_zones(INPUT_DIR / "zones.dat")
        assert len(df) == 2
        assert set(df.columns) >= {"id", "name"}
        assert df["id"].dtype == int

class TestReadZoneCosts:
    def test_reads_costs(self):
        df = read_zone_costs(INPUT_DIR / "zonecost.dat")
        assert len(df) == 8
        assert set(df.columns) >= {"pu", "zone", "cost"}

    def test_types(self):
        df = read_zone_costs(INPUT_DIR / "zonecost.dat")
        assert df["pu"].dtype == int
        assert df["zone"].dtype == int
        assert df["cost"].dtype == float

class TestReadZoneContributions:
    def test_reads_contributions(self):
        df = read_zone_contributions(INPUT_DIR / "zonecontrib.dat")
        assert len(df) == 4
        assert set(df.columns) >= {"feature", "zone", "contribution"}

    def test_values_in_range(self):
        df = read_zone_contributions(INPUT_DIR / "zonecontrib.dat")
        assert (df["contribution"] >= 0).all()
        assert (df["contribution"] <= 1).all()

class TestReadZoneTargets:
    def test_reads_targets(self):
        df = read_zone_targets(INPUT_DIR / "zonetarget.dat")
        assert len(df) == 4
        assert set(df.columns) >= {"zone", "feature", "target"}

class TestReadZoneBoundaryCosts:
    def test_reads_boundary_costs(self):
        df = read_zone_boundary_costs(INPUT_DIR / "zoneboundcost.dat")
        assert len(df) == 4
        assert set(df.columns) >= {"zone1", "zone2", "cost"}

class TestLoadZoneProject:
    def test_loads_full_project(self):
        zp = load_zone_project(DATA_DIR)
        assert isinstance(zp, ZonalProblem)
        assert zp.n_planning_units == 4
        assert zp.n_features == 2
        assert zp.n_zones == 2

    def test_validates_clean(self):
        zp = load_zone_project(DATA_DIR)
        assert zp.validate() == []

    def test_zone_costs_loaded(self):
        zp = load_zone_project(DATA_DIR)
        assert zp.get_zone_cost(1, 1) == 100.0
        assert zp.get_zone_cost(1, 2) == 50.0

    def test_contributions_loaded(self):
        zp = load_zone_project(DATA_DIR)
        assert zp.get_contribution(1, 1) == 1.0
        assert zp.get_contribution(1, 2) == 0.5


def _write_zonetarget(tmp_path: Path, rows: str) -> Path:
    path = tmp_path / "zonetarget.dat"
    path.write_text("zone,feature,target,targettype\n" + rows)
    return path


class TestZoneTargetType:
    def test_type_zero_is_unchanged_and_column_kept(self, tmp_path: Path):
        df = read_zone_targets(_write_zonetarget(tmp_path, "1,1,10.0,0\n2,2,3.0,0\n"))
        assert list(df["targettype"]) == [0, 0]
        assert list(df["target"]) == [10.0, 3.0]

    def test_type_one_resolves_to_fraction_of_feature_total(self, tmp_path: Path):
        problem = load_zone_project(DATA_DIR)          # feature 1 total raw amount = 29
        df = read_zone_targets(_write_zonetarget(tmp_path, "1,1,0.5,1\n2,2,3.0,0\n"))
        out = resolve_zone_target_types(df, problem.pu_vs_features)
        assert out.loc[0, "target"] == pytest.approx(14.5)
        assert out.loc[0, "targettype"] == 0                # resolved rows become type 0
        assert out.loc[1, "target"] == 3.0
        assert list(df["target"]) == [0.5, 3.0]             # input not mutated

    def test_resolution_is_idempotent_across_write_and_read(self, tmp_path: Path):
        problem = load_zone_project(DATA_DIR)
        df = read_zone_targets(_write_zonetarget(tmp_path, "1,1,0.5,1\n"))
        once = resolve_zone_target_types(df, problem.pu_vs_features)
        write_zone_targets(once, tmp_path / "again.dat")
        twice = resolve_zone_target_types(
            read_zone_targets(tmp_path / "again.dat"), problem.pu_vs_features,
        )
        assert twice.loc[0, "target"] == pytest.approx(14.5)

    def test_missing_column_passes_through(self):
        problem = load_zone_project(DATA_DIR)
        out = resolve_zone_target_types(problem.zone_targets, problem.pu_vs_features)
        pd.testing.assert_frame_equal(out, problem.zone_targets)

    @pytest.mark.parametrize("ttype", [2, 3])
    def test_occurrence_types_are_rejected_naming_the_row(self, tmp_path: Path, ttype: int):
        path = _write_zonetarget(tmp_path, f"1,1,10.0,0\n2,1,4,{ttype}\n")
        pattern = r"row 2 .*zone 2.*feature 1.*targettype " + str(ttype)
        with pytest.raises(ValueError, match=pattern):
            read_zone_targets(path)

    def test_unknown_type_is_rejected(self, tmp_path: Path):
        with pytest.raises(ValueError, match="targettype 7"):
            read_zone_targets(_write_zonetarget(tmp_path, "1,1,10.0,7\n"))

    def test_all_species_wildcard_is_rejected(self, tmp_path: Path):
        """MarZone's speciesid = -1 (zones.hpp:127-138) would resolve to a phantom target 0."""
        with pytest.raises(ValueError, match=r"row 1 has feature -1; MarZone's -1 'all species'"):
            read_zone_targets(_write_zonetarget(tmp_path, "1,-1,10.0,0\n"))

    def test_load_zone_project_applies_resolution(self, tmp_path: Path):
        _copy_zone_project(load_zone_project(DATA_DIR), tmp_path)
        (tmp_path / "input" / "zonetarget.dat").write_text(
            "zone,feature,target,targettype\n1,1,0.5,1\n2,2,3.0,0\n"
        )
        problem = load_zone_project(tmp_path)
        assert problem.zone_targets.loc[0, "target"] == pytest.approx(14.5)

    def test_validate_flags_unresolved_targettype(self, tmp_path: Path):
        """A frame straight from read_zone_targets (or built by hand) with targettype 1 would
        otherwise use 0.5 as an absolute amount and be trivially met."""
        _copy_zone_project(load_zone_project(DATA_DIR), tmp_path)
        zt_path = tmp_path / "input" / "zonetarget.dat"
        zt_path.write_text("zone,feature,target,targettype\n1,1,0.5,1\n2,2,3.0,0\n")
        raw = load_zone_project(DATA_DIR).copy_with(zone_targets=read_zone_targets(zt_path))
        msgs = [e for e in raw.validate() if "targettype" in e]
        assert len(msgs) == 1
        assert "1 row(s)" in msgs[0] and "resolve_zone_target_types" in msgs[0]
        assert load_zone_project(tmp_path).validate() == []


class TestPartialContributionTableWarning:
    def test_load_zone_project_warns_on_partial_contributions(self, tmp_path: Path):
        """Ruling R3: gaps are advisory — one UserWarning from the loader, validate() silent."""
        _copy_zone_project(load_zone_project(DATA_DIR), tmp_path)
        (tmp_path / "input" / "zonecontrib.dat").write_text(
            "feature,zone,contribution\n1,1,1.0\n"
        )
        with pytest.warns(UserWarning, match="3 unlisted"):
            problem = load_zone_project(tmp_path)
        assert problem.contribution_gaps() == [(1, 2), (2, 1), (2, 2)]
        assert problem.validate() == []

    def test_warning_names_the_configured_contribution_file(self, tmp_path: Path):
        problem = load_zone_project(DATA_DIR)
        renamed = problem.copy_with(
            parameters={**problem.parameters, "ZONECONTRIBNAME": "custom.dat"}
        )
        _copy_zone_project(renamed, tmp_path)
        (tmp_path / "input" / "zonecontrib.dat").unlink()
        (tmp_path / "input" / "custom.dat").write_text("feature,zone,contribution\n1,1,1.0\n")
        with pytest.warns(UserWarning, match="3 unlisted") as record:
            load_zone_project(tmp_path)
        messages = [str(w.message) for w in record]
        assert any("custom.dat" in m for m in messages)
        assert not any("zonecontrib.dat" in m for m in messages)

    def test_full_table_does_not_warn(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            problem = load_zone_project(DATA_DIR)      # fixture lists all four pairs
        assert problem.contribution_gaps() == []


def _copy_zone_project(problem: ZonalProblem, dest: Path) -> None:
    """Write ``problem`` into ``dest`` with the existing writers (ruling R1: there is no
    save_zone_project yet; this helper is its future body)."""
    save_project(problem, dest)                                   # base files + input.dat
    input_dir = dest / "input"
    write_zones(problem.zones, input_dir / "zones.dat")
    write_zone_costs(problem.zone_costs, input_dir / "zonecost.dat")
    write_zone_contributions(problem.zone_contributions, input_dir / "zonecontrib.dat")
    write_zone_targets(problem.zone_targets, input_dir / "zonetarget.dat")
    write_zone_boundary_costs(problem.zone_boundary_costs, input_dir / "zoneboundcost.dat")


class TestZoneTargetContribRoundTrip:
    def test_input_dat_round_trips_as_int(self, tmp_path: Path):
        write_input_dat({"ZONETARGETCONTRIB": 1, "BLM": 1.0}, tmp_path / "input.dat")
        params = read_input_dat(tmp_path / "input.dat")
        assert params["ZONETARGETCONTRIB"] == 1
        assert isinstance(params["ZONETARGETCONTRIB"], int)

    def test_full_project_round_trip_keeps_the_parameter(self, tmp_path: Path):
        problem = load_zone_project(DATA_DIR)
        mutated = problem.copy_with(parameters={**problem.parameters, "ZONETARGETCONTRIB": 1})
        _copy_zone_project(mutated, tmp_path)
        again = load_zone_project(tmp_path)
        assert again.zone_target_contrib() == 1
        assert again.validate() == []

    def test_out_of_domain_value_loads_but_fails_validation(self, tmp_path: Path):
        _copy_zone_project(load_zone_project(DATA_DIR), tmp_path)
        text = (tmp_path / "input.dat").read_text()
        (tmp_path / "input.dat").write_text(text + "ZONETARGETCONTRIB 2\n")
        problem = load_zone_project(tmp_path)
        assert any("ZONETARGETCONTRIB" in e for e in problem.validate())
