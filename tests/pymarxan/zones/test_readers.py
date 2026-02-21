from pathlib import Path

from pymarxan.zones.model import ZonalProblem
from pymarxan.zones.readers import (
    read_zones,
    read_zone_costs,
    read_zone_contributions,
    read_zone_targets,
    read_zone_boundary_costs,
    load_zone_project,
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
