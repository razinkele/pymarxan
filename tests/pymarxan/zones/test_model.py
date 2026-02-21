import pandas as pd

from pymarxan.zones.model import ZonalProblem


def _make_base_data():
    """Minimal base data: 4 PUs, 2 features."""
    planning_units = pd.DataFrame({
        "id": [1, 2, 3, 4],
        "cost": [10.0, 15.0, 20.0, 12.0],
        "status": [0, 0, 0, 0],
    })
    features = pd.DataFrame({
        "id": [1, 2],
        "name": ["sp_a", "sp_b"],
        "target": [20.0, 15.0],
        "spf": [1.0, 1.0],
    })
    pu_vs_features = pd.DataFrame({
        "species": [1, 1, 1, 1, 2, 2, 2, 2],
        "pu": [1, 2, 3, 4, 1, 2, 3, 4],
        "amount": [10.0, 8.0, 6.0, 5.0, 5.0, 7.0, 9.0, 4.0],
    })
    return planning_units, features, pu_vs_features


def _make_zone_data():
    """Two zones: 'protected' and 'sustainable'."""
    zones = pd.DataFrame({
        "id": [1, 2],
        "name": ["protected", "sustainable"],
    })
    zone_costs = pd.DataFrame({
        "pu": [1, 1, 2, 2, 3, 3, 4, 4],
        "zone": [1, 2, 1, 2, 1, 2, 1, 2],
        "cost": [100.0, 50.0, 150.0, 80.0, 200.0, 100.0, 120.0, 60.0],
    })
    zone_contributions = pd.DataFrame({
        "feature": [1, 1, 2, 2],
        "zone": [1, 2, 1, 2],
        "contribution": [1.0, 0.5, 1.0, 0.3],
    })
    zone_targets = pd.DataFrame({
        "zone": [1, 1, 2, 2],
        "feature": [1, 2, 1, 2],
        "target": [10.0, 8.0, 5.0, 3.0],
    })
    zone_boundary_costs = pd.DataFrame({
        "zone1": [1, 1, 2],
        "zone2": [1, 2, 2],
        "cost": [0.0, 50.0, 0.0],
    })
    return zones, zone_costs, zone_contributions, zone_targets, zone_boundary_costs


class TestZonalProblem:
    def test_create(self):
        pu, feat, puvspr = _make_base_data()
        zones, zc, zcontrib, zt, zbc = _make_zone_data()
        zp = ZonalProblem(
            planning_units=pu, features=feat, pu_vs_features=puvspr,
            zones=zones, zone_costs=zc,
            zone_contributions=zcontrib, zone_targets=zt,
            zone_boundary_costs=zbc,
        )
        assert zp.n_zones == 2
        assert zp.n_planning_units == 4
        assert zp.n_features == 2

    def test_zone_ids(self):
        pu, feat, puvspr = _make_base_data()
        zones, zc, zcontrib, zt, zbc = _make_zone_data()
        zp = ZonalProblem(
            planning_units=pu, features=feat, pu_vs_features=puvspr,
            zones=zones, zone_costs=zc,
        )
        assert zp.zone_ids == {1, 2}

    def test_get_zone_cost(self):
        pu, feat, puvspr = _make_base_data()
        zones, zc, zcontrib, zt, zbc = _make_zone_data()
        zp = ZonalProblem(
            planning_units=pu, features=feat, pu_vs_features=puvspr,
            zones=zones, zone_costs=zc,
        )
        assert zp.get_zone_cost(1, 1) == 100.0
        assert zp.get_zone_cost(1, 2) == 50.0

    def test_get_contribution(self):
        pu, feat, puvspr = _make_base_data()
        zones, zc, zcontrib, zt, zbc = _make_zone_data()
        zp = ZonalProblem(
            planning_units=pu, features=feat, pu_vs_features=puvspr,
            zones=zones, zone_costs=zc,
            zone_contributions=zcontrib,
        )
        assert zp.get_contribution(1, 1) == 1.0
        assert zp.get_contribution(1, 2) == 0.5

    def test_default_contribution_is_one(self):
        pu, feat, puvspr = _make_base_data()
        zones, zc, _, _, _ = _make_zone_data()
        zp = ZonalProblem(
            planning_units=pu, features=feat, pu_vs_features=puvspr,
            zones=zones, zone_costs=zc,
        )
        assert zp.get_contribution(1, 1) == 1.0

    def test_validate_valid(self):
        pu, feat, puvspr = _make_base_data()
        zones, zc, zcontrib, zt, zbc = _make_zone_data()
        zp = ZonalProblem(
            planning_units=pu, features=feat, pu_vs_features=puvspr,
            zones=zones, zone_costs=zc,
            zone_contributions=zcontrib, zone_targets=zt,
            zone_boundary_costs=zbc,
        )
        errors = zp.validate()
        assert errors == []

    def test_validate_missing_zone_cost(self):
        pu, feat, puvspr = _make_base_data()
        zones, _, _, _, _ = _make_zone_data()
        zc_incomplete = pd.DataFrame({
            "pu": [1, 1, 2, 2, 3, 3],
            "zone": [1, 2, 1, 2, 1, 2],
            "cost": [100.0, 50.0, 150.0, 80.0, 200.0, 100.0],
        })
        zp = ZonalProblem(
            planning_units=pu, features=feat, pu_vs_features=puvspr,
            zones=zones, zone_costs=zc_incomplete,
        )
        errors = zp.validate()
        assert any("zone_costs" in e for e in errors)
