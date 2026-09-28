import numpy as np
import pandas as pd
import pytest

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


class TestContributionSource:
    def _problem(self, **kw):
        pu, feat, puvspr = _make_base_data()
        zones, zc, zcontrib, zt, zbc = _make_zone_data()
        base = dict(
            planning_units=pu, features=feat, pu_vs_features=puvspr,
            zones=zones, zone_costs=zc,
        )
        base.update(kw)
        return ZonalProblem(**base)

    def test_zone_index_is_rank_in_sorted_ids_plus_one(self):
        zp = self._problem()
        assert zp.zone_index() == {1: 1, 2: 2}

    def test_feature_index_follows_features_order(self):
        zp = self._problem()
        assert zp.feature_index() == {1: 0, 2: 1}

    def test_default_is_one_without_table(self):
        zp = self._problem()
        assert zp.contribution_default() == 1.0
        assert zp.get_contribution(1, 2) == 1.0
        assert zp.contribution_lookup() == {}

    def test_default_is_zero_for_unlisted_pair_with_table(self):
        partial = pd.DataFrame({
            "feature": [1, 2], "zone": [1, 1], "contribution": [1.0, 0.8],
        })
        zp = self._problem(zone_contributions=partial)
        assert zp.contribution_default() == 0.0
        assert zp.get_contribution(1, 1) == 1.0
        assert zp.get_contribution(1, 2) == 0.0  # unlisted: MarZone zones.hpp:619
        assert zp.contribution_lookup() == {(1, 1): 1.0, (2, 1): 0.8}

    def test_contribution_matrix_layout(self):
        _, _, zcontrib, _, _ = _make_zone_data()
        zp = self._problem(zone_contributions=zcontrib)
        m = zp.contribution_matrix()
        assert m.shape == (3, 2)
        np.testing.assert_array_equal(m[0], [0.0, 0.0])
        np.testing.assert_array_equal(m[1], [1.0, 1.0])   # zone 1: f1=1.0, f2=1.0
        np.testing.assert_array_equal(m[2], [0.5, 0.3])   # zone 2: f1=0.5, f2=0.3

    def test_contribution_matrix_without_table_is_ones(self):
        m = self._problem().contribution_matrix()
        np.testing.assert_array_equal(m[1:], np.ones((2, 2)))
        np.testing.assert_array_equal(m[0], np.zeros(2))

    def test_zone_target_weight_default_is_ones(self):
        _, _, zcontrib, _, _ = _make_zone_data()
        zp = self._problem(zone_contributions=zcontrib)
        assert zp.zone_target_contrib() == 0
        w = zp.zone_target_weight_matrix()
        np.testing.assert_array_equal(w[1:], np.ones((2, 2)))
        np.testing.assert_array_equal(w[0], np.zeros(2))

    def test_zone_target_weight_is_contribution_when_flag_set(self):
        _, _, zcontrib, _, _ = _make_zone_data()
        zp = self._problem(
            zone_contributions=zcontrib, parameters={"ZONETARGETCONTRIB": 1},
        )
        assert zp.zone_target_contrib() == 1
        np.testing.assert_array_equal(
            zp.zone_target_weight_matrix(), zp.contribution_matrix(),
        )

    @pytest.mark.parametrize("bad", [2, "1", 1.5, -1])
    def test_zone_target_contrib_rejects_out_of_domain(self, bad):
        zp = self._problem(parameters={"ZONETARGETCONTRIB": bad})
        with pytest.raises(ValueError, match="ZONETARGETCONTRIB"):
            zp.zone_target_contrib()
        errors = zp.validate()
        assert any("ZONETARGETCONTRIB" in e for e in errors)

    def test_validate_reports_target2_in_zone_problem(self):
        pu, feat, puvspr = _make_base_data()
        feat = feat.copy()
        feat["target2"] = [5.0, 0.0]
        zp = self._problem(features=feat)
        assert any("target2" in e for e in zp.validate())

    def test_validate_reports_unknown_contribution_pair(self):
        bad = pd.DataFrame({
            "feature": [1, 9], "zone": [7, 1], "contribution": [1.0, 1.0],
        })
        zp = self._problem(zone_contributions=bad)
        msgs = [e for e in zp.validate() if "unknown" in e]
        assert len(msgs) == 1
        assert "feature 1, zone 7" in msgs[0] and "feature 9, zone 1" in msgs[0]
        assert "2 unknown" in msgs[0]

    def test_contribution_gaps_lists_unlisted_pairs(self):
        partial = pd.DataFrame({
            "feature": [1], "zone": [1], "contribution": [1.0],
        })
        zp = self._problem(zone_contributions=partial)
        assert zp.contribution_gaps() == [(1, 2), (2, 1), (2, 2)]
        assert zp.validate() == []                  # advisory, not an error (ruling R3)
        _, _, zcontrib, _, _ = _make_zone_data()
        zp_full = self._problem(zone_contributions=zcontrib)
        assert zp_full.contribution_gaps() == []
        zp_none = self._problem()
        assert zp_none.contribution_gaps() == []

    def test_validate_clean_when_all_pairs_listed(self):
        zones, zc, zcontrib, zt, zbc = _make_zone_data()
        zp = self._problem(
            zone_contributions=zcontrib, zone_targets=zt, zone_boundary_costs=zbc,
        )
        assert zp.validate() == []
