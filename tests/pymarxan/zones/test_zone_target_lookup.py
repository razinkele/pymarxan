"""zone_target_lookup / zone_target_issues / validate() / ZoneTargetWarning (#7, spec §1–2)."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from pymarxan.zones import ZoneTargetWarning
from pymarxan.zones.model import warn_zone_target_issues
from tests.pymarxan.zones.marzone_anchor import make_anchor_problem


def _with_targets(rows: list[tuple[object, object, float]]):
    p = make_anchor_problem()
    zt = pd.DataFrame(rows, columns=["zone", "feature", "target"])
    return p.copy_with(zone_targets=zt)


class TestLookup:
    def test_anchor(self):
        assert make_anchor_problem().zone_target_lookup() == {(2, 1): 10.0}

    def test_last_row_wins_both_orders(self):
        assert _with_targets([(2, 1, 25.0), (2, 1, 10.0)]).zone_target_lookup() == {(2, 1): 10.0}
        assert _with_targets([(2, 1, 10.0), (2, 1, 25.0)]).zone_target_lookup() == {(2, 1): 25.0}

    def test_unknown_ids_skipped(self):
        p = _with_targets([(9, 1, 10.0), (2, 99, 1.0), (2, 1, 10.0)])
        assert p.zone_target_lookup() == {(2, 1): 10.0}

    def test_positive_then_zero_switches_pair_off(self):
        p = _with_targets([(2, 1, 10.0), (2, 1, 0.0)])
        assert p.zone_target_lookup() == {}
        assert p.zone_target_lookup(include_inert=True) == {(2, 1): 0.0}

    def test_zero_then_positive_keeps_positive(self):
        assert _with_targets([(2, 1, 0.0), (2, 1, 10.0)]).zone_target_lookup() == {(2, 1): 10.0}

    def test_negative_is_inert(self):
        p = _with_targets([(1, 1, -5.0)])
        assert p.zone_target_lookup() == {}
        assert p.zone_target_lookup(include_inert=True) == {(1, 1): -5.0}

    def test_include_inert_never_includes_unknown_ids(self):
        p = _with_targets([(9, 1, 0.0), (2, 1, 0.0)])
        assert p.zone_target_lookup(include_inert=True) == {(2, 1): 0.0}

    def test_insertion_order_is_first_appearance(self):
        p = _with_targets([(2, 1, 1.0), (1, 1, 2.0), (2, 1, 3.0)])
        assert list(p.zone_target_lookup().items()) == [((2, 1), 3.0), ((1, 1), 2.0)]

    def test_odd_ids(self):
        p = _with_targets([(2.0, 1.0, 10.0), (np.nan, 1, 5.0), (1, np.nan, 5.0),
                           ("x", 1, 5.0), (2.5, 1, 5.0)])
        assert p.zone_target_lookup() == {(2, 1): 10.0}
        unknown, dups = p.zone_target_issues()
        assert [pos for pos, _, _ in unknown] == [1, 2, 3, 4]
        assert dups == []

    def test_none_and_missing_columns(self):
        p = make_anchor_problem()
        assert p.copy_with(zone_targets=None).zone_target_lookup() == {}
        bad = pd.DataFrame({"zone": [2], "feature": [1]})
        assert p.copy_with(zone_targets=bad).zone_target_lookup() == {}
        assert p.copy_with(zone_targets=bad).zone_target_issues() == ([], [])


class TestIssues:
    def test_clean_anchor(self):
        assert make_anchor_problem().zone_target_issues() == ([], [])

    def test_unknown_and_duplicates(self):
        p = _with_targets([(9, 1, 10.0), (2, 1, 25.0), (2, 99, 1.0), (2, 1, 10.0), (2, 1, 3.0)])
        unknown, dups = p.zone_target_issues()
        assert [(pos, z, f) for pos, z, f in unknown] == [(0, 9, 1), (2, 2, 99)]
        assert dups == [(2, 1)]


class TestValidate:
    def test_anchor_has_no_errors(self):
        assert make_anchor_problem().validate() == []

    def test_unknown_ids_reported(self):
        errors = _with_targets([(9, 1, 10.0), (2, 1, 10.0)]).validate()
        assert any("zone_targets" in e and "unknown" in e and "zone 9" in e for e in errors)

    def test_duplicates_reported(self):
        errors = _with_targets([(2, 1, 25.0), (2, 1, 10.0)]).validate()
        assert any("duplicate" in e and "last-wins" in e and "zone 2, feature 1" in e
                   for e in errors)

    def test_validate_never_raises_on_odd_ids(self):
        errors = _with_targets([("x", 1, 5.0), (2.5, 1, 5.0)]).validate()
        assert any("zone x" in e and "zone 2.5" in e for e in errors)

    def test_duplicate_contribution_rows_reported(self):
        p = make_anchor_problem()
        zc = pd.DataFrame({"feature": [1, 1, 1], "zone": [1, 2, 2],
                           "contribution": [1.0, 0.4, 0.5]})
        errors = p.copy_with(zone_contributions=zc).validate()
        assert any("zone_contributions" in e and "duplicate" in e and "feature 1, zone 2" in e
                   for e in errors)


class TestWarning:
    def test_silent_on_clean(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            warn_zone_target_issues(make_anchor_problem(), "ZoneSASolver")

    def test_one_warning_with_counts(self):
        p = _with_targets([(9, 1, 10.0), (2, 1, 25.0), (2, 1, 10.0)])
        with pytest.warns(ZoneTargetWarning) as rec:
            warn_zone_target_issues(p, "ZoneSASolver")
        assert len(rec) == 1
        msg = str(rec[0].message)
        assert msg.startswith("ZoneSASolver: zone_targets has")
        assert "1 row(s) with unknown" in msg and "1 duplicate" in msg

    def test_is_user_warning_subclass(self):
        assert issubclass(ZoneTargetWarning, UserWarning)


class TestValidateNonFinite:
    def test_clean_anchor(self):
        assert make_anchor_problem().validate() == []

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
    def test_non_finite_reported(self, bad):
        errs = _with_targets([(1, 1, bad), (2, 1, 10.0)]).validate()
        assert any("non-finite target" in e and "zone 1, feature 1" in e for e in errs)

    def test_non_numeric_never_raises(self):
        errs = _with_targets([(1, 1, "abc"), (2, 1, 10.0)]).validate()
        assert any("non-finite target" in e for e in errs)
