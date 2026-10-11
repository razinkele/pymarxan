"""PU zone rules: pulock / puzone / status precedence (#6 spec §2)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from pymarxan.zones.model import PUZoneRule, fixed_zones, rule_violations
from tests.pymarxan.zones.marzone_anchor import make_anchor_problem

ALL3 = frozenset({1, 2, 3})


def _table(rows):
    return pd.DataFrame(rows, columns=["pu", "zone"])


def _three_zone(statuses=(0, 0, 0), **tables):
    p = make_anchor_problem()
    zones = pd.DataFrame({"id": [1, 2, 3], "name": ["protect", "restore", "use"]})
    zc = pd.concat(
        [p.zone_costs, pd.DataFrame({"pu": [1, 2, 3], "zone": [3, 3, 3], "cost": [2.0] * 3})],
        ignore_index=True,
    )
    p = p.copy_with(
        zones=zones, zone_costs=zc, planning_units=p.planning_units.assign(status=list(statuses)),
    )
    return p.copy_with(**tables)


def _rule(allowed, may, start):
    return PUZoneRule(allowed=frozenset(allowed), may_be_unassigned=may, start=start)


def test_free_and_status_rows():
    rules = _three_zone((1, 2, 3)).pu_zone_rules()
    assert rules == [_rule(ALL3, True, 1), _rule({1}, False, 1), _rule(set(), True, 0)]


def test_lock_and_last_row_wins():
    p = _three_zone(pu_locks=_table([(1, 3), (1, 2)]))
    assert p.pu_lock_lookup() == {1: 2}
    assert p.pu_zone_rules()[0] == _rule({2}, False, 2)


def test_last_row_wins_before_unknown_zone_filter():
    p = _three_zone(pu_locks=_table([(1, 2), (1, 99)]))
    assert p.pu_lock_lookup() == {}
    assert p.pu_zone_rules()[0] == _rule(ALL3, True, None)


def test_unknown_pu_and_odd_ids_skipped():
    p = _three_zone(pu_locks=_table([(9, 2), (np.nan, 2), (2.0, 3.0), (3, 2.5)]))
    assert p.pu_lock_lookup() == {2: 3}


def test_lock_beats_status_3_and_status_3_beats_puzone():
    p = _three_zone((3, 3, 0), pu_locks=_table([(1, 2)]), pu_zones=_table([(2, 1), (2, 2)]))
    rules = p.pu_zone_rules()
    assert rules[0] == _rule({2}, False, 2)
    assert rules[1] == _rule(set(), True, 0)


def test_status_2_within_puzone():
    p = _three_zone((2, 2, 2), pu_zones=_table([(1, 2), (1, 3), (2, 3)]))
    rules = p.pu_zone_rules()
    assert rules[0] == _rule({2, 3}, False, None)
    assert rules[1] == _rule({3}, False, 3)
    assert rules[2] == _rule({1}, False, 1)


def test_puzone_rows():
    p = _three_zone(pu_zones=_table([(1, 2), (1, 3), (1, 3), (2, 99), (2, 98), (3, 1), (3, 99)]))
    assert p.pu_zone_lookup() == {1: frozenset({2, 3}), 3: frozenset({1})}
    rules = p.pu_zone_rules()
    assert rules[0] == _rule({2, 3}, False, None)
    assert rules[1] == _rule(ALL3, True, None)     # all-unknown puzone -> free
    assert rules[2] == _rule({1}, False, 1)        # partly unknown -> single zone -> lock


def test_missing_columns_mean_no_table():
    p = _three_zone(pu_locks=pd.DataFrame({"pu": [1]}), pu_zones=pd.DataFrame({"zone": [1]}))
    assert p.pu_lock_lookup() == {} and p.pu_zone_lookup() == {}


def test_non_contiguous_zone_ids():
    p = make_anchor_problem()
    relabel = {1: 5, 2: 9}
    p = p.copy_with(
        zones=p.zones.assign(id=p.zones["id"].map(relabel)),
        zone_costs=p.zone_costs.assign(zone=p.zone_costs["zone"].map(relabel)),
        zone_contributions=p.zone_contributions.assign(
            zone=p.zone_contributions["zone"].map(relabel),
        ),
        zone_targets=p.zone_targets.assign(zone=p.zone_targets["zone"].map(relabel)),
        planning_units=p.planning_units.assign(status=[2, 1, 0]),
        pu_zones=_table([(3, 9)]),
    )
    assert p.pu_zone_rules() == [
        _rule({5}, False, 5), _rule({5, 9}, True, 5), _rule({9}, False, 9),
    ]


def test_precedence_pairs():
    p = _three_zone(
        (1, 2, 3),
        pu_locks=_table([(2, 3)]),
        pu_zones=_table([(1, 2), (1, 3), (2, 1), (2, 2), (3, 2)]),
    )
    assert p.pu_zone_rules() == [
        _rule({2, 3}, False, None),   # status 1 + puzone -> puzone (start dropped)
        _rule({3}, False, 3),         # status 2 + puzone + lock -> lock
        _rule(set(), True, 0),        # status 3 + puzone -> status 3
    ]


def test_status_2_within_puzone_containing_first_zone():
    # spec §2: status 2 + puzone -> the puzone set, NOT narrowed to min(zone_ids)
    p = _three_zone((2, 0, 0), pu_zones=_table([(1, 1), (1, 2)]))
    assert p.pu_zone_rules()[0] == _rule({1, 2}, False, None)


def test_status_1_start_dropped_even_when_first_zone_allowed():
    p = _three_zone((1, 0, 0), pu_zones=_table([(1, 1), (1, 2)]))
    assert p.pu_zone_rules()[0] == _rule({1, 2}, False, None)


def test_invariant_no_unsatisfiable_rule():
    p = _three_zone((3, 2, 1), pu_locks=_table([(1, 99)]), pu_zones=_table([(2, 99), (3, 98)]))
    assert all(r.allowed or r.may_be_unassigned for r in p.pu_zone_rules())


def test_forced_property():
    assert _rule({2}, False, 2).forced
    assert not _rule({2}, True, None).forced
    assert not _rule({2, 3}, False, None).forced
    assert not _rule(set(), True, 0).forced


def test_rule_violations_and_fixed_zones():
    rules = [_rule({2}, False, 2), _rule(ALL3, True, None), _rule({2, 3}, False, None),
             _rule(set(), True, 0)]
    assert rule_violations(rules, np.array([2, 0, 3, 0])) == []
    assert rule_violations(rules, np.array([1, 0, 0, 2])) == [0, 2, 3]
    assert fixed_zones(rules) == {0: 2, 3: 0}
    assert fixed_zones([_rule({2}, True, None)]) == {}  # single-zone FREE rule is not fixed


def test_copy_with_carries_tables():
    p = _three_zone(pu_locks=_table([(1, 2)]))
    q = p.copy_with(planning_units=p.planning_units.copy())
    assert q.pu_locks is not None and q.pu_lock_lookup() == {1: 2}
