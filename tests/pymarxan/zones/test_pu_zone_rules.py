"""PU zone rules: pulock / puzone / status precedence (#6 spec §2)."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from pymarxan.solvers.base import SolverConfig
from pymarxan.zones import (
    ZoneHeuristicSolver,
    ZoneIISolver,
    ZoneInputWarning,
    ZoneLockWarning,
    ZoneMIPSolver,
    ZoneSASolver,
    ZoneTargetWarning,
)
from pymarxan.zones.model import PUZoneRule, fixed_zones, rule_violations, warn_zone_target_issues
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


def _issues(**kw):
    return _three_zone(**kw).pu_lock_issues()


def test_float_ids_render_as_ints_in_issue_messages():
    p = _three_zone(pu_locks=pd.DataFrame({"pu": [9.0], "zone": [1.0]}))
    text = " | ".join(p.pu_lock_issues())
    assert "PU 9, zone 1" in text and "9.0" not in text


def test_issue_rows_and_validate():
    p = _three_zone(
        (3, 3, 0),
        pu_locks=_table([(1, 2), (9, 1), (2, 1), (2, 3)]),
        pu_zones=_table([(1, 3), (3, 2), (3, 2), (2, 99)]),
    )
    issues = p.pu_lock_issues()
    text = " | ".join(issues)
    assert "unknown PU or zone" in text and "PU 9, zone 1" in text
    assert "repeats 1 PU(s); the last row wins" in text and "PU 2 → zone 3" in text
    assert "duplicate (pu, zone)" in text and "PU 3, zone 2" in text
    assert "single zone (treated as a lock" in text and "PU 3 → zone 2" in text
    assert "status 3 and a pu_locks row (the lock wins)" in text
    assert "outside the PU's pu_zones list (the lock wins)" in text and "PU 1 → zone 2" in text
    errors = p.validate()
    assert any("repeats 1 PU(s)" in e for e in errors)
    assert not any("outside the PU's pu_zones list" in e for e in errors)  # warning only


def test_all_unknown_puzone_reported():
    assert any("name only unknown zones (PU unrestricted)" in m
               for m in _issues(pu_zones=_table([(1, 98), (1, 99)])))


def test_status_3_with_puzone_reported():
    assert any("status 3 and pu_zones rows (status 3 wins)" in m
               for m in _issues(statuses=(3, 0, 0), pu_zones=_table([(1, 2), (1, 3)])))


def test_all_unknown_puzone_not_reported_for_locked_or_status_3_pu():
    issues = _issues(statuses=(3, 0, 0), pu_locks=_table([(2, 1)]),
                     pu_zones=_table([(1, 99), (2, 99)]))
    assert not any("name only unknown zones" in m for m in issues)


def test_single_zone_puzone_not_reported_for_locked_pu():
    issues = _issues(pu_locks=_table([(1, 3)]), pu_zones=_table([(1, 2)]))
    assert not any("single zone" in m for m in issues)
    assert any("outside the PU's pu_zones list" in m for m in issues)


def test_status_3_with_lock_and_puzone_reports_lock_only():
    issues = _issues(statuses=(3, 0, 0), pu_locks=_table([(1, 2)]),
                     pu_zones=_table([(1, 2), (1, 3)]))
    assert any("status 3 and a pu_locks row" in m for m in issues)
    assert not any("status 3 and pu_zones rows" in m for m in issues)


def test_missing_columns_reported():
    assert any("pu_locks missing columns ['zone'] (table ignored)" in m
               for m in _issues(pu_locks=pd.DataFrame({"pu": [1]})))


ERROR_ROWS = [
    ("unknown PU or zone", dict(pu_locks=_table([(9, 1)]))),
    ("repeats 1 PU(s)", dict(pu_locks=_table([(1, 1), (1, 2)]))),
    ("duplicate (pu, zone)", dict(pu_zones=_table([(1, 1), (1, 1), (1, 2)]))),
    ("name only unknown zones", dict(pu_zones=_table([(1, 99)]))),
    ("single zone (treated as a lock", dict(pu_zones=_table([(1, 2)]))),
    ("status 3 and a pu_locks row", dict(statuses=(3, 0, 0), pu_locks=_table([(1, 2)]))),
    ("status 3 and pu_zones rows",
     dict(statuses=(3, 0, 0), pu_zones=_table([(1, 1), (1, 2)]))),
    ("missing columns", dict(pu_locks=pd.DataFrame({"pu": [1]}))),
]


@pytest.mark.parametrize("needle,kw", ERROR_ROWS, ids=[r[0] for r in ERROR_ROWS])
def test_each_error_row_reaches_validate(needle, kw):
    kw = dict(kw)
    statuses = kw.pop("statuses", (0, 0, 0))
    assert any(needle in e for e in _three_zone(statuses, **kw).validate())


def test_repeated_lock_with_unknown_last_zone_says_not_locked():
    assert any("PU 1 → zone 99 (unknown zone, PU not locked)" in m
               for m in _issues(pu_locks=_table([(1, 2), (1, 99)])))


def test_repeated_known_lock_has_no_not_locked_suffix():
    text = " | ".join(_issues(pu_locks=_table([(1, 1), (1, 2)])))
    assert "PU 1 → zone 2" in text and "PU not locked" not in text


def test_triplicate_puzone_row_counted_once():
    issues = _issues(pu_zones=_table([(1, 2), (1, 2), (1, 2), (1, 3)]))
    assert any("has 1 duplicate (pu, zone) row(s)" in m for m in issues)


def test_status_3_single_zone_puzone_reported_only_as_status_3():
    issues = _issues(statuses=(3, 0, 0), pu_zones=_table([(1, 2)]))
    assert any("status 3 and pu_zones rows" in m for m in issues)
    assert not any("single zone" in m for m in issues)


def test_no_tables_no_scan():
    assert _three_zone((1, 2, 3)).pu_lock_issues() == []


def test_clean_problem_has_no_issues():
    p = _three_zone((2, 1, 0), pu_locks=_table([(1, 2)]), pu_zones=_table([(2, 1), (2, 2)]))
    assert p.pu_lock_issues() == []
    assert p.validate() == []


def test_warning_hierarchy():
    assert issubclass(ZoneTargetWarning, ZoneInputWarning)
    assert issubclass(ZoneLockWarning, ZoneInputWarning)
    assert not issubclass(ZoneLockWarning, ZoneTargetWarning)


SOLVERS = [ZoneMIPSolver, ZoneSASolver, ZoneHeuristicSolver, ZoneIISolver]


@pytest.mark.parametrize("cls", SOLVERS, ids=lambda c: c.__name__)
def test_each_solver_warns_once_about_locks(cls):
    # two issue classes (unknown PU, repeated PU): still ONE ZoneLockWarning, joined by " | "
    p = _three_zone(pu_locks=_table([(1, 2), (9, 1), (2, 1), (2, 3)]))
    p = p.copy_with(parameters={**p.parameters, "NUMITNS": 500})
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        cls().solve(p, SolverConfig(num_solutions=2, seed=1))
    hits = [w for w in rec if issubclass(w.category, ZoneLockWarning)]
    assert len(hits) == 1
    assert str(hits[0].message).startswith(f"{cls.__name__}: PU zone restrictions:")
    assert "PU 9, zone 1" in str(hits[0].message) and "the last row wins" in str(hits[0].message)


def test_ii_improve_warns_once_about_locks():
    p = _three_zone(pu_locks=_table([(1, 2), (9, 1)]))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        (sol,) = ZoneHeuristicSolver().solve(p, SolverConfig(num_solutions=1, seed=1))
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        ZoneIISolver().improve(p, sol)
    assert len([w for w in rec if issubclass(w.category, ZoneLockWarning)]) == 1


def test_target_wrapper_emits_only_target_warning():
    p = _three_zone(pu_locks=_table([(9, 1)])).copy_with(
        zone_targets=pd.DataFrame({"zone": [9], "feature": [1], "target": [1.0]}),
    )
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        warn_zone_target_issues(p, "ZoneSASolver")
    assert [w.category for w in rec] == [ZoneTargetWarning]


def test_lock_warning_points_at_the_solve_caller():
    p = _three_zone(pu_locks=_table([(9, 1)]))
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        ZoneHeuristicSolver().solve(p, SolverConfig(num_solutions=1, seed=1))
    (w,) = [w for w in rec if issubclass(w.category, ZoneLockWarning)]
    assert w.filename == __file__


def test_partly_unknown_puzone_reported_as_single_zone():
    issues = _issues(pu_zones=_table([(1, 1), (1, 99)]))
    assert any("single zone (treated as a lock" in m and "PU 1 → zone 1" in m for m in issues)
    assert any("unknown PU or zone" in m and "PU 1, zone 99" in m for m in issues)


@pytest.mark.parametrize("bad", [np.nan, "x"], ids=["nan", "text"])
def test_validate_never_raises_on_bad_status_with_locks(bad):
    # per-row status parse: "x" raises under astype(int); NaN casts silently (pin: still skipped)
    p = _three_zone((0, bad, 0), pu_locks=_table([(1, 2)]))
    assert isinstance(p.validate(), list)
    assert isinstance(p.pu_lock_issues(), list)


def test_dispatcher_emits_one_of_each_kind_in_one_call():
    # spec §3: at most one warning of EACH kind; Testing 5: ZoneInputWarning catches both
    p = _three_zone(pu_locks=_table([(9, 1)])).copy_with(
        zone_targets=pd.DataFrame({"zone": [9], "feature": [1], "target": [1.0]}),
    )
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        ZoneHeuristicSolver().solve(p, SolverConfig(num_solutions=1, seed=1))
    kinds = sorted(w.category.__name__ for w in rec if issubclass(w.category, ZoneInputWarning))
    assert kinds == ["ZoneLockWarning", "ZoneTargetWarning"]
