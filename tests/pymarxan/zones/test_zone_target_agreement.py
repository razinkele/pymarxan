"""All zone components agree on bad zone_targets rows (#7, spec §1 + cache addendum).

The dup_* DataFrame/cache cases and the cache delta test are pins (those paths were already
last-wins / consistent); the RED cases are unknown ids in the objective, the writer, the
negative-weight cache mask and the public name.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from pymarxan.solvers.base import SolverConfig
from pymarxan.solvers.run_mode import RunModePipeline
from pymarxan.zones import (
    ZoneHeuristicSolver,
    ZoneIISolver,
    ZoneMIPSolver,
    ZoneSASolver,
    ZoneTargetWarning,
    compute_zone_achieved,
)
from pymarxan.zones.cache import ZoneProblemCache
from pymarxan.zones.objective import (
    _compute_zone_achieved,
    build_zone_solution,
    check_zone_targets,
    compute_zone_objective,
    compute_zone_shortfalls,
)
from pymarxan.zones.writers import write_zone_summary
from tests.pymarxan.zones.marzone_anchor import OPTIMUM, make_anchor_problem

ASSIGN = np.array(OPTIMUM)
VARIANTS = {
    "dup_25_then_10": [(2, 1, 25.0), (2, 1, 10.0)],
    "dup_10_then_25": [(2, 1, 10.0), (2, 1, 25.0)],
    "dup_zone1_10_then_20": [(1, 1, 10.0), (1, 1, 20.0)],
    "unknown_zone": [(9, 1, 10.0), (2, 1, 10.0)],
    "unknown_feature": [(2, 99, 1.0), (2, 1, 10.0)],
    "all_three": [(9, 1, 10.0), (2, 99, 1.0), (2, 1, 25.0), (2, 1, 10.0)],
}
NEG_CONTRIB = pd.DataFrame({"feature": [1, 1], "zone": [1, 2], "contribution": [-0.5, 0.4]})


def _variant(rows, **kw):
    p = make_anchor_problem(**kw)
    return p.copy_with(zone_targets=pd.DataFrame(rows, columns=["zone", "feature", "target"]))


def _hand_normalized(p):
    lookup = p.zone_target_lookup()
    zt = pd.DataFrame(
        [(z, f, t) for (z, f), t in lookup.items()], columns=["zone", "feature", "target"],
    )
    return p.copy_with(zone_targets=zt if len(zt) else None)


def _cache_objective(p, assignment, blm=0.0):
    cache = ZoneProblemCache.from_zone_problem(p)
    return cache.compute_full_zone_objective(
        assignment, held=cache.compute_held(assignment), blm=blm,
    )


@pytest.mark.parametrize("name", sorted(VARIANTS))
def test_dataframe_and_cache_agree_with_hand_normalized(name):
    p = _variant(VARIANTS[name], spf=10.0)
    expected = compute_zone_objective(_hand_normalized(p), ASSIGN, 0.0)
    assert compute_zone_objective(p, ASSIGN, 0.0) == pytest.approx(expected)
    assert _cache_objective(p, ASSIGN) == pytest.approx(expected)


def test_unknown_zone_no_phantom_penalty():
    # Before #7: 17.0 (spf 1) — the unknown zone 9 row added a 10.0 shortfall.
    assert compute_zone_objective(_variant(VARIANTS["unknown_zone"]), ASSIGN, 0.0) == 7.0


def test_met_flags_and_shortfalls_keyed_by_lookup():
    p = _variant(VARIANTS["all_three"])
    assert set(check_zone_targets(p, ASSIGN)) == {(2, 1)}
    assert set(compute_zone_shortfalls(p, ASSIGN)) == {(2, 1)}
    sol = build_zone_solution(p, ASSIGN, 0.0, solver_name="t")
    assert set(sol.metadata["zone_targets_met"]) == {"z2_f1"}


def test_all_rows_inert_add_no_penalty():
    p = _variant([(9, 1, 10.0), (2, 1, 10.0), (2, 1, 0.0)])
    none = make_anchor_problem().copy_with(zone_targets=None)
    # Known inert pair stays in the met-flags, as met (check_overall_targets convention).
    assert check_zone_targets(p, ASSIGN) == {(2, 1): True}
    assert compute_zone_shortfalls(p, ASSIGN) == {}
    assert compute_zone_objective(p, ASSIGN, 0.0) == compute_zone_objective(none, ASSIGN, 0.0)
    assert _cache_objective(p, ASSIGN) == pytest.approx(_cache_objective(none, ASSIGN))


def test_inert_pair_met_even_when_weighted_achievement_is_negative():
    p = make_anchor_problem(zone_target_contrib=1).copy_with(
        zone_contributions=NEG_CONTRIB,
        zone_targets=pd.DataFrame({"zone": [1], "feature": [1], "target": [0.0]}),
    )
    assert check_zone_targets(p, ASSIGN) == {(1, 1): True}  # achieved -5, target inert


def test_cache_no_phantom_penalty_for_untargeted_pair_with_negative_weight():
    p = make_anchor_problem(zone_target_contrib=1).copy_with(zone_contributions=NEG_CONTRIB)
    df = compute_zone_objective(p, ASSIGN, 0.0)
    assert df == 21.0  # cost 7 + overall (15 - (-5 + 8)) = 12 + zone 2 (10 - 8) = 2
    assert _cache_objective(p, ASSIGN) == pytest.approx(df)  # was 26.0


def test_cache_mask_gate():
    assert ZoneProblemCache.from_zone_problem(make_anchor_problem()).zone_target_active is None
    pos = make_anchor_problem(zone_target_contrib=1)
    assert ZoneProblemCache.from_zone_problem(pos).zone_target_active is None
    neg = pos.copy_with(zone_contributions=NEG_CONTRIB)
    active = ZoneProblemCache.from_zone_problem(neg).zone_target_active
    assert active is not None
    assert not active[1, 0] and active[2, 0]  # zone 1 untargeted, zone 2 targeted


def test_cache_delta_matches_full_with_negative_weight():
    p = make_anchor_problem(zone_target_contrib=1).copy_with(zone_contributions=NEG_CONTRIB)
    cache = ZoneProblemCache.from_zone_problem(p)
    a = ASSIGN.copy()
    held = cache.compute_held(a)
    before = cache.compute_full_zone_objective(a, held=held, blm=0.0)
    delta = cache.compute_delta_zone_objective(0, int(a[0]), 2, a, held=held, blm=0.0)
    b = a.copy()
    b[0] = 2
    after = cache.compute_full_zone_objective(b, held=cache.compute_held(b), blm=0.0)
    assert before + delta == pytest.approx(after)


def test_cache_matches_objective_when_misslevel_nonpositive_and_negative_weight():
    p = make_anchor_problem(zone_target_contrib=1, misslevel=0.0).copy_with(
        zone_contributions=NEG_CONTRIB,
        zone_targets=pd.DataFrame({"zone": [1], "feature": [1], "target": [10.0]}),
    )
    a = np.array([1, 2, 2])
    df = compute_zone_objective(p, a, 0.0)
    assert df == 12.0
    assert _cache_objective(p, a) == pytest.approx(df)  # was 7.0
    cache = ZoneProblemCache.from_zone_problem(p)
    held = cache.compute_held(a)
    before = cache.compute_full_zone_objective(a, held=held, blm=0.0)
    delta = cache.compute_delta_zone_objective(0, int(a[0]), 2, a, held=held, blm=0.0)
    b = a.copy()
    b[0] = 2
    after = cache.compute_full_zone_objective(b, held=cache.compute_held(b), blm=0.0)
    assert before + delta == pytest.approx(after)


def test_writer_one_row_per_known_pair(tmp_path):
    p = _variant(VARIANTS["all_three"] + [(1, 1, 0.0)])
    sol = build_zone_solution(p, ASSIGN, 0.0, solver_name="t")
    out = tmp_path / "summary.csv"
    write_zone_summary(p, [sol], out)
    zone_rows = pd.read_csv(out).query("tier == 'zone'")
    assert list(zip(zone_rows["zone"], zone_rows["feature"], zone_rows["target"],
                    zone_rows["times_met"], strict=True)) == [(2, 1, 10.0, 1), (1, 1, 0.0, 1)]


def test_public_compute_zone_achieved():
    assert compute_zone_achieved is _compute_zone_achieved
    assert compute_zone_achieved(make_anchor_problem(), ASSIGN) == {(1, 1): 10.0, (2, 1): 20.0}


def test_no_warnings_from_objective_helpers():
    p = _variant(VARIANTS["all_three"])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        compute_zone_objective(p, ASSIGN, 0.0)
        _cache_objective(p, ASSIGN)


CONFIG = SolverConfig(num_solutions=1, seed=1)
SOLVERS = [ZoneMIPSolver, ZoneSASolver, ZoneHeuristicSolver, ZoneIISolver]
STAGES = {1: 1, 2: 2, 5: 3}  # zone-solver calls per run (run_mode.py docstring)


def _fast(p):
    return p.copy_with(parameters={**p.parameters, "NUMITNS": 1000})


def _mip(p):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return ZoneMIPSolver().solve(p, CONFIG)


def _zone_target_warnings(rec):
    return [w for w in rec if issubclass(w.category, ZoneTargetWarning)]


@pytest.mark.parametrize("name", sorted(VARIANTS))
def test_mip_matches_hand_normalized(name):
    p = _variant(VARIANTS[name])
    got, want = _mip(p), _mip(_hand_normalized(p))
    assert [list(s.zone_assignment) for s in got] == [list(s.zone_assignment) for s in want]
    assert [s.objective for s in got] == pytest.approx([s.objective for s in want])


def test_mip_duplicate_rows_no_longer_raise():
    sols = _mip(_variant(VARIANTS["dup_25_then_10"]))
    assert len(sols) == 1 and sols[0].objective == pytest.approx(7.0)


def test_mip_last_wins_is_discriminating():
    # Last row (zone-1 target 20) needs two PUs in zone 1: (1, 1, 0) at 11.0.
    # First-wins (10) would give the anchor optimum (1, 2, 2) at 7.0.
    sols = _mip(_variant(VARIANTS["dup_zone1_10_then_20"]))
    assert [list(s.zone_assignment) for s in sols] == [[1, 1, 0]]
    assert sols[0].objective == pytest.approx(11.0)


def test_all_rows_inert_mip_matches_none():
    p = _variant([(9, 1, 10.0), (2, 1, 10.0), (2, 1, 0.0)])
    none = make_anchor_problem().copy_with(zone_targets=None)
    got, want = _mip(p), _mip(none)
    assert [list(s.zone_assignment) for s in got] == [list(s.zone_assignment) for s in want]
    assert [s.objective for s in got] == pytest.approx([s.objective for s in want])


@pytest.mark.parametrize("cls", SOLVERS, ids=lambda c: c.__name__)
def test_each_solver_warns_exactly_once(cls):
    p = _fast(_variant(VARIANTS["all_three"], spf=10.0))
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        cls().solve(p, SolverConfig(num_solutions=2, seed=1))
    hits = _zone_target_warnings(rec)
    assert len(hits) == 1
    assert str(hits[0].message).startswith(f"{cls.__name__}: zone_targets has")


@pytest.mark.parametrize("cls", SOLVERS, ids=lambda c: c.__name__)
def test_each_solver_silent_on_clean_anchor(cls):
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        cls().solve(_fast(make_anchor_problem(spf=10.0)), CONFIG)
    assert not _zone_target_warnings(rec)


def test_ii_improve_warns_once():
    p = _fast(_variant(VARIANTS["all_three"], spf=10.0))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sol = ZoneHeuristicSolver().solve(p, CONFIG)[0]
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        ZoneIISolver().improve(p, sol)
    assert len(_zone_target_warnings(rec)) == 1


@pytest.mark.parametrize("runmode", sorted(STAGES))
def test_pipeline_warns_once_per_solver_call(runmode):
    p = _fast(_variant(VARIANTS["all_three"], spf=10.0))
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        sols = RunModePipeline(runmode=runmode).solve(p, SolverConfig(num_solutions=2, seed=1))
    assert len(sols) == 2
    assert len(_zone_target_warnings(rec)) == 2 * STAGES[runmode]
