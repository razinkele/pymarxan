"""diagnose_zone_infeasibility + ZoneMIPSolver failure warnings (#7, spec §3)."""
from __future__ import annotations

import warnings

import pandas as pd
import pytest

from pymarxan.solvers.base import SolverConfig
from pymarxan.zones import ZoneMIPNoSolutionWarning, ZoneMIPSolver, diagnose_zone_infeasibility
from pymarxan.zones.diagnostics import PUZoneRule, _pu_zone_rules
from tests.pymarxan.zones.marzone_anchor import make_anchor_problem

CONFIG = SolverConfig(num_solutions=1, seed=1)


def _status(p, statuses):
    return p.copy_with(planning_units=p.planning_units.assign(status=statuses))


def _contrib(p, rows):
    return p.copy_with(
        zone_contributions=pd.DataFrame(rows, columns=["feature", "zone", "contribution"]),
    )


def _mip_warning(p) -> str:
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        assert ZoneMIPSolver().solve(p, CONFIG) == []
    msgs = [str(w.message) for w in rec if issubclass(w.category, ZoneMIPNoSolutionWarning)]
    assert len(msgs) == 1, msgs
    return msgs[0]


class TestRules:
    def test_status_convention(self):
        p = _status(make_anchor_problem(), [0, 2, 3])
        assert _pu_zone_rules(p) == [
            PUZoneRule(frozenset({1, 2}), False),
            PUZoneRule(frozenset({1}), True),
            PUZoneRule(frozenset(), False),
        ]

    def test_missing_status_column_means_free(self):
        p = make_anchor_problem()
        p = p.copy_with(planning_units=p.planning_units.drop(columns="status"))
        assert all(r == PUZoneRule(frozenset({1, 2}), False) for r in _pu_zone_rules(p))


class TestDiagnose:
    def test_anchor_reachable(self):
        assert diagnose_zone_infeasibility(make_anchor_problem()) == []

    def test_two_lockouts_make_overall_unreachable(self):
        msgs = diagnose_zone_infeasibility(_status(make_anchor_problem(), [3, 3, 0]))
        assert msgs == ["overall target for feature 1 is 15 but at most 10 is reachable"]

    def test_partial_contribution_table(self):
        p = _contrib(make_anchor_problem(), [(1, 2, 0.4)])  # zone 1 now contributes 0
        assert diagnose_zone_infeasibility(p) == [
            "overall target for feature 1 is 15 but at most 12 is reachable"
        ]

    def test_locked_in_forced_negative_counts(self):
        p = _status(_contrib(make_anchor_problem(), [(1, 1, -0.5), (1, 2, 0.4)]), [2, 0, 0])
        # PU1 forced into zone 1: -5; PUs 2-3 best = max(0, -5, 4) = 4 each -> 3
        assert diagnose_zone_infeasibility(p) == [
            "overall target for feature 1 is 15 but at most 3 is reachable"
        ]

    def test_free_pu_may_stay_unassigned_when_every_zone_is_negative(self):
        p = _contrib(make_anchor_problem(), [(1, 1, -0.5), (1, 2, -0.1)])
        assert diagnose_zone_infeasibility(p) == [
            "overall target for feature 1 is 15 but at most 0 is reachable"
        ]

    def test_zone_target_with_contrib_weight_and_forced_negative(self):
        p = make_anchor_problem(zone_target_contrib=1)
        p = _contrib(p, [(1, 1, -0.5), (1, 2, 0.4)])
        p = _status(p, [2, 0, 0])
        p = p.copy_with(
            zone_targets=pd.DataFrame({"zone": [1], "feature": [1], "target": [1.0]}),
            features=p.features.assign(target=0.0),
        )
        # zone 1 weight -0.5: PU1 forced -5; free PUs max(0, -5) = 0 -> -5
        assert diagnose_zone_infeasibility(p) == [
            "zone target (zone 1, feature 1) is 1 but at most -5 is reachable"
        ]

    def test_zero_targets_never_reported(self):
        p = _status(_contrib(make_anchor_problem(), [(1, 1, -0.5), (1, 2, 0.4)]), [2, 0, 0])
        p = p.copy_with(features=p.features.assign(target=0.0), zone_targets=None)
        assert diagnose_zone_infeasibility(p) == []

    def test_misslevel_scales_the_overall_target(self):
        p = _status(make_anchor_problem(misslevel=0.5), [3, 3, 0])
        assert diagnose_zone_infeasibility(p) == []  # 7.5 <= 10

    def test_misslevel_scales_zone_targets(self):
        p = _status(make_anchor_problem(misslevel=0.5), [3, 0, 0]).copy_with(
            zone_targets=pd.DataFrame({"zone": [2], "feature": [1], "target": [30.0]}),
        )
        assert diagnose_zone_infeasibility(p) == []  # 15 <= 20

    def test_duplicate_puvspr_rows_summed(self):
        p = _status(make_anchor_problem(), [3, 3, 0])
        puv = pd.DataFrame({"species": [1, 1, 1, 1], "pu": [1, 2, 3, 3],
                            "amount": [10.0, 10.0, 5.0, 5.0]})
        assert diagnose_zone_infeasibility(p.copy_with(pu_vs_features=puv)) == [
            "overall target for feature 1 is 15 but at most 10 is reachable"
        ]


class TestMIPWarning:
    def test_names_unreachable_target(self):
        msg = _mip_warning(_status(make_anchor_problem(), [3, 3, 0]))
        assert msg.startswith("ZoneMIPSolver: infeasible. Unreachable targets: ")
        assert "feature 1 is 15 but at most 10" in msg

    def test_compete_wording_when_each_target_reachable(self):
        p = _status(make_anchor_problem(), [3, 0, 0])
        assert diagnose_zone_infeasibility(p) == []
        msg = _mip_warning(p)
        assert "every target is reachable on its own" in msg
        assert "ZoneSASolver" in msg

    def test_ghost_feature_named_in_mip_warning(self):
        p = make_anchor_problem()
        ghost = pd.DataFrame({"id": [2], "name": ["ghost"], "target": [5.0], "spf": [1.0]})
        p = p.copy_with(features=pd.concat([p.features, ghost], ignore_index=True))
        msg = _mip_warning(p)
        assert "overall target for feature 2 is 5 but at most 0 is reachable" in msg

    def test_diagnostic_crash_still_returns_empty(self, monkeypatch):
        import pymarxan.zones.mip_solver as ms

        def boom(problem):
            raise RuntimeError("boom")

        monkeypatch.setattr(ms, "diagnose_zone_infeasibility", boom)
        msg = _mip_warning(_status(make_anchor_problem(), [3, 3, 0]))
        assert "diagnostic failed" in msg
        assert "reachable on its own" not in msg

    def test_feasible_solve_emits_no_failure_warning(self):
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            sols = ZoneMIPSolver().solve(make_anchor_problem(), CONFIG)
        assert sols and sols[0].objective == pytest.approx(7.0)
        assert not [w for w in rec if issubclass(w.category, ZoneMIPNoSolutionWarning)]


class TestWarnNoSolutionBranches:
    def test_not_solved_mentions_time_limit(self):
        import pulp

        from pymarxan.zones.mip_solver import _warn_no_solution

        with pytest.warns(ZoneMIPNoSolutionWarning) as rec:
            _warn_no_solution(make_anchor_problem(), pulp.constants.LpStatusNotSolved)
        msg = str(rec[0].message)
        assert "time limit" in msg and "MIP_TIME_LIMIT" in msg

    def test_other_status_generic(self):
        import pulp

        from pymarxan.zones.mip_solver import _warn_no_solution

        with pytest.warns(ZoneMIPNoSolutionWarning) as rec:
            _warn_no_solution(make_anchor_problem(), pulp.constants.LpStatusUndefined)
        assert str(rec[0].message).startswith("ZoneMIPSolver: no solution (status")

    def test_infeasible_more_than_five_truncated(self, monkeypatch):
        import pulp

        import pymarxan.zones.mip_solver as ms

        monkeypatch.setattr(
            ms, "diagnose_zone_infeasibility", lambda p: [f"u{i}" for i in range(7)],
        )
        with pytest.warns(ZoneMIPNoSolutionWarning) as rec:
            ms._warn_no_solution(make_anchor_problem(), pulp.constants.LpStatusInfeasible)
        assert "(+2 more)" in str(rec[0].message)
