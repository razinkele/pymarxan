"""Snyder–Haight–ReVelle two-stage stochastic reserve-selection MIP."""
from __future__ import annotations

import itertools

import pytest

from pymarxan.temporal.two_stage import two_stage_reserve_mip


def _problem(budget1=1.0, budget2=1.0):
    """3 sites, 3 features, 2 scenarios.

    contains: f1∈{s0}, f2∈{s1}, f3∈{s2}. Costs all 1. Scenario A (p .5): all
    sites available in stage 2; Scenario B (p .5): only s0,s1 available (s2 lost).
    """
    contains = {1: {0}, 2: {1}, 3: {2}}
    cost = {0: 1.0, 1: 1.0, 2: 1.0}
    scenarios = [(0.5, {0, 1, 2}), (0.5, {0, 1})]
    return dict(
        sites=[0, 1, 2],
        features=[1, 2, 3],
        contains=contains,
        cost=cost,
        scenarios=scenarios,
        budget1=budget1,
        budget2=budget2,
    )


def _brute_force(p) -> float:
    sites, feats = p["sites"], p["features"]
    contains, cost = p["contains"], p["cost"]
    scenarios = p["scenarios"]
    b1, b2 = p["budget1"], p["budget2"]

    def covered(selected):
        return sum(1 for f in feats if contains[f] & selected)

    best = -1.0
    for k in range(len(sites) + 1):
        for s1 in itertools.combinations(sites, k):
            s1 = set(s1)
            if sum(cost[i] for i in s1) > b1 + 1e-9:
                continue
            exp = 0.0
            for prob, avail in scenarios:
                # best stage-2 add from available, not already chosen, within b2
                pool = [i for i in avail if i not in s1]
                best_s = covered(s1)
                for kk in range(len(pool) + 1):
                    for add in itertools.combinations(pool, kk):
                        if sum(cost[i] for i in add) > b2 + 1e-9:
                            continue
                        best_s = max(best_s, covered(s1 | set(add)))
                exp += prob * best_s
            best = max(best, exp)
    return best


def test_mip_matches_brute_force():
    for b1, b2 in [(0.0, 0.0), (1.0, 1.0), (2.0, 1.0), (1.0, 2.0), (3.0, 3.0)]:
        p = _problem(b1, b2)
        sol = two_stage_reserve_mip(**p)
        assert sol.optimal is True
        assert sol.expected_coverage == pytest.approx(_brute_force(p), abs=1e-6)


def test_recourse_handles_scenario_loss():
    """Stage-1 budget for one site + stage-2 for one more = 2 sites per
    scenario; with 3 features the optimum covers 2 per scenario → expected 2.0,
    matching brute force (and the recourse keeps both scenarios at 2)."""
    p = _problem(budget1=1.0, budget2=1.0)
    sol = two_stage_reserve_mip(**p)
    assert sol.expected_coverage == pytest.approx(_brute_force(p), abs=1e-6)
    assert sol.expected_coverage == pytest.approx(2.0)


def test_zero_budget_zero_coverage():
    p = _problem(budget1=0.0, budget2=0.0)
    sol = two_stage_reserve_mip(**p)
    assert sol.expected_coverage == pytest.approx(0.0)
    assert sol.stage1 == set()


def test_weights_applied():
    p = _problem(budget1=1.0, budget2=0.0)
    p["weights"] = {1: 10.0, 2: 1.0, 3: 1.0}
    sol = two_stage_reserve_mip(**p)
    # f1 (weight 10) is in s0, available everywhere → protect s0 in stage 1
    assert 0 in sol.stage1


def test_recourse_only_exposes_scenario_loss():
    """Recourse-only (no stage-1): scenario A can recover all 3 features but
    scenario B can only reach f1,f2 (s2 is lost and can't be added in stage 2)
    → expected = .5*3 + .5*2 = 2.5. This is where future loss actually bites."""
    p = _problem(budget1=0.0, budget2=3.0)
    sol = two_stage_reserve_mip(**p)
    assert sol.expected_coverage == pytest.approx(2.5)


def test_stage1_secures_against_loss():
    """With enough stage-1 budget, protecting s2 *now* secures f3 in every
    scenario (stage-1 sites are scenario-independent) → full coverage 3.0."""
    p = _problem(budget1=3.0, budget2=3.0)
    sol = two_stage_reserve_mip(**p)
    assert sol.expected_coverage == pytest.approx(3.0)
    assert 2 in sol.stage1  # the at-risk site is protected early
