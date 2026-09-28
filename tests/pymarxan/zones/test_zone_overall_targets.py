"""Overall (contribution-weighted) feature targets — objective module, spec §3.2."""
from __future__ import annotations

import numpy as np
import pytest

from pymarxan.zones.objective import (
    check_overall_targets,
    compute_overall_achieved,
    compute_overall_penalty,
    compute_overall_shortfall,
    compute_overall_shortfalls,
    compute_zone_shortfalls,
)
from tests.pymarxan.zones.marzone_anchor import (
    OPTIMUM,
    OPTIMUM_COST,
    all_assignments,
    make_anchor_problem,
    oracle,
)


class TestOracleSelfConsistency:
    """The oracle itself must reproduce the spec table before anything is compared to it."""

    def test_hard_optimum_is_unique(self):
        feasible = [(oracle(a)["cost"], a) for a in all_assignments() if oracle(a)["feasible"]]
        best = min(c for c, _ in feasible)
        assert best == OPTIMUM_COST
        assert [a for c, a in feasible if c == best] == [OPTIMUM]

    def test_spf_one_penalised_minimum_is_infeasible(self):
        best = min(all_assignments(), key=lambda a: oracle(a, spf=1.0)["objective"])
        assert best == (2, 2, 2)
        assert oracle(best, spf=1.0)["objective"] == 6.0

    def test_spf_ten_penalised_argmin_equals_hard_optimum(self):
        best = min(all_assignments(), key=lambda a: oracle(a, spf=10.0)["objective"])
        assert best == OPTIMUM
        assert oracle((2, 2, 2), spf=10.0)["objective"] == 33.0

    def test_contrib_weighted_zone_targets_infeasible_everywhere(self):
        assert not any(oracle(a, zone_target_contrib=1)["feasible"] for a in all_assignments())


class TestOverallAchieved:
    def test_matches_oracle_for_all_assignments(self):
        p = make_anchor_problem()
        for a in all_assignments():
            got = compute_overall_achieved(p, np.array(a))
            assert got == {1: pytest.approx(oracle(a)["overall"])}

    def test_unassigned_contributes_nothing(self):
        p = make_anchor_problem()
        assert compute_overall_achieved(p, np.array([0, 0, 0])) == {1: 0.0}

    def test_amounts_override_is_used(self):
        p = make_anchor_problem()
        amounts = np.full((3, 1), 2.0)
        got = compute_overall_achieved(p, np.array([1, 2, 0]), amounts=amounts)
        assert got == {1: pytest.approx(2.0 * 1.0 + 2.0 * 0.4)}

    def test_default_contribution_without_table_is_one(self):
        p = make_anchor_problem()
        p = p.copy_with(zone_contributions=None)
        assert compute_overall_achieved(p, np.array([2, 2, 2])) == {1: pytest.approx(30.0)}

    def test_default_contribution_with_partial_table_is_zero(self):
        p = make_anchor_problem()
        partial = p.zone_contributions[p.zone_contributions["zone"] == 1]
        p = p.copy_with(zone_contributions=partial)
        assert compute_overall_achieved(p, np.array([2, 2, 1])) == {1: pytest.approx(10.0)}


class TestOverallTargetsAndPenalty:
    def test_met_flags_follow_misslevel(self):
        p = make_anchor_problem()
        assert check_overall_targets(p, np.array([2, 2, 2])) == {1: False}   # 12 < 15
        assert check_overall_targets(p, np.array(OPTIMUM)) == {1: True}     # 18 >= 15
        p.parameters["MISSLEVEL"] = 0.8
        assert check_overall_targets(p, np.array([2, 2, 2])) == {1: True}    # 12 >= 12

    def test_inert_target_is_met(self):
        p = make_anchor_problem()
        p.features.loc[0, "target"] = 0.0
        assert check_overall_targets(p, np.array([0, 0, 0])) == {1: True}
        assert compute_overall_shortfalls(p, np.array([0, 0, 0])) == {}

    def test_penalty_is_spf_times_shortfall(self):
        p = make_anchor_problem(spf=10.0)
        assert compute_overall_penalty(p, np.array([2, 2, 2])) == pytest.approx(30.0)
        assert compute_overall_shortfall(p, np.array([2, 2, 2])) == pytest.approx(3.0)
        assert compute_overall_shortfalls(p, np.array([2, 2, 2])) == {1: pytest.approx(3.0)}
        assert compute_overall_penalty(p, np.array(OPTIMUM)) == 0.0

    def test_penalty_scales_with_misslevel(self):
        p = make_anchor_problem(spf=1.0, misslevel=0.5)
        # target 7.5; (0, 2, 0) achieves 4 -> shortfall 3.5
        assert compute_overall_penalty(p, np.array([0, 2, 0])) == pytest.approx(3.5)

    def test_spf_column_absent_defaults_to_one(self):
        p = make_anchor_problem()
        p = p.copy_with(features=p.features.drop(columns=["spf"]))
        assert compute_overall_penalty(p, np.array([2, 2, 2])) == pytest.approx(3.0)


class TestZoneShortfalls:
    def test_per_pair_shortfalls_on_anchor(self):
        p = make_anchor_problem()
        # Task 2 is additive: _compute_zone_achieved is still contribution-weighted here, so
        # only assignments whose zone-2 shortfall is invariant under both semantics are pinned.
        # The raw pin [2, 0, 0] -> 0.0 lands in Task 3 (test_zone_target_met_on_raw_amounts).
        # 30 raw / 12 weighted: met under both semantics
        assert compute_zone_shortfalls(p, np.array([2, 2, 2])) == {(2, 1): 0.0}
        assert compute_zone_shortfalls(p, np.array([1, 1, 0])) == {(2, 1): 10.0}

    def test_empty_without_zone_targets(self):
        p = make_anchor_problem().copy_with(zone_targets=None)
        assert compute_zone_shortfalls(p, np.array([1, 1, 1])) == {}


class TestZoneTargetsRawOnAnchor:
    def test_zone_target_met_on_raw_amounts(self):
        from pymarxan.zones.objective import check_zone_targets
        p = make_anchor_problem()
        assert check_zone_targets(p, np.array([2, 0, 0])) == {(2, 1): True}     # 10 raw >= 10
        # 10 raw in zone 2 (was 4.0 weighted before this task's flip)
        assert compute_zone_shortfalls(p, np.array([2, 0, 0])) == {(2, 1): 0.0}

    def test_zone_target_weighted_under_flag(self):
        from pymarxan.zones.objective import check_zone_targets
        p = make_anchor_problem(zone_target_contrib=1)
        assert check_zone_targets(p, np.array([2, 2, 0])) == {(2, 1): False}    # 8 < 10
        assert check_zone_targets(p, np.array([2, 2, 2])) == {(2, 1): True}     # 12 >= 10


class TestZoneObjectiveIncludesBothTiers:
    @pytest.mark.parametrize("spf", [1.0, 10.0])
    @pytest.mark.parametrize("flag", [0, 1])
    def test_compute_zone_objective_equals_oracle(self, spf, flag):
        from pymarxan.zones.objective import compute_zone_objective
        p = make_anchor_problem(spf=spf, zone_target_contrib=flag)
        for a in all_assignments():
            got = compute_zone_objective(p, np.array(a), 0.0)
            assert got == pytest.approx(oracle(a, spf=spf, zone_target_contrib=flag)["objective"])

    def test_spf_one_penalised_minimum_is_the_infeasible_corner(self):
        from pymarxan.zones.objective import compute_zone_objective
        p = make_anchor_problem(spf=1.0)
        assert compute_zone_objective(p, np.array([2, 2, 2]), 0.0) == pytest.approx(6.0)
        assert check_overall_targets(p, np.array([2, 2, 2])) == {1: False}
