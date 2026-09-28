"""Overall (contribution-weighted) feature targets — objective module, spec §3.2."""
from __future__ import annotations

import numpy as np
import pytest

from pymarxan.zones.objective import (
    check_overall_targets,
    check_zone_targets,
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
        # Only assignments whose zone-2 shortfall is the same under raw and
        # contribution-weighted accumulation are pinned here; the raw-only case
        # [2, 0, 0] -> 0.0 is pinned by test_zone_target_met_on_raw_amounts.
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
        # 10 raw in zone 2 (it would be 4.0 under contribution weighting)
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


class TestBuildZoneSolution:
    def test_fields_on_anchor(self):
        from pymarxan.zones.objective import build_zone_solution, compute_zone_objective
        p = make_anchor_problem(spf=10.0)
        sol = build_zone_solution(p, np.array([2, 2, 2]), 0.0, solver_name="test", run=3)
        assert sol.cost == 3.0
        assert sol.objective == compute_zone_objective(p, np.array([2, 2, 2]), 0.0)
        assert sol.objective == pytest.approx(33.0)
        assert sol.targets_met == {1: False}
        assert sol.all_targets_met is False
        assert sol.penalty == pytest.approx(30.0)
        assert sol.shortfall == pytest.approx(3.0)
        assert sol.metadata["zone_targets_met"] == {"z2_f1": True}
        assert sol.metadata["overall_penalty"] == pytest.approx(30.0)
        assert sol.metadata["zone_penalty"] == 0.0
        assert sol.metadata["solver"] == "test"
        assert sol.metadata["run"] == 3
        assert sol.metadata["zone_boundary_cost"] == 0.0
        np.testing.assert_array_equal(sol.selected, [True, True, True])
        np.testing.assert_array_equal(sol.zone_assignment, [2, 2, 2])

    def test_run_omitted_when_none(self):
        from pymarxan.zones.objective import build_zone_solution
        p = make_anchor_problem()
        sol = build_zone_solution(p, np.array(OPTIMUM), 0.0, solver_name="test")
        assert "run" not in sol.metadata
        assert sol.penalty == 0.0 and sol.shortfall == 0.0

    def test_assignment_is_copied(self):
        from pymarxan.zones.objective import build_zone_solution
        p = make_anchor_problem()
        a = np.array(OPTIMUM)
        sol = build_zone_solution(p, a, 0.0, solver_name="test")
        a[0] = 0
        assert sol.zone_assignment is not None
        assert int(sol.zone_assignment[0]) == 1


class TestMetToleranceDoesNotHideShortfalls:
    """The met test tolerates float rounding (relative 1e-9), not real shortfalls."""

    def test_overall_target_short_by_relative_1e_3_is_unmet(self):
        p = make_anchor_problem()
        p.features.loc[0, "target"] = 18.0 * (1.0 + 1e-3)   # OPTIMUM achieves 18
        assert check_overall_targets(p, np.array(OPTIMUM)) == {1: False}
        p.features.loc[0, "target"] = 18.0 * (1.0 + 1e-12)
        assert check_overall_targets(p, np.array(OPTIMUM)) == {1: True}

    def test_zone_target_short_by_relative_1e_3_is_unmet(self):
        p = make_anchor_problem()
        zt = p.zone_targets.copy()
        zt.loc[0, "target"] = 20.0 * (1.0 + 1e-3)            # OPTIMUM holds 20 raw in zone 2
        p = p.copy_with(zone_targets=zt)
        assert check_zone_targets(p, np.array(OPTIMUM)) == {(2, 1): False}
        zt.loc[0, "target"] = 20.0 * (1.0 + 1e-12)
        p = p.copy_with(zone_targets=zt)
        assert check_zone_targets(p, np.array(OPTIMUM)) == {(2, 1): True}
