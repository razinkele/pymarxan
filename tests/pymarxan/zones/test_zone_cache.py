"""Tests for ZoneProblemCache — precomputed zone arrays with delta computation."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pymarxan.zones.cache import ZoneProblemCache
from pymarxan.zones.objective import compute_zone_objective
from pymarxan.zones.readers import load_zone_project
from tests.pymarxan.zones.marzone_anchor import all_assignments, make_anchor_problem, oracle

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "zones"


@pytest.fixture
def zone_problem():
    """Load the standard zone test problem (4 PUs, 2 features, 2 zones)."""
    return load_zone_project(DATA_DIR)


@pytest.fixture
def cache(zone_problem):
    """Build a ZoneProblemCache from the zone test problem."""
    return ZoneProblemCache.from_zone_problem(zone_problem)


class TestConstruction:
    """Verify cache construction produces correct shapes and metadata."""

    def test_n_pu(self, cache):
        assert cache.n_pu == 4

    def test_n_feat(self, cache):
        assert cache.n_feat == 2

    def test_n_zones(self, cache):
        assert cache.n_zones == 2

    def test_zone_cost_matrix_shape(self, cache):
        # Shape: (n_pu, n_zones+1) — col 0 = unassigned, cols 1..n = zones
        assert cache.zone_cost_matrix.shape == (4, 3)

    def test_zone_cost_matrix_unassigned_col_zero(self, cache):
        # Column 0 (unassigned) should be all zeros
        np.testing.assert_array_equal(
            cache.zone_cost_matrix[:, 0], np.zeros(4)
        )

    def test_zone_cost_matrix_values(self, cache):
        # From zonecost.dat: PU1 zone1=100, zone2=50, etc.
        # zone_id_to_col maps zone IDs to columns
        col1 = cache.zone_id_to_col[1]
        col2 = cache.zone_id_to_col[2]
        assert cache.zone_cost_matrix[0, col1] == 100.0
        assert cache.zone_cost_matrix[0, col2] == 50.0
        assert cache.zone_cost_matrix[1, col1] == 150.0
        assert cache.zone_cost_matrix[1, col2] == 80.0

    def test_contribution_matrix_shape(self, cache):
        # Shape: (n_zones+1, n_feat)
        assert cache.contribution_matrix.shape == (3, 2)

    def test_contribution_matrix_unassigned_row_zero(self, cache):
        # Row 0 (unassigned) should be all zeros
        np.testing.assert_array_equal(
            cache.contribution_matrix[0, :], np.zeros(2)
        )

    def test_contribution_matrix_values(self, cache):
        # From zonecontrib.dat: feature1/zone1=1.0, feature1/zone2=0.5,
        # feature2/zone1=1.0, feature2/zone2=0.3
        col1 = cache.zone_id_to_col[1]
        col2 = cache.zone_id_to_col[2]
        feat_col_0 = cache.feat_id_to_col[1]  # feature 1
        feat_col_1 = cache.feat_id_to_col[2]  # feature 2
        assert cache.contribution_matrix[col1, feat_col_0] == 1.0
        assert cache.contribution_matrix[col2, feat_col_0] == 0.5
        assert cache.contribution_matrix[col1, feat_col_1] == 1.0
        assert cache.contribution_matrix[col2, feat_col_1] == 0.3

    def test_zone_target_matrix_shape(self, cache):
        # Shape: (n_zones+1, n_feat)
        assert cache.zone_target_matrix.shape == (3, 2)

    def test_zone_target_matrix_unassigned_row_zero(self, cache):
        # Row 0 (unassigned) should be all zeros (no target for unassigned)
        np.testing.assert_array_equal(
            cache.zone_target_matrix[0, :], np.zeros(2)
        )

    def test_zone_target_matrix_values(self, cache):
        # From zonetarget.dat: zone1/feature1=10, zone1/feature2=8,
        # zone2/feature1=5, zone2/feature2=3
        col1 = cache.zone_id_to_col[1]
        col2 = cache.zone_id_to_col[2]
        feat_col_0 = cache.feat_id_to_col[1]
        feat_col_1 = cache.feat_id_to_col[2]
        assert cache.zone_target_matrix[col1, feat_col_0] == 10.0
        assert cache.zone_target_matrix[col1, feat_col_1] == 8.0
        assert cache.zone_target_matrix[col2, feat_col_0] == 5.0
        assert cache.zone_target_matrix[col2, feat_col_1] == 3.0

    def test_zone_boundary_costs_dict(self, cache):
        # From zoneboundcost.dat: (1,2)=50, (2,1)=50, (1,1)=0, (2,2)=0
        col1 = cache.zone_id_to_col[1]
        col2 = cache.zone_id_to_col[2]
        assert cache.zone_boundary_costs.get((col1, col2), 0.0) == 50.0
        assert cache.zone_boundary_costs.get((col2, col1), 0.0) == 50.0

    def test_zone_boundary_costs_symmetric(self, cache):
        """Zone boundary costs must be stored in both directions."""
        for (z1, z2), cost in list(cache.zone_boundary_costs.items()):
            reverse = cache.zone_boundary_costs.get((z2, z1), None)
            assert reverse is not None, f"Missing reverse for ({z1},{z2})"
            assert reverse == cost

    def test_pu_feat_matrix_shape(self, cache):
        assert cache.pu_feat_matrix.shape == (4, 2)

    def test_inherits_base_fields(self, cache):
        """Cache should carry over ProblemCache-like fields."""
        assert cache.neighbors is not None
        assert cache.self_boundary is not None
        assert len(cache.self_boundary) == 4
        assert cache.feat_spf is not None
        assert len(cache.feat_spf) == 2


class TestHeldPerZone:
    """Verify ZoneHeld.per_zone (raw) computation."""

    def test_all_unassigned(self, cache):
        assignment = np.zeros(4, dtype=int)
        held = cache.compute_held(assignment)
        assert held.per_zone.shape == (3, 2)  # (n_zones+1, n_feat)
        # All unassigned => all held is zero (contribution[0, :] = 0)
        np.testing.assert_array_almost_equal(held.per_zone, np.zeros((3, 2)))

    def test_all_zone1(self, cache):
        assignment = np.array([1, 1, 1, 1], dtype=int)
        held = cache.compute_held(assignment)
        col1 = cache.zone_id_to_col[1]
        # All PUs in zone 1, contribution for zone1 is [1.0, 1.0]
        # PU amounts: feature1=[10,8,6,5]=29, feature2=[5,7,9,4]=25
        # held[col1, 0] = 29 * 1.0, held[col1, 1] = 25 * 1.0
        assert held.per_zone[col1, cache.feat_id_to_col[1]] == pytest.approx(29.0)
        assert held.per_zone[col1, cache.feat_id_to_col[2]] == pytest.approx(25.0)

    def test_mixed_assignment(self, cache):
        # PU0->zone1, PU1->zone2, PU2->zone1, PU3->unassigned
        assignment = np.array([1, 2, 1, 0], dtype=int)
        held = cache.compute_held(assignment)
        col1 = cache.zone_id_to_col[1]
        col2 = cache.zone_id_to_col[2]
        f0 = cache.feat_id_to_col[1]
        f1 = cache.feat_id_to_col[2]
        # Zone1: PU0(10,5) + PU2(6,9) contrib=[1.0, 1.0]
        # held[col1, f0] = (10+6)*1.0 = 16
        # held[col1, f1] = (5+9)*1.0 = 14
        assert held.per_zone[col1, f0] == pytest.approx(16.0)
        assert held.per_zone[col1, f1] == pytest.approx(14.0)
        # Zone2: PU1(8,7) — raw amounts (MarZone reserve.hpp:164); contributions no longer apply
        # to zone targets. Old pins were 4.0 / 2.1 (8×0.5, 7×0.3).
        assert held.per_zone[col2, f0] == pytest.approx(8.0)
        assert held.per_zone[col2, f1] == pytest.approx(7.0)

    def test_update_held(self, cache):
        """Incremental update matches full recomputation."""
        assignment = np.array([1, 2, 1, 0], dtype=int)
        held = cache.compute_held(assignment)

        # Change PU1 from zone2 -> zone1
        old_zone = 2
        new_zone = 1
        idx = 1
        cache.update_held(held, idx, old_zone, new_zone)

        # Compute from scratch with updated assignment
        assignment[idx] = new_zone
        held_expected = cache.compute_held(assignment)

        np.testing.assert_array_almost_equal(held.per_zone, held_expected.per_zone)
        np.testing.assert_array_almost_equal(held.overall, held_expected.overall)

    def test_update_to_unassigned(self, cache):
        """Moving a PU to unassigned (zone 0)."""
        assignment = np.array([1, 2, 1, 2], dtype=int)
        held = cache.compute_held(assignment)

        # Change PU3 from zone2 -> unassigned
        cache.update_held(held, 3, 2, 0)
        assignment[3] = 0
        held_expected = cache.compute_held(assignment)

        np.testing.assert_array_almost_equal(held.per_zone, held_expected.per_zone)
        np.testing.assert_array_almost_equal(held.overall, held_expected.overall)

    def test_update_from_unassigned(self, cache):
        """Moving a PU from unassigned (zone 0) to a zone."""
        assignment = np.array([0, 0, 0, 0], dtype=int)
        held = cache.compute_held(assignment)

        # Change PU0 from unassigned -> zone2
        cache.update_held(held, 0, 0, 2)
        assignment[0] = 2
        held_expected = cache.compute_held(assignment)

        np.testing.assert_array_almost_equal(held.per_zone, held_expected.per_zone)
        np.testing.assert_array_almost_equal(held.overall, held_expected.overall)


class TestFullObjective:
    """Verify compute_full_zone_objective matches the DataFrame-based implementation."""

    def test_matches_reference_random(self, cache, zone_problem):
        """Full objective matches compute_zone_objective() for 10 random assignments."""
        rng = np.random.default_rng(42)
        zone_options = [0, 1, 2]
        blm = float(zone_problem.parameters.get("BLM", 1.0))

        for _ in range(10):
            assignment = np.array(
                [rng.choice(zone_options) for _ in range(4)], dtype=int
            )
            held = cache.compute_held(assignment)
            cached_obj = cache.compute_full_zone_objective(
                assignment, held=held, blm=blm
            )
            ref_obj = compute_zone_objective(zone_problem, assignment, blm)
            assert cached_obj == pytest.approx(
                ref_obj, abs=1e-10
            ), f"Mismatch for assignment={assignment}"

    def test_all_unassigned(self, cache, zone_problem):
        """All unassigned => objective should be 0 + penalty."""
        assignment = np.zeros(4, dtype=int)
        blm = 1.0
        held = cache.compute_held(assignment)
        cached_obj = cache.compute_full_zone_objective(assignment, held=held, blm=blm)
        ref_obj = compute_zone_objective(zone_problem, assignment, blm)
        assert cached_obj == pytest.approx(ref_obj, abs=1e-10)

    def test_all_zone1(self, cache, zone_problem):
        """All PUs in zone 1."""
        assignment = np.array([1, 1, 1, 1], dtype=int)
        blm = 1.0
        held = cache.compute_held(assignment)
        cached_obj = cache.compute_full_zone_objective(assignment, held=held, blm=blm)
        ref_obj = compute_zone_objective(zone_problem, assignment, blm)
        assert cached_obj == pytest.approx(ref_obj, abs=1e-10)

    def test_blm_zero(self, cache, zone_problem):
        """BLM=0 => no boundary contribution."""
        assignment = np.array([1, 2, 1, 2], dtype=int)
        blm = 0.0
        held = cache.compute_held(assignment)
        cached_obj = cache.compute_full_zone_objective(assignment, held=held, blm=blm)
        ref_obj = compute_zone_objective(zone_problem, assignment, blm)
        assert cached_obj == pytest.approx(ref_obj, abs=1e-10)


class TestDeltaObjective:
    """Verify delta == full_after - full_before for zone changes."""

    def _verify_delta(self, cache, zone_problem, assignment, idx, new_zone, blm):
        """Helper: verify delta matches difference of full computations."""
        old_zone = int(assignment[idx])
        held = cache.compute_held(assignment)

        # Full objective before
        obj_before = cache.compute_full_zone_objective(assignment, held=held, blm=blm)

        # Compute delta
        delta = cache.compute_delta_zone_objective(
            idx, old_zone, new_zone, assignment, held=held, blm=blm
        )

        # Apply change and compute full objective after
        assignment_after = assignment.copy()
        assignment_after[idx] = new_zone
        held_after = cache.compute_held(assignment_after)
        obj_after = cache.compute_full_zone_objective(
            assignment_after, held=held_after, blm=blm
        )

        expected_delta = obj_after - obj_before
        assert delta == pytest.approx(
            expected_delta, abs=1e-10
        ), (
            f"Delta mismatch: idx={idx}, {old_zone}->{new_zone}, "
            f"delta={delta}, expected={expected_delta}, "
            f"before={obj_before}, after={obj_after}"
        )

    def test_zone_to_zone(self, cache, zone_problem):
        """Changing a PU from one zone to another."""
        assignment = np.array([1, 2, 1, 2], dtype=int)
        blm = 1.0
        self._verify_delta(cache, zone_problem, assignment, 0, 2, blm)

    def test_zone_to_unassigned(self, cache, zone_problem):
        """Removing a PU from a zone (going to unassigned)."""
        assignment = np.array([1, 2, 1, 2], dtype=int)
        blm = 1.0
        self._verify_delta(cache, zone_problem, assignment, 1, 0, blm)

    def test_unassigned_to_zone(self, cache, zone_problem):
        """Assigning a PU from unassigned to a zone."""
        assignment = np.array([0, 0, 1, 2], dtype=int)
        blm = 1.0
        self._verify_delta(cache, zone_problem, assignment, 0, 1, blm)

    def test_unassigned_to_unassigned(self, cache, zone_problem):
        """No-op: unassigned to unassigned should be zero delta."""
        assignment = np.array([0, 1, 2, 0], dtype=int)
        blm = 1.0
        held = cache.compute_held(assignment)
        delta = cache.compute_delta_zone_objective(
            0, 0, 0, assignment, held=held, blm=blm
        )
        assert delta == pytest.approx(0.0, abs=1e-10)

    def test_same_zone_no_change(self, cache, zone_problem):
        """No-op: zone1 to zone1 should be zero delta."""
        assignment = np.array([1, 2, 1, 2], dtype=int)
        blm = 1.0
        held = cache.compute_held(assignment)
        delta = cache.compute_delta_zone_objective(
            0, 1, 1, assignment, held=held, blm=blm
        )
        assert delta == pytest.approx(0.0, abs=1e-10)

    def test_blm_zero_delta(self, cache, zone_problem):
        """BLM=0: boundary should not contribute to delta."""
        assignment = np.array([1, 2, 1, 0], dtype=int)
        blm = 0.0
        self._verify_delta(cache, zone_problem, assignment, 3, 1, blm)

    def test_sequential_changes(self, cache, zone_problem):
        """Apply a sequence of zone changes and verify each delta."""
        blm = 1.0
        assignment = np.array([0, 0, 0, 0], dtype=int)
        changes = [
            (0, 1),  # PU0 -> zone1
            (1, 2),  # PU1 -> zone2
            (2, 1),  # PU2 -> zone1
            (3, 2),  # PU3 -> zone2
            (0, 2),  # PU0 -> zone2
            (1, 0),  # PU1 -> unassigned
        ]

        for idx, new_zone in changes:
            self._verify_delta(
                cache, zone_problem, assignment.copy(), idx, new_zone, blm
            )
            assignment[idx] = new_zone

    def test_random_changes(self, cache, zone_problem):
        """Random assignment changes all produce correct deltas."""
        rng = np.random.default_rng(123)
        zone_options = [0, 1, 2]
        blm = 1.0
        assignment = np.array(
            [rng.choice(zone_options) for _ in range(4)], dtype=int
        )

        for _ in range(20):
            idx = rng.integers(4)
            new_zone = rng.choice(zone_options)
            self._verify_delta(
                cache, zone_problem, assignment.copy(), idx, new_zone, blm
            )
            assignment[idx] = new_zone


class TestMissLevel:
    """Verify MISSLEVEL is applied to zone target matrix."""

    def test_misslevel_scales_targets(self, zone_problem):
        """With MISSLEVEL=0.5, targets should be halved."""
        import copy
        problem = copy.deepcopy(zone_problem)
        problem.parameters["MISSLEVEL"] = 0.5
        cache = ZoneProblemCache.from_zone_problem(problem)

        col1 = cache.zone_id_to_col[1]
        f0 = cache.feat_id_to_col[1]
        # Original target for zone1/feature1 is 10.0
        assert cache.zone_target_matrix[col1, f0] == pytest.approx(5.0)

    def test_misslevel_default_no_change(self, zone_problem):
        """Default MISSLEVEL=1.0 should leave targets unchanged."""
        cache = ZoneProblemCache.from_zone_problem(zone_problem)
        col1 = cache.zone_id_to_col[1]
        f0 = cache.feat_id_to_col[1]
        assert cache.zone_target_matrix[col1, f0] == pytest.approx(10.0)

    def test_penalty_uses_scaled_targets(self, zone_problem):
        """Cache penalty should agree with objective.compute_zone_penalty."""
        import copy
        problem = copy.deepcopy(zone_problem)
        problem.parameters["MISSLEVEL"] = 0.5
        cache = ZoneProblemCache.from_zone_problem(problem)

        assignment = np.array([1, 2, 1, 2], dtype=int)
        held = cache.compute_held(assignment)
        cached_obj = cache.compute_full_zone_objective(assignment, held=held, blm=0.0)

        from pymarxan.zones.objective import compute_zone_objective
        ref_obj = compute_zone_objective(problem, assignment, 0.0)
        assert cached_obj == pytest.approx(ref_obj, abs=1e-10)


class TestZoneTargetWeight:
    def test_weight_is_ones_by_default(self, cache):
        np.testing.assert_array_equal(cache.zone_target_weight[1:], np.ones((2, 2)))
        np.testing.assert_array_equal(cache.zone_target_weight[0], np.zeros(2))

    def test_weight_is_contribution_matrix_when_flag_set(self, zone_problem):
        zone_problem.parameters["ZONETARGETCONTRIB"] = 1
        c = ZoneProblemCache.from_zone_problem(zone_problem)
        np.testing.assert_array_equal(c.zone_target_weight, c.contribution_matrix)
        np.testing.assert_array_equal(c.contribution_matrix, zone_problem.contribution_matrix())

    def test_cache_rejects_out_of_domain_flag(self, zone_problem):
        zone_problem.parameters["ZONETARGETCONTRIB"] = 2
        with pytest.raises(ValueError, match="ZONETARGETCONTRIB"):
            ZoneProblemCache.from_zone_problem(zone_problem)

    def test_full_objective_matches_reference_in_weighted_mode(self, zone_problem):
        zone_problem.parameters["ZONETARGETCONTRIB"] = 1
        c = ZoneProblemCache.from_zone_problem(zone_problem)
        rng = np.random.default_rng(7)
        for _ in range(10):
            assignment = rng.integers(0, 3, size=4)
            held = c.compute_held(assignment)
            assert c.compute_full_zone_objective(assignment, held=held, blm=1.0) == pytest.approx(
                compute_zone_objective(zone_problem, assignment, 1.0), abs=1e-10,
            )

    def test_delta_matches_reference_in_weighted_mode(self, zone_problem):
        zone_problem.parameters["ZONETARGETCONTRIB"] = 1
        c = ZoneProblemCache.from_zone_problem(zone_problem)
        rng = np.random.default_rng(11)
        for _ in range(20):
            assignment = rng.integers(0, 3, size=4)
            idx = int(rng.integers(4))
            old_zone = int(assignment[idx])
            new_zone = int(rng.choice([z for z in (0, 1, 2) if z != old_zone]))
            held = c.compute_held(assignment)
            before = c.compute_full_zone_objective(assignment, held=held, blm=1.0)
            delta = c.compute_delta_zone_objective(
                idx, old_zone, new_zone, assignment, held=held, blm=1.0,
            )
            after_assignment = assignment.copy()
            after_assignment[idx] = new_zone
            after = c.compute_full_zone_objective(
                after_assignment, held=c.compute_held(after_assignment), blm=1.0,
            )
            assert delta == pytest.approx(after - before, abs=1e-10)


class TestZoneHeld:
    @pytest.mark.parametrize("flag", [0, 1])
    def test_overall_equals_contribution_weighted_per_zone(self, zone_problem, flag):
        zone_problem.parameters["ZONETARGETCONTRIB"] = flag
        c = ZoneProblemCache.from_zone_problem(zone_problem)
        rng = np.random.default_rng(3)
        for _ in range(10):
            a = rng.integers(0, 3, size=4)
            held = c.compute_held(a)
            np.testing.assert_allclose(
                (c.contribution_matrix * held.per_zone).sum(axis=0), held.overall, atol=1e-12,
            )

    @pytest.mark.parametrize("flag", [0, 1])
    def test_per_zone_is_raw_in_both_modes(self, zone_problem, flag):
        zone_problem.parameters["ZONETARGETCONTRIB"] = flag
        c = ZoneProblemCache.from_zone_problem(zone_problem)
        held = c.compute_held(np.array([1, 2, 1, 0]))
        assert held.per_zone[c.zone_id_to_col[2], c.feat_id_to_col[1]] == pytest.approx(8.0)

    def test_overall_is_contribution_weighted(self, cache):
        held = cache.compute_held(np.array([1, 2, 1, 0]))
        # f1: PU0 10×1.0 + PU2 6×1.0 + PU1 8×0.5 = 20; f2: 5 + 9 + 7×0.3 = 16.1
        assert held.overall[cache.feat_id_to_col[1]] == pytest.approx(20.0)
        assert held.overall[cache.feat_id_to_col[2]] == pytest.approx(16.1)

    def test_update_keeps_invariant_through_a_walk(self, cache):
        rng = np.random.default_rng(5)
        a = rng.integers(0, 3, size=4)
        held = cache.compute_held(a)
        for _ in range(50):
            idx = int(rng.integers(4))
            new = int(rng.integers(0, 3))
            cache.update_held(held, idx, int(a[idx]), new)
            a[idx] = new
        fresh = cache.compute_held(a)
        np.testing.assert_allclose(held.per_zone, fresh.per_zone, atol=1e-12)
        np.testing.assert_allclose(held.overall, fresh.overall, atol=1e-12)


class TestOverallTermInCache:
    def test_overall_target_vector_applies_misslevel_and_zeroes_inert(self, zone_problem):
        zone_problem.parameters["MISSLEVEL"] = 0.5
        zone_problem.features.loc[1, "target"] = 0.0
        c = ZoneProblemCache.from_zone_problem(zone_problem)
        np.testing.assert_array_equal(c.overall_target_vector, [10.0, 0.0])
        assert c.has_overall_targets

    def test_no_overall_targets_flag(self, zone_problem):
        zone_problem.features["target"] = 0.0
        c = ZoneProblemCache.from_zone_problem(zone_problem)
        assert not c.has_overall_targets
        held = c.compute_held(np.array([0, 0, 0, 0]))
        assert c._compute_overall_penalty(held.overall) == 0.0

    def test_contrib_differs_gate(self, zone_problem):
        c = ZoneProblemCache.from_zone_problem(zone_problem)
        assert c.contrib_differs[1, 2]          # fixture contributions differ (1.0 vs 0.5/0.3)
        assert c.contrib_differs[0, 1]
        assert not c.contrib_differs[1, 1]
        plain = ZoneProblemCache.from_zone_problem(
            zone_problem.copy_with(zone_contributions=None),
        )
        assert not plain.contrib_differs[1, 2]  # default contributions: zone-to-zone moves free
        assert plain.contrib_differs[0, 2]      # assigning an unassigned PU still changes A_f

    def test_gated_move_has_zero_overall_delta(self, zone_problem):
        plain = ZoneProblemCache.from_zone_problem(
            zone_problem.copy_with(zone_contributions=None),
        )
        held = plain.compute_held(np.array([1, 1, 2, 2]))
        assert plain._overall_penalty_delta(0, 1, 2, held.overall) == 0.0

    @pytest.mark.parametrize("flag", [0, 1])
    def test_delta_matches_full_recomputation_random_flips(self, zone_problem, flag):
        zone_problem.parameters["ZONETARGETCONTRIB"] = flag
        c = ZoneProblemCache.from_zone_problem(zone_problem)
        rng = np.random.default_rng(13)
        for _ in range(40):
            a = rng.integers(0, 3, size=4)
            idx = int(rng.integers(4))
            old = int(a[idx])
            new = int(rng.choice([z for z in (0, 1, 2) if z != old]))
            held = c.compute_held(a)
            before = c.compute_full_zone_objective(a, held=held, blm=1.0)
            delta = c.compute_delta_zone_objective(idx, old, new, a, held=held, blm=1.0)
            b = a.copy()
            b[idx] = new
            after = c.compute_full_zone_objective(b, held=c.compute_held(b), blm=1.0)
            assert delta == pytest.approx(after - before, abs=1e-10)

    def test_spf_column_absent_defaults_to_one(self):
        p = make_anchor_problem()
        p = p.copy_with(features=p.features.drop(columns=["spf"]))
        c = ZoneProblemCache.from_zone_problem(p)
        np.testing.assert_array_equal(c.feat_spf, [1.0])


class TestCacheAgainstAnchorOracle:
    @pytest.mark.parametrize("spf", [1.0, 10.0])
    @pytest.mark.parametrize("flag", [0, 1])
    def test_full_objective_equals_oracle_for_all_27(self, spf, flag):
        p = make_anchor_problem(spf=spf, zone_target_contrib=flag)
        c = ZoneProblemCache.from_zone_problem(p)
        for a in all_assignments():
            arr = np.array(a)
            got = c.compute_full_zone_objective(arr, held=c.compute_held(arr), blm=0.0)
            assert got == pytest.approx(oracle(a, spf=spf, zone_target_contrib=flag)["objective"])
