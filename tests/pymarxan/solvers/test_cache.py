"""Tests for ProblemCache — precomputed arrays and delta computation."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pymarxan.io.readers import load_project
from pymarxan.solvers.cache import ProblemCache
from pymarxan.solvers.utils import compute_objective

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "simple"


@pytest.fixture()
def problem():
    return load_project(DATA_DIR)


@pytest.fixture()
def cache(problem):
    return ProblemCache.from_problem(problem)


@pytest.fixture()
def pu_index(problem):
    pu_ids = problem.planning_units["id"].tolist()
    return {pid: i for i, pid in enumerate(pu_ids)}


# ------------------------------------------------------------------
# 1. Construction tests
# ------------------------------------------------------------------


class TestConstruction:
    def test_costs_shape_and_values(self, problem, cache):
        assert cache.costs.shape == (6,)
        expected = problem.planning_units["cost"].values.astype(np.float64)
        np.testing.assert_array_equal(cache.costs, expected)

    def test_statuses_shape_and_values(self, problem, cache):
        assert cache.statuses.shape == (6,)
        expected = problem.planning_units["status"].values.astype(np.int32)
        np.testing.assert_array_equal(cache.statuses, expected)

    def test_pu_id_to_idx(self, problem, cache):
        pu_ids = problem.planning_units["id"].tolist()
        for i, pid in enumerate(pu_ids):
            assert cache.pu_id_to_idx[pid] == i

    def test_pu_feat_matrix_shape(self, cache):
        assert cache.pu_feat_matrix.shape == (6, 3)

    def test_pu_feat_matrix_values(self, cache):
        # PU 1 (idx 0): species 1 -> 10.0, species 2 -> 5.0, species 3 -> 0.0
        np.testing.assert_allclose(cache.pu_feat_matrix[0], [10.0, 5.0, 0.0])
        # PU 2 (idx 1): species 1 -> 15.0, species 2 -> 10.0, species 3 -> 6.0
        np.testing.assert_allclose(cache.pu_feat_matrix[1], [15.0, 10.0, 6.0])
        # PU 6 (idx 5): species 1 -> 10.0, species 2 -> 7.0, species 3 -> 8.0
        np.testing.assert_allclose(cache.pu_feat_matrix[5], [10.0, 7.0, 8.0])

    def test_feat_targets_shape_and_values(self, cache):
        assert cache.feat_targets.shape == (3,)
        np.testing.assert_array_equal(cache.feat_targets, [30.0, 20.0, 15.0])

    def test_feat_spf_shape_and_values(self, cache):
        assert cache.feat_spf.shape == (3,)
        np.testing.assert_array_equal(cache.feat_spf, [1.0, 1.0, 1.0])

    def test_feat_id_to_col(self, cache):
        assert cache.feat_id_to_col == {1: 0, 2: 1, 3: 2}

    def test_neighbors_length(self, cache):
        assert len(cache.neighbors) == 6

    def test_neighbors_symmetry(self, cache):
        """If (j, w) in neighbors[i], then (i, w) must be in neighbors[j]."""
        for i, nbrs in enumerate(cache.neighbors):
            for j, w in nbrs:
                found = any(ni == i and nw == w for ni, nw in cache.neighbors[j])
                assert found, (
                    f"Neighbor symmetry broken: ({i},{j},w={w}) "
                    f"not mirrored"
                )

    def test_neighbors_content(self, cache):
        # PU 1 (idx 0) neighbors: PU 2 (idx 1) with weight 1.0
        assert (1, 1.0) in cache.neighbors[0]
        # PU 3 (idx 2) neighbors: PU 2 (idx 1) and PU 4 (idx 3)
        nbr_ids = {j for j, _ in cache.neighbors[2]}
        assert nbr_ids == {1, 3}

    def test_self_boundary_shape_and_values(self, cache):
        assert cache.self_boundary.shape == (6,)
        # From bound.dat: PU1=2.0, PU2=1.0, PU3=1.0, PU4=1.0, PU5=1.0, PU6=2.0
        np.testing.assert_array_equal(
            cache.self_boundary, [2.0, 1.0, 1.0, 1.0, 1.0, 2.0]
        )

    def test_cached_scalars(self, problem, cache):
        assert cache.misslevel == float(
            problem.parameters.get("MISSLEVEL", 1.0)
        )
        assert cache.cost_thresh == float(
            problem.parameters.get("COSTTHRESH", 0.0)
        )
        assert cache.thresh_pen1 == float(
            problem.parameters.get("THRESHPEN1", 0.0)
        )
        assert cache.thresh_pen2 == float(
            problem.parameters.get("THRESHPEN2", 0.0)
        )


# ------------------------------------------------------------------
# 2. compute_held tests
# ------------------------------------------------------------------


class TestComputeHeld:
    def test_all_selected(self, cache):
        selected = np.ones(6, dtype=bool)
        held = cache.compute_held(selected)
        assert held.shape == (3,)
        # Feature 1 total: 10+15+5+12+8+10 = 60
        # Feature 2 total: 5+10+8+3+12+7 = 45
        # Feature 3 total: 0+6+10+4+5+8 = 33
        np.testing.assert_allclose(held, [60.0, 45.0, 33.0])

    def test_none_selected(self, cache):
        selected = np.zeros(6, dtype=bool)
        held = cache.compute_held(selected)
        np.testing.assert_array_equal(held, [0.0, 0.0, 0.0])

    def test_partial_selection(self, cache):
        # Select PU 1 and PU 2 (idx 0, 1)
        selected = np.array([True, True, False, False, False, False])
        held = cache.compute_held(selected)
        # Feature 1: 10+15 = 25
        # Feature 2: 5+10 = 15
        # Feature 3: 0+6 = 6
        np.testing.assert_allclose(held, [25.0, 15.0, 6.0])


# ------------------------------------------------------------------
# 3. compute_full_objective matches compute_objective
# ------------------------------------------------------------------


class TestComputeFullObjective:
    def test_matches_reference_all_selected(self, problem, cache, pu_index):
        selected = np.ones(6, dtype=bool)
        blm = 1.0
        held = cache.compute_held(selected)
        cache_obj = cache.compute_full_objective(selected, held, blm)
        ref_obj = compute_objective(problem, selected, pu_index, blm)
        assert abs(cache_obj - ref_obj) < 1e-10

    def test_matches_reference_none_selected(self, problem, cache, pu_index):
        selected = np.zeros(6, dtype=bool)
        blm = 1.0
        held = cache.compute_held(selected)
        cache_obj = cache.compute_full_objective(selected, held, blm)
        ref_obj = compute_objective(problem, selected, pu_index, blm)
        assert abs(cache_obj - ref_obj) < 1e-10

    def test_matches_reference_random_20(self, problem, cache, pu_index):
        rng = np.random.default_rng(42)
        blm = 1.0
        for _ in range(20):
            selected = rng.random(6) > 0.5
            held = cache.compute_held(selected)
            cache_obj = cache.compute_full_objective(selected, held, blm)
            ref_obj = compute_objective(problem, selected, pu_index, blm)
            assert abs(cache_obj - ref_obj) < 1e-10, (
                f"Mismatch for selection {selected}: "
                f"cache={cache_obj}, ref={ref_obj}"
            )

    def test_matches_reference_varying_blm(self, problem, cache, pu_index):
        rng = np.random.default_rng(99)
        for blm in [0.0, 0.5, 1.0, 2.0, 10.0]:
            selected = rng.random(6) > 0.5
            held = cache.compute_held(selected)
            cache_obj = cache.compute_full_objective(selected, held, blm)
            ref_obj = compute_objective(problem, selected, pu_index, blm)
            assert abs(cache_obj - ref_obj) < 1e-10


# ------------------------------------------------------------------
# 4. compute_delta_objective: delta == full_after - full_before
# ------------------------------------------------------------------


class TestComputeDeltaObjective:
    def test_delta_equals_diff_every_pu(self, cache):
        """Flip each PU and verify delta == full_after - full_before."""
        blm = 1.0
        rng = np.random.default_rng(123)
        selected = rng.random(6) > 0.5

        held = cache.compute_held(selected)
        total_cost = float(np.sum(cache.costs[selected]))
        obj_before = cache.compute_full_objective(selected, held, blm)

        for idx in range(6):
            delta = cache.compute_delta_objective(
                idx, selected, held, total_cost, blm
            )

            # Flip
            flipped = selected.copy()
            flipped[idx] = not flipped[idx]
            held_after = cache.compute_held(flipped)
            obj_after = cache.compute_full_objective(flipped, held_after, blm)

            expected_delta = obj_after - obj_before
            assert abs(delta - expected_delta) < 1e-10, (
                f"Delta mismatch at idx={idx}: "
                f"delta={delta}, expected={expected_delta}"
            )

    def test_delta_add_and_remove(self, cache):
        """Test both adding (select) and removing (deselect) a PU."""
        blm = 1.0
        # Start with none selected: adding PU 0
        selected_none = np.zeros(6, dtype=bool)
        held_none = cache.compute_held(selected_none)
        cost_none = 0.0
        obj_none = cache.compute_full_objective(selected_none, held_none, blm)

        delta_add = cache.compute_delta_objective(
            0, selected_none, held_none, cost_none, blm
        )
        sel_after = selected_none.copy()
        sel_after[0] = True
        held_after = cache.compute_held(sel_after)
        obj_after = cache.compute_full_objective(sel_after, held_after, blm)
        assert abs(delta_add - (obj_after - obj_none)) < 1e-10

        # Start with all selected: removing PU 0
        selected_all = np.ones(6, dtype=bool)
        held_all = cache.compute_held(selected_all)
        cost_all = float(np.sum(cache.costs))
        obj_all = cache.compute_full_objective(selected_all, held_all, blm)

        delta_rm = cache.compute_delta_objective(
            0, selected_all, held_all, cost_all, blm
        )
        sel_rm = selected_all.copy()
        sel_rm[0] = False
        held_rm = cache.compute_held(sel_rm)
        obj_rm = cache.compute_full_objective(sel_rm, held_rm, blm)
        assert abs(delta_rm - (obj_rm - obj_all)) < 1e-10

    def test_delta_various_blm(self, cache):
        rng = np.random.default_rng(77)
        for blm in [0.0, 0.5, 2.0, 5.0]:
            selected = rng.random(6) > 0.5
            held = cache.compute_held(selected)
            total_cost = float(np.sum(cache.costs[selected]))
            obj_before = cache.compute_full_objective(selected, held, blm)

            for idx in range(6):
                delta = cache.compute_delta_objective(
                    idx, selected, held, total_cost, blm
                )
                flipped = selected.copy()
                flipped[idx] = not flipped[idx]
                held_after = cache.compute_held(flipped)
                obj_after = cache.compute_full_objective(
                    flipped, held_after, blm
                )
                expected = obj_after - obj_before
                assert abs(delta - expected) < 1e-10, (
                    f"blm={blm}, idx={idx}: delta={delta}, expected={expected}"
                )

    def test_delta_all_selections(self, cache):
        """Exhaustive: test delta for every possible selection x every PU."""
        blm = 1.0
        n = 6
        for bits in range(2**n):
            selected = np.array(
                [(bits >> i) & 1 for i in range(n)], dtype=bool
            )
            held = cache.compute_held(selected)
            total_cost = float(np.sum(cache.costs[selected]))
            obj_before = cache.compute_full_objective(selected, held, blm)

            for idx in range(n):
                delta = cache.compute_delta_objective(
                    idx, selected, held, total_cost, blm
                )
                flipped = selected.copy()
                flipped[idx] = not flipped[idx]
                held_after = cache.compute_held(flipped)
                obj_after = cache.compute_full_objective(
                    flipped, held_after, blm
                )
                expected = obj_after - obj_before
                assert abs(delta - expected) < 1e-10, (
                    f"bits={bits:06b}, idx={idx}: "
                    f"delta={delta}, expected={expected}"
                )


# ------------------------------------------------------------------
# 5. Delta with COSTTHRESH parameters
# ------------------------------------------------------------------


class TestDeltaWithCostThresh:
    @pytest.fixture()
    def problem_ct(self, problem):
        """Problem with COSTTHRESH set."""
        problem.parameters["COSTTHRESH"] = 50.0
        problem.parameters["THRESHPEN1"] = 10.0
        problem.parameters["THRESHPEN2"] = 5.0
        return problem

    @pytest.fixture()
    def cache_ct(self, problem_ct):
        return ProblemCache.from_problem(problem_ct)

    @pytest.fixture()
    def pu_index_ct(self, problem_ct):
        pu_ids = problem_ct.planning_units["id"].tolist()
        return {pid: i for i, pid in enumerate(pu_ids)}

    def test_costthresh_scalars(self, cache_ct):
        assert cache_ct.cost_thresh == 50.0
        assert cache_ct.thresh_pen1 == 10.0
        assert cache_ct.thresh_pen2 == 5.0

    def test_full_objective_matches_with_costthresh(
        self, problem_ct, cache_ct, pu_index_ct
    ):
        rng = np.random.default_rng(55)
        blm = 1.0
        for _ in range(20):
            selected = rng.random(6) > 0.5
            held = cache_ct.compute_held(selected)
            cache_obj = cache_ct.compute_full_objective(selected, held, blm)
            ref_obj = compute_objective(
                problem_ct, selected, pu_index_ct, blm
            )
            assert abs(cache_obj - ref_obj) < 1e-10

    def test_delta_matches_with_costthresh(self, cache_ct):
        blm = 1.0
        rng = np.random.default_rng(66)
        for _ in range(10):
            selected = rng.random(6) > 0.5
            held = cache_ct.compute_held(selected)
            total_cost = float(np.sum(cache_ct.costs[selected]))
            obj_before = cache_ct.compute_full_objective(selected, held, blm)

            for idx in range(6):
                delta = cache_ct.compute_delta_objective(
                    idx, selected, held, total_cost, blm
                )
                flipped = selected.copy()
                flipped[idx] = not flipped[idx]
                held_after = cache_ct.compute_held(flipped)
                obj_after = cache_ct.compute_full_objective(
                    flipped, held_after, blm
                )
                expected = obj_after - obj_before
                assert abs(delta - expected) < 1e-10, (
                    f"COSTTHRESH delta mismatch at idx={idx}: "
                    f"delta={delta}, expected={expected}"
                )

    def test_delta_crosses_threshold(self, cache_ct):
        """Test delta when flipping a PU crosses the cost threshold."""
        blm = 1.0
        # Select PUs 1-4 (costs: 10+15+20+12 = 57 > 50 threshold)
        selected = np.array([True, True, True, True, False, False])
        held = cache_ct.compute_held(selected)
        total_cost = float(np.sum(cache_ct.costs[selected]))
        assert total_cost > 50.0  # Verify above threshold
        obj_before = cache_ct.compute_full_objective(selected, held, blm)

        # Remove PU 4 (cost 12): new cost = 45 < 50 => threshold penalty gone
        idx = 3
        delta = cache_ct.compute_delta_objective(
            idx, selected, held, total_cost, blm
        )
        flipped = selected.copy()
        flipped[idx] = False
        held_after = cache_ct.compute_held(flipped)
        obj_after = cache_ct.compute_full_objective(flipped, held_after, blm)
        expected = obj_after - obj_before
        assert abs(delta - expected) < 1e-10


# ------------------------------------------------------------------
# 6. Delta with MISSLEVEL
# ------------------------------------------------------------------


class TestDeltaWithMisslevel:
    @pytest.fixture()
    def problem_ml(self, problem):
        """Problem with MISSLEVEL < 1.0 (relaxed targets)."""
        problem.parameters["MISSLEVEL"] = 0.8
        return problem

    @pytest.fixture()
    def cache_ml(self, problem_ml):
        return ProblemCache.from_problem(problem_ml)

    @pytest.fixture()
    def pu_index_ml(self, problem_ml):
        pu_ids = problem_ml.planning_units["id"].tolist()
        return {pid: i for i, pid in enumerate(pu_ids)}

    def test_misslevel_scalar(self, cache_ml):
        assert cache_ml.misslevel == 0.8

    def test_full_objective_matches_with_misslevel(
        self, problem_ml, cache_ml, pu_index_ml
    ):
        rng = np.random.default_rng(88)
        blm = 1.0
        for _ in range(20):
            selected = rng.random(6) > 0.5
            held = cache_ml.compute_held(selected)
            cache_obj = cache_ml.compute_full_objective(selected, held, blm)
            ref_obj = compute_objective(
                problem_ml, selected, pu_index_ml, blm
            )
            assert abs(cache_obj - ref_obj) < 1e-10

    def test_delta_matches_with_misslevel(self, cache_ml):
        blm = 1.0
        rng = np.random.default_rng(44)
        for _ in range(10):
            selected = rng.random(6) > 0.5
            held = cache_ml.compute_held(selected)
            total_cost = float(np.sum(cache_ml.costs[selected]))
            obj_before = cache_ml.compute_full_objective(selected, held, blm)

            for idx in range(6):
                delta = cache_ml.compute_delta_objective(
                    idx, selected, held, total_cost, blm
                )
                flipped = selected.copy()
                flipped[idx] = not flipped[idx]
                held_after = cache_ml.compute_held(flipped)
                obj_after = cache_ml.compute_full_objective(
                    flipped, held_after, blm
                )
                expected = obj_after - obj_before
                assert abs(delta - expected) < 1e-10
