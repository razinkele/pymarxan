"""Tests for greedy_mesh_restore (MESH-maximizing restoration optimizer)."""
from __future__ import annotations

import numpy as np
import pytest

from pymarxan.models.grid import GridGeometry
from pymarxan.restoration import (
    MeshRestorationResult,
    RestorationProblem,
    compute_mesh,
    greedy_mesh_restore,
)


def _grid(nrow=1, ncol=5):
    return GridGeometry(x_min=0.0, y_max=float(nrow), cell_width=1.0, cell_height=1.0,
                        mask=np.ones((nrow, ncol), dtype=bool))


def _strip_problem(cost=None):
    # 1x5 strip; ends (0,4) already habitat (2 patches); middle 3 (1,2,3) restorable.
    g = _grid(1, 5)
    existing = np.array([True, False, False, False, True])
    restorable = np.array([False, True, True, True, False])
    return RestorationProblem(g, existing, restorable, cost)


def test_returns_result():
    rp = _strip_problem()
    res = greedy_mesh_restore(rp, budget=1.0)
    assert isinstance(res, MeshRestorationResult)
    assert res.restored.dtype == bool
    assert res.n_restored == 1
    assert res.total_cost == pytest.approx(1.0)


def test_plan_is_subset_of_restorable_and_roundtrips():
    rp = _strip_problem()
    res = greedy_mesh_restore(rp, budget=2.0)
    assert not (res.restored & ~rp.restorable).any()  # ⊆ restorable
    hm = rp.existing_habitat | res.restored
    assert res.mesh == pytest.approx(compute_mesh(rp.grid, hm).mesh)
    assert res.baseline_mesh == pytest.approx(rp.baseline_mesh().mesh)


def test_greedy_picks_bridge_cell_first():
    # ends habitat; restoring cell 1 extends the left end-patch {0}->{0,1} (area 2), tied with
    # cell 3; tie -> lowest index 1. cell 2 (isolated) gains less.
    rp = _strip_problem()
    res = greedy_mesh_restore(rp, budget=1.0)
    assert list(np.flatnonzero(res.restored)) == [1]


def test_budget_fills_uniform_cost():
    rp = _strip_problem()
    res = greedy_mesh_restore(rp, budget=3.0)  # uniform cost 1 -> all 3 middle cells
    assert res.n_restored == 3
    # whole strip is one patch -> maximum MESH
    assert res.mesh == pytest.approx(compute_mesh(rp.grid, np.ones(5, bool)).mesh)


def test_mesh_curve_monotone_and_shaped():
    rp = _strip_problem()
    res = greedy_mesh_restore(rp, budget=2.0)
    curve = res.mesh_curve
    assert curve[0] == pytest.approx(res.baseline_mesh)
    assert curve[-1] == pytest.approx(res.mesh)
    assert len(curve) == res.n_restored + 1
    assert np.all(np.diff(curve) >= -1e-12)  # non-decreasing


def test_cost_curve_and_order():
    # non-uniform cost -> cost_curve is cumulative spend (not step count), order = pick sequence.
    cost = np.array([1.0, 2.0, 3.0, 4.0, 1.0])
    rp = _strip_problem(cost=cost)
    res = greedy_mesh_restore(rp, budget=10.0)
    assert res.cost_curve[0] == 0.0
    assert res.cost_curve[-1] == pytest.approx(res.total_cost)
    assert len(res.cost_curve) == res.n_restored + 1
    assert np.all(np.diff(res.cost_curve) >= -1e-12)  # non-decreasing cumulative cost
    assert len(res.order) == res.n_restored
    # reconstruct the plan from order -> matches restored mask
    recon = np.zeros(rp.n_pu, bool)
    recon[res.order] = True
    assert np.array_equal(recon, res.restored)
    # cumulative cost matches the order's costs
    assert res.cost_curve[-1] == pytest.approx(cost[res.order].sum())


def test_cost_budget_honored_nonuniform():
    cost = np.array([1.0, 5.0, 1.0, 5.0, 1.0])  # only cell 2 affordable at budget 1
    rp = _strip_problem(cost=cost)
    res = greedy_mesh_restore(rp, budget=1.0)
    assert res.total_cost <= 1.0
    assert list(np.flatnonzero(res.restored)) == [2]


def test_gain_per_cost_vs_gain_differ():
    # cell 1 (cost 4) biggest raw gain but poor per-cost; budget 4.
    cost = np.array([1.0, 4.0, 1.0, 1.0, 1.0])
    rp = _strip_problem(cost=cost)
    by_gain = greedy_mesh_restore(rp, budget=4.0, criterion="gain")
    by_ratio = greedy_mesh_restore(rp, budget=4.0, criterion="gain_per_cost")
    assert list(np.flatnonzero(by_gain.restored)) == [1]
    assert list(np.flatnonzero(by_ratio.restored)) == [2, 3]


def test_zero_cost_cell_restored_first():
    cost = np.array([1.0, 1.0, 0.0, 1.0, 1.0])  # cell 2 free
    rp = _strip_problem(cost=cost)
    res = greedy_mesh_restore(rp, budget=0.0)  # only the free cell is affordable
    assert list(np.flatnonzero(res.restored)) == [2]
    assert res.total_cost == pytest.approx(0.0)


def test_budget_zero_empty_plan():
    rp = _strip_problem(cost=np.ones(5))
    res = greedy_mesh_restore(rp, budget=0.0)
    assert res.n_restored == 0
    assert res.mesh == pytest.approx(res.baseline_mesh)
    assert list(res.mesh_curve) == [pytest.approx(res.baseline_mesh)]
    assert list(res.cost_curve) == [0.0]
    assert res.order == []


def test_no_restorable_cells():
    g = _grid(1, 3)
    rp = RestorationProblem(g, np.array([True, False, True]), np.zeros(3, bool))
    res = greedy_mesh_restore(rp, budget=10.0)
    assert res.n_restored == 0


def test_negative_budget_and_bad_criterion_raise():
    rp = _strip_problem()
    with pytest.raises(ValueError):
        greedy_mesh_restore(rp, budget=-1.0)
    with pytest.raises(ValueError):
        greedy_mesh_restore(rp, budget=1.0, criterion="nope")
