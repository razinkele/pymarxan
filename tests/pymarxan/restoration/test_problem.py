"""Tests for RestorationProblem (restoration data model)."""
from __future__ import annotations

import numpy as np
import pytest

from pymarxan.models.grid import GridGeometry
from pymarxan.restoration import MeshResult, RestorationProblem, compute_mesh


def _grid(nrow=3, ncol=3, mask=None):
    if mask is None:
        mask = np.ones((nrow, ncol), dtype=bool)
    return GridGeometry(x_min=0.0, y_max=float(nrow), cell_width=1.0, cell_height=1.0, mask=mask)


def _rp(existing, restorable, cost=None, nrow=3, ncol=3):
    return RestorationProblem(_grid(nrow, ncol), np.asarray(existing, bool),
                              np.asarray(restorable, bool), cost)


def test_cost_defaults_to_uniform():
    rp = _rp(np.zeros(9), np.ones(9))
    assert rp.cost.shape == (9,)
    assert np.all(rp.cost == 1.0)


def test_n_pu():
    assert _rp(np.zeros(9), np.ones(9)).n_pu == 9


def test_restorable_indices():
    restorable = np.array([0, 1, 0, 1, 1, 0, 0, 0, 0], bool)
    rp = _rp(np.zeros(9), restorable)
    assert list(rp.restorable_indices) == [1, 3, 4]


def test_habitat_mask_unions_existing_and_restored():
    existing = np.zeros(9, bool)
    existing[0] = True
    restorable = np.zeros(9, bool)
    restorable[1] = restorable[2] = True
    rp = _rp(existing, restorable)
    restored = np.zeros(9, bool)
    restored[1] = True
    hm = rp.habitat_mask(restored)
    assert hm[0] and hm[1] and not hm[2]


def test_habitat_mask_rejects_non_restorable():
    rp = _rp(np.zeros(9), np.array([1, 1, 0, 0, 0, 0, 0, 0, 0], bool))
    bad = np.zeros(9, bool)
    bad[3] = True  # cell 3 is not restorable
    with pytest.raises(ValueError):
        rp.habitat_mask(bad)


def test_habitat_mask_wrong_length_raises():
    rp = _rp(np.zeros(9), np.ones(9))
    with pytest.raises(ValueError):
        rp.habitat_mask(np.zeros(5, bool))


def test_restore_mesh_matches_compute_mesh():
    existing = np.zeros(9, bool)
    existing[0] = True
    restorable = np.ones(9, bool)
    restorable[0] = False  # cell 0 already habitat
    rp = _rp(existing, restorable)
    restored = np.zeros(9, bool)
    restored[1] = restored[3] = True
    r = rp.restore_mesh(restored)
    assert isinstance(r, MeshResult)
    expected = compute_mesh(rp.grid, rp.habitat_mask(restored))
    assert r.mesh == pytest.approx(expected.mesh)


def test_baseline_mesh_is_existing_only():
    existing = np.ones(9, bool)
    rp = _rp(existing, np.zeros(9))
    assert rp.baseline_mesh().mesh == pytest.approx(compute_mesh(rp.grid, existing).mesh)


def test_restore_increases_mesh_over_baseline():
    # 1x3 strip, ends habitat (2 patches), middle restorable -> restoring bridges -> MESH up.
    g = _grid(1, 3)
    existing = np.array([True, False, True])
    restorable = np.array([False, True, False])
    rp = RestorationProblem(g, existing, restorable)
    restored = np.array([False, True, False])
    assert rp.restore_mesh(restored).mesh > rp.baseline_mesh().mesh


def test_restoration_cost_sums_selected():
    cost = np.arange(9, dtype=float)  # 0..8
    rp = _rp(np.zeros(9), np.ones(9), cost=cost)
    restored = np.zeros(9, bool)
    restored[2] = restored[5] = True
    assert rp.restoration_cost(restored) == pytest.approx(7.0)  # 2 + 5


def test_validate_clean():
    existing = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0], bool)
    restorable = np.array([0, 1, 1, 0, 0, 0, 0, 0, 0], bool)
    assert _rp(existing, restorable).validate() == []


def test_validate_flags_overlap():
    both = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0], bool)
    errs = _rp(both, both).validate()
    assert any("disjoint" in e.lower() or "overlap" in e.lower() for e in errs)


def test_validate_flags_wrong_length_and_bad_cost():
    g = _grid(3, 3)
    rp = RestorationProblem(g, np.zeros(5, bool), np.ones(9, bool))  # existing wrong length
    assert rp.validate()  # non-empty
    rp2 = RestorationProblem(g, np.zeros(9, bool), np.ones(9, bool), np.full(9, -1.0))
    assert any("cost" in e.lower() for e in rp2.validate())


# --- from_arrays ---

def test_from_arrays_basic():
    existing = np.array([[1, 0, 0], [0, 0, 0], [0, 0, 1]], dtype=float)
    restorable = np.array([[0, 1, 1], [1, 1, 1], [1, 1, 0]], dtype=float)
    rp = RestorationProblem.from_arrays(existing, restorable, x_min=0.0, y_max=3.0,
                                        cell_width=1.0, cell_height=1.0)
    assert rp.n_pu == 9
    assert rp.existing_habitat.dtype == bool
    assert rp.existing_habitat.sum() == 2  # cells (0,0) and (2,2)
    assert rp.restorable.sum() == 7
    assert np.all(rp.cost == 1.0)
    assert rp.validate() == []  # existing/restorable disjoint by construction here


def test_from_arrays_validity_from_existing_footprint_with_nodata():
    nan = np.nan
    existing = np.array([[1, 0, nan], [0, 0, 0], [nan, 0, 1]], dtype=float)  # 2 nodata -> 7 valid
    restorable = np.zeros((3, 3), dtype=float)
    rp = RestorationProblem.from_arrays(existing, restorable, x_min=0.0, y_max=3.0,
                                        cell_width=1.0, cell_height=1.0, nodata=None)
    assert rp.n_pu == 7  # nodata cells dropped from the study region


def test_from_arrays_explicit_mask_precedence():
    existing = np.ones((2, 2), dtype=float)
    restorable = np.array([[0, 0], [0, 0]], dtype=float)  # no restorable outside the mask
    mask = np.array([[1, 1], [0, 1]], dtype=float)  # 3 valid
    rp = RestorationProblem.from_arrays(existing, restorable, x_min=0.0, y_max=2.0,
                                        cell_width=1.0, cell_height=1.0, mask_array=mask)
    assert rp.n_pu == 3


def test_from_arrays_warns_on_dropped_restorable():
    # restorable data on a cell that is nodata in existing_habitat -> dropped + warned (M1).
    existing = np.array([[1, 0], [0, np.nan]], dtype=float)  # (1,1) nodata -> outside study region
    restorable = np.array([[0, 1], [1, 1]], dtype=float)     # (1,1) restorable but dropped
    with pytest.warns(UserWarning, match="outside the study region"):
        rp = RestorationProblem.from_arrays(existing, restorable, x_min=0.0, y_max=2.0,
                                            cell_width=1.0, cell_height=1.0)
    assert rp.n_pu == 3


def test_from_arrays_cost_default_and_shape_check():
    existing = np.zeros((2, 2), dtype=float)
    restorable = np.ones((2, 2), dtype=float)
    rp = RestorationProblem.from_arrays(existing, restorable, x_min=0.0, y_max=2.0,
                                        cell_width=1.0, cell_height=1.0)
    assert np.all(rp.cost == 1.0)
    with pytest.raises(ValueError):
        RestorationProblem.from_arrays(existing, np.ones((3, 3)), x_min=0.0, y_max=2.0,
                                       cell_width=1.0, cell_height=1.0)


def test_from_arrays_empty_mask_raises():
    existing = np.full((2, 2), np.nan)
    with pytest.raises(ValueError):
        RestorationProblem.from_arrays(existing, np.zeros((2, 2)), x_min=0.0, y_max=2.0,
                                       cell_width=1.0, cell_height=1.0)


def test_inline_nodata_mask_matches_s2():
    # M2 regression: the inlined _nodata_mask must match spatial.raster._nodata_mask behaviourally.
    from pymarxan.restoration.problem import _nodata_mask as inline_nd
    from pymarxan.spatial.raster import _nodata_mask as s2_nd
    for arr, nd in (
        (np.array([1.0, np.nan, 3.0]), None),
        (np.array([1.0, 2.0, -9999.0]), -9999.0),
        (np.array([1, 2, 3], dtype=int), 2),
    ):
        assert np.array_equal(inline_nd(arr, nd), s2_nd(arr, nd))
