"""Tests for restoptr-style MESH (effective mesh size, Jaeger 2000)."""
from __future__ import annotations

import numpy as np
import pytest

from pymarxan.models.grid import GridGeometry
from pymarxan.restoration import MeshResult, compute_mesh


def _grid(nrow=3, ncol=3, mask=None):
    if mask is None:
        mask = np.ones((nrow, ncol), dtype=bool)
    return GridGeometry(x_min=0.0, y_max=float(nrow), cell_width=1.0, cell_height=1.0, mask=mask)


def test_full_habitat_single_patch_is_max():
    g = _grid(3, 3)  # A_total = 9, cell_area = 1
    r = compute_mesh(g, np.ones(9, bool), cell_area=1.0)
    assert isinstance(r, MeshResult)
    assert r.n_patches == 1
    assert r.total_area == 9.0
    assert r.mesh == pytest.approx(9.0)  # 9^2 / 9 = 9 == A_total (max)


def test_no_habitat_is_zero():
    g = _grid(3, 3)
    r = compute_mesh(g, np.zeros(9, bool), cell_area=1.0)
    assert r.n_patches == 0
    assert r.mesh == 0.0
    assert list(r.patch_areas) == []


def test_two_separated_single_cells():
    # habitat at opposite corners (0,0) and (2,2): non-adjacent under rook -> 2 patches.
    g = _grid(3, 3)
    mask = np.zeros(9, bool)
    mask[0] = True   # cell (0,0), row-major index 0
    mask[8] = True   # cell (2,2), row-major index 8
    r = compute_mesh(g, mask, cell_area=1.0)
    assert r.n_patches == 2
    assert r.mesh == pytest.approx((1 + 1) / 9)  # 0.2222


def test_block_plus_isolated_cell():
    # 4-cell connected block (top-left 2x2) + 1 isolated cell (2,2). areas 4 and 1.
    g = _grid(3, 3)
    mask = np.zeros(9, bool)
    for i in (0, 1, 3, 4):  # (0,0),(0,1),(1,0),(1,1) -> connected 2x2 block
        mask[i] = True
    mask[8] = True  # (2,2) isolated
    r = compute_mesh(g, mask, cell_area=1.0)
    assert r.n_patches == 2
    assert sorted(r.patch_areas, reverse=True) == [4.0, 1.0]
    assert r.mesh == pytest.approx((16 + 1) / 9)  # 1.8889


def test_rook_vs_queen_diagonal():
    # two diagonally-touching cells: 2 patches under rook, 1 under queen.
    g = _grid(2, 2)
    mask = np.array([True, False, False, True])  # (0,0) and (1,1)
    assert compute_mesh(g, mask, connectivity="rook", cell_area=1.0).n_patches == 2
    assert compute_mesh(g, mask, connectivity="queen", cell_area=1.0).n_patches == 1


def test_mesh_scales_with_cell_area():
    g = _grid(3, 3)
    mask = np.ones(9, bool)
    base = compute_mesh(g, mask, cell_area=1.0).mesh
    scaled = compute_mesh(g, mask, cell_area=4.0).mesh
    assert scaled == pytest.approx(4.0 * base)  # MESH linear in cell_area


def test_default_cell_area_from_grid():
    # cell 2x3 -> cell_area 6; full 2x2 grid, one patch: A_total = 4*6 = 24, mesh = 24.
    g = GridGeometry(x_min=0.0, y_max=6.0, cell_width=2.0, cell_height=3.0,
                     mask=np.ones((2, 2), bool))
    r = compute_mesh(g, np.ones(4, bool))
    assert r.total_area == pytest.approx(24.0)
    assert r.mesh == pytest.approx(24.0)


def test_monotone_bridging_increases_mesh():
    # 1x3 strip: two ends habitat (2 patches) -> add the middle -> 1 patch, MESH increases.
    g = _grid(1, 3)
    ends = np.array([True, False, True])
    bridged = np.array([True, True, True])
    assert compute_mesh(g, bridged, cell_area=1.0).mesh > compute_mesh(g, ends, cell_area=1.0).mesh


def test_masked_nonrectangular_grid_mapping():
    # invalid centre cell; habitat on the 8 border cells, all rook-connected around the hole
    # -> 1 ring patch. Confirms habitat_mask maps to the right 2-D positions.
    m2d = np.ones((3, 3), bool)
    m2d[1, 1] = False  # centre invalid -> 8 valid PUs
    g = _grid(3, 3, mask=m2d)
    r = compute_mesh(g, np.ones(8, bool), cell_area=1.0)
    assert r.total_area == 8.0
    assert r.n_patches == 1
    assert r.mesh == pytest.approx(64.0 / 8.0)  # 8^2 / 8 = 8


def test_coherence_and_division_properties():
    # full 3x3 one patch: C = mesh/total = 9/9 = 1, D = 0.
    g = _grid(3, 3)
    full = compute_mesh(g, np.ones(9, bool), cell_area=1.0)
    assert full.coherence == pytest.approx(1.0)
    assert full.division == pytest.approx(0.0)
    # two isolated corners: C = Σ(A_i/A_total)² = 2·(1/9)² = 2/81 = mesh/total_area.
    mask = np.zeros(9, bool)
    mask[0] = mask[8] = True
    frag = compute_mesh(g, mask, cell_area=1.0)
    assert frag.coherence == pytest.approx(2 / 81)  # (2/9) / 9
    assert frag.division == pytest.approx(1.0 - 2 / 81)


def test_wrong_length_mask_raises():
    g = _grid(3, 3)
    with pytest.raises(ValueError):
        compute_mesh(g, np.ones(5, bool))


def test_unknown_connectivity_raises():
    g = _grid(3, 3)
    with pytest.raises(ValueError):
        compute_mesh(g, np.ones(9, bool), connectivity="diagonal")


def test_nonpositive_cell_area_raises():
    g = _grid(3, 3)
    with pytest.raises(ValueError):
        compute_mesh(g, np.ones(9, bool), cell_area=0.0)
    with pytest.raises(ValueError):
        compute_mesh(g, np.ones(9, bool), cell_area=-2.0)
