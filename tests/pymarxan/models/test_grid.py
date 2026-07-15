"""Tests for GridGeometry (raster-grid PUs, S1)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pymarxan.models.grid import GridGeometry


def test_n_pu_and_valid_cells_row_major():
    mask = np.ones((3, 3), dtype=bool)
    mask[1, 1] = False
    mask[2, 0] = False
    grid = GridGeometry(0.0, 3.0, 1.0, 1.0, mask)
    assert grid.n_pu == 7
    assert grid.shape == (3, 3)
    cells = grid.valid_cells()
    assert cells[0] == (0, 0)
    assert (1, 1) not in cells and (2, 0) not in cells
    assert cells == sorted(cells)  # row-major == sorted by (r, c)


def test_cell_centroids_2x2():
    grid = GridGeometry(0.0, 2.0, 1.0, 1.0, np.ones((2, 2), dtype=bool))
    # PU order (0,0),(0,1),(1,0),(1,1); centroid = (x_min+(c+.5)w, y_max-(r+.5)h)
    expected = np.array([[0.5, 1.5], [1.5, 1.5], [0.5, 0.5], [1.5, 0.5]])
    assert np.allclose(grid.cell_centroids(), expected)


def test_cell_bounds_2x2():
    grid = GridGeometry(0.0, 2.0, 1.0, 1.0, np.ones((2, 2), dtype=bool))
    # (0,0): minx0 miny1 maxx1 maxy2
    assert grid.cell_bounds()[0] == (0.0, 1.0, 1.0, 2.0)


def test_build_boundary_hand_computed_2x2():
    grid = GridGeometry(0.0, 2.0, 1.0, 1.0, np.ones((2, 2), dtype=bool))
    df = grid.build_boundary()  # ids 1..4
    shared = df[df.id1 != df.id2]
    self_rows = df[df.id1 == df.id2]
    assert len(shared) == 4 and set(shared.boundary) == {1.0}
    assert len(self_rows) == 4 and set(self_rows.boundary) == {2.0}


def test_single_cell_boundary():
    grid = GridGeometry(0.0, 1.0, 1.0, 1.0, np.ones((1, 1), dtype=bool))
    df = grid.build_boundary()
    assert len(df) == 1
    row = df.iloc[0]
    assert row.id1 == row.id2 and row.boundary == 4.0  # full perimeter


def test_build_boundary_len_guard():
    grid = GridGeometry(0.0, 2.0, 1.0, 1.0, np.ones((2, 2), dtype=bool))
    with pytest.raises(ValueError, match="entries"):
        grid.build_boundary(np.array([1, 2, 3]))  # 3 != 4


def test_build_boundary_duplicate_ids_guard():
    grid = GridGeometry(0.0, 2.0, 1.0, 1.0, np.ones((2, 2), dtype=bool))
    with pytest.raises(ValueError, match="unique"):
        grid.build_boundary(np.array([1, 2, 2, 3]))  # right length, dup id


def test_non_square_cells():
    # cell_width=2 (x), cell_height=3 (y); origin (10, 20). Guards the
    # direction->edge mapping + centroid/bounds axis scaling (a w/h swap here
    # WOULD change the assertions, unlike any w==h==1 test).
    grid = GridGeometry(10.0, 20.0, 2.0, 3.0, np.ones((2, 2), dtype=bool))
    # PU order (0,0),(0,1),(1,0),(1,1): x = 10 + (c+.5)*2, y = 20 - (r+.5)*3
    expected_c = np.array([[11.0, 18.5], [13.0, 18.5], [11.0, 15.5], [13.0, 15.5]])
    assert np.allclose(grid.cell_centroids(), expected_c)
    # (0,0) bounds: minx10 maxx12, maxy20 miny17
    assert grid.cell_bounds()[0] == (10.0, 17.0, 12.0, 20.0)
    df = grid.build_boundary()
    shared = df[df.id1 != df.id2]
    # horizontal (left-right) neighbors share a vertical edge = cell_height = 3;
    # vertical (up-down) neighbors share a horizontal edge = cell_width = 2.
    horiz = shared[shared.id1.isin([1, 3]) & shared.id2.isin([2, 4])]
    vert = shared[shared.id1.isin([1, 2]) & shared.id2.isin([3, 4])]
    assert set(horiz.boundary) == {3.0}  # NOT 2.0 — would fail on a w/h swap
    assert set(vert.boundary) == {2.0}


def test_validation():
    good = np.ones((2, 2), dtype=bool)
    with pytest.raises(ValueError, match="2-D boolean"):
        GridGeometry(0.0, 2.0, 1.0, 1.0, np.ones(4, dtype=bool))  # 1-D
    with pytest.raises(ValueError, match="2-D boolean"):
        GridGeometry(0.0, 2.0, 1.0, 1.0, np.ones((2, 2), dtype=int))  # not bool
    with pytest.raises(ValueError, match="> 0"):
        GridGeometry(0.0, 2.0, 0.0, 1.0, good)  # cell_width 0
    with pytest.raises(ValueError, match="no valid cells"):
        GridGeometry(0.0, 2.0, 1.0, 1.0, np.zeros((2, 2), dtype=bool))  # all False


@pytest.mark.spatial
def test_build_boundary_matches_shapely_full_grid():
    import geopandas as gpd
    from shapely.geometry import box

    from pymarxan.spatial.boundary import compute_boundary

    grid = GridGeometry(0.0, 3.0, 1.0, 1.0, np.ones((3, 3), dtype=bool))
    ids = np.arange(1, grid.n_pu + 1)
    gdf = gpd.GeoDataFrame({"id": ids}, geometry=[box(*b) for b in grid.cell_bounds()])
    analytic = grid.build_boundary(ids).sort_values(["id1", "id2"]).reset_index(drop=True)
    shapely_b = compute_boundary(gdf).sort_values(["id1", "id2"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(analytic, shapely_b, check_dtype=False)


@pytest.mark.spatial
def test_build_boundary_matches_shapely_masked_grid():
    import geopandas as gpd
    from shapely.geometry import box

    from pymarxan.spatial.boundary import compute_boundary

    mask = np.ones((3, 3), dtype=bool)
    mask[0, 0] = False  # remove a corner → exposed edges
    grid = GridGeometry(0.0, 3.0, 1.0, 1.0, mask)
    ids = np.arange(1, grid.n_pu + 1)
    gdf = gpd.GeoDataFrame({"id": ids}, geometry=[box(*b) for b in grid.cell_bounds()])
    analytic = grid.build_boundary(ids).sort_values(["id1", "id2"]).reset_index(drop=True)
    shapely_b = compute_boundary(gdf).sort_values(["id1", "id2"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(analytic, shapely_b, check_dtype=False)


@pytest.mark.spatial
def test_build_boundary_matches_shapely_center_hole():
    import geopandas as gpd
    from shapely.geometry import box

    from pymarxan.spatial.boundary import compute_boundary

    mask = np.ones((3, 3), dtype=bool)
    mask[1, 1] = False  # fully-surrounded masked-out cell → its 4 neighbors gain self-edge
    grid = GridGeometry(0.0, 3.0, 1.0, 1.0, mask)
    ids = np.arange(1, grid.n_pu + 1)
    gdf = gpd.GeoDataFrame({"id": ids}, geometry=[box(*b) for b in grid.cell_bounds()])
    analytic = grid.build_boundary(ids).sort_values(["id1", "id2"]).reset_index(drop=True)
    shapely_b = compute_boundary(gdf).sort_values(["id1", "id2"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(analytic, shapely_b, check_dtype=False)


@pytest.mark.spatial
def test_build_boundary_matches_shapely_non_square():
    import geopandas as gpd
    from shapely.geometry import box

    from pymarxan.spatial.boundary import compute_boundary

    grid = GridGeometry(10.0, 20.0, 2.0, 3.0, np.ones((3, 3), dtype=bool))  # w != h
    ids = np.arange(1, grid.n_pu + 1)
    gdf = gpd.GeoDataFrame({"id": ids}, geometry=[box(*b) for b in grid.cell_bounds()])
    analytic = grid.build_boundary(ids).sort_values(["id1", "id2"]).reset_index(drop=True)
    shapely_b = compute_boundary(gdf).sort_values(["id1", "id2"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(analytic, shapely_b, check_dtype=False)


def _tiny_problem():
    from pymarxan.models.problem import ConservationProblem

    pu = pd.DataFrame({"id": [1, 2], "cost": [1.0, 1.0], "status": [0, 0]})
    feats = pd.DataFrame({"id": [1], "name": ["a"], "target": [1.0], "spf": [1.0]})
    pvf = pd.DataFrame({"species": [1, 1], "pu": [1, 2], "amount": [1.0, 1.0]})
    return ConservationProblem(pu, feats, pvf)


def test_problem_grid_defaults_none():
    assert _tiny_problem().grid is None


def test_copy_with_and_clone_preserve_grid():
    g = GridGeometry(0.0, 2.0, 1.0, 1.0, np.ones((2, 2), dtype=bool))
    p = _tiny_problem().copy_with(grid=g)
    assert p.grid is g
    # preserved through a later copy_with that overrides an unrelated field
    assert p.copy_with(parameters={"BLM": 1.0}).grid is g
    # clone deep-copies (independent grid, still present)
    cloned = p.clone()
    assert cloned.grid is not None and cloned.grid is not g
    assert cloned.grid.n_pu == 4


def test_validate_grid_count_mismatch():
    # _tiny_problem has 2 PUs; a 2x2 grid (n_pu=4) disagrees -> validate error
    g4 = GridGeometry(0.0, 2.0, 1.0, 1.0, np.ones((2, 2), dtype=bool))
    errs = _tiny_problem().copy_with(grid=g4).validate()
    assert any("grid" in e and "planning" in e for e in errs)
    # a matching grid (n_pu == 2) yields no grid error
    g2 = GridGeometry(0.0, 2.0, 1.0, 1.0, np.array([[True, True]], dtype=bool))
    errs_ok = _tiny_problem().copy_with(grid=g2).validate()
    assert not any("grid" in e for e in errs_ok)


def _reference_build_boundary(grid, pu_ids=None):
    """The pre-vectorization per-cell loop — the multiset oracle."""
    cells = grid.valid_cells()
    n = len(cells)
    if pu_ids is None:
        pu_ids = np.arange(1, n + 1)
    pu_ids = np.asarray(pu_ids)
    cell_to_id = {cell: int(pu_ids[i]) for i, cell in enumerate(cells)}
    shared = {int(pid): 0.0 for pid in pu_ids}
    rows = []
    for (r, c), pid in cell_to_id.items():
        for nbr, edge in (((r, c + 1), grid.cell_height), ((r + 1, c), grid.cell_width)):
            nid = cell_to_id.get(nbr)
            if nid is not None:
                rows.append({"id1": pid, "id2": nid, "boundary": edge})
                shared[pid] += edge
                shared[nid] += edge
    perimeter = 2.0 * (grid.cell_width + grid.cell_height)
    for pid in cell_to_id.values():
        sb = perimeter - shared[pid]
        if sb > 1e-10:
            rows.append({"id1": pid, "id2": pid, "boundary": sb})
    return pd.DataFrame(rows, columns=["id1", "id2", "boundary"])


def _sorted(df):
    return df.sort_values(["id1", "id2"]).reset_index(drop=True)


def test_build_boundary_vectorized_matches_reference_loop():
    rng = np.random.default_rng(0)
    hole = np.ones((3, 3), dtype=bool)
    hole[1, 1] = False
    masked = rng.random((5, 5)) < 0.7
    masked[0, 0] = True  # guarantee at least one valid cell
    cases = [
        GridGeometry(0.0, 4.0, 1.0, 1.0, np.ones((4, 4), dtype=bool)),   # full
        GridGeometry(10.0, 20.0, 2.0, 3.0, masked),                     # non-square + holes
        GridGeometry(0.0, 1.0, 1.0, 1.0, np.ones((1, 5), dtype=bool)),  # 1xN strip
        GridGeometry(0.0, 5.0, 1.0, 1.0, np.ones((5, 1), dtype=bool)),  # Nx1 strip
        GridGeometry(0.0, 3.0, 1.0, 1.0, hole),                         # center hole
        GridGeometry(0.0, 1.0, 1.0, 1.0, np.ones((1, 1), dtype=bool)),  # single cell
        GridGeometry(0.0, 2.1, 0.3, 0.7, hole.copy()),                  # non-integer (ULP)
        GridGeometry(0.0, 3.0, 1.0, 1.0, np.asfortranarray(hole)),      # Fortran-order mask
    ]
    for grid in cases:
        got = _sorted(grid.build_boundary())
        ref = _sorted(_reference_build_boundary(grid))
        pd.testing.assert_frame_equal(got, ref, check_dtype=False)


def test_build_boundary_vectorized_arbitrary_pu_ids():
    grid = GridGeometry(0.0, 2.0, 1.0, 1.0, np.ones((2, 2), dtype=bool))
    ids = np.array([5, 3, 8, 1])  # non-sequential
    got = _sorted(grid.build_boundary(ids))
    ref = _sorted(_reference_build_boundary(grid, ids))
    pd.testing.assert_frame_equal(got, ref, check_dtype=False)


def test_build_boundary_scale_smoke():
    # 200x200 full grid: 2*200*199 shared rows + (4*200-4) border self rows.
    grid = GridGeometry(0.0, 200.0, 1.0, 1.0, np.ones((200, 200), dtype=bool))
    df = grid.build_boundary()
    assert len(df) == 2 * 200 * 199 + (4 * 200 - 4)
