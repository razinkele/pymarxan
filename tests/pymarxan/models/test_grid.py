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
