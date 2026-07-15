"""Tests for raster-grid ingestion (S2)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pymarxan.spatial.raster import from_arrays


def test_from_arrays_basic_roundtrip():
    f1 = np.array([[1.0, 2.0, 0.0], [0.0, 3.0, 4.0], [5.0, 0.0, 6.0]])
    f2 = np.array([[0.0, 0.0, 1.0], [2.0, 0.0, 0.0], [0.0, 3.0, 0.0]])
    p = from_arrays({1: f1, 2: f2}, x_min=0, y_max=3, cell_width=1, cell_height=1)
    assert len(p.planning_units) == 9
    assert list(p.planning_units["id"]) == list(range(1, 10))
    assert p.grid is not None and p.grid.n_pu == 9
    assert p.validate() == []  # a constructed grid problem is valid
    m = p.build_pu_feature_matrix()  # (9, 2); columns sorted feature ids [1, 2]
    assert np.allclose(m[:, 0], f1.ravel())  # 0-amount cells fill 0 in the dense matrix
    assert np.allclose(m[:, 1], f2.ravel())


def test_validity_feature_union_drops_nan():
    f1 = np.array([[1.0, np.nan], [np.nan, 2.0]])
    p = from_arrays({1: f1}, x_min=0, y_max=2, cell_width=1, cell_height=1)
    assert p.grid.n_pu == 2  # only (0,0) and (1,1) have data
    assert list(p.planning_units["id"]) == [1, 2]


def test_validity_cost_footprint_over_union():
    f1 = np.ones((2, 2))  # every cell has feature data
    cost = np.array([[5.0, np.nan], [np.nan, 7.0]])  # cost defines the study area
    p = from_arrays({1: f1}, x_min=0, y_max=2, cell_width=1, cell_height=1, cost_array=cost)
    assert p.grid.n_pu == 2
    assert list(p.planning_units["cost"]) == [5.0, 7.0]


def test_validity_mask_over_cost_and_features():
    f1 = np.ones((2, 2))
    cost = np.ones((2, 2))
    mask = np.array([[1, 0], [0, 0]])
    p = from_arrays(
        {1: f1}, x_min=0, y_max=2, cell_width=1, cell_height=1,
        cost_array=cost, mask_array=mask,
    )
    assert p.grid.n_pu == 1


def test_sparse_zero_and_nodata_no_row():
    f1 = np.array([[0.0, 5.0], [np.nan, 2.0]])  # union validity drops (1,0)
    p = from_arrays({1: f1}, x_min=0, y_max=2, cell_width=1, cell_height=1)
    assert p.grid.n_pu == 3  # (0,0),(0,1),(1,1)
    assert len(p.pu_vs_features) == 2  # (0,0)=0 → no row; 5 and 2 → rows
    assert set(p.pu_vs_features["amount"]) == {5.0, 2.0}


def test_cost_default_one_and_status():
    f1 = np.ones((2, 2))
    status = np.array([[0, 2], [0, 3]])
    p = from_arrays(
        {1: f1}, x_min=0, y_max=2, cell_width=1, cell_height=1, status_array=status,
    )
    assert list(p.planning_units["cost"]) == [1.0, 1.0, 1.0, 1.0]
    assert list(p.planning_units["status"]) == [0, 2, 0, 3]


def test_cost_nodata_in_mask_warns_and_defaults():
    # mask admits a cell the cost layer marks nodata -> warn + default cost 1.0
    f1 = np.ones((2, 2))
    cost = np.array([[5.0, np.nan], [7.0, 9.0]])
    mask = np.array([[1, 1], [0, 0]])  # (0,0) and (0,1) valid; (0,1) cost is nodata
    with pytest.warns(UserWarning, match="nodata cost"):
        p = from_arrays(
            {1: f1}, x_min=0, y_max=2, cell_width=1, cell_height=1,
            cost_array=cost, mask_array=mask,
        )
    assert list(p.planning_units["cost"]) == [5.0, 1.0]  # nodata cost -> 1.0


def test_holey_mask_cross_layer_alignment():
    # Non-symmetric validity + distinct cost + distinct per-feature values: catches any
    # cross-layer transpose/misindex. Mask keeps (0,0),(1,2),(2,1) (row-major PU order).
    mask = np.zeros((3, 3), dtype=int)
    mask[0, 0] = 1
    mask[1, 2] = 1
    mask[2, 1] = 1
    cost = np.arange(9, dtype=float).reshape(3, 3)  # cost = row-major index
    f1 = (np.arange(9, dtype=float).reshape(3, 3) + 1) * 10  # (idx+1)*10
    p = from_arrays(
        {1: f1}, x_min=0, y_max=3, cell_width=1, cell_height=1,
        cost_array=cost, mask_array=mask,
    )
    # valid cells row-major: (0,0)=idx0, (1,2)=idx5, (2,1)=idx7
    assert list(p.planning_units["cost"]) == [0.0, 5.0, 7.0]
    amt = dict(zip(p.pu_vs_features["pu"], p.pu_vs_features["amount"]))
    assert amt == {1: 10.0, 2: 60.0, 3: 80.0}  # (idx+1)*10 at those cells


def test_invalid_status_raises():
    f1 = np.ones((2, 2))
    status = np.array([[0, 7], [0, 0]])
    with pytest.raises(ValueError, match="status"):
        from_arrays(
            {1: f1}, x_min=0, y_max=2, cell_width=1, cell_height=1, status_array=status,
        )


def test_noninteger_status_raises():
    f1 = np.ones((2, 2))
    status = np.array([[0.0, 2.7], [0.0, 0.0]])
    with pytest.raises(ValueError, match="status"):
        from_arrays(
            {1: f1}, x_min=0, y_max=2, cell_width=1, cell_height=1, status_array=status,
        )


def test_shape_mismatch_raises():
    with pytest.raises(ValueError, match="shape"):
        from_arrays(
            {1: np.ones((2, 2)), 2: np.ones((3, 3))},
            x_min=0, y_max=2, cell_width=1, cell_height=1,
        )


def test_boundary_wired_and_toggle():
    f1 = np.ones((2, 2))
    p = from_arrays({1: f1}, x_min=0, y_max=2, cell_width=1, cell_height=1)
    assert p.boundary is not None
    pd.testing.assert_frame_equal(
        p.boundary.reset_index(drop=True),
        p.grid.build_boundary(np.array([1, 2, 3, 4])).reset_index(drop=True),
    )
    p2 = from_arrays(
        {1: f1}, x_min=0, y_max=2, cell_width=1, cell_height=1, include_boundary=False,
    )
    assert p2.boundary is None


def test_feature_names():
    f = {1: np.ones((1, 2)), 2: np.ones((1, 2))}
    p = from_arrays(
        f, x_min=0, y_max=1, cell_width=1, cell_height=1, feature_names={1: "seagrass"},
    )
    assert dict(zip(p.features["id"], p.features["name"])) == {1: "seagrass", 2: "feature_2"}


def test_empty_study_area_raises():
    f1 = np.full((2, 2), np.nan)
    with pytest.raises(ValueError, match="valid"):
        from_arrays({1: f1}, x_min=0, y_max=2, cell_width=1, cell_height=1)


def test_empty_features_raises():
    with pytest.raises(ValueError, match="non-empty"):
        from_arrays({}, x_min=0, y_max=1, cell_width=1, cell_height=1)


from affine import Affine  # noqa: E402  (grouped with the wrapper tests below)

from pymarxan.spatial.raster import from_rasters  # noqa: E402

# 3x3 grid: x_min=0, y_max=3, cell 1x1, north-up.
REF_TF = Affine(1.0, 0.0, 0.0, 0.0, -1.0, 3.0)


def _write(tmp_path, name, array, *, transform=REF_TF, crs="EPSG:3035", nodata=None):
    import rasterio

    array = np.asarray(array, dtype="float32")
    count = 1 if array.ndim == 2 else array.shape[0]
    height, width = array.shape[-2], array.shape[-1]
    path = tmp_path / name
    with rasterio.open(
        path, "w", driver="GTiff", height=height, width=width, count=count,
        dtype="float32", crs=crs, transform=transform, nodata=nodata,
    ) as ds:
        if array.ndim == 2:
            ds.write(array, 1)
        else:
            ds.write(array)
    return path


@pytest.mark.spatial
def test_from_rasters_single_band_matches_from_arrays(tmp_path):
    f1 = np.array([[1, 2, 0], [0, 3, 4], [5, 0, 6]], dtype="float32")
    f2 = np.array([[0, 0, 1], [2, 0, 0], [0, 3, 0]], dtype="float32")
    p = from_rasters({1: _write(tmp_path, "f1.tif", f1), 2: _write(tmp_path, "f2.tif", f2)})
    ref = from_arrays(
        {1: f1.astype(float), 2: f2.astype(float)},
        x_min=0, y_max=3, cell_width=1, cell_height=1,
    )
    assert list(p.planning_units["id"]) == list(ref.planning_units["id"])
    assert np.allclose(p.build_pu_feature_matrix(), ref.build_pu_feature_matrix())
    assert p.grid is not None and p.grid.n_pu == 9


@pytest.mark.spatial
def test_from_rasters_multiband_tuple(tmp_path):
    f1 = np.array([[1, 2, 0], [0, 3, 4], [5, 0, 6]], dtype="float32")
    f2 = np.array([[0, 0, 1], [2, 0, 0], [0, 3, 0]], dtype="float32")
    stack = np.stack([f1, f2])  # (2, 3, 3)
    path = _write(tmp_path, "stack.tif", stack)
    p = from_rasters({1: (path, 1), 2: (path, 2)})
    ref = from_arrays(
        {1: f1.astype(float), 2: f2.astype(float)},
        x_min=0, y_max=3, cell_width=1, cell_height=1,
    )
    assert np.allclose(p.build_pu_feature_matrix(), ref.build_pu_feature_matrix())


@pytest.mark.spatial
def test_from_rasters_nodata_to_nan(tmp_path):
    f1 = np.array([[1, -9999, 3], [4, 5, 6], [7, 8, 9]], dtype="float32")
    path = _write(tmp_path, "f.tif", f1, nodata=-9999)
    p = from_rasters({1: path})
    assert p.grid.n_pu == 8  # the -9999 cell is dropped


@pytest.mark.spatial
def test_from_rasters_cost_and_status(tmp_path):
    f1 = np.ones((3, 3), dtype="float32")
    cost = np.full((3, 3), 4.0, dtype="float32")
    status = np.zeros((3, 3), dtype="float32")
    status[0, 0] = 2
    p = from_rasters(
        {1: _write(tmp_path, "f.tif", f1)},
        cost_raster=_write(tmp_path, "cost.tif", cost),
        status_raster=_write(tmp_path, "status.tif", status),
    )
    assert set(p.planning_units["cost"]) == {4.0}
    assert p.planning_units.iloc[0]["status"] == 2


@pytest.mark.spatial
def test_from_rasters_transform_mismatch_raises(tmp_path):
    a = _write(tmp_path, "a.tif", np.ones((3, 3), dtype="float32"))
    b = _write(tmp_path, "b.tif", np.ones((3, 3), dtype="float32"),
               transform=Affine(2.0, 0.0, 0.0, 0.0, -2.0, 6.0))
    with pytest.raises(ValueError, match="transform"):
        from_rasters({1: a, 2: b})


@pytest.mark.spatial
def test_from_rasters_crs_mismatch_raises(tmp_path):
    a = _write(tmp_path, "a.tif", np.ones((3, 3), dtype="float32"), crs="EPSG:3035")
    b = _write(tmp_path, "b.tif", np.ones((3, 3), dtype="float32"), crs="EPSG:4326")
    with pytest.raises(ValueError, match="CRS"):
        from_rasters({1: a, 2: b})


@pytest.mark.spatial
def test_from_rasters_rotated_transform_raises(tmp_path):
    rot = Affine(1.0, 0.5, 0.0, 0.5, -1.0, 3.0)  # b, d non-zero
    a = _write(tmp_path, "a.tif", np.ones((3, 3), dtype="float32"), transform=rot)
    with pytest.raises(ValueError, match="rotat|north"):
        from_rasters({1: a})


@pytest.mark.spatial
def test_from_rasters_south_up_transform_raises(tmp_path):
    south_up = Affine(1.0, 0.0, 100.0, 0.0, 1.0, 0.0)  # e >= 0 (non-identity → georeferenced)
    a = _write(tmp_path, "a.tif", np.ones((3, 3), dtype="float32"), transform=south_up)
    with pytest.raises(ValueError, match="north-up"):
        from_rasters({1: a})


@pytest.mark.spatial
def test_from_rasters_aligned_large_origin_no_false_reject(tmp_path):
    # Two truly-aligned rasters at a projected origin (~4.3e6) whose origin differs by a
    # sub-micron amount must NOT be rejected (the tolerance scales to cell size); a real
    # half-cell shift MUST be rejected.
    tf = Affine(100.0, 0.0, 4_321_000.0, 0.0, -100.0, 3_210_000.0)
    tf_eps = Affine(100.0, 0.0, 4_321_000.0000005, 0.0, -100.0, 3_210_000.0)
    tf_shift = Affine(100.0, 0.0, 4_321_050.0, 0.0, -100.0, 3_210_000.0)  # +0.5 cell
    a = _write(tmp_path, "a.tif", np.ones((3, 3), dtype="float32"), transform=tf)
    b = _write(tmp_path, "b.tif", np.ones((3, 3), dtype="float32"), transform=tf_eps)
    p = from_rasters({1: a, 2: b})  # must not raise
    assert p.grid.n_pu == 9
    c = _write(tmp_path, "c.tif", np.ones((3, 3), dtype="float32"), transform=tf_shift)
    with pytest.raises(ValueError, match="transform"):
        from_rasters({1: a, 2: c})


@pytest.mark.spatial
def test_from_rasters_no_crs_builds(tmp_path):
    # All rasters lack a CRS → build succeeds, grid.crs is None.
    a = _write(tmp_path, "a.tif", np.ones((2, 2), dtype="float32"), crs=None,
               transform=Affine(1.0, 0.0, 0.0, 0.0, -1.0, 2.0))
    p = from_rasters({1: a})
    assert p.grid.n_pu == 4 and p.grid.crs is None
