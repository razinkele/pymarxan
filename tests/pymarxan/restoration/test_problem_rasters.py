"""from_rasters ingestion for RestorationProblem (rasterio round-trip)."""
from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.spatial
rasterio = pytest.importorskip("rasterio")
from rasterio.transform import from_origin  # noqa: E402

from pymarxan.restoration import RestorationProblem  # noqa: E402


def _write(path, arr, *, nodata=None):
    arr = np.asarray(arr, dtype="float32")
    tf = from_origin(0.0, arr.shape[0], 1.0, 1.0)  # x_min=0, y_max=nrow, 1x1 cells, north-up
    with rasterio.open(path, "w", driver="GTiff", height=arr.shape[0], width=arr.shape[1],
                       count=1, dtype="float32", transform=tf, nodata=nodata) as dst:
        dst.write(arr, 1)


def test_from_rasters_matches_from_arrays(tmp_path):
    existing = np.array([[1, 0, 0], [0, 0, 0], [0, 0, 1]], dtype="float32")
    restorable = np.array([[0, 1, 1], [1, 1, 1], [1, 1, 0]], dtype="float32")
    ep, rpath = tmp_path / "e.tif", tmp_path / "r.tif"
    _write(ep, existing)
    _write(rpath, restorable)
    rp = RestorationProblem.from_rasters(str(ep), str(rpath))
    ref = RestorationProblem.from_arrays(existing.astype(float), restorable.astype(float),
                                         x_min=0.0, y_max=3.0, cell_width=1.0, cell_height=1.0)
    assert rp.n_pu == ref.n_pu == 9
    assert np.array_equal(rp.existing_habitat, ref.existing_habitat)
    assert np.array_equal(rp.restorable, ref.restorable)
    assert rp.grid.cell_width == 1.0 and rp.grid.cell_height == 1.0
    assert rp.grid.x_min == 0.0 and rp.grid.y_max == 3.0


def test_from_rasters_with_cost(tmp_path):
    existing = np.zeros((2, 2), dtype="float32")
    restorable = np.ones((2, 2), dtype="float32")
    cost = np.array([[1, 2], [3, 4]], dtype="float32")
    for name, arr in (("e", existing), ("r", restorable), ("c", cost)):
        _write(tmp_path / f"{name}.tif", arr)
    rp = RestorationProblem.from_rasters(str(tmp_path / "e.tif"), str(tmp_path / "r.tif"),
                                         cost=str(tmp_path / "c.tif"))
    assert list(rp.cost) == [1.0, 2.0, 3.0, 4.0]


def test_from_rasters_misaligned_raises(tmp_path):
    _write(tmp_path / "e.tif", np.ones((3, 3), "float32"))
    _write(tmp_path / "r.tif", np.ones((2, 2), "float32"))  # wrong shape
    with pytest.raises(ValueError):
        RestorationProblem.from_rasters(str(tmp_path / "e.tif"), str(tmp_path / "r.tif"))
