# Raster-grid PUs — S2 implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a `ConservationProblem` (carrying a S1 `GridGeometry`) directly from aligned rasters/arrays — the grid-native ingestion path.

**Architecture:** A pure-numpy `from_arrays(...)` core holds all ingestion logic (validity mask → grid → PU table → sparse `pu_vs_features` → analytic boundary). A thin `from_rasters(...)` wrapper reads rasters with rasterio, normalizes nodata → NaN, guards axis-aligned north-up alignment, and delegates to the core. No model change (S1 added the `grid` field); no solver change.

**Tech Stack:** Python 3.12+, NumPy, pandas, rasterio (already an optional dep).

**Design spec:** `docs/plans/2026-07-15-raster-grid-s2-design.md`. Scoping: `docs/plans/2026-07-15-raster-grid-pus-scoping.md`. S1 model: `src/pymarxan/models/grid.py`.

## Global Constraints

- Python 3.12+, `from __future__ import annotations`, full type hints.
- No new third-party dependency (rasterio already declared).
- Tests **must** run under the `shiny` micromamba env: `/opt/micromamba/envs/shiny/bin/pytest`.
- Lint: ruff (E, F, I, UP; line length 99). Types: mypy clean. Coverage ≥ 75%.
- The bar before done: `make check` green.
- **Positional-alignment contract (S1):** valid-cell row-major order == PU order == `build_pu_feature_matrix` row order == `build_boundary` id order. `planning_units.id` = `1..n_pu` in that order.
- **Reprojection deferred:** misaligned/rotated/non-north-up rasters raise, they are not warped.
- `ConservationProblem(planning_units, features, pu_vs_features, boundary=None, parameters=..., grid=<kw_only>)`.

## File Structure

- Create: `src/pymarxan/spatial/raster.py` — `from_arrays` + `from_rasters` (+ private helpers).
- Modify: `src/pymarxan/spatial/__init__.py` — re-export both.
- Create: `tests/pymarxan/spatial/test_raster.py`.
- Modify: `CHANGELOG.md`.

---

### Task 1: `from_arrays` (pure numpy core)

**Files:**
- Create: `src/pymarxan/spatial/raster.py`
- Test: `tests/pymarxan/spatial/test_raster.py`

**Interfaces:**
- Produces: `from_arrays(feature_arrays: dict[int, np.ndarray], *, x_min, y_max, cell_width, cell_height, crs=None, cost_array=None, status_array=None, mask_array=None, feature_names=None, nodata=None, include_boundary=True) -> ConservationProblem`.

- [ ] **Step 1: Write the failing tests**

Create `tests/pymarxan/spatial/test_raster.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `/opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/spatial/test_raster.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'pymarxan.spatial.raster'`.

- [ ] **Step 3: Implement `from_arrays`**

Create `src/pymarxan/spatial/raster.py`:

```python
"""Raster-grid ingestion for grid-based planning units (S2).

``from_arrays`` is the pure-numpy core (aligned arrays → ConservationProblem with a
GridGeometry); ``from_rasters`` is the thin rasterio wrapper (Task 2).
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from pymarxan.models.grid import GridGeometry
from pymarxan.models.problem import ConservationProblem

_VALID_STATUS = frozenset({0, 1, 2, 3})


def _nodata_mask(arr: np.ndarray, nodata: float | None) -> np.ndarray:
    """Boolean array, True where ``arr`` is nodata (NaN always; ``== nodata`` sentinel)."""
    if np.issubdtype(arr.dtype, np.floating):
        m = np.isnan(arr)
    else:
        m = np.zeros(arr.shape, dtype=bool)
    if nodata is not None:
        m = m | (arr == nodata)
    return m


def from_arrays(
    feature_arrays: dict[int, np.ndarray],
    *,
    x_min: float,
    y_max: float,
    cell_width: float,
    cell_height: float,
    crs: str | None = None,
    cost_array: np.ndarray | None = None,
    status_array: np.ndarray | None = None,
    mask_array: np.ndarray | None = None,
    feature_names: dict[int, str] | None = None,
    nodata: float | None = None,
    include_boundary: bool = True,
) -> ConservationProblem:
    """Build a ConservationProblem (with a GridGeometry) from aligned numpy arrays.

    Valid grid cells (see the validity precedence) become planning units in row-major
    order, ids ``1..n_pu``. Feature amounts must be non-negative; values ``<= 0`` (and
    nodata) yield no ``pu_vs_features`` row (the dense matrix restores 0). The generated
    ``features`` table uses placeholder ``target=0.0`` / ``spf=1.0`` — set real targets
    afterwards (e.g. via the feature-override machinery) before solving. See the S2 design
    spec for full semantics.
    """
    if not feature_arrays:
        raise ValueError("feature_arrays must be non-empty")

    feat_ids = sorted(feature_arrays)
    feats = {fid: np.asarray(feature_arrays[fid], dtype=float) for fid in feat_ids}
    shape = feats[feat_ids[0]].shape
    if len(shape) != 2:
        raise ValueError(f"feature arrays must be 2-D, got shape {shape}")

    # --- shape check (all arrays share one shape) ---
    for fid in feat_ids:
        if feats[fid].shape != shape:
            raise ValueError(f"feature array {fid} shape {feats[fid].shape} != {shape}")
    extras = {"cost_array": cost_array, "status_array": status_array, "mask_array": mask_array}
    for name, a in extras.items():
        if a is not None and np.asarray(a).shape != shape:
            raise ValueError(f"{name} shape {np.asarray(a).shape} != feature shape {shape}")

    # --- validity mask: mask -> cost -> feature-union ---
    if mask_array is not None:
        m = np.asarray(mask_array, dtype=float)
        valid = (m != 0) & ~_nodata_mask(m, nodata)
    elif cost_array is not None:
        c = np.asarray(cost_array, dtype=float)
        valid = ~_nodata_mask(c, nodata)
    else:
        valid = np.zeros(shape, dtype=bool)
        for a in feats.values():
            valid = valid | ~_nodata_mask(a, nodata)
    if not valid.any():
        raise ValueError("no valid cells (the validity mask is empty)")

    grid = GridGeometry(x_min, y_max, cell_width, cell_height, valid, crs)
    n_pu = grid.n_pu
    pu_ids = np.arange(1, n_pu + 1)
    rows, cols = np.nonzero(valid)  # row-major == PU order == valid_cells()

    # --- cost ---
    if cost_array is not None:
        c = np.asarray(cost_array, dtype=float)
        cost_vals = c[rows, cols]
        cost_nd = _nodata_mask(c, nodata)[rows, cols]
        if cost_nd.any():
            # Only reachable when a mask/feature-union admits a cell the cost layer
            # marks nodata (in the cost-footprint case those cells are already excluded).
            warnings.warn(
                f"{int(cost_nd.sum())} valid cell(s) have nodata cost; defaulting to 1.0",
                stacklevel=2,
            )
        cost_vals = np.where(cost_nd, 1.0, cost_vals)
    else:
        cost_vals = np.ones(n_pu, dtype=float)

    # --- status (integer-valued, in {0,1,2,3}; nodata -> 0) ---
    status_vals = np.zeros(n_pu, dtype=int)
    if status_array is not None:
        s = np.asarray(status_array, dtype=float)
        s_valid = s[rows, cols]
        real = ~_nodata_mask(s, nodata)[rows, cols]
        sv = s_valid[real]
        iv = np.round(sv).astype(int)
        if np.any(iv != sv) or not set(iv.tolist()).issubset(_VALID_STATUS):
            raise ValueError(
                f"status values must be integers in {sorted(_VALID_STATUS)}, got {sorted(set(sv.tolist()))}"
            )
        status_vals[real] = iv

    planning_units = pd.DataFrame({"id": pu_ids, "cost": cost_vals, "status": status_vals})

    # --- features table ---
    names = feature_names or {}
    features = pd.DataFrame(
        {
            "id": feat_ids,
            "name": [names.get(fid, f"feature_{fid}") for fid in feat_ids],
            "target": [0.0] * len(feat_ids),
            "spf": [1.0] * len(feat_ids),
        }
    )

    # --- pu_vs_features (sparse long form: row per (cell, feature) with amount > 0) ---
    frames: list[pd.DataFrame] = []
    for fid in feat_ids:
        a = feats[fid]
        vals = a[rows, cols]
        keep = (~_nodata_mask(a, nodata)[rows, cols]) & (vals > 0)
        if keep.any():
            frames.append(
                pd.DataFrame({"species": fid, "pu": pu_ids[keep], "amount": vals[keep]})
            )
    if frames:
        pu_vs_features = pd.concat(frames, ignore_index=True)
    else:
        pu_vs_features = pd.DataFrame(columns=["species", "pu", "amount"])

    boundary = grid.build_boundary(pu_ids) if include_boundary else None

    return ConservationProblem(
        planning_units, features, pu_vs_features, boundary=boundary, grid=grid
    )
```

(The `rasterio` / `affine` / `pathlib` imports belong to `from_rasters` and are added in Task 2, so Task 1 has no unused imports.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `/opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/spatial/test_raster.py -q`
Expected: PASS (15 tests).

- [ ] **Step 5: Commit**

```bash
git add src/pymarxan/spatial/raster.py tests/pymarxan/spatial/test_raster.py
git commit -m "feat(spatial): from_arrays — build a grid ConservationProblem from aligned numpy arrays"
```

---

### Task 2: `from_rasters` wrapper + exports + CHANGELOG

**Files:**
- Modify: `src/pymarxan/spatial/raster.py`
- Modify: `src/pymarxan/spatial/__init__.py`
- Test: `tests/pymarxan/spatial/test_raster.py` (append)
- Modify: `CHANGELOG.md`

**Interfaces:**
- Consumes: `from_arrays` (Task 1).
- Produces: `from_rasters(feature_rasters: dict[int, str | Path | tuple[str | Path, int]], *, cost_raster=None, status_raster=None, mask_raster=None, feature_names=None, include_boundary=True) -> ConservationProblem`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/pymarxan/spatial/test_raster.py`:

```python
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
```

- [ ] **Step 2: Run them to verify they fail**

Run: `/opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/spatial/test_raster.py -k "from_rasters" -q`
Expected: FAIL — `from_rasters` not importable / not defined.

- [ ] **Step 3: Implement `from_rasters` + helpers**

First add the wrapper's imports to the top of `src/pymarxan/spatial/raster.py` (isort
order: stdlib `pathlib`, then third-party `affine`/`rasterio`, before the first-party
`pymarxan.*` block):

```python
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from affine import Affine
from rasterio.crs import CRS

from pymarxan.models.grid import GridGeometry
from pymarxan.models.problem import ConservationProblem
```

Then append the wrapper + helpers:

```python
def _transforms_close(a: Affine, b: Affine, tol: float = 1e-6) -> bool:
    # Scale the tolerance to cell size: at projected-CRS origins (easting ~1e6-1e7)
    # one float64 ULP already exceeds a fixed 1e-9, so a fixed absolute tolerance would
    # false-reject two truly-aligned rasters whose origin was recomputed differently.
    scale = max(abs(a.a), abs(a.e), 1.0)
    return all(abs(x - y) <= tol * scale for x, y in zip(a[:6], b[:6]))


def _require_north_up(tf: Affine) -> None:
    if abs(tf.b) > 1e-12 or abs(tf.d) > 1e-12:
        raise ValueError(
            "rotated/sheared rasters are not supported (transform has non-zero rotation/shear)"
        )
    if tf.a <= 0 or tf.e >= 0:
        raise ValueError(
            "only axis-aligned north-up rasters are supported "
            "(need transform.a > 0 and transform.e < 0)"
        )


def _read(
    path: str | Path, band: int
) -> tuple[np.ndarray, Affine, tuple[int, int], CRS | None]:
    with rasterio.open(path) as src:
        arr = src.read(band).astype(float)
        if src.nodata is not None:
            arr = np.where(arr == src.nodata, np.nan, arr)
        return arr, src.transform, (src.height, src.width), src.crs


def _read_aligned(
    path: str | Path, band: int, label: str,
    ref_tf: Affine, ref_shape: tuple[int, int], ref_crs: CRS | None,
) -> np.ndarray:
    arr, tf, shp, crs = _read(path, band)
    if shp != ref_shape:
        raise ValueError(f"{label} shape {shp} != reference {ref_shape}")
    if not _transforms_close(tf, ref_tf):
        raise ValueError(f"{label} transform {tf!r} != reference {ref_tf!r}")
    if crs != ref_crs:
        raise ValueError(f"{label} CRS {crs} != reference {ref_crs}")
    return arr


def from_rasters(
    feature_rasters: dict[int, str | Path | tuple[str | Path, int]],
    *,
    cost_raster: str | Path | None = None,
    status_raster: str | Path | None = None,
    mask_raster: str | Path | None = None,
    feature_names: dict[int, str] | None = None,
    include_boundary: bool = True,
) -> ConservationProblem:
    """Build a ConservationProblem from aligned rasters (thin wrapper over from_arrays).

    ``feature_rasters`` values are a single-band path or a ``(path, band)`` tuple (bands
    are 1-indexed). All rasters must share the reference raster's grid (transform, shape,
    CRS) — misaligned/rotated/non-north-up rasters raise; they are not warped.
    """
    if not feature_rasters:
        raise ValueError("feature_rasters must be non-empty")

    def _spec(v: str | Path | tuple[str | Path, int]) -> tuple[str | Path, int]:
        if isinstance(v, tuple):
            return v[0], int(v[1])
        return v, 1

    feat_ids = sorted(feature_rasters)

    # Reference grid from the first feature raster.
    first_path, first_band = _spec(feature_rasters[feat_ids[0]])
    first_arr, ref_tf, ref_shape, ref_crs = _read(first_path, first_band)
    _require_north_up(ref_tf)

    feature_arrays: dict[int, np.ndarray] = {feat_ids[0]: first_arr}
    for fid in feat_ids[1:]:
        path, band = _spec(feature_rasters[fid])
        feature_arrays[fid] = _read_aligned(
            path, band, f"feature {fid}", ref_tf, ref_shape, ref_crs
        )

    cost_array = (
        _read_aligned(cost_raster, 1, "cost_raster", ref_tf, ref_shape, ref_crs)
        if cost_raster is not None else None
    )
    status_array = (
        _read_aligned(status_raster, 1, "status_raster", ref_tf, ref_shape, ref_crs)
        if status_raster is not None else None
    )
    mask_array = (
        _read_aligned(mask_raster, 1, "mask_raster", ref_tf, ref_shape, ref_crs)
        if mask_raster is not None else None
    )

    return from_arrays(
        feature_arrays,
        x_min=ref_tf.c,
        y_max=ref_tf.f,
        cell_width=ref_tf.a,
        cell_height=-ref_tf.e,
        crs=ref_crs.to_string() if ref_crs is not None else None,
        cost_array=cost_array,
        status_array=status_array,
        mask_array=mask_array,
        feature_names=feature_names,
        include_boundary=include_boundary,
    )
```

- [ ] **Step 4: Run the wrapper tests to verify they pass**

Run: `/opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/spatial/test_raster.py -q`
Expected: PASS (25 tests — 15 core + 10 wrapper).

- [ ] **Step 5: Export from `spatial/__init__.py`**

Add the import (isort: `pymarxan.spatial.raster` sorts after `.importers`, before `.wdpa`) and the two names to `__all__`:

```python
from pymarxan.spatial.raster import from_arrays, from_rasters
```

Add `"from_arrays"` and `"from_rasters"` to the `__all__` list (keep it alphabetically sorted, as the existing list is).

- [ ] **Step 6: Add the CHANGELOG entry**

Under `## [Unreleased]` → `### Added` in `CHANGELOG.md` (create the headers if empty):

```markdown
- **Raster-grid ingestion (`from_rasters` / `from_arrays`, S2).** ``spatial/raster.py``
  builds a ``ConservationProblem`` (carrying a S1 ``GridGeometry``) directly from aligned
  rasters — a pure-NumPy ``from_arrays`` core plus a thin rasterio ``from_rasters`` wrapper.
  One feature container (``dict[int, path | (path, band)]``) covers separate files and a
  multi-band stack; validity precedence mask → cost → feature-union; sparse
  ``pu_vs_features``; the analytic boundary is wired in by default. Rotated / non-north-up /
  misaligned rasters raise (reprojection deferred). +25 tests.
```

- [ ] **Step 7: Run the full check**

Run: `PATH="/opt/micromamba/envs/shiny/bin:$HOME/.local/bin:$PWD/.venv/bin:$PATH" make check`
Expected: green — 0 ruff, 0 mypy, full suite + 25 new. (`test_solutions_are_different` flake → rerun once.)

Note: the CLAUDE.md `micromamba.sh` activation path may not exist on this machine; the `PATH=...` prefix above is the working invocation.

- [ ] **Step 8: Commit**

```bash
git add src/pymarxan/spatial/raster.py src/pymarxan/spatial/__init__.py \
        tests/pymarxan/spatial/test_raster.py CHANGELOG.md
git commit -m "feat(spatial): from_rasters — grid ConservationProblem from aligned rasters (S2)"
```

---

## Post-plan notes

- **Design review:** run `multi-agent-design-review` on this plan before/at execution — the parity-relevant surface is the nodata/validity precedence, the sparse `pu_vs_features` (amount > 0), and the transform→grid-scalar mapping (`cell_height = -transform.e`).
- **Parity:** ingestion only; no solver/objective math. The round-trip anchor (`build_pu_feature_matrix` == input arrays at valid cells) + S1 boundary parity keep the constructed problem faithful; the 35.0 min-set anchor is untouched (full suite).
- **Deferred (own specs):** S3 scale (sparse matrices, windowed reads, heuristic-only path), S4 UI/mapping + `has_geometry`. `exactextract` + reprojection remain a later fast-path.
- **`tests/pymarxan/spatial/` exists?** If not, the first test-file write creates it; add an `__init__.py` if the sibling test dirs have one.
```
