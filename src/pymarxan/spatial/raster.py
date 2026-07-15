"""Raster-grid ingestion for grid-based planning units (S2).

``from_arrays`` is the pure-numpy core (aligned arrays → ConservationProblem with a
GridGeometry); ``from_rasters`` is the thin rasterio wrapper (Task 2).
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from affine import Affine
from rasterio.crs import CRS

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
    return np.asarray(m, dtype=bool)


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
                f"status values must be integers in {sorted(_VALID_STATUS)}, "
                f"got {sorted(set(sv.tolist()))}"
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
