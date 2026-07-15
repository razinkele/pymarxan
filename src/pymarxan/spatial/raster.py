"""Raster-grid ingestion for grid-based planning units (S2 + S3c).

``from_arrays`` is the pure-numpy core (aligned arrays → ConservationProblem with a
GridGeometry); ``from_rasters`` is the rasterio wrapper, with a windowed (tiled) read
path (S3c) for large rasters selected by ``window_size``.
"""
from __future__ import annotations

import warnings
from contextlib import ExitStack
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import rasterio
from affine import Affine
from rasterio.crs import CRS
from rasterio.windows import Window

from pymarxan.models.grid import GridGeometry
from pymarxan.models.problem import ConservationProblem

_VALID_STATUS = frozenset({0, 1, 2, 3})
_DEFAULT_TILE = 1024
_WINDOW_AUTO_BYTES = 512 * 1024 * 1024  # 512 MiB dense-stack threshold for "auto"


def _spec(v: str | Path | tuple[str | Path, int]) -> tuple[str | Path, int]:
    """Normalize a feature-raster spec to ``(path, band)`` (bands are 1-indexed)."""
    if isinstance(v, tuple):
        return v[0], int(v[1])
    return v, 1


def _nodata_mask(arr: np.ndarray, nodata: float | None) -> np.ndarray:
    """Boolean array, True where ``arr`` is nodata (NaN always; ``== nodata`` sentinel)."""
    if np.issubdtype(arr.dtype, np.floating):
        m = np.isnan(arr)
    else:
        m = np.zeros(arr.shape, dtype=bool)
    if nodata is not None:
        m = m | (arr == nodata)
    return np.asarray(m, dtype=bool)


def _validate_status_ints(sv_real: np.ndarray) -> np.ndarray:
    """Round + validate non-nodata status values (must be integer-valued in {0,1,2,3})."""
    iv = np.round(sv_real).astype(int)
    if np.any(iv != sv_real) or not set(iv.tolist()).issubset(_VALID_STATUS):
        raise ValueError(
            f"status values must be integers in {sorted(_VALID_STATUS)}, "
            f"got {sorted(set(sv_real.tolist()))}"
        )
    return iv


def _features_table(feat_ids: list[int], feature_names: dict[int, str] | None) -> pd.DataFrame:
    """Placeholder features table (target=0.0, spf=1.0 — set real targets afterwards)."""
    names = feature_names or {}
    return pd.DataFrame(
        {
            "id": feat_ids,
            "name": [names.get(fid, f"feature_{fid}") for fid in feat_ids],
            "target": [0.0] * len(feat_ids),
            "spf": [1.0] * len(feat_ids),
        }
    )


def _assemble_problem(
    *,
    x_min: float,
    y_max: float,
    cell_width: float,
    cell_height: float,
    crs: str | None,
    mask: np.ndarray,
    feat_ids: list[int],
    feature_names: dict[int, str] | None,
    cost_vals: np.ndarray,
    status_vals: np.ndarray,
    pvf_frames: list[pd.DataFrame],
    include_boundary: bool,
) -> ConservationProblem:
    """Build the ConservationProblem from per-PU arrays (shared by both ingest paths)."""
    grid = GridGeometry(x_min, y_max, cell_width, cell_height, mask, crs)
    pu_ids = np.arange(1, grid.n_pu + 1)
    planning_units = pd.DataFrame({"id": pu_ids, "cost": cost_vals, "status": status_vals})
    features = _features_table(feat_ids, feature_names)
    if pvf_frames:
        pu_vs_features = pd.concat(pvf_frames, ignore_index=True)
    else:
        pu_vs_features = pd.DataFrame(columns=["species", "pu", "amount"])
    boundary = grid.build_boundary(pu_ids) if include_boundary else None
    return ConservationProblem(
        planning_units, features, pu_vs_features, boundary=boundary, grid=grid
    )


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

    rows, cols = np.nonzero(valid)  # row-major == PU order == valid_cells()
    n_pu = rows.size
    pu_ids = np.arange(1, n_pu + 1)

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
        real = ~_nodata_mask(s, nodata)[rows, cols]
        status_vals[real] = _validate_status_ints(s[rows, cols][real])

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

    return _assemble_problem(
        x_min=x_min, y_max=y_max, cell_width=cell_width, cell_height=cell_height,
        crs=crs, mask=valid, feat_ids=feat_ids, feature_names=feature_names,
        cost_vals=cost_vals, status_vals=status_vals, pvf_frames=frames,
        include_boundary=include_boundary,
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


def _check_align(
    label: str, shp: tuple[int, int], tf: Affine, crs: CRS | None,
    ref_tf: Affine, ref_shape: tuple[int, int], ref_crs: CRS | None,
) -> None:
    """Alignment check shared by the full-array and windowed paths."""
    if shp != ref_shape:
        raise ValueError(f"{label} shape {shp} != reference {ref_shape}")
    if not _transforms_close(tf, ref_tf):
        raise ValueError(f"{label} transform {tf!r} != reference {ref_tf!r}")
    if crs != ref_crs:
        raise ValueError(f"{label} CRS {crs} != reference {ref_crs}")


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
    _check_align(label, shp, tf, crs, ref_tf, ref_shape, ref_crs)
    return arr


def _tiles(height: int, width: int, tile: int):
    """Yield row-major Window tiles of side ``tile`` (edge tiles clamped)."""
    for row_off in range(0, height, tile):
        h = min(tile, height - row_off)
        for col_off in range(0, width, tile):
            w = min(tile, width - col_off)
            yield Window(col_off, row_off, w, h)


def _read_win(src: rasterio.DatasetReader, band: int, win: Window) -> np.ndarray:
    """Read one window as float with the source nodata normalized to NaN (as ``_read``)."""
    arr: np.ndarray = src.read(band, window=win).astype(float)
    if src.nodata is not None:
        arr = np.where(arr == src.nodata, np.nan, arr)
    return arr


def _check_meta(
    src: rasterio.DatasetReader, label: str,
    ref_tf: Affine, ref_shape: tuple[int, int], ref_crs: CRS | None,
) -> None:
    """Metadata-only alignment check (no read)."""
    _check_align(label, (src.height, src.width), src.transform, src.crs,
                 ref_tf, ref_shape, ref_crs)


def _resolve_windowed(
    first_path: str | Path, n_feat: int,
    window_size: int | Literal["auto"] | None,
) -> tuple[bool, int]:
    """Return (windowed, tile). ``None`` -> full; positive int -> windowed(that tile);
    ``"auto"`` -> windowed(default tile) when the dense stack exceeds the threshold."""
    if window_size is None:
        return False, _DEFAULT_TILE
    if isinstance(window_size, bool):
        raise ValueError("window_size must be an int, 'auto', or None; got a bool")
    if isinstance(window_size, int):
        if window_size < 1:
            raise ValueError("window_size must be a positive int")
        return True, window_size
    if window_size != "auto":
        raise ValueError(f"window_size must be an int, 'auto', or None; got {window_size!r}")
    with rasterio.open(first_path) as src:
        dense_bytes = src.height * src.width * n_feat * 8
    return (dense_bytes > _WINDOW_AUTO_BYTES), _DEFAULT_TILE


def from_rasters(
    feature_rasters: dict[int, str | Path | tuple[str | Path, int]],
    *,
    cost_raster: str | Path | None = None,
    status_raster: str | Path | None = None,
    mask_raster: str | Path | None = None,
    feature_names: dict[int, str] | None = None,
    include_boundary: bool = True,
    window_size: int | Literal["auto"] | None = "auto",
) -> ConservationProblem:
    """Build a ConservationProblem from aligned rasters (wrapper over from_arrays).

    ``feature_rasters`` values are a single-band path or a ``(path, band)`` tuple (bands
    are 1-indexed). All rasters must share the reference raster's grid (transform, shape,
    CRS) — misaligned/rotated/non-north-up rasters raise; they are not warped.

    ``window_size`` (``int | "auto" | None``, default ``"auto"``) selects the ingest path:
    ``None`` reads each raster whole; a positive int reads in square tiles of that side;
    ``"auto"`` reads windowed when the estimated dense stack exceeds a threshold.
    ``include_boundary`` defaults to ``True`` on both paths (the analytic ``build_boundary``
    is O(n)); pass ``include_boundary=False`` to skip it (e.g. a no-BLM run at extreme scale).
    """
    if not feature_rasters:
        raise ValueError("feature_rasters must be non-empty")
    feat_ids = sorted(feature_rasters)
    first_path, first_band = _spec(feature_rasters[feat_ids[0]])

    # --- S3c switch: windowed vs full-array ---
    windowed, tile = _resolve_windowed(first_path, len(feat_ids), window_size)
    if windowed:
        return _from_rasters_windowed(
            feature_rasters, feat_ids, tile,
            cost_raster=cost_raster, status_raster=status_raster, mask_raster=mask_raster,
            feature_names=feature_names, include_boundary=include_boundary,
        )

    # --- full-array path ---
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


def _from_rasters_windowed(
    feature_rasters: dict[int, str | Path | tuple[str | Path, int]],
    feat_ids: list[int],
    tile: int,
    *,
    cost_raster: str | Path | None,
    status_raster: str | Path | None,
    mask_raster: str | Path | None,
    feature_names: dict[int, str] | None,
    include_boundary: bool,
) -> ConservationProblem:
    """Windowed (tiled) ingestion — same ConservationProblem as the full-array path."""
    with ExitStack() as stack:
        # --- open + align-check all rasters (metadata only) ---
        first_path, first_band = _spec(feature_rasters[feat_ids[0]])
        first_src = stack.enter_context(rasterio.open(first_path))
        _require_north_up(first_src.transform)
        ref_tf, ref_crs = first_src.transform, first_src.crs
        height, width = first_src.height, first_src.width
        ref_shape = (height, width)

        feat_src: dict[int, tuple[rasterio.DatasetReader, int]] = {
            feat_ids[0]: (first_src, first_band)
        }
        for fid in feat_ids[1:]:
            path, band = _spec(feature_rasters[fid])
            src = stack.enter_context(rasterio.open(path))
            _check_meta(src, f"feature {fid}", ref_tf, ref_shape, ref_crs)
            feat_src[fid] = (src, band)

        def _open_extra(path: str | Path | None, label: str):
            if path is None:
                return None
            src = stack.enter_context(rasterio.open(path))
            _check_meta(src, label, ref_tf, ref_shape, ref_crs)
            return src

        cost_src = _open_extra(cost_raster, "cost_raster")
        status_src = _open_extra(status_raster, "status_raster")
        mask_src = _open_extra(mask_raster, "mask_raster")

        # --- Pass 1: validity mask ---
        mask = np.zeros((height, width), dtype=bool)
        for win in _tiles(height, width, tile):
            sl = (slice(int(win.row_off), int(win.row_off) + int(win.height)),
                  slice(int(win.col_off), int(win.col_off) + int(win.width)))
            if mask_src is not None:
                m = _read_win(mask_src, 1, win)
                valid_w = (m != 0) & ~np.isnan(m)
            elif cost_src is not None:
                valid_w = ~np.isnan(_read_win(cost_src, 1, win))
            else:
                valid_w = np.zeros((int(win.height), int(win.width)), dtype=bool)
                for src, band in feat_src.values():
                    valid_w |= ~np.isnan(_read_win(src, band, win))
            mask[sl] = valid_w
        flat_valid = np.flatnonzero(mask.ravel())  # int64, ascending == PU order
        n_pu = int(flat_valid.size)
        if n_pu == 0:
            raise ValueError("no valid cells (the validity mask is empty)")

        # --- Pass 2: cost / status / feature rows ---
        cost_vals = np.ones(n_pu, dtype=float)
        status_vals = np.zeros(n_pu, dtype=int)
        cost_nd_count = 0
        frames_by_feat: dict[int, list[pd.DataFrame]] = {fid: [] for fid in feat_ids}
        for win in _tiles(height, width, tile):
            sl = (slice(int(win.row_off), int(win.row_off) + int(win.height)),
                  slice(int(win.col_off), int(win.col_off) + int(win.width)))
            win_mask = mask[sl]
            if not win_mask.any():
                continue
            vr, vc = np.nonzero(win_mask)
            gflat = (np.int64(win.row_off) + vr) * np.int64(width) + (np.int64(win.col_off) + vc)
            pu_idx = np.searchsorted(flat_valid, gflat)  # 0-based row-major rank
            ids_w = pu_idx + 1

            if cost_src is not None:
                cv = _read_win(cost_src, 1, win)[vr, vc]
                nd = np.isnan(cv)
                cost_nd_count += int(nd.sum())
                cost_vals[pu_idx] = np.where(nd, 1.0, cv)
            if status_src is not None:
                sv = _read_win(status_src, 1, win)[vr, vc]
                real = ~np.isnan(sv)
                if real.any():
                    status_vals[pu_idx[real]] = _validate_status_ints(sv[real])
            for fid, (src, band) in feat_src.items():
                av = _read_win(src, band, win)[vr, vc]
                keep = (~np.isnan(av)) & (av > 0)
                if keep.any():
                    frames_by_feat[fid].append(
                        pd.DataFrame({"species": fid, "pu": ids_w[keep], "amount": av[keep]})
                    )

        if cost_nd_count:
            warnings.warn(
                f"{cost_nd_count} valid cell(s) have nodata cost; defaulting to 1.0",
                stacklevel=2,
            )

        frames = [df for fid in feat_ids for df in frames_by_feat[fid]]
        return _assemble_problem(
            x_min=ref_tf.c, y_max=ref_tf.f, cell_width=ref_tf.a, cell_height=-ref_tf.e,
            crs=ref_crs.to_string() if ref_crs is not None else None,
            mask=mask, feat_ids=feat_ids, feature_names=feature_names,
            cost_vals=cost_vals, status_vals=status_vals, pvf_frames=frames,
            include_boundary=include_boundary,
        )
