# Raster-grid PUs — S3c implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a windowed (tiled) read path to `from_rasters` so large rasters ingest without loading full `(H×W)` arrays, producing the same `ConservationProblem` as the full-array path.

**Architecture:** A two-pass windowed builder inside `spatial/raster.py`: pass 1 tiles the grid to build the `(H×W)` bool validity mask → `flat_valid` (an `(n_pu,)` sorted index); pass 2 tiles again to sample cost/status and emit the sparse `pu_vs_features`, mapping each cell to its global row-major PU id via `searchsorted(flat_valid, global_flat_index)`. A `window_size` switch (`int | "auto" | None`) selects windowed vs the unchanged S2 full-array path. The pure `from_arrays` core is untouched; no solver/objective change.

**Tech Stack:** Python 3.12+, NumPy, pandas, rasterio (`rasterio.windows.Window`, `contextlib.ExitStack`).

**Design spec:** `docs/plans/2026-07-15-raster-grid-s3c-design.md`. Builds on S2 `spatial/raster.py` (shipped v0.18.0). S3a (sparse cache) + S3b (MIP guard) deferred.

## Global Constraints

- Python 3.12+, `from __future__ import annotations`, full type hints.
- No new third-party dependency.
- Tests **must** run under the `shiny` micromamba env: `/opt/micromamba/envs/shiny/bin/pytest`.
- Lint: ruff (E, F, I, UP; line length 99). Types: mypy clean. Coverage ≥ 75%.
- The bar before done: `make check` green.
- **Reuse S2 helpers** unchanged: `_spec`, `_transforms_close`, `_require_north_up`, `_nodata_mask`, `_VALID_STATUS`, `from_arrays`. Do not duplicate their logic.
- **Positional-alignment contract:** PU ids are `1..n_pu` in global **row-major** valid-cell order (`flat_valid` sorted ascending; `searchsorted` returns the exact rank).
- **int64 indices:** compute global flat indices in int64 (`flat_valid` from `np.flatnonzero` is int64; keep `gflat` int64) to avoid overflow on large grids.
- **Windowed == full:** the windowed path must yield a `ConservationProblem` semantically identical to the full path (same mask, PU ids, cost/status, `build_pu_feature_matrix()`, and `pu_vs_features` as a multiset).

## File Structure

- Modify: `src/pymarxan/spatial/raster.py` — `window_size`/`include_boundary` params on `from_rasters` + private windowed helpers.
- Test: `tests/pymarxan/spatial/test_raster.py` (append).
- Modify: `CHANGELOG.md`.

---

### Task 1: Windowed ingestion in `from_rasters`

**Files:**
- Modify: `src/pymarxan/spatial/raster.py`
- Test: `tests/pymarxan/spatial/test_raster.py` (append)
- Modify: `CHANGELOG.md`

**Interfaces:**
- Consumes: S2 `from_rasters`, `from_arrays`, `_spec`, `_transforms_close`, `_require_north_up`, `_VALID_STATUS`.
- Produces: `from_rasters(..., include_boundary: bool | None = None, window_size: int | Literal["auto"] | None = "auto")` — windowed when `window_size` is an int or `"auto"` resolves large; else the unchanged full path.

- [ ] **Step 1: Write the failing tests**

Append to `tests/pymarxan/spatial/test_raster.py` (the S2 `_write` helper and `Affine`/`from_rasters` imports already exist there; `REF_TF` is the 3×3 grid transform — these tests build their own 5×5 transform):

```python
from pymarxan.spatial import raster as _raster_mod  # noqa: E402  (for monkeypatch)

# 5x5 north-up grid, x_min=0 y_max=5 cell 1x1.
TF5 = Affine(1.0, 0.0, 0.0, 0.0, -1.0, 5.0)


def _grid5(vals):
    return np.asarray(vals, dtype="float32")


@pytest.mark.spatial
def test_windowed_equals_full(tmp_path):
    # 5x5, 2 features, one masked-out cell (nodata in both features).
    f1 = np.arange(25, dtype="float32").reshape(5, 5)
    f2 = (np.arange(25, dtype="float32").reshape(5, 5) % 3)  # some zeros
    f1[2, 2] = -1.0
    f2[2, 2] = -1.0
    p1 = _write(tmp_path, "f1.tif", f1, transform=TF5, nodata=-1)
    p2 = _write(tmp_path, "f2.tif", f2, transform=TF5, nodata=-1)
    win = from_rasters({1: p1, 2: p2}, window_size=2, include_boundary=True)
    full = from_rasters({1: p1, 2: p2}, window_size=None, include_boundary=True)
    assert list(win.planning_units["id"]) == list(full.planning_units["id"])
    assert np.array_equal(win.grid.mask, full.grid.mask)
    assert list(win.planning_units["cost"]) == list(full.planning_units["cost"])
    assert list(win.planning_units["status"]) == list(full.planning_units["status"])
    assert np.allclose(win.build_pu_feature_matrix(), full.build_pu_feature_matrix())
    sort = lambda d: d.sort_values(["species", "pu"]).reset_index(drop=True)  # noqa: E731
    pd.testing.assert_frame_equal(sort(win.pu_vs_features), sort(full.pu_vs_features),
                                  check_dtype=False)
    pd.testing.assert_frame_equal(
        win.boundary.sort_values(["id1", "id2"]).reset_index(drop=True),
        full.boundary.sort_values(["id1", "id2"]).reset_index(drop=True),
        check_dtype=False,
    )


@pytest.mark.spatial
def test_windowed_pu_ids_row_major_across_tiles(tmp_path):
    # window_size=2 straddles rows; PU ids must follow GLOBAL row-major order.
    f1 = np.ones((5, 5), dtype="float32")
    p1 = _write(tmp_path, "f1.tif", f1, transform=TF5)
    p = from_rasters({1: p1}, window_size=2)
    # A cell in tile (0,1) [row 0, col 2] has a lower id than one in tile (1,0) [row 2, col 0].
    # id = row*5 + col + 1 for a full grid.
    assert p.grid.n_pu == 25
    assert list(p.planning_units["id"]) == list(range(1, 26))  # full grid -> 1..25 row-major


@pytest.mark.spatial
def test_windowed_validity_precedence(tmp_path):
    f1 = np.ones((5, 5), dtype="float32")
    cost = np.ones((5, 5), dtype="float32")
    cost[3, 3] = np.nan  # will be written as nodata
    mask = np.ones((5, 5), dtype="float32")
    mask[0, 4] = 0.0
    pf = _write(tmp_path, "f.tif", f1, transform=TF5)
    pc = _write(tmp_path, "c.tif", np.where(np.isnan(cost), -1, cost), transform=TF5, nodata=-1)
    pm = _write(tmp_path, "m.tif", mask, transform=TF5)
    # mask wins over cost/features
    pm_res = from_rasters({1: pf}, cost_raster=pc, mask_raster=pm, window_size=2)
    full = from_rasters({1: pf}, cost_raster=pc, mask_raster=pm, window_size=None)
    assert pm_res.grid.n_pu == full.grid.n_pu == 24  # one masked cell
    # cost footprint (no mask): the nodata-cost cell drops
    cwin = from_rasters({1: pf}, cost_raster=pc, window_size=2)
    cfull = from_rasters({1: pf}, cost_raster=pc, window_size=None)
    assert cwin.grid.n_pu == cfull.grid.n_pu == 24


@pytest.mark.spatial
def test_windowed_cost_status(tmp_path):
    f1 = np.ones((5, 5), dtype="float32")
    cost = np.full((5, 5), 4.0, dtype="float32")
    status = np.zeros((5, 5), dtype="float32")
    status[0, 0] = 2
    status[4, 4] = 3
    p = from_rasters(
        {1: _write(tmp_path, "f.tif", f1, transform=TF5)},
        cost_raster=_write(tmp_path, "c.tif", cost, transform=TF5),
        status_raster=_write(tmp_path, "s.tif", status, transform=TF5),
        window_size=2,
    )
    assert set(p.planning_units["cost"]) == {4.0}
    assert p.planning_units.iloc[0]["status"] == 2  # (0,0) -> id 1
    assert p.planning_units.iloc[-1]["status"] == 3  # (4,4) -> id 25


@pytest.mark.spatial
def test_windowed_invalid_status_raises(tmp_path):
    f1 = np.ones((5, 5), dtype="float32")
    status = np.zeros((5, 5), dtype="float32")
    status[2, 2] = 7  # out of range, in an interior tile
    with pytest.raises(ValueError, match="status"):
        from_rasters(
            {1: _write(tmp_path, "f.tif", f1, transform=TF5)},
            status_raster=_write(tmp_path, "s.tif", status, transform=TF5),
            window_size=2,
        )


@pytest.mark.spatial
def test_windowed_guards_fire(tmp_path):
    # rotated reference raster raises before any window read
    rot = Affine(1.0, 0.5, 0.0, 0.5, -1.0, 5.0)
    a = _write(tmp_path, "a.tif", np.ones((5, 5), dtype="float32"), transform=rot)
    with pytest.raises(ValueError, match="rotat|north"):
        from_rasters({1: a}, window_size=2)
    # transform mismatch under windowing
    b = _write(tmp_path, "b.tif", np.ones((5, 5), dtype="float32"), transform=TF5)
    c = _write(tmp_path, "c.tif", np.ones((5, 5), dtype="float32"),
               transform=Affine(2.0, 0.0, 0.0, 0.0, -2.0, 10.0))
    with pytest.raises(ValueError, match="transform"):
        from_rasters({1: b, 2: c}, window_size=2)


@pytest.mark.spatial
def test_windowed_include_boundary_resolution(tmp_path):
    f1 = np.ones((5, 5), dtype="float32")
    p1 = _write(tmp_path, "f.tif", f1, transform=TF5)
    # windowed defaults to no boundary
    assert from_rasters({1: p1}, window_size=2).boundary is None
    # explicit True builds it
    assert from_rasters({1: p1}, window_size=2, include_boundary=True).boundary is not None
    # full path still defaults to a built boundary
    assert from_rasters({1: p1}, window_size=None).boundary is not None


@pytest.mark.spatial
def test_window_size_larger_than_grid_equals_full(tmp_path):
    f1 = np.arange(25, dtype="float32").reshape(5, 5)
    p1 = _write(tmp_path, "f.tif", f1, transform=TF5)
    big = from_rasters({1: p1}, window_size=100, include_boundary=True)  # 1 tile
    full = from_rasters({1: p1}, window_size=None, include_boundary=True)
    assert np.allclose(big.build_pu_feature_matrix(), full.build_pu_feature_matrix())


@pytest.mark.spatial
def test_auto_small_takes_full_path(tmp_path):
    f1 = np.arange(25, dtype="float32").reshape(5, 5)
    p1 = _write(tmp_path, "f.tif", f1, transform=TF5)
    auto = from_rasters({1: p1}, window_size="auto")  # tiny -> full
    assert auto.boundary is not None  # full path builds boundary by default
    ref = from_rasters({1: p1}, window_size=None)
    assert np.allclose(auto.build_pu_feature_matrix(), ref.build_pu_feature_matrix())


@pytest.mark.spatial
def test_auto_large_takes_windowed_path(tmp_path, monkeypatch):
    monkeypatch.setattr(_raster_mod, "_WINDOW_AUTO_BYTES", 8)  # force windowed
    f1 = np.arange(25, dtype="float32").reshape(5, 5)
    p1 = _write(tmp_path, "f.tif", f1, transform=TF5)
    auto = from_rasters({1: p1}, window_size="auto", include_boundary=True)
    ref = from_rasters({1: p1}, window_size=None, include_boundary=True)
    assert np.allclose(auto.build_pu_feature_matrix(), ref.build_pu_feature_matrix())
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `/opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/spatial/test_raster.py -k "windowed or window_size or auto_" -q`
Expected: FAIL — `from_rasters()` got an unexpected keyword argument `window_size`.

- [ ] **Step 3: Add the windowed helpers + switch**

In `src/pymarxan/spatial/raster.py`, add `Literal` to the typing import and `ExitStack` (top of file):

```python
from contextlib import ExitStack
from pathlib import Path
from typing import Literal
```

(Keep the existing `import warnings`, numpy/pandas/rasterio/affine/CRS imports; add `from rasterio.windows import Window`.)

Add module constants near `_VALID_STATUS`:

```python
_DEFAULT_TILE = 1024
_WINDOW_AUTO_BYTES = 512 * 1024 * 1024  # 512 MiB dense-stack threshold for "auto"
```

Add helpers (after the S2 helpers, before or after `from_rasters`):

```python
def _tiles(height: int, width: int, tile: int):
    """Yield row-major Window tiles of side ``tile`` (edge tiles clamped)."""
    for row_off in range(0, height, tile):
        h = min(tile, height - row_off)
        for col_off in range(0, width, tile):
            w = min(tile, width - col_off)
            yield Window(col_off, row_off, w, h)


def _read_win(src: rasterio.DatasetReader, band: int, win: Window) -> np.ndarray:
    """Read one window as float with the source nodata normalized to NaN (as ``_read``)."""
    arr = src.read(band, window=win).astype(float)
    if src.nodata is not None:
        arr = np.where(arr == src.nodata, np.nan, arr)
    return arr


def _check_meta(src: rasterio.DatasetReader, label: str,
                ref_tf: Affine, ref_shape: tuple[int, int], ref_crs: CRS | None) -> None:
    """Metadata-only alignment check (no read)."""
    shp = (src.height, src.width)
    if shp != ref_shape:
        raise ValueError(f"{label} shape {shp} != reference {ref_shape}")
    if not _transforms_close(src.transform, ref_tf):
        raise ValueError(f"{label} transform {src.transform!r} != reference {ref_tf!r}")
    if src.crs != ref_crs:
        raise ValueError(f"{label} CRS {src.crs} != reference {ref_crs}")


def _resolve_windowed(
    first_path: str | Path, n_feat: int,
    window_size: int | Literal["auto"] | None,
) -> tuple[bool, int]:
    """Return (windowed, tile). ``None`` -> full; int -> windowed(that tile);
    ``"auto"`` -> windowed(default tile) when the dense stack exceeds the threshold."""
    if window_size is None:
        return False, _DEFAULT_TILE
    if isinstance(window_size, int):
        return True, window_size
    if window_size != "auto":
        raise ValueError(f"window_size must be an int, 'auto', or None; got {window_size!r}")
    with rasterio.open(first_path) as src:
        dense_bytes = src.height * src.width * n_feat * 8
    return (dense_bytes > _WINDOW_AUTO_BYTES), _DEFAULT_TILE
```

Replace the `from_rasters` signature + top of body. Change the signature to:

```python
def from_rasters(
    feature_rasters: dict[int, str | Path | tuple[str | Path, int]],
    *,
    cost_raster: str | Path | None = None,
    status_raster: str | Path | None = None,
    mask_raster: str | Path | None = None,
    feature_names: dict[int, str] | None = None,
    include_boundary: bool | None = None,
    window_size: int | Literal["auto"] | None = "auto",
) -> ConservationProblem:
```

The current S2 body begins (after the `if not feature_rasters:` guard) with:

```python
    def _spec(...): ...
    feat_ids = sorted(feature_rasters)
    first_path, first_band = _spec(feature_rasters[feat_ids[0]])
    first_arr, ref_tf, ref_shape, ref_crs = _read(first_path, first_band)
    _require_north_up(ref_tf)
    # ... reads every raster, delegates to from_arrays ...
```

Edit it so that **immediately after** the existing `first_path, first_band = _spec(...)` line
(and before the existing `first_arr, ... = _read(...)` line) the switch is inserted:

```python
    first_path, first_band = _spec(feature_rasters[feat_ids[0]])   # existing line, keep

    # --- S3c switch: windowed vs full-array ---
    windowed, tile = _resolve_windowed(first_path, len(feat_ids), window_size)
    if include_boundary is None:
        include_boundary = not windowed
    if windowed:
        return _from_rasters_windowed(
            feature_rasters, feat_ids, tile,
            cost_raster=cost_raster, status_raster=status_raster, mask_raster=mask_raster,
            feature_names=feature_names, include_boundary=include_boundary,
        )

    # --- full-array path below is the UNCHANGED S2 body ---
    first_arr, ref_tf, ref_shape, ref_crs = _read(first_path, first_band)   # existing, keep
    # ... everything from here down is exactly the existing S2 body; it already
    #     threads `include_boundary` (now a resolved bool) through to from_arrays. ...
```

No other line of the existing full-array body changes.

Add the windowed builder:

```python
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
    with ExitStack() as stack:
        # --- open + align-check all rasters (metadata only) ---
        first_path, first_band = _spec(feature_rasters[feat_ids[0]])
        first_src = stack.enter_context(rasterio.open(first_path))
        _require_north_up(first_src.transform)
        ref_tf, ref_crs = first_src.transform, first_src.crs
        height, width = first_src.height, first_src.width
        ref_shape = (height, width)

        feat_src: dict[int, tuple[rasterio.DatasetReader, int]] = {feat_ids[0]: (first_src, first_band)}
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
        pu_ids = np.arange(1, n_pu + 1)

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
                svr = sv[real]
                iv = np.round(svr).astype(int)
                if np.any(iv != svr) or not set(iv.tolist()).issubset(_VALID_STATUS):
                    raise ValueError(
                        f"status values must be integers in {sorted(_VALID_STATUS)}, "
                        f"got {sorted(set(svr.tolist()))}"
                    )
                status_vals[pu_idx[real]] = iv
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

        grid = GridGeometry(
            ref_tf.c, ref_tf.f, ref_tf.a, -ref_tf.e, mask,
            ref_crs.to_string() if ref_crs is not None else None,
        )
        planning_units = pd.DataFrame({"id": pu_ids, "cost": cost_vals, "status": status_vals})
        names = feature_names or {}
        features = pd.DataFrame(
            {
                "id": feat_ids,
                "name": [names.get(fid, f"feature_{fid}") for fid in feat_ids],
                "target": [0.0] * len(feat_ids),
                "spf": [1.0] * len(feat_ids),
            }
        )
        frames = [df for fid in feat_ids for df in frames_by_feat[fid]]
        if frames:
            pu_vs_features = pd.concat(frames, ignore_index=True)
        else:
            pu_vs_features = pd.DataFrame(columns=["species", "pu", "amount"])
        boundary = grid.build_boundary(pu_ids) if include_boundary else None
        return ConservationProblem(
            planning_units, features, pu_vs_features, boundary=boundary, grid=grid
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `/opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/spatial/test_raster.py -q`
Expected: PASS (S2's 25 + 10 new = 35 tests).

- [ ] **Step 5: Add the CHANGELOG entry**

Under `## [Unreleased]` → `### Added`:

```markdown
- **Windowed raster ingestion (`from_rasters(window_size=...)`, S3c).** Large rasters ingest
  in tiles without loading full ``(H×W)`` arrays: a two-pass windowed builder (bool validity
  mask → ``flat_valid`` index → sparse ``pu_vs_features`` via row-major ``searchsorted``),
  bit-identical to the full-array path. ``window_size`` (``int | "auto" | None``, default
  ``"auto"``) auto-switches on the estimated dense-stack size; the windowed path defaults
  ``include_boundary`` off (S1 ``build_boundary`` is a per-cell Python loop — a scale
  bottleneck to vectorize later). +10 tests.
```

- [ ] **Step 6: Run the full check**

Run: `PATH="/opt/micromamba/envs/shiny/bin:$HOME/.local/bin:$PWD/.venv/bin:$PATH" make check`
Expected: green — 0 ruff, 0 mypy, full suite + 10 new. (`test_solutions_are_different` flake → rerun once.)

Note: the CLAUDE.md `micromamba.sh` activation path may not exist; the `PATH=...` prefix above is the working invocation.

- [ ] **Step 7: Commit**

```bash
git add src/pymarxan/spatial/raster.py tests/pymarxan/spatial/test_raster.py CHANGELOG.md
git commit -m "feat(spatial): windowed raster ingestion — from_rasters(window_size=...) (S3c)"
```

---

## Review fixes folded in (see `...-s3c-review.md`)

The 3-lens review (grounding agent RAN the builder → byte-identical to the full path)
requires these — apply them during execution:
1. **Promote `_spec` to module level** (it is currently a nested closure inside
   `from_rasters`; `_from_rasters_windowed` calls it → `NameError` otherwise).
2. **`_read_win`:** `arr: np.ndarray = src.read(band, window=win).astype(float)` then
   `return arr` (avoids `no-any-return` under `warn_return_any=true`). `rasterio.DatasetReader`
   annotations are fine — no `# type: ignore` needed (rasterio is `--ignore-missing-imports`).
3. **Extract shared helpers** and refactor `from_arrays` to use them (kill windowed/full
   drift): `_validate_status_ints(sv_real)`, `_features_table(feat_ids, feature_names)`,
   `_check_align(label, shp, tf, crs, ref_tf, ref_shape, ref_crs)` (used by `_read_aligned`
   + the windowed metadata check), and `_assemble_problem(*, x_min, y_max, cell_width,
   cell_height, crs, mask, feat_ids, feature_names, cost_vals, status_vals, pvf_frames,
   include_boundary)`.
4. **Validate `window_size`:** reject `bool` / `<= 0` with a clear `ValueError`.
5. **Warn on auto-skipped boundary:** when `"auto"` resolves to windowed and
   `include_boundary is None`, `warnings.warn` that the boundary was skipped for scale.
6. **Tests:** make `test_windowed_pu_ids_row_major` non-vacuous (nodata hole + assert a
   post-hole cell's amount lands on the hole-shifted id); assert the cost-nodata warning
   fires **once** and the cell cost defaults to `1.0`; add a `window_size<=0` guard test and
   an auto-skip-boundary-warning test. New count ≈ +13 tests.

## Post-plan notes

- **Design review:** DONE (`...-s3c-review.md`). Risk surface confirmed sound; the row-major `searchsorted` mapping was verified byte-identical against the real full path.
- **Parity:** ingestion only; no solver/objective math. The windowed==full anchor + the S2 round-trip keep the ingested problem faithful; the 35.0 min-set anchor is untouched (full suite).
- **Deferred:** S3a (sparse `ProblemCache` matrix — unlocks SA/greedy at scale) and S3b (MIP-at-scale guard). Vectorizing `build_boundary` is a standalone `models/grid.py` task (geometry, not solver-cache) so `include_boundary` scales too.
- `mypy`: `rasterio.DatasetReader` annotations are fine (rasterio runs under `--ignore-missing-imports`). The one real mypy risk is `no-any-return` on any helper that bare-`return`s a `src.read(...)`-derived value — annotate the local `arr: np.ndarray` (see review fix 2).
```
