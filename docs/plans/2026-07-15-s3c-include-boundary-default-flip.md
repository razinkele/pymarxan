# S3c `from_rasters` — flip windowed `include_boundary` default — spec + plan

**Date:** 2026-07-15
**Status:** Approved (brainstorm), streamlined (no spec-review loop / design review — a default
flip + warning removal, per the review skill's "not worth a formal review" bar).
**Scope:** `spatial/raster.py::from_rasters` only. Follows the `build_boundary` vectorization
(v0.21.0), which removed the reason the windowed path skipped the boundary.

## Motivation

S3c (v0.19.0) defaulted `include_boundary` **off** on the windowed path and warned when
auto-windowed, because S1's `build_boundary` was a per-cell Python loop that didn't scale.
v0.21.0 vectorized `build_boundary` (O(n)), so that reason is gone: the windowed path should
build the boundary by default, matching the full-array path, so BLM works out of the box.

## Change

In `from_rasters`:
- Signature: `include_boundary: bool | None = None` → **`include_boundary: bool = True`**.
- Delete the path-dependent resolution + warning:
  ```python
  if include_boundary is None:
      include_boundary = not windowed
      if windowed and window_size == "auto":
          warnings.warn("boundary skipped ...")
  ```
  `include_boundary` is now a plain bool (default `True`); explicit `False` still opts out.
- Docstring: replace the "windowed path defaults `include_boundary` to `False`" note with
  "defaults to `True` on both paths".

`warnings` stays imported (`from_arrays` still uses it for the cost-nodata warning).

## Memory note

At ~1M cells the boundary is a ~3M-row / ~72 MB DataFrame — real but modest (far below the
feature matrix S3a addressed), and now fast to build. A caller who wants to skip it (e.g. a
no-BLM run at extreme scale) passes `include_boundary=False`.

## Tests (TDD)

- **Update `test_windowed_include_boundary_resolution`:** windowed now defaults to a *built*
  boundary (`from_rasters(..., window_size=2).boundary is not None`); explicit
  `include_boundary=False` → `None`; full path still builds by default.
- **Remove `test_auto_windowed_skips_boundary_warns`** (no more warning). The
  `_raster_mod` import stays (used by `test_auto_large_takes_windowed_path`).
- **Anchor:** `test_windowed_equals_full` (passes `include_boundary=True` explicitly) and the
  `build_boundary` parity tests are unaffected; `make check` green.

## References

`build_boundary` vectorization (v0.21.0); S3c design/impl (`2026-07-15-raster-grid-s3c-*`).
