# Raster-grid PUs — S3c multi-agent design review — synthesis

**Date:** 2026-07-15
**Reviewed:** `2026-07-15-raster-grid-s3c-design.md` + `...-s3c-implementation.md`
**Lenses:** Architect, Codebase-grounding, Independent re-design.

## Verdict

**Design sound and independently converged; two code blockers + refinements.** The
grounding agent **ran the windowed builder against the real repo `from_rasters`** and
confirmed byte-identical output (ids/mask/cost/status/`build_pu_feature_matrix`/`pu_vs_
features` multiset/boundary), and hand-verified the `searchsorted(flat_valid, gflat)`
row-major mapping through the masked-hole case. The independent re-design reached the same
design on every non-obvious call (two-pass, `flat_valid`+searchsorted vs a full `(H×W)`
index, multiset parity, per-window nodata==`_read`, accumulate-warn-once, int64 indices,
boundary-is-the-real-wall). No CRITICAL/HIGH *design* issues; the two HIGH/MEDIUM items are
code defects in the plan's literal code.

## Findings folded in

- **HIGH (grounding) — `_spec` is a nested closure, not module-level.** `_from_rasters_
  windowed` (module-level) calls `_spec` → `NameError` on every windowed call; all 10 new
  tests would error. *Folded:* promote `_spec` from inside `from_rasters` to a module-level
  function (both paths use it).

- **MEDIUM (grounding) — `_read_win` trips `no-any-return`.** `src` is `Any` (rasterio
  untyped) so `return src.read(...).astype(float)` fails mypy under `warn_return_any=true`,
  breaking `make types`. *Folded:* `arr: np.ndarray = src.read(band, window=win).astype(
  float)` then `return arr`. Also corrected the plan's post-note: `rasterio.DatasetReader`
  is a valid annotation that passes mypy (rasterio is `--ignore-missing-imports`), it does
  **not** need a `# type: ignore`.

- **MEDIUM (architect) — ingestion logic duplicated across paths (drift risk).** The
  windowed builder re-implemented status validation, the features table, the alignment
  check, and the `GridGeometry`/`planning_units`/`ConservationProblem` construction from
  `from_arrays` — a future parity fix to `from_arrays` would silently skip the windowed
  path. *Folded:* extracted shared module-level helpers — `_validate_status_ints`,
  `_features_table`, `_check_align` (used by `_read_aligned` + the windowed metadata check),
  and `_assemble_problem` (grid + planning_units + features + `pu_vs_features` concat +
  boundary + `ConservationProblem`) — and refactored `from_arrays` to call them so both
  paths build the model through one code path.

- **MEDIUM (architect) — `include_boundary` silently coupled to the `"auto"` switch.**
  Scaling a `from_rasters(rasters)` call up past `_WINDOW_AUTO_BYTES` silently flips the
  boundary from populated to `None`. *Folded:* when `window_size="auto"` resolves to
  windowed **and** `include_boundary is None`, `warnings.warn` that the boundary was skipped
  for scale (with the explicit-opt-in route). An int `window_size` is an explicit windowing
  choice → no warning.

- **MEDIUM (independent) — vacuous row-major test + unasserted cost-nodata warning.**
  `test_windowed_pu_ids_row_major` asserted `ids == 1..25` on a *full* grid, which holds
  regardless of the mapping. *Folded:* rewrote it with a nodata hole and an assertion that a
  specific post-hole cell's amount lands on the hole-shifted PU id (actually exercises
  `searchsorted`). Added a cost-nodata test asserting the warning fires **exactly once**
  (`len(record)==1`) and the cell's cost defaults to `1.0`.

- **LOW — `window_size` not validated.** `isinstance(_, int)` also matches `bool`; `0`/
  negative give opaque errors. *Folded:* reject non-positive / bool `window_size` with a
  clear `ValueError` + a test.

- **LOW (architect, doc) — re-file `build_boundary` vectorization.** It's a `models/grid.py`
  geometry concern, not an "S3a companion" (S3a is the solver cache). *Folded:* design doc
  re-files it as a standalone `models/grid.py` task.

## Noted, no action

- `_resolve_windowed` reopens the first raster for the "auto" size estimate (also opened by
  the windowed helper via `ExitStack`): metadata-only, negligible.
- `"auto"` dense estimate ignores cost/status/mask layers: only shifts the threshold, never
  correctness. Left as-is.
- Windowed status error message lists only the offending tile's values (full path lists all)
  — raise/no-raise identical, produced problem identical; message text differs. Cosmetic.
- `flat_valid` is up to 8× the bool mask when fully valid, but same order as the mandatory
  output tables and never `O(H×W×n_feat)`. Within budget.
