# Raster-grid PUs — S1 multi-agent design review — synthesis

**Date:** 2026-07-15
**Reviewed:** `2026-07-15-raster-grid-s1-design.md` + `...-s1-implementation.md`
**Lenses:** Architect, Codebase-grounding, Independent re-design. (Scientific-accuracy
skipped — S1 is pure grid geometry; its one correctness claim, analytic boundary ==
shapely, is computational, covered by grounding + independent re-design.)

## Verdict

**Architecturally sound; no CRITICAL or HIGH correctness defect.** Grounding verified
every factual claim about existing code as VERIFIED (`copy_with`/`clone` forwarding,
kw_only-field block, no import cycle, `compute_boundary` column/row/threshold shape,
`build_pu_feature_matrix` row order, `has_geometry` safety, the `generate_planning_grid`
bottom-up-numbering workaround). The independent re-design converged on every structural
and mathematical decision (fields, centroid y-sign, right+down emission, direction→edge
mapping, `1e-10` threshold, sort-before-compare parity) and hand-traced the masked 3×3
parity case against shapely — they match. Two of the plan's choices were judged *better*
than the reviewer's first instinct (shape-from-`mask.shape`; `id1=cell, id2=neighbor`
mirroring shapely's positional pairing rather than `min/max` canonicalization).

## Findings folded in

- **HIGH — non-square cells untested (test gap).** Every test used `w == h == 1.0`, so a
  `cell_width`/`cell_height` swap in `build_boundary` — or a dropped `*cell_width` in
  centroids — would pass the entire suite. This is exactly the logic the spec flagged as
  tricky. *Folded:* added a hand-computed non-square (`w=2, h=3`) `build_boundary` test, a
  non-square shapely-parity test, and non-square centroid/bounds assertions.

- **MEDIUM — `validate()` had no `grid` branch (pattern break + S2 safety net).** Every
  prior optional field (`boundary`/`probability`/`connectivity`) added a `validate()`
  branch. *Folded:* `validate()` now cross-checks `grid.n_pu == len(planning_units)` when a
  grid is present — the checkable half of the positional-alignment contract; catches an
  S2 grid built from a different mask than `planning_units`. + a test.

- **MEDIUM — center-hole parity (masked interior cell).** The masked parity test removed a
  corner; a fully-surrounded masked-out neighbor is the cleaner "next to a masked cell"
  case. *Folded:* added a center-removed (`3×3`, `mask[1,1]=False`) shapely-parity test.

- **LOW — re-export `GridGeometry` from `models/__init__.py`** for discoverability parity
  with `ConservationProblem`. *Folded* into Task 2.

- **LOW — `pu_ids` uniqueness guard.** `build_boundary` is a standalone public method;
  duplicate ids would silently corrupt the `shared[nid]` accumulation. *Folded:* uniqueness
  check alongside the existing `len(pu_ids)==n_pu` guard, + a test.

- **LOW (doc) — reword "id1 < id2"** so it reads as a property of the default sequential
  ids, not a hard invariant S2 might lean on; note the three grid/geometry modules
  (`models/grid.py`, `spatial/grid.py`, `models/geometry.py`) in the `GridGeometry`
  docstring; soften the "usable without geo extras" line (`problem.py` hard-imports
  geopandas at module level for `has_geometry`, so a grid-carrying *problem* still needs
  it; only the standalone class is geo-free).

## Noted, no action (with reason)

- Self-boundary via `perimeter − shared` cancellation vs. exposed-edge sum: the plan
  *mirrors shapely's own* formula, so parity is preserved; the direct-sum alternative would
  itself diverge from shapely at extreme magnitudes. No realistic case; leave as-is.
- NaN/inf guards on the float origin/cell-size: YAGNI (no path produces them).
- Non-vectorized `valid_cells()`/`build_boundary` loops: one-time setup, not a solver inner
  loop, so outside the `bench` per-flip perf guard; a vectorized build is a natural S3 item.
- Sub-`1e-10` cell-size shared-edge threshold difference: pathological only; `cell_* > 0`
  is enforced.
- grid↔boundary auto-wiring is deliberately S2's job (ingestion builds `boundary` from
  `grid.build_boundary(planning_units["id"])` with the same row-major id assignment).
