# S4b — raster-aware Shiny map — design review synthesis

**Date:** 2026-07-15
**Reviewed:** `2026-07-15-s4b-raster-map-ui-{design,implementation}.md`
**Lenses:** Architect, Codebase-grounding (RAN reprojection/build_pu_map/centroids live).
(Scientific + independent-redesign skipped — a UI dispatch refactor, no science / novel algorithm.)

## Verdict

**Architecture sound; three code-placement fixes + plan-prose corrections.** Both agents
empirically confirmed the load-bearing bits in the shiny env: `_grid_bounds_latlon` reprojects an
EPSG:3035 grid to correct `((south,west),(north,east))` degrees (`always_xy`→(lon,lat), numpy
arrays, None/4326 passthrough); `build_pu_map`'s three branches (9-cell grid → 9 Rectangles;
`max_cells=2`→None; vector→GeoJSON; synthetic→Rectangles); `compute_centroids` relocation has no
import cycle; all 5 target modules have the exact block to replace and a `render.text` sink; the
`_allow_widget_outside_session` fixture is autouse in test_map_utils.py; the grid PU order ==
planning_units color order invariant holds for `from_arrays`.

## Findings folded in

- **HIGH — relocate `compute_centroids` import to network_view TOP LEVEL (outside the ipyleaflet
  `try`).** The plan's "add to the existing `from .map_utils import …`" lands it inside
  `try: import ipyleaflet`, but `compute_centroids` is used in the **no-ipyleaflet** fallback
  (`network_view.py:255`) and imported at module level by `tests/.../test_network_view.py`. Gating
  it → `NameError`/`ImportError` when ipyleaflet is absent, invisible to CI (shiny env has it).
  *Folded:* import `compute_centroids` at network_view top level, unconditionally (`map_utils`
  imports fine without ipyleaflet).

- **MEDIUM — `too_large_for_map` must be None-safe and placed after the None-guard.** It calls
  `has_grid(problem)` → `problem.grid`; `problem=None` (no project loaded) → `AttributeError`, and
  the summary renders guard `if p is None` *after* where the plan said to put the cap check.
  *Folded:* `too_large_for_map` returns `problem is not None and has_grid(problem) and
  n_planning_units > max_cells`; the module cap-check goes **after** each summary's existing
  `if p is None: return …` guard.

- **MEDIUM — Zonation's cap message would vanish.** `zonation_panel`'s text render is `summary`
  (not `map_summary`), keyed on `_result()` and never fetching `problem()`. *Folded:* Task 2 adds
  `p = problem()` + a `too_large_for_map(p)` branch to zonation's `summary`.

- **LOW — plan-prose corrections.** `comparison_map` is a **single** overlay map (per-cell
  `comparison_color(in_a, in_b)`), not an A/B pair — corrected the "apply to both" note. Dropped
  the stray `zoom=` param from the design signature (impl already omits it; the builders keep their
  own zoom defaults). `grid_builder.py` is a 7th identical dispatch site — left out **on purpose**
  (a spatial *input* module that previews vector-materialized grids, so `has_geometry` is normally
  True); noted explicitly rather than folded in. Added a one-line order-invariant comment to
  `build_pu_map` (colors are in `planning_units` order == grid PU/`cell_bounds` order — a future
  non-`from_arrays` grid source that broke that would mis-color silently, the Zonation-Phase-C
  row-order footgun).

## Noted, no action

- Vector/synthetic branches stay uncapped (a classic large-`n_pu` Marxan project is pre-existing,
  out of S4b's raster scope).
- The live Playwright check remains best-effort (dir-load modal resists headless automation, as in
  Zonation Phase D); the `map_utils` helper unit tests are the hard guard.
