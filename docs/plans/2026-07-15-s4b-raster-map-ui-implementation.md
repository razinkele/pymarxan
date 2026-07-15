# S4b — raster-aware Shiny map — implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Render raster-grid (`GridGeometry`) problems on the Shiny maps using their real cell footprint, via a shared `build_pu_map` dispatcher.

**Architecture:** A `has_grid` predicate + `map_utils` helpers (`build_pu_map`, `pu_centroids_latlon`, `_grid_bounds_latlon`, `too_large_for_map`, relocated `compute_centroids`). The 5 solution-display maps + `network_view` become thin `build_pu_map(...)` calls. `has_geometry` stays vector-only.

**Tech Stack:** Python 3.12+, NumPy, pyproj (via geopandas), ipyleaflet (optional).

**Design spec:** `docs/plans/2026-07-15-s4b-raster-map-ui-design.md`. Builds on S1 (`cell_bounds`/`cell_centroids`), S2/S3 (grid ingestion).

## Global Constraints

- Python 3.12+, `from __future__ import annotations`, full type hints.
- Tests **must** run under the `shiny` micromamba env: `/opt/micromamba/envs/shiny/bin/pytest`.
- Lint: ruff (E, F, I, UP; line length 99). Types: mypy clean. Coverage ≥ 75%.
- The bar before done: `make check` green.
- **`has_geometry` is untouched** (12+ modules read `planning_units.geometry`). Grid support goes through `has_grid` + `build_pu_map`.
- **Cap on the grid branch only** (`max_cells=5000`); vector/synthetic branches unchanged.
- Tests building an `ipyleaflet.Map` outside a session use the `_allow_widget_outside_session` fixture (already in `tests/pymarxan_shiny/test_map_utils.py`).

## File Structure

- Modify: `src/pymarxan/models/problem.py` (+ `has_grid`), `src/pymarxan/models/__init__.py` (export).
- Modify: `src/pymarxan_shiny/modules/mapping/map_utils.py` (helpers + relocated `compute_centroids`).
- Modify: `src/pymarxan_shiny/modules/mapping/{frequency_map,solution_map,comparison_map,spatial_grid,network_view}.py`, `src/pymarxan_shiny/modules/zonation/zonation_panel.py` (refactor to `build_pu_map`).
- Test: `tests/pymarxan_shiny/test_map_utils.py` (append helper tests).
- Modify: `CHANGELOG.md`.

---

### Task 1: `has_grid` + `map_utils` helpers

**Files:**
- Modify: `src/pymarxan/models/problem.py`, `src/pymarxan/models/__init__.py`
- Modify: `src/pymarxan_shiny/modules/mapping/map_utils.py`, `.../mapping/network_view.py` (compute_centroids import)
- Test: `tests/pymarxan_shiny/test_map_utils.py`

**Interfaces:**
- Produces: `has_grid(problem) -> bool`; `map_utils.build_pu_map(problem, colors, *, max_cells=5000) -> ipyleaflet.Map | None`; `pu_centroids_latlon(problem) -> list[tuple[float,float]]`; `too_large_for_map(problem, max_cells=5000) -> bool`; `_grid_bounds_latlon(grid) -> list`; `compute_centroids` (relocated here).

- [ ] **Step 1: Write the failing tests**

Append to `tests/pymarxan_shiny/test_map_utils.py`:

```python
import numpy as np  # noqa: E402  (if not already imported at top)

from pymarxan.models.problem import has_geometry, has_grid  # noqa: E402
from pymarxan.spatial.raster import from_arrays  # noqa: E402
from pymarxan_shiny.modules.mapping.map_utils import (  # noqa: E402
    build_pu_map,
    pu_centroids_latlon,
    too_large_for_map,
    _grid_bounds_latlon,
)


def _grid_problem(nrows=3, ncols=3, crs="EPSG:3035", x_min=4_321_000.0, y_max=3_210_000.0):
    return from_arrays(
        {1: np.ones((nrows, ncols))},
        x_min=x_min, y_max=y_max, cell_width=100.0, cell_height=100.0, crs=crs,
        include_boundary=False,
    )


def test_has_grid_predicate():
    p = _grid_problem()
    assert has_grid(p) and not has_geometry(p)  # grid, but no vector geometry
    from pymarxan.models.problem import ConservationProblem
    import pandas as pd
    plain = ConservationProblem(
        pd.DataFrame({"id": [1], "cost": [1.0], "status": [0]}),
        pd.DataFrame({"id": [1], "name": ["a"], "target": [1.0], "spf": [1.0]}),
        pd.DataFrame({"species": [1], "pu": [1], "amount": [1.0]}),
    )
    assert not has_grid(plain)


def test_grid_bounds_latlon_reprojects():
    p = _grid_problem()  # EPSG:3035, realistic European coords
    bounds = _grid_bounds_latlon(p.grid)
    assert len(bounds) == p.grid.n_pu
    for (s, w), (n, e) in bounds:
        assert s < n and w < e          # south<north, west<east
        assert -90 <= s <= 90 and -180 <= w <= 180  # plausible degrees


def test_grid_bounds_latlon_none_crs_passthrough():
    p = _grid_problem(crs=None)  # no CRS -> raw coords, no reprojection
    bounds = _grid_bounds_latlon(p.grid)
    assert len(bounds) == p.grid.n_pu
    # raw passthrough: cell_bounds (minx,miny,maxx,maxy) -> ((miny,minx),(maxy,maxx))
    minx, miny, maxx, maxy = p.grid.cell_bounds()[0]
    assert bounds[0] == ((miny, minx), (maxy, maxx))


def test_build_pu_map_grid_branch(_allow_widget_outside_session):
    p = _grid_problem()
    m = build_pu_map(p, ["#ff0000"] * p.grid.n_pu)
    assert m is not None
    from ipyleaflet import Rectangle
    assert sum(isinstance(layer, Rectangle) for layer in m.layers) == p.grid.n_pu


def test_build_pu_map_cap(_allow_widget_outside_session):
    p = _grid_problem(nrows=3, ncols=3)  # 9 cells
    assert build_pu_map(p, ["#f00"] * 9, max_cells=2) is None  # over cap
    assert too_large_for_map(p, max_cells=2) is True
    assert build_pu_map(p, ["#f00"] * 9, max_cells=100) is not None  # under cap
    assert too_large_for_map(p, max_cells=100) is False


def test_build_pu_map_vector_branch(_allow_widget_outside_session):
    import geopandas as gpd
    from shapely.geometry import Polygon

    from pymarxan.models.problem import ConservationProblem
    import pandas as pd
    pu = gpd.GeoDataFrame(
        {"id": [1, 2], "cost": [1.0, 1.0], "status": [0, 0]},
        geometry=[Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
                  Polygon([(1, 0), (2, 0), (2, 1), (1, 1)])],
        crs="EPSG:4326",
    )
    p = ConservationProblem(
        pu, pd.DataFrame({"id": [1], "name": ["a"], "target": [1.0], "spf": [1.0]}),
        pd.DataFrame({"species": [1, 1], "pu": [1, 2], "amount": [1.0, 1.0]}),
    )
    m = build_pu_map(p, ["#f00", "#0f0"])
    from ipyleaflet import GeoJSON
    assert m is not None and any(isinstance(x, GeoJSON) for x in m.layers)


def test_pu_centroids_latlon_grid():
    p = _grid_problem()
    cents = pu_centroids_latlon(p)
    assert len(cents) == p.grid.n_pu
    for lat, lon in cents:
        assert -90 <= lat <= 90 and -180 <= lon <= 180
```

- [ ] **Step 2: Run to verify they fail**

Run: `/opt/micromamba/envs/shiny/bin/pytest tests/pymarxan_shiny/test_map_utils.py -k "has_grid or grid_bounds or build_pu_map or centroids or cap" -q`
Expected: FAIL — `has_grid`/`build_pu_map`/etc. not importable.

- [ ] **Step 3: Add `has_grid` to the model**

In `src/pymarxan/models/problem.py`, right after `has_geometry`:

```python
def has_grid(problem: ConservationProblem) -> bool:
    """Check if the problem carries a raster GridGeometry (grid-cell PUs)."""
    return problem.grid is not None
```

In `src/pymarxan/models/__init__.py`, add `has_grid` to the import from `.problem` and to `__all__` (keep alphabetical).

- [ ] **Step 4: Add the `map_utils` helpers**

In `src/pymarxan_shiny/modules/mapping/map_utils.py`, add imports at top and the helpers.
Add to the import block:

```python
import numpy as np

from pymarxan.models.problem import has_geometry, has_grid
```

Add the module constant + helpers:

```python
_MAX_MAP_CELLS = 5000


def compute_centroids(
    grid: list[tuple[tuple[float, float], tuple[float, float]]],
) -> list[tuple[float, float]]:
    """Center (lat, lon) of each bounding box. (Relocated from network_view.)"""
    return [((s + n) / 2, (w + e) / 2) for (s, w), (n, e) in grid]


def too_large_for_map(problem, max_cells: int = _MAX_MAP_CELLS) -> bool:
    """True when a grid problem has too many cells to render as rectangles.

    None-safe (``problem`` may be None when no project is loaded)."""
    return (
        problem is not None
        and has_grid(problem)
        and problem.n_planning_units > max_cells
    )


def _latlon_transformer(crs):
    """A pyproj Transformer to EPSG:4326, or None (use raw coords) for None / already-4326 /
    unparseable CRS."""
    if crs is None:
        return None
    try:
        from pyproj import CRS, Transformer
        if CRS.from_user_input(crs).to_epsg() == 4326:
            return None
        return Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    except Exception:
        return None


def _grid_bounds_latlon(grid) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """``grid.cell_bounds()`` -> ``[((south, west), (north, east))]`` in EPSG:4326."""
    arr = np.asarray(grid.cell_bounds(), dtype=float)  # (n, 4): minx, miny, maxx, maxy
    minx, miny, maxx, maxy = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]
    tr = _latlon_transformer(grid.crs)
    if tr is not None:
        west, south = tr.transform(minx, miny)  # always_xy -> (lon, lat)
        east, north = tr.transform(maxx, maxy)
    else:
        west, south, east, north = minx, miny, maxx, maxy
    return [((float(s), float(w)), (float(n), float(e)))
            for s, w, n, e in zip(south, west, north, east)]


def build_pu_map(problem, colors: list[str], *, max_cells: int = _MAX_MAP_CELLS):
    """Base ipyleaflet map for a problem's PUs: vector geometry, raster grid, or synthetic
    grid. Returns None when ipyleaflet is missing, or a grid exceeds ``max_cells``.

    ``colors`` is one entry per PU in ``planning_units`` order — which, for a grid problem,
    coincides with ``cell_bounds()``/``cell_centroids()`` PU (row-major) order, so cells,
    centroids, and colors line up. (A grid source that didn't preserve that order would
    mis-color silently.)"""
    if not _HAS_IPYLEAFLET:
        return None
    if has_geometry(problem):
        return create_geo_map(problem.planning_units, colors)
    if has_grid(problem):
        if problem.n_planning_units > max_cells:
            return None
        return create_grid_map(_grid_bounds_latlon(problem.grid), colors)
    from pymarxan.models.geometry import generate_grid
    return create_grid_map(generate_grid(problem.n_planning_units), colors)


def pu_centroids_latlon(problem) -> list[tuple[float, float]]:
    """(lat, lon) centroid per PU (for network overlays); same 3-way dispatch as build_pu_map."""
    if has_geometry(problem):
        pus = problem.planning_units
        if pus.crs is not None and pus.crs.to_epsg() != 4326:
            pus = pus.to_crs("EPSG:4326")
        return [(g.centroid.y, g.centroid.x) for g in pus.geometry]
    if has_grid(problem):
        cents = np.asarray(problem.grid.cell_centroids(), dtype=float)  # (n, 2): x, y
        x, y = cents[:, 0], cents[:, 1]
        tr = _latlon_transformer(problem.grid.crs)
        lon, lat = (tr.transform(x, y) if tr is not None else (x, y))
        return [(float(la), float(lo)) for la, lo in zip(lat, lon)]
    from pymarxan.models.geometry import generate_grid
    return compute_centroids(generate_grid(problem.n_planning_units))
```

Then in `src/pymarxan_shiny/modules/mapping/network_view.py`, **remove** the local
`compute_centroids` definition and import it from `map_utils` at **module top level, OUTSIDE the
`try: import ipyleaflet` block** (a dedicated `from pymarxan_shiny.modules.mapping.map_utils import
compute_centroids`). **Do not** add it to the ipyleaflet-gated import: `compute_centroids` is used
by the non-ipyleaflet fallback render and imported at top level by `test_network_view.py`, so
gating it would `NameError`/`ImportError` when ipyleaflet is absent (invisible to CI, which has
ipyleaflet). `map_utils` imports fine without ipyleaflet.

- [ ] **Step 5: Run to verify they pass**

Run: `/opt/micromamba/envs/shiny/bin/pytest tests/pymarxan_shiny/test_map_utils.py -q`
Expected: PASS (existing + ~7 new).

- [ ] **Step 6: Commit**

```bash
git add src/pymarxan/models/problem.py src/pymarxan/models/__init__.py \
        src/pymarxan_shiny/modules/mapping/map_utils.py \
        src/pymarxan_shiny/modules/mapping/network_view.py \
        tests/pymarxan_shiny/test_map_utils.py
git commit -m "feat(shiny): has_grid + build_pu_map/pu_centroids_latlon map helpers (S4b)"
```

---

### Task 2: refactor the map modules + CHANGELOG

**Files:**
- Modify: `.../mapping/{frequency_map,solution_map,comparison_map,spatial_grid,network_view}.py`, `.../zonation/zonation_panel.py`
- Modify: `CHANGELOG.md`

**Interfaces:**
- Consumes: `build_pu_map`, `pu_centroids_latlon`, `too_large_for_map` (Task 1).

- [ ] **Step 1: Refactor the 5 solution-display maps**

For each of `frequency_map`, `solution_map`, `comparison_map`, `spatial_grid`, and
`zonation/zonation_panel`: in the `@render_widget def map()` (whatever it's named), replace the
```python
if has_geometry(p):
    return create_geo_map(p.planning_units, colors)
grid = generate_grid(n_pu)
return create_grid_map(grid, colors)
```
block with:
```python
return build_pu_map(p, colors)
```
and in the module's summary/status `render.text`, add the cap message **immediately after that
render's existing `if p is None: return …` guard** (not before it — `too_large_for_map` is
None-safe but the message only makes sense once a project is loaded):
```python
if too_large_for_map(p):
    return f"Grid too large to map ({p.n_planning_units} cells); use the analysis/table views."
```
Update imports: `from .map_utils import build_pu_map, too_large_for_map` (drop now-unused
`create_geo_map`/`create_grid_map`/`generate_grid`/`has_geometry` imports if nothing else uses
them in that file — ruff F401 will flag leftovers). Each of these has a single `map()` render
(e.g. `comparison_map` is one overlay map colored by `comparison_color(in_a, in_b)`, **not** an
A/B pair) and a problem-keyed summary text.

**Zonation exception:** `zonation_panel`'s text render is `summary` (not `map_summary`) and is
keyed on the ranking `_result()`, not `problem()` — so add `p = problem()` at the top of `summary`
and the `too_large_for_map(p)` branch there, else its cap message would silently vanish.

- [ ] **Step 2: Refactor `network_view`**

Replace its `if has_geometry(p): … else: generate_grid/compute_centroids` block with:
```python
m = build_pu_map(p, colors)
if m is None:
    return None  # ipyleaflet missing or grid too large
centroids = pu_centroids_latlon(p)
```
keeping the subsequent capped-polyline overlay loop unchanged (it reads `centroids[i]`).
Add `too_large_for_map` to `map_summary` as in Step 1. Update imports.

- [ ] **Step 3: Run the mapping + model test suites (no regression)**

Run: `/opt/micromamba/envs/shiny/bin/pytest tests/pymarxan_shiny/ tests/pymarxan/models/ -q`
Expected: PASS — the existing per-module tests (test_frequency_map, test_solution_map,
test_comparison_map, …) still pass on the vector path, and the Task 1 helper tests pass.

- [ ] **Step 4: CHANGELOG + full check**

Add under `## [Unreleased]` → `### Added`:
```markdown
- **Raster-grid problems render on the Shiny maps (S4b).** A new `has_grid` predicate + a shared
  `build_pu_map` dispatcher draw grid (raster-ingested) planning units on their real cell
  footprint (`GridGeometry.cell_bounds()`, reprojected to lat/lon) across the frequency / solution
  / comparison / spatial-grid / Zonation maps and the connectivity network view. Large grids
  (> 5000 cells) show a "too large to map" message instead of freezing the browser. `has_geometry`
  is unchanged (vector-only).
```

Run: `PATH="/opt/micromamba/envs/shiny/bin:$HOME/.local/bin:$PWD/.venv/bin:$PATH" make check`
Expected: green — 0 ruff, 0 mypy, full suite. (`test_solutions_are_different` flake → rerun once.)

- [ ] **Step 5: Live check (best-effort)**

Launch `make app`, load a small grid project (or construct one), and screenshot a map tab showing
grid cells. The dir-load modal may resist headless Playwright (as in Zonation Phase D) — the Task 1
helper unit tests are the hard guard; note the outcome.

- [ ] **Step 6: Commit**

```bash
git add src/pymarxan_shiny/modules/ CHANGELOG.md
git commit -m "feat(shiny): route the maps + network view through build_pu_map (grid support, S4b)"
```

---

## Post-plan notes

- **Design review:** run `multi-agent-design-review` — the risk surface is the CRS reprojection
  (`always_xy`/lon-lat order, None/unparseable CRS fallback), the cap message plumbing
  (`render_widget` → None + summary text), the `compute_centroids` relocation (no import cycle),
  and no regression on the vector path across the 6 refactored modules.
- **Parity:** UI-only; no solver/objective change; the 35.0 anchor is untouched.
- **Deferred:** downsampling large grids for the map (the cap skips instead, per the brainstorm).
