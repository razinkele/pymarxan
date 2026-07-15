# Raster-grid PUs — S1 implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `GridGeometry` (grid-cells-as-PUs descriptor + analytic rook-adjacency boundary) and carry it as a `kw_only` field on `ConservationProblem`.

**Architecture:** A pure numpy/pandas `GridGeometry` (origin + cell size + validity mask + CRS). Valid cells in row-major order ARE the PU order. `build_boundary()` generates the boundary matrix analytically (right + down neighbors → `id1 < id2`; self-boundary = perimeter − shared), matching shapely `compute_boundary` on identical cells. One `kw_only` field on the problem; solvers untouched. No new deps.

**Tech Stack:** Python 3.12+, NumPy, pandas. (The boundary *parity test* also uses geopandas/shapely, which are already core deps.)

**Design spec:** `docs/plans/2026-07-15-raster-grid-s1-design.md`; scoping: `docs/plans/2026-07-15-raster-grid-pus-scoping.md`.

## Global Constraints

- Python 3.12+, `from __future__ import annotations`, full type hints.
- Zero new third-party dependencies (`GridGeometry` is pure numpy/pandas).
- Tests **must** run under the `shiny` micromamba env: `/opt/micromamba/envs/shiny/bin/pytest`.
- Lint: ruff (E, F, I, UP; line length 99). Types: mypy clean. Coverage threshold 75%.
- The bar before done: `make check` green.
- New optional `ConservationProblem` fields must be `kw_only=True` with defaults (CLAUDE.md). `copy_with` (`dataclass_fields`) and `clone` (`copy.deepcopy`) both forward it automatically.
- `GridGeometry` is `@dataclass(eq=False)` (its numpy `mask` breaks the auto `__eq__`).
- **Positional-alignment contract:** valid-cell row-major order == PU order == `build_pu_feature_matrix` row order. `build_boundary` checks `len(pu_ids)==n_pu`; the *order* is a documented contract.

## File Structure

- Create: `src/pymarxan/models/grid.py` — `GridGeometry`.
- Modify: `src/pymarxan/models/problem.py` — `grid` kw_only field + import + `validate()` branch.
- Modify: `src/pymarxan/models/__init__.py` — re-export `GridGeometry`.
- Create: `tests/pymarxan/models/test_grid.py`.
- Modify: `tests/pymarxan/models/test_problem.py` (or the S1 test file) — the field tests.
- Modify: `CHANGELOG.md`.

---

### Task 1: `GridGeometry`

**Files:**
- Create: `src/pymarxan/models/grid.py`
- Test: `tests/pymarxan/models/test_grid.py`

**Interfaces:**
- Produces: `GridGeometry(x_min, y_max, cell_width, cell_height, mask, crs=None)` with `n_pu`, `shape`, `valid_cells()`, `cell_centroids()`, `cell_bounds()`, `build_boundary(pu_ids=None)`.

- [ ] **Step 1: Write the failing tests**

Create `tests/pymarxan/models/test_grid.py`:

```python
"""Tests for GridGeometry (raster-grid PUs, S1)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pymarxan.models.grid import GridGeometry


def test_n_pu_and_valid_cells_row_major():
    mask = np.ones((3, 3), dtype=bool)
    mask[1, 1] = False
    mask[2, 0] = False
    grid = GridGeometry(0.0, 3.0, 1.0, 1.0, mask)
    assert grid.n_pu == 7
    assert grid.shape == (3, 3)
    cells = grid.valid_cells()
    assert cells[0] == (0, 0)
    assert (1, 1) not in cells and (2, 0) not in cells
    assert cells == sorted(cells)  # row-major == sorted by (r, c)


def test_cell_centroids_2x2():
    grid = GridGeometry(0.0, 2.0, 1.0, 1.0, np.ones((2, 2), dtype=bool))
    # PU order (0,0),(0,1),(1,0),(1,1); centroid = (x_min+(c+.5)w, y_max-(r+.5)h)
    expected = np.array([[0.5, 1.5], [1.5, 1.5], [0.5, 0.5], [1.5, 0.5]])
    assert np.allclose(grid.cell_centroids(), expected)


def test_cell_bounds_2x2():
    grid = GridGeometry(0.0, 2.0, 1.0, 1.0, np.ones((2, 2), dtype=bool))
    # (0,0): minx0 miny1 maxx1 maxy2
    assert grid.cell_bounds()[0] == (0.0, 1.0, 1.0, 2.0)


def test_build_boundary_hand_computed_2x2():
    grid = GridGeometry(0.0, 2.0, 1.0, 1.0, np.ones((2, 2), dtype=bool))
    df = grid.build_boundary()  # ids 1..4
    shared = df[df.id1 != df.id2]
    self_rows = df[df.id1 == df.id2]
    assert len(shared) == 4 and set(shared.boundary) == {1.0}
    assert len(self_rows) == 4 and set(self_rows.boundary) == {2.0}


def test_single_cell_boundary():
    grid = GridGeometry(0.0, 1.0, 1.0, 1.0, np.ones((1, 1), dtype=bool))
    df = grid.build_boundary()
    assert len(df) == 1
    row = df.iloc[0]
    assert row.id1 == row.id2 and row.boundary == 4.0  # full perimeter


def test_build_boundary_len_guard():
    grid = GridGeometry(0.0, 2.0, 1.0, 1.0, np.ones((2, 2), dtype=bool))
    with pytest.raises(ValueError, match="entries"):
        grid.build_boundary(np.array([1, 2, 3]))  # 3 != 4


def test_build_boundary_duplicate_ids_guard():
    grid = GridGeometry(0.0, 2.0, 1.0, 1.0, np.ones((2, 2), dtype=bool))
    with pytest.raises(ValueError, match="unique"):
        grid.build_boundary(np.array([1, 2, 2, 3]))  # right length, dup id


def test_non_square_cells():
    # cell_width=2 (x), cell_height=3 (y); origin (10, 20). Guards the
    # direction->edge mapping + centroid/bounds axis scaling (a w/h swap here
    # WOULD change the assertions, unlike any w==h==1 test).
    grid = GridGeometry(10.0, 20.0, 2.0, 3.0, np.ones((2, 2), dtype=bool))
    # PU order (0,0),(0,1),(1,0),(1,1): x = 10 + (c+.5)*2, y = 20 - (r+.5)*3
    expected_c = np.array([[11.0, 18.5], [13.0, 18.5], [11.0, 15.5], [13.0, 15.5]])
    assert np.allclose(grid.cell_centroids(), expected_c)
    # (0,0) bounds: minx10 maxx12, maxy20 miny17
    assert grid.cell_bounds()[0] == (10.0, 17.0, 12.0, 20.0)
    df = grid.build_boundary()
    shared = df[df.id1 != df.id2]
    # horizontal (left-right) neighbors share a vertical edge = cell_height = 3;
    # vertical (up-down) neighbors share a horizontal edge = cell_width = 2.
    horiz = shared[shared.id1.isin([1, 3]) & shared.id2.isin([2, 4])]
    vert = shared[shared.id1.isin([1, 2]) & shared.id2.isin([3, 4])]
    assert set(horiz.boundary) == {3.0}  # NOT 2.0 — would fail on a w/h swap
    assert set(vert.boundary) == {2.0}


def test_validation():
    good = np.ones((2, 2), dtype=bool)
    with pytest.raises(ValueError, match="2-D boolean"):
        GridGeometry(0.0, 2.0, 1.0, 1.0, np.ones(4, dtype=bool))  # 1-D
    with pytest.raises(ValueError, match="2-D boolean"):
        GridGeometry(0.0, 2.0, 1.0, 1.0, np.ones((2, 2), dtype=int))  # not bool
    with pytest.raises(ValueError, match="> 0"):
        GridGeometry(0.0, 2.0, 0.0, 1.0, good)  # cell_width 0
    with pytest.raises(ValueError, match="no valid cells"):
        GridGeometry(0.0, 2.0, 1.0, 1.0, np.zeros((2, 2), dtype=bool))  # all False


@pytest.mark.spatial
def test_build_boundary_matches_shapely_full_grid():
    import geopandas as gpd
    from shapely.geometry import box

    from pymarxan.spatial.boundary import compute_boundary

    grid = GridGeometry(0.0, 3.0, 1.0, 1.0, np.ones((3, 3), dtype=bool))
    ids = np.arange(1, grid.n_pu + 1)
    gdf = gpd.GeoDataFrame({"id": ids}, geometry=[box(*b) for b in grid.cell_bounds()])
    analytic = grid.build_boundary(ids).sort_values(["id1", "id2"]).reset_index(drop=True)
    shapely_b = compute_boundary(gdf).sort_values(["id1", "id2"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(analytic, shapely_b, check_dtype=False)


@pytest.mark.spatial
def test_build_boundary_matches_shapely_masked_grid():
    import geopandas as gpd
    from shapely.geometry import box

    from pymarxan.spatial.boundary import compute_boundary

    mask = np.ones((3, 3), dtype=bool)
    mask[0, 0] = False  # remove a corner → exposed edges
    grid = GridGeometry(0.0, 3.0, 1.0, 1.0, mask)
    ids = np.arange(1, grid.n_pu + 1)
    gdf = gpd.GeoDataFrame({"id": ids}, geometry=[box(*b) for b in grid.cell_bounds()])
    analytic = grid.build_boundary(ids).sort_values(["id1", "id2"]).reset_index(drop=True)
    shapely_b = compute_boundary(gdf).sort_values(["id1", "id2"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(analytic, shapely_b, check_dtype=False)


@pytest.mark.spatial
def test_build_boundary_matches_shapely_center_hole():
    import geopandas as gpd
    from shapely.geometry import box

    from pymarxan.spatial.boundary import compute_boundary

    mask = np.ones((3, 3), dtype=bool)
    mask[1, 1] = False  # fully-surrounded masked-out cell → its 4 neighbors gain self-edge
    grid = GridGeometry(0.0, 3.0, 1.0, 1.0, mask)
    ids = np.arange(1, grid.n_pu + 1)
    gdf = gpd.GeoDataFrame({"id": ids}, geometry=[box(*b) for b in grid.cell_bounds()])
    analytic = grid.build_boundary(ids).sort_values(["id1", "id2"]).reset_index(drop=True)
    shapely_b = compute_boundary(gdf).sort_values(["id1", "id2"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(analytic, shapely_b, check_dtype=False)


@pytest.mark.spatial
def test_build_boundary_matches_shapely_non_square():
    import geopandas as gpd
    from shapely.geometry import box

    from pymarxan.spatial.boundary import compute_boundary

    grid = GridGeometry(10.0, 20.0, 2.0, 3.0, np.ones((3, 3), dtype=bool))  # w != h
    ids = np.arange(1, grid.n_pu + 1)
    gdf = gpd.GeoDataFrame({"id": ids}, geometry=[box(*b) for b in grid.cell_bounds()])
    analytic = grid.build_boundary(ids).sort_values(["id1", "id2"]).reset_index(drop=True)
    shapely_b = compute_boundary(gdf).sort_values(["id1", "id2"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(analytic, shapely_b, check_dtype=False)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `/opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/models/test_grid.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pymarxan.models.grid'`.

- [ ] **Step 3: Implement `GridGeometry`**

Create `src/pymarxan/models/grid.py`:

```python
"""Grid-geometry descriptor for raster-grid planning units (S1)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(eq=False)
class GridGeometry:
    """A north-up, axis-aligned grid whose valid cells are planning units.

    ``mask`` is an ``(nrows, ncols)`` boolean array; ``True`` marks a valid cell
    (a planning unit). The valid cells in row-major order (row 0 = top) ARE the
    planning-unit order: PU row ``i`` <-> the ``i``-th valid cell <-> row ``i``
    of ``build_pu_feature_matrix``. Pure numpy/pandas (no shapely/rasterio).
    ``eq=False`` because the numpy ``mask`` breaks the auto ``__eq__``. Treat as
    immutable — do not mutate ``mask`` after construction.

    The analytic, polygon-free counterpart to two sibling modules it should not be
    confused with: ``spatial/grid.py`` (``generate_planning_grid`` — a materialized
    vector square grid) and ``models/geometry.py`` (``generate_grid`` — synthetic
    bounding boxes for the map).
    """

    x_min: float
    y_max: float
    cell_width: float
    cell_height: float
    mask: np.ndarray
    crs: str | None = None

    def __post_init__(self) -> None:
        mask = np.asarray(self.mask)
        if mask.ndim != 2 or mask.dtype != bool:
            raise ValueError("mask must be a 2-D boolean array")
        if self.cell_width <= 0 or self.cell_height <= 0:
            raise ValueError("cell_width and cell_height must be > 0")
        if not mask.any():
            raise ValueError("mask has no valid cells (all False)")
        self.mask = mask

    @property
    def shape(self) -> tuple[int, int]:
        return (self.mask.shape[0], self.mask.shape[1])

    @property
    def n_pu(self) -> int:
        return int(self.mask.sum())

    def valid_cells(self) -> list[tuple[int, int]]:
        """Valid ``(row, col)`` cells in row-major (top-down) order = PU order."""
        rows, cols = np.nonzero(self.mask)
        return list(zip(rows.tolist(), cols.tolist()))

    def cell_centroids(self) -> np.ndarray:
        """``(n_pu, 2)`` array of ``(x, y)`` cell centroids in PU order."""
        cells = self.valid_cells()
        out = np.empty((len(cells), 2), dtype=float)
        for i, (r, c) in enumerate(cells):
            out[i, 0] = self.x_min + (c + 0.5) * self.cell_width
            out[i, 1] = self.y_max - (r + 0.5) * self.cell_height
        return out

    def cell_bounds(self) -> list[tuple[float, float, float, float]]:
        """``(minx, miny, maxx, maxy)`` per PU in PU order."""
        out: list[tuple[float, float, float, float]] = []
        for r, c in self.valid_cells():
            minx = self.x_min + c * self.cell_width
            maxx = minx + self.cell_width
            maxy = self.y_max - r * self.cell_height
            miny = maxy - self.cell_height
            out.append((minx, miny, maxx, maxy))
        return out

    def build_boundary(self, pu_ids: np.ndarray | None = None) -> pd.DataFrame:
        """Analytic rook-adjacency boundary (columns ``id1``, ``id2``,
        ``boundary``), matching ``spatial.boundary.compute_boundary`` on the
        equivalent vector grid. ``pu_ids`` aligns to valid-cell order (defaults
        to ``1..n_pu``)."""
        cells = self.valid_cells()
        n = len(cells)
        if pu_ids is None:
            pu_ids = np.arange(1, n + 1)
        pu_ids = np.asarray(pu_ids)
        if len(pu_ids) != n:
            raise ValueError(f"pu_ids must have {n} entries, got {len(pu_ids)}")
        if len(set(pu_ids.tolist())) != n:
            raise ValueError("pu_ids must be unique")
        cell_to_id = {cell: int(pu_ids[i]) for i, cell in enumerate(cells)}
        shared: dict[int, float] = {int(pid): 0.0 for pid in pu_ids}
        rows: list[dict] = []
        for (r, c), pid in cell_to_id.items():
            # right neighbor shares a vertical edge (length cell_height);
            # down neighbor shares a horizontal edge (length cell_width).
            for nbr, edge in (((r, c + 1), self.cell_height), ((r + 1, c), self.cell_width)):
                nid = cell_to_id.get(nbr)
                if nid is not None:
                    rows.append({"id1": pid, "id2": nid, "boundary": edge})
                    shared[pid] += edge
                    shared[nid] += edge
        perimeter = 2.0 * (self.cell_width + self.cell_height)
        for pid in cell_to_id.values():
            self_b = perimeter - shared[pid]
            if self_b > 1e-10:
                rows.append({"id1": pid, "id2": pid, "boundary": self_b})
        return pd.DataFrame(rows, columns=["id1", "id2", "boundary"])
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `/opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/models/test_grid.py -v`
Expected: PASS (13 tests).

- [ ] **Step 5: Commit**

```bash
git add src/pymarxan/models/grid.py tests/pymarxan/models/test_grid.py
git commit -m "feat(models): GridGeometry — grid-cells-as-PUs + analytic rook-adjacency boundary"
```

---

### Task 2: `ConservationProblem.grid` field + CHANGELOG

**Files:**
- Modify: `src/pymarxan/models/problem.py`
- Test: `tests/pymarxan/models/test_grid.py` (append the field tests)
- Modify: `CHANGELOG.md`

- [ ] **Step 1: Write the failing tests**

Append to `tests/pymarxan/models/test_grid.py`:

```python
def _tiny_problem():
    from pymarxan.models.problem import ConservationProblem

    pu = pd.DataFrame({"id": [1, 2], "cost": [1.0, 1.0], "status": [0, 0]})
    feats = pd.DataFrame({"id": [1], "name": ["a"], "target": [1.0], "spf": [1.0]})
    pvf = pd.DataFrame({"species": [1, 1], "pu": [1, 2], "amount": [1.0, 1.0]})
    return ConservationProblem(pu, feats, pvf)


def test_problem_grid_defaults_none():
    assert _tiny_problem().grid is None


def test_copy_with_and_clone_preserve_grid():
    g = GridGeometry(0.0, 2.0, 1.0, 1.0, np.ones((2, 2), dtype=bool))
    p = _tiny_problem().copy_with(grid=g)
    assert p.grid is g
    # preserved through a later copy_with that overrides an unrelated field
    assert p.copy_with(parameters={"BLM": 1.0}).grid is g
    # clone deep-copies (independent grid, still present)
    cloned = p.clone()
    assert cloned.grid is not None and cloned.grid is not g
    assert cloned.grid.n_pu == 4


def test_validate_grid_count_mismatch():
    # _tiny_problem has 2 PUs; a 2x2 grid (n_pu=4) disagrees -> validate error
    g4 = GridGeometry(0.0, 2.0, 1.0, 1.0, np.ones((2, 2), dtype=bool))
    errs = _tiny_problem().copy_with(grid=g4).validate()
    assert any("grid" in e and "planning" in e for e in errs)
    # a matching grid (n_pu == 2) yields no grid error
    g2 = GridGeometry(0.0, 2.0, 1.0, 1.0, np.array([[True, True]], dtype=bool))
    errs_ok = _tiny_problem().copy_with(grid=g2).validate()
    assert not any("grid" in e for e in errs_ok)
```

- [ ] **Step 2: Run them to verify they fail**

Run: `/opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/models/test_grid.py -k "grid_defaults or preserve_grid" -v`
Expected: FAIL — `ConservationProblem` has no `grid` attribute (`TypeError`/`AttributeError`).

- [ ] **Step 3: Add the field to `ConservationProblem`**

In `src/pymarxan/models/problem.py`, add the import after the existing model imports (isort: `pymarxan.models` package import sorts before `pymarxan.models.grid`):

```python
from pymarxan.models.grid import GridGeometry
```

Add the field to the dataclass, after the `connectivity` kw_only field:

```python
    connectivity: pd.DataFrame | None = field(default=None, kw_only=True)
    grid: GridGeometry | None = field(default=None, kw_only=True)
```

Add a `grid` branch to `validate()`, right before its final `return errors`, mirroring
the existing `boundary`/`probability`/`connectivity` branches:

```python
        # --- Grid validation ---
        if self.grid is not None and self.grid.n_pu != len(self.planning_units):
            errors.append(
                f"grid.n_pu ({self.grid.n_pu}) does not match the number of "
                f"planning_units ({len(self.planning_units)})"
            )
```

Re-export `GridGeometry` from `src/pymarxan/models/__init__.py` for discoverability
parity with `ConservationProblem`:

```python
from pymarxan.models.grid import GridGeometry
from pymarxan.models.problem import (
    ConservationProblem,
    apply_feature_overrides,
    has_geometry,
)

__all__ = [
    "ConservationProblem",
    "GridGeometry",
    "apply_feature_overrides",
    "has_geometry",
]
```

- [ ] **Step 4: Run the field tests to verify they pass**

Run: `/opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/models/test_grid.py -v`
Expected: PASS (16 tests — 13 GridGeometry + 3 field).

- [ ] **Step 5: Add the CHANGELOG entry**

Under `## [Unreleased]` → `### Added` in `CHANGELOG.md` (create the headers if empty):

```markdown
- **Grid-geometry model for raster-grid planning units (`GridGeometry`, S1).**
  ``models/grid.py`` — a pure-NumPy grid descriptor (origin + cell size + validity
  mask + CRS) whose valid cells are planning units, with an analytic rook-adjacency
  ``build_boundary()`` (matches the shapely ``compute_boundary``, without materializing
  a polygon per cell) — carried as a ``kw_only`` ``grid`` field on
  ``ConservationProblem``. First step toward raster-grid PUs; ingestion (S2) follows.
  +16 tests.
```

- [ ] **Step 6: Run the full check**

Run: `PATH="/opt/micromamba/envs/shiny/bin:$HOME/.local/bin:$PWD/.venv/bin:$PATH" make check`
Expected: green — 0 ruff, 0 mypy, full suite + 16 new. (`test_solutions_are_different` flake → rerun once.)

Note: the CLAUDE.md `micromamba.sh` activation path may not exist on this machine; the `PATH=...` prefix above is the working invocation.

- [ ] **Step 7: Commit**

```bash
git add src/pymarxan/models/problem.py src/pymarxan/models/__init__.py \
        tests/pymarxan/models/test_grid.py CHANGELOG.md
git commit -m "feat(models): carry GridGeometry as a kw_only grid field on ConservationProblem"
```

---

## Post-plan notes

- **Design review:** the user requested `multi-agent-design-review` — worth a pass on the boundary-math parity (the analytic vs shapely equivalence, incl. the masked/exposed-edge case) and the `kw_only`-field / import placement.
- **Parity:** adds a model field + a new boundary generator; no solver/objective math changes, and the analytic boundary is anchored to equal shapely `compute_boundary`. 35.0 anchor untouched (in the full suite).
- **Deferred (own specs):** S2 raster ingestion (`from_rasters`, `exactextract`), S3 scale (sparse), S4 UI + `has_geometry`/mapping. This S1 is the self-contained core they build on.
