# restoptr MESH measure — implementation plan

> **For agentic workers:** TDD, one bite-sized step at a time. Steps use `- [ ]`.

**Goal:** Add `pymarxan.restoration.compute_mesh` — the effective mesh size (Jaeger 2000) of a
habitat map on a raster `GridGeometry` — as a pure, independently testable landscape-index measure.

**Architecture:** New pure-Python core subpackage `src/pymarxan/restoration/` (no UI, no solver deps),
alongside `adequacy`/`phylo`/`zonation`. One scipy call (`scipy.ndimage.label`, already a dep) for
raster connected-components; the rest is numpy. No `ConservationProblem` dependency.

**Tech stack:** Python 3.12+, numpy, `scipy.ndimage.label`. `from __future__ import annotations`,
full type hints. Tests under the `shiny` micromamba env.

> **⚠ Review fixes folded** (`...-review.md`, science VERIFIED, no CRITICAL/HIGH). The code below
> is superseded by these where they conflict: `MeshResult` is `@dataclass(eq=False)` (numpy field)
> with `coherence`/`division` `@property`s; `compute_mesh` validates `cell_area > 0`; `n_patches`
> derives from `patch_areas.size`; the `mesh.py`/`__init__.py` docstrings record the
> `scipy.ndimage.label` vs `count_connected_components` distinction and the `restoration` vs
> `connectivity` split (IIC/PC home = `restoration`); tests add `cell_area<=0` + coherence/division;
> `test_l_patch...` → `test_block_plus_isolated_cell`.

## Global constraints

- MESH `= Σ A_i² / A_total`; `A_total = grid.n_pu · cell_area` (**total landscape area — every
  valid cell — NOT habitat area**); `A_i` = (cells in patch) · cell_area; patches = connected
  components of habitat cells (rook default / queen optional).
- Pure new subpackage — no solver/objective change; the parity anchor (35.0) is untouched.
- `habitat_mask` is `(n_pu,)` bool in the grid's row-major valid-cell order (== PU order).

---

### Task 1: `compute_mesh` + `MeshResult`

**Files:**
- Create: `src/pymarxan/restoration/__init__.py`, `src/pymarxan/restoration/mesh.py`
- Test: `tests/pymarxan/restoration/test_mesh.py`

**Interfaces produced:**
- `MeshResult(mesh: float, n_patches: int, patch_areas: np.ndarray, total_area: float)` (dataclass).
- `compute_mesh(grid: GridGeometry, habitat_mask: np.ndarray, *, connectivity: str = "rook",
  cell_area: float | None = None) -> MeshResult`.

- [ ] **Step 1: Write the failing tests.** Create `tests/pymarxan/restoration/test_mesh.py`:

```python
"""Tests for restoptr-style MESH (effective mesh size, Jaeger 2000)."""
from __future__ import annotations

import numpy as np
import pytest

from pymarxan.models.grid import GridGeometry
from pymarxan.restoration import MeshResult, compute_mesh


def _grid(nrow=3, ncol=3, mask=None):
    if mask is None:
        mask = np.ones((nrow, ncol), dtype=bool)
    return GridGeometry(x_min=0.0, y_max=float(nrow), cell_width=1.0, cell_height=1.0, mask=mask)


def test_full_habitat_single_patch_is_max():
    g = _grid(3, 3)  # A_total = 9, cell_area = 1
    r = compute_mesh(g, np.ones(9, bool), cell_area=1.0)
    assert isinstance(r, MeshResult)
    assert r.n_patches == 1
    assert r.total_area == 9.0
    assert r.mesh == pytest.approx(9.0)  # 9^2 / 9 = 9 == A_total (max)


def test_no_habitat_is_zero():
    g = _grid(3, 3)
    r = compute_mesh(g, np.zeros(9, bool), cell_area=1.0)
    assert r.n_patches == 0
    assert r.mesh == 0.0
    assert list(r.patch_areas) == []


def test_two_separated_single_cells():
    # habitat at opposite corners (0,0) and (2,2): non-adjacent under rook -> 2 patches.
    g = _grid(3, 3)
    mask = np.zeros(9, bool)
    mask[0] = True   # cell (0,0), row-major index 0
    mask[8] = True   # cell (2,2), row-major index 8
    r = compute_mesh(g, mask, cell_area=1.0)
    assert r.n_patches == 2
    assert r.mesh == pytest.approx((1 + 1) / 9)  # 0.2222


def test_l_patch_plus_isolated_cell():
    # 4-cell connected block (top-left 2x2) + 1 isolated cell (2,2). areas 4 and 1.
    g = _grid(3, 3)
    mask = np.zeros(9, bool)
    for i in (0, 1, 3, 4):  # (0,0),(0,1),(1,0),(1,1) -> connected 2x2
        mask[i] = True
    mask[8] = True  # (2,2) isolated
    r = compute_mesh(g, mask, cell_area=1.0)
    assert r.n_patches == 2
    assert sorted(r.patch_areas, reverse=True) == [4.0, 1.0]
    assert r.mesh == pytest.approx((16 + 1) / 9)  # 1.8889


def test_rook_vs_queen_diagonal():
    # two diagonally-touching cells: 2 patches under rook, 1 under queen.
    g = _grid(2, 2)
    mask = np.array([True, False, False, True])  # (0,0) and (1,1)
    assert compute_mesh(g, mask, connectivity="rook", cell_area=1.0).n_patches == 2
    assert compute_mesh(g, mask, connectivity="queen", cell_area=1.0).n_patches == 1


def test_mesh_scales_with_cell_area():
    g = _grid(3, 3)
    mask = np.ones(9, bool)
    base = compute_mesh(g, mask, cell_area=1.0).mesh
    scaled = compute_mesh(g, mask, cell_area=4.0).mesh
    assert scaled == pytest.approx(4.0 * base)  # MESH linear in cell_area


def test_default_cell_area_from_grid():
    # cell 2x3 -> cell_area 6; full 2x2 grid, one patch: A_total = 4*6 = 24, mesh = 24.
    g = GridGeometry(x_min=0.0, y_max=6.0, cell_width=2.0, cell_height=3.0,
                     mask=np.ones((2, 2), bool))
    r = compute_mesh(g, np.ones(4, bool))
    assert r.total_area == pytest.approx(24.0)
    assert r.mesh == pytest.approx(24.0)


def test_monotone_bridging_increases_mesh():
    # 1x3 strip: two ends habitat (2 patches) -> add the middle -> 1 patch, MESH increases.
    g = _grid(1, 3)
    ends = np.array([True, False, True])
    bridged = np.array([True, True, True])
    assert compute_mesh(g, bridged, cell_area=1.0).mesh > compute_mesh(g, ends, cell_area=1.0).mesh


def test_masked_nonrectangular_grid_mapping():
    # invalid centre cell; habitat on the 8 border cells, all rook-connected around the hole
    # -> 1 ring patch. Confirms habitat_mask maps to the right 2-D positions.
    m2d = np.ones((3, 3), bool)
    m2d[1, 1] = False  # centre invalid -> 8 valid PUs
    g = _grid(3, 3, mask=m2d)
    r = compute_mesh(g, np.ones(8, bool), cell_area=1.0)
    assert r.total_area == 8.0
    assert r.n_patches == 1
    assert r.mesh == pytest.approx(64.0 / 8.0)  # 8^2 / 8 = 8


def test_wrong_length_mask_raises():
    g = _grid(3, 3)
    with pytest.raises(ValueError):
        compute_mesh(g, np.ones(5, bool))


def test_unknown_connectivity_raises():
    g = _grid(3, 3)
    with pytest.raises(ValueError):
        compute_mesh(g, np.ones(9, bool), connectivity="diagonal")
```

- [ ] **Step 2: Run to verify they fail.**

Run: `/opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/restoration/test_mesh.py -q`
Expected: FAIL — `pymarxan.restoration` does not exist.

- [ ] **Step 3: Implement.** Create `src/pymarxan/restoration/mesh.py`:

```python
"""restoptr-style MESH — effective mesh size (Jaeger 2000) of a habitat map on a raster grid."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from pymarxan.models.grid import GridGeometry

_STRUCTURES = {
    "rook": np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=int),
    "queen": np.ones((3, 3), dtype=int),
}


@dataclass
class MeshResult:
    """Effective mesh size of a habitat map. ``mesh`` is m_eff in area units."""

    mesh: float
    n_patches: int
    patch_areas: np.ndarray  # (n_patches,) area per patch, descending
    total_area: float  # A_total = n_valid_cells · cell_area


def compute_mesh(
    grid: GridGeometry,
    habitat_mask: np.ndarray,
    *,
    connectivity: str = "rook",
    cell_area: float | None = None,
) -> MeshResult:
    """MESH (effective mesh size, Jaeger 2000): ``Σ A_i² / A_total`` over habitat patches
    (connected components), with ``A_total`` = total landscape area (every valid cell).

    ``habitat_mask`` is ``(grid.n_pu,)`` bool in the grid's row-major valid-cell (== PU) order.
    """
    if connectivity not in _STRUCTURES:
        msg = f"connectivity must be 'rook' or 'queen', got {connectivity!r}"
        raise ValueError(msg)
    habitat_mask = np.asarray(habitat_mask).astype(bool)
    if habitat_mask.shape != (grid.n_pu,):
        msg = f"habitat_mask must have length {grid.n_pu}, got {habitat_mask.shape}"
        raise ValueError(msg)

    area = float(cell_area) if cell_area is not None else grid.cell_width * grid.cell_height
    total_area = grid.n_pu * area

    hab2d = np.zeros(grid.shape, dtype=bool)
    hab2d[grid.mask] = habitat_mask
    labels, n = ndimage.label(hab2d, structure=_STRUCTURES[connectivity])
    counts = np.bincount(labels.ravel())[1:] if n > 0 else np.array([], dtype=np.int64)
    patch_areas = np.sort(counts.astype(float) * area)[::-1]

    mesh = float((patch_areas**2).sum() / total_area) if total_area > 0 else 0.0
    return MeshResult(
        mesh=mesh,
        n_patches=int(n),
        patch_areas=patch_areas,
        total_area=float(total_area),
    )
```

Create `src/pymarxan/restoration/__init__.py`:

```python
"""restoptr-style ecological restoration planning (landscape indices)."""
from __future__ import annotations

from pymarxan.restoration.mesh import MeshResult, compute_mesh

__all__ = ["MeshResult", "compute_mesh"]
```

- [ ] **Step 4: Run to verify they pass.**

Run: `/opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/restoration/test_mesh.py -q`
Expected: PASS (11 tests).

- [ ] **Step 5: ruff + mypy.**

Run: `ruff check src/pymarxan/restoration/ tests/pymarxan/restoration/ && mypy src/pymarxan/restoration/`
Expected: clean. (mypy `no-any-return` guard: `mesh` is wrapped in `float(...)`.)

- [ ] **Step 6: Parity + CHANGELOG.**

Run: `/opt/micromamba/envs/shiny/bin/python examples/validate_marxan_parity.py` → 35.0 intact.
Add to `CHANGELOG.md` `[Unreleased]`:

```markdown
### Added

- **Landscape fragmentation measure (restoptr-style, `pymarxan.restoration`).** `compute_mesh(grid,
  habitat_mask)` returns the **effective mesh size** (MESH, Jaeger 2000) of a habitat map on a
  raster `GridGeometry` — `Σ patchArea² / totalLandscapeArea` over habitat patches (connected
  components; rook 4-connectivity by default, queen optional) — the flagship index of restoptr-style
  restoration planning (Justeau-Allaire et al. 2021). Pure measure; no solver change. The
  restoration data model and a MESH-maximizing optimizer are follow-ons.
```

- [ ] **Step 7: Full check + commit.**

Run: `PATH="/opt/micromamba/envs/shiny/bin:$HOME/.local/bin:$PWD/.venv/bin:$PATH" make check`
Commit: `feat(restoration): compute_mesh — restoptr effective mesh size (Jaeger 2000)`.

## Self-review

- **Spec coverage:** MESH formula ✓ (Step 3), A_total=landscape area ✓, rook/queen ✓, cell_area
  default ✓, all edge cases (empty/full/wrong-length/unknown-connectivity/masked-grid) ✓ have tests.
- **Placeholders:** none — all steps have concrete code.
- **Type consistency:** `MeshResult` fields and `compute_mesh` signature match the spec and the
  tests; `patch_areas` is a float ndarray sorted descending in both the impl and the test assertions.
- **Design-review handoff:** run architect / grounding / **scientific-accuracy (scite)** on the
  MESH formula + the `A_total` denominator (the parity-critical detail) + the Jaeger/Justeau-Allaire
  citations before merge — the scientific lens is load-bearing here.
