# restoptr restoration data model — implementation plan

> **For agentic workers:** TDD, one bite-sized step at a time. Steps use `- [ ]`. Tests under the
> `shiny` micromamba env.

**Goal:** Add `pymarxan.restoration.RestorationProblem` — a standalone restoration data model (grid +
existing-habitat / restorable / cost cell states) with a `habitat_mask(restored)` bridge to
`compute_mesh`, `validate()`, and `from_arrays` / `from_rasters` ingestion.

**Architecture:** New module `src/pymarxan/restoration/problem.py`. The core (`RestorationProblem` +
`from_arrays`) is **rasterio-free** (inlined `_nodata_mask`); `from_rasters` lazily imports the S2
rasterio helpers from `pymarxan.spatial.raster`. Bridges to `compute_mesh` (v0.28.0).

**Tech stack:** Python 3.12+, numpy; rasterio only inside `from_rasters` (lazy). `from __future__
import annotations`, full type hints.

## Global constraints

- All arrays `(grid.n_pu,)` in the grid's row-major valid-cell (== PU) order.
- `RestorationProblem` is `@dataclass(eq=False)` (numpy fields).
- Pure new subpackage code — no solver/objective change; the parity anchor (35.0) is untouched.
- `import pymarxan.restoration` must not require rasterio (only `from_rasters` needs it, lazily).

---

### Task 1: `RestorationProblem` core (dataclass + bridge + validate + from_arrays)

**Files:**
- Create: `src/pymarxan/restoration/problem.py`
- Modify: `src/pymarxan/restoration/__init__.py` (export `RestorationProblem`)
- Test: `tests/pymarxan/restoration/test_problem.py`

**Interfaces produced:**
- `RestorationProblem(grid, existing_habitat, restorable, cost=None)` — `@dataclass(eq=False)`;
  `n_pu`, `habitat_mask(restored)`, `baseline_mesh(**kw)`, `restore_mesh(restored, **kw)`,
  `restoration_cost(restored)`, `validate() -> list[str]`.
- `RestorationProblem.from_arrays(existing_habitat_2d, restorable_2d, *, cost_2d=None, x_min, y_max,
  cell_width, cell_height, crs=None, mask_array=None, nodata=None) -> RestorationProblem`.

- [ ] **Step 1: Write the failing tests.** Create `tests/pymarxan/restoration/test_problem.py`:

```python
"""Tests for RestorationProblem (restoration data model)."""
from __future__ import annotations

import numpy as np
import pytest

from pymarxan.models.grid import GridGeometry
from pymarxan.restoration import MeshResult, RestorationProblem, compute_mesh


def _grid(nrow=3, ncol=3, mask=None):
    if mask is None:
        mask = np.ones((nrow, ncol), dtype=bool)
    return GridGeometry(x_min=0.0, y_max=float(nrow), cell_width=1.0, cell_height=1.0, mask=mask)


def _rp(existing, restorable, cost=None, nrow=3, ncol=3):
    return RestorationProblem(_grid(nrow, ncol), np.asarray(existing, bool),
                              np.asarray(restorable, bool), cost)


def test_cost_defaults_to_uniform():
    rp = _rp(np.zeros(9), np.ones(9))
    assert rp.cost.shape == (9,)
    assert np.all(rp.cost == 1.0)


def test_n_pu():
    assert _rp(np.zeros(9), np.ones(9)).n_pu == 9


def test_habitat_mask_unions_existing_and_restored():
    existing = np.zeros(9, bool); existing[0] = True
    restorable = np.zeros(9, bool); restorable[1] = restorable[2] = True
    rp = _rp(existing, restorable)
    restored = np.zeros(9, bool); restored[1] = True
    hm = rp.habitat_mask(restored)
    assert hm[0] and hm[1] and not hm[2]


def test_habitat_mask_rejects_non_restorable():
    rp = _rp(np.zeros(9), np.array([1, 1, 0, 0, 0, 0, 0, 0, 0], bool))
    bad = np.zeros(9, bool); bad[3] = True  # cell 3 is not restorable
    with pytest.raises(ValueError):
        rp.habitat_mask(bad)


def test_habitat_mask_wrong_length_raises():
    rp = _rp(np.zeros(9), np.ones(9))
    with pytest.raises(ValueError):
        rp.habitat_mask(np.zeros(5, bool))


def test_restore_mesh_matches_compute_mesh():
    existing = np.zeros(9, bool); existing[0] = True
    restorable = np.ones(9, bool); restorable[0] = False  # cell 0 already habitat
    rp = _rp(existing, restorable)
    restored = np.zeros(9, bool); restored[1] = restored[3] = True
    r = rp.restore_mesh(restored)
    assert isinstance(r, MeshResult)
    expected = compute_mesh(rp.grid, rp.habitat_mask(restored))
    assert r.mesh == pytest.approx(expected.mesh)


def test_baseline_mesh_is_existing_only():
    existing = np.ones(9, bool)
    rp = _rp(existing, np.zeros(9))
    assert rp.baseline_mesh().mesh == pytest.approx(compute_mesh(rp.grid, existing).mesh)


def test_restore_increases_mesh_over_baseline():
    # 1x3 strip, ends already habitat (2 patches), middle restorable -> restoring bridges -> higher.
    g = _grid(1, 3)
    existing = np.array([True, False, True])
    restorable = np.array([False, True, False])
    rp = RestorationProblem(g, existing, restorable)
    restored = np.array([False, True, False])
    assert rp.restore_mesh(restored).mesh > rp.baseline_mesh().mesh


def test_restoration_cost_sums_selected():
    cost = np.arange(9, dtype=float)  # 0..8
    rp = _rp(np.zeros(9), np.ones(9), cost=cost)
    restored = np.zeros(9, bool); restored[2] = restored[5] = True
    assert rp.restoration_cost(restored) == pytest.approx(7.0)  # 2 + 5


def test_validate_clean():
    existing = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0], bool)
    restorable = np.array([0, 1, 1, 0, 0, 0, 0, 0, 0], bool)
    assert _rp(existing, restorable).validate() == []


def test_validate_flags_overlap():
    both = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0], bool)
    errs = _rp(both, both).validate()
    assert any("disjoint" in e.lower() or "overlap" in e.lower() for e in errs)


def test_validate_flags_wrong_length_and_bad_cost():
    g = _grid(3, 3)
    rp = RestorationProblem(g, np.zeros(5, bool), np.ones(9, bool))  # existing wrong length
    assert rp.validate()  # non-empty
    rp2 = RestorationProblem(g, np.zeros(9, bool), np.ones(9, bool), np.full(9, -1.0))
    assert any("cost" in e.lower() for e in rp2.validate())


# --- from_arrays ---

def test_from_arrays_basic():
    existing = np.array([[1, 0, 0], [0, 0, 0], [0, 0, 1]], dtype=float)
    restorable = np.array([[0, 1, 1], [1, 1, 1], [1, 1, 0]], dtype=float)
    rp = RestorationProblem.from_arrays(existing, restorable, x_min=0.0, y_max=3.0,
                                        cell_width=1.0, cell_height=1.0)
    assert rp.n_pu == 9
    assert rp.existing_habitat.dtype == bool
    assert rp.existing_habitat.sum() == 2  # cells (0,0) and (2,2)
    assert rp.restorable.sum() == 7
    assert np.all(rp.cost == 1.0)
    assert rp.validate() == []  # existing/restorable disjoint by construction here


def test_from_arrays_validity_from_existing_footprint_with_nodata():
    nan = np.nan
    existing = np.array([[1, 0, nan], [0, 0, 0], [nan, 0, 1]], dtype=float)  # 2 nodata -> 7 valid
    restorable = np.zeros((3, 3), dtype=float)
    rp = RestorationProblem.from_arrays(existing, restorable, x_min=0.0, y_max=3.0,
                                        cell_width=1.0, cell_height=1.0, nodata=None)
    assert rp.n_pu == 7  # nodata cells dropped from the study region


def test_from_arrays_explicit_mask_precedence():
    existing = np.ones((2, 2), dtype=float)
    restorable = np.ones((2, 2), dtype=float)
    mask = np.array([[1, 1], [0, 1]], dtype=float)  # 3 valid
    rp = RestorationProblem.from_arrays(existing, restorable, x_min=0.0, y_max=2.0,
                                        cell_width=1.0, cell_height=1.0, mask_array=mask)
    assert rp.n_pu == 3


def test_from_arrays_cost_default_and_shape_check():
    existing = np.zeros((2, 2), dtype=float)
    restorable = np.ones((2, 2), dtype=float)
    rp = RestorationProblem.from_arrays(existing, restorable, x_min=0.0, y_max=2.0,
                                        cell_width=1.0, cell_height=1.0)
    assert np.all(rp.cost == 1.0)
    with pytest.raises(ValueError):
        RestorationProblem.from_arrays(existing, np.ones((3, 3)), x_min=0.0, y_max=2.0,
                                       cell_width=1.0, cell_height=1.0)


def test_from_arrays_empty_mask_raises():
    existing = np.full((2, 2), np.nan)
    with pytest.raises(ValueError):
        RestorationProblem.from_arrays(existing, np.zeros((2, 2)), x_min=0.0, y_max=2.0,
                                       cell_width=1.0, cell_height=1.0)
```

- [ ] **Step 2: Run to verify they fail.**
Run: `/opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/restoration/test_problem.py -q`
Expected: FAIL — `RestorationProblem` not importable.

- [ ] **Step 3: Implement `problem.py`.**

```python
"""RestorationProblem — restoration data model (grid + habitat/restorable/cost cell states).

Standalone domain model (cf. RiverNetwork / PhylogeneticTree) for restoptr-style restoration:
which restorable cells to convert to habitat, evaluated by ``compute_mesh`` via
``habitat_mask(restored)``. Core + ``from_arrays`` are rasterio-free (inlined ``_nodata_mask``);
``from_rasters`` lazily imports the S2 rasterio helpers.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

from pymarxan.models.grid import GridGeometry
from pymarxan.restoration.mesh import MeshResult, compute_mesh


def _nodata_mask(arr: np.ndarray, nodata: float | None) -> np.ndarray:
    """True where ``arr`` is nodata (NaN always; ``== nodata`` sentinel). Pure numpy (S2 parity)."""
    m = np.isnan(arr) if np.issubdtype(arr.dtype, np.floating) else np.zeros(arr.shape, bool)
    if nodata is not None:
        m = m | (arr == nodata)
    return np.asarray(m, dtype=bool)


@dataclass(eq=False)  # numpy fields break the auto __eq__ (repo convention, cf. GridGeometry)
class RestorationProblem:
    grid: GridGeometry
    existing_habitat: np.ndarray
    restorable: np.ndarray
    cost: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.existing_habitat = np.asarray(self.existing_habitat).astype(bool)
        self.restorable = np.asarray(self.restorable).astype(bool)
        if self.cost is None:
            self.cost = np.ones(self.grid.n_pu, dtype=float)
        else:
            self.cost = np.asarray(self.cost).astype(float)

    @property
    def n_pu(self) -> int:
        return self.grid.n_pu

    def _check_restored(self, restored: np.ndarray) -> np.ndarray:
        restored = np.asarray(restored).astype(bool)
        if restored.shape != (self.n_pu,):
            raise ValueError(f"restored must have length {self.n_pu}, got {restored.shape}")
        if bool((restored & ~self.restorable).any()):
            raise ValueError("restored contains cells that are not restorable")
        return restored

    def habitat_mask(self, restored: np.ndarray) -> np.ndarray:
        """Post-restoration habitat map: ``existing_habitat | restored`` (restored ⊆ restorable)."""
        return self.existing_habitat | self._check_restored(restored)

    def baseline_mesh(self, **mesh_kwargs) -> MeshResult:
        """MESH of the current (pre-restoration) habitat map."""
        return compute_mesh(self.grid, self.existing_habitat, **mesh_kwargs)

    def restore_mesh(self, restored: np.ndarray, **mesh_kwargs) -> MeshResult:
        """MESH of the post-restoration habitat map."""
        return compute_mesh(self.grid, self.habitat_mask(restored), **mesh_kwargs)

    def restoration_cost(self, restored: np.ndarray) -> float:
        """Total cost of a restoration plan."""
        restored = self._check_restored(restored)
        assert self.cost is not None  # set in __post_init__
        return float(self.cost[restored].sum())

    def validate(self) -> list[str]:
        """Return a list of data errors (empty when valid); does not raise."""
        errs: list[str] = []
        n = self.n_pu
        for name, arr in (("existing_habitat", self.existing_habitat),
                          ("restorable", self.restorable), ("cost", self.cost)):
            if arr is None or arr.shape != (n,):
                errs.append(f"{name} must have length {n}, got "
                            f"{None if arr is None else arr.shape}")
        if (self.existing_habitat.shape == (n,) and self.restorable.shape == (n,)
                and bool((self.existing_habitat & self.restorable).any())):
            errs.append("existing_habitat and restorable must be disjoint "
                        "(an already-habitat cell cannot be restored)")
        if self.cost is not None and self.cost.shape == (n,) and not (
                np.all(np.isfinite(self.cost)) and np.all(self.cost >= 0)):
            errs.append("cost must be finite and >= 0")
        return errs

    @classmethod
    def from_arrays(
        cls,
        existing_habitat: np.ndarray,
        restorable: np.ndarray,
        *,
        cost: np.ndarray | None = None,
        x_min: float,
        y_max: float,
        cell_width: float,
        cell_height: float,
        crs: str | None = None,
        mask_array: np.ndarray | None = None,
        nodata: float | None = None,
    ) -> RestorationProblem:
        eh = np.asarray(existing_habitat, dtype=float)
        rs = np.asarray(restorable, dtype=float)
        shape = eh.shape
        if len(shape) != 2:
            raise ValueError(f"arrays must be 2-D, got shape {shape}")
        for label, a in (("restorable", rs),
                         ("cost", None if cost is None else np.asarray(cost, float)),
                         ("mask_array", None if mask_array is None else np.asarray(mask_array, float))):
            if a is not None and a.shape != shape:
                raise ValueError(f"{label} shape {a.shape} != existing_habitat shape {shape}")

        # validity precedence: explicit mask -> existing_habitat non-nodata footprint
        if mask_array is not None:
            m = np.asarray(mask_array, dtype=float)
            valid = (m != 0) & ~_nodata_mask(m, nodata)
        else:
            valid = ~_nodata_mask(eh, nodata)
        if not valid.any():
            raise ValueError("no valid cells (the validity mask is empty)")

        rows, cols = np.nonzero(valid)  # row-major == PU order
        grid = GridGeometry(x_min, y_max, cell_width, cell_height, valid, crs)

        existing_vec = (eh[rows, cols] > 0) & ~_nodata_mask(eh, nodata)[rows, cols]
        rest_vec = (rs[rows, cols] > 0) & ~_nodata_mask(rs, nodata)[rows, cols]
        if cost is None:
            cost_vec = np.ones(rows.size, dtype=float)
        else:
            c = np.asarray(cost, dtype=float)
            cost_vec = c[rows, cols]
            nd = _nodata_mask(c, nodata)[rows, cols]
            if nd.any():
                warnings.warn(f"{int(nd.sum())} valid cell(s) have nodata cost; defaulting to 1.0",
                              stacklevel=2)
            cost_vec = np.where(nd, 1.0, cost_vec)
        return cls(grid, existing_vec, rest_vec, cost_vec)
```

Add the `from_rasters` classmethod (Task 2). Update `src/pymarxan/restoration/__init__.py`:

```python
from pymarxan.restoration.mesh import MeshResult, compute_mesh
from pymarxan.restoration.problem import RestorationProblem

__all__ = ["MeshResult", "RestorationProblem", "compute_mesh"]
```

- [ ] **Step 4: Run to verify pass.**
Run: `/opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/restoration/test_problem.py -q`
Expected: PASS (Task-1 tests). ruff + mypy clean
(`mypy src/pymarxan/restoration/ --ignore-missing-imports`; the `assert self.cost is not None`
guards the optional-field access).

- [ ] **Step 5: Commit** — `feat(restoration): RestorationProblem data model + from_arrays`.

---

### Task 2: `from_rasters` ingestion

**Files:**
- Modify: `src/pymarxan/restoration/problem.py` (add `from_rasters` classmethod)
- Test: `tests/pymarxan/restoration/test_problem_rasters.py` (marker: `spatial`)

**Interfaces produced:**
- `RestorationProblem.from_rasters(existing_habitat, restorable, *, cost=None, band=1)
  -> RestorationProblem`.

- [ ] **Step 1: Write the failing tests.** Create `tests/pymarxan/restoration/test_problem_rasters.py`:

```python
"""from_rasters ingestion for RestorationProblem (rasterio round-trip)."""
from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.spatial
rasterio = pytest.importorskip("rasterio")
from rasterio.transform import from_origin  # noqa: E402

from pymarxan.restoration import RestorationProblem  # noqa: E402


def _write(path, arr, *, nodata=None):
    arr = np.asarray(arr, dtype="float32")
    tf = from_origin(0.0, arr.shape[0], 1.0, 1.0)  # x_min=0, y_max=nrow, 1x1 cells, north-up
    with rasterio.open(path, "w", driver="GTiff", height=arr.shape[0], width=arr.shape[1],
                       count=1, dtype="float32", transform=tf, nodata=nodata) as dst:
        dst.write(arr, 1)


def test_from_rasters_matches_from_arrays(tmp_path):
    existing = np.array([[1, 0, 0], [0, 0, 0], [0, 0, 1]], dtype="float32")
    restorable = np.array([[0, 1, 1], [1, 1, 1], [1, 1, 0]], dtype="float32")
    ep, rpath = tmp_path / "e.tif", tmp_path / "r.tif"
    _write(ep, existing)
    _write(rpath, restorable)
    rp = RestorationProblem.from_rasters(ep, rpath)
    ref = RestorationProblem.from_arrays(existing.astype(float), restorable.astype(float),
                                         x_min=0.0, y_max=3.0, cell_width=1.0, cell_height=1.0)
    assert rp.n_pu == ref.n_pu == 9
    assert np.array_equal(rp.existing_habitat, ref.existing_habitat)
    assert np.array_equal(rp.restorable, ref.restorable)


def test_from_rasters_misaligned_raises(tmp_path):
    _write(tmp_path / "e.tif", np.ones((3, 3), "float32"))
    _write(tmp_path / "r.tif", np.ones((2, 2), "float32"))  # wrong shape
    with pytest.raises(ValueError):
        RestorationProblem.from_rasters(tmp_path / "e.tif", tmp_path / "r.tif")
```

- [ ] **Step 2: Run to verify fail** — `from_rasters` not implemented.

- [ ] **Step 3: Implement `from_rasters`** in `problem.py`:

```python
    @classmethod
    def from_rasters(
        cls,
        existing_habitat: str,
        restorable: str,
        *,
        cost: str | None = None,
        band: int = 1,
    ) -> RestorationProblem:
        """Build from aligned single-band rasters (rasterio). Lazily imports the S2 helpers."""
        from pymarxan.spatial.raster import _check_align, _read, _require_north_up

        eh, tf, shape, crs = _read(existing_habitat, band)
        _require_north_up(tf)
        rs = _read_aligned_layer(restorable, band, "restorable", tf, shape, crs, _read, _check_align)
        cost_arr = (None if cost is None
                    else _read_aligned_layer(cost, band, "cost", tf, shape, crs, _read, _check_align))
        # north-up transform: x_min = c, y_max = f, cell_width = a, cell_height = -e
        return cls.from_arrays(
            eh, rs, cost=cost_arr,
            x_min=tf.c, y_max=tf.f, cell_width=tf.a, cell_height=-tf.e,
            crs=(str(crs) if crs is not None else None),
        )
```

with a tiny module-level helper (keeps the lazy import local):

```python
def _read_aligned_layer(path, band, label, ref_tf, ref_shape, ref_crs, _read, _check_align):
    arr, tf, shape, crs = _read(path, band)
    _check_align(label, shape, tf, crs, ref_tf, ref_shape, ref_crs)
    return arr
```

*(Or import `_read_aligned` from `spatial.raster` directly — it does the read+check in one call;
prefer that if its signature matches: `_read_aligned(path, band, label, ref_tf, ref_shape,
ref_crs)`.)*

- [ ] **Step 4: Run to verify pass** (marker `spatial`, needs rasterio in the shiny env).

- [ ] **Step 5: Parity + CHANGELOG + full check.**
Run: `/opt/micromamba/envs/shiny/bin/python examples/validate_marxan_parity.py` → 35.0.
Add to `CHANGELOG.md` `[Unreleased]`:

```markdown
- **Restoration data model (restoptr-style, `pymarxan.restoration`).** `RestorationProblem` (grid +
  existing-habitat / restorable / cost cell states) makes MESH actionable: `habitat_mask(restored)`
  bridges a restoration plan to `compute_mesh`, with `baseline_mesh` / `restore_mesh` /
  `restoration_cost` / `validate`, built via `from_arrays` (pure numpy) or `from_rasters` (aligned
  single-band rasters). The budget/min-max-restore constraints and the MESH-maximizing optimizer are
  follow-ons.
```

Run: `PATH="/opt/micromamba/envs/shiny/bin:$HOME/.local/bin:$PWD/.venv/bin:$PATH" make check`
- [ ] **Step 6: Commit** — `feat(restoration): from_rasters ingestion + docs`.

## Self-review

- **Spec coverage:** dataclass + `cost` default ✓; `habitat_mask` union + subset check ✓;
  `baseline_mesh`/`restore_mesh`/`restoration_cost` ✓; `validate` (disjoint/length/cost) ✓;
  `from_arrays` (validity precedence, binarize, cost/nodata, shape/empty checks) ✓; `from_rasters`
  (align + delegate, misalignment raises) ✓. Rasterio-free core (inline `_nodata_mask`, lazy import
  in `from_rasters`) ✓.
- **Placeholders:** none — all steps have concrete code.
- **Type consistency:** `cost: np.ndarray | None` with `__post_init__` fill + `assert ... is not
  None` guard at the read site; `RestorationProblem` fields/methods match the tests.
- **Design-review handoff:** architect / grounding / independent-redesign on the model boundary,
  the `from_arrays` validity precedence, the rasterio-free contract, and the `compute_mesh` bridge
  (science is settled from the MESH review). Grounding should RUN `from_arrays`/`from_rasters` and
  confirm the `_read_aligned` signature reused from `spatial.raster`.
```
