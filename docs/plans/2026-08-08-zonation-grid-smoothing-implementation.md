# Grid-Convolution Smoothing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Raster-scale distribution smoothing for grid problems: `GridSmoothingSpec` → truncated-window 2-D negative-exponential convolution, mass-conserving, sparse column-wise, no PU cap.

**Architecture:** Math in `connectivity/smoothing.py` (`smooth_distribution_grid`, mirroring `smooth_distribution`'s one-or-many contract); spec dataclass in `zonation/smoothing.py`; isinstance dispatch in `rank_removal`'s smoothing branch building a sparse smoothed CSR column-by-column; type widening in `ZonationSolver`.

**Tech Stack:** numpy, scipy.signal.oaconvolve, scipy.sparse; pytest.

**Spec:** `docs/plans/2026-08-08-zonation-grid-smoothing-design.md` — §3 (math, normative), §5 (validation), §6 (contracts), §7 (tests).

## Global Constraints

- Tests ONLY via `/opt/micromamba/envs/shiny/bin/pytest`; ruff `~/.local/bin/ruff` (line 99); mypy clean; `from __future__ import annotations`.
- `SmoothingSpec` behavior byte-for-byte unchanged, including the 50k cap (which must NOT apply to `GridSmoothingSpec`).
- The v0.32 engine (both selection paths) is consumed unchanged — grid smoothing only changes how the CSR is built. Never materialize a dense `(n_pu, n_feat)` matrix on the grid path.
- PU order invariant: scatter/gather uses `np.flatnonzero(mask.reshape(-1))` (row-major valid-cell order == CSR row order). A mismatch is silent wrong ranking.
- Branch `feat/zonation-grid-smoothing` (created; spec committed). Parity anchor untouched; `make check` green at end. Known SA flake: rerun-alone protocol.

---

### Task 1: `smooth_distribution_grid` (the math)

**Files:**
- Modify: `src/pymarxan/connectivity/smoothing.py` (append)
- Create: `tests/pymarxan/connectivity/test_grid_smoothing.py`

**Interfaces:**
- Consumes: `pymarxan.models.grid.GridGeometry` (fields `x_min, y_max, cell_width, cell_height, mask, crs`; `shape`, `cell_centroids()`), existing `smooth_distribution` + `distance_matrix_from_points` (the vector oracle for tests).
- Produces: `smooth_distribution_grid(amounts: np.ndarray, grid: GridGeometry, alpha: float, *, truncate: float = 1e-9) -> np.ndarray` — `(n_pu,)` or `(n_pu, m)` in, same shape out. Tasks 2–3 call it.

- [ ] **Step 1: Write the failing tests**

```python
"""Grid-convolution distribution smoothing (design 2026-08-08 §3/§6).

The vector path (smooth_distribution over a dense kernel) is the oracle: on a
grid whose truncation window covers every pairwise offset, the grid kernel is
IDENTICAL to the untruncated vector kernel, so results agree to FP regrouping
(allclose, never bitwise — convolution sums in a different order).
"""
from __future__ import annotations

import numpy as np
import pytest

from pymarxan.connectivity.smoothing import (
    distance_matrix_from_points,
    smooth_distribution,
    smooth_distribution_grid,
)
from pymarxan.models.grid import GridGeometry


def _grid(mask: np.ndarray, cw: float = 1.0, ch: float = 1.0) -> GridGeometry:
    return GridGeometry(
        x_min=0.0, y_max=float(mask.shape[0]) * ch, cell_width=cw, cell_height=ch,
        mask=np.asarray(mask, dtype=bool),
    )


def _vector_oracle(q: np.ndarray, grid: GridGeometry, alpha: float) -> np.ndarray:
    d = distance_matrix_from_points(grid.cell_centroids())
    return smooth_distribution(q, d, alpha)


@pytest.mark.parametrize("cw,ch", [(1.0, 1.0), (2.0, 0.5)])
def test_grid_matches_vector_full_window(cw: float, ch: float) -> None:
    rng = np.random.default_rng(0)
    mask = np.ones((6, 5), dtype=bool)
    grid = _grid(mask, cw, ch)
    q = rng.uniform(0, 3, size=grid.n_pu)
    # alpha small enough that the clipped window covers the whole grid.
    out = smooth_distribution_grid(q, grid, alpha=0.3)
    np.testing.assert_allclose(out, _vector_oracle(q, grid, 0.3), rtol=1e-10, atol=1e-13)


def test_grid_matches_vector_with_holes() -> None:
    rng = np.random.default_rng(1)
    mask = np.ones((7, 6), dtype=bool)
    mask[2:4, 2:4] = False  # interior hole
    mask[0, 0] = False      # corner notch
    grid = _grid(mask)
    q = rng.uniform(0, 2, size=grid.n_pu)
    out = smooth_distribution_grid(q, grid, alpha=0.4)
    np.testing.assert_allclose(out, _vector_oracle(q, grid, 0.4), rtol=1e-10, atol=1e-13)


def test_multi_column_matches_per_column() -> None:
    rng = np.random.default_rng(2)
    grid = _grid(np.ones((5, 5), dtype=bool))
    q = rng.uniform(0, 1, size=(grid.n_pu, 3))
    out = smooth_distribution_grid(q, grid, alpha=0.5)
    assert out.shape == q.shape
    for j in range(3):
        np.testing.assert_array_equal(
            out[:, j], smooth_distribution_grid(q[:, j], grid, alpha=0.5)
        )


@pytest.mark.parametrize("truncate", [1e-9, 1e-2])
def test_mass_conserved_at_any_truncation(truncate: float) -> None:
    rng = np.random.default_rng(3)
    mask = np.ones((10, 9), dtype=bool)
    mask[5, 3:6] = False
    grid = _grid(mask)
    q = rng.uniform(0, 4, size=grid.n_pu)
    out = smooth_distribution_grid(q, grid, alpha=1.2, truncate=truncate)
    # Conservation is exact at ANY truncation: the normalizer uses the same
    # truncated kernel (design §3).
    np.testing.assert_allclose(out.sum(), q.sum(), rtol=1e-12)


def test_single_source_compact_support_exact_zeros() -> None:
    mask = np.ones((15, 15), dtype=bool)
    grid = _grid(mask)
    q = np.zeros(grid.n_pu)
    src = 7 * 15 + 7  # centre cell (full mask: PU index == flat index)
    q[src] = 5.0
    out = smooth_distribution_grid(q, grid, alpha=1.5, truncate=1e-3)
    r = int(np.ceil(-np.log(1e-3) / 1.5))  # window radius in cells
    og = out.reshape(15, 15)
    inside = np.zeros((15, 15), dtype=bool)
    inside[max(0, 7 - r) : 7 + r + 1, max(0, 7 - r) : 7 + r + 1] = True
    # EXACT zeros outside the window (FFT dust masked by support dilation).
    assert (og[~inside] == 0.0).all()
    assert og[7, 7] > 0.0


def test_isolated_invalid_region_no_nan() -> None:
    # Invalid cells far from any valid cell have Z == 0; the masked division
    # must not let 0/0 NaN leak into the convolution (design §3 fix).
    mask = np.zeros((20, 20), dtype=bool)
    mask[:3, :3] = True  # valid island in one corner; rest invalid
    grid = _grid(mask)
    q = np.arange(1.0, grid.n_pu + 1)
    out = smooth_distribution_grid(q, grid, alpha=2.0, truncate=1e-6)
    assert np.isfinite(out).all()
    np.testing.assert_allclose(out.sum(), q.sum(), rtol=1e-12)


def test_huge_alpha_near_identity_and_single_column_grid() -> None:
    grid = _grid(np.ones((4, 4), dtype=bool))
    q = np.arange(1.0, 17.0)
    out = smooth_distribution_grid(q, grid, alpha=50.0)
    np.testing.assert_allclose(out, q, rtol=1e-8)  # self-weight dominates
    col = _grid(np.ones((6, 1), dtype=bool))  # x-radius clips to 0
    qc = np.arange(1.0, 7.0)
    outc = smooth_distribution_grid(qc, col, alpha=0.5)
    np.testing.assert_allclose(outc.sum(), qc.sum(), rtol=1e-12)


def test_nonnegative_under_fft_roundoff() -> None:
    # Design-review HIGH (found independently by two lenses): a high-amplitude
    # blob plus a remote unit source makes true in-window corner values sink
    # below the FFT noise floor — unclamped output goes NEGATIVE inside the
    # reach mask (3,570 cells at default truncate in the review's probe), which
    # would silently break the heap's monotonicity invariant downstream.
    mask = np.ones((256, 256), dtype=bool)
    grid = _grid(mask)
    q = np.zeros(grid.n_pu)
    qg = q.reshape(256, 256)
    qg[40:80, 40:80] = 1e12  # high-amplitude blob
    qg[200, 200] = 1.0       # remote lone unit source
    out = smooth_distribution_grid(qg.reshape(-1), grid, alpha=0.5, truncate=1e-9)
    assert (out >= 0.0).all()
    np.testing.assert_allclose(out.sum(), qg.sum(), rtol=1e-10)


def test_matches_explicit_truncated_kernel_oracle() -> None:
    # Pin the two-convolution formulation against an INDEPENDENTLY built
    # window-bounded column-normalized kernel matrix (stronger than the
    # full-window case: exercises the truncation itself). Both review lenses
    # measured <=6.3e-16.
    rng = np.random.default_rng(5)
    mask = np.ones((10, 9), dtype=bool)
    mask[4, 2:5] = False
    grid = _grid(mask)
    q = rng.uniform(0, 3, size=grid.n_pu)
    alpha, truncate = 1.0, 1e-2
    out = smooth_distribution_grid(q, grid, alpha, truncate=truncate)
    # Explicit oracle: truncated window, column-normalized over valid cells.
    cents = grid.cell_centroids()
    rx = int(np.ceil(-np.log(truncate) / (alpha * 1.0)))
    d = distance_matrix_from_points(cents)
    dxy = np.abs(cents[:, None, :] - cents[None, :, :])
    in_window = (dxy[:, :, 0] <= rx * 1.0 + 1e-12) & (dxy[:, :, 1] <= rx * 1.0 + 1e-12)
    K = np.where(in_window, np.exp(-alpha * d), 0.0)
    Kn = K / K.sum(axis=0)
    np.testing.assert_allclose(out, Kn @ q, rtol=1e-10, atol=1e-13)


def test_validation() -> None:
    grid = _grid(np.ones((3, 3), dtype=bool))
    q = np.ones(9)
    with pytest.raises(ValueError, match="alpha"):
        smooth_distribution_grid(q, grid, alpha=0.0)
    for bad in (0.0, 1.0, -0.5, 2.0):
        with pytest.raises(ValueError, match="truncate"):
            smooth_distribution_grid(q, grid, alpha=1.0, truncate=bad)
    with pytest.raises(ValueError, match="rows"):
        smooth_distribution_grid(np.ones(5), grid, alpha=1.0)
```

- [ ] **Step 2: Run to verify failure**

Run: `/opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/connectivity/test_grid_smoothing.py -v`
Expected: collection error / ImportError (`smooth_distribution_grid` undefined).

- [ ] **Step 3: Implement.** Append to `src/pymarxan/connectivity/smoothing.py`:

```python
def smooth_distribution_grid(
    amounts: np.ndarray,
    grid: "GridGeometry",
    alpha: float,
    *,
    truncate: float = 1e-9,
) -> np.ndarray:
    """Grid-convolution distribution smoothing (raster analogue of
    :func:`smooth_distribution`).

    Same negative-exponential, mass-conserving (source-normalised) kernel, but
    evaluated as a truncated-window 2-D convolution on the grid — O(cells·log)
    per feature instead of an n×n kernel. Mass is conserved over valid cells
    EXACTLY at any ``truncate``, because the normaliser uses the same truncated
    kernel (each source's outgoing weights sum to 1 over its reachable valid
    cells). Truncation is window-bounded: when the clipped window covers the
    whole grid, results equal the untruncated vector kernel's up to FP
    regrouping (allclose, not bitwise).

    Args:
        amounts: ``(n_pu,)`` or ``(n_pu, m)`` per-PU amounts, rows in the
            grid's row-major valid-cell order (== CSR row order).
        grid: The problem's :class:`GridGeometry`.
        alpha: Negative-exponential decay rate (> 0), per CRS distance unit.
        truncate: Window cutoff in kernel-value terms, in (0, 1): the window
            radius per axis is ``ceil(-ln(truncate) / (alpha·cell_size))``,
            clipped to the grid extent.

    Returns:
        Smoothed amounts, same shape as ``amounts``.
    """
    from scipy.signal import oaconvolve

    if alpha <= 0:
        raise ValueError(f"alpha must be > 0, got {alpha}")
    if not 0.0 < truncate < 1.0:
        raise ValueError(f"truncate must be in (0, 1), got {truncate}")
    amounts = np.asarray(amounts, dtype=float)
    one_d = amounts.ndim == 1
    cols = amounts[:, None] if one_d else amounts
    mask = grid.mask
    nrows, ncols_g = mask.shape
    flat_valid = np.flatnonzero(mask.reshape(-1))
    if cols.shape[0] != flat_valid.size:
        raise ValueError(
            f"amounts must have {flat_valid.size} rows, got {cols.shape[0]}"
        )

    cw, ch = abs(grid.cell_width), abs(grid.cell_height)
    rx = min(int(np.ceil(-np.log(truncate) / (alpha * cw))), ncols_g - 1)
    ry = min(int(np.ceil(-np.log(truncate) / (alpha * ch))), nrows - 1)
    dy, dx = np.meshgrid(
        np.arange(-ry, ry + 1), np.arange(-rx, rx + 1), indexing="ij"
    )
    kernel = np.exp(-alpha * np.sqrt((dx * cw) ** 2 + (dy * ch) ** 2))

    m = mask.astype(float)
    z = oaconvolve(m, kernel, mode="same")
    box = np.ones(kernel.shape)  # for the reach mask (box conv beats
    # scipy.ndimage.binary_dilation, which is slow for large structures)

    out = np.empty_like(cols)
    for j in range(cols.shape[1]):
        plane = np.zeros(mask.shape)
        plane.reshape(-1)[flat_valid] = cols[:, j]
        ratio = np.zeros_like(plane)
        # Masked division (design §3): at invalid cells far from any valid
        # cell z == 0, and a bare plane/z would seed NaN that the convolution
        # spreads everywhere.
        np.divide(plane, z, out=ratio, where=mask)
        smoothed = oaconvolve(ratio, kernel, mode="same")
        # Restore exact compact support: FFT writes ~1e-16 dust everywhere;
        # true values are zero outside (source support ⊕ window) ∩ mask.
        reach = oaconvolve((plane != 0).astype(float), box, mode="same") > 0.5
        smoothed[~(reach & mask)] = 0.0
        # Clamp FFT roundoff INSIDE the reach: true corner values ~q·truncate^√2
        # can sink below the FFT noise floor (which scales with total landscape
        # mass) and come out negative — and negative amounts would silently
        # invert the warp=1 heap's monotonicity invariant downstream (they
        # bypass raw-amount validation). Design-review HIGH; clipped mass is at
        # noise scale, far inside the conservation tolerance.
        np.maximum(smoothed, 0.0, out=smoothed)
        out[:, j] = smoothed.reshape(-1)[flat_valid]
    return out[:, 0] if one_d else out
```

Use a PLAIN `from pymarxan.models.grid import GridGeometry` import with an
unquoted annotation — no cycle exists (precedent: `connectivity/features.py`
already imports `pymarxan.models.problem`; review-verified). Two docstring
additions to the function (review findings #2/#8): (a) "Zonation's own smoothing
is unnormalized kernel accumulation (Moilanen et al. 2005,
doi:10.1098/rspb.2005.3164); this function additionally conserves mass per
source — an edge-corrected variant whose rankings match the unnormalized
transform wherever a source's truncated window fits inside the valid mask
(CAZ/ABF scores are invariant to per-feature constant scaling), differing only
near edges/holes"; (b) "Common Zonation guidance sets alpha = 2 / (species mean
dispersal distance) — the mean-dispersal identity of the 2-D kernel (Westwood
et al. 2020, doi:10.3390/d12020061); alpha is in inverse CRS distance units.
Zonation 5 itself truncates the kernel tail (Moilanen et al. 2022,
doi:10.1111/2041-210X.13819)." ALSO one drive-by fix in the SAME file (review
finding #7, numerically verified): `smooth_distribution`'s docstring says
column normalization means "each destination unit's incoming kernel weights sum
to 1" — the code is per-SOURCE outgoing; change that sentence to "each source
unit's outgoing kernel weights sum to 1, so the redistribution conserves total
amount". No behavior change.

- [ ] **Step 4: Run tests**

Run: `/opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/connectivity/test_grid_smoothing.py -v`
Expected: ALL PASS (12 collected items — 11 functions, one 2-way parametrize).
Then ruff + mypy on both files.

- [ ] **Step 5: Commit**

```bash
git add src/pymarxan/connectivity/smoothing.py tests/pymarxan/connectivity/test_grid_smoothing.py
git commit -m "feat(connectivity): smooth_distribution_grid — truncated-window grid convolution"
```

---

### Task 2: `GridSmoothingSpec` + exports

**Files:**
- Modify: `src/pymarxan/zonation/smoothing.py` (append), `src/pymarxan/zonation/__init__.py` (export)
- Test: `tests/pymarxan/zonation/test_smoothing.py` (append)

**Interfaces:**
- Consumes: `smooth_distribution_grid` (Task 1).
- Produces: `GridSmoothingSpec(alpha: float, truncate: float = 1e-9)` with `apply(amounts: np.ndarray, grid: GridGeometry) -> np.ndarray`; exported from `pymarxan.zonation` exactly like `SmoothingSpec` (check `zonation/__init__.py` for the existing export line and mirror it; also mirror any top-level `pymarxan/__init__.py` re-export IF SmoothingSpec has one — grep first, do not invent one).

- [ ] **Step 1: Failing tests** (append to `tests/pymarxan/zonation/test_smoothing.py`):

```python
# --- GridSmoothingSpec (grid-convolution smoothing) -----------------------
from pymarxan.models.grid import GridGeometry
from pymarxan.zonation.smoothing import GridSmoothingSpec


def test_grid_spec_validation() -> None:
    with pytest.raises(ValueError, match="alpha"):
        GridSmoothingSpec(alpha=0.0)
    with pytest.raises(ValueError, match="truncate"):
        GridSmoothingSpec(alpha=1.0, truncate=1.0)
    with pytest.raises(ValueError, match="truncate"):
        GridSmoothingSpec(alpha=1.0, truncate=0.0)


def test_grid_spec_apply_delegates() -> None:
    grid = GridGeometry(
        x_min=0.0, y_max=4.0, cell_width=1.0, cell_height=1.0,
        mask=np.ones((4, 4), dtype=bool),
    )
    q = np.arange(1.0, 17.0)
    spec = GridSmoothingSpec(alpha=0.8)
    from pymarxan.connectivity.smoothing import smooth_distribution_grid

    np.testing.assert_array_equal(
        spec.apply(q, grid), smooth_distribution_grid(q, grid, 0.8, truncate=1e-9)
    )


def test_grid_spec_exported_from_zonation() -> None:
    from pymarxan.zonation import GridSmoothingSpec as exported

    assert exported is GridSmoothingSpec
```

- [ ] **Step 2: Verify failures** (ImportError). **Step 3: Implement** — append to `zonation/smoothing.py`:

```python
@dataclass
class GridSmoothingSpec:  # no eq=False: two float fields, auto __eq__ works
    """Raster-scale distribution smoothing for grid problems.

    The grid-convolution counterpart of :class:`SmoothingSpec`: same
    negative-exponential mass-conserving kernel, evaluated as a truncated 2-D
    convolution on ``problem.grid`` — no n×n kernel, no PU cap. Accepted by
    the same ``smoothing=`` parameter of ``rank_removal`` / ``ZonationSolver``;
    requires the problem to carry a :class:`GridGeometry`.
    """

    alpha: float
    truncate: float = 1e-9

    def __post_init__(self) -> None:
        if self.alpha <= 0:
            raise ValueError(f"alpha must be > 0, got {self.alpha}")
        if not 0.0 < self.truncate < 1.0:
            raise ValueError(f"truncate must be in (0, 1), got {self.truncate}")

    def apply(self, amounts: np.ndarray, grid: GridGeometry) -> np.ndarray:
        """Smooth one or many per-PU feature columns on the grid."""
        return smooth_distribution_grid(
            amounts, grid, self.alpha, truncate=self.truncate
        )
```

with `from pymarxan.connectivity.smoothing import smooth_distribution_grid` and
`from pymarxan.models.grid import GridGeometry` added to the module imports, and
the `zonation/__init__.py` export extended alongside `SmoothingSpec`'s.

- [ ] **Step 4: Run** `tests/pymarxan/zonation/test_smoothing.py -v` → ALL PASS; ruff+mypy. **Step 5: Commit** `feat(zonation): GridSmoothingSpec`.

---

### Task 3: `rank_removal` dispatch + solver widening + integration tests

**Files:**
- Modify: `src/pymarxan/zonation/rank_removal.py`, `src/pymarxan/solvers/zonation_solver.py`
- Test: `tests/pymarxan/zonation/test_rank_removal_scale.py` (append)

**Interfaces:**
- Consumes: `GridSmoothingSpec` (Task 2), `problem.grid` (`ConservationProblem` kw-only field), existing engine unchanged.
- Produces: `rank_removal(..., smoothing: SmoothingSpec | GridSmoothingSpec | None = None)`; same widening in `ZonationSolver.__init__`.

- [ ] **Step 1: Failing tests** (append to the scale file; `GridGeometry` import may need adding at top):

```python
# --- Grid-convolution smoothing dispatch (v0.33 phase) --------------------
from pymarxan.models.grid import GridGeometry
from pymarxan.zonation.smoothing import GridSmoothingSpec


def _grid_smoothing_problem(nrows: int = 6, ncols: int = 5) -> ConservationProblem:
    rng = np.random.default_rng(23)
    mask = np.ones((nrows, ncols), dtype=bool)
    mask[1, 1] = False
    grid = GridGeometry(
        x_min=0.0, y_max=float(nrows), cell_width=1.0, cell_height=1.0, mask=mask
    )
    n = int(mask.sum())
    pu = pd.DataFrame(
        {"id": list(range(1, n + 1)), "cost": rng.integers(1, 4, n).astype(float),
         "status": [0] * n}
    )
    feats = pd.DataFrame(
        {"id": [1, 2], "name": ["a", "b"], "target": [1.0, 1.0], "spf": [1.0, 1.0]}
    )
    amounts = rng.integers(0, 4, size=(n, 2)).astype(float)
    rows = [
        {"species": fid, "pu": i + 1, "amount": amounts[i, j]}
        for i in range(n)
        for j, fid in enumerate([1, 2])
        if amounts[i, j]
    ]
    return ConservationProblem(pu, feats, pd.DataFrame(rows), grid=grid)


def test_grid_smoothing_requires_grid_problem() -> None:
    p = _random_problem(0)  # no grid
    with pytest.raises(ValueError, match="grid"):
        rank_removal(p, smoothing=GridSmoothingSpec(alpha=1.0))


@pytest.mark.parametrize("rule", ["caz", "abf"])
def test_grid_smoothing_heap_equals_batch(rule: str) -> None:
    p = _grid_smoothing_problem()
    spec = GridSmoothingSpec(alpha=0.7)
    _assert_equal_results(
        rank_removal(p, rule=rule, warp=1, smoothing=spec),
        rank_removal(p, rule=rule, warp=1, smoothing=spec, _force_batch=True),
    )


@pytest.mark.parametrize("rule", ["caz", "abf"])
def test_grid_smoothing_self_consistency(rule: str) -> None:
    # rank_removal(problem, GridSmoothingSpec) must equal rank_removal on a
    # problem REBUILT from the manually smoothed matrix (same values through
    # the same engine -> orders equal exactly). This is the oracle for the
    # dispatch path — the dense test oracle never learns GridSmoothingSpec.
    from pymarxan.connectivity.smoothing import smooth_distribution_grid

    p = _grid_smoothing_problem()
    spec = GridSmoothingSpec(alpha=0.7)
    q = p.build_pu_feature_matrix()
    smoothed = smooth_distribution_grid(q, p.grid, spec.alpha, truncate=spec.truncate)
    rows = [
        {"species": fid, "pu": i + 1, "amount": smoothed[i, j]}
        for i in range(smoothed.shape[0])
        for j, fid in enumerate([1, 2])
        if smoothed[i, j]
    ]
    p2 = ConservationProblem(
        p.planning_units.copy(), p.features.copy(), pd.DataFrame(rows), grid=p.grid
    )
    a = rank_removal(p, rule=rule, smoothing=spec)
    b = rank_removal(p2, rule=rule)
    assert a.removal_order == b.removal_order
    assert a.priority_rank == b.priority_rank


def test_grid_smoothing_no_pu_cap() -> None:
    # >50k cells must be ACCEPTED with GridSmoothingSpec (the cap is the dense
    # SmoothingSpec's). Trivially sparse feature keeps this fast.
    mask = np.ones((300, 200), dtype=bool)  # 60_000 cells
    grid = GridGeometry(
        x_min=0.0, y_max=300.0, cell_width=1.0, cell_height=1.0, mask=mask
    )
    n = 60_000
    pu = pd.DataFrame({"id": range(1, n + 1), "cost": [1.0] * n, "status": [0] * n})
    feats = pd.DataFrame({"id": [1], "name": ["a"], "target": [1.0], "spf": [1.0]})
    pvf = pd.DataFrame(
        [{"species": 1, "pu": 30_100, "amount": 5.0}]  # one source cell
    )
    p = ConservationProblem(pu, feats, pvf, grid=grid)
    res = rank_removal(p, smoothing=GridSmoothingSpec(alpha=1.0), warp=600)
    assert len(res.removal_order) == n


# NOTE (review #6): do NOT add a new cap test — the existing
# test_smoothing_capped_at_vector_scale already pins the cap with the identical
# monkeypatch. Instead EDIT that existing test: add a second context-managed
# assertion (same fixture) matching "GridSmoothingSpec" in the message, pinning
# the new redirect the cap message gains in this task.


@pytest.mark.parametrize("rule", ["caz"])
def test_zonation_solver_grid_smoothing(rule: str) -> None:
    from pymarxan.solvers.zonation_solver import ZonationSolver

    p = _grid_smoothing_problem()
    solver = ZonationSolver(rule=rule, smoothing=GridSmoothingSpec(alpha=0.7))
    sols = solver.solve(p, {})
    assert len(sols) == 1
    assert sols[0].metadata.get("smoothed") is True
```

NOTE on the solver test: check `ZonationSolver.__init__`'s actual signature
(`rule` may be positional/kw with different name — read
`src/pymarxan/solvers/zonation_solver.py:40-70` and adapt the constructor call
and the metadata key access to what exists; the metadata stash uses
`build_solution`, key names visible in that file around line 88).

- [ ] **Step 2: Verify failures.** ImportError/TypeError/no-raise as applicable.

- [ ] **Step 3: Implement.** In `rank_removal.py`:

Imports: `from pymarxan.zonation.smoothing import GridSmoothingSpec, SmoothingSpec`;
signature + docstring type: `smoothing: SmoothingSpec | GridSmoothingSpec | None = None`.

Replace the smoothing-cap guard and matrix build:

```python
    n_pu_total = problem.n_planning_units
    if isinstance(smoothing, GridSmoothingSpec) and problem.grid is None:
        raise ValueError(
            "GridSmoothingSpec requires a grid problem (problem.grid is None); "
            "use SmoothingSpec for vector problems"
        )
    if isinstance(smoothing, SmoothingSpec) and n_pu_total > _SMOOTHING_MAX_PU:
        # Positive isinstance (review #5): a future third spec type must not
        # silently inherit the dense cap or the dense apply path.
        raise ValueError(
            f"smoothing builds a dense {n_pu_total}x{n_pu_total} kernel and is "
            f"vector-scale only (n_pu <= {_SMOOTHING_MAX_PU}); use "
            "GridSmoothingSpec for grid problems at raster scale "
            "(problems constructed with grid=GridGeometry(...))."
        )
```

and the build branch:

```python
    if isinstance(smoothing, GridSmoothingSpec):
        assert problem.grid is not None  # guarded above
        base = problem.build_pu_feature_csr().tocsc()
        n_rows = base.shape[0]
        data_parts: list[np.ndarray] = []
        idx_parts: list[np.ndarray] = []
        indptr = [0]
        for j in range(base.shape[1]):
            col = np.zeros(n_rows)
            sl = slice(base.indptr[j], base.indptr[j + 1])
            col[base.indices[sl]] = base.data[sl]
            sc = smoothing.apply(col, problem.grid)
            nz = np.flatnonzero(sc)
            data_parts.append(sc[nz])
            idx_parts.append(nz)
            indptr.append(indptr[-1] + nz.size)
        from scipy.sparse import csc_matrix

        q = csc_matrix(
            (
                np.concatenate(data_parts) if data_parts else np.zeros(0),
                np.concatenate(idx_parts) if idx_parts else np.zeros(0, dtype=np.intp),
                np.asarray(indptr, dtype=np.intp),
            ),
            shape=base.shape,
        ).tocsr()
    elif isinstance(smoothing, SmoothingSpec):
        q = csr_matrix(smoothing.apply(problem.build_pu_feature_matrix()))
    elif smoothing is not None:
        raise TypeError(
            f"unsupported smoothing spec: {type(smoothing).__name__}"
        )
    else:
        q = problem.build_pu_feature_csr()
```

Line anchors (grounding-verified): the current cap guard is at
`rank_removal.py:171-176`, the build branch at `:180-183`, and
`q.eliminate_zeros()` at `:184` — it MUST remain after the new dispatch (the
snippet keeps it).

(Per-column `apply` recomputes the kernel and Z each call — deliberate, for
memory flatness; overhead is ~1.5× in convolution count (3 per column vs 2 + one
shared Z). Do not "optimize" by batching all columns dense.) Docstring: the sentence "Smoothing stays vector-scale
(n_pu <= 50_000)." becomes "Dense-kernel ``SmoothingSpec`` smoothing stays
vector-scale (n_pu <= 50_000); ``GridSmoothingSpec`` (grid problems) has no
cap — truncated 2-D convolution, mass-conserving."

In `zonation_solver.py`: widen the import + the `smoothing:` parameter type to
`SmoothingSpec | GridSmoothingSpec | None`. No behavior change.

- [ ] **Step 4: Run** the zonation dir + solver tests + the new -k "grid_smoothing or vector_smoothing_cap" selection → ALL PASS; whole scale file green; ruff + mypy on all touched files.

- [ ] **Step 5: Commit** `feat(zonation): GridSmoothingSpec dispatch in rank_removal + solver type widening`.

---

### Task 4: Bench, CHANGELOG, gate

**Files:**
- Modify: `tests/benchmarks/bench_zonation.py` (append), `CHANGELOG.md`

- [ ] **Step 1: Append the bench**

```python
def test_grid_smoothing_rank_removal_budget() -> None:
    from pymarxan.zonation.smoothing import GridSmoothingSpec

    p = _grid_problem(300, with_grid=True)  # 90_000 cells, compact blocks
    # alpha=2.0 -> window radius ~11: small enough that smoothing genuinely
    # preserves sparsity (review #3: alpha=0.5/truncate=1e-9 gives radius 42
    # and 36%-dense output, which cannot witness the sparsity claim).
    spec = GridSmoothingSpec(alpha=2.0)
    t0 = time.perf_counter()
    res = rank_removal(p, rule="caz", warp=90, smoothing=spec)
    elapsed = time.perf_counter() - t0
    assert len(res.removal_order) == 90_000
    assert elapsed < 120.0, f"grid-smoothed rank_removal took {elapsed:.1f}s"
    # Sparsity witness: rebuild the smoothed matrix the way the dispatch does
    # and assert it stays well under dense.
    from pymarxan.connectivity.smoothing import smooth_distribution_grid

    base = p.build_pu_feature_csr()
    csc = base.tocsc()
    nnz = 0
    for j in range(base.shape[1]):
        col = np.zeros(base.shape[0])
        sl = slice(csc.indptr[j], csc.indptr[j + 1])
        col[csc.indices[sl]] = csc.data[sl]
        nnz += int(
            (smooth_distribution_grid(col, p.grid, 2.0) > 0).sum()
        )
    assert nnz < 0.25 * base.shape[0] * base.shape[1], f"smoothed nnz {nnz}"
```

NOTE: `_grid_problem` builds a plain problem WITHOUT `grid=` — check its
definition; if it lacks a `GridGeometry`, extend the helper with an optional
`with_grid: bool = False` parameter that attaches
`GridGeometry(x_min=0.0, y_max=float(side), cell_width=1.0, cell_height=1.0,
mask=np.ones((side, side), dtype=bool))` and pass `with_grid=True` here —
keeping the two existing benches' calls unchanged.

- [ ] **Step 2: Run the bench deliberately** (`-m bench`), record time. If over
budget, STOP and report BLOCKED with the time (measure-first discipline).

- [ ] **Step 3: CHANGELOG** `[Unreleased]`:

```markdown
### Added
- `zonation.GridSmoothingSpec` + `connectivity.smooth_distribution_grid`:
  raster-scale distribution smoothing for grid problems via truncated-window
  2-D negative-exponential convolution — mass-conserving at any truncation
  (the normalizer uses the same truncated kernel), sparse column-wise (no
  dense n×n kernel, no dense PU×feature matrix), no 50k-PU cap. Accepted by
  the existing `smoothing=` parameter of `rank_removal`/`ZonationSolver`;
  the dense-kernel `SmoothingSpec` and its vector-scale cap are unchanged.
```

- [ ] **Step 4: Full gate** `PATH="/opt/micromamba/envs/shiny/bin:$HOME/.local/bin:$PWD/.venv/bin:$PATH" make check` → green (~1930 tests); SA-flake rerun protocol. **Step 5: Commit** `docs(zonation): grid-smoothing changelog + bench`.

---

## Self-review (done at write time)

- Spec coverage: §3→T1, §4 spec-class→T2, §4 dispatch/solver→T3, §5→T1+T3 tests, §6→T1 (allclose/conservation/support) + T3 (heap==batch, self-consistency), §7.1-3→T1, §7.4-6→T3, §7.7→T4, §7.8→T4 gate, §9 exports→T2.
- Types consistent: `smooth_distribution_grid(amounts, grid, alpha, *, truncate)` used identically in T1/T2/T3; `GridSmoothingSpec.apply(amounts, grid)` T2→T3.
- Two flagged verify-don't-assume points for implementers: ZonationSolver constructor signature/metadata keys (T3 note), `_grid_problem` grid attachment (T4 note).
- No placeholders.
