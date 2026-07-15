# `GridGeometry.build_boundary` vectorization — implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the Python loops in `GridGeometry.build_boundary` with O(n) numpy so `include_boundary` scales to million-cell grids, with an identical output multiset.

**Architecture:** Rewrite one method body (`models/grid.py`). Scatter `pu_ids` into an `id_grid`, derive right/down shared edges from shifted-mask AND, derive self-boundary from exposed sides, `np.concatenate` into the `DataFrame`. Same signature, guards, columns, `1e-10` threshold, and output rows (order may differ — nothing depends on it).

**Tech Stack:** Python 3.12+, NumPy, pandas.

**Design spec:** `docs/plans/2026-07-15-build-boundary-vectorization-design.md`. Current code: `src/pymarxan/models/grid.py::GridGeometry.build_boundary`. Parity vs `spatial/boundary.py::compute_boundary`.

## Global Constraints

- Python 3.12+, `from __future__ import annotations`, full type hints.
- No new dependency.
- Tests **must** run under the `shiny` micromamba env: `/opt/micromamba/envs/shiny/bin/pytest`.
- Lint: ruff (E, F, I, UP; line length 99). Types: mypy clean. Coverage ≥ 75%.
- The bar before done: `make check` green.
- **No behaviour change:** same signature `build_boundary(pu_ids=None) -> pd.DataFrame`; same `len`/uniqueness guards; columns `["id1","id2","boundary"]`; `> 1e-10` self threshold; identical output **multiset** (row order may differ — the parity tests sort, the hand-computed tests check sets/counts).
- **Do NOT call `self.valid_cells()`** in the rewrite (it builds a Python list of `n` tuples — a scale bottleneck); use `np.flatnonzero(mask.reshape(-1))`.

## File Structure

- Modify: `src/pymarxan/models/grid.py` — rewrite `build_boundary`.
- Test: `tests/pymarxan/models/test_grid.py` (append).

---

### Task 1: Vectorize `build_boundary`

**Files:**
- Modify: `src/pymarxan/models/grid.py`
- Test: `tests/pymarxan/models/test_grid.py` (append)

**Interfaces:**
- Unchanged: `build_boundary(pu_ids: np.ndarray | None = None) -> pd.DataFrame`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/pymarxan/models/test_grid.py`:

```python
def _reference_build_boundary(grid, pu_ids=None):
    """The pre-vectorization per-cell loop — the multiset oracle."""
    cells = grid.valid_cells()
    n = len(cells)
    if pu_ids is None:
        pu_ids = np.arange(1, n + 1)
    pu_ids = np.asarray(pu_ids)
    cell_to_id = {cell: int(pu_ids[i]) for i, cell in enumerate(cells)}
    shared = {int(pid): 0.0 for pid in pu_ids}
    rows = []
    for (r, c), pid in cell_to_id.items():
        for nbr, edge in (((r, c + 1), grid.cell_height), ((r + 1, c), grid.cell_width)):
            nid = cell_to_id.get(nbr)
            if nid is not None:
                rows.append({"id1": pid, "id2": nid, "boundary": edge})
                shared[pid] += edge
                shared[nid] += edge
    perimeter = 2.0 * (grid.cell_width + grid.cell_height)
    for pid in cell_to_id.values():
        sb = perimeter - shared[pid]
        if sb > 1e-10:
            rows.append({"id1": pid, "id2": pid, "boundary": sb})
    return pd.DataFrame(rows, columns=["id1", "id2", "boundary"])


def _sorted(df):
    return df.sort_values(["id1", "id2"]).reset_index(drop=True)


def test_build_boundary_vectorized_matches_reference_loop():
    rng = np.random.default_rng(0)
    hole = np.ones((3, 3), dtype=bool)
    hole[1, 1] = False
    masked = rng.random((5, 5)) < 0.7
    masked[0, 0] = True  # guarantee at least one valid cell
    cases = [
        GridGeometry(0.0, 4.0, 1.0, 1.0, np.ones((4, 4), dtype=bool)),   # full
        GridGeometry(10.0, 20.0, 2.0, 3.0, masked),                     # non-square + holes
        GridGeometry(0.0, 1.0, 1.0, 1.0, np.ones((1, 5), dtype=bool)),  # 1xN strip
        GridGeometry(0.0, 5.0, 1.0, 1.0, np.ones((5, 1), dtype=bool)),  # Nx1 strip
        GridGeometry(0.0, 3.0, 1.0, 1.0, hole),                         # center hole
        GridGeometry(0.0, 1.0, 1.0, 1.0, np.ones((1, 1), dtype=bool)),  # single cell
        GridGeometry(0.0, 2.1, 0.3, 0.7, hole.copy()),                  # non-integer cells (ULP path)
        GridGeometry(0.0, 3.0, 1.0, 1.0, np.asfortranarray(hole)),      # Fortran-order mask
    ]
    for grid in cases:
        got = _sorted(grid.build_boundary())
        ref = _sorted(_reference_build_boundary(grid))
        pd.testing.assert_frame_equal(got, ref, check_dtype=False)


def test_build_boundary_vectorized_arbitrary_pu_ids():
    grid = GridGeometry(0.0, 2.0, 1.0, 1.0, np.ones((2, 2), dtype=bool))
    ids = np.array([5, 3, 8, 1])  # non-sequential
    got = _sorted(grid.build_boundary(ids))
    ref = _sorted(_reference_build_boundary(grid, ids))
    pd.testing.assert_frame_equal(got, ref, check_dtype=False)


def test_build_boundary_scale_smoke():
    # 200x200 full grid: 2*200*199 shared rows + (4*200-4) border self rows.
    grid = GridGeometry(0.0, 200.0, 1.0, 1.0, np.ones((200, 200), dtype=bool))
    df = grid.build_boundary()
    assert len(df) == 2 * 200 * 199 + (4 * 200 - 4)
```

- [ ] **Step 2: Run to verify they fail (or pass by accident on the old loop)**

Run: `/opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/models/test_grid.py -k "vectorized or scale_smoke" -v`
Expected: the `matches_reference_loop` / `arbitrary_pu_ids` tests **pass on the old loop** (they compare to a reference identical to the old code) — that's fine; they lock the contract before and after. The `scale_smoke` also passes on the old loop but slowly. (This task is a refactor guarded by these tests, not a red→green feature; the real signal is that they stay green after the rewrite **and** the existing S1 parity tests do too.)

- [ ] **Step 3: Rewrite `build_boundary`**

Replace the body of `GridGeometry.build_boundary` in `src/pymarxan/models/grid.py` (keep the docstring, note it's vectorized):

```python
    def build_boundary(self, pu_ids: np.ndarray | None = None) -> pd.DataFrame:
        """Analytic rook-adjacency boundary (columns ``id1``, ``id2``,
        ``boundary``), matching ``spatial.boundary.compute_boundary`` on the
        equivalent vector grid. ``pu_ids`` aligns to valid-cell order (defaults
        to ``1..n_pu``). Vectorized (numpy), O(n_pu)."""
        mask = self.mask
        flat_valid = np.flatnonzero(mask.reshape(-1))  # row-major (C-order) == PU order
        n = int(flat_valid.size)
        if pu_ids is None:
            pu_ids = np.arange(1, n + 1)
        pu_ids = np.asarray(pu_ids)
        if len(pu_ids) != n:
            raise ValueError(f"pu_ids must have {n} entries, got {len(pu_ids)}")
        if len(np.unique(pu_ids)) != n:
            raise ValueError("pu_ids must be unique")

        # id grid: pu_ids scattered at valid cells (row-major); invalid cells unused.
        id_grid = np.zeros(mask.shape, dtype=np.int64)
        id_grid.reshape(-1)[flat_valid] = pu_ids  # id_grid is C-contiguous -> writable view
        id_at_valid = id_grid.reshape(-1)[flat_valid]  # pu_ids as int64, in PU order

        # Right edges (valid cell + valid right neighbor share a vertical edge = cell_height).
        both_r = mask[:, :-1] & mask[:, 1:]
        r_id1 = id_grid[:, :-1][both_r]
        r_id2 = id_grid[:, 1:][both_r]

        # Down edges (valid cell + valid down neighbor share a horizontal edge = cell_width).
        both_d = mask[:-1, :] & mask[1:, :]
        d_id1 = id_grid[:-1, :][both_d]
        d_id2 = id_grid[1:, :][both_d]

        # Self-boundary = exposed sides = perimeter - shared, per valid cell.
        has_left = np.zeros_like(mask)
        has_left[:, 1:] = mask[:, :-1]
        has_right = np.zeros_like(mask)
        has_right[:, :-1] = mask[:, 1:]
        has_up = np.zeros_like(mask)
        has_up[1:, :] = mask[:-1, :]
        has_down = np.zeros_like(mask)
        has_down[:-1, :] = mask[1:, :]
        self_grid = (
            (2 - has_left.astype(np.int64) - has_right.astype(np.int64)) * self.cell_height
            + (2 - has_up.astype(np.int64) - has_down.astype(np.int64)) * self.cell_width
        )
        self_vals = self_grid.reshape(-1)[flat_valid]
        keep = self_vals > 1e-10
        s_ids = id_at_valid[keep]
        s_vals = self_vals[keep]

        id1 = np.concatenate([r_id1, d_id1, s_ids])
        id2 = np.concatenate([r_id2, d_id2, s_ids])
        boundary = np.concatenate([
            np.full(r_id1.size, self.cell_height),
            np.full(d_id1.size, self.cell_width),
            s_vals,
        ])
        return pd.DataFrame({"id1": id1, "id2": id2, "boundary": boundary})
```

- [ ] **Step 4: Run the grid tests + verify pass**

Run: `/opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/models/test_grid.py -q`
Expected: PASS — the existing S1 tests (shapely parity, hand-computed 2×2, non-square, single-cell, guards, center-hole) **and** the new reference-loop / arbitrary-ids / scale-smoke tests all green.

- [ ] **Step 5: Run the boundary parity harness (spatial)**

Run: `/opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/models/test_grid.py tests/pymarxan/spatial/test_raster.py -q -m "spatial or not spatial"`
Expected: PASS — the S3c windowed `include_boundary=True` path (which calls `build_boundary`) and the shapely-parity tests confirm equality end-to-end.

- [ ] **Step 6: CHANGELOG + full check**

Add under `## [Unreleased]` → `### Changed` in `CHANGELOG.md` (create the header if absent):
```markdown
- **Vectorized `GridGeometry.build_boundary`.** The analytic rook-adjacency boundary is now
  built with O(n) numpy array ops (shifted-mask edges + exposed-side self-boundary) instead of
  a per-cell Python loop — so `include_boundary` scales to million-cell raster grids (identical
  output; the shapely-parity anchor is unchanged).
```

Run: `PATH="/opt/micromamba/envs/shiny/bin:$HOME/.local/bin:$PWD/.venv/bin:$PATH" make check`
Expected: green — 0 ruff, 0 mypy, full suite. (`test_solutions_are_different` flake → rerun once.)

Note: the CLAUDE.md `micromamba.sh` activation path may not exist; the `PATH=...` prefix is the working invocation.

- [ ] **Step 7: Commit**

```bash
git add src/pymarxan/models/grid.py tests/pymarxan/models/test_grid.py CHANGELOG.md
git commit -m "perf(grid): vectorize GridGeometry.build_boundary (O(n) numpy, identical output)"
```

---

## Post-plan notes

- **Design review:** run `multi-agent-design-review` — the risk surface is the shift-index correctness (right/down edges, the four `has_*` grids), the self-boundary `perimeter - shared` algebraic identity, and the `id_grid.reshape(-1)` writable-view / C-order assumption. The grounding agent should run vectorized-vs-loop-vs-shapely.
- **Parity:** no solver/objective change; the 35.0 anchor is untouched. `build_boundary`'s output multiset is unchanged (the reference-loop test + the S1 shapely-parity set are the guards).
- **Follow-on (optional, not this task):** flip the S3c windowed `include_boundary` default back to build-by-default now that it scales — a separate `spatial/raster.py` decision, deferred.
