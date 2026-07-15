# Raster-grid PUs — S3a implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Shrink `ProblemCache` memory (3 dense `(n_pu×n_feat)` matrices → one CSR) so SA / iterative-improvement run on large problems, with bit-identical solver behaviour.

**Architecture:** (1) Gate `expected_matrix`/`var_matrix` construction on `probmode==3` (dead weight otherwise). (2) Store the feature amounts as a `scipy.sparse` CSR (`ConservationProblem.build_pu_feature_csr()`), make `ProblemCache.pu_feat_matrix` a cached densify-property for the niche dense consumers (clumping/separation/probmode-3), and route the hot loop (compute_held, delta, held-update in SA/II) through CSR methods — so a plain problem never densifies. The delta becomes O(nnz_row) and stays bit-identical.

**Tech Stack:** Python 3.12+, NumPy, pandas, scipy.sparse (already a dep).

**Design spec:** `docs/plans/2026-07-15-raster-grid-s3a-design.md`. Cache: `solvers/cache.py`. Parity: `examples/validate_marxan_parity.py`, `marxan-parity-check` skill.

## Global Constraints

- Python 3.12+, `from __future__ import annotations`, full type hints.
- No new third-party dependency.
- Tests **must** run under the `shiny` micromamba env: `/opt/micromamba/envs/shiny/bin/pytest`.
- Lint: ruff (E, F, I, UP; line length 99). Types: mypy (`--ignore-missing-imports`) clean. Coverage ≥ 75%.
- The bar before done: `make check` green **and** `make bench` non-regressing.
- **Parity anchor:** MIP == 35.0 on `tests/data/simple`; SA/greedy ≥ 35.0; `validate_marxan_parity.py` green. The delta + held-update are bit-identical; `compute_held` is `allclose` (scipy vs numpy sum order) — the simple project's integer amounts sum exactly, so its SA trajectory is unchanged.
- **CSR data is float64** (matching `build_pu_feature_matrix`). Duplicate `(pu,species)` rows are summed; unknown-id rows dropped.

## File Structure

- Modify: `src/pymarxan/models/problem.py` — add `build_pu_feature_csr()`.
- Modify: `src/pymarxan/solvers/cache.py` — CSR field + property + `from_problem` + `compute_held` + `apply_flip_to_held` + delta + gate expected/var.
- Modify: `src/pymarxan/solvers/simulated_annealing.py` (1 line), `src/pymarxan/solvers/iterative_improvement.py` (3 sites).
- Test: `tests/pymarxan/models/test_problem.py` (or a new test file), `tests/pymarxan/solvers/test_cache.py`.

---

### Task 1: `ConservationProblem.build_pu_feature_csr()`

**Files:**
- Modify: `src/pymarxan/models/problem.py`
- Test: `tests/pymarxan/models/test_problem.py` (append) — or wherever `build_pu_feature_matrix` is tested.

**Interfaces:**
- Produces: `build_pu_feature_csr() -> scipy.sparse.csr_matrix`, shape `(n_pu, n_feat)`, `build_pu_feature_csr().toarray() == build_pu_feature_matrix()`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/pymarxan/models/test_problem.py`:

```python
def test_build_pu_feature_csr_matches_dense(tiny_problem):
    csr = tiny_problem.build_pu_feature_csr()
    assert np.array_equal(csr.toarray(), tiny_problem.build_pu_feature_matrix())


def test_build_pu_feature_csr_sums_duplicates_and_drops_unknown():
    import pandas as pd

    from pymarxan.models.problem import ConservationProblem

    pu = pd.DataFrame({"id": [1, 2], "cost": [1.0, 1.0], "status": [0, 0]})
    feats = pd.DataFrame({"id": [1], "name": ["a"], "target": [1.0], "spf": [1.0]})
    pvf = pd.DataFrame(
        {
            "species": [1, 1, 1, 99],       # 99 = unknown feature -> dropped
            "pu": [1, 1, 2, 1],             # (1,1) appears twice -> summed
            "amount": [2.0, 3.0, 4.0, 5.0],
        }
    )
    p = ConservationProblem(pu, feats, pvf)
    csr = p.build_pu_feature_csr()
    assert np.array_equal(csr.toarray(), p.build_pu_feature_matrix())
    assert csr.toarray()[0, 0] == 5.0  # 2 + 3 summed
    assert csr.toarray()[1, 0] == 4.0


def test_build_pu_feature_csr_empty_pvf():
    import pandas as pd

    from pymarxan.models.problem import ConservationProblem

    pu = pd.DataFrame({"id": [1, 2], "cost": [1.0, 1.0], "status": [0, 0]})
    feats = pd.DataFrame({"id": [1], "name": ["a"], "target": [1.0], "spf": [1.0]})
    pvf = pd.DataFrame({"species": [], "pu": [], "amount": []})
    csr = ConservationProblem(pu, feats, pvf).build_pu_feature_csr()
    assert csr.shape == (2, 1)
    assert csr.nnz == 0


def test_build_pu_feature_csr_edge_cases():
    import pandas as pd

    from pymarxan.models.problem import ConservationProblem

    # feature 2 present in no PU (empty column); PU 3 has no features (empty row);
    # a negative amount is stored but does not count as "present".
    pu = pd.DataFrame({"id": [1, 2, 3], "cost": [1.0, 1.0, 1.0], "status": [0, 0, 0]})
    feats = pd.DataFrame({"id": [1, 2], "name": ["a", "b"], "target": [1.0, 1.0],
                          "spf": [1.0, 1.0]})
    pvf = pd.DataFrame({"species": [1, 1], "pu": [1, 2], "amount": [5.0, -3.0]})
    p = ConservationProblem(pu, feats, pvf)
    assert np.array_equal(p.build_pu_feature_csr().toarray(), p.build_pu_feature_matrix())
```

- [ ] **Step 2: Run to verify they fail**

Run: `/opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/models/test_problem.py -k "csr" -v`
Expected: FAIL — `ConservationProblem` has no `build_pu_feature_csr`.

- [ ] **Step 3: Implement `build_pu_feature_csr`**

First add a `TYPE_CHECKING` import at the top of `src/pymarxan/models/problem.py` so the
`-> csr_matrix` annotation type-checks without importing scipy at module-load time (the file
has `from __future__ import annotations`, so the annotation is a string at runtime):

```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from scipy.sparse import csr_matrix
```

Then add to `ConservationProblem` (right after `build_pu_feature_matrix`) in `src/pymarxan/models/problem.py`:

```python
    def build_pu_feature_csr(self) -> csr_matrix:
        """Sparse ``(n_pu, n_feat)`` CSR of feature amounts per PU.

        Rows in ``planning_units`` order, columns in ``features["id"]`` order.
        Duplicate ``(pu, species)`` rows are summed and unknown-id rows dropped —
        the sparse equivalent of ``build_pu_feature_matrix`` (``csr.toarray()`` equals it).
        ``sum_duplicates()`` is load-bearing: it makes each row's column indices **unique**,
        which ``ProblemCache.apply_flip_to_held`` relies on (numpy fancy in-place add is
        last-wins, not accumulate).
        """
        from scipy.sparse import csr_matrix

        n_pu = self.n_planning_units
        n_feat = self.n_features
        feat_ids = self.features["id"].values
        feat_id_to_col = {int(fid): j for j, fid in enumerate(feat_ids)}

        pv = self.pu_vs_features
        rows = pd.Series(pv["pu"].values).map(self.pu_id_to_index).to_numpy()
        cols = pd.Series(pv["species"].values).map(feat_id_to_col).to_numpy()
        amt = np.asarray(pv["amount"].values, dtype=np.float64)
        keep = ~(pd.isna(rows) | pd.isna(cols))

        csr = csr_matrix(
            (amt[keep], (rows[keep].astype(np.int64), cols[keep].astype(np.int64))),
            shape=(n_pu, n_feat),
        )
        csr.sum_duplicates()
        return csr
```

- [ ] **Step 4: Run to verify they pass**

Run: `/opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/models/test_problem.py -k "csr" -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/pymarxan/models/problem.py tests/pymarxan/models/test_problem.py
git commit -m "feat(models): build_pu_feature_csr — sparse CSR feature matrix (== dense builder)"
```

---

### Task 2: CSR-backed `ProblemCache` + gate probmode matrices

**Files:**
- Modify: `src/pymarxan/solvers/cache.py`
- Test: `tests/pymarxan/solvers/test_cache.py` (append)

**Interfaces:**
- Consumes: `build_pu_feature_csr` (Task 1).
- Produces: `ProblemCache.pu_feat_csr` (field), `pu_feat_matrix` (cached property), `apply_flip_to_held(held, idx, sign)` (method); `compute_held`/`compute_delta_objective` unchanged signatures.

- [ ] **Step 1: Write the failing tests**

Append to `tests/pymarxan/solvers/test_cache.py`:

```python
import numpy as np

from pymarxan.solvers.cache import ProblemCache


def _random_problem(seed=0, n_pu=12, n_feat=4):
    import pandas as pd

    from pymarxan.models.problem import ConservationProblem

    rng = np.random.default_rng(seed)
    pu = pd.DataFrame({"id": np.arange(1, n_pu + 1), "cost": rng.random(n_pu) + 0.5,
                       "status": np.zeros(n_pu, int)})
    feats = pd.DataFrame({"id": np.arange(1, n_feat + 1), "name": [f"f{j}" for j in range(n_feat)],
                          "target": rng.random(n_feat) * 3, "spf": np.ones(n_feat)})
    rows = []
    for pid in range(1, n_pu + 1):
        for sp in range(1, n_feat + 1):
            if rng.random() < 0.5:
                rows.append({"species": sp, "pu": pid, "amount": float(rng.integers(1, 5))})
    pvf = pd.DataFrame(rows)
    return ConservationProblem(pu, feats, pvf)


def test_cache_csr_field_and_property():
    p = _random_problem()
    cache = ProblemCache.from_problem(p)
    # dense property equals the dense builder
    assert np.array_equal(cache.pu_feat_matrix, p.build_pu_feature_matrix())
    # cached: same object on second access
    assert cache.pu_feat_matrix is cache.pu_feat_matrix


def test_cache_is_construction_safe():
    # frozen dataclass with a CSR field must not raise on build (compare=False)
    ProblemCache.from_problem(_random_problem())  # no exception


def test_compute_held_allclose_to_dense():
    p = _random_problem()
    cache = ProblemCache.from_problem(p)
    dense = p.build_pu_feature_matrix()
    rng = np.random.default_rng(1)
    for _ in range(5):
        sel = rng.random(cache.n_pu) < 0.5
        assert np.allclose(cache.compute_held(sel), dense[sel].sum(axis=0))


def test_apply_flip_to_held_bit_identical():
    p = _random_problem()
    cache = ProblemCache.from_problem(p)
    dense = p.build_pu_feature_matrix()
    held = cache.compute_held(np.zeros(cache.n_pu, bool))
    for idx, sign in [(0, 1.0), (3, 1.0), (3, -1.0), (7, 1.0)]:
        expected = held + sign * dense[idx]
        got = held.copy()
        cache.apply_flip_to_held(got, idx, sign)
        assert np.array_equal(got, expected)  # bit-identical (only omits + 0.0)


import pytest  # noqa: E402


@pytest.mark.parametrize("n_pu, n_feat", [(12, 4), (40, 160)])
def test_delta_matches_dense_penalty(n_pu, n_feat):
    # The CSR delta equals the dense-formula delta. NOT bitwise `==` at large n_feat:
    # the per-nonzero-column np.dot regroups the retained terms (non-associative), so it
    # drifts <= a few ULP for float targets/SPF. The (40, 160) case exercises that raster
    # regime; use a relative tolerance. (Integer-amount problems + the 35.0 anchor are exact.)
    p = _random_problem(n_pu=n_pu, n_feat=n_feat)
    cache = ProblemCache.from_problem(p)
    dense = p.build_pu_feature_matrix()
    rng = np.random.default_rng(2)
    sel = rng.random(cache.n_pu) < 0.5
    held = cache.compute_held(sel)
    total_cost = float(cache.costs[sel].sum())
    eff = cache.feat_targets * cache.misslevel
    for idx in range(cache.n_pu):
        sign = -1.0 if sel[idx] else 1.0
        old_sf = np.maximum(0.0, eff - held)
        new_sf = np.maximum(0.0, eff - (held + sign * dense[idx]))
        dense_pen = float(np.dot(cache._det_spf, new_sf - old_sf))
        # blm=0 → boundary term drops out → whole delta == cost_delta + penalty_delta.
        got = cache.compute_delta_objective(idx, sel, held, total_cost, blm=0.0)
        expected = sign * cache.costs[idx] + dense_pen
        assert abs(got - expected) <= 1e-9 * (1.0 + abs(expected))


def test_expected_var_gated_on_probmode():
    p = _random_problem()
    cache = ProblemCache.from_problem(p)  # probmode 0
    assert cache.expected_matrix.size == 0 and cache.var_matrix.size == 0
```

- [ ] **Step 2: Run to verify they fail**

Run: `/opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/solvers/test_cache.py -k "csr or held or delta or expected_var or construction" -v`
Expected: FAIL — no `pu_feat_csr`/`apply_flip_to_held`; `expected_matrix` non-empty for probmode 0.

- [ ] **Step 3: Refactor `cache.py`**

1. **Imports** (top): add
```python
from functools import cached_property
from scipy.sparse import csr_matrix
```

2. **Field** (replace `pu_feat_matrix: np.ndarray` at ~line 83):
```python
    pu_feat_csr: csr_matrix = field(compare=False, repr=False)
```
and add the property (anywhere in the class body, e.g. just above `compute_held`):
```python
    @cached_property
    def pu_feat_matrix(self) -> np.ndarray:
        """Dense (n_pu, n_feat) view — densified on first access (clumping / separation /
        probmode-3 / analysis). A plain problem never triggers this. NB do not add
        ``slots=True`` to this dataclass — it would break ``cached_property``."""
        if self.n_pu * self.n_feat > 20_000_000:  # ~160 MB float64 — the sparse win is lost here
            warnings.warn(
                "Densifying a large ProblemCache feature matrix "
                f"({self.n_pu}x{self.n_feat}); clumping/separation/probmode-3 at raster "
                "scale is not yet sparse.",
                stacklevel=2,
            )
        return np.asarray(self.pu_feat_csr.toarray())
```

3. **`from_problem`:**
   - Replace `pu_feat_matrix = problem.build_pu_feature_matrix()` (~line 176) with:
     ```python
     pu_feat_csr = problem.build_pu_feature_csr()
     ```
   - **Gate expected/var (Piece 1):** replace the unconditional `prob_matrix`/`expected_matrix`/`var_matrix` block (~lines 243-261) with:
     ```python
     if probmode == 3:
         dense = np.asarray(pu_feat_csr.toarray())
         prob_matrix = np.ones_like(dense)
         if (
             problem.pu_vs_features is not None
             and "prob" in problem.pu_vs_features.columns
         ):
             pv_pu = problem.pu_vs_features["pu"].values
             pv_sp = problem.pu_vs_features["species"].values
             pv_pr = problem.pu_vs_features["prob"].values
             for k in range(len(pv_pu)):
                 pi = pu_id_to_idx.get(int(pv_pu[k]))
                 fj = feat_id_to_col.get(int(pv_sp[k]))
                 if pi is not None and fj is not None:
                     prob_matrix[pi, fj] = float(pv_pr[k])
         expected_matrix = dense * prob_matrix
         var_matrix = (dense ** 2) * prob_matrix * (1.0 - prob_matrix)
     else:
         expected_matrix = np.zeros((0, 0))
         var_matrix = np.zeros((0, 0))
     ```
     (`probmode` is already read above this block.)
   - **`feat_uses_pu` from CSC (no densify)** — replace the `for j ... np.where(pu_feat_matrix[:, j] > 0)` loop (~lines 310-313) with:
     ```python
     csc = pu_feat_csr.tocsc()
     feat_uses_pu = []
     for j in range(n_feat):
         seg = slice(csc.indptr[j], csc.indptr[j + 1])
         col_rows = csc.indices[seg]
         col_data = csc.data[seg]
         feat_uses_pu.append(col_rows[col_data > 0].astype(np.int32))
     ```
   - **`pu_to_sep_feats`** (~lines 375-380, `separation_active` only) — densify locally there:
     ```python
     if separation_active:
         sep_dense = np.asarray(pu_feat_csr.toarray())
         sep_col_ids = np.where(sep_active_mask)[0].astype(np.int32)
         for i in range(n_pu):
             mask_at_pu = sep_dense[i, sep_col_ids] > 0
             pu_to_sep_feats.append(sep_col_ids[mask_at_pu])
     else:
         pu_to_sep_feats = [np.zeros(0, dtype=np.int32)] * n_pu
     ```
   - In the final `return cls(...)`, replace `pu_feat_matrix=pu_feat_matrix,` with `pu_feat_csr=pu_feat_csr,`.
   - **Docstrings:** update the class Attributes block (currently documents `pu_feat_matrix : np.ndarray`) to add `pu_feat_csr` and note `pu_feat_matrix` is now a lazily-densified property; update the module's inverse-index-discipline note to say the CSR is the source of truth and `feat_uses_pu` derives from its CSC form.

4. **`compute_held`** (replace the body — `flatnonzero` is unambiguous across scipy versions vs
   boolean row-indexing):
```python
        rows = np.flatnonzero(selected)
        result = np.asarray(self.pu_feat_csr[rows].sum(axis=0)).ravel()
        return result
```

5. **`apply_flip_to_held`** (new method, add near `compute_held`):
```python
    def apply_flip_to_held(self, held: np.ndarray, idx: int, sign: float) -> None:
        """In-place held update for flipping PU ``idx``: ``held[cols] += sign*amounts``.

        Bit-identical to ``held += sign * pu_feat_matrix[idx]`` (only omits ``+ 0.0`` on
        features absent from the PU), and O(nnz in the row).
        """
        csr = self.pu_feat_csr
        s, e = csr.indptr[idx], csr.indptr[idx + 1]
        cols = csr.indices[s:e]
        held[cols] += sign * csr.data[s:e]
```

6. **`compute_delta_objective` penalty block** — replace (~lines 615-622):
```python
        s = self.pu_feat_csr.indptr[idx]
        e = self.pu_feat_csr.indptr[idx + 1]
        cols = self.pu_feat_csr.indices[s:e]
        amts = self.pu_feat_csr.data[s:e]
        eff = self.feat_targets[cols] * self.misslevel
        old_shortfalls = np.maximum(0.0, eff - held[cols])
        new_shortfalls = np.maximum(0.0, eff - (held[cols] + sign * amts))
        penalty_delta = float(np.dot(self._det_spf[cols], new_shortfalls - old_shortfalls))
```
(The probmode-3 delta branch below is unchanged.)

- [ ] **Step 4: Run to verify they pass**

Run: `/opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/solvers/test_cache.py -q`
Expected: PASS. Also run the existing solver/objective suite to catch regressions:
`/opt/micromamba/envs/shiny/bin/pytest tests/ -q -k "solver or objective or penalty or clump or separation or probmode or cache"`

- [ ] **Step 5: Commit**

```bash
git add src/pymarxan/solvers/cache.py tests/pymarxan/solvers/test_cache.py
git commit -m "feat(solvers): CSR-backed ProblemCache — sparse pu_feat_matrix + gate probmode matrices"
```

---

### Task 3: Reroute the SA / II held-update + verify parity & bench

**Files:**
- Modify: `src/pymarxan/solvers/simulated_annealing.py`, `src/pymarxan/solvers/iterative_improvement.py`

**Interfaces:**
- Consumes: `ProblemCache.apply_flip_to_held` (Task 2). No new tests — the existing SA/II/parity suites are the guard (behaviour is bit-identical).

- [ ] **Step 1: Reroute the four call sites**

- `simulated_annealing.py:261`: `held += sign * cache.pu_feat_matrix[idx]` →
  ```python
  cache.apply_flip_to_held(held, idx, sign)
  ```
- `iterative_improvement.py:220`: `held -= cache.pu_feat_matrix[i]` →
  ```python
  cache.apply_flip_to_held(held, i, -1.0)
  ```
- `iterative_improvement.py:302`: `held += cache.pu_feat_matrix[i]` →
  ```python
  cache.apply_flip_to_held(held, i, 1.0)
  ```
- `iterative_improvement.py:355`: `new_held = held - cache.pu_feat_matrix[r] + cache.pu_feat_matrix[a]` →
  ```python
  new_held = held.copy()
  cache.apply_flip_to_held(new_held, r, -1.0)
  cache.apply_flip_to_held(new_held, a, 1.0)
  ```

- [ ] **Step 2: Add a "plain problem never densifies" test**

Append to `tests/pymarxan/solvers/test_cache.py`:

```python
def test_plain_solve_does_not_densify():
    # After compute_held + a delta, a plain problem's cache has NOT materialized the dense
    # matrix (the cached_property slot is empty).
    p = _random_problem()
    cache = ProblemCache.from_problem(p)
    sel = np.zeros(cache.n_pu, bool)
    held = cache.compute_held(sel)
    cache.compute_full_objective(sel, held, blm=1.0)  # also once per SA run
    cache.compute_delta_objective(0, sel, held, 0.0, blm=1.0)
    cache.apply_flip_to_held(held, 0, 1.0)
    assert "pu_feat_matrix" not in cache.__dict__  # never densified
```

- [ ] **Step 3: Run the parity harness + solver suite**

```bash
/opt/micromamba/envs/shiny/bin/python examples/validate_marxan_parity.py
/opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/solvers/ tests/ -q -k "parity or validate or simulated or iterative or greedy"
```
Expected: MIP == 35.0; SA/greedy ≥ 35.0; all green. (`test_solutions_are_different` flake → rerun once.)

- [ ] **Step 4: Full check + bench**

```bash
PATH="/opt/micromamba/envs/shiny/bin:$HOME/.local/bin:$PWD/.venv/bin:$PATH" make check
PATH="/opt/micromamba/envs/shiny/bin:$HOME/.local/bin:$PWD/.venv/bin:$PATH" make bench
```
Expected: `make check` green (0 ruff/0 mypy); `make bench` non-regressing (the O(nnz_row) delta is ≤ the old O(n_feat) per-flip). Note: `bench` is flaky on a busy machine — if a per-flip budget trips, rerun on a quiet machine and compare against the committed budget; a genuine regression (not noise) blocks.

- [ ] **Step 5: CHANGELOG + commit**

Add under `## [Unreleased]` → `### Changed` (create the header if absent):
```markdown
- **Sparse solver cache (S3a).** ``ProblemCache`` now stores the feature amounts as a
  ``scipy.sparse`` CSR (``ConservationProblem.build_pu_feature_csr``) instead of a dense
  ``(n_pu×n_feat)`` matrix, and builds the PROBMODE-3 ``expected``/``var`` matrices only when
  needed — cutting SA / iterative-improvement cache memory by ~10–40× on large sparse
  (raster) problems. ``cache.pu_feat_matrix`` is preserved as a lazily-densified property for
  clumping / separation / analysis; the delta is now O(nnz-per-PU). Solver results unchanged
  on integer-amount problems (MIP still 35.0 on the reference problem; SA/greedy ≥ 35.0); the
  delta / `compute_held` differ only by float summation order (≤ a few ULP) on arbitrary-float
  problems. Scope: plain SA / iterative-improvement — clumping / separation / probmode-3 /
  analysis / zone problems still densify (future work).
```

```bash
git add -A
git commit -m "feat(solvers): route SA/II held-update through CSR apply_flip_to_held (S3a)"
```

---

## Post-plan notes

- **Design review:** run `multi-agent-design-review` before/at execution — the risk surface is the parity-critical delta/held-update (bit-identical claim), the frozen-dataclass CSR field (`compare=False`), the `compute_held` allclose-not-bitwise nuance, and the probmode-3/clumping/separation densify paths staying correct.
- **Parity:** this is solver-internal; the `35.0` anchor + `validate_marxan_parity.py` + the dense-vs-CSR unit tests are the guard. The delta/held-update are bit-identical; `compute_held` differs only in float sum order (exact for the integer-amount simple project).
- **Deferred:** S3b (MIP-at-scale guard); `build_boundary` vectorization (`models/grid.py`) so `include_boundary` scales for windowed ingestion. Together with S3c, S3a means large raster problems both ingest and solve (SA/greedy) without OOM.
