# SA Optimises Connectivity (v0.44.0) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Simulated annealing and iterative improvement optimise connectivity (`CONNECTIVITY_WEIGHT`, symmetric and asymmetric) with exactly the term the MIP optimises and `build_solution` reports, so SA's internal objective equals the objective it reports.

**Architecture:** `ProblemCache` gains connectivity adjacency arrays in CSR form (built once in `from_problem`, the same pattern as boundary) plus `conn_mode` / `conn_weight`. `compute_full_objective` adds the term, and `compute_delta_objective` adds an O(degree) per-flip delta. Neither SA nor iterative improvement changes: both already score every move through these two methods. With connectivity off the delta costs one integer comparison.

**Tech Stack:** Python 3.11+, numpy, pytest.

**Spec:** `docs/plans/2026-10-17-sa-connectivity-design.md` (approved 2026-10-17). Executors read it. The spec is the authority, and this plan argues from it.

**Provenance:** every code block below is extracted (`git show`) from a prototype built task by task in a scratch worktree, one commit per task, with the full gate green at every commit. Prototype commits: Task 1 `cd4bb33`, Task 2 `79ef8c3`, Task 3 `f71ef1c` (base `06e0be5`). Diffs apply with `git apply --whitespace=nowarn` on the previous task's tree. Every block was replayed that way on a fresh worktree from the base, and each task's tree matched the prototype commit. Editors that strip trailing whitespace from pasted context lines break a plain `git apply`. New files are given in full.

## Global Constraints

- Python 3.11+ floor; `from __future__ import annotations` as the first import in every new file; full type hints. mypy runs as `mypy src/pymarxan/ src/pymarxan_shiny/ --ignore-missing-imports`.
- ruff rules E, F, I, UP; line length 99. **Run `ruff check` on the test files too at every task handoff.**
- Tests run ONLY under `/opt/micromamba/envs/shiny/bin/pytest`, never `.venv`. In a worktree without `.venv`, use `PYTHONPATH=src:.` with the shiny pytest, `~/.local/bin/ruff`, and the main checkout's `.venv/bin/mypy`.
- **Semantics are exactly `compute_connectivity_penalty`'s** (v0.43.0; spec §3):
  - off when `problem.connectivity is None` or `CONNECTIVITY_WEIGHT <= 0`;
  - rows whose ids are not planning units are skipped;
  - symmetric: `−w·Σ v·[x_i ∧ x_j]`, the first row of an `(i, j)` / `(j, i)` pair wins, and a self-pair rewards `v` when selected;
  - asymmetric (`ASYMMETRIC_CONNECTIVITY = 1`): `w·Σ v·[x_i ∧ ¬x_j]` per row, duplicates add up, self-rows dropped.
- **Performance model:** no per-flip allocation beyond what boundary already does; with `conn_mode == 0` the delta adds one integer comparison. Benches are comparative only (`bench` marker; never an absolute wall-clock budget).
- **Nothing else changes:** no golden is regenerated, no existing test changes, and the `tests/data/simple` anchor (MIP cost 35.0; SA/greedy at or above) holds.
- Seed every SA run (never `seed=None`).
- Commit trailer on every commit:
  ```
  Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01AZEiop9qqxVCrMHKqSDYP8
  ```

## Review Focus

These are inputs a user will plausibly hit that the spec's acceptance list doesn't name. Each is pinned by a test in its owning task.
1. **A full N×N matrix upload** lists every pair twice, as `(i, j)` and `(j, i)`: it must count once (first row wins), not double. → symmetric dedupe array test (Task 1) and the full = evaluator property test, whose fixtures include reversed rows.
2. **Connectivity rows naming units that are not planning units** (a stale matrix after deleting PUs) are skipped, not a crash and not a term. → the unknown-id row (id 99) in both property-test fixtures and `test_unknown_ids_only_is_empty_but_on` (Task 1).
3. **An asymmetric self-row `(i, i)`** must not give a phantom `+v` on adding. → self-row dropped (Task 1) and delta exactness over every PU (Task 2).
4. **`CONNECTIVITY_WEIGHT = 0` with a table loaded** (the Shiny slider at 0) costs nothing per flip and changes nothing. → off-case tests (Task 1) and the comparative bench (Task 3).
5. **Iterative improvement's removal pass** must not drop a unit whose removal breaks a strong connection. → II test with a weight-0 control (Task 2).

## Design-review rulings (4-lens review, synthesis `docs/plans/2026-10-17-sa-connectivity-review.md`)

- **R1, Tasks 1–2:** the asymmetric delta does not use `~selected`. `conn_out_rowsum` (Σ out-values per row) is precomputed, and the delta is `rowsum − dot(val, sel)`, so 0/1 int64 and uint8 selections give the bool answer (pinned by tests).
- **R2, Task 2:** the headline SA test also asserts that the plan's connectivity term is non-zero (without it the symmetric case passed on the old code); SA behaviour tests keep only seeds that fail on the old code.
- **R3, Task 1:** the full-objective term uses a precomputed CSR row index, not an O(edges) array rebuilt per call.
- **R4, Task 2:** a cross-copy test: on a small fixture with duplicate, reversed and self rows, the brute-force optimum of the cache's full objective equals the MIP's `mip_objective_value`, and seeded SA reaches it (symmetric and asymmetric). A shared normaliser for the three copies is backlogged.
- **R5, Task 3:** docs: prioritizr equivalence holds for clean input (pymarxan's own rules stated for conflicting, repeated, self and unknown rows); the Zonation solver and the external Marxan binary only report the term; the term is distinct from Marxan's boundary-file connectivity × BLM; the CHANGELOG names downstream effects (BLM calibration, replacement cost, automatic starting temperature), that plans may gain units and cost, and the MIP's edge cases; five backlog lines.
- **Not adopted:** merging the asymmetric out/in arrays (the bench now prints the asymmetric cost); tuple / mode-code cosmetics; II swap and two-step tests (ITIMPTYPE 3 uses the full objective, covered by full = evaluator).

## Rulings made while prototyping

Where the spec was silent, these choices were made; each names its task.
- **Task 1, symmetric self-pairs:** de-duplicated like any pair (the first `(i, i)` row wins), because the evaluator's de-duplication covers self keys too.
- **Task 1, unknown ids:** dropped before de-duplication, as in the evaluator.
- **Task 1, all-unknown table:** connectivity stays on (`conn_mode` 1) with empty arrays; the term is 0 either way.
- **Task 1, defaults for a hand-built cache:** the CSR start arrays default to `zeros(1)`; `from_problem` always builds full-size arrays.
- **Task 2, asymmetric SA fixture:** costs (1, 10, 2) rather than (1, 10, 1), so that PU 1 alone is the cheapest pick for an SA that ignores connectivity, and the test fails without the change.
- **Task 2, delta form:** `np.dot(val, sel[nbrs])` rather than boolean-mask sums (about +20% per call with a degree-4 graph instead of +35%).
- **Task 2, R4 fixture weight:** `CONNECTIVITY_WEIGHT` 3.0; at 1.5 a connectivity-blind SA already found the asymmetric optimum, so the test could not fail before Task 2.
- **Task 1, row sums:** `conn_out_rowsum` is built with `np.bincount` (`np.add.reduceat` mishandles trailing empty rows).
- **Task 3, bench tolerance:** `t_off <= max(1.05·t_base, t_base + 2 µs·N)`.
- **Between Tasks 1 and 2** SA's tracked objective is offset by the initial connectivity value while its plans do not yet change; iterative improvement's swap pass (ITIMPTYPE 3), which uses the full objective, already sees connectivity. Task 2 closes both.

---

### Task 1: Cache connectivity arrays and full-objective term

**Files:**
- Modify: `src/pymarxan/solvers/cache.py`
- Create: `tests/pymarxan/solvers/test_cache_connectivity.py`

**Interfaces:**
- Consumes: `compute_connectivity_penalty(problem, selected, pu_index) -> float` (`solvers/utils.py`, v0.43.0) — the semantics to match, used as the test oracle.
- Produces (`pymarxan.solvers.cache.ProblemCache`, new frozen-dataclass fields appended with defaults): `conn_mode: int` (0 off, 1 symmetric, 2 asymmetric), `conn_weight: float`, `conn_self: np.ndarray` (n_pu,), `conn_out_rowsum: np.ndarray` (n_pu,, Σ out-values per row; R1), CSR `conn_out_start` / `conn_out_idx` / `conn_out_val` (symmetric: each de-duplicated pair in both directions; asymmetric: out-edges, duplicates summed), CSR `conn_in_start` / `conn_in_idx` / `conn_in_val` (asymmetric in-edges; empty otherwise); `conn_out_row: np.ndarray` (source row of each out CSR entry, so the full-objective term rebuilds nothing per call; R3); module-level `_build_connectivity(...)` called from `from_problem`; `ProblemCache._compute_connectivity(selected) -> float` (already multiplied by `conn_weight`), added in `compute_full_objective`.

- [ ] **Step 1: Write the failing tests**

Create `tests/pymarxan/solvers/test_cache_connectivity.py`:

```python
"""ProblemCache connectivity: adjacency arrays and the full-objective term.

The cache's connectivity term must equal ``compute_connectivity_penalty`` (the
reported objective, which equals the MIP's term) on every binary selection.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pymarxan.models.problem import ConservationProblem
from pymarxan.solvers.cache import ProblemCache
from pymarxan.solvers.utils import compute_connectivity_penalty

N_PU = 6


def _problem(
    connectivity: pd.DataFrame | None,
    weight: float = 1.0,
    asymmetric: bool = False,
) -> ConservationProblem:
    """6 PUs (ids 1..6), 2 features; integer data throughout."""
    pu = pd.DataFrame({
        "id": list(range(1, N_PU + 1)),
        "cost": [3.0, 5.0, 2.0, 7.0, 4.0, 6.0],
        "status": [0] * N_PU,
    })
    feat = pd.DataFrame({
        "id": [1, 2], "name": ["a", "b"], "target": [6.0, 4.0], "spf": [10.0, 10.0],
    })
    puvspr = pd.DataFrame({
        "species": [1, 1, 1, 2, 2, 2],
        "pu": [1, 3, 5, 2, 4, 6],
        "amount": [3.0, 2.0, 4.0, 2.0, 3.0, 1.0],
    })
    boundary = pd.DataFrame({
        "id1": [1, 2, 3, 4, 5, 1],
        "id2": [2, 3, 4, 5, 6, 1],
        "boundary": [1.0, 1.0, 1.0, 1.0, 1.0, 2.0],
    })
    problem = ConservationProblem(
        planning_units=pu, features=feat, pu_vs_features=puvspr,
        boundary=boundary, connectivity=connectivity,
    )
    problem.parameters["CONNECTIVITY_WEIGHT"] = weight
    if asymmetric:
        problem.parameters["ASYMMETRIC_CONNECTIVITY"] = 1
    return problem


# Symmetric fixture: reversed duplicate with a DIFFERENT value (first row wins),
# an exact duplicate, a self-pair, an unknown id, and an isolated PU (6).
SYM_CONN = pd.DataFrame({
    "id1": [1, 2, 2, 3, 4, 4, 99, 3],
    "id2": [2, 1, 3, 2, 4, 5, 1, 5],
    "value": [4.0, 9.0, 3.0, 8.0, 5.0, 2.0, 7.0, 1.0],
})

# Asymmetric fixture: duplicate (1, 2) rows (sum), the reverse (2, 1), a self-row
# (dropped), an unknown id, and a second source into 2.
ASYM_CONN = pd.DataFrame({
    "id1": [1, 1, 2, 3, 3, 4, 99, 5],
    "id2": [2, 2, 1, 3, 2, 5, 2, 4],
    "value": [4.0, 6.0, 3.0, 9.0, 2.0, 5.0, 7.0, 1.0],
})


def _csr_rows(start: np.ndarray, idx: np.ndarray, val: np.ndarray) -> dict[int, dict[int, float]]:
    out: dict[int, dict[int, float]] = {}
    for i in range(len(start) - 1):
        row = {int(j): float(v) for j, v in zip(idx[start[i]:start[i + 1]],
                                               val[start[i]:start[i + 1]], strict=True)}
        assert len(row) == start[i + 1] - start[i], "duplicate neighbour in CSR row"
        out[i] = row
    return out


class TestSymmetricArrays:
    def test_mode_and_weight(self) -> None:
        cache = ProblemCache.from_problem(_problem(SYM_CONN, weight=2.5))
        assert cache.conn_mode == 1
        assert cache.conn_weight == 2.5

    def test_pairs_deduped_first_row_wins_both_directions(self) -> None:
        cache = ProblemCache.from_problem(_problem(SYM_CONN))
        rows = _csr_rows(cache.conn_out_start, cache.conn_out_idx, cache.conn_out_val)
        # PU index = id - 1. Pair {1,2}: first row (1,2,4) wins over (2,1,9).
        # Pair {2,3}: first row (2,3,3) wins over (3,2,8).
        assert rows == {
            0: {1: 4.0},
            1: {0: 4.0, 2: 3.0},
            2: {1: 3.0, 4: 1.0},
            3: {4: 2.0},
            4: {3: 2.0, 2: 1.0},
            5: {},
        }

    def test_self_pair_goes_to_conn_self(self) -> None:
        cache = ProblemCache.from_problem(_problem(SYM_CONN))
        np.testing.assert_array_equal(cache.conn_self, [0.0, 0.0, 0.0, 5.0, 0.0, 0.0])

    def test_in_lists_empty_for_symmetric(self) -> None:
        cache = ProblemCache.from_problem(_problem(SYM_CONN))
        assert cache.conn_in_idx.size == 0
        assert cache.conn_in_val.size == 0
        np.testing.assert_array_equal(cache.conn_in_start, np.zeros(N_PU + 1))

    def test_csr_dtypes_match_boundary(self) -> None:
        cache = ProblemCache.from_problem(_problem(SYM_CONN))
        assert cache.conn_out_start.dtype == np.int32
        assert cache.conn_out_idx.dtype == np.int32
        assert cache.conn_out_val.dtype == np.float64
        assert cache.conn_self.dtype == np.float64
        assert cache.conn_out_start.shape == (N_PU + 1,)


class TestPrecomputedRowArrays:
    """Row index per CSR entry and per-row out-value sums, built once (no per-call work)."""

    @pytest.mark.parametrize(("conn", "asymmetric"), [(SYM_CONN, False), (ASYM_CONN, True)])
    def test_conn_out_row_matches_csr(self, conn: pd.DataFrame, asymmetric: bool) -> None:
        cache = ProblemCache.from_problem(_problem(conn, asymmetric=asymmetric))
        want = np.repeat(np.arange(N_PU), np.diff(cache.conn_out_start))
        assert cache.conn_out_row.dtype == np.int32
        np.testing.assert_array_equal(cache.conn_out_row, want)

    def test_out_rowsum_asymmetric(self) -> None:
        cache = ProblemCache.from_problem(_problem(ASYM_CONN, asymmetric=True))
        assert cache.conn_out_rowsum.dtype == np.float64
        np.testing.assert_array_equal(cache.conn_out_rowsum, [10.0, 3.0, 2.0, 5.0, 1.0, 0.0])

    def test_out_rowsum_zero_for_symmetric_and_off(self) -> None:
        for problem in (_problem(SYM_CONN), _problem(SYM_CONN, weight=0.0), _problem(None)):
            cache = ProblemCache.from_problem(problem)
            np.testing.assert_array_equal(cache.conn_out_rowsum, np.zeros(N_PU))
            assert cache.conn_out_row.size == cache.conn_out_idx.size


class TestAsymmetricArrays:
    def test_mode(self) -> None:
        cache = ProblemCache.from_problem(_problem(ASYM_CONN, asymmetric=True))
        assert cache.conn_mode == 2
        assert cache.conn_weight == 1.0

    def test_out_edges_duplicates_summed_self_row_dropped(self) -> None:
        cache = ProblemCache.from_problem(_problem(ASYM_CONN, asymmetric=True))
        rows = _csr_rows(cache.conn_out_start, cache.conn_out_idx, cache.conn_out_val)
        assert rows == {
            0: {1: 10.0},
            1: {0: 3.0},
            2: {1: 2.0},
            3: {4: 5.0},
            4: {3: 1.0},
            5: {},
        }

    def test_in_edges(self) -> None:
        cache = ProblemCache.from_problem(_problem(ASYM_CONN, asymmetric=True))
        rows = _csr_rows(cache.conn_in_start, cache.conn_in_idx, cache.conn_in_val)
        assert rows == {
            0: {1: 3.0},
            1: {0: 10.0, 2: 2.0},
            2: {},
            3: {4: 1.0},
            4: {3: 5.0},
            5: {},
        }

    def test_no_self_rewards(self) -> None:
        cache = ProblemCache.from_problem(_problem(ASYM_CONN, asymmetric=True))
        np.testing.assert_array_equal(cache.conn_self, np.zeros(N_PU))


class TestOff:
    @pytest.mark.parametrize(
        ("conn", "weight"),
        [(None, 1.0), (SYM_CONN, 0.0), (SYM_CONN, -1.0), (ASYM_CONN, 0.0)],
    )
    def test_off_cases(self, conn: pd.DataFrame | None, weight: float) -> None:
        problem = _problem(conn, weight=weight, asymmetric=conn is ASYM_CONN)
        cache = ProblemCache.from_problem(problem)
        assert cache.conn_mode == 0
        assert cache.conn_weight == 0.0
        assert cache.conn_out_idx.size == 0
        assert cache.conn_in_idx.size == 0
        np.testing.assert_array_equal(cache.conn_out_start, np.zeros(N_PU + 1))
        np.testing.assert_array_equal(cache.conn_self, np.zeros(N_PU))
        # The term is 0: the objective equals the no-connectivity objective.
        base = ProblemCache.from_problem(_problem(None))
        rng = np.random.default_rng(7)
        for _ in range(10):
            sel = rng.random(N_PU) < 0.5
            held = cache.compute_held(sel)
            assert cache.compute_full_objective(sel, held, 1.0) == (
                base.compute_full_objective(sel, held, 1.0)
            )

    def test_unknown_ids_only_is_empty_but_on(self) -> None:
        conn = pd.DataFrame({"id1": [99, 1], "id2": [1, 98], "value": [5.0, 5.0]})
        cache = ProblemCache.from_problem(_problem(conn))
        assert cache.conn_mode == 1
        assert cache.conn_out_idx.size == 0
        np.testing.assert_array_equal(cache.conn_out_start, np.zeros(N_PU + 1))


class TestFullObjectiveMatchesEvaluator:
    @pytest.mark.parametrize(
        ("conn", "asymmetric", "weight"),
        [(SYM_CONN, False, 1.0), (SYM_CONN, False, 2.5),
         (ASYM_CONN, True, 1.0), (ASYM_CONN, True, 0.75)],
    )
    def test_full_minus_base_equals_connectivity_penalty(
        self, conn: pd.DataFrame, asymmetric: bool, weight: float,
    ) -> None:
        problem = _problem(conn, weight=weight, asymmetric=asymmetric)
        cache = ProblemCache.from_problem(problem)
        base = ProblemCache.from_problem(_problem(None))
        rng = np.random.default_rng(20261017)
        selections = [np.zeros(N_PU, dtype=bool), np.ones(N_PU, dtype=bool)]
        selections += [rng.random(N_PU) < 0.5 for _ in range(30)]
        nonzero = 0
        for sel in selections:
            held = cache.compute_held(sel)
            got = cache.compute_full_objective(sel, held, 1.0) - (
                base.compute_full_objective(sel, held, 1.0)
            )
            want = compute_connectivity_penalty(problem, sel, cache.pu_id_to_idx)
            assert got == pytest.approx(want, abs=1e-9)
            nonzero += want != 0.0
        assert nonzero >= 20  # the fixture actually exercises the term
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `PYTHONPATH=src:. /opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/solvers/test_cache_connectivity.py -q`
Expected: FAIL — 22 failed (`AttributeError` on the missing `conn_*` fields)

- [ ] **Step 3: Implement**

Modify `src/pymarxan/solvers/cache.py` (apply this diff):

```diff
diff --git a/src/pymarxan/solvers/cache.py b/src/pymarxan/solvers/cache.py
index 7abea85..fcf600c 100644
--- a/src/pymarxan/solvers/cache.py
+++ b/src/pymarxan/solvers/cache.py
@@ -19,19 +19,27 @@ The deterministic-path SPF mask excludes features handled by constraint
 states (Phase 19 ClumpState for ``target2 > 0``; Phase 20 SepState for
 ``sepnum > 1``) to prevent double-counting. The compound mask is
 precomputed as ``_det_spf`` and consumed by both ``compute_full_objective``
 and ``compute_delta_objective``. When adding a new constraint type, edit
 the single mask definition in ``from_problem`` — not the call sites.
+
+Connectivity
+------------
+``CONNECTIVITY_WEIGHT > 0`` with a connectivity table builds CSR adjacency arrays
+(``conn_out_*``; asymmetric also ``conn_in_*``) plus symmetric self-pair rewards
+(``conn_self``), mirroring ``compute_connectivity_penalty`` (the MIP's term). Both
+``compute_full_objective`` and ``compute_delta_objective`` read them; when ``conn_mode``
+is 0 (off) the delta pays one integer comparison.
 """
 from __future__ import annotations
 
 import warnings
 from dataclasses import dataclass, field
 from functools import cached_property
 
 import numpy as np
-from scipy.sparse import csr_matrix
+from scipy.sparse import coo_matrix, csr_matrix
 
 from pymarxan.models.problem import ConservationProblem
 from pymarxan.solvers.utils import compute_cost_threshold_penalty
 
 
@@ -139,10 +147,28 @@ class ProblemCache:
     # Precomputed deterministic-penalty SPF mask: feat_spf * (target2 <= 0) *
     # (sepnum <= 1). Single source of truth (round-3 H14) — both compute_full_objective
     # and compute_delta_objective read this rather than reconstructing the
     # mask. Falls back to feat_spf when neither constraint is active.
     _det_spf: np.ndarray = field(default_factory=lambda: np.zeros(0))
+    # Connectivity (CONNECTIVITY_WEIGHT; mirrors compute_connectivity_penalty / the MIP).
+    # conn_mode: 0 off (no table or weight <= 0), 1 symmetric, 2 asymmetric. CSR layout
+    # as for boundary. Symmetric: each de-duplicated pair stored in both directions in
+    # conn_out_*, self-pair rewards in conn_self, conn_in_* empty. Asymmetric: out-edges
+    # i -> j and in-edges k -> i, duplicate rows summed, self-rows dropped.
+    # conn_out_row[e] = source row of CSR entry e (full objective, no per-call rebuild);
+    # conn_out_rowsum[i] = Σ out-values of row i (asymmetric delta; zeros otherwise).
+    conn_mode: int = 0
+    conn_weight: float = 0.0
+    conn_self: np.ndarray = field(default_factory=lambda: np.zeros(0))
+    conn_out_start: np.ndarray = field(default_factory=lambda: np.zeros(1, dtype=np.int32))
+    conn_out_idx: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int32))
+    conn_out_val: np.ndarray = field(default_factory=lambda: np.zeros(0))
+    conn_in_start: np.ndarray = field(default_factory=lambda: np.zeros(1, dtype=np.int32))
+    conn_in_idx: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int32))
+    conn_in_val: np.ndarray = field(default_factory=lambda: np.zeros(0))
+    conn_out_row: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int32))
+    conn_out_rowsum: np.ndarray = field(default_factory=lambda: np.zeros(0))
 
     @cached_property
     def pu_feat_matrix(self) -> np.ndarray:
         """Dense (n_pu, n_feat) view — densified on first access (clumping / separation /
         probmode-3 / analysis). A plain problem never triggers this."""
@@ -434,10 +460,24 @@ class ProblemCache:
         if clumping_active:
             det_spf = det_spf * (feat_target2 <= 0).astype(np.float64)
         if separation_active:
             det_spf = det_spf * (feat_sepnum <= 1).astype(np.float64)
 
+        (
+            conn_mode, conn_weight, conn_self,
+            conn_out_start, conn_out_idx, conn_out_val,
+            conn_in_start, conn_in_idx, conn_in_val,
+        ) = _build_connectivity(problem, pu_id_to_idx, n_pu)
+        conn_out_row = np.repeat(
+            np.arange(n_pu, dtype=np.int32), np.diff(conn_out_start),
+        ).astype(np.int32)
+        conn_out_rowsum = (
+            np.bincount(conn_out_row, weights=conn_out_val, minlength=n_pu)
+            if conn_mode == 2
+            else np.zeros(n_pu, dtype=np.float64)
+        )
+
         return cls(
             n_pu=n_pu,
             n_feat=n_feat,
             costs=costs,
             statuses=statuses,
@@ -468,10 +508,21 @@ class ProblemCache:
             feat_sepnum=feat_sepnum,
             pu_coords=pu_coords,
             separation_active=separation_active,
             pu_to_sep_feats=pu_to_sep_feats,
             _det_spf=det_spf,
+            conn_mode=conn_mode,
+            conn_weight=conn_weight,
+            conn_self=conn_self,
+            conn_out_start=conn_out_start,
+            conn_out_idx=conn_out_idx,
+            conn_out_val=conn_out_val,
+            conn_in_start=conn_in_start,
+            conn_in_idx=conn_in_idx,
+            conn_in_val=conn_in_val,
+            conn_out_row=conn_out_row,
+            conn_out_rowsum=conn_out_rowsum,
         )
 
     # ------------------------------------------------------------------
     # Compute methods
     # ------------------------------------------------------------------
@@ -546,10 +597,11 @@ class ProblemCache:
         blm: float,
     ) -> float:
         """Compute the full Marxan objective value.
 
         objective = cost + BLM * boundary + penalty [+ cost_threshold_penalty]
+        [+ connectivity]
 
         Parameters
         ----------
         selected : np.ndarray
             (n_pu,) boolean selection array.
@@ -589,10 +641,14 @@ class ProblemCache:
         if self.probmode == 3:
             held_expected = self.expected_matrix[selected].sum(axis=0)
             held_variance = self.var_matrix[selected].sum(axis=0)
             obj += self._compute_zscore_penalty(held_expected, held_variance)
 
+        # Connectivity term (no-op when conn_mode == 0)
+        if self.conn_mode:
+            obj += self._compute_connectivity(selected)
+
         return obj
 
     def compute_delta_objective(
         self,
         idx: int,
@@ -730,10 +786,29 @@ class ProblemCache:
 
     # ------------------------------------------------------------------
     # Internal helpers
     # ------------------------------------------------------------------
 
+    def _compute_connectivity(self, selected: np.ndarray) -> float:
+        """Connectivity term for a binary selection; equals ``compute_connectivity_penalty``.
+
+        Symmetric: ``-w * (sum of conn_self over selected + sum over pairs with both ends
+        selected)`` (each pair counted once via ``nbr > row``). Asymmetric: ``w * sum over
+        out-edges whose source is selected and sink is not``.
+        """
+        if self.conn_mode == 0:
+            return 0.0
+        sel = np.asarray(selected, dtype=bool)
+        rows = self.conn_out_row
+        nbrs = self.conn_out_idx
+        if self.conn_mode == 1:
+            both = sel[rows] & sel[nbrs] & (nbrs > rows)
+            total = float(np.sum(self.conn_self[sel])) + float(np.sum(self.conn_out_val[both]))
+            return -self.conn_weight * total
+        broken = sel[rows] & ~sel[nbrs]
+        return self.conn_weight * float(np.sum(self.conn_out_val[broken]))
+
     def _compute_boundary(self, selected: np.ndarray) -> float:
         """Compute total boundary for a selection using cached arrays.
 
         Parameters
         ----------
@@ -780,5 +855,92 @@ class ProblemCache:
         # If i is selected and j is not: i adds boundary w.
         # If j is not selected, we don't iterate j. So only counted once.
         # Correct.
         
         return total
+
+
+_ConnArrays = tuple[
+    int, float, np.ndarray,
+    np.ndarray, np.ndarray, np.ndarray,
+    np.ndarray, np.ndarray, np.ndarray,
+]
+
+
+def _build_connectivity(
+    problem: ConservationProblem,
+    pu_id_to_idx: dict[int, int],
+    n_pu: int,
+) -> _ConnArrays:
+    """Build the connectivity arrays (see the ``conn_*`` fields of ``ProblemCache``).
+
+    Semantics are exactly ``compute_connectivity_penalty``: off when there is no table or
+    ``CONNECTIVITY_WEIGHT <= 0``; rows with an id that is not a planning unit are skipped;
+    symmetric pairs are de-duplicated by unordered key with the first row winning (a
+    self-pair goes to ``conn_self``); asymmetric duplicate rows add up and self-rows
+    (always 0) are dropped. Returns ``(mode, weight, conn_self, out_start, out_idx,
+    out_val, in_start, in_idx, in_val)``.
+    """
+    conn_self = np.zeros(n_pu, dtype=np.float64)
+    empty_idx = np.zeros(0, dtype=np.int32)
+    empty_val = np.zeros(0, dtype=np.float64)
+    weight = float(problem.parameters.get("CONNECTIVITY_WEIGHT", 0.0))
+    if problem.connectivity is None or weight <= 0.0:
+        return (
+            0, 0.0, conn_self,
+            np.zeros(n_pu + 1, dtype=np.int32), empty_idx, empty_val,
+            np.zeros(n_pu + 1, dtype=np.int32), empty_idx, empty_val,
+        )
+    asymmetric = bool(int(problem.parameters.get("ASYMMETRIC_CONNECTIVITY", 0)))
+
+    conn = problem.connectivity
+    n_rows = len(conn)
+    src = np.fromiter(
+        (pu_id_to_idx.get(int(i), -1) for i in conn["id1"].values), np.int64, n_rows,
+    )
+    dst = np.fromiter(
+        (pu_id_to_idx.get(int(j), -1) for j in conn["id2"].values), np.int64, n_rows,
+    )
+    val = np.asarray(conn["value"].values, dtype=np.float64)
+    known = (src >= 0) & (dst >= 0)
+    src, dst, val = src[known], dst[known], val[known]
+
+    if asymmetric:
+        mode = 2
+        nonself = src != dst
+        src, dst, val = src[nonself], dst[nonself], val[nonself]
+    else:
+        mode = 1
+        lo = np.minimum(src, dst)
+        hi = np.maximum(src, dst)
+        # First row of each unordered pair wins (np.unique's index is the first occurrence).
+        _, first = np.unique(lo * n_pu + hi, return_index=True)
+        first.sort()
+        lo, hi, val = lo[first], hi[first], val[first]
+        is_self = lo == hi
+        conn_self[lo[is_self]] = val[is_self]
+        lo, hi, val = lo[~is_self], hi[~is_self], val[~is_self]
+        src = np.concatenate([lo, hi])
+        dst = np.concatenate([hi, lo])
+        val = np.concatenate([val, val])
+
+    # COO -> CSR sums duplicate (i, j) entries (asymmetric duplicates add up).
+    out = coo_matrix((val, (src, dst)), shape=(n_pu, n_pu)).tocsr()
+    out.sum_duplicates()
+    out.sort_indices()
+    out_start = out.indptr.astype(np.int32)
+    out_idx = out.indices.astype(np.int32)
+    out_val = out.data.astype(np.float64)
+    if mode == 2:
+        inc = out.transpose().tocsr()
+        inc.sort_indices()
+        in_start = inc.indptr.astype(np.int32)
+        in_idx = inc.indices.astype(np.int32)
+        in_val = inc.data.astype(np.float64)
+    else:
+        in_start = np.zeros(n_pu + 1, dtype=np.int32)
+        in_idx, in_val = empty_idx, empty_val
+    return (
+        mode, weight, conn_self,
+        out_start, out_idx, out_val,
+        in_start, in_idx, in_val,
+    )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `PYTHONPATH=src:. /opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/solvers/test_cache_connectivity.py -q`
Expected: 22 passed

- [ ] **Step 5: Gate**

Run: `~/.local/bin/ruff check src tests && /home/razinka/marxan/.venv/bin/mypy src/pymarxan/ src/pymarxan_shiny/ --ignore-missing-imports && PYTHONPATH=src:. /opt/micromamba/envs/shiny/bin/pytest tests/ -q -m "not bench" -p no:cacheprovider`
Expected: ruff and mypy clean; the full suite passes, no golden regenerated, no existing test changed (prototype: 3583 passed after Task 1, 3605 after Tasks 2 and 3; 101 skipped HiGHS cases, which run in CI; 21 bench deselected after Task 3, 20 before).

- [ ] **Step 6: Commit**

```bash
git add src/pymarxan/solvers/cache.py tests/pymarxan/solvers/test_cache_connectivity.py
git commit -m "feat(cache): connectivity adjacency arrays and full-objective term

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01AZEiop9qqxVCrMHKqSDYP8"
```

---

### Task 2: Delta term: SA and iterative improvement optimise connectivity

**Files:**
- Modify: `src/pymarxan/solvers/cache.py`
- Modify: `tests/pymarxan/solvers/test_cache_connectivity.py`
- Create: `tests/pymarxan/solvers/test_sa_connectivity.py`

**Interfaces:**
- Consumes: Task 1's `conn_mode`, `conn_weight`, `conn_self`, `conn_out_*`, `conn_in_*`, `_compute_connectivity`.
- Produces: `ProblemCache.compute_delta_objective` includes the connectivity delta (spec §5): symmetric `−s·w·(conn_self[idx] + Σ_out v·sel[nbr])`, asymmetric `s·w·Σ_out v·(1 − sel[j]) − s·w·Σ_in v·sel[k]`, guarded by `if self.conn_mode:`. No change to `simulated_annealing.py` or `iterative_improvement.py`.

- [ ] **Step 1: Write the failing tests**

Modify `tests/pymarxan/solvers/test_cache_connectivity.py` (apply this diff):

```diff
diff --git a/tests/pymarxan/solvers/test_cache_connectivity.py b/tests/pymarxan/solvers/test_cache_connectivity.py
index e7d3fd6..5bac105 100644
--- a/tests/pymarxan/solvers/test_cache_connectivity.py
+++ b/tests/pymarxan/solvers/test_cache_connectivity.py
@@ -228,5 +228,65 @@ class TestFullObjectiveMatchesEvaluator:
             )
             want = compute_connectivity_penalty(problem, sel, cache.pu_id_to_idx)
             assert got == pytest.approx(want, abs=1e-9)
             nonzero += want != 0.0
         assert nonzero >= 20  # the fixture actually exercises the term
+
+
+class TestDeltaExactness:
+    @pytest.mark.parametrize(
+        ("conn", "asymmetric", "weight"),
+        [(SYM_CONN, False, 1.0), (SYM_CONN, False, 2.5),
+         (ASYM_CONN, True, 1.0), (ASYM_CONN, True, 0.75)],
+    )
+    def test_delta_equals_full_difference(
+        self, conn: pd.DataFrame, asymmetric: bool, weight: float,
+    ) -> None:
+        cache = ProblemCache.from_problem(_problem(conn, weight=weight, asymmetric=asymmetric))
+        blm = 1.0
+        rng = np.random.default_rng(42)
+        states = [np.zeros(N_PU, dtype=bool), np.ones(N_PU, dtype=bool)]
+        states += [rng.random(N_PU) < 0.5 for _ in range(8)]
+        for sel in states:
+            held = cache.compute_held(sel)
+            before = cache.compute_full_objective(sel, held, blm)
+            total_cost = float(np.sum(cache.costs[sel]))
+            for idx in range(N_PU):
+                delta = cache.compute_delta_objective(idx, sel, held, total_cost, blm)
+                flipped = sel.copy()
+                flipped[idx] = not flipped[idx]
+                after = cache.compute_full_objective(flipped, cache.compute_held(flipped), blm)
+                assert delta == pytest.approx(after - before, abs=1e-9), (sel, idx)
+
+    @pytest.mark.parametrize(("conn", "asymmetric"), [(SYM_CONN, False), (ASYM_CONN, True)])
+    @pytest.mark.parametrize("dtype", [np.int64, np.uint8])
+    def test_integer_selection_dtypes_match_bool(
+        self, conn: pd.DataFrame, asymmetric: bool, dtype: type,
+    ) -> None:
+        """A 0/1 integer selection gives the bool delta (no bitwise ``~`` on ints)."""
+        cache = ProblemCache.from_problem(_problem(conn, asymmetric=asymmetric))
+        rng = np.random.default_rng(3)
+        for sel in [np.zeros(N_PU, dtype=bool)] + [rng.random(N_PU) < 0.5 for _ in range(6)]:
+            held = cache.compute_held(sel)
+            before = cache.compute_full_objective(sel, held, 1.0)
+            cost = float(np.sum(cache.costs[sel]))
+            as_int = sel.astype(dtype)
+            for idx in range(N_PU):
+                want = cache.compute_delta_objective(idx, sel, held, cost, 1.0)
+                got = cache.compute_delta_objective(idx, as_int, held, cost, 1.0)
+                assert got == pytest.approx(want, abs=1e-9), (dtype, sel, idx)
+                flipped = sel.copy()
+                flipped[idx] = not flipped[idx]
+                after = cache.compute_full_objective(flipped, cache.compute_held(flipped), 1.0)
+                assert got == pytest.approx(after - before, abs=1e-9), (dtype, sel, idx)
+
+    def test_off_delta_unchanged(self) -> None:
+        """Weight 0 with a table: delta identical to the no-table cache."""
+        cache = ProblemCache.from_problem(_problem(SYM_CONN, weight=0.0))
+        base = ProblemCache.from_problem(_problem(None))
+        sel = np.array([True, False, True, True, False, True])
+        held = cache.compute_held(sel)
+        cost = float(np.sum(cache.costs[sel]))
+        for idx in range(N_PU):
+            assert cache.compute_delta_objective(idx, sel, held, cost, 1.0) == (
+                base.compute_delta_objective(idx, sel, held, cost, 1.0)
+            )
```

Create `tests/pymarxan/solvers/test_sa_connectivity.py`:

```python
"""Simulated annealing and iterative improvement optimise connectivity (v0.44.0).

Before v0.44.0 the SA/II cache had no connectivity term: they searched as if
connectivity did not exist, and SA's tracked best objective disagreed with the
objective it reported.
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import pytest

from pymarxan.models.problem import ConservationProblem
from pymarxan.solvers.base import SolverConfig
from pymarxan.solvers.cache import ProblemCache
from pymarxan.solvers.iterative_improvement import IterativeImprovementSolver
from pymarxan.solvers.mip_solver import MIPSolver
from pymarxan.solvers.simulated_annealing import SimulatedAnnealingSolver
from pymarxan.solvers.utils import compute_connectivity_penalty


def _grid_problem(
    connectivity: pd.DataFrame | None,
    asymmetric: bool = False,
) -> ConservationProblem:
    """4x4 grid, 3 features; every number is an integer so SA's accumulated
    objective is exact and can be compared to the reported one at 1e-9."""
    rng = np.random.default_rng(5)
    n = 16
    pu = pd.DataFrame({
        "id": np.arange(1, n + 1),
        "cost": rng.integers(1, 10, size=n).astype(float),
        "status": np.zeros(n, dtype=int),
    })
    feat = pd.DataFrame({
        "id": [1, 2, 3], "name": ["a", "b", "c"],
        "target": [8.0, 6.0, 5.0], "spf": [20.0, 20.0, 20.0],
    })
    rows = [(f, p, float(rng.integers(1, 5)))
            for f in (1, 2, 3) for p in range(1, n + 1) if rng.random() < 0.4]
    puvspr = pd.DataFrame(rows, columns=["species", "pu", "amount"])
    b_rows = []
    for i in range(1, n + 1):
        if i % 4 != 0:
            b_rows.append((i, i + 1, 1.0))
        if i + 4 <= n:
            b_rows.append((i, i + 4, 1.0))
    boundary = pd.DataFrame(b_rows, columns=["id1", "id2", "boundary"])
    problem = ConservationProblem(
        planning_units=pu, features=feat, pu_vs_features=puvspr,
        boundary=boundary, connectivity=connectivity,
    )
    problem.parameters.update({
        "BLM": 1.0, "NUMITNS": 20_000, "NUMTEMP": 100, "CONNECTIVITY_WEIGHT": 2.0,
    })
    if asymmetric:
        problem.parameters["ASYMMETRIC_CONNECTIVITY"] = 1
    return problem


def _grid_connectivity() -> pd.DataFrame:
    rng = np.random.default_rng(11)
    rows = [(int(rng.integers(1, 17)), int(rng.integers(1, 17)), float(rng.integers(1, 9)))
            for _ in range(40)]
    return pd.DataFrame(rows, columns=["id1", "id2", "value"])


def _solve_sa(problem: ConservationProblem, seed: int = 42):
    sols = SimulatedAnnealingSolver().solve(problem, SolverConfig(num_solutions=1, seed=seed))
    assert len(sols) == 1
    return sols[0]


class TestSAObjectiveConsistency:
    def test_control_without_connectivity(self) -> None:
        sol = _solve_sa(_grid_problem(None))
        assert sol.objective == pytest.approx(
            sol.metadata["history"]["best_objective"][-1], abs=1e-9,
        )

    @pytest.mark.parametrize("asymmetric", [False, True])
    def test_reported_objective_equals_tracked_best(self, asymmetric: bool) -> None:
        problem = _grid_problem(_grid_connectivity(), asymmetric=asymmetric)
        sol = _solve_sa(problem)
        # The plan's connectivity term is non-zero, so the equality below exercises it.
        pu_index = {int(pid): i for i, pid in enumerate(problem.planning_units["id"])}
        assert compute_connectivity_penalty(problem, sol.selected, pu_index) != 0.0
        assert sol.objective == pytest.approx(
            sol.metadata["history"]["best_objective"][-1], abs=1e-9,
        )


def _three_pu(
    connectivity: pd.DataFrame | None,
    costs: tuple[float, float, float],
    params: dict[str, float],
) -> ConservationProblem:
    """The test_connectivity_mip fixture shape: 3 PUs, 1 feature, any one PU meets it."""
    problem = ConservationProblem(
        planning_units=pd.DataFrame({"id": [1, 2, 3], "cost": list(costs), "status": [0] * 3}),
        features=pd.DataFrame({"id": [1], "name": ["f1"], "target": [1.0], "spf": [1.0]}),
        pu_vs_features=pd.DataFrame({
            "species": [1, 1, 1], "pu": [1, 2, 3], "amount": [5.0, 5.0, 5.0],
        }),
        connectivity=connectivity,
    )
    problem.parameters.update({"NUMITNS": 2000, "NUMTEMP": 20, **params})
    return problem


def _ids(sol) -> set[int]:
    return {i + 1 for i in np.flatnonzero(sol.selected)}


class TestSABehaviour:
    # Seeds chosen so each case fails on a connectivity-blind SA (v0.43.0 cache).
    @pytest.mark.parametrize("seed", [1, 3, 6])
    def test_symmetric_strong_link_selects_both(self, seed: int) -> None:
        conn = pd.DataFrame({"id1": [1, 2], "id2": [2, 3], "value": [100.0, 0.1]})
        problem = _three_pu(conn, (1.0, 10.0, 1.0), {"CONNECTIVITY_WEIGHT": 1.0})
        assert {1, 2} <= _ids(_solve_sa(problem, seed))

    @pytest.mark.parametrize("seed", [1, 3, 6])
    def test_asymmetric_never_source_without_sink(self, seed: int) -> None:
        # PU 1 is the cheapest single unit, so a connectivity-blind SA picks it alone.
        conn = pd.DataFrame({"id1": [1], "id2": [2], "value": [50.0]})
        problem = _three_pu(
            conn, (1.0, 10.0, 2.0),
            {"CONNECTIVITY_WEIGHT": 1.0, "ASYMMETRIC_CONNECTIVITY": 1},
        )
        chosen = _ids(_solve_sa(problem, seed))
        assert 1 not in chosen or 2 in chosen, chosen


class TestIterativeImprovement:
    CONN = pd.DataFrame({"id1": [1], "id2": [2], "value": [100.0]})

    def _solve(self, weight: float) -> set[int]:
        problem = _three_pu(
            self.CONN, (1.0, 10.0, 1.0), {"CONNECTIVITY_WEIGHT": weight, "ITIMPTYPE": 1},
        )
        sols = IterativeImprovementSolver().solve(problem, SolverConfig(num_solutions=1))
        return _ids(sols[0])

    def test_removal_drops_expensive_unit_without_connectivity(self) -> None:
        assert self._solve(0.0) == {3}

    def test_removal_keeps_strongly_connected_pair(self) -> None:
        assert self._solve(1.0) == {1, 2}


# Cross-copy fixture: duplicate, reversed (different value) and self rows; no unknown ids,
# non-negative values (the MIP rejects unknown ids and cannot bound negative asymmetric values).
_CLEAN_SYM = pd.DataFrame({
    "id1": [1, 2, 2, 3, 4, 4, 3, 1],
    "id2": [2, 1, 3, 2, 4, 5, 5, 2],
    "value": [4.0, 9.0, 3.0, 8.0, 5.0, 2.0, 1.0, 6.0],
})
_CLEAN_ASYM = pd.DataFrame({
    "id1": [1, 1, 2, 3, 3, 4, 5, 6],
    "id2": [2, 2, 1, 3, 2, 5, 4, 3],
    "value": [4.0, 6.0, 3.0, 9.0, 2.0, 5.0, 1.0, 7.0],
})


def _six_pu(connectivity: pd.DataFrame, asymmetric: bool) -> ConservationProblem:
    """6 PUs, 2 features, BLM 1; SPF high enough that the penalty optimum meets targets."""
    problem = ConservationProblem(
        planning_units=pd.DataFrame({
            "id": [1, 2, 3, 4, 5, 6],
            "cost": [3.0, 5.0, 2.0, 7.0, 4.0, 6.0],
            "status": [0] * 6,
        }),
        features=pd.DataFrame({
            "id": [1, 2], "name": ["a", "b"], "target": [6.0, 4.0], "spf": [1000.0, 1000.0],
        }),
        pu_vs_features=pd.DataFrame({
            "species": [1, 1, 1, 2, 2, 2],
            "pu": [1, 3, 5, 2, 4, 6],
            "amount": [3.0, 2.0, 4.0, 2.0, 3.0, 1.0],
        }),
        boundary=pd.DataFrame({
            "id1": [1, 2, 3, 4, 5, 1],
            "id2": [2, 3, 4, 5, 6, 1],
            "boundary": [1.0, 1.0, 1.0, 1.0, 1.0, 2.0],
        }),
        connectivity=connectivity,
    )
    problem.parameters.update({
        "BLM": 1.0, "CONNECTIVITY_WEIGHT": 3.0, "NUMITNS": 20_000, "NUMTEMP": 100,
    })
    if asymmetric:
        problem.parameters["ASYMMETRIC_CONNECTIVITY"] = 1
    return problem


class TestCrossCopyAgreement:
    """Cache (brute force), MIP and SA agree on the optimum of the same connectivity problem."""

    @pytest.mark.parametrize(("conn", "asymmetric"), [(_CLEAN_SYM, False), (_CLEAN_ASYM, True)])
    def test_brute_force_mip_and_sa_agree(self, conn: pd.DataFrame, asymmetric: bool) -> None:
        problem = _six_pu(conn, asymmetric)
        cache = ProblemCache.from_problem(problem)
        brute = min(
            cache.compute_full_objective(sel, cache.compute_held(sel), 1.0)
            for sel in (np.array(bits, dtype=bool)
                        for bits in itertools.product([False, True], repeat=6))
        )
        mip = MIPSolver(mip_backend="cbc").solve(problem, SolverConfig(num_solutions=1))[0]
        assert mip.all_targets_met
        assert mip.metadata["mip_objective_value"] == pytest.approx(brute, abs=1e-6)
        sa = _solve_sa(problem)
        assert sa.objective == pytest.approx(brute, abs=1e-9)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `PYTHONPATH=src:. /opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/solvers/test_cache_connectivity.py tests/pymarxan/solvers/test_sa_connectivity.py -q`
Expected: FAIL — 19 failed, 25 passed — 4 delta-exactness, 4 integer-dtype (R1), 2 headline SA consistency (symmetric fails on the non-zero connectivity check, asymmetric on reported vs tracked), 6 SA behaviour (seeds 1, 3, 6), 1 II removal and 2 cross-copy SA (R4); Task 1's 22 tests and 3 controls pass

- [ ] **Step 3: Implement**

Modify `src/pymarxan/solvers/cache.py` (apply this diff):

```diff
diff --git a/src/pymarxan/solvers/cache.py b/src/pymarxan/solvers/cache.py
index fcf600c..e8bf3c4 100644
--- a/src/pymarxan/solvers/cache.py
+++ b/src/pymarxan/solvers/cache.py
@@ -780,10 +780,38 @@ class ProblemCache:
             prob_after = self._compute_zscore_penalty(
                 held_exp_after, held_var_after,
             )
             delta += prob_after - prob_before
 
+        # --- Connectivity delta (one integer comparison when off) ---
+        # O(degree), against the pre-flip selection; s = sign (+1 add, -1 remove).
+        #   symmetric:  -s·w·(conn_self[idx] + Σ_out v·sel[nbr])
+        #   asymmetric:  s·w·(rowsum[idx] − Σ_out v·sel[j]) − s·w·Σ_in v·sel[k]
+        # np.dot on the 0/1 selection (any dtype) — never ``~selected``, which is bitwise
+        # on integer arrays.
+        if self.conn_mode:
+            cs = self.conn_out_start[idx]
+            ce = self.conn_out_start[idx + 1]
+            out_sel = 0.0
+            if cs < ce:
+                out_sel = float(
+                    np.dot(self.conn_out_val[cs:ce], selected[self.conn_out_idx[cs:ce]])
+                )
+            if self.conn_mode == 1:
+                delta -= sign * self.conn_weight * (self.conn_self[idx] + out_sel)
+            else:
+                in_sel = 0.0
+                ks = self.conn_in_start[idx]
+                ke = self.conn_in_start[idx + 1]
+                if ks < ke:
+                    in_sel = float(
+                        np.dot(self.conn_in_val[ks:ke], selected[self.conn_in_idx[ks:ke]])
+                    )
+                delta += sign * self.conn_weight * (
+                    self.conn_out_rowsum[idx] - out_sel - in_sel
+                )
+
         return float(delta)
 
     # ------------------------------------------------------------------
     # Internal helpers
     # ------------------------------------------------------------------
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `PYTHONPATH=src:. /opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/solvers/test_cache_connectivity.py tests/pymarxan/solvers/test_sa_connectivity.py -q`
Expected: 44 passed

- [ ] **Step 5: Gate**

Run: `~/.local/bin/ruff check src tests && /home/razinka/marxan/.venv/bin/mypy src/pymarxan/ src/pymarxan_shiny/ --ignore-missing-imports && PYTHONPATH=src:. /opt/micromamba/envs/shiny/bin/pytest tests/ -q -m "not bench" -p no:cacheprovider`
Expected: ruff and mypy clean; the full suite passes, no golden regenerated, no existing test changed (prototype: 3583 passed after Task 1, 3605 after Tasks 2 and 3; 101 skipped HiGHS cases, which run in CI; 21 bench deselected after Task 3, 20 before).

- [ ] **Step 6: Commit**

```bash
git add src/pymarxan/solvers/cache.py tests/pymarxan/solvers/test_cache_connectivity.py tests/pymarxan/solvers/test_sa_connectivity.py
git commit -m "feat(sa): simulated annealing and iterative improvement optimise connectivity

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01AZEiop9qqxVCrMHKqSDYP8"
```

---

### Task 3: Bench, changelog, manual, roadmap

**Files:**
- Modify: `CHANGELOG.md`
- Modify: `docs/USER_MANUAL.md`
- Modify: `docs/plans/2026-10-11-prioritizr-parity-roadmap.md`
- Modify: `tests/benchmarks/bench_sa.py`

**Interfaces:**
- Consumes: Tasks 1–2.
- Produces: `tests/benchmarks/bench_sa.py` comparative connectivity bench (`bench` marker; excluded from CI and `make check`); CHANGELOG `[Unreleased]` `### Changed`; `docs/USER_MANUAL.md` connectivity paragraph; roadmap backlog (SA item done; PROBMODE 1 premium gap added). No source change.

- [ ] **Step 1: Write the comparative bench**

Modify `tests/benchmarks/bench_sa.py` (apply this diff):

```diff
diff --git a/tests/benchmarks/bench_sa.py b/tests/benchmarks/bench_sa.py
index 38bf0e1..50e65ad 100644
--- a/tests/benchmarks/bench_sa.py
+++ b/tests/benchmarks/bench_sa.py
@@ -1,13 +1,16 @@
 """Simulated annealing performance benchmarks."""
 from __future__ import annotations
 
 import time
 
+import numpy as np
 import pytest
 
+from pymarxan.models.problem import ConservationProblem
 from pymarxan.solvers.base import SolverConfig
+from pymarxan.solvers.cache import ProblemCache
 from pymarxan.solvers.simulated_annealing import SimulatedAnnealingSolver
 from tests.benchmarks.conftest import make_problem
 
 pytestmark = pytest.mark.bench
 
@@ -63,5 +66,73 @@ class TestSAPerformance:
         elapsed = time.perf_counter() - start
 
         assert len(solutions) == 1
         assert solutions[0].cost > 0
         assert elapsed < 60.0, f"Large SA took {elapsed:.2f}s, expected <60s"
+
+
+def _delta_loop_time(problem: ConservationProblem, n_calls: int, reps: int = 5) -> float:
+    """Best-of-``reps`` seconds for ``n_calls`` compute_delta_objective calls.
+
+    Identical seeded selection and flip sequence for every problem of the same size, so
+    timings are directly comparable within one test.
+    """
+    cache = ProblemCache.from_problem(problem)
+    rng = np.random.default_rng(0)
+    selected = rng.random(cache.n_pu) < 0.5
+    held = cache.compute_held(selected)
+    total_cost = float(np.sum(cache.costs[selected]))
+    flips = rng.integers(cache.n_pu, size=n_calls)
+    best = float("inf")
+    for _ in range(reps):
+        start = time.perf_counter()
+        for idx in flips:
+            cache.compute_delta_objective(int(idx), selected, held, total_cost, 1.0)
+        best = min(best, time.perf_counter() - start)
+    return best
+
+
+class TestSAConnectivityDeltaCost:
+    """v0.44.0: connectivity in the SA delta. Comparative only (machine-relative)."""
+
+    N_CALLS = 20_000
+
+    def _problems(self) -> tuple[ConservationProblem, ...]:
+        """No table; table + weight 0; symmetric weight 1; asymmetric weight 1."""
+        base = make_problem(n_pu=1000, n_feat=50, density=0.3, seed=42)
+        assert base.boundary is not None
+        conn = base.boundary.rename(columns={"boundary": "value"})  # grid graph, degree ~4
+        conn["value"] = 1.0
+        # copy_with is shallow: give each variant its own parameters dict.
+        off = base.copy_with(
+            connectivity=conn, parameters={**base.parameters, "CONNECTIVITY_WEIGHT": 0.0},
+        )
+        on = base.copy_with(
+            connectivity=conn, parameters={**base.parameters, "CONNECTIVITY_WEIGHT": 1.0},
+        )
+        asym = base.copy_with(
+            connectivity=conn,
+            parameters={
+                **base.parameters, "CONNECTIVITY_WEIGHT": 1.0, "ASYMMETRIC_CONNECTIVITY": 1,
+            },
+        )
+        return base, off, on, asym
+
+    def test_connectivity_off_costs_nothing(self) -> None:
+        base, off, on, asym = self._problems()
+        assert ProblemCache.from_problem(off).conn_mode == 0
+        assert ProblemCache.from_problem(on).conn_mode == 1
+        assert ProblemCache.from_problem(asym).conn_mode == 2
+        t_base = _delta_loop_time(base, self.N_CALLS)
+        t_off = _delta_loop_time(off, self.N_CALLS)
+        t_on = _delta_loop_time(on, self.N_CALLS)
+        t_asym = _delta_loop_time(asym, self.N_CALLS)
+        per_call = 1e6 / self.N_CALLS
+        print(
+            f"\ndelta per call: no table {t_base * per_call:.2f} us, "
+            f"table+weight 0 {t_off * per_call:.2f} us (x{t_off / t_base:.3f}), "
+            f"degree~4 weight 1 symmetric {t_on * per_call:.2f} us (x{t_on / t_base:.3f}), "
+            f"asymmetric {t_asym * per_call:.2f} us (x{t_asym / t_base:.3f})"
+        )
+        # Off must be free: <= 5% over the no-table baseline, with a 2 us/call floor
+        # absorbing timer noise on fast machines.
+        assert t_off <= max(1.05 * t_base, t_base + 2e-6 * self.N_CALLS)
```

- [ ] **Step 2: Write the docs**

Modify `CHANGELOG.md` (apply this diff):

```diff
diff --git a/CHANGELOG.md b/CHANGELOG.md
index bc9d2c9..2d285b6 100644
--- a/CHANGELOG.md
+++ b/CHANGELOG.md
@@ -5,10 +5,32 @@ All notable changes to **pymarxan** are documented in this file.
 The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
 and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
 
 ## [Unreleased]
 
+### Changed
+- **Simulated annealing and iterative improvement now optimise connectivity (the
+  `CONNECTIVITY_WEIGHT` term; connectivity encoded in the boundary file under `BLM` was already
+  optimised), so their plans change on every problem with `CONNECTIVITY_WEIGHT > 0` and a
+  connectivity table.** The `ProblemCache` behind SA and iterative improvement had no
+  connectivity term: they searched as if it did not exist and only the MIP optimised it. The
+  cache now holds connectivity adjacency arrays and adds the term `compute_connectivity_penalty`
+  reports to its full objective and its O(degree) per-flip delta: symmetric pairs rewarded when
+  both units are selected (first row of a pair wins, a self-pair rewards its unit), asymmetric
+  rows charged when the source is selected without the sink (duplicate rows add up, self-rows
+  ignored). Expect SA/II plans to gain strongly connected units (symmetric) or the sinks of
+  selected sources (asymmetric), at higher cost, as the MIP's already do. Everything built on
+  SA inherits the change: BLM calibration, replacement cost, the automatic starting
+  temperature and RUNMODEs that chain SA into iterative improvement. SA's tracked
+  current/best objective and its history now equal the reported `Solution.objective` on
+  connectivity problems. The term equals the MIP's on input the MIP accepts; the MIP still
+  raises on connectivity rows that name unknown planning units and cannot solve negative
+  asymmetric values, both of which SA and the evaluator accept. Problems without connectivity,
+  or with a weight of 0 or below, are unaffected (the delta pays one integer comparison).
+  Greedy heuristics (HEURTYPE 0–7), the Zonation solver, the external Marxan binary and the
+  zone solvers are unchanged.
+
 ## [0.43.0] — 2026-10-17
 
 ### Changed
 - **Reported connectivity now matches the MIP's connectivity term, so reported objectives
   change on every problem with connectivity, for every solver.** `compute_connectivity_penalty`
```

Modify `docs/USER_MANUAL.md` (apply this diff):

```diff
diff --git a/docs/USER_MANUAL.md b/docs/USER_MANUAL.md
index fdebcce..10e9ebe 100644
--- a/docs/USER_MANUAL.md
+++ b/docs/USER_MANUAL.md
@@ -260,13 +260,31 @@ below switches it off):
   A pair listed twice, as `(i, j)` and `(j, i)` (as in a full matrix), counts once: the first
   row wins.
 - **Asymmetric** (`ASYMMETRIC_CONNECTIVITY = 1`): a penalty of `v` for each row whose source
   `id1` is selected without its sink `id2`. Repeated `(id1, id2)` rows add up.
 
-These are prioritizr's connectivity and asymmetric-connectivity penalties. The MIP solver
-optimises them; simulated annealing and the other heuristics report them in the objective but
-do not optimise them.
+These follow prioritizr's `add_connectivity_penalties()` and
+`add_asym_connectivity_penalties()` (each symmetric pair counted once, as prioritizr does by
+using one triangle of the matrix). Edge cases differ: for a pair given twice with different
+values prioritizr keeps the `id1 > id2` row and sums repeated rows, where pymarxan keeps the
+first row; prioritizr treats an asymmetric self-row `(i, i)` as a reward for selecting `i`,
+where pymarxan, like Marxan's `ASYMMETRICCONNECTIVITY`, ignores it; and rows naming unknown
+planning units are skipped here but are an error in prioritizr. For clean input (no
+conflicting reversed rows, no repeated symmetric rows, no self-rows, no unknown ids) the terms
+are the same.
+
+pymarxan's MIP solver, simulated annealing and iterative improvement optimise them. Every other
+solver — the greedy heuristics (HEURTYPE 0–7), the Zonation solver and the external Marxan
+executable (which has no `CONNECTIVITY_WEIGHT`) — only reports them in the objective.
+
+This term is separate from Marxan's own way of using connectivity, which Marxan Connect also
+follows (Beger et al. 2010; Daigle et al. 2020): connection strengths written into the boundary
+file and weighted by `BLM`, penalising a unit selected without the units it is connected to.
+That boundary-based connectivity has always been optimised by simulated annealing. pymarxan
+does not read Marxan's `ASYMMETRICCONNECTIVITY` option; its asymmetric term has the same form
+(source selected without sink, self-rows ignored, repeated rows added) but applies to the
+connectivity table and `CONNECTIVITY_WEIGHT`.
 
 ### Connectivity Metrics
 
 Compute graph-theoretic metrics from the loaded connectivity matrix:
 
```

Modify `docs/plans/2026-10-11-prioritizr-parity-roadmap.md` (apply this diff):

```diff
diff --git a/docs/plans/2026-10-11-prioritizr-parity-roadmap.md b/docs/plans/2026-10-11-prioritizr-parity-roadmap.md
index c5ed9a3..b2dc87c 100644
--- a/docs/plans/2026-10-11-prioritizr-parity-roadmap.md
+++ b/docs/plans/2026-10-11-prioritizr-parity-roadmap.md
@@ -216,9 +216,29 @@ Fold these into a stage when one touches the same code:
 - `simulate_data` / `simulate_species` / `simulate_cost` synthetic project generator, useful for
   tutorials and the JOSS paper.
 - ~~Align `compute_connectivity_penalty` with the MIP connectivity term~~ — done in v0.43
   (symmetric reward-only, first row of a pair wins; asymmetric `v·max(0, x_i − x_j)`; weight
   ≤ 0 ignored). Goldens' `conn_*` objectives re-recorded in place.
-- SA / heuristics do not optimise connectivity (ProblemCache has no connectivity delta); they
-  only report it.
+- ~~SA / heuristics do not optimise connectivity~~ — done in v0.44 for SA and iterative
+  improvement (`ProblemCache` connectivity arrays in the full objective and the per-flip
+  delta; `docs/plans/2026-10-17-sa-connectivity-design.md`). Greedy heuristics still only
+  report it.
+- SA / iterative improvement ignore the PROBMODE 1 risk premium: the `ProblemCache` objective
+  has no PROBMODE 1 term, so SA's tracked objective disagrees with the reported one (cache 44.0
+  vs reported 69.2 on `simple` with PROBMODE 1, PROBABILITYWEIGHTING 2, probabilities
+  `linspace(0.1, 0.6)`, selection {2, 4, 6}).
+- One shared connectivity normaliser for the evaluator, the MIP and `ProblemCache` (today each
+  re-implements the rules). Fold in two MIP gaps: a connectivity row naming an unknown planning
+  unit raises `KeyError` (the evaluator and SA skip it), and negative asymmetric values leave
+  the MIP unbounded / raise `IndexError` (SA and the evaluator accept them).
+- String PU ids split SA's reported and tracked objectives (pre-existing, not connectivity
+  specific): `build_solution` keys `pu_index` on the raw id while the cache uses `int(id)`.
+  Normalise the PU id dtype at load or in `build_solution`.
+- Marxan's `ASYMMETRICCONNECTIVITY 1` in input.dat is ignored: the boundary file is run as a
+  symmetric boundary penalty. pymarxan's asymmetric term applies only to the connectivity table.
+- Marxan's greedy heuristics (`GreedyScore` / `RareScore`, `heuristics.cpp`) include
+  `BLM · ConnectionCost2` in the unit's cost; `HeuristicSolver._score_pu` uses cost only.
+- Zone SA: `ZoneProblemCache` full objective and delta disagree on connectivity self-pairs
+  (verified by probe: a self-row `(i, i, v)` gives full-objective difference 0 but a delta of
+  `+2v` when `i` leaves a zone). Drop self-rows at build or give them the binary rule.
 - Per-PU decision bounds (prioritizr `add_manual_bounded_constraints`) and fractional zone
   shares for `ZoneMIPSolver`.
```

- [ ] **Step 3: Run the tests to verify they pass**

Run: `PYTHONPATH=src:. /opt/micromamba/envs/shiny/bin/pytest tests/benchmarks/bench_sa.py -m bench -k connectivity -q -s`
Expected: passes; prints the ratios (prototype, 1000 PUs, 20k delta calls, best of 5: weight 0 with a table ×0.98–×1.01 of no table; degree-4 grid at weight 1: symmetric ×1.19–×1.23, asymmetric ×1.36–×1.38, print-only). Timing is machine-relative: a failure here on a loaded machine is re-run, not fixed by loosening the bound.

- [ ] **Step 4: Gate**

Run: `~/.local/bin/ruff check src tests && /home/razinka/marxan/.venv/bin/mypy src/pymarxan/ src/pymarxan_shiny/ --ignore-missing-imports && PYTHONPATH=src:. /opt/micromamba/envs/shiny/bin/pytest tests/ -q -m "not bench" -p no:cacheprovider`
Expected: ruff and mypy clean; the full suite passes, no golden regenerated, no existing test changed (prototype: 3583 passed after Task 1, 3605 after Tasks 2 and 3; 101 skipped HiGHS cases, which run in CI; 21 bench deselected after Task 3, 20 before).

- [ ] **Step 5: Commit**

```bash
git add CHANGELOG.md docs/USER_MANUAL.md docs/plans/2026-10-11-prioritizr-parity-roadmap.md tests/benchmarks/bench_sa.py
git commit -m "docs(sa-connectivity): bench, changelog, manual, roadmap

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01AZEiop9qqxVCrMHKqSDYP8"
```
