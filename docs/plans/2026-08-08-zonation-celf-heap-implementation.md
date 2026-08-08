# CELF Lazy-Heap warp=1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `warp=1` (exact single-cell greedy) feasible at ~1M cells via a CELF-style lazy min-heap that is bitwise-identical to the existing batch path, plus a `curve_every` parameter and array-backed curve storage.

**Architecture:** All inside `src/pymarxan/zonation/rank_removal.py`. Task 1: `curve_every` + preallocated-array curves (both paths). Task 2: extract shared per-removal bookkeeping into `remove_cell` (pure refactor, existing suite is the net). Task 3: the heap path (auto at warp=1), phase-masked FP-extinction repair, non-finite guards, advisory scope change, heap-vs-batch equivalence suite. Task 4: constructed extinction fixture + pinning tests + warp=1 bench. Task 5: docstring/CHANGELOG + full gate.

**Tech Stack:** numpy, scipy.sparse, heapq (stdlib), pandas, pytest.

**Spec:** `docs/plans/2026-08-08-zonation-celf-heap-design.md` — §3 (invariant), §5 (repair), §6 (curves), §7 (equivalence contract) are normative.

## Global Constraints

- Tests ONLY via `/opt/micromamba/envs/shiny/bin/pytest`; ruff at `~/.local/bin/ruff` (E,F,I,UP; line 99); mypy clean; `from __future__ import annotations`.
- Public API delta: exactly one new kw-only param `curve_every: int = 1`; one private test-only kw `_force_batch: bool = False`. `ZonationResult`/`ZonationSolver` untouched.
- The batch path's behavior must be bit-for-bit unchanged for every existing input (the 55-test scale suite + whole zonation dir are the net after every task).
- Heap path taken iff `warp == 1 and not _force_batch and not _force_full_rescore`.
- Parity anchor 35.0 untouched (no Marxan solver touched); `make check` green at the end.
- Branch: `feat/zonation-celf-heap` (already created; spec committed).
- Known flake `test_solutions_are_different`: rerun before treating as real.

---

### Task 1: `curve_every` + array-backed curves (both paths)

**Files:**
- Modify: `src/pymarxan/zonation/rank_removal.py` (signature, validation, `record_curve`, batch-loop call sites, result assembly)
- Test: `tests/pymarxan/zonation/test_rank_removal_scale.py` (append)

**Interfaces:**
- Produces: `rank_removal(..., curve_every: int = 1)`; module-internal recording rule "initial + `n_removed % curve_every == 0` at batch ends + always final" that Task 3's heap path must reuse verbatim; `record_curve()` closure writing into a preallocated array with `n_curve_rows`/`last_recorded_at` nonlocals.

- [ ] **Step 1: Append failing tests**

```python
# --- CELF phase, Task 1: curve_every + array curves -----------------------
def test_curve_every_validation() -> None:
    p = _random_problem(0)
    for bad in (0, -3, 2.5, "5", None):
        with pytest.raises(ValueError, match="curve_every"):
            rank_removal(p, curve_every=bad)  # type: ignore[arg-type]
    # np.integer must be ACCEPTED (design-review #5): the likeliest caller type
    # for a raster-scale memory knob is numpy-derived (n_pu // 1000 etc.).
    res = rank_removal(p, curve_every=np.int64(7))  # type: ignore[arg-type]
    assert len(res.removal_order) == p.n_planning_units


@pytest.mark.parametrize("rule", ["caz", "abf"])
def test_curve_every_thins_rows_exactly(rule: str) -> None:
    # Thinned rows must be a bitwise row-subset of the curve_every=1 run:
    # initial row, every k-th removal, and always the final state (design §6).
    p = _random_problem(3, n_pu=40)
    full = rank_removal(p, rule=rule, warp=1)
    thin = rank_removal(p, rule=rule, warp=1, curve_every=7)
    assert full.removal_order == thin.removal_order
    fc = full.performance_curves.to_numpy()
    tc = thin.performance_curves.to_numpy()
    k, n = 7, 40
    idxs = [0] + list(range(k, n + 1, k))
    if idxs[-1] != n:
        idxs.append(n)
    np.testing.assert_array_equal(tc, fc[idxs])
    assert list(thin.performance_curves.columns) == list(full.performance_curves.columns)
```

- [ ] **Step 2: Run to verify failures**

Run: `/opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/zonation/test_rank_removal_scale.py -v -k curve_every`
Expected: 3 FAIL (`TypeError: unexpected keyword argument 'curve_every'`).

- [ ] **Step 3: Implement.** In `rank_removal.py`:

Add to the signature (after `smoothing`, before `_force_full_rescore`):

```python
    curve_every: int = 1,
```

Immediately after the `rule` check (before `_validate_inputs`), with
`import operator` added to the stdlib import group:

```python
    try:
        curve_every = operator.index(curve_every)
    except TypeError:
        raise ValueError(
            f"curve_every must be an integer >= 1, got {curve_every!r}"
        ) from None
    if curve_every < 1:
        raise ValueError(
            f"curve_every must be an integer >= 1, got {curve_every!r}"
        )
```

(`operator.index` accepts Python int and `np.integer` exactly; rejects
float/str/None — design-review #5.)

Replace the `removal_order`/`curve_rows` block and `record_curve` (currently the
dict-appending closure) with:

```python
    removal_order: list[int] = []
    curve_cols = ["prop_landscape_remaining", "prop_cost_remaining"] + [
        f"feat_{int(fid)}" for fid in feat_ids
    ]
    curves = np.empty((3 + n_pu // curve_every, 2 + n_feat), dtype=float)
    n_curve_rows = 0
    last_recorded_at = -1  # n_removed value of the most recent recorded row

    def record_curve() -> None:
        nonlocal n_curve_rows, last_recorded_at
        curves[n_curve_rows, 0] = n_remaining / n_pu
        # max(): float-cost sequential drift can leave a tiny negative
        # residual at run end (design §7 site 2); exact for integer costs.
        curves[n_curve_rows, 1] = max(cost_remaining, 0.0) / cost_total
        curves[n_curve_rows, 2:] = np.where(T > 0, Q / T_safe, 1.0)
        n_curve_rows += 1
        last_recorded_at = n_pu - n_remaining
```

In the batch loop, replace the trailing unconditional `record_curve()` with:

```python
        n_removed = n_pu - n_remaining
        if n_removed % curve_every == 0:
            record_curve()
```

After the loop (before `priority_rank`):

```python
    if last_recorded_at != n_pu:
        record_curve()
```

And in the result assembly, replace `pd.DataFrame(curve_rows)` with:

```python
        performance_curves=pd.DataFrame(curves[:n_curve_rows], columns=curve_cols),
```

Value-identity notes (keep true): the three row expressions are exactly the
old dict values (`n_remaining / n_pu`, `max(cost_remaining, 0.0) / cost_total`,
`np.where(T > 0, Q / T_safe, 1.0)`), written as float64 — so every oracle
`assert_frame_equal(check_exact=True)` must keep passing at the default
`curve_every=1` (n_feat==0 works: the `2:` slice assignment takes an empty
array). Preallocation bound: 1 initial + at most `n_pu // curve_every`
multiples + 1 final ≤ `2 + n_pu // curve_every` rows.

- [ ] **Step 4: Run the whole zonation dir**

Run: `/opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/zonation/ -q`
Expected: ALL PASS (75 = 72 existing + 3 new; the entire oracle suite passing IS the curve_every=1 identity proof).

- [ ] **Step 5: Commit**

```bash
git add src/pymarxan/zonation/rank_removal.py tests/pymarxan/zonation/test_rank_removal_scale.py
git commit -m "feat(zonation): curve_every parameter + array-backed performance curves"
```

---

### Task 2: Extract `remove_cell` (pure refactor)

**Files:**
- Modify: `src/pymarxan/zonation/rank_removal.py` (batch loop body)

**Interfaces:**
- Produces: local closure `remove_cell(idx: int) -> tuple[np.ndarray, np.ndarray]` returning `(cols, crossed)`; Task 3's heap path calls it and consumes both elements. `n_remaining`/`cost_remaining` become per-removal updates inside it.

- [ ] **Step 1: Implement the extraction.** Define after `candidate_indices` (before `rescore`):

```python
    def remove_cell(idx: int) -> tuple[np.ndarray, np.ndarray]:
        """Shared per-removal bookkeeping for both selection paths.

        Returns ``(cols, crossed)``: the removed row's feature columns, and the
        subset whose remaining total crossed from >0 to <=0 with this removal
        (FP-residue extinction — float amounts only; consumed by the heap
        path's invariant repair, design §5).
        """
        nonlocal n_remaining, cost_remaining
        removal_order.append(int(pu_ids[idx]))
        remaining[idx] = False
        s, e = indptr[idx], indptr[idx + 1]
        cols = indices[s:e]
        prev_pos = Q[cols] > 0
        Q[cols] -= data[s:e]  # sequential, matching the dense engine
        crossed = cols[prev_pos & (Q[cols] <= 0)]
        n_remaining -= 1
        cost_remaining -= float(c[idx])
        return cols, crossed
```

NOTE: move the existing `indptr, indices, data = q.indptr, q.indices, q.data`
line above this function definition for clarity (closures late-bind, so this is
stylistic, not load-bearing — the assignment already precedes both loops). In the batch loop, replace the per-removal body:

```python
        changed_parts: list[np.ndarray] = []
        for idx in removed:
            cols, _crossed = remove_cell(int(idx))
            changed_parts.append(cols)
```

and DELETE the now-redundant `n_remaining -= removed.size` line (the decrement
moved inside `remove_cell`). Everything else in the loop (progress guard,
holder marking from `changed_parts`, the Task-1 curve recording) stays.

- [ ] **Step 2: Run the full net — behavioral no-op proof**

Run: `/opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/zonation/ tests/pymarxan/solvers/test_zonation_solver.py -q`
Expected: ALL PASS, zero changes to any expectation. Also ruff + mypy on the file.
Then run the existing warp-batch bench once
(`/opt/micromamba/envs/shiny/bin/pytest tests/benchmarks/bench_zonation.py::test_rank_removal_scale_budget -m bench -v`)
and note its time in the commit message — `remove_cell` adds O(nnz_row)
crossed-detection per removal to the batch path (design §2 accepts this; the
bench is the regression net).

- [ ] **Step 3: Commit**

```bash
git add src/pymarxan/zonation/rank_removal.py
git commit -m "refactor(zonation): extract shared per-removal bookkeeping into remove_cell"
```

---

### Task 3: The CELF heap path

**Files:**
- Modify: `src/pymarxan/zonation/rank_removal.py`
- Test: `tests/pymarxan/zonation/test_rank_removal_scale.py` (append + one edit)

**Interfaces:**
- Consumes: `remove_cell` (Task 2), the Task-1 recording rule, existing `rescore`/`dirty`/`candidate_indices`/`csc`.
- Produces: `_force_batch: bool = False` private kwarg (Task 4 tests use it); module constant `_NO_PROGRESS_MSG`.

- [ ] **Step 1: Append failing tests**

```python
# --- CELF phase, Task 3: heap-vs-batch bitwise equivalence ----------------
@pytest.mark.parametrize("rule", ["caz", "abf"])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_heap_equals_batch_integer(rule: str, seed: int) -> None:
    p = _random_problem(seed)
    _assert_equal_results(
        rank_removal(p, rule=rule, warp=1),
        rank_removal(p, rule=rule, warp=1, _force_batch=True),
    )


@pytest.mark.parametrize("rule", ["caz", "abf"])
def test_heap_equals_batch_families(rule: str) -> None:
    # Locks, float amounts+costs, wide features, locks+float — design §8.1.
    fixtures = [
        _random_problem(7, statuses=True),
        _random_problem(11, integer=False),
        _random_problem(5, n_pu=60, n_feat=25),
        _random_problem(21, n_pu=120, n_feat=8, statuses=True, integer=False),
    ]
    for p in fixtures:
        _assert_equal_results(
            rank_removal(p, rule=rule, warp=1),
            rank_removal(p, rule=rule, warp=1, _force_batch=True),
        )


@pytest.mark.parametrize("rule", ["caz", "abf"])
def test_heap_equals_batch_smoothing(rule: str) -> None:
    rng = np.random.default_rng(17)
    p = _random_problem(17, n_pu=18, n_feat=4)
    spec = SmoothingSpec(alpha=0.5, coords=rng.uniform(0, 10, size=(18, 2)))
    _assert_equal_results(
        rank_removal(p, rule=rule, warp=1, smoothing=spec),
        rank_removal(p, rule=rule, warp=1, smoothing=spec, _force_batch=True),
    )


def test_heap_equals_batch_edge_shapes() -> None:
    pu = pd.DataFrame({"id": [1, 2, 3], "cost": [1.0, 2.0, 3.0], "status": [0, 0, 0]})
    feats0 = pd.DataFrame({"id": [], "name": [], "target": [], "spf": []})
    pvf0 = pd.DataFrame(columns=["species", "pu", "amount"])
    p0 = ConservationProblem(pu, feats0, pvf0)
    _assert_equal_results(
        rank_removal(p0, warp=1), rank_removal(p0, warp=1, _force_batch=True)
    )


@pytest.mark.parametrize("rule", ["caz", "abf"])
def test_heap_equals_batch_float_sweep(rule: str) -> None:
    # GENERAL heap-vs-batch coverage. Review #2 measured 0 extinction crossings
    # in 120 such runs — the constructed Task-4 fixtures, NOT this sweep, are
    # the §5 repair's net.
    for seed in range(30):
        p = _random_problem(seed, n_pu=50, n_feat=5, integer=False)
        _assert_equal_results(
            rank_removal(p, rule=rule, warp=1),
            rank_removal(p, rule=rule, warp=1, _force_batch=True),
        )


@pytest.mark.parametrize("rule", ["caz", "abf"])
def test_heap_equals_batch_weights(rule: str) -> None:
    # w != 1 is a distinct rescore input never covered by other families;
    # weight-0 features meet the dead-mask logic (review #6).
    p = _random_problem(13, n_pu=40, n_feat=4)
    w = {1: 3.5, 2: 0.0}
    _assert_equal_results(
        rank_removal(p, rule=rule, warp=1, weights=w),
        rank_removal(p, rule=rule, warp=1, weights=w, _force_batch=True),
    )


def test_heap_equals_batch_all_ties() -> None:
    # Sustained equal-key regime: every delta bitwise-equal each step — pure
    # (score, index) tuple-order selection with maximal stale-duplicate
    # traffic; the tie-break-through-staleness argument's stress case
    # (review #10). Expected order is directly assertable: ascending PU index.
    n = 30
    pu = pd.DataFrame(
        {"id": list(range(1, n + 1)), "cost": [1.0] * n, "status": [0] * n}
    )
    feats = pd.DataFrame({"id": [1], "name": ["a"], "target": [1.0], "spf": [1.0]})
    pvf = pd.DataFrame(
        [{"species": 1, "pu": i, "amount": 2.0} for i in range(1, n + 1)]
    )
    p = ConservationProblem(pu, feats, pvf)
    for rule in ("caz", "abf"):
        heap_res = rank_removal(p, rule=rule, warp=1, use_cost=False)
        _assert_equal_results(
            heap_res,
            rank_removal(p, rule=rule, warp=1, use_cost=False, _force_batch=True),
        )
        assert heap_res.removal_order == list(range(1, n + 1))


def test_heap_equals_batch_small_families() -> None:
    # Stored-zero, duplicate-(pu,species), and featureless-PU fixtures — cheap
    # direct coverage beyond the transitive oracle route (reviews #6/#10).
    pu3 = pd.DataFrame({"id": [1, 2, 3], "cost": [1.0] * 3, "status": [0] * 3})
    feats2 = pd.DataFrame(
        {"id": [1, 2], "name": ["a", "b"], "target": [1.0, 1.0], "spf": [1.0, 1.0]}
    )
    feats1 = pd.DataFrame({"id": [1], "name": ["a"], "target": [1.0], "spf": [1.0]})
    fixtures = [
        ConservationProblem(  # stored zero
            pu3,
            feats2,
            pd.DataFrame(
                [
                    {"species": 1, "pu": 1, "amount": 2.0},
                    {"species": 1, "pu": 2, "amount": 0.0},
                    {"species": 2, "pu": 2, "amount": 3.0},
                    {"species": 2, "pu": 3, "amount": 1.0},
                ]
            ),
        ),
        ConservationProblem(  # duplicate (pu, species) rows
            pu3.iloc[:2].reset_index(drop=True),
            feats1,
            pd.DataFrame(
                [
                    {"species": 1, "pu": 1, "amount": 2.0},
                    {"species": 1, "pu": 1, "amount": 3.0},
                    {"species": 1, "pu": 2, "amount": 4.0},
                ]
            ),
        ),
        ConservationProblem(  # featureless PUs 1 and 3
            pu3, feats1, pd.DataFrame([{"species": 1, "pu": 2, "amount": 3.0}])
        ),
    ]
    for p in fixtures:
        for rule in ("caz", "abf"):
            _assert_equal_results(
                rank_removal(p, rule=rule, warp=1),
                rank_removal(p, rule=rule, warp=1, _force_batch=True),
            )
```

Also two EDITS to existing tests: (a) `test_warp_advisory_helper` — append
`(1_000_000, 1)` to its silent-cases tuple list (warp=1 is the fast path now and
must not warn); (b) `test_curve_every_thins_rows_exactly` — parametrize over
`_force_batch` (add `@pytest.mark.parametrize("force_batch", [False, True])`,
thread `_force_batch=force_batch` into both calls) so the batch path keeps
direct `curve_every>1` coverage after warp=1 re-routes to the heap (review #10).

- [ ] **Step 2: Verify failures**

Run: `/opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/zonation/test_rank_removal_scale.py -v -k "heap_equals or advisory_helper"`
Expected: all heap tests FAIL (`TypeError: ... '_force_batch'`); the advisory
test FAILS on the new `(1_000_000, 1)` case (it currently warns).

- [ ] **Step 3: Implement.** In `rank_removal.py`:

Top-level: `import heapq` (stdlib group). Module constant (replace the inline
message in the batch path's progress guard with it):

```python
_NO_PROGRESS_MSG = (
    "rank_removal made no progress: non-finite scores (extreme "
    "amounts/weights can overflow w/Q); cannot rank this input"
)
```

Advisory condition gains `warp > 1`:

```python
    if warp > 1 and n_pu > _WARP_ADVISORY_MIN_PU and warp < n_pu // 10_000:
```

Signature: add `_force_batch: bool = False` after `_force_full_rescore`.

Selection dispatch — wrap the existing batch `while` loop:

```python
    use_heap = warp == 1 and not _force_batch and not _force_full_rescore

    if use_heap:
        while n_remaining > 0:
            cand = candidate_indices()  # one lock-phase at a time
            phase_mask = np.zeros(n_pu, dtype=bool)
            phase_mask[cand] = True
            rescore(cand[dirty[cand]])
            # NaN-only guard (design §4.3 / review #4): +inf keys are totally
            # ordered and must NOT raise (the batch path completes on all-inf
            # regimes); only NaN corrupts heapq ordering.
            if cand.size and np.isnan(delta[cand]).any():
                raise RuntimeError(_NO_PROGRESS_MSG)
            heap = [(float(delta[i]), int(i)) for i in cand]
            heapq.heapify(heap)
            phase_left = cand.size
            while phase_left > 0:
                if not heap:
                    raise RuntimeError(_NO_PROGRESS_MSG)
                s_val, i = heapq.heappop(heap)
                if not remaining[i]:
                    continue  # lazy deletion: cell already removed
                if dirty[i]:
                    # Buffered dirty rescore (review #1, the CRITICAL perf fix:
                    # single-row rescores cost ~72.5us in scipy slice overhead
                    # vs 1.5us/row vectorized; the per-pop variant measured
                    # ~18x SLOWER than the batch path). Drain the contiguous
                    # removed/dirty prefix of the heap, then rescore the
                    # deduplicated dirty buffer in ONE vectorized call.
                    buf = [i]
                    while heap:
                        s2, i2 = heap[0]
                        if not remaining[i2]:
                            heapq.heappop(heap)
                            continue
                        if dirty[i2]:
                            heapq.heappop(heap)
                            buf.append(i2)
                            continue
                        break
                    rows = np.unique(np.asarray(buf, dtype=np.intp))
                    rescore(rows)
                    if np.isnan(delta[rows]).any():
                        raise RuntimeError(_NO_PROGRESS_MSG)
                    for h in rows:
                        heapq.heappush(heap, (float(delta[h]), int(h)))
                    continue
                if s_val != delta[i]:
                    # Superseded duplicate. Safe ONLY because delta[] is written
                    # solely by rescore(), and every heap-path rescore is
                    # followed by a push/heapify of the rescored cells — a
                    # mismatched key always has a fresher sibling in the heap.
                    # (Rescoring without pushing would silently break argmin.)
                    continue
                # Fresh top == true global argmin (design §3), ties by index
                # via tuple order — accept.
                assert not dirty[i]
                cols, crossed = remove_cell(i)
                if cols.size:
                    holders = np.concatenate(
                        [csc.indices[csc.indptr[j] : csc.indptr[j + 1]] for j in cols]
                    )
                    dirty[holders] = True
                for j in crossed:
                    # FP-residue extinction repair (design §5): holders' true
                    # scores just DROPPED, so cached keys are no longer lower
                    # bounds — rescore and re-push, current phase only
                    # (phase_mask is load-bearing: an out-of-phase push would
                    # let a locked-in cell be selected early).
                    col = csc.indices[csc.indptr[j] : csc.indptr[j + 1]]
                    repair = col[remaining[col] & phase_mask[col]]
                    if repair.size:
                        rescore(repair)
                        if np.isnan(delta[repair]).any():
                            raise RuntimeError(_NO_PROGRESS_MSG)
                        for h in repair:
                            heapq.heappush(heap, (float(delta[h]), int(h)))
                phase_left -= 1
                n_removed = n_pu - n_remaining
                if n_removed % curve_every == 0:
                    record_curve()
    else:
        # ... existing batch while-loop, unchanged ...
```

The Task-1 final-record check (`if last_recorded_at != n_pu: record_curve()`)
and the `priority_rank` assembly stay at function level BELOW this entire
if/else, shared by both paths — do not indent them into the `else` branch (the
heap path depends on the final-record for the always-final row at
`curve_every>1`).

Also rewrite `_warn_if_small_warp`'s docstring (its "warp=1 selection alone is
O(n^2)" opener becomes false):

```python
    """Advise (warn-and-proceed, S3b precedent) when warp is too small to scale.

    warp=1 routes to the exact lazy-heap path and is fast; the advisory covers
    2 <= warp < n_pu // 10_000 at raster scale, where batch selection pays an
    O(candidates) partition per small batch. Large warp (10-100) is documented
    Zonation practice as a computation-time vs solution-refinement trade-off;
    ``warp ~ n_pu/1000`` is pymarxan performance advice for batch mode.
    Silence with ``warnings.filterwarnings`` when a small warp is deliberate.
    """
```

Notes (keep true):
- Every heap entry is `(float, int)` — homogeneous tuples, and NaN never enters
  (guarded at init, dirty-repush, and repair-push), so heapq ordering is sound.
- The `s_val != delta[i]` skip is float equality on purpose: `delta[i]` is only
  ever written by `rescore`, and every rescore that touches a heap-relevant cell
  pushes a fresh matching entry, so a mismatched key always has a fresher
  sibling in the heap.
- `phase_mask` on the repair is load-bearing (design §5): pushing an
  out-of-phase holder would let a locked-in cell be selected early. Out-of-phase
  holders are already dirty via the standard marking and get rescored at their
  phase's init.
- The empty-heap guard is belt-and-braces (an invariant violation, not an
  expected path) — same `RuntimeError` contract as the batch progress guard.

- [ ] **Step 4: Run everything**

Run: `/opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/zonation/ tests/pymarxan/solvers/test_zonation_solver.py -q`
Expected: ALL PASS — the new heap tests, the advisory edit, AND every
pre-existing warp=1 fixture (which now runs the heap against the dense oracle).
`test_dirty_set_equals_full_rescore` at warp=1 now compares heap vs
batch-full-rescore — must stay green. If a heap-vs-batch test fails: print both
removal orders, find the first divergence, and check (in order) the freshness
test, the repair's phase mask, and duplicate-skip logic. The batch path is
ground truth — never adjust it.
Then ruff + mypy on both files.

- [ ] **Step 5: Commit**

```bash
git add src/pymarxan/zonation/rank_removal.py tests/pymarxan/zonation/test_rank_removal_scale.py
git commit -m "feat(zonation): CELF lazy-heap exact selection for warp=1"
```

---

### Task 4: Extinction fixture, pinning tests, warp=1 bench

**Files:**
- Test: `tests/pymarxan/zonation/test_rank_removal_scale.py` (append)
- Modify: `tests/benchmarks/bench_zonation.py` (append)

**Interfaces:**
- Consumes: `_force_batch` (Task 3), `_grid_problem` (existing bench helper), `_pvf_problem`-style manual fixtures.

- [ ] **Step 1: Append the tests**

```python
# --- CELF phase, Task 4: extinction construction + pinning ----------------
def _extinction_problem(statuses: list[int]) -> ConservationProblem:
    """Detector-verified FP-residue extinction fixture (review #2, verified).

    Levers (both are load-bearing — the review's original 'huge costs on big
    holders' reasoning was backwards, a near-zero-amount holder at cost 1 is
    the global argmin and leaves FIRST):
    - the residue carrier PU3 gets cost 1e-15 so its score (1.67e-2) stays
      ABOVE the big holders' (5e-4, 1e-3) until the crossing;
    - detector PU6 (sole holder of f3, score 1/200 = 5e-3) sits strictly
      inside the carrier's stale-key/post-crossing-true-score gap (0, 1.67e-2),
      so a missing, broken, or phase-inverted repair provably FLIPS the order
      to [1,2,6,3,5,4] — this test VERIFIES the repair, not merely exercises it
      (execution-verified against a deliberately repair-less heap).
    """
    pu = pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5, 6],
            "cost": [1000.0, 1000.0, 1e-15, 1.0, 1.0, 200.0],
            "status": statuses,
        }
    )
    feats = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "name": ["a", "b", "c"],
            "target": [1.0] * 3,
            "spf": [1.0] * 3,
        }
    )
    pvf = pd.DataFrame(
        [
            {"species": 1, "pu": 1, "amount": 0.3},
            {"species": 1, "pu": 2, "amount": 0.3},
            {"species": 1, "pu": 3, "amount": 1e-17},
            {"species": 2, "pu": 4, "amount": 5.0},
            {"species": 2, "pu": 5, "amount": 3.0},
            {"species": 3, "pu": 6, "amount": 1.0},
        ]
    )
    return ConservationProblem(pu, feats, pvf)


@pytest.mark.parametrize("rule", ["caz", "abf"])
def test_fp_residue_extinction_heap_equals_batch(rule: str) -> None:
    # Crossing INSIDE the normal phase -> the repair-push path itself runs.
    assert (0.3 + 0.3 + 1e-17) - 0.3 - 0.3 <= 0.0  # the arithmetic premise
    p = _extinction_problem([0, 0, 0, 0, 0, 0])
    heap_res = rank_removal(p, rule=rule, warp=1)
    _assert_equal_results(
        heap_res, rank_removal(p, rule=rule, warp=1, _force_batch=True)
    )
    order = heap_res.removal_order
    # Big holders precede the carrier (crossing happens while PU3 remains),
    # and the detector-sensitive order is pinned exactly.
    assert max(order.index(1), order.index(2)) < order.index(3)
    assert order == [1, 2, 3, 6, 5, 4]


@pytest.mark.parametrize("rule", ["caz", "abf"])
def test_fp_residue_extinction_cross_phase(rule: str) -> None:
    # Big holders locked OUT (review #3): the crossing happens in the
    # locked-out phase while the carrier sits in the normal phase — exercising
    # the repair-push-skipped + dirty-carry + phase-init-rescore path that
    # design §5 declares safe.
    p = _extinction_problem([3, 3, 0, 0, 0, 0])
    _assert_equal_results(
        rank_removal(p, rule=rule, warp=1),
        rank_removal(p, rule=rule, warp=1, _force_batch=True),
    )


def test_subnormal_raises_on_both_paths() -> None:
    # The NaN-poisoned no-progress input must raise on the heap path (at init
    # or repush) AND still on the forced batch path.
    pu = pd.DataFrame(
        {"id": [1, 2, 3, 4], "cost": [1.0] * 4, "status": [0] * 4}
    )
    feats = pd.DataFrame(
        {"id": [1, 2], "name": ["a", "b"], "target": [1.0, 1.0], "spf": [1.0, 1.0]}
    )
    pvf = pd.DataFrame(
        [
            {"species": 1, "pu": 1, "amount": 5e-324},
            {"species": 2, "pu": 2, "amount": 1.0},
            {"species": 2, "pu": 3, "amount": 2.0},
            {"species": 2, "pu": 4, "amount": 3.0},
        ]
    )
    p = ConservationProblem(pu, feats, pvf)
    for kwargs in ({}, {"_force_batch": True}):
        with pytest.raises(RuntimeError, match="no progress"):
            rank_removal(p, warp=1, **kwargs)


def test_curve_every_with_warp_batches() -> None:
    # warp>1: records land on batch boundaries whose cumulative removal count
    # is a multiple of curve_every; always the final row (design §6).
    p = _random_problem(9, n_pu=30)
    full = rank_removal(p, warp=5)               # rows at n_removed 0,5,...,30
    thin = rank_removal(p, warp=5, curve_every=10)
    fc = full.performance_curves.to_numpy()
    tc = thin.performance_curves.to_numpy()
    np.testing.assert_array_equal(tc, fc[[0, 2, 4, 6]])
```

- [ ] **Step 2: Run them**

Run: `/opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/zonation/test_rank_removal_scale.py -v -k "fp_residue or subnormal_raises_on_both or curve_every_with_warp"`
Expected: ALL PASS (Task 3 already implemented the machinery; these pin it).
The fixture's expected order `[1, 2, 3, 6, 5, 4]` was execution-verified during
design review; if it fails, suspect the ENGINE (repair or phase mask), not the
fixture — debug against `_force_batch=True` before touching any assertion.

- [ ] **Step 3: Append the bench** to `tests/benchmarks/bench_zonation.py`:

```python
def test_rank_removal_warp1_heap_budget() -> None:
    p = _grid_problem(300)  # 90_000 cells
    t0 = time.perf_counter()
    res = rank_removal(p, rule="caz", warp=1, curve_every=1000)
    elapsed = time.perf_counter() - t0
    assert len(res.removal_order) == 90_000
    # curve_every=1000 so the bench measures selection, not curve I/O.
    # Reference points on this machine (design review): batch warp=1 on this
    # geometry = 120.7s; the naive per-pop heap DNF'd — the buffered-pop loop
    # is what makes this budget possible.
    assert elapsed < 60.0, f"warp=1 heap rank_removal took {elapsed:.1f}s"
```

- [ ] **Step 4: MEASURE first, then run the bench**

Run: `/opt/micromamba/envs/shiny/bin/pytest tests/benchmarks/bench_zonation.py -m bench -v`
Expected: all three benches PASS. **If the warp=1 bench exceeds 60 s, STOP and
report BLOCKED with the measured time and the pop/rescore counts** (design §8.7:
the budget is a claim to verify, not to force) — do not raise the budget or ship
perf wording without the controller's decision. Record the measured time in the
commit message and report; Task 5's docstring perf wording depends on it.

- [ ] **Step 5: Commit**

```bash
git add tests/pymarxan/zonation/test_rank_removal_scale.py tests/benchmarks/bench_zonation.py
git commit -m "test(zonation): FP-extinction fixture, dual-path NaN guard, warp=1 bench"
```

---

### Task 5: Docstring, CHANGELOG, full gate

**Files:**
- Modify: `src/pymarxan/zonation/rank_removal.py` (docstring only), `CHANGELOG.md`

- [ ] **Step 1: Docstring.** In the `rank_removal` docstring, replace the sentence beginning "Init is O(nnz); million-cell rasters rank in minutes at raster-appropriate ``warp``" through "...silence via ``warnings.filterwarnings``)." with:

```
    Init is O(nnz). ``warp=1`` selects via lazy-greedy (accelerated greedy,
    Minoux 1978; popularized as CELF, Leskovec et al. 2007) mirrored to
    minimization: removal only increases remaining cells' scores, so cached
    keys are lower bounds on a min-heap and a popped fresh top is the true
    argmin. The warp=1 trajectory is bitwise-identical to batch selection for
    every input whose scores never evaluate to NaN (float amounts and +inf
    regimes included; NaN-producing runs raise ``RuntimeError`` on the heap
    path where batch selection may return a NaN-ordered tail) — exact with
    respect to the greedy removal sequence; the ranking itself remains a
    heuristic prioritization, not a provably optimal reserve. Single-cell
    removal is thereby feasible at raster scale (bench: 90k cells well under a
    minute; pass ``curve_every`` to keep curve memory bounded). ``warp>1`` uses
    batch selection; an advisory warns for small ``warp>1`` at raster scale
    (silence via ``warnings.filterwarnings``). ``curve_every=k`` records the
    initial state, every k-th removal (at batch boundaries when ``warp>1``),
    and always the final state. Landscape-spanning features degrade either path
    toward O(n^2) holder-marking.
```

(If Task 4's measured bench time materially beats/misses "well under a minute",
adjust that clause to the measurement — never ship an unmeasured number.)

Also extend the private-kwarg sentence: ``_force_full_rescore`` and
``_force_batch`` are test-only: the first disables the dirty-set shortcut (and
forces batch selection), the second forces batch selection at ``warp=1``.

- [ ] **Step 2: CHANGELOG** under `[Unreleased]`:

```markdown
### Added
- `zonation.rank_removal(curve_every=...)`: record performance-curve rows every
  k-th removal instead of every step — the memory knob for `warp=1` at raster
  scale; curve storage is now a preallocated array either way.

### Changed
- `zonation.rank_removal` with `warp=1` now selects via a lazy-greedy min-heap
  (Minoux 1978; cf. CELF): exact single-cell greedy at raster scale,
  bitwise-identical to the previous warp=1 path whenever scores stay NaN-free
  (float amounts and +inf regimes included; NaN-producing runs now fail fast
  with `RuntimeError`; FP-residue feature extinction is repaired eagerly,
  phase-scoped). The small-warp advisory no longer fires for `warp=1`.
```

- [ ] **Step 3: Full gate**

Run: `PATH="/opt/micromamba/envs/shiny/bin:$HOME/.local/bin:$PWD/.venv/bin:$PATH" make check`
Expected: green (~1895+ tests). Known SA flake: rerun alone if it is the only failure.

- [ ] **Step 4: Commit**

```bash
git add src/pymarxan/zonation/rank_removal.py CHANGELOG.md
git commit -m "docs(zonation): CELF warp=1 contract + curve_every changelog"
```

---

## Self-review (done at write time)

- Spec coverage: §3/§4→T3, §5→T3 (+T4 fixture), §6→T1 (+T4 warp>1 pinning), §7→T3 tests, §8.1→T3, §8.2→T4, §8.3→T3 Step 4 (existing suite), §8.4→T1+T4, §8.5→T3 (advisory) + T4 (subnormal), §8.6→T4 bench, §8.7→T5, §10→T5.
- Type consistency: `remove_cell(idx:int)->(cols,crossed)` defined T2, consumed T3; `_force_batch` defined T3, consumed T4; `curve_every` defined T1, used T3/T4; `_NO_PROGRESS_MSG` defined T3 and reused by the batch guard.
- No placeholders; all code complete.
- Ordering note: T1's thinning test runs warp=1 pre-heap (batch) and post-heap (heap) — valid both ways by design.
