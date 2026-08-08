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

Immediately after the `rule` check (before `_validate_inputs`):

```python
    if not isinstance(curve_every, int) or curve_every < 1:
        raise ValueError(
            f"curve_every must be an integer >= 1, got {curve_every!r}"
        )
```

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

NOTE: `remove_cell` references `indptr`/`indices`/`data`, which are currently
assigned AFTER `candidate_indices` — keep the existing
`indptr, indices, data = q.indptr, q.indices, q.data` line but move it above
this function definition. In the batch loop, replace the per-removal body:

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
    # Statistical net for the §5 extinction repair: 30 float seeds.
    for seed in range(30):
        p = _random_problem(seed, n_pu=50, n_feat=5, integer=False)
        _assert_equal_results(
            rank_removal(p, rule=rule, warp=1),
            rank_removal(p, rule=rule, warp=1, _force_batch=True),
        )
```

Also EDIT the existing `test_warp_advisory_helper`: append `(1_000_000, 1)` to
its silent-cases tuple list (warp=1 is the fast path now and must not warn).

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
            if cand.size and not np.isfinite(delta[cand]).all():
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
                    rescore(np.array([i], dtype=np.intp))
                    v = float(delta[i])
                    if not np.isfinite(v):
                        raise RuntimeError(_NO_PROGRESS_MSG)
                    heapq.heappush(heap, (v, i))
                    continue
                if s_val != delta[i]:
                    continue  # superseded duplicate; a fresher entry exists
                # Fresh top == true global argmin (design §3), ties by index
                # via tuple order — accept.
                cols, crossed = remove_cell(i)
                if cols.size:
                    holders = np.concatenate(
                        [csc.indices[csc.indptr[j] : csc.indptr[j + 1]] for j in cols]
                    )
                    dirty[holders] = True
                for j in crossed:
                    # FP-residue extinction repair (design §5): holders' true
                    # scores just DROPPED, so cached keys are no longer lower
                    # bounds — rescore and re-push, current phase only.
                    col = csc.indices[csc.indptr[j] : csc.indptr[j + 1]]
                    repair = col[remaining[col] & phase_mask[col]]
                    if repair.size:
                        rescore(repair)
                        if not np.isfinite(delta[repair]).all():
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
@pytest.mark.parametrize("rule", ["caz", "abf"])
def test_fp_residue_extinction_heap_equals_batch(rule: str) -> None:
    # Constructed FP-residue extinction (design §8.2): feature 1's big holders
    # (PUs 1, 2) carry huge costs so they are removed first WITHIN the normal
    # phase; the sequential residue 0.6 - 0.3 - 0.3 == 0.0 <= 0 extinguishes
    # feature 1 while its tiny holder (PU 3) remains — the exact case where
    # cached heap keys stop being lower bounds and the repair must fire.
    assert (0.3 + 0.3 + 1e-17) - 0.3 - 0.3 <= 0.0  # the arithmetic premise
    pu = pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5],
            "cost": [1000.0, 1000.0, 1.0, 1.0, 1.0],
            "status": [0, 0, 0, 0, 0],
        }
    )
    feats = pd.DataFrame(
        {
            "id": [1, 2],
            "name": ["a", "b"],
            "target": [1.0, 1.0],
            "spf": [1.0, 1.0],
        }
    )
    pvf = pd.DataFrame(
        [
            {"species": 1, "pu": 1, "amount": 0.3},
            {"species": 1, "pu": 2, "amount": 0.3},
            {"species": 1, "pu": 3, "amount": 1e-17},
            {"species": 2, "pu": 4, "amount": 5.0},
            {"species": 2, "pu": 5, "amount": 3.0},
        ]
    )
    p = ConservationProblem(pu, feats, pvf)
    heap_res = rank_removal(p, rule=rule, warp=1)
    batch_res = rank_removal(p, rule=rule, warp=1, _force_batch=True)
    _assert_equal_results(heap_res, batch_res)
    # Prove the construction actually crosses before PU 3 leaves: the big
    # holders precede the residue carrier in the removal order.
    order = heap_res.removal_order
    assert max(order.index(1), order.index(2)) < order.index(3)


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
If `test_fp_residue_extinction_heap_equals_batch`'s order assertion fails, the
fixture's costs/amounts need re-tuning so PUs 1–2 genuinely go first — do NOT
weaken the equality assertion.

- [ ] **Step 3: Append the bench** to `tests/benchmarks/bench_zonation.py`:

```python
def test_rank_removal_warp1_heap_budget() -> None:
    p = _grid_problem(300)  # 90_000 cells
    t0 = time.perf_counter()
    res = rank_removal(p, rule="caz", warp=1, curve_every=1000)
    elapsed = time.perf_counter() - t0
    assert len(res.removal_order) == 90_000
    # curve_every=1000 so the bench measures selection, not curve I/O.
    assert elapsed < 60.0, f"warp=1 heap rank_removal took {elapsed:.1f}s"
```

- [ ] **Step 4: Run the bench deliberately; record the time**

Run: `/opt/micromamba/envs/shiny/bin/pytest tests/benchmarks/bench_zonation.py -m bench -v`
Expected: both benches PASS; record the warp=1 elapsed time in the commit
message and report.

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
    Init is O(nnz). ``warp=1`` selects via a CELF-style lazy min-heap (cached
    scores are valid lower bounds because removal only decreases remaining
    totals) and is EXACT — bitwise-identical to the batch selection at
    ``warp=1``, float amounts included — so single-cell removal is feasible at
    raster scale (~1M cells in minutes; pass ``curve_every`` to keep curve
    memory bounded). ``warp>1`` uses batch selection; an advisory warns for
    small ``warp>1`` at raster scale (silence via ``warnings.filterwarnings``).
    ``curve_every=k`` records the initial state, every k-th removal (at batch
    boundaries when ``warp>1``), and always the final state. Landscape-spanning
    features degrade either path toward O(n^2) holder-marking.
```

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
- `zonation.rank_removal` with `warp=1` now selects via a CELF-style lazy
  min-heap: exact single-cell greedy at raster scale, bitwise-identical to the
  previous warp=1 path (float amounts included; FP-residue feature extinction
  is repaired eagerly, phase-scoped). The small-warp advisory no longer fires
  for `warp=1`.
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
