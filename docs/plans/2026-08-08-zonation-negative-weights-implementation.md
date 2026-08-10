# Negative-Weight Features Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Support negative (opportunity-cost) feature weights in `rank_removal`: CAZ gains a meaningful negative semantic, warp=1 auto-routes to batch selection, and `w >= 0` results stay bitwise identical.

**Architecture:** Three edits to one file. Task 1 generalizes the CAZ reduction behind a `has_neg_w` flag that validation still keeps False (a deliberately unreachable branch; the existing suite proves the refactor is a no-op). Task 2 drops the validation raise, routes warp=1 away from the heap with a warning, and brings the branch alive with hand-computed semantic tests. Task 3 documents and gates.

**Tech Stack:** numpy, scipy.sparse, pytest.

**Spec:** `docs/plans/2026-08-08-zonation-negative-weights-design.md` — §3 (CAZ semantics, normative), §4 (implementation), §5 (invariants), §6 (tests).

## Global Constraints

- Tests ONLY via `/opt/micromamba/envs/shiny/bin/pytest`; ruff `~/.local/bin/ruff` (E,F,I,UP; line 99); mypy `.venv/bin/mypy src/pymarxan/zonation/rank_removal.py --ignore-missing-imports`; `from __future__ import annotations`.
- **`w >= 0` behavior must stay bitwise identical.** The existing suite — including the test-local dense oracle `_dense_rank_removal`, which keeps the OLD CAZ formula — is the proof and must NOT be modified.
- Amounts stay `>= 0` and finite; only *weights* may now be negative.
- ABF is already correct for either sign — do not change its reduction.
- Branch `feat/zonation-negative-weights` (created; spec committed). Parity anchor untouched; `make check` green at the end.
- Known SA flake `test_solutions_are_different`: rerun alone if it is the only failure.
- Ruff the test files too before every commit (an E402 slipped two reviews in the CELF phase).

---

### Task 1: Generalize the CAZ reduction (behavioral no-op)

The new branch is **unreachable** until Task 2 drops the validation raise — deliberate, so the risky hot-path edit lands with the existing suite as its only net.

**Files:**
- Modify: `src/pymarxan/zonation/rank_removal.py` (`rescore`, `:310-333`; `w` fill block, `:231-235`)

**Interfaces:**
- Produces: closure variable `has_neg_w: bool` (read by `rescore` and, in Task 2, by the heap-routing line); `rescore`'s reduction split by `rule` and `has_neg_w`.

- [ ] **Step 1: Add the flag.** After the `w` fill block (immediately after `rank_removal.py:235`, before the `use_cost` block):

```python
    has_neg_w = bool((w < 0).any())
```

- [ ] **Step 2: Replace the reduction.** In `rescore`, replace the loop body (`rank_removal.py:327-332`) with:

```python
        for s in range(0, rows.size, _RESCORE_CHUNK):
            chunk = rows[s : s + _RESCORE_CHUNK]
            qd = q[chunk].toarray()
            r = qd * fac
            if rule == "abf":
                r[:, dead] = 0.0
                out = r.sum(axis=1)
            elif has_neg_w:
                # CAZ with negative weights: the max must range over the
                # features this cell HOLDS and that are still extant.
                # Structural zeros for absent features would otherwise
                # outrank every negative term, silently ignoring
                # opportunity-cost weights (design spec §3). Held-ness comes
                # from the AMOUNT buffer, not from `r == 0`: a zero-weighted
                # held feature also yields 0.0, and with negatives present
                # including vs excluding that 0 changes the answer.
                r[qd == 0] = -np.inf
                r[:, dead] = -np.inf
                out = r.max(axis=1)
                # A row with nothing held-and-extant floors to 0.0, matching
                # the shipped formula. `== -inf` not `~isfinite`: +inf is a
                # legal score in the subnormal-Q regime the RuntimeError
                # guard owns, and must not be rewritten to 0.0.
                out[out == -np.inf] = 0.0
            else:
                r[:, dead] = 0.0
                out = r.max(axis=1)
            delta[chunk] = out / c[chunk]
```

Naming `qd` keeps two chunk buffers alive at once (~26 MB each at
`_RESCORE_CHUNK=32768`, `n_feat=100`) — accepted per spec §4.2. Values are
unchanged: `qd * fac` is the same expression as the inlined form.

- [ ] **Step 3: Prove the no-op.** Run the full zonation + solver + panel suite:

Run: `/opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/zonation/ tests/pymarxan/solvers/test_zonation_solver.py tests/pymarxan_shiny/test_zonation_panel.py -q`
Expected: ALL PASS with zero expectation changes (~130 tests). Every CAZ test still runs the `else` branch, since `has_neg_w` is False for all of them; the dense oracle's continued agreement is the bitwise proof. Then ruff + mypy on the file.

- [ ] **Step 4: Commit**

```bash
git add src/pymarxan/zonation/rank_removal.py
git commit -m "refactor(zonation): generalize CAZ reduction behind has_neg_w flag"
```

---

### Task 2: Enable negative weights (validation, routing, semantics)

**Files:**
- Modify: `src/pymarxan/zonation/rank_removal.py` (`_validate_inputs` `:78-87`; new helper; `:247`; `:335`)
- Test: `tests/pymarxan/zonation/test_rank_removal_scale.py` (append)

**Interfaces:**
- Consumes: `has_neg_w` (Task 1).
- Produces: `_warn_if_negative_weights_at_warp1(warp: int, has_neg_w: bool) -> None`; `rank_removal` accepting negative weight values.

- [ ] **Step 1: Write the failing tests.** Append to `tests/pymarxan/zonation/test_rank_removal_scale.py`:

```python
# --- Negative (opportunity-cost) feature weights -------------------------
def _cost_weight_problem(costs: list[float] | None = None) -> ConservationProblem:
    """PU1 benefit-only, PU2 empty, PU3 weak cost, PU4 strong cost.

    Hand-computable and — critically — DISCRIMINATING: under the shipped
    max-with-structural-zeros formula PU2/PU3/PU4 all score 0.0 and break the
    tie by index, giving [2, 3, 4, 1]; under max-over-held-and-extant they
    score 0.0 / -1/11 / -10/11, giving [4, 3, 2, 1]. A fixture where both
    formulas agree would prove nothing.
    """
    n = 4
    pu = pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "cost": [1.0] * n if costs is None else costs,
            "status": [0] * n,
        }
    )
    feats = pd.DataFrame(
        {"id": [1, 2], "name": ["benefit", "cost"], "target": [1.0, 1.0],
         "spf": [1.0, 1.0]}
    )
    pvf = pd.DataFrame(
        [
            {"species": 1, "pu": 1, "amount": 10.0},
            {"species": 2, "pu": 3, "amount": 1.0},
            {"species": 2, "pu": 4, "amount": 10.0},
        ]
    )
    return ConservationProblem(pu, feats, pvf)


NEG_W = {1: 1.0, 2: -1.0}


def test_caz_negative_weight_removes_cost_cells_first() -> None:
    # Hand-computed, Q1=10 Q2=11:
    #   PU1 max{10/10}=1.0 | PU2 empty->0.0 | PU3 max{-1/11}=-0.0909
    #   PU4 max{-10/11}=-0.9091  -> remove PU4; Q2=1
    #   PU3 max{-1/1}=-1.0       -> remove PU3; Q2=0 (extinct)
    #   PU2 0.0 < PU1 1.0        -> remove PU2, then PU1.
    res = rank_removal(_cost_weight_problem(), rule="caz", weights=NEG_W, warp=1)
    assert res.removal_order == [4, 3, 2, 1]
    # The formerly-inexpressible case: a cell holding ONLY a negatively
    # weighted feature ranks below a cell holding nothing at all.
    order = res.removal_order
    assert order.index(4) < order.index(2)


def test_abf_negative_weight_mixed_cell() -> None:
    # Separate fixture: PU3 holds BOTH, so its net score flips sign as Q2
    # shrinks (0.0 -> -0.6667) — a concrete demonstration of the
    # non-monotonicity that makes the warp=1 lazy heap unusable.
    #   Q1=Q2=15: PU1 10/15=0.6667 | PU2 -10/15=-0.6667 | PU3 5/15-5/15=0.0
    #   | PU4 0.0 -> remove PU2; Q2=5
    #   PU3 5/15-5/5=-0.6667      -> remove PU3; Q1=10, Q2=0
    #   PU4 0.0 < PU1 10/10=1.0   -> remove PU4, then PU1.
    pu = pd.DataFrame(
        {"id": [1, 2, 3, 4], "cost": [1.0] * 4, "status": [0] * 4}
    )
    feats = pd.DataFrame(
        {"id": [1, 2], "name": ["benefit", "cost"], "target": [1.0, 1.0],
         "spf": [1.0, 1.0]}
    )
    pvf = pd.DataFrame(
        [
            {"species": 1, "pu": 1, "amount": 10.0},
            {"species": 2, "pu": 2, "amount": 10.0},
            {"species": 1, "pu": 3, "amount": 5.0},
            {"species": 2, "pu": 3, "amount": 5.0},
        ]
    )
    p = ConservationProblem(pu, feats, pvf)
    res = rank_removal(p, rule="abf", weights=NEG_W, warp=1)
    assert res.removal_order == [2, 3, 4, 1]


def test_caz_zero_weight_held_feature_participates() -> None:
    # A zero-weighted feature is HELD, so it enters the max at 0.0: a cell
    # holding {w=0, w<0} scores 0.0, not the negative term (design §3 — the
    # reason the mask reads the amount buffer, not `r == 0`). PU3 therefore
    # ties the empty PU2 at 0.0 and loses the tie on index, while PU4 (cost
    # only) still goes first.
    pu = pd.DataFrame({"id": [1, 2, 3, 4], "cost": [1.0] * 4, "status": [0] * 4})
    feats = pd.DataFrame(
        {"id": [1, 2, 3], "name": ["benefit", "cost", "zero"],
         "target": [1.0] * 3, "spf": [1.0] * 3}
    )
    pvf = pd.DataFrame(
        [
            {"species": 1, "pu": 1, "amount": 10.0},
            {"species": 2, "pu": 3, "amount": 5.0},
            {"species": 3, "pu": 3, "amount": 7.0},
            {"species": 2, "pu": 4, "amount": 5.0},
        ]
    )
    p = ConservationProblem(pu, feats, pvf)
    res = rank_removal(p, rule="caz", weights={1: 1.0, 2: -1.0, 3: 0.0}, warp=1)
    assert res.removal_order[0] == 4          # only-cost cell first
    assert res.removal_order.index(2) < res.removal_order.index(3)  # 0.0 tie


def test_negative_weight_routes_off_the_heap(monkeypatch: pytest.MonkeyPatch) -> None:
    # heapq.heapify is called once per lock-phase by the heap path and never
    # by the batch path — a loud probe rather than counting pops.
    import heapq

    def _boom(*_a: object, **_k: object) -> None:
        raise AssertionError("heap path taken with negative weights")

    monkeypatch.setattr(heapq, "heapify", _boom)
    p = _cost_weight_problem()
    with pytest.warns(UserWarning, match="non-monotone"):
        res = rank_removal(p, rule="caz", weights=NEG_W, warp=1)
    assert res.removal_order == [4, 3, 2, 1]
    # The probe has teeth: all-positive weights at warp=1 DO reach heapify.
    with pytest.raises(AssertionError, match="heap path taken"):
        rank_removal(p, rule="caz", weights={1: 1.0, 2: 1.0}, warp=1)


def test_no_heap_warning_without_negative_weights() -> None:
    import warnings as _warnings

    p = _cost_weight_problem()
    with _warnings.catch_warnings():
        _warnings.simplefilter("error")
        rank_removal(p, rule="caz", weights={1: 1.0, 2: 2.0}, warp=1)


@pytest.mark.parametrize("rule", ["caz", "abf"])
@pytest.mark.parametrize("warp", [1, 3])
def test_negative_weight_self_consistency(rule: str, warp: int) -> None:
    # Only monotonicity was lost; the dirty-set shortcut and the batch
    # selection are still valid, so these must agree exactly at a FIXED warp.
    p = _cost_weight_problem()
    base = rank_removal(p, rule=rule, weights=NEG_W, warp=warp)
    _assert_equal_results(
        base, rank_removal(p, rule=rule, weights=NEG_W, warp=warp, _force_batch=True)
    )
    _assert_equal_results(
        base,
        rank_removal(
            p, rule=rule, weights=NEG_W, warp=warp, _force_full_rescore=True
        ),
    )


def test_negative_weight_cost_division() -> None:
    # Dividing a NEGATIVE score by a larger positive cost makes it LESS
    # negative, so the pricier cost-carrying cell is removed later even
    # though it carries identical amounts. Pinned as current behavior;
    # flagged in the spec's open questions for the science lens.
    pu = pd.DataFrame(
        {"id": [1, 2, 3], "cost": [1.0, 1.0, 2.0], "status": [0, 0, 0]}
    )
    feats = pd.DataFrame(
        {"id": [1, 2], "name": ["benefit", "cost"], "target": [1.0, 1.0],
         "spf": [1.0, 1.0]}
    )
    pvf = pd.DataFrame(
        [
            {"species": 1, "pu": 1, "amount": 10.0},
            {"species": 2, "pu": 2, "amount": 10.0},
            {"species": 2, "pu": 3, "amount": 10.0},
        ]
    )
    p = ConservationProblem(pu, feats, pvf)
    res = rank_removal(p, rule="caz", weights=NEG_W, warp=1, use_cost=True)
    assert res.removal_order == [2, 3, 1]


def test_negative_weight_still_rejects_nan_and_negative_amounts() -> None:
    p = _cost_weight_problem()
    with pytest.raises(ValueError, match="weights must be finite"):
        rank_removal(p, weights={1: float("nan")})
    pu = pd.DataFrame({"id": [1, 2], "cost": [1.0, 1.0], "status": [0, 0]})
    feats = pd.DataFrame({"id": [1], "name": ["a"], "target": [1.0], "spf": [1.0]})
    pvf = pd.DataFrame(
        [{"species": 1, "pu": 1, "amount": 2.0},
         {"species": 1, "pu": 2, "amount": -1.0}]
    )
    with pytest.raises(ValueError, match="amounts must be >= 0"):
        rank_removal(ConservationProblem(pu, feats, pvf), weights={1: -1.0})
```

- [ ] **Step 2: Run to verify failures**

Run: `/opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/zonation/test_rank_removal_scale.py -v -k "negative_weight or caz_negative or abf_negative or zero_weight_held or heap_warning"`
Expected: FAIL — most with `ValueError: feature weights must be >= 0 ...`.

- [ ] **Step 3: Drop the validation raise.** In `_validate_inputs`, delete the `if (wv < 0).any(): raise ...` block (`rank_removal.py:81-87`), keeping the finiteness check. Update its docstring sentence to:

```
    Negative weights are supported (Zonation v3+ opportunity-cost features;
    Moilanen et al. 2011, doi:10.1890/10-1865.1); amounts must stay
    nonnegative — Zonation expresses negative features through weights, not
    occurrence levels.
```

- [ ] **Step 4: Add the routing warning helper** beside `_warn_if_small_warp`:

```python
def _warn_if_negative_weights_at_warp1(warp: int, has_neg_w: bool) -> None:
    """Advise that negative weights disable the exact warp=1 lazy heap.

    A term ``w_j q_ij / Q_j`` with ``w_j < 0`` grows MORE negative as ``Q_j``
    shrinks, so scores are no longer nondecreasing under removal, cached heap
    keys stop being lower bounds, and the lazy-greedy exactness argument
    fails. Batch selection has no such dependency (its dirty set tracks
    changed inputs, not monotone scores), so it is used instead.
    """
    if warp == 1 and has_neg_w:
        warnings.warn(
            "negative feature weights make rank_removal scores non-monotone, "
            "so the exact warp=1 lazy heap is unavailable; using batch "
            "selection instead (identical results, slower). Silence via "
            "warnings.filterwarnings if deliberate.",
            stacklevel=3,
        )
```

- [ ] **Step 5: Wire the call and the routing.** After `_warn_if_small_warp(n_pu, warp)` (`rank_removal.py:247`) add:

```python
    _warn_if_negative_weights_at_warp1(warp, has_neg_w)
```

and extend the routing line (`rank_removal.py:335`):

```python
    use_heap = (
        warp == 1
        and not _force_batch
        and not _force_full_rescore
        and not has_neg_w
    )
```

- [ ] **Step 6: Run everything**

Run: `/opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/zonation/ tests/pymarxan/solvers/test_zonation_solver.py -q`
Expected: ALL PASS — the new tests plus every pre-existing one (unchanged). If a hand-computed order assert fails, recompute it from the fixture BEFORE touching the engine: the arithmetic is in the test comments and the engine is more likely right than the comment. Then ruff (both files) + mypy.

- [ ] **Step 7: Commit**

```bash
git add src/pymarxan/zonation/rank_removal.py tests/pymarxan/zonation/test_rank_removal_scale.py
git commit -m "feat(zonation): negative (opportunity-cost) feature weights"
```

---

### Task 3: Docstring, CHANGELOG, gate

**Files:**
- Modify: `src/pymarxan/zonation/rank_removal.py` (docstring only), `CHANGELOG.md`

- [ ] **Step 1: Docstring.** In `rank_removal`'s docstring, after the sentence describing CAZ/ABF, insert:

```
    Feature weights may be negative (Zonation v3+ opportunity-cost features;
    Moilanen et al. 2011, doi:10.1890/10-1865.1) so that cells carrying them
    are preferentially removed. With any negative weight, CAZ maximizes over
    the features a cell HOLDS and that are still extant (a cell holding
    nothing extant scores 0.0); with all-nonnegative weights this is exactly
    the previous formula, bit for bit. Negative weights also make scores
    non-monotone under removal, so ``warp=1`` uses batch selection rather
    than the exact lazy heap — identical results, without the heap's speed
    (a filterable ``UserWarning`` says so).
```

- [ ] **Step 2: CHANGELOG** under `[Unreleased]`:

```markdown
### Added
- `zonation.rank_removal` accepts negative feature weights — Zonation v3+
  opportunity-cost / threat layers (Moilanen et al. 2011,
  doi:10.1890/10-1865.1) — so cells carrying them rank for removal first.
  `ZonationSolver` inherits this through its existing `weights` parameter.

### Changed
- CAZ with negative weights maximizes over the features a planning unit holds
  and that remain extant, instead of over all features; a unit holding nothing
  extant scores 0.0. For all-nonnegative weights the result is bit-for-bit
  identical to previous releases. Negative weights make scores non-monotone,
  so `warp=1` routes to batch selection instead of the exact lazy heap
  (identical results, slower; a filterable `UserWarning` explains).
```

- [ ] **Step 3: Full gate**

Run: `PATH="/opt/micromamba/envs/shiny/bin:$HOME/.local/bin:$PWD/.venv/bin:$PATH" make check`
Expected: green (~1945 tests). Known SA flake: rerun alone if it is the only failure.

- [ ] **Step 4: Commit**

```bash
git add src/pymarxan/zonation/rank_removal.py CHANGELOG.md
git commit -m "docs(zonation): negative-weight contract + changelog"
```

---

## Self-review (done at write time)

- Spec coverage: §3→T1 kernel, §4.1→T2 Step 3, §4.2→T1, §4.3→T1 Step 1, §4.4→T2 Steps 4-5, §4.5→T3, §5 (invariants)→T1 Step 3 no-op proof, §6.1→T1/T2 existing suite, §6.2-4→T2 hand-computed tests, §6.6→T2 routing probe, §6.7→T2 self-consistency, §6.8→T2 cost test, §6.9→T3 gate.
- **Spec amendment needed (fold before/with the review):** §6.5 promises a test where a cell's only held feature is extinct. That state is unreachable by construction — a remaining holder with `q_ij > 0` keeps `Q_j > 0` — except through the FP-residue path the CELF phase already covers. The `-inf` floor is instead exercised by empty cells (same code line), asserted via PU2 scoring 0.0. Amend §6.5 rather than write an impossible test.
- **New open question for the science lens:** dividing a *negative* score by cost makes an expensive threat-carrying cell rank for removal LATER than an identical cheap one (pinned by `test_negative_weight_cost_division`). Mathematically consistent with `delta/c`, but the conservation semantics deserve checking against Moilanen 2011.
- Type consistency: `has_neg_w: bool` defined T1, consumed T1 (`rescore`) and T2 (routing, warning); `_warn_if_negative_weights_at_warp1(warp, has_neg_w)` defined and called in T2.
- No placeholders; all code complete. Fixture discrimination verified by hand (old `[2,3,4,1]` vs new `[4,3,2,1]`).
