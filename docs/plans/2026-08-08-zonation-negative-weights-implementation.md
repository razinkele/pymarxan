# Negative-Weight Features Implementation Plan (ABF-only)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Support negative (opportunity-cost) feature weights under `rule="abf"` with `use_cost=False`, routing `warp=1` to batch selection — and raise clearly for the two combinations the design review established as unsound.

**Architecture:** **No change to `rescore`.** ABF's `sum` is already correct for either sign; the phase is validation, routing, docs, and one solver metadata marker. Task 1 flips validation, adds the two guards and the heap routing, and lands the ABF semantic tests. Task 2 documents the divergence from Moilanen 2011, marks the curve-reading inversion, and gates.

**Tech Stack:** numpy, pytest.

**Spec:** `docs/plans/2026-08-08-zonation-negative-weights-design.md` (revised post-review) — §2 (ABF-only rationale), §3 (`use_cost` rationale), §4 (the honest semantic), §8 (tests). Review synthesis: `2026-08-08-zonation-negative-weights-review.md`.

## Global Constraints

- Tests ONLY via `/opt/micromamba/envs/shiny/bin/pytest`; ruff `~/.local/bin/ruff` (E,F,I,UP; line 99) **on test files too**; mypy `.venv/bin/mypy src/pymarxan/ src/pymarxan_shiny/ --ignore-missing-imports`; `from __future__ import annotations`.
- **Do not modify `rescore` (`rank_removal.py:310-333`).** Positive-weight behavior must be unchanged *by construction*. A CAZ kernel change was the first draft's approach and was withdrawn — see review findings #1 and #4.
- Amounts stay `>= 0` and finite; only *weights* may now be negative.
- Branch `feat/zonation-negative-weights` (created; spec, plan and review synthesis committed).
- Known SA flake `test_solutions_are_different`: rerun alone if it is the only failure.

---

### Task 1: Validation, guards, routing, ABF semantics

**Files:**
- Modify: `src/pymarxan/zonation/rank_removal.py` (`_validate_inputs` `:78-87`; new helper beside `_warn_if_small_warp`; after the `w` fill at `:235`; advisory call site `:247`; routing `:335`)
- Test: `tests/pymarxan/zonation/test_rank_removal_scale.py` (append + one edit)

**Interfaces:**
- Produces: closure `has_neg_w: bool`; `_warn_if_negative_weights_at_warp1(warp: int, has_neg_w: bool) -> None`; `rank_removal` accepting negative weights under `rule="abf"` + `use_cost=False`.

- [ ] **Step 1: Edit the existing test that asserts the deleted raise.** Replace `test_negative_or_nan_weight_raises` (`test_rank_removal_scale.py:421-426`) with:

```python
def test_nan_weight_raises_and_negative_weight_is_accepted() -> None:
    # The negative-weight raise is gone (ABF opportunity-cost features);
    # the NaN check stays. Review finding #6: this test previously asserted
    # `weights must be >= 0`, which this phase deliberately removes.
    p = _random_problem(0)
    with pytest.raises(ValueError, match="weights must be finite"):
        rank_removal(p, weights={1: float("nan")})
    res = rank_removal(p, rule="abf", weights={1: -2.0}, use_cost=False, warp=4)
    assert len(res.removal_order) == p.n_planning_units
```

- [ ] **Step 2: Append the new tests**

```python
# --- Negative (opportunity-cost) feature weights, ABF-only ---------------
NEG_W = {1: 1.0, 2: -1.0}


def _threat_problem() -> ConservationProblem:
    """PU1 benefit-only, PU2 threat-only, PU3 holds both, PU4 empty.

    Hand-computed ABF trace with weights {1:+1, 2:-1}, unit costs, Q1=Q2=15:
      PU1 10/15=0.6667 | PU2 -10/15=-0.6667 | PU3 5/15-5/15=0.0 | PU4 0.0
      -> remove PU2; Q2=5
      PU3 5/15-5/5 = -0.6667   -> remove PU3; Q1=10, Q2=0 (extinct)
      PU4 0.0 < PU1 10/10=1.0  -> remove PU4, then PU1.
    PU3's score FLIPS SIGN as Q2 shrinks (0.0 -> -0.6667): the concrete
    demonstration of the non-monotonicity that makes the warp=1 lazy heap
    unusable with negative weights.
    """
    pu = pd.DataFrame({"id": [1, 2, 3, 4], "cost": [1.0] * 4, "status": [0] * 4})
    feats = pd.DataFrame(
        {"id": [1, 2], "name": ["benefit", "threat"], "target": [1.0, 1.0],
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
    return ConservationProblem(pu, feats, pvf)


def test_abf_negative_weight_removes_threat_cells_first() -> None:
    res = rank_removal(
        _threat_problem(), rule="abf", weights=NEG_W, use_cost=False, warp=1
    )
    assert res.removal_order == [2, 3, 4, 1]
    # A cell holding ONLY a threat ranks before a cell holding nothing.
    assert res.removal_order.index(2) < res.removal_order.index(4)


def test_caz_with_negative_weight_raises() -> None:
    # Design §2: max is not a trade-off operator — a positive term always
    # masks a negative one, so CAZ + negative weights is degenerate.
    with pytest.raises(ValueError, match="rule='abf'"):
        rank_removal(
            _threat_problem(), rule="caz", weights=NEG_W, use_cost=False
        )
    # Positive-weight CAZ is untouched.
    res = rank_removal(_threat_problem(), rule="caz", weights={1: 1.0, 2: 2.0})
    assert len(res.removal_order) == 4


def test_use_cost_with_negative_weight_raises() -> None:
    # Design §3: dividing a negative score by cost inverts the cost response
    # (the costlier threat cell would be removed later).
    with pytest.raises(ValueError, match="use_cost"):
        rank_removal(_threat_problem(), rule="abf", weights=NEG_W, use_cost=True)
    # use_cost=True is unaffected without negative weights.
    res = rank_removal(_threat_problem(), rule="abf", weights={1: 1.0, 2: 2.0})
    assert len(res.removal_order) == 4


def test_negative_weight_routes_off_the_heap(monkeypatch: pytest.MonkeyPatch) -> None:
    # heapq.heapify runs once per lock-phase on the heap path and never on the
    # batch path — a loud probe rather than counting pops.
    import heapq

    def _boom(*_a: object, **_k: object) -> None:
        raise AssertionError("heap path taken with negative weights")

    monkeypatch.setattr(heapq, "heapify", _boom)
    p = _threat_problem()
    with pytest.warns(UserWarning, match="non-monotone"):
        res = rank_removal(p, rule="abf", weights=NEG_W, use_cost=False, warp=1)
    assert res.removal_order == [2, 3, 4, 1]
    # The probe has teeth: positive weights at warp=1 DO reach heapify.
    with pytest.raises(AssertionError, match="heap path taken"):
        rank_removal(p, rule="abf", weights={1: 1.0, 2: 2.0}, warp=1)


def test_no_heap_warning_without_negative_weights() -> None:
    import warnings as _warnings

    with _warnings.catch_warnings():
        _warnings.simplefilter("error")
        rank_removal(_threat_problem(), rule="abf", weights={1: 1.0, 2: 2.0}, warp=1)


@pytest.mark.parametrize("warp", [1, 3])
def test_negative_weight_self_consistency(warp: int) -> None:
    # Only monotonicity was lost; the dirty-set shortcut and batch selection
    # stay valid, so these must agree exactly at a FIXED warp.
    p = _threat_problem()
    kw = {"rule": "abf", "weights": NEG_W, "use_cost": False, "warp": warp}
    base = rank_removal(p, **kw)  # type: ignore[arg-type]
    _assert_equal_results(base, rank_removal(p, **kw, _force_batch=True))  # type: ignore[arg-type]
    _assert_equal_results(
        base, rank_removal(p, **kw, _force_full_rescore=True)  # type: ignore[arg-type]
    )


def test_negative_weight_near_extinction_magnitudes() -> None:
    # Spec §7 / review finding #8: negative terms are unbounded BELOW as
    # Q -> 0, a magnitude regime the progress guard has only met from the
    # positive side. The run must complete (or raise the documented
    # RuntimeError) — never hang or emit NaN ranks.
    pu = pd.DataFrame({"id": [1, 2, 3], "cost": [1.0] * 3, "status": [0] * 3})
    feats = pd.DataFrame(
        {"id": [1, 2], "name": ["a", "b"], "target": [1.0, 1.0], "spf": [1.0, 1.0]}
    )
    pvf = pd.DataFrame(
        [
            {"species": 1, "pu": 1, "amount": 1e-300},
            {"species": 2, "pu": 2, "amount": 1.0},
            {"species": 2, "pu": 3, "amount": 2.0},
        ]
    )
    p = ConservationProblem(pu, feats, pvf)
    res = rank_removal(p, rule="abf", weights={1: -1.0}, use_cost=False, warp=1)
    assert len(res.removal_order) == 3
    assert all(np.isfinite(v) for v in res.priority_rank.values())
```

- [ ] **Step 3: Run to verify failures**

Run: `/opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/zonation/test_rank_removal_scale.py -v -k "negative_weight or abf_negative or caz_with_negative or use_cost_with_negative or nan_weight_raises or heap_warning"`
Expected: FAIL, most with `ValueError: feature weights must be >= 0 ...`.

- [ ] **Step 4: Drop the validation raise.** In `_validate_inputs`, delete the `if (wv < 0).any(): raise ...` block (`rank_removal.py:81-87`), keeping the finiteness check, and replace the docstring's negative-weight sentence with:

```
    Negative weights are supported under ``rule="abf"`` (Zonation v3+
    opportunity-cost features; Moilanen et al. 2011, doi:10.1890/10-1865.1);
    amounts must stay nonnegative — Zonation expresses negative features
    through weights, not occurrence levels.
```

- [ ] **Step 5: Add the flag and the two guards.** Immediately after the `w` fill block (after `rank_removal.py:235`) — the flag must read the *filled* `w`, so a negative weight keyed to an absent feature id is correctly ignored:

```python
    has_neg_w = bool((w < 0).any())
    if has_neg_w and rule == "caz":
        raise ValueError(
            "negative feature weights are not supported with rule='caz': the "
            "core-area max cannot trade a negative term against a positive "
            "one (any positive term masks it entirely), so the weight would "
            "be silently inert. Use rule='abf', whose additive form is the "
            "one Zonation's negative-feature machinery is defined for "
            "(Moilanen et al. 2011, doi:10.1890/10-1865.1)."
        )
    if has_neg_w and use_cost:
        raise ValueError(
            "negative feature weights cannot be combined with use_cost=True: "
            "dividing a negative score by cost inverts the cost response, so "
            "a costlier threat-carrying cell would rank for removal LATER "
            "than an identical cheap one. Pass use_cost=False, or express "
            "costs as negatively weighted features (Moilanen et al. 2011, "
            "doi:10.1890/10-1865.1)."
        )
```

- [ ] **Step 6: Add the routing warning helper** beside `_warn_if_small_warp`:

```python
def _warn_if_negative_weights_at_warp1(warp: int, has_neg_w: bool) -> None:
    """Advise that negative weights disable the exact warp=1 lazy heap.

    A term ``w_j q_ij / Q_j`` with ``w_j < 0`` grows MORE negative as ``Q_j``
    shrinks, so scores are no longer nondecreasing under removal, cached heap
    keys stop being lower bounds, and the lazy-greedy exactness argument
    fails. Batch selection has no such dependency — its dirty set tracks
    changed inputs, not monotone scores — so it is used instead.
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

- [ ] **Step 7: Wire the call and the routing.** After `_warn_if_small_warp(n_pu, warp)` (`:247`):

```python
    _warn_if_negative_weights_at_warp1(warp, has_neg_w)
```

and extend the routing line (`:335`):

```python
    use_heap = (
        warp == 1
        and not _force_batch
        and not _force_full_rescore
        and not has_neg_w
    )
```

- [ ] **Step 8: Run everything**

Run: `/opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/zonation/ tests/pymarxan/solvers/test_zonation_solver.py tests/pymarxan_shiny/test_zonation_panel.py -q`
Expected: ALL PASS — the new tests plus every pre-existing one. `rescore` is untouched, so positive-weight results cannot have moved. If the hand-computed `[2, 3, 4, 1]` fails, recompute from the fixture comment before touching the engine (the review verified this arithmetic). Then ruff (both files) + mypy.

- [ ] **Step 9: Commit**

```bash
git add src/pymarxan/zonation/rank_removal.py tests/pymarxan/zonation/test_rank_removal_scale.py
git commit -m "feat(zonation): negative (opportunity-cost) feature weights, ABF-only"
```

---

### Task 2: Solver marker, docs, CHANGELOG, gate

**Files:**
- Modify: `src/pymarxan/solvers/zonation_solver.py` (metadata dict `:84-90`), `src/pymarxan/zonation/rank_removal.py` (docstrings), `src/pymarxan/zonation/result.py` (`performance_curves` docstring), `CHANGELOG.md`
- Test: `tests/pymarxan/zonation/test_rank_removal_scale.py` (append one)

**Interfaces:**
- Consumes: `has_neg_w` semantics from Task 1.
- Produces: `Solution.metadata["negative_weight_features"]: list[int]`.

- [ ] **Step 1: Failing solver test.** Append:

```python
def test_solver_negative_weights_metadata_marker() -> None:
    from pymarxan.solvers.zonation_solver import ZonationSolver

    solver = ZonationSolver(
        rule="abf", weights=NEG_W, use_cost=False, warp=2, top_fraction=0.5
    )
    sols = solver.solve(_threat_problem(), {})
    assert len(sols) == 1
    # Curves for a negatively weighted feature read INVERSELY (fraction of
    # the threat still inside the reserve; lower is better), so consumers
    # need to know which series to flip — spec §6.
    assert sols[0].metadata.get("negative_weight_features") == [2]
```

- [ ] **Step 2: Run to verify it fails** (`assert None == [2]`).

- [ ] **Step 3: Add the marker.** In `zonation_solver.py`'s metadata dict (after `"smoothing_alpha"`, `:89`):

```python
            "negative_weight_features": sorted(
                int(fid) for fid, wt in (self.weights or {}).items() if wt < 0
            ),
```

- [ ] **Step 4: Docstrings.** In `rank_removal`'s docstring, after the CAZ/ABF sentence, insert:

```
    Feature weights may be negative under ``rule="abf"`` (Zonation v3+
    opportunity-cost / alternative-land-use features; Moilanen et al. 2011,
    doi:10.1890/10-1865.1) so that cells carrying them rank for removal first.
    Two combinations raise ``ValueError``: ``rule="caz"`` (a max cannot trade a
    negative term against a positive one, so the weight would be silently
    inert) and ``use_cost=True`` (dividing a negative score by cost inverts
    the cost response). Negative weights also make scores non-monotone under
    removal, so ``warp=1`` uses batch selection rather than the exact lazy
    heap — identical results, without the heap's speed (a filterable
    ``UserWarning`` says so).

    Semantics note: this engine's proportional / remaining-sum marginal makes a
    negatively weighted feature INCREASINGLY urgent to exclude as it nears
    elimination. Moilanen et al. 2011 instead inverts the benefit function for
    negative features (``z_k = 4``) so they become DECREASINGLY urgent once
    mostly excluded; removal orders differ materially. The citation is for the
    concept — negative weights represent opportunity costs — not for identical
    dynamics; the faithful form needs the per-feature benefit exponent listed
    as a future extension.

    Performance curves for a negatively weighted feature read inversely: the
    stored fraction is the share of that feature still INSIDE the remaining
    set, so lower is better. ``ZonationSolver`` records the affected feature
    ids in ``Solution.metadata["negative_weight_features"]``.
```

In `result.py`, extend the `performance_curves` description with one sentence:
`For negatively weighted features the retained fraction reads inversely (lower is better) — see rank_removal's docstring.`

- [ ] **Step 5: CHANGELOG** under `[Unreleased]`:

```markdown
### Added
- `zonation.rank_removal` accepts negative feature weights under `rule="abf"` —
  Zonation v3+ opportunity-cost / alternative-land-use layers (concept:
  Moilanen et al. 2011, doi:10.1890/10-1865.1) — so cells carrying them rank
  for removal first. `ZonationSolver` inherits this through its existing
  `weights` parameter and records the affected ids in
  `Solution.metadata["negative_weight_features"]`, since performance curves for
  those features read inversely (lower is better).

### Changed
- `warp=1` routes to batch selection when any weight is negative: such scores
  are non-monotone under removal, so the exact lazy heap's lower-bound argument
  does not hold (identical results, slower; a filterable `UserWarning`
  explains). `rule="caz"` and `use_cost=True` raise `ValueError` when combined
  with negative weights — a max cannot trade a negative term against a positive
  one, and dividing a negative score by cost inverts the cost response.
  Positive-weight behavior is unchanged: no scoring code was modified.
```

- [ ] **Step 6: Full gate**

Run: `PATH="/opt/micromamba/envs/shiny/bin:$HOME/.local/bin:$PWD/.venv/bin:$PATH" make check`
Expected: green (~1945 tests). Known SA flake: rerun alone if it is the only failure.

- [ ] **Step 7: Commit**

```bash
git add src/pymarxan/zonation/rank_removal.py src/pymarxan/zonation/result.py src/pymarxan/solvers/zonation_solver.py tests/pymarxan/zonation/test_rank_removal_scale.py CHANGELOG.md
git commit -m "docs(zonation): negative-weight contract, curve-reading marker, changelog"
```

---

## Self-review (done at write time)

- Spec coverage: §2 (CAZ raise)→T1 Step 5 + test; §3 (`use_cost` raise)→T1 Step 5 + test; §4 (honest semantic)→T2 Step 4; §5.1→T1 Step 4; §5.2→T1 Step 5; §5.3→T1 Steps 6-7; §5.4→T2 Step 4; §5.5→T2 Steps 1-3; §6→T2 Steps 1/4; §7→T1 near-extinction test; §8.1→T1 Step 1; §8.2-3→T1 Step 2; §8.4→T1 Step 2; §8.5→T1 Step 2; §8.6→T1 Step 2; §8.7→T2 Step 1; §8.8→T1 Step 8; §8.9→T2 Step 6.
- `rescore` appears in no task's Files list — the constraint is structural, not aspirational.
- Type consistency: `has_neg_w: bool` defined T1 Step 5, used Steps 5/7; helper signature identical at definition (Step 6) and call (Step 7); `NEG_W` and `_threat_problem` defined T1 Step 2, reused T2 Step 1.
- The near-extinction test accepts either completion or the documented `RuntimeError`; as written it asserts completion, since with `w=-1` and one holder the `0.0*inf` NaN path needs a *second* feature column to be absent — verified reachable in this fixture. If it raises instead, the implementer should switch the assertion to `pytest.raises(RuntimeError, match="no progress")` and note it — both are documented behavior, and which one occurs is an engine fact, not a design choice.
- No placeholders.
