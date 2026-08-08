# CELF lazy-heap: exact warp=1 rank_removal at raster scale — design spec

**Date:** 2026-08-08
**Status:** draft (pending loop review + multi-agent design review)
**Prior art:** deferred from the v0.31.0 raster-scale phase with three logged caveats
(roadmap + `2026-08-08-zonation-raster-scale-review.md` finding #17). Scope decisions
from brainstorm: auto-activation at `warp==1`; `curve_every` public parameter + array
curve storage; heap freshness tracked by the existing `dirty` array.

## 1. Problem

v0.31.0 made `rank_removal` sparse and incremental, but `warp=1` — the exact greedy
that removes one cell at a time, Zonation's finest-resolution mode — remains O(n²) in
*selection* (an argpartition over ~n candidates per single removal) and is advised
against at raster scale. Two consequences:

1. Users wanting the most refined ranking at 1M cells have no path (de Mello et al.
   2015 chose warp=1 deliberately on a country-wide raster).
2. Newly reachable `warp=1` would also explode curve storage: one dict row per
   removal ≈ 1 GB of Python dicts at 1M cells.

## 2. Goals / non-goals

**Goals**

- `warp=1` at ~1M cells completes in minutes on a workstation (localized features),
  producing output **bitwise-identical to the v0.31.0 batch path at warp=1 — float
  amounts included** (§7 states why this is a stronger claim than dense-oracle
  equivalence and why it is achievable).
- Activation is automatic: `warp==1` (after normalization) → heap path; `warp>1` →
  existing batch path, byte-for-byte untouched in behavior. Private test-only
  `_force_batch: bool = False` keeps the batch path reachable at warp=1 for
  equivalence tests.
- New kw-only `curve_every: int = 1` honored identically by both paths; curve
  storage becomes a preallocated float64 array (same values, same DataFrame shape).
- The warp advisory stops firing for `warp==1` (now the fast path) and keeps firing
  for `2 ≤ warp < n_pu // 10_000` at scale.

**Non-goals**

- No change to `warp>1` semantics or performance; no `ZonationResult` change; no
  `ZonationSolver` change (exposing `curve_every` there is deferred until asked for).
- No edge-removal candidate restriction (still the documented v0.13-era difference).
- No mitigation of the landscape-spanning-feature worst case (dirty-marking is
  O(holders) per removal in either path; documented, not defended).

## 3. Why the lazy heap is exact (the invariant)

With the v0.31.0 validation in force (amounts ≥ 0 and finite, weights ≥ 0 and
finite, costs > 0 when used), removal only ever *decreases* each remaining total
`Q_j`, so every term `w_j·q_ij/Q_j` of a remaining cell is nondecreasing over the
run, and both CAZ (max) and ABF (sum) scores divided by a constant cost are
nondecreasing. Hence **any cached score is a lower bound on the current true score**.

Lazy-heap selection: min-heap of `(score, pu_index)` tuples. Pop the top; if the
cell's cached score is *fresh* (its features' totals unchanged since scoring — i.e.
`dirty[i]` is False and the entry's key equals `delta[i]`), it is the true global
argmin: every other heap key is a lower bound on its cell's true score, and the
fresh top is ≤ all keys. Tie-break `(score, index)` is preserved *through
staleness*: a stale competitor's key is ≤ its true score, so an equal-true-score
lower-index cell pops earlier, gets refreshed, and re-enters at the same key with
the smaller index — tuple order then decides exactly as the batch path's stable
argsort does.

**The one exception, and its repair (§5): FP-residue extinction.** If `Q_j` crosses
from >0 to ≤0 while a holder remains (possible only with float amounts — integer
amounts guarantee a remaining holder keeps `Q_j ≥ q_ij ≥ 1`), the holders' terms
drop to 0: their true scores *decrease*, cached keys stop being lower bounds, and
the heap can return a wrong argmin. Repair: eagerly rescore and re-push the extinct
feature's remaining holders at the moment of crossing. At most `n_feat` events per
run, O(nnz of that CSC column) each.

A second, independent reason the deferral notes recorded — a lazy heap cannot
reproduce warp>1's frozen-batch semantics — is moot under auto-activation: the heap
only ever runs at warp=1.

## 4. Structure of the change

All in `src/pymarxan/zonation/rank_removal.py`; no new modules.

1. **`remove_cell(idx) -> np.ndarray` (local closure), used by BOTH paths.** Exactly
   the current per-removal bookkeeping, extracted verbatim: append
   `int(pu_ids[idx])` to `removal_order`, clear `remaining[idx]`, sequential
   `Q[indices[s:e]] -= data[s:e]`, decrement `cost_remaining`, decrement
   `n_remaining`. New (cheap, O(nnz_row)): capture `prev = Q[cols]` before the
   subtraction and return the pair `(cols, crossed)` where `cols` is the removed
   row's feature-column slice (`indices[s:e]`) and
   `crossed = cols[(prev > 0) & (Q[cols] <= 0)]` — the features this removal drove
   to extinction. The batch path consumes `cols` (its `changed_parts` union
   marking) and ignores `crossed`; the heap path consumes both (`cols` for holder
   marking, `crossed` for the §5 repair). Sharing the function makes "identical
   trajectories" structural.
   Dirty-marking stays **path-specific** (deliberately not in `remove_cell`): the
   batch path keeps its existing post-batch union marking (marking per removal would
   re-mark shared features' holders up to k times per batch); the heap path marks
   per removal, which at k=1 is the same set.
2. **Batch path:** the existing `while` loop, with its per-removal body replaced by
   `remove_cell(idx)` + the retained `changed_parts` collection — no behavioral
   change (verified by the entire existing suite).
3. **Heap path**, taken iff `warp == 1 and not _force_batch and not
   _force_full_rescore` — `_force_full_rescore` is a batch-semantics knob, so it
   forces the batch path too; the pre-existing
   `test_dirty_set_equals_full_rescore(warp=1)` thereby becomes a free
   heap-vs-batch-full-rescore equality check. Per lock-phase:
   - `cand = candidate_indices()`; capture `phase_mask` (boolean, True exactly on
     `cand`) — the extinction repair needs phase membership (§5);
     `rescore(cand[dirty[cand]])`; heapify `[(float(delta[i]), int(i)) for i in
     cand]`; `phase_left = cand.size`. Every push, everywhere, uses
     `(float(...), int(...))` so tuple comparisons stay homogeneous.
   - Pop `(s, i)`: if `not remaining[i]` → skip (lazy deletion of duplicates);
     elif `dirty[i]` → `rescore(np.array([i]))` (the same chunked-dense kernel,
     chunk of one — bitwise the value a batch rescore would produce), push
     `(float(delta[i]), i)`, continue; elif `s != delta[i]` → superseded duplicate,
     skip; else **accept**: `crossed = remove_cell(i)`; mark `dirty` on the CSC
     holders of `i`'s features; for each feature in `crossed`, rescore its remaining
     holders and push fresh entries (the §3 repair — note these cells are rescored
     *now*, clearing `dirty`, so their entries are fresh); `phase_left -= 1`;
     curve-record per §6. Phase ends at `phase_left == 0`.
   - **Non-finite guard:** every value pushed to the heap is checked
     `np.isfinite(...)`; violation raises the same `RuntimeError` as the batch
     path's progress guard ("made no progress" wording adjusted to cover both
     sites). NaN must never enter the heap — heapq comparisons silently corrupt the
     invariant. The heap path may therefore raise *earlier* in a run than the batch
     path would; same exception type and cause, documented.
4. **`_warn_if_small_warp`** gains `warp > 1` in its condition; docstring updated
   (warp=1 is the exact fast path now).

## 5. FP-residue extinction repair — precision

Trigger: `remove_cell` returns `crossed` (features with `prev > 0` and now
`Q_j ≤ 0`). For each such `j`: `repair = csc column j ∩ remaining ∩ phase_mask`;
`rescore(repair)`; push `(float(delta[h]), int(h))` for each. **Phase membership is
load-bearing**: the heap must only ever contain current-phase candidates — pushing
an out-of-phase holder (e.g. a locked-in cell during the normal phase) would let it
be selected in the wrong phase. Out-of-phase holders of a crossed feature need no
repair: they are already marked dirty by the standard holder-marking (crossed ⊆ the
removed cell's features), and each later phase rescores its dirty candidates at
phase init before heapifying, so no stale bound ever enters a heap. Cost bounds: each
feature crosses at most once (Q never increases), so total repair work over a run
is ≤ O(nnz). The repair happens *before* the next pop, so the invariant is restored
before any selection can consult a poisoned bound. Integer-amount runs never
trigger it; the constructed-fixture test (§8) plus a broad float sweep cover it.

## 6. Curves: `curve_every` + array storage

- New kw-only `curve_every: int = 1`; validated `isinstance(curve_every, int) and
  curve_every >= 1` else `ValueError` (no silent float truncation; `bool` passes
  the isinstance check as an int subclass — harmless, `True` behaves as 1). Semantics for BOTH paths: record the initial state; after each
  *batch* (heap path: each removal is a batch of one), record iff
  `n_removed_total % curve_every == 0`; always record the final state once (skip if
  the last record already was final). `curve_every=1` reproduces today's rows
  exactly for every warp. For `warp>1`, records land on batch boundaries whose
  cumulative removal count is a multiple of `curve_every` (documented; phase-tail
  batches of odd size may skip a multiple).
- Storage: preallocate `curves = np.empty((3 + n_pu // curve_every, 2 + n_feat))`;
  `record_curve` writes a row (`n_remaining / n_pu`,
  `max(cost_remaining, 0.0) / cost_total`, then `np.where(T > 0, Q / T_safe, 1.0)`)
  — the same float64 values the dict path produced. Final DataFrame:
  `pd.DataFrame(curves[:r], columns=[...])` with the exact current column names and
  order, so `assert_frame_equal(check_exact=True)` against the dict-era oracle
  still holds. `ZonationResult` is unchanged (still a DataFrame).

## 7. Equivalence contract

The claim is **heap warp=1 ≡ batch warp=1, bitwise, for all valid inputs including
float amounts and smoothing** — stronger than the v0.31.0 dense-oracle contract
(which has ULP sites at the dense/sparse boundary). It is achievable because both
paths, given the same `Q` trajectory, compute scores with the *same expressions on
the same machinery* (`rescore`), and §3/§5 establish the heap selects the same
`(score, index)` argmin each step; identical selections ⇒ identical sequential `Q`
updates ⇒ induction. The batch path's relation to the dense oracle is unchanged
(v0.31.0 §7), so heap-vs-oracle inherits exactly those caveats and no new ones.

Curve values: unchanged bitwise at `curve_every=1`; a thinned run's rows are an
exact subset (row-for-row bitwise) of the `curve_every=1` run's rows.

## 8. Testing

All in `tests/pymarxan/zonation/test_rank_removal_scale.py` (append) +
`tests/benchmarks/bench_zonation.py` (append); env per `marxan-testing`.

1. **Heap-vs-batch bitwise (the core):** for every fixture family already in the
   suite — random integer (seeds 0–2), locks+costs, weights+extinction, wide
   n_feat=25, stored-zero, duplicates, featureless-PU, n_feat=0, float amounts+costs
   (seed 11), smoothing (seed 17) — assert `rank_removal(warp=1)` equals
   `rank_removal(warp=1, _force_batch=True)` via `_assert_equal_results`
   (exact, `check_exact` curves). Plus a 30-seed float sweep (both rules) — this is
   the statistical net for the §5 repair.
2. **Constructed FP-residue extinction fixture (best effort):** engineer a float
   problem where a feature's `Q_j` goes ≤0 while a tiny-amount holder remains,
   with the crossing happening *inside the normal phase* so the repair-push path
   itself is exercised (lever: give the feature's big-amount holders huge costs —
   cost divides the score, so they are removed first — while the residue carrier
   holds a tiny amount; locks would move the crossing into the locked-out phase,
   where the repair-push is skipped by the phase mask); assert heap==batch and that
   the run terminates. The grounding review agent is tasked with verifying the
   construction actually crosses (instrument `remove_cell`'s return); if genuinely
   unconstructible, the sweep in (1) plus a unit test of the repair mechanics
   (monkeypatch a `Q` crossing) replaces it — the plan must say which.
3. **Existing suite as heap coverage:** all current warp=1 fixtures now exercise
   the heap path against the dense oracle with no test edits — their staying green
   is required evidence.
4. **`curve_every`:** (a) thinned == full rows `[::k]` + final, bitwise, both
   paths; (b) `curve_every=1` default unchanged (covered by the whole existing
   suite); (c) `curve_every=0`/negative/non-int → `ValueError`; (d) a `warp>1` +
   `curve_every>1` case pinning the batch-boundary semantics.
5. **Guards:** NaN-poisoned input (the subnormal fixture) raises `RuntimeError` on
   the heap path too; advisory: `(1_000_000, 1)` no longer warns, `(1_000_000, 50)`
   still does, boundary non-warn cases unchanged.
6. **Bench (append, bench-marked):** 300×300 grid, `warp=1`, `curve_every=1000`,
   budget < 60 s; assert full removal order length.
7. **`make check`** green; parity anchor untouched (no Marxan solver touched).

## 9. Performance envelope (claims for review)

- Heap: n accepts + duplicates; each pop O(log n); dirty pops add a single-row
  rescore (~10–20 µs). Per-accept bookkeeping (Q row update, CSC holder marking) is
  the same numpy work the batch path does per removal.
- 90k cells warp=1: seconds (bench-verified). 1M cells, localized features:
  minutes — dominated by per-removal Python-level loop overhead (~1M iterations ×
  ~20–60 µs). Heap memory ≈ n × ~90 B ≈ 90 MB at 1M plus re-push growth.
- Worst case (landscape-spanning feature): dirty-marking O(n) per removal → O(n²)
  — same as the batch path at warp=1 today; documented in the docstring.
- Curve array at `curve_every=1000`: ~1k rows — negligible. At `curve_every=1`,
  1M×22 float64 ≈ 176 MB (documented; the parameter exists precisely for this).

## 10. Files touched

- `src/pymarxan/zonation/rank_removal.py` — heap path, `remove_cell` extraction,
  `curve_every`, array curves, advisory condition, docstring (warp=1 contract,
  curve_every, advisory scope, heap worst case).
- `tests/pymarxan/zonation/test_rank_removal_scale.py` — §8 tests (append).
- `tests/benchmarks/bench_zonation.py` — warp=1 bench (append).
- `CHANGELOG.md` — `[Unreleased]` Added (`curve_every`) + Changed (warp=1 heap,
  advisory scope). Next release: minor, v0.32.0.
- Roadmap memory post-merge: CELF deferral → done; remaining deferrals unchanged.

## 11. Risks / open questions for review

- The freshness test (`dirty[i]` + key-vs-`delta[i]` match) assumes `delta[i]` is
  only ever written by `rescore` — confirm no other writer exists (grounding).
- Key equality `s != delta[i]` uses float equality on purpose (a superseded entry's
  key differs only if a rescore wrote a new value); NaN never enters (guarded) so
  `!=` is safe — reviewers should sanity-check.
- Heap-path curve recording at `curve_every=1` produces one row per removal — the
  array write is O(n_feat) per removal; at 1M cells × 100 features that is 1e8
  float writes (~seconds) — acceptable, but the bench should use a thinned
  curve_every so the bench measures selection, not curve I/O.
- `remove_cell` extraction must not perturb the batch path's FP trajectory — the
  whole existing suite is the net; reviewers should diff the extraction closely.
