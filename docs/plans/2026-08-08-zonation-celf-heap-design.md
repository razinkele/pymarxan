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
   2015 chose warp=1 deliberately on a biome-wide raster — the Brazilian Cerrado;
   full-text verified: "we chose to remove one pixel at a time, which results in a
   more refined solution").
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

- No change to `warp>1` *behavior* (bitwise); the shared `remove_cell` adds
  O(nnz_row) crossed-detection bookkeeping per removal to the batch path — accepted,
  with the existing warp-scale bench as the regression net. No `ZonationResult`
  change; no `ZonationSolver` change (exposing `curve_every` there is deferred).
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

This monotonicity holds in float64, not merely over the reals (design-review #8):
IEEE-754 correctly-rounded ops are weakly monotone and numpy's per-row reduction
tree is fixed for fixed `n_feat`, so cached keys are lower bounds *bitwise*;
overflow to +inf preserves order (inf tuples are totally ordered in a heap), and
NaN — the only non-ordered value — is guarded out of the heap (§4.3).

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
     elif `dirty[i]` → **buffered dirty rescore** (design-review #1, the CRITICAL
     performance fix — single-row rescores cost ~72.5 µs each in scipy slice
     overhead vs 1.5 µs/row vectorized, and the naive per-pop variant measured
     ~18× SLOWER than the batch path): buffer `i`, then keep popping while the
     heap top is removed (drop) or dirty (buffer), stopping at the first fresh
     top; rescore the deduplicated buffer in ONE vectorized `rescore()` call;
     push every buffered cell back at its new key; continue the outer loop.
     Values are bitwise what per-row rescores would produce (the kernel is
     chunk-shape-independent), and accepts still happen only on a fresh top, so
     selection semantics are unchanged. Elif `s != delta[i]` → superseded
     duplicate, skip (an inline comment records the invariant this relies on:
     `delta` is written only by `rescore`, and every heap-path rescore is
     followed by a push/heapify of the rescored cells; plus `assert not dirty[i]`
     before accept). Else **accept**: `cols, crossed = remove_cell(i)`; mark
     `dirty` on the CSC holders of `cols`; for each feature in `crossed`, rescore
     its remaining current-phase holders and push fresh entries (the §3 repair);
     `phase_left -= 1`; curve-record per §6. Phase ends at `phase_left == 0`.
   - **NaN guard (design-review #4 — NaN only, not isfinite):** every value about
     to be pushed (phase init, buffered rescore, repair) is checked with
     `np.isnan`; a NaN raises the batch path's `RuntimeError`. NaN must never
     enter the heap — heapq comparisons silently corrupt the invariant — but
     **+inf keys are totally ordered and stay**: an isfinite guard would raise on
     all-inf score regimes where the batch path completes (verified
     counterexample: two PUs sharing a 1e-310 feature → batch returns [1, 2]),
     needlessly shrinking the equivalence domain. With NaN-only guarding, the
     heap may still raise where batch limps through a NaN-ordered tail — that
     carve-out is §7's, and it also converts the late-rescore inf-timing hazard
     (a dirty cell rescored later than batch would, seeing a non-held feature's
     factor transition finite→inf → 0·inf = NaN) into a loud error, never a
     wrong order.
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

- New kw-only `curve_every: int = 1`; normalized via `operator.index(curve_every)`
  in try/except → `ValueError`, then `>= 1` checked (design-review #5: accepts
  Python int AND `np.integer` — the likeliest caller type for a raster memory knob
  — while rejecting floats/strings/None with no silent truncation; `bool` indexes
  to 0/1, `True` harmless, `False` fails the ≥1 check). Semantics for BOTH paths: record the initial state; after each
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

The claim is **heap warp=1 ≡ batch warp=1, bitwise, for every valid input whose
scores never evaluate to NaN during the run — float amounts, +inf score regimes,
and smoothing included** (design-review #4 scoped this: with NaN-only guarding,
all-inf regimes stay bitwise-identical; on NaN-producing runs the heap raises
`RuntimeError` where the batch path may either raise its progress guard or limp to
a NaN-ordered tail — failing fast on a semantically garbage ranking is the better
behavior and is documented, not hidden). It is achievable because both paths,
given the same `Q` trajectory, compute scores with the *same expressions on the
same machinery* (`rescore`), and §3/§5 establish the heap selects the same
`(score, index)` argmin each step; identical selections ⇒ identical sequential `Q`
updates ⇒ induction. The batch path's relation to the dense oracle is unchanged
(v0.31.0 §7), so heap-vs-oracle inherits exactly those caveats and no new ones.

Curve values: unchanged bitwise at `curve_every=1`; a thinned run's rows are an
exact subset (row-for-row bitwise) of the `curve_every=1` run's rows.

## 8. Testing

All in `tests/pymarxan/zonation/test_rank_removal_scale.py` (append) +
`tests/benchmarks/bench_zonation.py` (append); env per `marxan-testing`.

1. **Heap-vs-batch bitwise (the core):** for every fixture family — random integer
   (seeds 0–2), locks+costs, **weights incl. a weight-0 feature** (design-review
   #6: w≠1 is a distinct rescore input; w=0 meets the dead-mask), wide n_feat=25,
   stored-zero, duplicates, featureless-PU, n_feat=0, float amounts+costs (seed
   11), smoothing (seed 17), and an **all-ties fixture** (n_pu≈30, one feature,
   equal amounts, `use_cost=False` — sustained equal-key regime stressing the
   tie-break-through-staleness argument; expected order = ascending PU index,
   asserted directly too) — assert `rank_removal(warp=1)` equals
   `rank_removal(warp=1, _force_batch=True)` via `_assert_equal_results`. Plus a
   30-seed float sweep (both rules) as **general** coverage — review #2 measured 0
   extinction crossings in 120 such runs, so the sweep is explicitly NOT the
   repair's net; the constructed fixtures below are.
2. **Detector-verified FP-residue extinction fixture (design-review #2, verified
   construction):** f1 = {PU1: 0.3 @cost 1000, PU2: 0.3 @cost 1000, PU3 (residue
   carrier): 1e-17 @cost **1e-15**}, f2 = {PU4: 5.0, PU5: 3.0} @cost 1, f3 =
   {PU6 (detector): 1.0 @cost 200}. The carrier's cost keeps its score
   (1.67e-2) ABOVE the big holders' (5e-4, 1e-3) until the crossing — the original
   "huge costs on big holders" lever was backwards: a near-zero-amount holder at
   cost 1 is the global argmin and leaves first. The detector's score (5e-3) sits
   strictly inside the carrier's stale-key/post-crossing-true-score gap, so a
   missing, broken, or phase-inverted repair provably flips the order
   ([1,2,6,3,5,4] instead of [1,2,3,6,5,4] — execution-verified, both rules):
   this *verifies* the repair, not merely exercises it. Assert heap==batch, the
   expected order, and that PUs 1–2 precede PU 3.
3. **Cross-phase extinction fixture (design-review #3):** the same construction
   with PUs 1–2 locked out (status 3) — the crossing happens in the locked-out
   phase while the carrier sits in the normal phase, exercising the
   repair-push-skipped + dirty-carry + phase-init-rescore path that §5 declares
   safe. Assert heap==batch.
4. **Existing suite as heap coverage:** all current warp=1 fixtures now exercise
   the heap path against the dense oracle with no test edits — their staying green
   is required evidence.
5. **`curve_every`:** (a) thinned == full rows `[::k]` + final, bitwise,
   **parametrized over `_force_batch`** (design-review #10: the batch path at
   `curve_every>1` must keep direct coverage after warp=1 re-routes to the heap);
   (b) `curve_every=1` default unchanged (whole existing suite); (c)
   `curve_every=0`/negative/`2.5`/`"5"`/`None` → `ValueError` while `np.int64(7)`
   is accepted; (d) a `warp>1` + `curve_every>1` case pinning batch-boundary
   semantics.
6. **Guards:** the subnormal NaN fixture raises `RuntimeError` on BOTH paths;
   advisory: `(1_000_000, 1)` no longer warns, `(1_000_000, 50)` still does,
   boundary non-warn cases unchanged; `_warn_if_small_warp`'s own docstring is
   rewritten (its "warp=1 is O(n²)" opener becomes false — design-review #10).
7. **Bench (append, bench-marked):** 300×300 grid, `warp=1`, `curve_every=1000` —
   **measure first, then assert**: review #1 measured the batch path itself at
   120.7 s on this geometry/machine and the naive heap at DNF, so the buffered
   implementation must be timed before any budget or docstring perf claim ships;
   target budget < 60 s, escalate if the measurement misses it. Assert full
   removal-order length.
8. **`make check`** green; parity anchor untouched (no Marxan solver touched).

## 9. Performance envelope (claims for review)

- **Measured basis (design-review #1, side=100 grid geometry):** ~62 dirty pops
  per accept; a single-row `rescore` costs ~72.5 µs (scipy slice overhead) vs
  1.5 µs/row vectorized — hence the buffered-pop loop in §4.3, which replaces ~62
  single-row calls (~4.5 ms) per accept with one vectorized call (~90–120 µs), an
  ~18× inner-loop reduction. The naive per-pop variant measured 47.8 s vs the
  batch path's 2.7 s at side=100 and DNF at side=300 — never regress to it.
- Reference points on this machine: batch warp=1 at side=300 (90k cells) is
  120.7 s. The buffered heap's projection is well under that; the Task-4 bench
  MEASURES before asserting the < 60 s budget (§8.7). 1M-cell claims wait for the
  bench — no "minutes at 1M" wording ships unmeasured.
- Worst case (landscape-spanning feature): holder-marking O(n) per removal → O(n²)
  — same as the batch path at warp=1 today; documented in the docstring.
- Heap memory ≈ n × ~90 B ≈ 90 MB at 1M plus re-push growth. Curve array at
  `curve_every=1000`: ~1k rows — negligible. At `curve_every=1`, 1M×22 float64 ≈
  176 MB (the parameter exists precisely for this).

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
