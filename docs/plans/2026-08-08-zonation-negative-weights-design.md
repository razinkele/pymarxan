# Negative-weight (opportunity-cost) features — design spec

**Date:** 2026-08-08
**Status:** draft (pending loop review + multi-agent design review)
**Provenance:** the deferral named in `_validate_inputs`'s own error message
("negative weights, used by Zonation v3+ for opportunity-cost features, are not
yet supported"), logged at v0.31.0 after the science lens established that
negative weighting is real published Zonation practice (Moilanen et al. 2011,
doi:10.1890/10-1865.1: "negatively weighted features can be taken to represent
multiple (opportunity) costs"). Brainstorm decisions: CAZ max restricted to
held-and-extant features; warp=1 auto-routes to batch selection; full 4-lens
review.

## 1. Problem

`rank_removal` rejects negative feature weights outright. That blocks a
documented Zonation v3+ workflow class — opportunity costs, threats, and
alternative land uses expressed as negatively weighted layers, so that cells
carrying them are *preferentially removed*. Two things stand in the way, and
they are different in kind:

1. **CAZ cannot express a negative term.** `rescore` builds a dense row
   (`rank_removal.py:328-331`) where absent features are structural zeros, then
   takes `max`. A negative term can never beat those zeros, so negative weights
   would be **silently inert** under `rule="caz"` — the worst failure mode,
   since nothing raises and the ranking merely ignores the user's intent.
   ABF's `sum` has no such problem: absent features contribute 0 to a sum
   regardless of the sign of anything else, so ABF is already correct.
2. **Negative weights break the warp=1 heap's exactness proof.** v0.32's lazy
   heap is exact because removal only *decreases* `Q_j`, making every term
   `w_j·q_ij/Q_j` nondecreasing, so cached keys are lower bounds
   (`rank_removal.py:124`). With `w_j < 0` and `q_ij > 0` the term becomes *more
   negative* as `Q_j` shrinks: scores can decrease, cached keys become upper
   bounds, and a lazily-popped fresh top is no longer the true argmin. This is
   not repairable lazily (unlike the FP-residue extinction case, which is a
   bounded, detectable event): a stale overestimate can hide the true minimum
   arbitrarily deep in the heap.

The batch path has no such dependency — its dirty-set tracks *whether a cell's
inputs changed*, not whether scores move monotonically — so it stays exact with
negative weights untouched.

## 2. Goals / non-goals

**Goals**

- Accept negative feature weights (finite, any sign) in `rank_removal` and
  therefore `ZonationSolver`; amounts remain `>= 0`.
- CAZ gains a meaningful negative-weight semantic: `delta_i = max_j` over the
  features cell `i` **holds and that are still extant**, floored to `0.0` when
  the cell holds nothing extant.
- **Exact backward compatibility for `w >= 0`**: bit-identical results for every
  existing input, proven by the untouched existing suite (including the
  test-local dense oracle).
- `warp=1` with negative weights auto-routes to batch selection (exact, slower)
  with one filterable `UserWarning`; no silent wrong answers, no hard failure.

**Non-goals**

- No change to ABF's formula (already correct), to curves, smoothing, locks,
  `ZonationResult`, or the solver's signature.
- No negative *amounts* (Zonation expresses negative features through weights;
  amounts are occurrence levels — science-verified at v0.31).
- No attempt to preserve the heap under negative weights (§1.2 — not lazily
  recoverable; the batch route is the answer).
- No UI exposure of weights (unchanged; weights are already a `rank_removal`
  parameter the Shiny panel does not surface).

## 3. CAZ semantics (normative)

Let `held_i` = the set of features with a stored (nonzero) amount in row `i`,
and `extant` = features with `Q_j > 0`.

```
delta_i = max_{j ∈ held_i ∩ extant} ( w_j · q_ij / Q_j ) / c_i      (nonempty)
delta_i = 0.0                                                        (empty set)
```

**Equivalence to the shipped formula for `w >= 0`** (the load-bearing claim):
every term is then `>= 0`, and the shipped code computes
`max({terms of held∩extant} ∪ {0 for absent} ∪ {0 for dead})`. If the new set is
nonempty its maximum is `>= 0`, so adding zeros cannot change it; if the new set
is empty the shipped max over all-zeros is `0.0`, which the floor reproduces
exactly. `max` is order-free and selects an existing float, so the surviving
value is **bitwise identical**. This is why the existing suite — including the
dense oracle, which keeps the *old* formula — is the compatibility proof and
must not be modified.

**Why "held" must come from the amount buffer, not from `r == 0`:** a *zero
weight* (legal, and already tested) makes a held term exactly `0.0`, which is
indistinguishable from an absent feature by value. With negatives present the
distinction is observable: for a cell holding one zero-weighted and one
negatively-weighted feature, including the `0` yields `delta = 0` while
excluding it yields the negative term. The mask is therefore `qd != 0` on the
dense amount buffer (valid because `eliminate_zeros()` guarantees stored
amounts are nonzero, and validation guarantees them positive).

**Extinct features are excluded, not zeroed.** A cell holding only extinct
features scores `0.0` (via the floor) — identical to today for `w >= 0`, and it
avoids an extinct feature masking a negative term at `max(0, negative) = 0`.
For **ABF this distinction does not exist**: excluding a term from a sum and
zeroing it are the same operation, which is why ABF keeps `r[:, dead] = 0.0`
unchanged under both weight regimes.

## 4. Implementation

All in `src/pymarxan/zonation/rank_removal.py`.

1. **`_validate_inputs`** (`:81-87`): delete the negative-weight raise; keep the
   finiteness check. One docstring sentence updated.
2. **`rescore`** (`:310-333`): bind the dense amount buffer
   (`qd = q[chunk].toarray()`; `r = qd * fac`) — a naming change only, values
   identical — and split the CAZ reduction:

   ```python
   if rule == "caz":
       if has_neg_w:
           r[qd == 0] = -np.inf     # absent: not held
           r[:, dead] = -np.inf     # extinct: no longer counts
           out = r.max(axis=1)
           out[out == -np.inf] = 0.0
       else:
           r[:, dead] = 0.0         # shipped path, untouched
           out = r.max(axis=1)
   else:
       r[:, dead] = 0.0
       out = r.sum(axis=1)          # ABF: correct for either sign
   ```

   `out == -np.inf` (not `~np.isfinite`): `+inf` is a legal score in the
   subnormal-`Q` regime that v0.32's `RuntimeError` guard owns, and must not be
   silently rewritten to `0.0`.
   Naming `qd` keeps two chunk buffers alive simultaneously (~26 MB each at
   `_RESCORE_CHUNK=32768`, `n_feat=100`) instead of one — accepted; the existing
   benches are the regression net.
3. **`has_neg_w`**: `bool((w < 0).any())`, computed once immediately after the
   `w` array is filled from the `weights` dict (before `rescore` is defined, so
   the closure captures it).
4. **Heap routing** (`:335`): `use_heap = warp == 1 and not _force_batch and
   not _force_full_rescore and not has_neg_w`. The warning fires from a
   module-level helper (mirroring `_warn_if_small_warp`) called immediately
   after the `warp` clamp, i.e. beside the existing advisory call, when
   `warp == 1 and has_neg_w`: negative weights make scores non-monotone, so the
   exact lazy heap is unavailable and batch selection is used — same results,
   slower. Filterable `UserWarning`.
5. **Docstring**: negative weights supported with the CAZ rule stated; the
   warp=1 routing consequence; the Moilanen 2011 citation.

## 5. What provably does not change

Weights enter *only* `fac = w / Q_safe` inside `rescore`. Therefore: `Q`, `T`,
and every performance-curve value are weight-independent; locks/phases,
`curve_every`, smoothing (applied to amounts before weighting), `top_fraction`,
`priority_rank`, and `ZonationResult` are untouched. Selection is sign-agnostic
(`argpartition`/`argsort` order any finite floats), and `use_cost` divides by a
validated-positive cost, so sign is preserved. The `RuntimeError` progress guard
is unaffected — negative weights produce finite scores, never NaN.
`ZonationSolver` needs no edit: it already forwards a `weights` dict
(`zonation_solver.py:45`), so it inherits the capability with the validation
change alone. Note the solver's `warp` **defaults to 1**
(`zonation_solver.py:44`), so every negative-weight `solve()` takes the batch
route and emits the routing warning — intended (the user should know the exact
heap is off), but it sharpens the §9 question about warning granularity.

## 6. Testing

Appended to `tests/pymarxan/zonation/test_rank_removal_scale.py`; the existing
suite is modified only where noted.

1. **Backward compatibility (no new test):** the entire existing suite green,
   unmodified — the dense oracle still encodes the *old* CAZ formula, so its
   continued agreement is the bitwise proof of §3's equivalence claim.
2. **Hand-computed semantics**, both rules, on one small fixture with a mixed
   positive/negative weight set: assert the exact `removal_order` derived by
   hand in the test's comment (a negatively-weighted cell must be removed
   before a neutral one).
3. **The formerly-inexpressible case:** a cell holding *only* a
   negatively-weighted feature must rank below (be removed before) a cell
   holding nothing — impossible under the shipped `max`-with-zeros formula.
4. **Zero-weight × negative-weight interaction** (§3): a cell holding one
   `w=0` feature and one `w<0` feature scores the negative term, not `0`.
5. **Extinct-feature exclusion:** a cell whose only held feature has gone
   extinct scores exactly `0.0`.
6. **Heap routing:** with a negative weight at `warp=1`, monkeypatch
   `heapq.heapify` (called once per lock-phase by the heap path, never by the
   batch path) to raise `AssertionError`; the run must complete normally —
   a positive, loud probe rather than counting pops. The `UserWarning` must
   fire; with all-nonnegative weights at `warp=1` it must not (and `heapify`
   must then be reached, confirming the probe has teeth).
7. **Self-consistency:** negative weights, `warp ∈ {1, 3, n}`, `use_cost` both,
   `_force_full_rescore` on/off — all agree (the dirty-set shortcut is still
   valid; only monotonicity was lost).
8. **Cost interaction:** negative delta divided by differing positive costs
   reorders as expected (hand-computed).
9. `make check` green; parity anchor untouched (no Marxan solver touched).

## 7. Performance

The negative branch adds two masked writes and one comparison per chunk versus
the shipped path, and is taken *only* when a negative weight exists — the
positive-weight hot path executes the same instructions as v0.33. The extra
chunk buffer (§4.2) is the one unconditional cost. Existing benches
(`bench_zonation.py`: warp-batch, warp=1 heap, grid-smoothed) are the regression
net; no new bench — this phase changes semantics, not scale.

## 8. Files touched

- `src/pymarxan/zonation/rank_removal.py` — validation, `rescore`, routing,
  warning helper, docstring.
- `tests/pymarxan/zonation/test_rank_removal_scale.py` — §6 tests (append).
- `CHANGELOG.md` — `[Unreleased]` Added + Changed → **v0.34.0**.
- Roadmap memory post-merge: negative weights done; edge removal remains.

## 9. Risks / open questions for review

- **Science (primary):** does Zonation itself admit negative weights under CAZ,
  or only in the additive/ABF framework (Moilanen 2011 is framed additively)?
  And does Zonation's CAZ maximize over held features or over all features?
  If CAZ+negative is not a real Zonation combination, the fallback is to raise
  for `rule="caz"` and ship ABF-only — the spec should be revised, not the code
  patched after the fact.
- **Grounding:** verify the §3 bitwise-equivalence claim by execution across the
  existing fixture families (positive weights, both rules, warps, locks,
  extinction, smoothing) — not by reading; and verify negative-weight results
  against an independently written brute-force scorer.
- Does any *other* consumer assume `delta >= 0`? (`analysis`, the Shiny panel,
  `ZonationSolver.build_solution` metadata — grounding to grep and confirm.)
- Is one warning per call the right granularity, or should it fire only once per
  process? (Precedent `_warn_if_small_warp` fires per call — but `ZonationSolver`
  defaults to `warp=1`, so negative-weight users hit this on every solve.)
- Should `ZonationSolver` gain a test asserting negative weights flow through
  end-to-end, or is the `rank_removal` coverage sufficient given it is a pure
  pass-through?
