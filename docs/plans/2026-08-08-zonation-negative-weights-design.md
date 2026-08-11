# Negative-weight (opportunity-cost) features — design spec

**Date:** 2026-08-08
**Status:** revised after multi-agent design review (`wf_255edcca-6b7`, all four lenses
`revise`); synthesis in `2026-08-08-zonation-negative-weights-review.md`.
**Provenance:** the deferral named in `_validate_inputs`'s own error message
("negative weights, used by Zonation v3+ for opportunity-cost features, are not
yet supported"), logged at v0.31.0.

> **What the review changed.** The first draft supported negative weights under
> both rules via a restricted CAZ `max`. The science lens showed that
> combination is structurally degenerate (§2) and that our ABF marginal is
> directionally opposite to the paper cited as warrant (§4); the architect lens
> found the draft's `-inf` sentinel inverted orderings. The revision **ships ABF
> only, with no kernel change at all** — strictly smaller, and positive-weight
> behavior becomes untouched rather than provably-equivalent.

## 1. Problem

`rank_removal` rejects negative feature weights outright. That blocks a
documented Zonation v3+ workflow: opportunity costs, threats, and alternative
land uses expressed as negatively weighted layers so that cells carrying them
are *preferentially removed* (Moilanen et al. 2011, doi:10.1890/10-1865.1 —
"negatively weighted features can be taken to represent multiple (opportunity)
costs", made available "in the Zonation v.3 software").

## 2. Why ABF only

Under ABF the score is a **sum**, so a negative term genuinely trades off
against positive ones — and absent features contribute exactly `0` to a sum
regardless of any sign. The shipped kernel is therefore already correct for
negative weights: **no code change to `rescore` at all.**

Under CAZ the score is a **max**, which is not a trade-off operator. Whenever a
cell holds any positively weighted feature whose term exceeds the negative term,
the negative term is discarded entirely: a cell holding `q_benefit = 0.1` and
`q_threat = 1e6` scores `+0.0100`, bitwise identical to a cell carrying no
threat (review-verified by execution). The threat's magnitude is irrelevant. A
restricted "max over held-and-extant features" — the first draft's design — does
not fix this; it only moves the silent inertness onto *mixed* cells, which is
exactly where an opportunity-cost layer bites (Moilanen 2011's own case study
uses near-ubiquitous agriculture and urban-potential layers). Moilanen 2011
places CAZ inside its multi-criterion Eq. 3 but every worked negative-feature
mechanism in the paper is ABF, "which aggregates value by direct summation
across features" — summation is the mechanism that makes the trade-off
expressible, and CAZ's `max` has no benefit function to invert.

**Therefore: `rule="caz"` with any negative weight raises `ValueError`.**
Positive-weight CAZ is untouched — a stronger backward-compatibility guarantee
than the first draft's bitwise-equivalence argument, since no CAZ code changes.

## 3. Why `use_cost=True` is rejected alongside negative weights

`delta/c` is faithful for positive features (Moilanen et al. 2005,
doi:10.1098/rspb.2005.3164 Eq. 2.1), but benefit-cost ratio ranking is
order-consistent only while the numerator keeps one sign. With `delta = -2` at
costs 1 / 2 / 10 the scores are `-2.0 / -1.0 / -0.2`: the **costlier**
threat-carrying cell is removed **later**. The conservation reading is
unambiguous — a cell that both carries a threat and is expensive to protect is
the one to concede first — so the formula does the opposite of what a user
would expect, and the composition has no published basis: Moilanen 2011's stated
purpose for negative features is to *replace* the single-cost divisor ("different
costs can be accounted for in our framework", the gain being that negative
features carry *multiple* costs where `c_i` carries one).

**Therefore: `use_cost=True` with any negative weight raises `ValueError`**,
directing the user to express costs as negative features (the paper's own
practice) or to pass `use_cost=False`. This is the conservative direction:
relaxing it later is backward-compatible, tightening it later would not be.

## 4. The semantic we ship, stated honestly

Our ABF marginal is `w_j · q_ij / Q_j`, proportional to `1/R_j` — the
proportional / remaining-sum member of the additive-benefit family that
pymarxan has used since v0.13. For a **negative** feature this makes exclusion
*increasingly* urgent as the feature nears elimination (`|marginal|` runs
1 → 100 as `R` falls 1.0 → 0.01).

Moilanen 2011 does the **opposite**, deliberately: it inverts the benefit
function for negative features (`z_k = 1/0.25 = 4` against `z_j = 0.25` for
biodiversity) so they become *decreasingly* important to exclude once mostly
excluded — "implementing the notion of generalized complementarity". Their
marginal runs 4.0 → 4e-6 over the same range. **Removal orders differ
materially.**

This deviation is *not* covered by pymarxan's existing documented departure from
`z ≈ 0.25`, because that departure preserves direction for positive features
(both forms increase as `R` falls; only steepness differs) and **changes kind at
the sign boundary**. Nor is it covered by the grid-smoothing precedent of
shipping a documented variant: that was licensed by proven ranking-*inertness*,
whereas this is ranking-*active*.

So the docstring and CHANGELOG cite Moilanen 2011 for the **concept only**
(negative weights represent opportunity costs / alternative land uses, and their
carriers should be removed first) and state the divergence plainly. The faithful
form needs a per-feature benefit exponent, which attaches to the docstring's
pre-existing deferral ("the concave power-benefit generalization is a future
extension") and is logged in the roadmap — not attempted here.

## 5. What changes

All in `src/pymarxan/zonation/rank_removal.py`; **`rescore` is not touched.**

1. **`_validate_inputs`** (`:81-87`): delete the negative-weight raise; keep the
   finiteness check; update the docstring sentence.
2. **Two new guards** in `rank_removal`, after the `w` array is filled (so they
   see weights aligned to actual features, not raw dict values keyed to absent
   ids): `has_neg_w = bool((w < 0).any())`; then
   - `rule == "caz" and has_neg_w` → `ValueError` naming `rule="abf"` and the
     reason (`max` cannot trade off);
   - `use_cost and has_neg_w` → `ValueError` naming both escapes.
3. **Heap routing** (`:335`): `use_heap = ... and not has_neg_w`, plus a
   filterable `UserWarning` from a module-level helper called beside
   `_warn_if_small_warp`. Review-verified as correct and necessary: with
   `w_j < 0` the term is strictly *decreasing* under removal, so cached heap keys
   become upper bounds and the Minoux-1978 lazy-greedy exactness argument fails.
   Batch selection is unaffected — its dirty set tracks changed *inputs*, never
   monotone scores.
4. **Docs**: the §4 divergence; the curve-reading inversion (§6); a `ValueError`
   summary line.
5. **`ZonationSolver`**: no signature change (it already forwards `weights`), but
   its metadata gains `negative_weight_features` — the ids given negative
   weights — following the existing `smoothed` / `smoothing_alpha` provenance
   precedent, so curve consumers can flip the affected series.

## 6. Curve-reading inversion (new failure mode)

`performance_curves` stores `Q/T` per feature. For a positive feature that reads
"fraction of the feature retained" — higher is better, the convention every
consumer and the Shiny panel's plot assumes. For a **negative** feature the
identical number means "fraction of the threat still inside the reserve", where
**lower is better**. Curves are weight-independent (§7), so nothing in the
output distinguishes them: a mixed run yields a plot whose rows must be read in
opposite directions. Before this change the uniform reading was always valid.
Mitigated by documenting it and by the §5.5 metadata marker.

## 7. What provably does not change

Weights enter *only* `fac = w / Q_safe` inside `rescore`, which this phase does
not modify. So `Q`, `T`, and every curve **value** are weight-independent;
locks/phases, `curve_every`, smoothing, `top_fraction`, `priority_rank` and
`ZonationResult` are untouched; selection is sign-agnostic. The `RuntimeError`
progress guard still fires for ABF with negative weights (review-verified on the
pinned subnormal fixture) — the `0.0 * inf → NaN` mechanism is unchanged.
Note negative terms are unbounded *below* as `Q → 0` (`Q=1e-9` → `-1e9`), a
magnitude regime the guard has only met from the positive side; §8.5 covers it.

## 8. Testing

Appended to `tests/pymarxan/zonation/test_rank_removal_scale.py`, plus one
**edit** to an existing test.

1. **Existing test edit (review finding #6):** `test_negative_or_nan_weight_raises`
   (`:421-425`) asserts the raise being deleted. Its NaN half stays; its negative
   half is replaced by assertions on the two new guards.
2. **ABF semantics, hand-computed** (fixture arithmetic verified by the review):
   PU1 benefit-10, PU2 threat-10, PU3 holds both at 5, PU4 empty, `use_cost=False`,
   weights `{1: +1, 2: -1}` → `[2, 3, 4, 1]`. PU3's score flips sign as `Q₂`
   shrinks (0.0 → −0.667), a concrete demonstration of the non-monotonicity that
   makes the heap unusable.
3. **A cell holding only a threat** ranks before a cell holding nothing.
4. **Guards:** CAZ + negative → `ValueError`; `use_cost=True` + negative →
   `ValueError`; both messages assert-matched. Positive-weight CAZ and
   `use_cost=True` still work (no regression).
5. **Heap routing:** monkeypatch `heapq.heapify` (called once per lock-phase by
   the heap path, never by the batch path) to raise; a negative-weight `warp=1`
   run must complete and must emit the `UserWarning`; a positive-weight `warp=1`
   run must reach `heapify` (proving the probe has teeth); no warning without
   negative weights. Plus the near-extinction magnitude regime (§7) against the
   progress guard.
6. **Self-consistency** at fixed warp: default == `_force_batch=True` ==
   `_force_full_rescore=True` (only monotonicity was lost; the dirty-set
   shortcut and batch selection remain valid).
7. **Solver:** `ZonationSolver` end-to-end with negative weights
   (`use_cost=False`, `rule="abf"`), asserting the `negative_weight_features`
   metadata marker.
8. **The whole existing suite green** — with `rescore` untouched, positive-weight
   behavior is unchanged by construction, not by argument.
9. `make check` green; parity anchor untouched.

## 9. Files touched

- `src/pymarxan/zonation/rank_removal.py` — validation, two guards, routing,
  warning helper, docstrings.
- `src/pymarxan/solvers/zonation_solver.py` — metadata marker only.
- `tests/pymarxan/zonation/test_rank_removal_scale.py` — §8 (append + one edit).
- `CHANGELOG.md` — `[Unreleased]` → **v0.34.0**.
- Roadmap memory post-merge: negative weights shipped ABF-only; **new deferral:
  per-feature benefit exponent (`z_j`) for Moilanen-2011-faithful negative-feature
  dynamics**, joined to the existing power-benefit deferral. Edge removal remains.

## 10. Review outcome

Reviewed by `wf_255edcca-6b7`; all findings folded (synthesis doc). Verified and
not to be re-litigated: heap bypass correctness; removal-direction correctness;
the "Zonation v3+" attribution; that nothing downstream assumes `delta >= 0`;
that warning granularity is a non-issue (Python's default filter shows it once
per call site per process). The first draft's `-inf` sentinel defect is recorded
in the synthesis as the reason a value-sentinel must never be used for masking in
this engine, should a later phase add a masked reduction.
