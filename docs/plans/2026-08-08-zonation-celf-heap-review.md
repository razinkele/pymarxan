# CELF lazy-heap — multi-agent design review synthesis

**Date:** 2026-08-08 · **Run:** Workflow `wf_5de28a2d-584`, 4 lenses, ~471k tokens, 29 min.
**Verdicts:** architect approve-with-fixes · grounding **revise** · science approve-with-fixes · re-design approve-with-fixes.
**Empirical core:** grounding transcribed Tasks 1–3 verbatim and ran 216 bitwise matrix checks + a
240-run float sweep (heap==batch AND scratch-batch==real v0.31 engine, curves `check_exact`) — the
functional design is **confirmed sound**. The re-design lens independently derived a near-identical
architecture (heap, dirty+key freshness, phase-masked repair, kernel reuse) — strong mutual
validation. The science lens proof-walked §3 adversarially and pronounced it airtight, with three
lemmas worth recording. But two findings force real changes: a CRITICAL performance refutation and
a HIGH test-coverage refutation.

## Accepted findings and resolutions

| # | Sev | Lens | Finding | Resolution |
|---|-----|------|---------|-----------|
| 1 | **CRITICAL** | grounding | **Performance premise refuted.** Heap as designed is ~18× slower than batch at warp=1 (side=100: 47.8 s vs 2.7 s; side=300: batch 120.7 s, heap DNF >9 min). Cause: ~62 dirty-pop **single-row** rescores per accept, each paying ~72.5 µs scipy slice overhead (design §9 estimated 10–20 µs; vectorized cost is 1.5 µs/row). | **Buffered dirty pops** (grounding's empirically-projected fix): pop consecutive entries, buffering dirty ones (skipping removed) until a fresh top surfaces; rescore the deduplicated buffer in ONE vectorized `rescore()` call; push all back at new keys; re-evaluate. Selection semantics unchanged (accept still requires a fresh top; values bitwise-identical — same kernel, chunk-shape-independent), so the §7 contract survives. §9 rewritten to measured numbers; Task-4 bench budget stays 60 s but is flagged re-measure-first. |
| 2 | HIGH (3 lenses + grounding independently) | all | **Extinction fixture broken.** The tiny-amount residue carrier is the global argmin and leaves FIRST (order [3,1,2,5,4]) — crossing happens with zero holders, repair never fires, the fixture's own order assertion fails. Worse (science 8b, grounding): even a re-tuned fixture *exercises* the repair without *verifying* it, and the 30-seed float sweep produced **0 crossings in 120 runs** — the "statistical net" claim was empirically vacuous. | Fixture replaced with the science lens's **verified detector construction**: carrier cost 1e-15 (score dominates big holders until crossing) + detector PU6 (sole holder of f3, amount 1.0, cost 200) whose score 5e-3 sits strictly inside the carrier's stale-key/true-score gap — a missing/broken/phase-inverted repair provably flips the order ([1,2,6,3,5,4] vs [1,2,3,6,5,4], both rules, execution-verified). Sweep demoted to general coverage. |
| 3 | MED* | re-design | **No cross-phase extinction coverage** — the §5 phase-mask-skip + dirty-carry + phase-init-rescore argument had zero planned tests. | Added fixture: same construction with the big holders locked out (status 3), so the crossing happens in the locked-out phase while the carrier sits in the normal phase. |
| 4 | MED | science + re-design | **§7 "bitwise for all valid inputs" falsified**: `isfinite` guards raise where batch completes (verified: all-inf deltas from shared 1e-310 feature → batch returns [1,2]; inf+NaN co-held case). inf keys are heapq-safe — only NaN corrupts ordering. | **Guard NaN only** (`np.isnan` at the three sites): the all-inf class becomes bitwise-identical; contract scoped to "identical whenever no score evaluates to NaN; NaN-producing runs raise RuntimeError on the heap path where batch may limp to a NaN-ordered tail". The guard still moots the late-rescore inf-timing divergence the re-design lens found (loud raise, never a wrong order). |
| 5 | MED | architect + both others | `isinstance(curve_every, int)` rejects `np.integer` — the likeliest caller type for a raster memory knob. | `operator.index(curve_every)` in try/except → ValueError; `np.int64(7)` added to the accepted validation cases. |
| 6 | MED | architect | Weights never tested bitwise heap-vs-batch (w≠1 is a distinct rescore input; w=0 interacts with the dead-mask). | Weights case (incl. a weight-0 feature) added to the families test; stored-zero/duplicates/featureless noted as transitively covered via the oracle suite (+3 cheap direct lines added anyway per re-design). |
| 7 | MED | science | "CELF-style" alone misattributes: the lazy/accelerated greedy is **Minoux 1978** (doi:10.1007/BFb0006528); CELF (Leskovec et al. 2007, doi:10.1145/1281192.1281239) popularized it in a submodular-maximization frame that does not apply here (our claim is exactness with monotone lower bounds, not an approximation guarantee). | Docstring/CHANGELOG: "lazy-greedy selection (Minoux 1978; popularized as CELF, Leskovec et al. 2007), mirrored to minimization". |
| 8 | LOW | science | §3 proof stated in real arithmetic; the bitwise claim needs two FP lemmas (IEEE-754 correctly-rounded ops are weakly monotone; the per-row pairwise-summation tree is fixed for fixed n_feat) — both hold. Also: +inf preserves order; NaN is the only non-ordered value. | One-sentence lemma note added to §3. |
| 9 | LOW | science | "Exact" risks optimality misreading; de Mello 2015 is biome-wide (Cerrado), not country-wide (their warp=1 quote verified from full text). | Docstring gains "exact w.r.t. the greedy removal sequence; the ranking remains a heuristic prioritization"; §1 wording fixed. |
| 10 | LOW | architect + re-design | Assorted plan hardening: `_warn_if_small_warp` docstring goes stale (promised update had no task step); final-record check placement ambiguous after the Task-3 dispatch wrap; batch-path `curve_every>1` loses direct coverage post-Task-3; §2 "no warp>1 performance change" contradicted by remove_cell's O(nnz_row) crossed-computation; "must move indptr" is stylistic (closures late-bind); invariant comment needed at the rescore/discard sites (+ `assert not dirty[i]` before accept); all-ties fixture missing (sustained equal-key regime is the tie-break proof's stress case). | All folded: helper-docstring step added; "final-record stays below the whole if/else" sentence; thinning test parametrized over `_force_batch`; §2 reworded (behavioral no-op, O(nnz_row) bookkeeping accepted, existing warp bench is the regression net); softened wording; comments+assert added; all-ties fixture added (n_pu=30, one feature, equal amounts, `use_cost=False` → ascending-index order assertable directly). |

## Key confirmations (verified, do not re-litigate)

- **216/216 + 240/240 bitwise**: heap==batch and scratch==v0.31 across every fixture family incl.
  warp 2/3/4/7/80 neutrality, `use_cost=False`, n_pu=1, curve_every edges; array-built DataFrame
  `assert_frame_equal(check_exact=True)`-identical to the dict-built one; whole existing suite green
  with the transcribed engine swapped in-tree (89 zonation/solver + 6 shiny-panel tests).
- **§3 proof airtight** (science, adversarial walk): (i) every remaining phase cell always retains
  ≥1 entry with key == `delta[k]`; (ii) `delta[k]` ≤ true(k) under monotonicity+repair; (iii) the
  equal-key stale-competitor tie case forces a contradiction. Repair-before-next-pop sufficient;
  dirty-mark-then-repair ordering correct; FP-residue extinction confirmed the ONLY break
  (subnormal-positive Q explodes scores *upward* — order-preserving).
- The repair is **provably load-bearing**: grounding built a repair-disabled heap and produced a
  first-divergence-at-position-2 counterexample fixture.
- `delta` written only by `rescore` (three sites cited); advisory silent/warn truth table verified
  incl. boundary (1M, 2) warns; subnormal fixture raises on both paths in ms; preallocation bound
  never overflowed under instrumentation; `remove_cell` extraction faithful and mid-batch-reader-free.
- Structure verdicts: inline heap + if/else dispatch correct at two strategies (extraction trigger:
  a third selection mode, e.g. edge removal); `_force_batch` naming and `_force_full_rescore`→batch
  routing endorsed; empty-heap guard keep; per-push `float()`/`int()` exact.
- References (scite-verified, unretracted): Minoux 1978; Leskovec et al. 2007; de Mello et al. 2015
  (full-text warp=1 quote); Moilanen 2005/2007 (re-confirmed).

## Outcome

Design spec + implementation plan patched in place (same commit as this doc): buffered dirty-pop
rescore loop (#1), detector + cross-phase extinction fixtures (#2/#3), NaN-only guards + scoped §7
contract (#4), `operator.index` validation (#5), weights/all-ties/coverage additions (#6/#10),
Minoux attribution + wording (#7–#9). Execution may proceed; the Task-4 bench measures before any
budget/docstring perf claim ships.
