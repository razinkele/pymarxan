# Phase 19 — Multi-Agent Review Synthesis

**Date:** 2026-05-16
**Subject:** `2026-05-16-phase19-design.md` + `2026-05-16-phase19-implementation.md`
**Reviewers:** four subagents from disjoint angles (architect, codebase-grounding, scientific-accuracy, independent-re-design). Same pattern as Phase 18. Transcripts archived in agent task outputs.

## TL;DR

Phase 19's draft has **two classes of problem**:

1. **Marxan-parity errors that mirror Phase 18.** The scientific-accuracy agent read `clumping.cpp::PartialPen4` and `score_change.cpp::computeChangePenalty` and caught **four HIGH-severity formulation bugs** the draft would have shipped as silent semantic drift. The most damning: **CLUMPTYPE 1 in the draft is wrong** (draft says capped at target2; Marxan says half-amount), and **CLUMPTYPE 2 in the draft is identical to CLUMPTYPE 0** (Marxan's CLUMPTYPE 2 is quadratic `amount²/target2`). Plus the shortfall formula uses raw `max(0, T·MISSLEVEL − amount)` while Marxan uses normalised `(T − amount)/T` multiplied by a *baseline penalty* × SPF (not SPF alone). And MISSLEVEL is reporting-only in Marxan classic — pymarxan's existing penalty path uses it in-objective, which is a pre-existing pymarxan extension over Marxan, not a Phase 19 design choice.

2. **A real performance problem.** The architect agent flagged that the naive full-scipy-csgraph recompute is O(n_pu + edges) per SA flip — at 1M iterations on a 10k-PU problem, that's 50B operations just for clumping bookkeeping. The independent re-design agent independently proposed an **incremental `ClumpState` with union-find / bounded local BFS** as the cure. This is the headline architectural change Phase 19 needs.

Plus four smaller plan-mechanic errors caught by the codebase-grounding agent (Task 6 framing, ZoneMIPSolver __init__, feature_table guards, "four solvers" naming).

The plan is salvageable but needs a substantial revision before execution. Two of the issues require user input because they're design trade-offs, not bugs.

## Findings by severity

### CRITICAL (must fix before execution)

#### C1 — CLUMPTYPE 1 formula is wrong

**Source:** Scientific agent, `clumping.cpp::PartialPen4`:
```cpp
case 1: return amount / 2.0;             // "nicer step function"
```
**Draft says:** `min(amount_in_clump, target2)` (capped contribution).
**Marxan classic actually:** `amount / 2.0` (sub-threshold clumps contribute *half* their amount).

User Manual independently confirms: *"1. Partial clumps count half — Clumps smaller than the target score half their amount"*.

**Fix:** replace the CLUMPTYPE 1 row in the design's formulation table. Trivial code change once the formula is right.

#### C2 — CLUMPTYPE 2 formula is wrong (and duplicates CLUMPTYPE 0)

**Source:** Scientific agent, same file:
```cpp
case 2: if (spec[isp].target2)
            return (amount / spec[isp].target2 * amount);  // amount²/target2
```
**Draft says:** "amount if ≥ target2 else 0" — which is literally the same as CLUMPTYPE 0.
**Marxan classic actually:** *quadratic* `amount² / target2` for sub-target clumps. The User Manual phrasing "Score is proportional to the size of the clump" is misleading — the C++ is quadratic, not linear.

**Fix:** replace the CLUMPTYPE 2 row with `amount² / target2 if amount < target2 else amount`.

#### C3 — Shortfall is normalised in Marxan; draft uses raw

**Source:** Scientific agent, `clumping.cpp::ValueAdd`/`ValueRem`:
```cpp
shortfall = amount >= spec[isp].target
            ? 0
            : (spec[isp].target - amount) / spec[isp].target;
```
**Draft says:** `shortfall_j = max(0, target_j · MISSLEVEL − effective_amount_j)` — raw amount.
**Marxan classic actually:** `(target − amount) / target` — fractional in [0, 1].

The penalty is then assembled in `score_change.cpp::computeChangePenalty`:
```cpp
penalty += spec[isp].penalty * spec[isp].spf * (newamount - fractionAmount);
```
where `spec.penalty` is the **pre-computed baseline penalty** (cost to meet target via a greedy heuristic, calculated once at problem load). pymarxan's existing penalty path does NOT use a baseline-penalty multiplier — it's raw `SPF · shortfall`.

**Design call required (D1, below).** This is a calibration-affecting divergence. Options:
- **Align to Marxan**: add a baseline-penalty computation. Calibrated SPFs from existing Marxan studies would then work directly in pymarxan.
- **Keep pymarxan's existing convention**: raw `SPF · shortfall_raw`. Simpler; users have to recalibrate SPFs if they're porting Marxan projects.

The Phase 18 PROBMODE 3 path normalises by `ptarget`, an analogous Marxan-faithful choice. Consistency argues for aligning Phase 19 to Marxan too.

#### C4 — MISSLEVEL placement diverges from Marxan classic

**Source:** Scientific agent, grep of `clumping.cpp` and `computation.hpp` confirms MISSLEVEL is referenced **only in reporting** (`mvbest` "target met?" boolean), not inside the optimisation penalty.

**Draft says:** `target_j · MISSLEVEL − effective_amount_j` (MISSLEVEL inside the live objective).
**Marxan classic actually:** MISSLEVEL is reporting-only. The live penalty uses raw `target`.

**Pre-existing context:** pymarxan's existing deterministic penalty path (introduced before Phase 19, present in `compute_feature_shortfalls`) already uses MISSLEVEL in-objective. That's a **pre-existing pymarxan extension over Marxan classic**, not a Phase 19 invention.

**Design call required (D2, below).** Options:
- **Align Phase 19 with pymarxan's existing extension**: keep `MISSLEVEL` in the in-objective clumping shortfall, consistent with the rest of pymarxan. Document the divergence from Marxan classic in the design doc.
- **Align Phase 19 with Marxan strict**: drop MISSLEVEL from the clumping path. But then pymarxan's deterministic + clumping paths use *different* target-shortfall conventions — confusing and inconsistent.

Synthesis recommendation: keep MISSLEVEL in the clumping path (option 1) for internal consistency, but **document explicitly** in §"Assumptions" that this is a pymarxan extension over Marxan classic.

### HIGH (real bugs / real perf problems)

#### H1 — Delta computation too slow at scale (architect + independent re-design)

**Source:** Architect agent: naive `connected_components` per flip is O(n_pu + edges). 1M iterations × 10k PUs × 40k edges = ~50B ops just for clumping bookkeeping. The "only when has_any_clumping" gate doesn't help when clumping IS active.

The independent re-design agent went further and **proposed the cure**: an incremental `ClumpState` with union-find on add (`O(α)`) and bounded local BFS on remove. Mutable companion to the frozen `ProblemCache` — passed alongside `held` and `total_cost` through the SA loop.

**Fix:** adopt the independent agent's `ClumpState` design. Specifically:
- New `pymarxan/solvers/clumping.py::ClumpState` class.
- `ClumpState.from_selection(cache, selected) -> ClumpState` — initial O(edges + n_features × n_components) full build.
- `ClumpState.delta_penalty(cache, idx, adding) -> float` — incremental O(degree²) per flip via union-find / bounded BFS.
- `ClumpState.apply_flip(cache, idx, adding) -> None` — commits the flip to the in-memory state.
- SA / iterative-improvement maintain a `ClumpState` alongside `held`.

This makes the delta cost roughly **O(degree²)** per flip — for a grid that's O(constant). Acceptable for 1M-iteration SA runs.

#### H2 — Heuristic is clumping-blind during scoring (architect)

**Source:** Architect agent. `HeuristicSolver._score_pu` uses raw feature amounts; it has no concept of adjacency or clumping state. Under TARGET2, the heuristic will pick PUs that meet raw targets but badly miss clumping targets — and the post-hoc reporting will just show the user a bad solution.

**Fix options:**
- **A (full fix)**: refactor `_score_pu` to be clumping-aware for type-4 features. Significant work.
- **B (honest scope)**: re-title Task 9 as "HeuristicSolver: post-hoc clump reporting (scoring is clump-blind)" and document the limitation in `help_content.py`. Same shape as Phase 18's MIP "drop" strategy — the heuristic's contract under TARGET2 is "report the gap, don't optimise against it."

Synthesis recommendation: **option B for Phase 19**. A future phase can add a clumping-aware heuristic scorer. The honest title prevents users from expecting more.

#### H3 — `compute_delta_objective` signature trap (architect)

**Source:** Architect agent. Phase 18's PROBMODE 3 delta is per-PU decomposable; clumping is NOT (removing a PU can split a component). The delta requires constructing `selected_after = selected.copy(); selected_after[idx] = not selected[idx]` and computing components twice. The draft doesn't spell this out; an implementer following the PROBMODE 3 pattern would hit aliasing bugs.

**Fix:** explicit guidance in Task 8 of the impl plan. Or — better — adopt H1's incremental `ClumpState`, which sidesteps the issue entirely (the state is updated, not recomputed before/after).

#### H4 — Plan-mechanic errors (codebase agent)

Four:
1. **Task 6 framing**: `compute_penalty` / `compute_objective_terms` in `solvers/utils.py` only covers the `build_solution` / post-hoc path. The SA/II inner loop is via `ProblemCache.compute_full_objective` / `compute_delta_objective`. Task 8 covers this but Task 6's framing implies utils.py is sufficient.
2. **ZoneMIPSolver `__init__`**: does not currently exist; adding `mip_clump_strategy` requires creating an `__init__` from scratch, not editing one.
3. **feature_table**: extending columns requires touching `_COLUMN_ORDER` AND `validate_feature_edit` whitelist (currently `("target", "spf")`). Plan only mentions adding the columns.
4. **"All four solvers"**: explicit list — SA, MIP, Heuristic, IterativeImprovement.

All four are trivial fixes; flagged for completeness.

### MEDIUM

#### M1 — `Solution` is becoming a grab-bag of optional fields (architect)

`prob_shortfalls`, `prob_penalty`, now `clump_shortfalls`, `clump_penalty`. Phase 21 will add portfolio fields. A structured `SolutionMetrics` group is cleaner but not blocking. Flag for v0.3.

#### M2 — `compute_feature_components` return type inefficient (architect)

Dict-of-list-of-set is allocation-heavy in the SA hot loop. The independent agent's `ClumpState` approach uses NumPy integer arrays for component labels — cleaner and faster. Adopt the array-based representation if H1 is accepted (which it should be).

#### M3 — PROBMODE 3 × clumping test routing (architect)

Test item 11 ("both penalties present") should explicitly assert which feature's penalty flows through which path, not just aggregate non-zero correctness. Tactical change to the test plan.

#### M4 — SPF placement and baseline penalty (scientific)

Marxan's penalty: `spec.penalty (baseline) · spec.spf · shortfall_fraction`. Draft: `SPF · shortfall`. The `spec.penalty` baseline is the pre-computed cost to meet target via a greedy heuristic. **Tied to C3 above** — if D1 picks Marxan-strict, this needs the baseline-penalty machinery. If D1 picks pymarxan-convention, this is moot.

#### M5 — Sepnum coupling (scientific)

`NewPenalty4` also calls `computeSepPenalty` when `sepnum > 0`. Phase 19 declares separation out of scope, so `target2 > 0 AND sepnum > 0` should explicitly raise (or warn) until Phase 20 lands separation. Mechanical.

### LOW

- Adjacency reuse, clump-independence assumptions: design is correct.
- PROBMODE 3 × TARGET2 ordering: additive, independent. Document.
- Citations: scientific agent provided Ball-Possingham-Watts 2009 (foundational chapter) and Metcalfe 2015 (applied paper). Fold in.

## Comparison with independent re-design

The independent agent's design **converges** with the architect's perf concern: both propose `ClumpState` with incremental updates rather than full recompute. The agreement on this point is a strong signal.

The independent agent also proposes:
- **`mip_clump_strategy="bigM"`** as an exact MIP path via network-flow constraints. Out of scope for v0.2 per the design (correctly); flag for a future phase.
- **15 test cases including a regression guard** that `clumping_active == False` produces byte-identical results to pre-Phase-19. Worth adopting.

## Two open design calls for the user

These two decisions affect significant chunks of the implementation. Recommend asking before patching.

**D1 — Shortfall normalisation + SPF placement (C3, M4):**
- (a) **Align to Marxan strict**: fractional `(T−a)/T` × baseline_penalty × SPF. Calibrated Marxan SPFs port directly. Adds baseline-penalty precomputation.
- (b) **Keep pymarxan convention**: raw `max(0, T·MISSLEVEL − a)` × SPF. Simpler; consistent with existing deterministic path; users porting Marxan projects need to recalibrate SPFs.

Synthesis recommendation: (a) for Phase 19 — matches Phase 18's "Marxan-faithful" precedent and makes Marxan-classic project import meaningful. Baseline-penalty precomputation lives in `ProblemCache.from_problem`.

**D2 — MISSLEVEL placement (C4):**
- (c) **Keep pymarxan-extension**: MISSLEVEL inside the in-objective shortfall (matches pymarxan's existing deterministic path). Document the divergence from Marxan classic.
- (d) **Marxan strict**: MISSLEVEL is reporting-only. Pymarxan's existing path stays as-is; clumping path follows Marxan strict — inconsistent within pymarxan.

Synthesis recommendation: (c). Internal consistency outweighs Marxan parity here; document clearly.

## Recommended next actions

1. **Ask user** about D1 and D2 (two-option questions).
2. **Patch the design doc** with C1-C4 corrections, H1 (ClumpState incremental design), H2 (honest heuristic title), H3 (delta signature guidance), citations, sepnum-target2 reject, baseline-penalty if D1=(a), MISSLEVEL note if D2=(c).
3. **Patch the implementation plan** with H4 mechanical fixes (Task 6 framing, ZoneMIPSolver __init__, feature_table guards, four-solver naming), the ClumpState batches, and the byte-identical regression test.
4. **Defer execution** until the user confirms the patched plan.

## What this review cost / saved

~10 minutes of agent time. Caught **4 HIGH-severity Marxan-parity bugs** (CLUMPTYPE 1, CLUMPTYPE 2, shortfall normalisation, MISSLEVEL placement) and a **HIGH-severity perf concern** (delta cost at scale) that the v1 plan didn't address. Independent re-design agent contributed the `ClumpState` incremental design that cures the perf issue. Net: an estimated 2-4 days of "execute, discover wrong formula, revise" rework prevented.

---
*Synthesis written by the main session based on four subagent transcripts. Transcripts live in `/tmp/claude-1000/.../tasks/` and are referenced by ID in the source-control commit message.*
