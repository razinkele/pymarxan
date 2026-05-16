# Phase 19 — Clumping (TARGET2 + CLUMPTYPE + Type-4 species)

**Date:** 2026-05-16 (revised post-review)
**Target version:** v0.2.0 (full Marxan-classic parity)
**Parent plan:** [`docs/plans/2026-05-16-realignment.md`](2026-05-16-realignment.md)
**Review notes:** [`2026-05-16-phase19-review.md`](2026-05-16-phase19-review.md) — multi-agent review found 4 Marxan-parity bugs and a HIGH-severity perf concern. All corrected below.
**Effort:** ~1.5 weeks
**Confidence:** 85 % (raised from 75 % post-review; the incremental `ClumpState` design removes the delta-perf risk that drove the original lower confidence)

## Why

Marxan v4's `clumping.cpp` lets users specify a **minimum patch size** per feature — the "type-4 species" formulation. Heavily used in habitat-fragmentation-aware planning where a feature isn't usefully represented unless its total in a contiguous patch exceeds some threshold (e.g. "this species needs at least 50 hectares of contiguous habitat").

Without TARGET2, a reserve that scatters 5 PUs of 10 hectares each across the landscape may technically meet a 50-hectare target but actually be ecologically useless to the species. TARGET2 says "no, you need 50 hectares **in one patch**."

pymarxan currently has no patch-size mechanism. Phase 19 closes the gap.

### Foundational references

Confirmed by reading the Marxan v4 source (`clumping.cpp::PartialPen4`, `clumping.cpp::ValueAdd`/`ValueRem`, `score_change.cpp::computeChangePenalty`):

- Ball, I. R., Possingham, H. P., & Watts, M. (2009). Marxan and Relatives: Software for Spatial Conservation Prioritization. In *Spatial Conservation Prioritization* (pp. 185–195). Oxford University Press. https://doi.org/10.1093/oso/9780199547760.003.0014
- Metcalfe, K., Vaughan, G., Vaz, S., & Smith, R. J. (2015). Spatial, socio-economic, and ecological implications of incorporating minimum size constraints in marine protected area network design. *Conservation Biology, 29*(6), 1615–1625. https://doi.org/10.1111/cobi.12571
- Game, E. T., & Grantham, H. S. (2008). *Marxan User Manual: For Marxan version 1.8.10* (v2 updated 2021 by Marxan Solutions, §5.2 + §5.46 on CLUMPTYPE). https://marxansolutions.org/wp-content/uploads/2021/02/Marxan-User-Manual_2021.pdf

## What's in scope

- **Per-feature TARGET2** — minimum amount of feature *j* required in any single contiguous clump for the clump to "count" toward the feature target.
- **Per-feature CLUMPTYPE** (0, 1, 2) — how clumps that fall below TARGET2 are scored:
  - `CLUMPTYPE = 0`: clump counts fully if its amount ≥ TARGET2, else 0 (binary).
  - `CLUMPTYPE = 1`: clump counts min(amount_in_clump, TARGET2) — capped contribution.
  - `CLUMPTYPE = 2`: clump counts proportionally (amount_in_clump if ≥ TARGET2; else 0). *To be confirmed against Marxan reference — see Risk R1 below.*
- **"Type-4 species"** semantics — features with TARGET2 > 0 use the clumping-adjusted amount instead of raw sum in the deterministic shortfall penalty.
- **SA / heuristic / iterative-improvement** native support via `ProblemCache` extension (component recomputation per delta — slower than non-clumping, but correct).
- **MIP** — TARGET2 is not native to MILP. Strategy: same shape as Phase 18's MIP — `mip_clump_strategy="drop"` default solves without TARGET2 and reports the post-hoc gap on `Solution.clump_shortfalls`. A future `"big_m"` strategy could add big-M flow constraints for exact clumping but is out of scope for v0.2.0.
- **I/O**: optional `target2` and `clumptype` columns on `spec.dat`. Defaults (0 / 0) disable clumping for the feature. Writers omit columns when all defaults.
- **Shiny UI**: extend the feature table to expose target2/clumptype edits; show clump-shortfall column in `target_met`.

## What's NOT in scope

- **MIP big-M / network-flow formulation** of TARGET2. Deferred — same logic as Phase 18 MIP for SOCP.
- **Per-zone TARGET2** in `ZonalProblem`. Marxan with Zones doesn't ship clumping either; deferred.
- **Combined PROBMODE 3 × TARGET2.** First-pass interaction: clumping uses raw amounts, probability uses expected amounts via separate path. Test that both penalties accumulate without crossing wires. Detailed scrutiny in the review agent's pass.

## Formulation (precise, Marxan-faithful)

Given:
- `amount_ij`, `target_j`, `spf_j` as today.
- `target2_j` (new, default 0) — minimum amount per contiguous clump for feature *j*. `target2 ≤ 0` disables clumping for *j*.
- `clumptype_j` (new, default 0) — partial-clump scoring rule. Values {0, 1, 2}.
- Adjacency: standard boundary adjacency from `bound.dat` (already in `ProblemCache`).

**Connected components** for feature *j* under selection `x`:

A PU `i` "participates in feature *j*" iff `x_i = 1` AND `amount_ij > 0`. Components are connected sub-graphs of participating PUs under the boundary adjacency. A PU lacking the feature does **not** bridge clumps (confirmed against `rtnClumpSpecAtPu` in Marxan source).

**Per-clump occupancy** for feature *j* in component *c*:

```
occ(c, j) = Σ_{i ∈ c} amount_ij
```

**Per-clump effective contribution** — `PartialPen4` from `clumping.cpp` (verified against source line-by-line):

| CLUMPTYPE | Marxan source | Effective contribution when `occ < target2_j` |
|---|---|---|
| 0 (default) | `return 0.0` | `0` |
| 1 ("nicer step") | `return amount / 2.0` | `occ / 2` |
| 2 ("graduated/quadratic") | `return amount / target2 * amount` | `occ² / target2` |

When `occ ≥ target2_j`, all three CLUMPTYPEs return the full `occ`.

**Type-4 feature effective amount**:

```
held_eff_j = Σ_c PartialPen4(occ(c, j), target2_j, clumptype_j)
```

When `target2_j ≤ 0`, `held_eff_j = held_j` (raw amount sum). The clumping path short-circuits to the existing deterministic shortfall path.

**Fractional shortfall** (Marxan-faithful, per scientific review §C3):

```
if target_j > 0:
    fractional_shortfall_j = max(0, (target_j · MISSLEVEL − held_eff_j) / target_j)
else:
    fractional_shortfall_j = 0
```

Two divergences from Marxan-strict, both deliberate and documented:

- **Marxan-faithful**: pymarxan adopts the Marxan-classic normalisation `(T − amount) / T` rather than raw `T − amount`. This makes Marxan-calibrated SPFs port directly into pymarxan (per user decision D1 in the review).
- **pymarxan extension over Marxan classic**: the `· MISSLEVEL` multiplier inside the live shortfall. Marxan classic uses MISSLEVEL only as a reporting cutoff in `mvbest`. pymarxan's existing deterministic-target path uses MISSLEVEL in-objective and Phase 19 stays consistent with that (per user decision D2 in the review). Documented in §Assumptions below.

**Penalty contribution** of feature *j* — matches `score_change.cpp::computeChangePenalty`:

```
penalty_j = baseline_penalty_j · SPF_j · fractional_shortfall_j
```

`baseline_penalty_j` is a per-feature constant precomputed once at `ProblemCache.from_problem` build time. The Marxan-classic convention is "cost to meet the feature target via a greedy heuristic" — i.e. the minimum SPU-cost required to assemble `target_j` units of feature *j* from the cheapest available PUs. Phase 19's implementation will use this convention.

**Total clumping penalty**: `Σ_j penalty_j` over features with `target2_j > 0`. Added to the objective alongside the standard penalty for non-type-4 features.

## API surface changes

```python
# New optional columns on features
problem.features["target2"]                  # float, default 0; >0 activates clumping
problem.features["clumptype"]                # int 0/1/2, default 0

# New module pymarxan.solvers.clumping
class ClumpState:
    """Mutable companion to ProblemCache for incremental clumping bookkeeping.
    Maintained alongside `held` and `total_cost` through the SA / iterative-
    improvement inner loop. Frozen ProblemCache stays frozen; mutation lives
    here so the SA outer loop can flip PUs cheaply.
    """
    @classmethod
    def from_selection(cls, cache, selected) -> "ClumpState": ...
    def delta_penalty(self, cache, idx, adding) -> float: ...
    def apply_flip(self, cache, idx, adding) -> None: ...
    def held_effective(self) -> np.ndarray: ...

def compute_clump_penalty_from_scratch(cache, selected) -> tuple[np.ndarray, float]:
    """Reference implementation used in tests and full-objective evaluation;
    O(edges + n_features × n_components). NOT used in the SA hot loop."""

def evaluate_solution_clumping(problem, solution) -> tuple[dict[int, float], float]:
    """Post-hoc evaluator for the MIP 'drop' strategy."""

# Solver.supports_clumping() -> bool  default True; MIP "drop" returns True too

# Solution gains optional attrs (mirrors Phase 18 prob_* fields)
Solution.clump_shortfalls: dict[int, float] | None    # per-feature fractional shortfall
Solution.clump_penalty: float | None                  # baseline_penalty · SPF · shortfall, summed

# ProblemCache gains
ProblemCache.feat_target2: np.ndarray                 # (n_feat,) float; ≤0 disables
ProblemCache.feat_clumptype: np.ndarray               # (n_feat,) int8
ProblemCache.feat_baseline_penalty: np.ndarray        # (n_feat,) float; precomputed
ProblemCache.clumping_active: bool                    # any(feat_target2 > 0); zero-cost gate
ProblemCache.feat_uses_pu: list[np.ndarray]           # j -> sorted PU indices with amount > 0
```

### Incremental ClumpState design (H1 from review)

The naive recompute (`scipy.sparse.csgraph.connected_components` on each flip) is O(n_pu + edges) per SA flip. For a 10k-PU / 1M-iteration run that's prohibitive. Adopting the independent re-design agent's design:

- **On flip-add(idx)**: gather neighbouring component ids in PU `idx`'s adjacency that participate in feature `j`. Union them into a single component (smallest id). Add `amount[idx, j]` to its occupancy. Update `held_eff[j]` by un-applying the old per-clump `PartialPen4` contributions and applying the new one. Cost: O(degree) per type-4 feature in PU `idx`.

- **On flip-remove(idx)**: a component may split. Mark `idx` unassigned, then run bounded BFS from each former neighbour to reassign component ids — bounded because BFS terminates as soon as it reattaches to a different surviving neighbour. Cost: O(touched-component-size) per type-4 feature; worst case O(component-size) but typically O(degree).

Result: SA delta under clumping ≈ O(degree²) per flip, comparable to the existing PROBMODE 0 delta (which is O(features_per_pu)). Realistic problems (1k–10k PUs) stay tractable.

## Data flow

```
read_spec()       → features (with optional target2, clumptype columns)
                       │
                       ▼
ConservationProblem ───┴──▶ Solver
                                │
                                ▼
              ProblemCache.from_problem()  [build-once]
                  - existing fields unchanged
                  - new: feat_target2, feat_clumptype, feat_uses_pu
                  - new: feat_baseline_penalty (precompute per Marxan)
                  - new: clumping_active = bool(any(feat_target2 > 0))

              ClumpState.from_selection(cache, selected)  [SA setup]
                  - one-time full build via scipy.csgraph
                  - returns mutable companion to ProblemCache

              SA / IterativeImprovement inner loop  [per flip]
                  - compute_delta_objective() returns deterministic delta
                  - if cache.clumping_active:
                      clump_delta = clump_state.delta_penalty(cache, idx, adding)
                      total_delta = deterministic_delta + clump_delta
                  - on accept: clump_state.apply_flip(cache, idx, adding)

              compute_full_objective(selected)  [verification / build_solution]
                  - existing path produces deterministic objective
                  - if cache.clumping_active:
                      + compute_clump_penalty_from_scratch(cache, selected)

              HeuristicSolver  [post-hoc clump reporting]
                  - scoring stays raw-amount-based (clumping-blind)
                  - build_solution populates Solution.clump_shortfalls

              MIPSolver "drop" strategy (default)
                  - solves deterministic problem ignoring target2
                  - build_solution evaluates clumping post-hoc

              build_solution(problem, selected, ...)
                  - existing PROBMODE 3 post-hoc populate continues
                  - if any feat has target2 > 0:
                      evaluate_solution_clumping(...) populates
                      Solution.clump_shortfalls + clump_penalty
                  - single source of truth — every solver path
                    inherits clumping reporting through here
```

## File format

`spec.dat` extended with two optional columns:

```csv
id,name,target,spf,target2,clumptype
1,coral_reef,100,1.0,50.0,0
2,seagrass,80,2.0,0,0      # clumping inactive (target2=0)
```

Both optional. Defaults: `target2=0`, `clumptype=0`. Writers omit columns when all-default.

## Test strategy

Same TDD-per-task as Phase 18. Test categories (post-review):

1. **Connected-component math** — known adjacency + selection → known components; cross-check with `scipy.sparse.csgraph.connected_components`.
2. **CLUMPTYPE 0 binary** — single clump above target2 counts full; below target2 counts 0.
3. **CLUMPTYPE 1 = amount/2** — *Marxan-faithful*. Sub-target clump at occ=30 with target2=50 contributes 15, NOT 30. Distinguishes from the v1 capped formula.
4. **CLUMPTYPE 2 = quadratic** — *Marxan-faithful*. Sub-target clump at occ=30 with target2=50 contributes `30²/50 = 18`, NOT 30 and NOT 0. Distinguishes from the v1 wrong formula AND from CLUMPTYPE 0.
5. **Multi-component aggregation** — 2 components each meeting target2 → sum; one meeting, one not → CLUMPTYPE-dependent partial.
6. **target2 ≤ 0 disables clumping** — feature returns identical penalty to non-clumping path; clump_shortfalls[j] = 0.
7. **Schema defaults** — spec without target2/clumptype → all zeros; non-clumping path unchanged.
8. **Round-trip I/O** — save_project + load_project preserves target2 and clumptype columns; defaults elide on write.
9. **Delta correctness under clumping** — `ClumpState.delta_penalty` + `apply_flip` produces same `held_eff` as a from-scratch full recompute, for 200 random flips on a 50-PU grid (mirrors Phase 18 cache delta-correctness test).
10. **MIP "drop" strategy** — under TARGET2, MIP returns deterministic solution and populates Solution.clump_shortfalls + clump_penalty.
11. **Combined PROBMODE 3 + clumping** — both penalties present, both correctly accumulate; per-feature routing asserted (clumping uses `held_eff`, PROBMODE 3 uses Bernoulli E[T_j]).
12. **Capability method** — `solver.supports_clumping()` returns True for SA/Heur/IterImpr/MIP (MIP via drop fallback).
13. **Baseline penalty precompute** — `feat_baseline_penalty[j]` equals manually-computed greedy cost on a constructed problem.
14. **Sepnum × target2 reject** — `target2 > 0 AND sepnum > 0` raises NotImplementedError.
15. **Byte-identical regression** — running a non-clumping (target2 all 0) project produces a Solution with byte-identical `selected` + `objective` to pre-Phase-19. Pinned via a golden fixture in `tests/data/simple/` (from Phase 1).

Target: +15 new tests, total ≥1167, coverage stays ≥91%.

## Risks (post-review)

- **R1 (RESOLVED) — CLUMPTYPE 1 vs 2 semantics.** Scientific agent quoted `PartialPen4` directly: CLUMPTYPE 1 = `amount/2`, CLUMPTYPE 2 = `amount²/target2` (quadratic). Formulation table updated above. The User Manual phrasing "proportional to size" for CLUMPTYPE 2 is misleading — source-of-truth is the C++.

- **R2 (RESOLVED) — Delta computation cost.** Naive recompute was the v1 risk; the incremental `ClumpState` design (architect + independent agent convergence) brings delta per-flip to ~O(degree²), comparable to existing solvers. No benchmark gate needed at acceptance.

- **R3 (MEDIUM) — PROBMODE 3 × clumping interaction.** Two penalty systems run additively. Per scientific agent: Marxan source treats them as independent shortfalls on `spec[isp].amount`. Phase 19 follows this: deterministic shortfall path uses `held_eff` (clumping-adjusted), PROBMODE 3 path uses raw Bernoulli E[T_j] (no clumping adjustment). Test plan item 11 pins per-feature routing.

- **R4 (LOW) — Connectivity vs boundary adjacency.** Clumping uses boundary adjacency only (matches Marxan `clumping.cpp` and pymarxan's `FeatureContiguityConstraint`). A user with `connectivity.dat` but no `bound.dat` will see every selected PU as its own clump. Document but do not auto-fall-back.

- **R5 (LOW) — Floating-point drift.** Incremental `ClumpState` adds and removes `PartialPen4` contributions per flip. Floating-point summation can drift over millions of flips. Mitigation: a periodic full-recompute sanity check (every 100k flips by default) catches drift early; verified by the delta-correctness test (item 9 in §Test strategy).

- **R6 (NEW, LOW) — Sepnum interaction.** Marxan's `NewPenalty4` also calls `computeSepPenalty(iseparation, sepnum)`. Phase 20 lands separation distance. Until then, `target2 > 0 AND sepnum > 0` raises `NotImplementedError`. Mechanical guard.

## Assumptions

1. **Boundary adjacency defines contiguity** for clumping (matches Marxan source).
2. **Self-boundary entries don't affect connectivity** — only between-PU edges count for clump membership.
3. **Feature presence is required for clump bridging**: a PU with `amount_ij = 0` does NOT participate in feature *j*'s clumps even if selected (Marxan's `rtnClumpSpecAtPu` convention).
4. **MIP under TARGET2 falls back to drop + post-hoc** (same shape as Phase 18 PROBMODE 3).
5. **Marxan-faithful shortfall normalisation** — pymarxan adopts the Marxan-classic `(T − amount) / T` fractional shortfall × `baseline_penalty` × `SPF` (D1 from review). Calibrated Marxan SPFs port directly.
6. **MISSLEVEL is a pymarxan extension over Marxan classic** — Marxan uses MISSLEVEL only as a reporting cutoff in `mvbest`; pymarxan's pre-existing deterministic path uses it in-objective (`compute_feature_shortfalls`). Phase 19 stays consistent with pymarxan's existing path (D2 from review). Documented here so future contributors aren't confused.
7. **PROBMODE × TARGET2 are additive, not multiplicative** — independent shortfalls accumulated through different paths.
8. **Baseline penalty** is a per-feature constant precomputed once in `ProblemCache.from_problem`. Implementation: cost to assemble `target_j` units of feature *j* from the cheapest available PUs via a greedy heuristic. Matches Marxan's `compute_penalties()` semantics.

## Acceptance criteria

1. ✅ All new code has TDD-per-task unit tests.
2. ✅ `make check` stays green.
3. ✅ Shiny exposes target2/clumptype edits in the feature table.
4. ✅ CHANGELOG.md `[Unreleased]` gains Phase 19 entry.
5. ✅ Test count grows; coverage ≥91 %.
6. ✅ Scientific-accuracy review confirms CLUMPTYPE 0/1/2 ordering against `clumping.cpp`.
