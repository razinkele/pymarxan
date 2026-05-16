# Phase 18 — Probability Completion (PROB2D + PTARGET + Z-score)

**Date:** 2026-05-16 (revised post-review)
**Target version:** v0.2.0 (Marxan-classic parity)
**Parent plan:** [`docs/plans/2026-05-16-realignment.md`](2026-05-16-realignment.md)
**Review notes:** [`2026-05-16-phase18-review.md`](2026-05-16-phase18-review.md) — multi-agent review found three Marxan-parity bugs in the v1 formulation. Corrected below.
**Effort:** ~1 week
**Confidence:** 85 % (raised from 80 % post-review)

## Why

pymarxan already implements Marxan's two simpler probability modes:

| `PROBMODE` | What it does | Where |
|---|---|---|
| 1 | Risk premium: adds `PROBABILITYWEIGHTING · Σ prob_i · cost_i · x_i` to objective | `solvers/utils.py::compute_probability_penalty` |
| 2 | Persistence-adjusted amounts: replaces `amount_ij` with `amount_ij · (1 − prob_i)` | `mip_solver.py:183-220`, `heuristic.py:194-244` |

Both modes are **per-PU only** — every feature at PU *i* shares the same probability `prob_i`. This is Marxan's "PROB1D" format.

What's still missing — and what Phase 18 ships — is the **PROB2D + Z-score** formulation that climate-resilience and uncertain-suitability studies actually use, faithful to Marxan v4's `probability.cpp::computeProbMeasures`:

1. **Per-(PU, feature) Bernoulli probability** `p_ij` (carried as an optional `prob` column on `puvspr.dat`, file-compatible with Marxan). Variance is *derived* internally as `σ²_ij = amount_ij² · p_ij · (1 − p_ij)`.
2. A **probability target** `ptarget_j ∈ (0, 1)` per feature — "feature *j* must be represented in the reserve with probability ≥ ptarget_j". Default `−1` means "no probability target" (Marxan's convention).
3. **Z-score evaluation** (Marxan-faithful): `Z_j = (target_j − E[T_j]) / √Var[T_j]`. The reserve meets feature *j* with probability `P_j = probZUT(Z_j) = 1 − Φ(Z_j)` (upper-tail standard normal). Assumes independence across PUs and a CLT normal-approximation to the sum, both standard Marxan assumptions (Game 2008, Tulloch 2013).
4. **Normalised shortfall penalty**: when `P_j < ptarget_j`, penalty `SPF_j · (ptarget_j − P_j) / ptarget_j`, summed across features and scaled by `PROBABILITYWEIGHTING` (γ). Marxan uses the normalised form, not raw subtraction.

Without this, pymarxan can't reproduce results from papers like Carvalho et al. 2011 (climate-uncertainty) or Tulloch et al. 2013 (habitat-data uncertainty), and can't claim "full Marxan-classic parity" — the headline goal of v0.2.0.

### Foundational references

- Game, E. T., Watts, M. E., Wooldridge, S., & Possingham, H. P. (2008). Planning for persistence in marine reserves: A question of catastrophic importance. *Ecological Applications, 18*(3), 670–680. https://doi.org/10.1890/07-1027.1
- Tulloch, V. J., Possingham, H. P., Jupiter, S. D., Roelfsema, C., Tulloch, A. I. T., & Klein, C. J. (2013). Incorporating uncertainty associated with habitat data in marine reserve design. *Biological Conservation, 162*, 41–51. https://doi.org/10.1016/j.biocon.2013.01.003
- Carvalho, S. B., Brito, J. C., Crespo, E. G., Watts, M. E., & Possingham, H. P. (2011). Conservation planning under climate change: Toward accounting for uncertainty in predicted species distributions to increase confidence in conservation investments in space and time. *Biological Conservation, 144*(7), 2020–2030. https://doi.org/10.1016/j.biocon.2011.04.024

## What's in scope

- **`PROBMODE = 3`** (new) — Z-score / chance-constrained mode.
- **PROB2D file format** — `puvspr` extended with an optional `variance` column.
- **PTARGET column** — `features` extended with an optional `prob_target` column.
- **Z-score penalty** in `compute_probability_penalty` (renamed or split — see §4).
- **SA solver path** via `ProblemCache.compute_zscore_penalty` and incremental variance bookkeeping on flip.
- **Heuristic solver path** — straightforward, no caching needed.
- **Iterative improvement path** — uses the same penalty function.
- **I/O readers + writers** for PROB2D and PTARGET in Marxan-tabular format.
- **Tests**: 6–8 new tests covering edge cases (zero variance, all-locked, single PU, infeasible target, MISSLEVEL interaction).
- **Shiny UI**: one new field in the Solver tab to toggle PROBMODE 3 and display Z-scores per feature in the Results tab.

## What's NOT in scope (decided)

- **Per-zone PROB2D.** Marxan with Zones doesn't ship Z-score support either; deferred until users ask.
- **PROBMODE 1 + 3 combination.** Mode 3 supersedes Mode 1's risk-premium logic.
- **General per-cell variance (decoupled from Bernoulli model).** Marxan classic only supports the Bernoulli formulation; pymarxan ships parity. A future PROBMODE 4 could accept a free-form `variance` column for non-Bernoulli uncertainty models, but that diverges from Marxan and is not in this phase.

## MIP under PROBMODE 3 — strategy (decided)

The Z-score constraint `√(Σ σ²_ij · x_i) ≤ k` is SOCP/QCP territory; CBC can't handle it. Three viable strategies; Phase 18 ships strategy **(a)**:

(a) **`drop` (Phase 18 default)** — MIP solves the deterministic problem (probability term excluded from the objective). The returned `Solution` is then evaluated post-hoc under PROBMODE 3 and the Z-score / probability gap per feature is populated on `Solution.prob_shortfalls`. Users see the gap but can still use MIP for a fast deterministic baseline.

(b) **`piecewise`** — tangent-line approximation of `√Σ σ²` in CBC. Adds ~K constraints per feature. Deferred to a possible Phase 18.5 if users request it.

(c) **`socp`** — exact SOCP formulation via Gurobi/CPLEX. Deferred to Phase 21 (which adds those backends).

The `MIPSolver.__init__` gains a `mip_chance_strategy: Literal["drop", "piecewise", "socp"]` keyword, defaulting to `"drop"`. Phase 18 implements `"drop"`; `"piecewise"` and `"socp"` raise `NotImplementedError` with a Phase pointer.

A new `Solver.supports_probmode3() -> bool` capability method (default `True`, overridden to `False` on no solver — every solver including MIP "supports" mode 3, just with degraded semantics for MIP). The Shiny run-panel uses this to add a "Note: MIP drops chance constraint; reports gap post-hoc" banner when MIP + PROBMODE 3 are both selected.

## Formulation (precise, Marxan-faithful)

Given:
- `μ_ij` = `amount_ij` (read from `pu_vs_features.amount`)
- `p_ij` = per-cell Bernoulli probability (read from optional `pu_vs_features.prob`; default 0 = deterministic cell)
- `T_j` (deterministic target of feature *j*) — read from `features.target`
- `ptarget_j` (probability target of feature *j*) — read from `features.ptarget` (default `−1` = disabled)
- `γ` = `PROBABILITYWEIGHTING` (read from `parameters`, default 1.0)
- `x_i ∈ {0, 1}` (selection)
- `SPF_j` (species penalty factor) — read from `features.spf`

**Expected amount and variance per feature**, computed exactly as Marxan does (`probability.cpp` lines 38–41 for 1D, 88–91 for 2D — for our purposes these collapse to the 2D form with `p_ij` per cell):

```
E[T_j]   = Σ_i x_i · μ_ij · (1 − p_ij)
Var[T_j] = Σ_i x_i · μ_ij² · p_ij · (1 − p_ij)
```

(Note Marxan's 1D convention: `p` is per-PU **loss/threat probability** — the chance the PU is "lost". 2D `p` is per-cell. The wire format and the math are identical once vectorised. pymarxan uses the 2D convention internally.)

**Z-score (Marxan convention: `(T − E) / √V`)** — positive Z = shortfall:

When `Var[T_j] > 0`:
```
Z_j = (T_j − E[T_j]) / √Var[T_j]
P_j = probZUT(Z_j) = 1 − Φ(Z_j)     # upper tail; probability target met
```

When `Var[T_j] == 0` (no uncertainty for this feature in the reserve): `Z_j = 4` (Marxan's sentinel), `P_j ≈ 1`.

**Important — `MISSLEVEL` is NOT applied here.** Marxan classic uses the raw `T_j` inside the Z-score. MISSLEVEL applies only to the existing deterministic shortfall penalty path (`compute_feature_shortfalls`), not the probability path.

**Penalty contribution of feature *j*** under `PROBMODE = 3` (Marxan-faithful, normalised):

```
if ptarget_j ≤ 0:                       # feature has no probability target
    penalty_j = 0
elif P_j ≥ ptarget_j:                   # target met probabilistically
    penalty_j = 0
else:
    penalty_j = SPF_j · (ptarget_j − P_j) / ptarget_j
```

Total probability penalty applied to the objective: `γ · Σ_j penalty_j`. Added to the existing objective alongside the deterministic shortfall penalty (PROBMODE 3 is *additive*, not replacement — users may run with both deterministic targets and probability targets active).

## API surface changes

```python
# Existing — unchanged
problem.parameters["PROBMODE"]                   # 0/1/2/3 (3 is new)
problem.parameters["PROBABILITYWEIGHTING"]       # applies to all modes (was only mode 1)
problem.probability                              # PROB1D table (still used for per-PU loss prob)

# New optional column on pu_vs_features (Marxan-compatible wire format)
problem.pu_vs_features["prob"]                   # p_ij ∈ [0,1]; default 0 (deterministic cell)

# New optional column on features
problem.features["ptarget"]                      # p_j ∈ (0,1] or -1 (disabled); default -1

# New computation (lives in solvers/probability.py — new module)
from pymarxan.solvers.probability import (
    compute_expected_and_variance,         # build E[T_j] and Var[T_j] from cache
    compute_zscore_per_feature,            # Marxan's (T − E)/√V
    compute_zscore_penalty,                # γ · Σ SPF_j · (ptarget − P) / ptarget
    evaluate_solution_chance,              # post-hoc per-feature report
)

# New capability method on Solver
Solver.supports_probmode3() -> bool                  # default True
```

`compute_probability_penalty` in `solvers/utils.py` keeps its name and signature. Internally it dispatches:
- `PROBMODE == 1` → existing risk-premium logic (PROBABILITYWEIGHTING-scaled)
- `PROBMODE == 2` → returns 0.0 (handled by amount-substitution upstream)
- `PROBMODE == 3` → calls `compute_zscore_penalty` (PROBABILITYWEIGHTING-scaled)
- otherwise → returns 0.0

`Solution` gains two optional attributes populated under PROBMODE 3:
- `prob_shortfalls: dict[int, float] | None` — per-feature `max(0, ptarget − P_j)`
- `prob_penalty: float | None` — the γ-weighted sum that entered the objective

## Data flow

```
read_puvspr() → pu_vs_features (with optional 'prob' column = p_ij ∈ [0,1])
                       │
                       ▼
read_spec()  → features (with optional 'ptarget' column ∈ (0,1] or -1)
                       │
                       ▼
ConservationProblem ───┴───▶ Solver
                                │
                                ▼
              ProblemCache.from_problem()
                  - precomputes mu_matrix  (n_pu, n_feat)   = amount_ij
                  - precomputes prob_matrix (n_pu, n_feat)  = p_ij
                  - precomputes var_matrix (n_pu, n_feat)   = amount²·p·(1−p)
                  - precomputes feat_ptarget (n_feat,)
                  - precomputes pu_feat_idx[i] = sparse list of features in PU i
                                                  (preserves O(features_per_pu)
                                                   delta cost)
                  - tracks held_mean, held_var, held_prob_term incrementally

              SA + IterativeImprovementSolver:
                  - go through ProblemCache automatically; PROBMODE 3 logic
                    lives in the cache, not the solver

              HeuristicSolver:
                  - does NOT use ProblemCache. Adds inline PROBMODE 3 branch
                    that computes held_mean, held_var per candidate and calls
                    compute_zscore_penalty().

              MIPSolver / ZoneMIPSolver:
                  - strategy='drop' (default): probability term excluded from
                    the MIP objective. Returned Solution is post-hoc evaluated
                    under PROBMODE 3 via evaluate_solution_chance(); the gap
                    populates Solution.prob_shortfalls + Solution.prob_penalty.
                  - strategy='piecewise' / 'socp': NotImplementedError pointers
                    to Phase 18.5 / Phase 21.
```

## File format (Marxan-compatible)

Marxan-tabular `puvspr.dat` is extended with the optional `prob` column (Marxan v4's PROB2D wire format — see `input.cpp::readSparseMatrix`, "if the prob field is tagged on the end"):

```csv
species,pu,amount,prob
1,1,10.0,0.10
1,2,8.0,0.05
2,1,20.0,0.0
...
```

`prob` is optional. Readers default to `0.0` when absent (cell is deterministic — `(1−p)=1`, variance=0). Writers emit the column only when at least one row has non-zero `prob`. Files round-trip with Marxan C++.

`spec.dat` is extended with the optional `ptarget` column (Marxan's PTARGET2D — see `input.cpp::readSpecies`):

```csv
id,name,target,spf,ptarget
1,coral,100,1.0,0.95
2,kelp,50,2.0,-1
...
```

`ptarget` is optional. Readers default to `−1` when absent. `−1` means "no probability target for this feature" (Marxan's disabled-sentinel convention) — the feature contributes 0 to the probability penalty regardless of its Z-score. This is the correct default: existing legacy projects load unchanged and the new mode opts in feature-by-feature. Writers emit the column only when at least one row has `ptarget > 0`.

## Test strategy

Same TDD-per-task pattern as Phases 1–17. Test categories:

1. **Z-score math (Marxan-faithful)** — known `amount, p, T → known Z = (T−E)/√V` and `P = 1−Φ(Z)`; cross-checked against `scipy.stats.norm.sf` (which is `probZUT`).
2. **Numeric agreement with Marxan reference** — for a constructed 4-PU / 2-feature scenario with hand-computed expected `Z_j` and `P_j`, the pymarxan implementation matches to 1e-9. Doubles as a regression guard if anyone touches the formula later.
3. **Penalty normalisation** — verifies the penalty divides by `ptarget_j`: at `P=0.5, ptarget=0.95`, penalty per unit SPF is `(0.95−0.5)/0.95 ≈ 0.474`, not `0.45`.
4. **PROBMODE dispatch** — `compute_probability_penalty` returns the right thing for modes 0/1/2/3.
5. **Schema defaults** — `pu_vs_features` without `prob` column → variance/expected reduce to deterministic; `features` without `ptarget` (or `ptarget = -1`) → 0 contribution from probability penalty.
6. **Delta computation under PROB2D** — SA's `ProblemCache.compute_delta_objective` matches `compute_full_objective` after-before for ≥10 random flips at PROBMODE=3 on a problem where multiple features have non-zero variance.
7. **MIP drop strategy** — `MIPSolver.solve(problem_with_probmode_3)` returns a solution (no exception), and `Solution.prob_shortfalls` / `Solution.prob_penalty` are populated from post-hoc evaluation.
8. **MIP non-default strategy** — `MIPSolver(mip_chance_strategy="piecewise")` and `("socp")` raise NotImplementedError with phase pointers.
9. **Round-trip I/O** — `save_project + load_project` preserves `prob` and `ptarget` columns; missing-column defaults reproduce identical objective.
10. **MISSLEVEL NOT inside Z-score** — explicit regression test: changing MISSLEVEL changes the deterministic penalty but NOT `Z_j`, `P_j`, or `prob_penalty`. Documents the design decision.
11. **ptarget = -1 inactive** — feature with ptarget=-1 has 0 probability contribution regardless of Z.
12. **Capability method** — `MIPSolver().supports_probmode3()` returns True (since it falls back to drop strategy); `Solver` ABC has the method.

Target: +12–15 tests, total ≥1106, coverage stays ≥91 %.

## Migration / compatibility

- Existing projects (PROBMODE 0/1/2) are 100 % unaffected — no column reads, no default behaviour changes.
- Existing `.dat` files load identically; default `variance == 0` and `prob_target == 0.5` produce identical objective values for old projects.
- The new `variance` column in `pu_vs_features` is positional-optional — readers tolerate its absence; writers only emit it when meaningful.
- Same for `prob_target` on `features`.
- No API breakage. All existing tests stay green.

## Implementation order (preview)

See `2026-05-16-phase18-implementation.md` (revised post-review). Sketch:

1. Schema + I/O: `prob` column on `puvspr.dat`, `ptarget` column on `spec.dat`, sentinel-aware defaults.
2. `compute_expected_and_variance`, `compute_zscore_per_feature`, `compute_zscore_penalty` pure functions in `solvers/probability.py`.
3. PROBMODE dispatch in `compute_probability_penalty` (existing function widened).
4. ProblemCache: `prob_matrix`, `var_matrix`, `feat_ptarget`, sparse `pu_feat_idx` for `O(features_per_pu)` delta + held-variance vector.
5. `HeuristicSolver` inline PROBMODE 3 branch (does NOT use ProblemCache). SA and IterativeImprovementSolver inherit automatically from #4.
6. `MIPSolver` and `ZoneMIPSolver`: `mip_chance_strategy` kw, "drop" implementation + post-hoc evaluation, "piecewise"/"socp" raise NotImplementedError.
7. `Solver.supports_probmode3()` capability method.
8. Shiny: extend existing `probability_config.py` PROBMODE radio with "3" choice; add Z-score-per-feature display in `target_met.py`; add MIP-drop notice banner in `run_panel.py`.
9. End-to-end smoke test on a known synthetic problem in `tests/integration/test_phase18_smoke.py` (create the integration directory).

## Risks

- **Numerical stability** for very small `V_j` (lots of mostly-deterministic PUs). Use `np.clip(V_j, 1e-12, None)` before `√`. Memory carries the same trick from `simulated_annealing.py` clamping initial_temp at `0.001`.
- **Floating-point underflow** in `held_var` after subtraction (e.g. `held_var = -1e-18` on a flip-and-flip-back). Clamp `held_var = max(0, held_var)` after each update.
- **Φ(Z) at extremes**: `scipy.stats.norm.sf` is numerically fine in `[-37, 37]`; outside that range it returns 0.0 or 1.0 which is correct for our purposes. No special-casing.
- **Per-`norm.sf` call cost in SA hot loop**: ~5M evaluations for 100k iterations × 50 features. Architect agent flagged this as a potential benchmark regression. Mitigation: sparse per-PU feature index (cache only updates features actually present in the flipped PU) — keeps the inner loop closer to `O(features_per_pu)`. Verify with the existing `tests/benchmarks/` suite before claiming acceptance.
- **PROB2D file I/O** is a wire-format change — existing test fixtures don't have `prob` columns, so they exercise the default-zero path. We need at least one new test fixture under `tests/data/probabilistic/` that does.

## Assumptions (from review)

Three load-bearing modelling assumptions that pymarxan inherits unchanged from Marxan classic; document them in user-facing docs so practitioners know what they're buying:

1. **Independence across PUs** — `Var(Σ x_i A_ij) = Σ x_i σ²_ij`. Marxan has no covariance terms (verified by grep of `probability.cpp`). False under spatial autocorrelation; can over-estimate the reserve's confidence (Tulloch 2013).
2. **CLT-Normal approximation** — `Σ A_ij` is approximately Normal for moderate `|S|`. Holds poorly for very small reserves or strongly skewed amount distributions (Game 2008).
3. **Bernoulli per-cell uncertainty** — variance is fully determined by `amount · p · (1−p)`. Cannot represent uncertainty that decouples from amount magnitude (e.g. a fixed measurement-error variance). A future PROBMODE 4 could lift this; Phase 18 ships Marxan parity.

## Acceptance criteria (from §7 of realignment plan)

1. ✅ All new code has TDD-per-task unit tests.
2. ✅ `make check` stays green.
3. ✅ Shiny app exposes PROBMODE 3 toggle + Z-score display.
4. ✅ `CHANGELOG.md` gets an `## [Unreleased]` entry promoted later to `[0.2.0]`.
5. ✅ Test count grows; coverage ≥91 %.
