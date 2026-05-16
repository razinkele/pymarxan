# Phase 18 — Probability Completion (PROB2D + PTARGET + Z-score)

**Date:** 2026-05-16
**Target version:** v0.2.0 (Marxan-classic parity)
**Parent plan:** [`docs/plans/2026-05-16-realignment.md`](2026-05-16-realignment.md)
**Effort:** ~1 week
**Confidence:** 80 %

## Why

pymarxan already implements Marxan's two simpler probability modes:

| `PROBMODE` | What it does | Where |
|---|---|---|
| 1 | Risk premium: adds `PROBABILITYWEIGHTING · Σ prob_i · cost_i · x_i` to objective | `solvers/utils.py::compute_probability_penalty` |
| 2 | Persistence-adjusted amounts: replaces `amount_ij` with `amount_ij · (1 − prob_i)` | `mip_solver.py:183-220`, `heuristic.py:194-244` |

Both modes are **per-PU only** — every feature at PU *i* shares the same probability `prob_i`. This is Marxan's "PROB1D" format.

What's still missing — and what Phase 18 ships — is the **PROB2D + Z-score** formulation that climate-resilience and uncertain-suitability studies actually use:

1. Per-(PU, feature) **mean μ_ij** and **variance σ²_ij**.
2. A **probability target** PTARGET_j ∈ (0, 1) per feature — "feature *j* must be represented in the reserve with probability ≥ PTARGET_j".
3. **Z-score evaluation**: `Z_j = (Σ x_i μ_ij − target_j) / √(Σ x_i σ²_ij)`. The reserve meets feature *j* with probability `Φ(Z_j)` (assuming independence across PUs and a normal-approximation to the sum, both standard Marxan assumptions).
4. **Penalty term**: when `Φ(Z_j) < PTARGET_j`, a shortfall penalty proportional to `PTARGET_j − Φ(Z_j)`, weighted by SPF_j (same SPF as the deterministic target).

Without this, pymarxan can't reproduce results from papers that use `PROBMODE = 3` (the Z-score mode) and can't claim "full Marxan-classic parity" — the headline goal of v0.2.0.

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

- **MIP support for PROBMODE = 3.** The Z-score constraint involves `√(Σ σ²_ij·x_i)`, which is SOCP/QCP territory. CBC (pymarxan's default open-source backend) can't solve it; Gurobi/CPLEX could, but those backends arrive in Phase 21. Until then, calling `MIPSolver.solve()` on a problem with `PROBMODE = 3` raises a clear `NotImplementedError` pointing users at SA / heuristic / iterative-improvement and the Phase 21 issue.
- **Per-zone PROB2D.** Marxan with Zones doesn't ship Z-score support either; deferred until users ask.
- **PROBMODE 1 + 3 combination.** Mode 3 supersedes Mode 1's risk-premium logic.

## Formulation (precise)

Given:
- `μ_ij` (mean amount of feature *j* in PU *i*) — read from `pu_vs_features.amount`
- `σ²_ij` (variance of feature *j* in PU *i*) — read from `pu_vs_features.variance` (default 0 when column missing)
- `T_j` (deterministic target of feature *j*) — read from `features.target`
- `MISSLEVEL` (target-shortfall threshold, default 1.0)
- `p_j` (probability target of feature *j*) — read from `features.prob_target` (default 0.5 = even chance)
- `x_i ∈ {0, 1}` (selection)
- `SPF_j` (species penalty factor) — read from `features.spf`

Effective deterministic target: `T'_j = T_j · MISSLEVEL`

Reserve mean: `M_j = Σ_i x_i · μ_ij`
Reserve variance: `V_j = Σ_i x_i · σ²_ij`

When `V_j > 0`:
- `Z_j = (M_j − T'_j) / √V_j`
- `P_j = Φ(Z_j)` (cumulative normal)

When `V_j == 0` (deterministic case): `P_j = 1.0` if `M_j ≥ T'_j` else `0.0`.

**Penalty contribution of feature *j*** under `PROBMODE = 3`:
```
penalty_j = SPF_j · max(0, p_j − P_j)
```

Total probability penalty: `Σ_j penalty_j`. Added to the existing objective the same way `compute_penalty()` is.

## API surface changes

```python
# Existing — unchanged
problem.parameters["PROBMODE"]                   # 0/1/2/3 (3 is new)
problem.parameters["PROBABILITYWEIGHTING"]       # used only when PROBMODE==1
problem.probability                              # PROB1D table (unchanged)

# New optional columns
problem.pu_vs_features["variance"]               # σ²_ij; default 0
problem.features["prob_target"]                  # p_j ∈ (0,1); default 0.5

# New computation
from pymarxan.solvers.utils import (
    compute_probability_penalty,           # widened to dispatch on PROBMODE
    compute_zscore_penalty,                # new: PROBMODE=3 path
    compute_zscore_per_feature,            # new: returns dict[fid, Z_j]
)
```

`compute_probability_penalty` keeps its name and signature. Internally it dispatches:
- `PROBMODE == 1` → existing risk-premium logic
- `PROBMODE == 2` → returns 0.0 (handled by amount-substitution upstream)
- `PROBMODE == 3` → calls `compute_zscore_penalty`
- otherwise → returns 0.0

## Data flow

```
read_puvspr() → pu_vs_features (with optional 'variance' column)
                       │
                       ▼
read_spec()  → features (with optional 'prob_target' column)
                       │
                       ▼
ConservationProblem ───┴───▶ Solver
                                │
                                ▼
              SA: ProblemCache.from_problem()
                  - precomputes mu_matrix  (n_pu, n_feat)
                  - precomputes var_matrix (n_pu, n_feat)
                  - tracks reserve M_j, V_j incrementally on flip
                  - compute_zscore_penalty(M, V, T', p, SPF)

              Heuristic / iterative improvement:
                  - calls compute_zscore_penalty() directly with
                    np.dot(selected, mu/var matrix)

              MIP: raises NotImplementedError when PROBMODE==3
```

## File format

Marxan-tabular `puvspr.dat` becomes:

```csv
species,pu,amount,variance
1,1,10.0,0.5
1,2,8.0,1.2
...
```

`variance` is optional — readers default to `0.0` when absent (which makes the math degenerate gracefully to the deterministic case). Writers emit the column only when at least one row has a non-zero variance.

`spec.dat` becomes:

```csv
id,name,target,spf,prob_target
1,coral,100,1.0,0.95
2,kelp,50,2.0,0.8
...
```

`prob_target` is optional — readers default to `0.5` when absent (50% probability target — even-money chance — which is conventional and harmless when paired with `variance == 0`).

## Test strategy

Same TDD-per-task pattern as Phases 1–17. Test categories:

1. **Z-score math** — known μ/σ²/T → known Z and Φ(Z); cross-checked against `scipy.stats.norm.cdf`.
2. **Penalty math** — for a constructed scenario, the SA full-objective penalty equals the SPF-weighted shortfall in Φ space.
3. **PROBMODE dispatch** — `compute_probability_penalty` returns the right thing for modes 0/1/2/3.
4. **Schema defaults** — `pu_vs_features` without `variance` column → variance=0 throughout; `features` without `prob_target` → 0.5.
5. **Delta computation under PROB2D** — SA's `ProblemCache.compute_delta_objective` matches `compute_full_objective` after-before for ≥10 random flips at `PROBMODE=3`.
6. **MIP NotImplementedError** — `MIPSolver.solve(problem_with_probmode_3)` raises with a message pointing at SA.
7. **Round-trip I/O** — `save_project + load_project` preserves variance and prob_target columns.
8. **MISSLEVEL × PROBMODE 3** — verifies Z-score uses `T · MISSLEVEL` not raw `T`.

Target: +8–12 tests, total ≥1102, coverage stays ≥91 %.

## Migration / compatibility

- Existing projects (PROBMODE 0/1/2) are 100 % unaffected — no column reads, no default behaviour changes.
- Existing `.dat` files load identically; default `variance == 0` and `prob_target == 0.5` produce identical objective values for old projects.
- The new `variance` column in `pu_vs_features` is positional-optional — readers tolerate its absence; writers only emit it when meaningful.
- Same for `prob_target` on `features`.
- No API breakage. All existing tests stay green.

## Implementation order (preview)

See `2026-05-16-phase18-implementation.md` (written next). Sketch:

1. Schema + I/O: variance column, prob_target column, default handling.
2. `compute_zscore_per_feature` pure function.
3. `compute_zscore_penalty` pure function.
4. PROBMODE dispatch in `compute_probability_penalty`.
5. ProblemCache: variance matrix + held-variance vector + delta computation.
6. Heuristic / iterative improvement integration.
7. MIP NotImplementedError guard.
8. Shiny: PROBMODE radio + Z-score-per-feature display.
9. End-to-end smoke test on a known synthetic problem.

## Risks

- **Numerical stability** for very small `V_j` (lots of mostly-deterministic PUs). Use `np.clip(V_j, 1e-12, None)` before `√`. (Memory carries the same trick from `simulated_annealing.py::_temp_step` clamping at `0.001`.)
- **Φ(Z) at extremes**: `scipy.stats.norm.cdf` is numerically fine in `[-37, 37]`; outside that range it returns 0.0 or 1.0 which is correct for our purposes. No special-casing.
- **PROB2D file I/O** is a wire-format change — existing test fixtures don't have variance columns, so they exercise the default-zero path. We need at least one new test fixture under `tests/data/probabilistic/` that does.

## Acceptance criteria (from §7 of realignment plan)

1. ✅ All new code has TDD-per-task unit tests.
2. ✅ `make check` stays green.
3. ✅ Shiny app exposes PROBMODE 3 toggle + Z-score display.
4. ✅ `CHANGELOG.md` gets an `## [Unreleased]` entry promoted later to `[0.2.0]`.
5. ✅ Test count grows; coverage ≥91 %.
