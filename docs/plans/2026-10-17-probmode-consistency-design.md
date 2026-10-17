# PROBMODE 1/2 consistency across solvers (v0.45.0) — design

**Date:** 2026-10-17 · **Status:** approved in conversation; spec under review
**Follows:** v0.44.0 (SA optimises connectivity). **Precedes:** a separate stage implementing
Marxan's Probability 1D (chance constraint) — out of scope here.

## 1. Background (verified 2026-10-17)

- **Marxan C++** (`Marxan-source-code/marxan`, `probability.cpp`, `score_change.cpp`,
  `input.cpp:148`) has no `PROBMODE` parameter. A `prob` column in `pu.dat` switches on
  Probability 1D: per feature, expected `Σ amount·(1−p)`, variance `Σ amount²·p·(1−p)`,
  Z-score against `ptarget1d`, penalty `PROBABILITYWEIGHTING · Σ shortfall`. A `prob` column
  in `puvspr` switches on Probability 2D (presence polarity), which is pymarxan's PROBMODE 3.
- **pymarxan's PROBMODE 1** (risk premium `γ·Σ p_i·c_i·x_i`) and **PROBMODE 2** (feature
  amounts scaled by `(1 − p_i)`, deterministic) are pymarxan extensions predating Phase 18.
  They read the per-PU probability table (`problem.probability`, `prob.dat`).
- **The solvers disagree** (probe on `tests/data/simple`, selection {2,4,6}, probabilities
  `linspace(0.1, 0.6)`, `PROBABILITYWEIGHTING` 2):

| | MIP | Greedy | SA / II (`ProblemCache`) | Reported (`build_solution`) |
|---|---|---|---|---|
| Default `PROBMODE` with a probability table | 1 | 1 | **0** | 1 |
| PROBMODE 1 premium | ✓ | ✓ (effective cost) | **✗** (44.0 vs reported 69.2) | ✓ |
| PROBMODE 2 `(1 − p)` amounts | ✓ | ✓ | **✗** | **✗** (targets judged on raw amounts) |

  With all probabilities 0.5 under PROBMODE 2 the MIP selects every unit to meet the
  scaled targets, and `build_solution` judges them on raw amounts.

## 2. Decisions (user-approved)

| Decision | Choice |
|---|---|
| Scope | Staged: consistency now (this spec); Marxan Probability 1D as its own later stage. |
| Default mode | With a probability table and no `PROBMODE`, every path uses 1 (the cache changes from 0). |
| PROBMODE 2 "held" | **Expected amount everywhere**: `_compute_achieved` scales by `(1 − p)`, so targets met, shortfall, penalty, `compute_held`, writers, analysis and Shiny all report `Σ amount·(1 − p)`, consistent with the MIP's target rows and `MultiObjectiveMIP`'s met counts. |
| Labelling | Docs call PROBMODE 1/2 pymarxan extensions and say Marxan Probability 1D is not yet implemented. |

## 3. Semantics (exactly the MIP's)

Let `p_i` be the probability of PU `i` from `problem.probability` (missing PU → 0), `c_i` its
cost, `γ = PROBABILITYWEIGHTING` (default 1.0), `mode = PROBMODE` (default 1).

- **mode 1** (requires a probability table): objective term `γ · Σ_i x_i · p_i · c_i`. Off
  when `γ == 0` or no table. Amounts unchanged.
- **mode 2** (requires a probability table): every feature amount `a_ij` becomes
  `a_ij · (1 − p_i)` wherever held amounts are computed. No objective term.
- **mode 0 / 3 / no table:** unchanged behaviour.

## 4. Changes

**`solvers/cache.py` (`ProblemCache`)**
- Read `PROBMODE` with default 1 when `problem.probability` is present (0 otherwise), keeping
  PROBMODE 3 behaviour identical.
- mode 1: new field `prob_cost: (n_pu,) float64` = `γ·p_i·c_i` (zeros when off), defaulted.
  `compute_full_objective` adds `Σ prob_cost[selected]`; `compute_delta_objective` adds
  `sign · prob_cost[idx]` behind a cheap flag check.
- mode 2: scale the amounts used for `held`, the deterministic penalty and its delta by
  `(1 − p_i)` at build time. Every other consumer of the cache's amount arrays (ClumpState,
  SepState, SpaceState, constraints) must stay consistent with **its own** reporting
  counterpart under PROBMODE 2 — the prototype verifies each and either matches it or, if a
  combination cannot be made consistent cheaply, records it as a backlog item with the
  current behaviour unchanged (never a silent mismatch introduced by this change).

**`solvers/utils.py`**
- `_compute_achieved`: under mode 2 scale by `(1 − p)` (binary and fractional paths).
- `compute_probability_penalty` docstring: drop the false "handled upstream" claim.

**Docs:** CHANGELOG `[Unreleased]` (`### Changed`: SA/II plans change on probability
problems; PROBMODE 2 reported held/met/shortfall now use expected amounts; cache default mode;
`### Fixed`: the mismatches above); `docs/USER_MANUAL.md` PROBMODE table/rows relabelled as
pymarxan extensions with Marxan 1D/2D mapping; roadmap: Marxan Probability 1D stage, PROBMODE
backlog line removed.

## 5. Effects

- SA and iterative-improvement plans change on problems with a probability table in mode 1 or
  2 (including tables with no `PROBMODE`, now mode 1).
- PROBMODE 2 reports (targets met, shortfall, penalty, held, mvbest, representation, Shiny)
  change to expected amounts.
- The MIP and greedy are unchanged. Goldens: the Stage 4 `build_solution` golden has a
  PROBMODE 1 variant only (reported premium unchanged); verify no golden moves.
- The 35.0 anchor holds (no probability table).

## 6. Testing

- **SA consistency:** seeded SA's reported `Solution.objective` equals
  `metadata["history"]["best_objective"][-1]` for mode 0, 1 and 2, and for a probability table
  with no `PROBMODE` (fails today for 1, 2 and the default case).
- **Delta exactness:** every PU, both directions, mode 1 and mode 2, delta == full difference.
- **Full = evaluator:** cache full objective equals `build_solution(...).objective` on random
  selections (modes 1, 2, default), with a PU missing from the probability table.
- **Evaluator PROBMODE 2:** `compute_held` returns `Σ a·(1−p)`; the MIP's PROBMODE 2 plan is
  reported as meeting exactly the targets the MIP enforced (today it is judged on raw amounts).
- **Cross-solver:** on a ≤ 8-PU fixture, brute-force optimum of the cache objective ==
  `MIPSolver(mip_backend="cbc")` `mip_objective_value`, and seeded SA reaches it, modes 1 and 2.
- **Combinations:** PROBMODE 2 with clumping / separation: reported vs tracked objective (or the
  recorded backlog decision per §4).
- `make check` green; parity harness unchanged; no golden regenerated.

## 7. Out of scope

Marxan Probability 1D (next stage); reading Marxan's `pu.dat` `prob` column; PROBMODE 3;
zones; fractional decisions (SA is binary-only).
