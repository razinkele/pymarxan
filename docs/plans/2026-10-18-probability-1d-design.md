# Marxan Probability 1D (planned v0.46.0) — design

**Date:** 2026-10-18 · **Status:** design approved in conversation (approach A); written spec
not yet reviewed. **Next steps:** spec review loop → prototype-first plan → plan review → SDD.
**Follows:** v0.45.0 (PROBMODE 1/2 consistency), whose review established that Marxan C++ has no
`PROBMODE`, that pymarxan's PROBMODE 1/2 are extensions, and that Marxan v4's `probZUT` has an
integer-division bug (`1 / 2`) that inverts its probability penalty.

## 1. Reference: Marxan 3

Source of truth: `Marxan-source-code/marxan`, branch **`marxan3`**, `probability.c`
(`ComputeProbability1D`, `ChangeProbability1D`, `probZUT`), `input.c` (`pu.dat` `prob`,
`spec.dat` `ptarget1d` / `ptarget2d`, default −1), `marxan.c` (score assembly). Not v4.

Per feature `j`, with per-unit probability of **loss** `p_i` from `pu.dat` `prob`:

- expected `E_j = Σ_i a_ij·(1 − p_i)`, variance `V_j = Σ_i a_ij²·p_i·(1 − p_i)`;
- `Z_j = (target_j − E_j)/√V_j` if `V_j > 0`, else **`Z_j = 4`** (zero variance);
- `P_j = probZUT(Z_j)` = upper normal tail (`1 − probZUT(−Z)` for negative Z) = `norm.sf(Z_j)`;
- `penalty_j = [ptarget1d_j > P_j] · (ptarget1d_j − P_j)/ptarget1d_j` if `ptarget1d_j > 0`,
  else 0 (−1 = disabled);
- `probability1D = PROBABILITYWEIGHTING · Σ_j penalty_j`, added to the score **alongside** the
  ordinary representation penalty, which stays on raw amounts.

Quirk adopted for parity: zero variance gives `Z = 4`, `P ≈ 3.2e-5`, so the feature counts as
failing its probability target — including when every `p_i = 0` (no risk) and `E_j ≥ target_j`.

2D is the same with per-cell **presence** probability `q_ij` (puvspr `prob`) and `ptarget2d`.
1D is therefore exactly 2D with `q_ij = 1 − p_i`.

## 2. Decisions (user-approved, 2026-10-18)

| Decision | Choice |
|---|---|
| Activation / inputs | **Marxan-native.** `read_pu` keeps `prob`; `read_spec` accepts `ptarget1d` and `ptarget2d` (legacy `ptarget` read as `ptarget2d`). New **`PROBMODE 4` = Marxan Probability 1D**. When `input.dat` gives no `PROBMODE`, `load_project` auto-sets 4 if `pu.dat` has `prob` and any `ptarget1d > 0`, or 3 if puvspr has `prob` and any `ptarget2d > 0`. |
| Zero variance | **Align PROBMODE 3 too:** both modes use Marxan 3's `Z = 4` rule through one shared function; PROBMODE 3 penalties change for zero-variance features (CHANGELOG `Changed`). |
| Approach | **A — generalise PROBMODE 3's machinery** (one Z / P / penalty core; modes differ only in the source of `q` and the target column), not a parallel 1D module. |
| 1D probability source | `pu.dat` `prob`; if absent, pymarxan's `prob.dat` table (`problem.probability`); always probability of loss. |
| Both 1D and 2D requested | pymarxan runs one mode at a time: warn and use 1D (Marxan can run both; simultaneous support backlogged). |

## 3. Design

**Inputs** (`io/readers.py`, writers): keep `pu.dat` `prob`; whitelist `ptarget1d`, `ptarget2d`;
map legacy `ptarget` → `ptarget2d`; writers round-trip the new columns; `load_project`
auto-activation per §2.

**Chance-constraint core** (`solvers/probability.py`): one function producing per-feature Z, P
and penalty with Marxan 3's rules (zero variance `Z = 4`, step function, normalisation by the
probability target, targets ≤ 0 disabled). PROBMODE 3 and 4 differ only in `q` (2D: per-cell
presence; 1D: `1 − p_i`) and the target column (`ptarget2d` / `ptarget1d`). The deterministic
target penalty stays on raw amounts.

**Solvers:** `ProblemCache` builds `expected_matrix` / `var_matrix` from `q` for both modes, so
the existing PROBMODE 3 SA delta covers 1D; `MIPSolver`'s `mip_chance_strategy`
(`drop` / `piecewise` / `socp`) applies to mode 4, with the new zero-variance rule; greedy
unchanged (ignores chance constraints, as for mode 3); non-binary decisions rejected for mode 4
as for mode 3; the evaluator / `build_solution` / `evaluate_solution_chance` report mode 4.

**Existing users:** PROBMODE 3 penalties change for zero-variance features; legacy `ptarget`
projects keep working.

## 4. Tests (to be detailed in the spec review)

Hand-computed 1D example checked against a transcription of Marxan 3's code; mode 4 == mode 3
with `q = 1 − p`; zero-variance rule in both modes (incl. all `p = 0`); auto-activation from a
real-format project (pu.dat `prob`, spec.dat `ptarget1d`); SA delta == full difference; reported
== tracked objective; MIP strategies with mode 4; golden handling for any moving PROBMODE 3
record; legacy `ptarget` round trip.

## 5. Open points for the spec review

Exact interplay of auto-activation with an explicit `PROBMODE` 1/2 and a `pu.dat` `prob` column;
whether the Shiny probability panel gains a mode-4 option and `ptarget1d` editing; how the MIP
strategies treat zero-variance features (the `Z = 4` rule is a constant penalty there);
`evaluate_solution_chance` output columns for 1D; docs (manual, README, copilot instructions)
replacing "Probability 1D not yet implemented".
