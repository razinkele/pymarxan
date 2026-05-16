# Phase 18 — Multi-Agent Review Synthesis

**Date:** 2026-05-16
**Subject:** `2026-05-16-phase18-design.md` + `2026-05-16-phase18-implementation.md`
**Reviewers:** four subagents from disjoint angles (architect, codebase-grounding, scientific-accuracy, independent-re-design). Transcripts archived in agent task outputs.

## TL;DR

The Phase 18 design is **directionally correct but materially wrong on three fronts** that will fail "Marxan-classic parity":

1. **Marxan classic does NOT apply MISSLEVEL inside the Z-score** — the design does. Source: `probability.cpp::computeProbMeasures` references `target`, not `target × misslevel`. (Scientific agent, HIGH.)
2. **The wire format diverges from Marxan classic** — Marxan stores per-cell *occurrence/threat probability* `p` and derives variance internally as `amount² · p · (1−p)`. The design proposes a free-form `variance` column. Projects exchanged with Marxan C++ won't round-trip. (Scientific agent, HIGH.)
3. **The shortfall penalty is normalised differently** — Marxan computes `(prTarget − rawP) / prTarget`. The design computes `(prTarget − rawP)`. Penalty magnitude differs by a factor of `1/p_j`. (Scientific agent, MEDIUM.)

The **implementation plan** also has four mechanical errors caught by the code-explorer pass that must be fixed before Batch 1 can be executed mechanically:

4. Task 3 references a non-existent method `_validate_columns()`. The real method is `validate()`. (Codebase agent.)
5. Task 9 claims `IterativeImprovementSolver` doesn't use `ProblemCache` — it does. The proposed inline PROBMODE-3 logic in `iterative_improvement.py` would be redundant and divergent. (Codebase agent.)
6. Task 11 directs adding a PROBMODE radio to `run_panel.py`. There's already a dedicated `probability_config.py` module with a PROBMODE radio (currently only modes 1+2). The mode-3 entry belongs there. (Codebase agent.)
7. Task 12 says `tests/integration/` is an existing directory. It doesn't exist. (Codebase agent.)

Plus two structural design choices that two agents disagreed on:

8. **Where variance lives.** Architect agent argues for a separate `pu_vs_features_variance` field on `ConservationProblem` (mirroring the existing `probability` field). Independent re-design agent argues for a column on `pu_vs_features` (matches `amount` placement). Both have merit; the choice has knock-on effects on `validate()`, `clone()`, and `build_pu_feature_matrix`.

9. **Solver capability dispatch.** Architect agent argues the design should add `Solver.supports_probmode3()` mirroring the existing `supports_zones()` so the Shiny run-panel can pre-filter rather than catching `NotImplementedError` at runtime. The design currently has solver-level guards only.

## Findings by severity

### CRITICAL (must fix before execution)

#### C1 — MISSLEVEL inside the Z-score

**Source:** Scientific agent, grep of `Marxan-source-code/marxan` showed zero references to `MISSLEVEL` in `probability.cpp` or `computation.hpp`. The Marxan formula is `Z = (target − E) / √Var`, no MISSLEVEL multiplication.

**Design says:** `Z_j = (Σ x_i μ_ij − T_j·MISSLEVEL) / √(Σ x_i σ²_ij)`
**Should be:** `Z_j = (T_j − Σ x_i μ_ij) / √(Σ x_i σ²_ij)` (note also: sign flip — see C3)

**Impact:** if Phase 18 ships with MISSLEVEL in the Z-score, a user porting a project from Marxan C++ with default MISSLEVEL=1.0 will get bit-exact agreement. But with any non-default MISSLEVEL the penalty diverges. The "Marxan-classic parity" headline is broken.

**Fix:** drop `MISSLEVEL` from the Z-score expression. Document that MISSLEVEL applies only to the deterministic shortfall path, matching Marxan classic.

#### C2 — Wire-format compatibility

**Source:** Scientific agent, `input.cpp::readSparseMatrix`: "if the prob field is tagged on the end". Marxan's PROB2D file is `puvspr.dat` with a 4th column = `prob` (Bernoulli probability per PU-feature cell). Variance is derived inside `computeProbMeasures` as `amount² · prob · (1 − prob)`.

**Design says:** extend `puvspr.dat` with a free-form `variance` column.

**Two ways to reconcile:**
- **Option A (compatibility-first):** adopt Marxan's `prob` column convention. Variance computed internally. Pro: byte-for-byte compatible. Con: less expressive — pymarxan can only model Bernoulli variance, not general per-cell variance.
- **Option B (expressiveness-first):** keep `variance` but add a separate `prob`-mode reader that converts Marxan files. Document that pymarxan generalises Marxan's model. Pro: more flexible. Con: silent semantic drift from Marxan.

The agent recommends Option A for Phase 18 to claim parity, with a future PROBMODE 4 for general variance.

#### C3 — Shortfall normalisation and sign

**Source:** Scientific agent, `computation.hpp::computeProbMeasures` lines 74–97:
```cpp
rZ = (target - expected) / sqrt(variance);     // sign: T − E
rRawP = utils::probZUT(rZ);                    // upper tail
rShortfallPenalty = (prTarget - rRawP) / prTarget;     // normalised
```

**Design says:** `penalty_j = SPF_j · max(0, p_target_j − Φ(Z_j))` with `Z_j = (M − T)/√V`.

**Should be:** `penalty_j = SPF_j · max(0, (p_target_j − P_j) / p_target_j)` where `P_j = Φ((T − M)/√V)` (or equivalently `1 − Φ((M − T)/√V)`).

**Impact:** for a target of `p_j = 0.95` and observed `P_j = 0.5`, the design produces penalty `0.45 · SPF` while Marxan produces `(0.45/0.95) · SPF ≈ 0.474 · SPF`. Different magnitudes affect BLM calibration and solver convergence.

**Fix:** divide the shortfall by `p_target_j` (the agent suggests `(p − P)/p`), and verify the Z sign against `probZUT` by writing a numeric cross-check test against `Marxan-source-code/marxan/probability.cpp` outputs as part of Batch 2.

### HIGH (should fix; will create real bugs if left)

#### H1 — Default `prob_target` should be disabled (−1), not 0.5

**Source:** Scientific agent, `input.cpp::readSpecies`. Marxan defaults `ptarget1d` and `ptarget2d` to `−1` ("disabled" — feature contributes 0 to probability penalty). The design defaults to 0.5.

**Impact:** any user importing a legacy project gets a "50% probability" constraint silently switched on for every feature. Their solver objective will diverge from the C++ version's.

**Fix:** default to `−1.0` and treat negative values as "no probability target for this feature". Skip the feature in the penalty sum.

#### H2 — IterativeImprovementSolver doesn't need inline logic

**Source:** Codebase agent, `iterative_improvement.py:114` calls `ProblemCache.from_problem` and uses `compute_full_objective` / `compute_delta_objective` throughout.

**Impact:** the impl plan's Task 9 says "no ProblemCache changes — these solvers don't use one" and proposes inline PROBMODE 3 logic in `iterative_improvement.py`. This is wrong for that solver; the cache changes from Task 7 propagate to it automatically. Implementing inline logic would create two divergent code paths.

**Fix:** Task 9 covers `HeuristicSolver` (which genuinely doesn't use the cache) only. `IterativeImprovementSolver` inherits PROBMODE 3 support from Task 7's cache changes.

#### H3 — Existing `probability_config.py` is the right integration point, not `run_panel.py`

**Source:** Codebase agent, file at `src/pymarxan_shiny/modules/probability/probability_config.py` exists and currently exposes a PROBMODE radio with choices `"1"` and `"2"`.

**Impact:** the impl plan's Task 11 directs adding the PROBMODE control to `run_panel.py`. Following the plan literally would create a duplicate widget while leaving the real control showing only modes 1+2.

**Fix:** Task 11 adds `"3"` to the existing `probability_config.py` radio. Also adds conditional UI showing variance/ptarget columns when mode 3 is selected. The Z-score display in `target_met.py` stays as written.

#### H4 — Add `Solver.supports_probmode3()` capability

**Source:** Architect agent.

**Impact:** without a capability method, the Shiny run-panel can't grey-out MIP when PROBMODE=3 is active. Users discover the limitation by clicking Solve and getting an error. `supports_zones()` exists as the pattern.

**Fix:** add `supports_probmode3() -> bool` to `Solver` base. Default `True`. Override to `False` in `MIPSolver` and `ZoneMIPSolver`. Check in `RunModePipeline.solve()` and in the Shiny picker.

### MEDIUM (improves quality / future-proofs)

#### M1 — Variance storage choice: separate field vs. column

**Source:** disagreement between architect agent (separate field) and independent re-design (column on `pu_vs_features`).

**Trade-offs (in pymarxan's pattern context):**

| Aspect | Separate field `pu_vs_features_variance` | Column on `pu_vs_features` |
|---|---|---|
| Symmetry with existing fields | Matches `probability` (per-PU), `boundary`, `connectivity` | Matches the fact that variance is per-(PU,feature), same shape as `amount` |
| `clone()` / `copy_with()` machinery | Auto-handled (existing field copy) | Auto-handled (whole DataFrame copy) |
| `validate()` | Trivial: `if x is not None: check_columns(...)` | Needs column-presence check on optional column |
| `build_pu_feature_matrix()` | No changes; new `build_pu_feature_variance_matrix()` | Needs to ignore the extra column when building amount matrix |
| Marxan classic interop | Need a join when reading `puvspr.dat` with `prob` column | More natural one-to-one mapping |
| Where Marxan stores it | Single file, single row per (pu, species) | Single file, single row per (pu, species) |

**Recommendation:** the **column** approach (independent agent's choice) is actually more idiomatic for *Marxan-compatible data*, because Marxan itself stores `prob` in the same file. The architect's concern about `build_pu_feature_matrix` is real but tractable — that method already filters to specific columns. The trade-off is that `ConservationProblem.clone()` and `validate()` need explicit awareness of the optional column.

**Action:** go with the **column** approach (independent agent), but explicitly document the column as optional in `pu_vs_features` and add a `variance_amount_matrix()` method on `ConservationProblem` parallel to `build_pu_feature_matrix()`. `validate()` is unchanged because it already permits extra columns.

#### M2 — Sparse per-PU feature index in `ProblemCache`

**Source:** independent re-design agent.

**Impact:** the architect raised the perf concern (`norm.cdf` per feature per flip). The independent re-design proposes precomputing `pu_feat_idx[idx] : list[int]` — the sparse list of features non-zero in PU `idx`. Delta then only updates `held_var[j]` for those `j`, keeping `O(features_per_pu)` per flip.

**Action:** include the sparse index in Task 7's `ProblemCache` extension. Adds ~5 lines of construction code in `from_problem` and a few lines in `compute_delta_objective`.

#### M3 — Rename `"probability"` objective term key for mode 3

**Source:** architect agent.

**Impact:** `compute_objective_terms` returns a dict with key `"probability"`. Under PROBMODE 1 that's a risk premium; under PROBMODE 3 it's a chance-constraint shortfall — semantically different.

**Action:** keep the key `"probability"` but ensure the value documentation says "PROBMODE-dependent meaning". Don't break the API. Add a `"probability_mode"` companion key on the dict.

#### M4 — `write_spec` omission rule for `prob_target`

**Source:** architect agent.

**Impact:** the impl plan says "omit when all values equal 0.5". After fixing H1 (default to −1), the rule becomes "omit when all values are −1 (the default)", which is unambiguous and matches `variance`'s "omit when all zero" rule.

**Action:** trivial fix; cascades from H1.

### LOW (polish / nice-to-have)

#### L1 — PROBABILITYWEIGHTING applies to mode 3 too

**Source:** scientific agent. Marxan multiplies the summed shortfall by `PROBABILITYWEIGHTING` regardless of mode.

**Action:** apply the weighting consistently. One-line fix in `compute_zscore_penalty`.

#### L2 — Document the independence and normality assumptions

**Source:** scientific + independent.

**Action:** add a §11 "Assumptions" section to the design doc citing Game (2008) and Tulloch (2013), noting:
- Sums-of-amounts are CLT-approximated as Normal.
- Per-PU amounts are assumed independent (no covariance terms in Marxan's model either).
- Recommends users with strong spatial autocorrelation use SA over MIP (since neither path actually models covariance, but at least SA's stochastic exploration is more forgiving).

## Citations to fold into the design doc

The scientific agent surfaced three foundational papers that should be cited in the design's "Why" section:

- Game, E. T., Watts, M. E., Wooldridge, S., & Possingham, H. P. (2008). Planning for persistence in marine reserves: A question of catastrophic importance. *Ecological Applications, 18*(3), 670–680. https://doi.org/10.1890/07-1027.1
- Tulloch, V. J., Possingham, H. P., Jupiter, S. D., Roelfsema, C., Tulloch, A. I. T., & Klein, C. J. (2013). Incorporating uncertainty associated with habitat data in marine reserve design. *Biological Conservation, 162*, 41–51. https://doi.org/10.1016/j.biocon.2013.01.003
- Carvalho, S. B., Brito, J. C., Crespo, E. G., Watts, M. E., & Possingham, H. P. (2011). Conservation planning under climate change: Toward accounting for uncertainty in predicted species distributions to increase confidence in conservation investments in space and time. *Biological Conservation, 144*(7), 2020–2030. https://doi.org/10.1016/j.biocon.2011.04.024

## Comparison with independent re-design

The independent agent (no sight of my draft) landed within striking distance:
- Same Z-score core (deterministic-equivalent formulation `t_j + z_j·√Var ≥ E[T_j]`).
- Same variance column placement.
- Same sparse per-PU feature index.
- **Different MIP strategy**: enumerates `drop` / `piecewise` / `scenario`, defaults to `drop` with post-hoc evaluation rather than `NotImplementedError`. This is arguably better UX — users can still run MIP, get a deterministic solution, and see the gap to the chance-constraint optimum reported.
- 5 batches vs my 4 (added a "math + plumbing" Batch 1 separate from I/O).
- More tests (15 vs 10).

**Action:** consider adopting the independent agent's MIP "drop + post-hoc evaluation" strategy. It's more honest than `NotImplementedError` and ships in the same release window. Phase 21 still adds Gurobi/SOCP for users who need certified chance-constrained optimality.

## Recommended next actions

1. **Patch the design doc** with C1–C3 corrections (Marxan-parity bugs) and H1, H4, M2, M3, L1, L2 (smaller fixes + literature citations).
2. **Patch the implementation plan** with H2, H3, plus the four mechanical errors (Task 3 method name, Task 9 redundant inline logic, Task 11 wrong UI module, Task 12 missing directory).
3. **Decide on M1** (variance storage) — synthesis recommends column-on-`pu_vs_features` to match Marxan's wire format.
4. **Decide on MIP strategy** (independent agent's drop-and-post-hoc vs my NotImplementedError). Synthesis recommends drop-and-post-hoc.
5. **Re-circulate** the patched plan if any of (1)-(4) involve significant rewrites, or just proceed to Batch 1 if patches are surgical.

## What this review cost / saved

- Four agents in parallel, ~10 minutes total elapsed.
- Caught three Marxan-parity bugs (MISSLEVEL, normalisation, wire format) that would have shipped as silent semantic drift had Phase 18 executed off the v1 plan.
- Caught four mechanical errors in the impl plan that would have wasted Batch 1's first few tasks.
- Surfaced two alternative design choices (separate field vs column, MIP drop vs error) that deserve explicit user input rather than being implicit.

Net: ~10 minutes of agent time saved an estimated 1-2 days of "execute → discover error → revise → re-execute" rework.

---
*Synthesis written by the main session based on four subagent transcripts. Transcripts live in `/tmp/claude-1000/.../tasks/` and are referenced by ID in the source-control commit message.*
