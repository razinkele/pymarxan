# Zonation raster-scale — multi-agent design review synthesis

**Date:** 2026-08-08 · **Run:** Workflow `wf_d831dfab-404`, 4 parallel lenses, ~469k tokens, 12 min.
**Verdicts:** architect / grounding / science / re-design — all **approve-with-fixes**. No CRITICAL.
**Empirical core:** grounding EXECUTED the plan's Task-3 engine verbatim vs the verbatim dense
oracle — 746 exact-equality runs + 320 argpartition tie-stress runs + 60 float runs; re-design
independently re-implemented the algorithm and ran 240 further comparisons. CAZ: bitwise-identical
removal orders in **every** run. All 13 existing rank_removal tests pass on the sparse engine.
The plan's Task-4 bench measured **3.7 s** vs the 60 s budget.

## Accepted findings and resolutions

| # | Sev | Lens | Finding | Resolution |
|---|-----|------|---------|-----------|
| 1 | HIGH | architect | `from pymarxan.zonation import rank_removal as rr_module` binds the **function** (`__init__.py:10` re-export shadows the submodule; verified empirically — even `import ... as` binds the function). Both warp-advisory tests AttributeError forever; the pre-impl "expected FAIL" masks it. | Plan Task 2: `rr_module = importlib.import_module("pymarxan.zonation.rank_removal")` + comment. |
| 2 | MED* | science | NaN amounts/costs/weights slip all `< 0` guards; in the sparse selection `v=NaN` makes `below`/`ties` empty → **empty batch → infinite loop** (dense engine terminated, garbage order). *Correctness-critical.* | Validate finiteness up front with the sign checks (see #4); NaN tests added. |
| 3 | MED | grounding | Two reproduced ABF integer-amount divergences (seed 9/warp=n/use_cost=False, pos 61; grid(60)/warp=10, pos 3174): `np.add.reduceat` right-associates vs dense pairwise `sum(axis=1)` → 1-ULP row sums flip exact ties. Plan's seeds all pass, but the passing set is a numpy-version artifact. | **Superseded by adopting #9** — reduceat replaced by chunked-dense rescore; ABF becomes bitwise row-equal given equal `Q`, so integer amounts → both rules exact. |
| 4 | MED | re-design (+arch, grounding LOWs) | Negative/NaN validation ran on the duplicate-summed, post-smoothing matrix — contract inconsistent across paths; cancelling negatives passed. | Validate **raw inputs** before any matrix build: `pu_vs_features["amount"]` ≥ 0 & finite, weights ≥ 0 & finite, costs finite (plus existing `c > 0` when `use_cost`). One check, never moves between tasks. |
| 5 | MED | re-design | Smoothing path had **zero** oracle-equivalence coverage post-rewrite (oracle dropped the branch; smoothed = float regime). | Oracle keeps its smoothing branch; one ≤20-PU smoothing equivalence fixture added (deterministic seed; §7 float caveat documented). |
| 6 | MED | architect | Task-4 dirty-vs-full test on integer amounts had zero discriminating power beyond the Task-1 oracle. | Parametrize `integer` True **and** False. |
| 7 | MED | arch + science + re-design (3×, independently) | §7 omitted a **fourth regrouping site**: incremental `cost_remaining` vs dense per-batch masked pairwise sum — drifts O(n·ulp) for float costs, can end tiny-negative; fixtures dodged it (costs stayed integer even with `integer=False`). | §7 table gains the row; `record_curve` clamps `max(cost_remaining, 0.0)`; `_random_problem(integer=False)` now floats costs too; float test gets `atol=1e-12`. |
| 8 | MED | re-design | §9.1's ABF gap-verification promise silently dropped in the plan; exact asserts passed by fixture-family luck; no fixture near the production 10–100-feature width. | Mostly dissolved by #9 (integer ABF now exact by construction). Added a wide fixture (`n_feat=25`) to the equivalence matrix; float-regime seed-dependence documented in the module docstring. |
| 9 | — | re-design (stage-1 alternative) | **Adopted design change:** rescore dirty rows via chunked dense `(chunk, n_feat)` buffers reusing the dense engine's exact expressions (`q[chunk].toarray() * (w/Q_safe)`; `r[:, Q<=0]=0`; `max`/`sum(axis=1)`), instead of `reduceat` over CSR data. Per-row scores are **bitwise-identical** to the oracle given identical `Q` (numpy's per-row pairwise reduction depends only on the row). | Kills the reduceat empty-segment and CAZ-floor subtleties outright; contract strengthens to "integer amounts → both rules exact"; remaining FP sites: initial `Q` (float amounts only) + float-cost curves. Cost: a chunk loop, ~26 MB peak buffer at `n_feat=100`, modest rescore slowdown (bench headroom is 16×). |
| 10 | LOW | science | "Negative weights are scientifically meaningless" is **false** — Zonation v3+ uses negatively weighted features for opportunity costs (Moilanen et al. 2011, doi:10.1890/10-1865.1). Raise still right (dense engine's negative-weight behavior is silently broken via the implicit-zero max). | Reworded to "not yet supported" citing the DOI; negative-weight support logged as a deferred feature. |
| 11 | LOW | science | "Zonation's own raster practice is warp in the hundreds-plus" overclaims (published: warp=1 deliberate, default 10, 100 as trade-off; Zonation's raster device is **edge removal**, which pymarxan lacks — pre-existing v0.13 difference). | Docstring/warning reworded: warp 10–100 is documented Zonation practice as a time/refinement trade-off; `n//1000` framed as pymarxan performance advice; edge-removal difference noted in the module docstring. |
| 12 | LOW | grounding | 0-PU problem: dense returned a NaN curve row; sparse raises ZeroDivisionError. | Early clear `ValueError` for `n_pu == 0` (better than either); documented as the second deliberate behavior change; test added. |
| 13 | LOW | arch + re-design | Float-test rank bound (±1.5/n) unprincipled — a mid-run tie flip cascades arbitrarily; the bound is a determinism test in tolerance costume. | Replaced with exact-order assertion on the fixed seed + comment (on env upgrade: change seed, never widen). |
| 14 | LOW | re-design | Stored-zero CSR entries (explicit `amount=0.0` pvf rows) untested; `eliminate_zeros` was dead-path in the suite. | Equivalence fixture with an explicit zero row added. |
| 15 | LOW | science | §7 integer-exactness silently assumes sums < 2^53. | One sentence in §7 + docstring. |
| 16 | LOW | arch | Advisory has no parameter mute (S3b precedent has one); signature is frozen so none added. | Docstring + CHANGELOG name `warnings.filterwarnings` as the sanctioned mute. |
| 17 | LOW | science + re-design | CELF-deferral rationale recorded nowhere; two independent caveats found: (a) lazy lower-bound caching needs the nonnegativity validation; (b) monotonicity breaks at FP-residue feature extinction (a CAZ max term drops to 0 when Q_j crosses ≤ 0 while an FP-residue carrier remains); (c) a lazy heap cannot reproduce warp>1 frozen-batch semantics. | Recorded here + in the roadmap deferral entry. |

## Key confirmations (so we don't re-litigate)

- Dirty-set invalidation is **complete**: `delta_i` depends only on its own row, `w`, `c_i`, and
  `Q_j` of its own stored features; holders-of-changed-features is exactly the affected set
  (re-design probed extinct-mask transitions and ULP-underflow no-change subtractions — both safe).
- Argpartition boundary logic reproduces stable-argsort's first-k **including intra-batch emission
  order** (grounding: 320 tie-stress runs, 0 failures; re-design verified independently).
- Sequential per-cell `Q` subtraction in emission order is required and correct; batched
  subtraction would break the FP trajectory.
- `build_pu_feature_csr` contract verified at `problem.py:132-160` (canonical, fresh per call,
  bitwise `toarray()` equality incl. adversarial duplicates); `n_planning_units` is a `@property`
  (`problem.py:66`) and the Task-2 monkeypatch works verbatim; `STATUS_LOCKED_IN=2/OUT=3`;
  ZonationSolver passes `warp` (`zonation_solver.py:67`); complete caller set: solver, Shiny
  zonation panel (rule only), `__init__` export, one docstring mention.
- Bench conventions confirmed (`pyproject.toml:63,68`, `Makefile:4,10`, `ci.yml:48`);
  CHANGELOG `[Unreleased]` currently empty; `.github/copilot-instructions.md` has **no** zonation
  wording — plan's Task 5 Step 2 is a correctly-conditioned no-op.
- CAZ math preserved operation-for-operation and confirmed against the primary source
  (Moilanen et al. 2005, doi:10.1098/rspb.2005.3164, full text: remaining-total normalization,
  min-delta removal, cost division; Moilanen 2007, doi:10.1016/j.biocon.2006.09.008, real and
  unretracted). Warp = the real Zonation "warp factor" (de Mello et al. 2015,
  doi:10.1371/journal.pone.0133995). References verified via scite; none retracted.

## Outcome

Design spec and implementation plan patched in place (same commit as this doc):
kernel swap (#9), raw-input validation incl. finiteness (#2/#4), oracle smoothing branch +
fixture (#5), `importlib` fix (#1), §7 rewrite (#3/#7/#15), test hardening (#6/#12/#13/#14),
wording (#10/#11/#16). Execution may proceed (subagent-driven TDD).
