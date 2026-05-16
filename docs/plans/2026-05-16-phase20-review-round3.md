# Phase 20 — Round-3 review synthesis

**Date:** 2026-05-16
**Inputs:** four further parallel subagent reviews of the v3 plan from angles rounds 1+2 did not hit:

- **Failure-mode / observability** (`a3ad7e08fbccf56b3`) — what does a user see when things go subtly wrong: silent failures, hung states, warning visibility, typo handling, debugging UX.
- **Test design** (`ad44b2c3a860aef47`) — not whether tests exist but whether they're well-designed to catch regressions.
- **Backward compatibility / API stability** (`ab34bf91967011334`) — what breaks for v0.1 / v0.2.0a2 users; CHANGELOG draft.
- **Maintainability / six-months-from-now** (`ad0aea043617323eb`) — patterns that will rot; refactor timing.

Rounds 1+2 closed the parity bugs, performance blockers, validate-dead-code, NaN/coordinate guards, zone-solver no-ops, and writer gaps. Round 3 surfaced two CRITICAL items (SA crash visibility, warning filter behaviour), ~6 HIGH items, **one factual error in a round-2 task** (M5), and **one anti-test regression** introduced by the v3 patching. All resolved in this commit.

After this round, returns diminish. The plan is at the point where further review nibbles edges; v4 is the last patch round before Batch 1 execution.

---

## CRITICAL

### CR3 — SA worker thread crash mid-loop is invisible

**Found by:** Failure-mode (F1).

If anything inside `SepState.delta_penalty` raises (scipy `ValueError` from `pdist` on degenerate input, internal index-out-of-bounds from a corrupt `pu_to_sep_feats`, etc.), `run_panel._run`'s broad `except Exception` (run_panel.py:134-136) catches it but only sets `progress.status = "error"` and `progress.error = str(e)`. The progress bar stays at the last `pct` value; the user sees a stuck card with no traceback.

**v4 resolution (Task 17b):**

```python
except Exception as e:
    import traceback
    progress.status = "error"
    progress.error = f"{e!r}\n{traceback.format_exc()}"
    progress.message = "Solver failed (see Error below)"
```

Also applies to Phases 18 + 19 (same code path).

### CR4 — `warnings.warn` payload is half-functional for non-Shiny users

**Found by:** Failure-mode (F2).

The round-2 CR2 fix moved the warnings from `validate()` to `ProblemCache.from_problem` so they fire on every solve. But:
- Python's default `warnings` filter (`"default"`) deduplicates per `(message, category, module, lineno)`. Sequential solves on the same problem object → warning fires only the first time.
- `python -W error` mode → the warning becomes an exception, breaking solve. Plan never says this is intended.
- pytest with `filterwarnings = error::UserWarning` (strict-CI projects) → every test exercising a geographic-CRS sep problem fails.

**v4 resolution (pinned in Tasks 4, 10):** Use `warnings.warn(msg, UserWarning, stacklevel=2)` (already established by Phase 19's `compute_baseline_penalty`). Add explicit pytest assertion in `test_phase20_smoke.py`: `with pytest.warns(UserWarning): ProblemCache.from_problem(geographic_crs_problem)`. Document in `help_content.py` that pymarxan emits `UserWarning` from `ProblemCache.from_problem` and that strict-mode users should filter via `warnings.filterwarnings("always", category=UserWarning, module="pymarxan")`.

---

## HIGH

### H10 — Typo'd `spec.dat` column silently ignored

**Found by:** Failure-mode (F3).

`spec.dat` with `sepnnum=3` (typo) → `read_spec` accepts it silently as an unrecognised column; `sepnum` defaults to `1` (disabled). User wonders why their carefully calibrated 3-PU separation isn't doing anything. Same exposure for Phases 18 + 19 typos (`ptraget`, `targt2`, `clumptpe`).

**v4 resolution (new Task 0):** Add a known-column whitelist warning to `read_spec` covering all of Phase 18 + 19 + 20 columns. ~8 LOC, catches typos for all three constraint types in one shot:

```python
_KNOWN_SPEC_COLS = {"id", "name", "target", "prop", "spf", "ptarget",
                    "target2", "clumptype", "sepdistance", "sepnum"}
extra = set(df.columns) - _KNOWN_SPEC_COLS
if extra:
    warnings.warn(
        f"spec.dat has unrecognised columns {sorted(extra)} — these will "
        f"be ignored. Did you mean one of {sorted(_KNOWN_SPEC_COLS)}?",
        UserWarning, stacklevel=2,
    )
```

### H11 — `apply_feature_overrides` extension is uncovered

**Found by:** Test-design.

Round-2 M2 extended `_OVERRIDABLE_FIELDS` in `models/problem.py` to include `sepdistance` / `sepnum` / `target2` / `clumptype` / `ptarget`. **No test exercises this.** `ScenarioSet.clone_scenario({"sepnum": {1: 5}})` is the documented API for parameter sweeps; without a test, the extension can silently regress.

**v4 resolution (Task 14b — split out of Task 14):** Add `test_scenario_set_clones_with_sepnum_override` in `tests/pymarxan/models/test_problem.py`. ~15 LOC.

### H12 — Anti-test from design item 12 dropped during v3 patching

**Found by:** Test-design.

The v2 design's test-strategy item 12 said `"feat_sepnum == 0 is byte-identical to pre-Phase-20 — golden-fixture regression guard"`. The v3 implementation plan omits this task. Phase 19's parallel test (`TestTarget2ZeroIsRegressionFree::test_sa_objective_unchanged_when_target2_zero`) caught real bugs in the cache-mask compounding.

**v4 resolution (new Task 11c):** Add `test_sa_objective_unchanged_when_sepnum_disabled` in `tests/integration/test_phase20_smoke.py`. Solve same problem with and without `sepnum=1` column present; assert `SolverConfig(seed=42)` produces identical `Solution.objective`. This is the missing anti-test for the round-2 H1 compound mask. ~15 LOC.

### H13 — Bedrock test under-exercises sparse-density case

**Found by:** Test-design.

Plan's 200 random flips on one "mixed sep-active and inactive features" problem. On sparse-density problems (most flips touch zero sep-active features), the bedrock returns 0 trivially and exercises nothing. Phase 19's `ClumpState` bedrock parametrized over multiple clumptypes and PU counts.

**v4 resolution (Task 11):** Parametrize bedrock over `(n_pu, sep_density, seed)` with at least one configuration where `sep_density=1.0` (every PU contributes to a sep-active feature) and one where `sep_density=0.2`. Catches future incremental-KD-tree drift that the v1 full-recompute hides.

### H14 — `det_spf` mask is duplicated across two cache methods

**Found by:** Maintainability (#10a).

Round-1 H1 added the compound mask `det_spf = feat_spf * (feat_target2 <= 0) * (feat_sepnum <= 1)` to BOTH `compute_full_objective` (cache.py:379) AND `compute_delta_objective` (cache.py:483). 100 lines apart in the same file. A future maintainer adding Phase 22's mask will edit only one site and produce a silent objective/delta divergence.

**v4 resolution (Task 10):** Extract `def _det_spf(self) -> np.ndarray` as a `functools.cached_property` on `ProblemCache`. Both methods call it. Single source of truth. ~5 LOC, kills the whole class of "edit one of two" bug.

### H15 — Inverse-index pattern is invisible from outside the SA loop

**Found by:** Maintainability (#10b).

Phase 19's `feat_uses_pu` and Phase 20's `pu_to_sep_feats` are built at `from_problem` time, live on the frozen cache, and are consulted only by the matching State class. Phase 21's importance scoring will want a similar inverse index ("for each PU, which features get an importance contribution"); the next maintainer will write a fresh `np.where` per flip → tests pass, production benchmarks regress 5×.

**v4 resolution (Task 10):** Add a short module-level docstring to `ProblemCache` calling out the "inverse-index discipline" — when adding a constraint type, precompute the inverse PU→feature index at `from_problem` time, never inside the SA loop. ~6 lines, pure docs.

### H16 — No programmatic capability-matrix test

**Found by:** Maintainability (#10c).

Phase 20 adds `supports_separation()` to four base solvers AND overrides to `False` on four zone solvers AND raises `NotImplementedError` inside each zone `.solve()`. Twelve touch-points for one capability. Phase 21 importance will skip the zone override + zone raise because importance is "obviously" zone-safe — until it isn't.

**v4 resolution (new Task 18c):** Add `tests/integration/test_solver_capability_matrix.py` programmatically asserting every solver × every constraint × {supports + handles | doesn't-support + raises}. ~30 LOC, brittle bookkeeping wants a regression test.

---

## MEDIUM

### M7 — Round-2 task M5 description is factually wrong

**Found by:** Compatibility (§3).

Round-2 M5 claimed `mip_clump_strategy` "currently accepts any string, only fails at solve time on big_m". **This is wrong.** `src/pymarxan/solvers/mip_solver.py:57-61` already validates set-membership at `__init__` (raises `ValueError`).

**v4 resolution (Task 13):** Fix the task description. The actual change is wording-tightening for consistency with `mip_sep_strategy`'s error text, plus a shared helper:

```python
def _validate_mip_strategy(name, value, allowed, rejected_with_reason=None): ...
```

Used by all three strategy kwargs. Round-2's M5 "Phase 19 housekeeping" wording overstated the gap. Maintainability agent agrees: keep the kwargs separate (each has a different valid set), just fix the validation drift via shared helper.

### M8 — `build_solution` try/except is too broad

**Found by:** Failure-mode (F8).

Round-2 H2 wrapped `evaluate_solution_separation` in `try/except ValueError`. But internal bugs in `count_separation` raising `ValueError("internal state desync")` get swallowed as "no coordinates" — `sep_shortfalls = None`, user sees None and thinks separation wasn't configured.

**v4 resolution (Task 9 + Task 5):** Define a custom exception subclass in `pymarxan.solvers.separation`:

```python
class PUCoordinatesUnavailableError(ValueError):
    """PU coordinates required for separation but not derivable."""
```

`get_pu_coordinates` raises this specific class. `build_solution` catches only `PUCoordinatesUnavailableError`, not all `ValueError`. Other internal `ValueError`s propagate.

### M9 — "Targets met: 12/15" summary line is amount-only; ambiguous with separation

**Found by:** Failure-mode (F4).

`Solution.all_targets_met` is amount-based; doesn't consider sep gap. After Phase 20, `run_panel` shows "Targets met: 3/3" even when 2/3 features have separation shortfalls. Plan needs a forced decision: redefine semantics (breaks tests, Option B) or extend summary string (Option A, recommended).

**v4 resolution (Task 17b):** Adopt Option A. Extend `run_panel` summary line when applicable:

```
Cost: 1234.5, Targets met: 12/15, Sep gap: 2/3, Clump gap: 1/3, Prob gap: 0/5
```

Only show the constraint-specific tail when the corresponding `Solution.X_shortfalls is not None`. Document explicitly in the design Acceptance Criteria that `targets_met` remains amount-only.

### M10 — Float-boundary determinism warning half-implemented

**Found by:** Failure-mode (F5).

Round-2 H4 said "add a `validate()` warning when `sepdistance` matches grid spacing within 1e-9 tolerance". v3 documented this in design assumptions but did NOT port it into `ProblemCache.from_problem` (where the other warnings now live after round-2 CR2). Half-shipping is worse than nothing — users assume "no warning ⇒ safe".

**v4 resolution:** Pick one. Recommend **drop** — implementing it requires a min-pairwise-distance estimate over a sample of PUs, which adds complexity and false positives. Replace the design's documented "warning" promise with "users SHOULD set sepdistance strictly less-than or greater-than nominal grid spacing; matching values produce floating-point boundary non-determinism." Pure-doc commitment, no code.

### M11 — Test count under-shoots Phase 19 precedent

**Found by:** Test-design (#8).

v3 plan: +28 tests across ~5 files. Phase 19 added ~50 tests for comparable surface area. Math layer is most under-tested (collapsed to 3 functions vs Phase 19's 5 classes).

**v4 resolution:** Parametrize Tasks 6 + 7 (table-driven tests for `compute_sep_penalty` boundary behaviour and `count_separation` arrangements). +12 tests for ~$0$ extra LOC. Bring total to ~40.

### M12 — Performance test has no file home

**Found by:** Test-design (#7).

v3's "per-flip SepState under 200 µs benchmark" lives only as a Verification-section bullet.

**v4 resolution (new Task 11b):** Create `tests/benchmarks/bench_sep.py` mirroring `bench_sa.py`. Run via `make bench` (on-demand), not `make check`. Pins the round-2 CR1 memory-shape claim with an actual measurement.

---

## LOW / accepted as-is

- **F6** (no logging) — accept for now; `logger.debug` adds in Phase 20 are deferred to a v0.3 observability pass.
- **F9** (SepState thread-safety undocumented) — add a one-line "NOT thread-safe; private to solver call frame" docstring on SepState (and backport to ClumpState).
- **F10** (invalid-edit toast cell doesn't reset) — accept; users see the toast and re-enter.
- **Compat §4** (asymmetric column propagation) — accept; same shape as Phase 18 / 19. Document in CHANGELOG.
- **Compat §7** (one-way serialized-state compat across versions) — accept; same shape as Phase 18 / 19. Document in CHANGELOG.
- **Maintainability #1** (Solution → SolutionMetrics refactor) — defer to v0.3 / Phase 21.
- **Maintainability #2** (ProblemCache constraint blocks `_ProbBlock`/`_ClumpBlock`/`_SepBlock`) — **judgment call.** Maintainability agent recommends landing in Phase 20 Batch 3.5; compatibility agent shows the cost (~30 LOC mechanical refactor + downstream call-site rewrites). **v4 verdict: defer to Phase 21.** Phase 20 is already large; the refactor lands cleanly in Phase 21 alongside `_POST_HOC_EVALUATORS` registry. Track on v0.3 backlog.
- **Maintainability #4** (`build_solution` registry) — defer to Phase 21.
- **Maintainability #8** (rename phase-N smoke tests) — Phase 21 cleanup.

---

## CHANGELOG draft (from compatibility agent — adopted verbatim into v4 plan Task 18)

Compatibility agent provided a clean drop-in CHANGELOG entry covering Added/Changed/Notes sections. v4 plan adopts it without modification. Sketch saved at the bottom of `2026-05-16-phase20-implementation.md`.

---

## Patches landing in v4

- **`docs/plans/2026-05-16-phase20-design.md`** — assumption 11 added (separation module not re-exported at package level; matches Phase 19); R10/R11/R12 risks for warning visibility, capability matrix, test-density gap.
- **`docs/plans/2026-05-16-phase20-implementation.md`** — new Task 0 (read_spec column whitelist warning), new Task 11b (bench_sep.py), new Task 11c (sepnum=disabled byte-identical anti-test), new Task 14b (scenario clone with sepnum override test), new Task 18c (capability-matrix test). Task 4, 9, 10, 13, 17b, 18 patched for round-3 fixes. M5 task description corrected (factual error).

After v4, the plan is at execution-readiness. Confidence: 96 %. Further review at this stage would be diminishing returns; Batch 1 execution will surface any remaining issues against real code.
