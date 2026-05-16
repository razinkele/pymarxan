# Phase 20 — Round-2 review synthesis

**Date:** 2026-05-16
**Inputs:** four further parallel subagent reviews of the **v2** plan (post-round-1 patches), focused on angles the first round didn't hit:

- **Adversarial** (`a0d492f61296442e1`) — hostile inputs, numerical edge cases, mutation/aliasing, crash points.
- **Integration** (`af855669e66bea0c5`) — code Phase 20 touches indirectly: zone solvers, RunMode pipeline, Marxan binary wrapper, scenarios, output writers, untouched UI modules.
- **UX/ergonomics** (`aa0b6dc89a8860348`) — actual researcher workflows end-to-end.
- **Performance** (`a65cd899eed60011b`) — concrete cost numbers across small/medium/large problem sizes.

Round-1 caught the two Marxan-parity bugs (hyperbolic penalty curve, PU-id greedy ordering) and three plan-grounding bugs. Round-2 caught **two more CRITICAL items** — both blockers — and ~12 HIGH items spread across correctness, integration, and UX. All are listed below with their resolution in the v3 plan.

---

## CRITICAL (blocking — must fix before any Batch 1 code)

### CR1 — Pairwise distance matrix is ambiguously specified; the obvious reading is unusable

**Found by:** Adversarial (A3) + Performance (§3). Agreed independently.

The v2 design says SepState uses "per-affected-feature full recompute via vectorised count_separation with a precomputed pairwise squared-distance matrix". This is **ambiguous between three options**:

| Option | Memory | Per-flip cost |
|---|---|---|
| (a) Global `(n_pu, n_pu)` on `ProblemCache` | n_pu² × 8 bytes — **20 GB at n_pu=50000, 800 MB at 10000** | O(k²) slice |
| (b) Per-call `pdist(coords[candidates])` (k × k) | k² × 8 bytes — bounded by selection, freed per call | O(k²) build + greedy |
| (c) Per-feature `(|P_j|, |P_j|)` precomputed once | sum `|P_j|² × 8` ≈ 4 GB at large | O(|P_j|²) once + amortised |

Performance agent shows option (a) is infeasible above 10k PUs. Adversarial agent shows that even option (b) read naïvely (allocate 200 MB per flip at n_pu=5000) trashes the allocator in the SA hot loop.

**v3 resolution (pinned in Task 7):**

> `count_separation` allocates `dist_sq = scipy.spatial.distance.pdist(coords[candidates], "sqeuclidean")` on the **candidate sub-array only**, then unpacks to a `k × k` symmetric matrix via `squareform`. `k = len(candidates)`, never `n_pu`. The matrix lives in the function frame and is freed when the call returns. With early-exit at `|kept| == sepnum`, the typical call doesn't even materialise the full `k × k` — only the first `sepnum` rows are needed.

This makes the per-flip allocation O(k²) where `k = |selected ∩ has-feature|`, which is bounded by the realistic feature footprint (rarely > 1000 even on large problems).

### CR2 — `validate()` warnings are dead code

**Found by:** UX agent (§1, §2).

The v2 plan emits `UserWarning` from `ConservationProblem.validate()` for (a) geographic-CRS + `sepdistance > 0` and (b) `sepdistance == 0 AND sepnum > 1`. But `validate()` returns `list[str]` of errors — it does NOT raise warnings — and it is only called by the Shiny upload path. A user who:

1. Uploads a project (validate runs; no separation set → no warnings).
2. Edits `sepnum=3, sepdistance=5000` in the Shiny feature_table grid.
3. Clicks Run.

…hits the SA loop with degree-distances and no surface signal. The warnings the v2 plan added are unreachable.

**v3 resolution (pinned in Task 10):** Move both warnings into `ProblemCache.from_problem`, which fires on every solve (not just on upload). Use `warnings.warn(..., UserWarning)`. Then in `run_panel.py`, wrap the solve in `warnings.catch_warnings(record=True)` and replay any captured warnings via `ui.notification_show(type="warning")` so the user actually sees them. This is the same surface the Phase 19 `clumping.cpp::computeBaseline` warning uses — established pattern.

---

## HIGH (must fix before merge; correctness or silent-no-op)

### H1 — Zone solvers silently no-op separation

**Found by:** Integration agent (§1, §9).

`ZoneSASolver`, `ZoneIterativeImprovementSolver`, `ZoneHeuristicSolver`, `ZoneMIPSolver` construct `Solution(...)` directly — they don't call `build_solution`. They will never populate `sep_shortfalls`/`sep_penalty` regardless of the problem's separation state. The plan says "per-zone separation is out of scope" but doesn't add a guard.

**v3 resolution (new Task 13b):** Override `supports_separation()` to return `False` on the four zone solver classes. Have each `.solve()` raise `NotImplementedError("Zone solvers don't honour SEPDISTANCE/SEPNUM; deferred to v0.3. Use the non-zone solvers or set sepnum<=1.")` when the problem has any sep-active feature. Mirrors the existing TARGET2 deferral for zones.

### H2 — `build_solution` crashes a heuristic-only no-geometry user

**Found by:** Adversarial (B5).

A user running `HeuristicSolver` on a problem with `sepnum > 1` but no geometry / xloc / yloc: the heuristic completes, then `build_solution` calls `evaluate_solution_separation` which calls `get_pu_coordinates` which raises `ValueError`. The solver "fails" after producing a perfectly good selection.

**v3 resolution (pinned in Task 9):** In `build_solution`, the sep block uses `try/except ValueError` — on failure, sets `sep_shortfalls = None, sep_penalty = None` and emits a `UserWarning("Separation evaluation skipped: no PU coordinates available")`. The deterministic part of the solution still returns.

### H3 — NaN centroids silently corrupt count_separation

**Found by:** Adversarial (B1).

`has_geometry()` accepts a GeoDataFrame where some geometries are empty (only requires `not is_empty.all()`). Empty geometries produce NaN centroids. NaN-in-comparison is always False, so an empty-geom PU is always **rejected** from any admitted set — silent under-counting → falsely-high penalty.

**v3 resolution (pinned in Task 5):** `get_pu_coordinates` raises `ValueError("PU geometry contains N empty rows at indices [...]; cannot compute centroids for separation")` if any row's centroid contains NaN. Same guard for `xloc`/`yloc` NaN.

### H4 — Float-boundary determinism on `sepdistance` matching grid spacing

**Found by:** Adversarial (B2).

`geometry.centroid` introduces floating-point noise (~1e-12) from polygon math. Two PUs with nominal distance exactly equal to `sepdistance` flip between accepted/rejected based on rounding direction — SA results become non-reproducible across machines.

**v3 resolution (documented in design assumptions):** Document that `sepdistance` should be set strictly less than or strictly greater than any nominal grid spacing. Don't add a fuzzy tolerance — that hides the issue. Add a `validate()` warning when `sepdistance` matches the grid spacing within 1e-9 relative tolerance (detectable by checking against the min positive pairwise distance for a sample of 100 PUs).

### H5 — `SepState.delta_penalty` outer scan is O(n_feat) per flip

**Found by:** Adversarial (A2).

The v2 plan says delta_penalty filters affected features via `np.where(cache.pu_feat_matrix[idx] > 0)[0]` per flip — that's a fresh O(n_feat) scan on every SA iteration. Phase 19's `ClumpState` doesn't precompute an inverse index either (verified — it loops over `feat_uses_pu`). But the Phase 19 inner cost dominates, hiding this. Phase 20 makes it visible.

**v3 resolution (pinned in Task 10):** Add `pu_to_sep_feats: list[np.ndarray] | None` to `ProblemCache`. Element `i` is the array of separation-active feature column indices that contain PU `i`. Populated once in `from_problem` when `separation_active`. `SepState.delta_penalty` iterates `cache.pu_to_sep_feats[idx]` — O(features-at-this-PU), not O(n_feat). Same precomputed-inverse trick benefits Phase 19 if backported later.

### H6 — Negative `sepdistance` silently sign-stripped

**Found by:** Adversarial (B10).

`read_spec` for `sepdistance` (Task 1) coerces dtype but does NOT validate non-negativity. `cache` stores `-5.0`, the squared-distance comparison `dist_sq >= 25.0` loses the sign — behaves as `sepdistance = 5.0`. Silent.

**v3 resolution (pinned in Task 1):** `read_spec` raises `ValueError("sepdistance values must be >= 0; got negative at feature_id [...]")` if any row is negative.

### H7 — mvbest / summary writers don't get separation columns

**Found by:** UX agent (§5) + Integration agent (§7).

Marxan's reference `mvbest.csv` includes `SepDistance_Met` / `Separation_Count` columns. pymarxan's `write_mvbest` (writers.py:179-226) doesn't emit any of them, and the v2 plan doesn't add them. Phase 19 had the same gap for `clump_short` — but Phase 20 compounds it.

**v3 resolution (new Task 16b):** Extend `write_mvbest` to emit `Separation_Count` and `Separation_Met` columns when `solution.sep_shortfalls is not None`. Backport the equivalent `Clump_Short` for Phase 19 in the same commit — costs nothing extra and closes the parity gap noted in round 1.

### H8 — `run_panel` lacks `sep_mip_notice` banner

**Found by:** UX agent (§1) + Integration agent (§7).

`run_panel.py` shows `probmode3_mip_notice` when MIP + PROBMODE 3 are combined. No equivalent banner for MIP + sep-active. Users running default `mip_sep_strategy="drop"` get the same "deterministic solve + post-hoc gap" experience but with no UI notice.

**v3 resolution (pinned in Task 17):** Add `sep_mip_notice` banner; mirrors `probmode3_mip_notice` line-for-line. ~25 LOC.

### H9 — `feature_table.py`: invalid-edit toast missing; help description not extended

**Found by:** UX agent (§4).

`validate_feature_edit` returns `None` on invalid input → `_save` silently drops the edit. No toast, no inline error. Already a Phase 19 bug (clumptype has the same behaviour). The v2 plan doesn't fix it.

Also: the `ui.p(...)` description block at feature_table.py:48-61 explains target2/clumptype but Phase 20 doesn't extend it.

**v3 resolution (pinned in Task 15):** Add `ui.notification_show(f"Invalid {column} value '{raw}'; expected {rule}", type="warning")` when `validate_feature_edit` returns `None`. Extend the `ui.p` description block with one sentence on `sepdistance`/`sepnum`. Both fixes apply to Phase 19 columns retroactively (zero-cost win).

---

## MEDIUM (worth doing in v1)

### M1 — Combined Phase 18+19+20 SA cost grows ~3-4×

**Found by:** Performance agent (§6).

Three constraint paths run serially in the SA inner loop — no fusion. With all three on, expect 3-4× per-flip cost growth at medium problems. Plan doesn't disclose.

**v3 resolution:** Add a note to the design's "Risks" section (R8) and to the CHANGELOG entry that Phase 20 alongside Phase 18+19 on the same problem produces measurable SA slowdown. Doesn't change correctness; just sets user expectations.

### M2 — `apply_feature_overrides` doesn't allow overriding `sepdistance`/`sepnum`

**Found by:** Integration agent (§5).

`_OVERRIDABLE_FIELDS = {"target", "spf", "prop"}` at problem.py:334. ScenarioSet sweeps over `sepnum=1..5` are blocked from using the standard scenario API.

**v3 resolution (pinned in Task 14):** Extend the set to `{"target", "spf", "prop", "target2", "clumptype", "ptarget", "sepdistance", "sepnum"}`. One-line fix; backports Phase 18 + 19 coverage too.

### M3 — Marxan-binary cross-validation test missing

**Found by:** Integration agent (§4).

Phase 20's whole purpose is Marxan parity, but Task 18 doesn't include a "run Marxan binary on the same problem, compare sep_counts" test. Easy add via `pytest.importorskip` / skipif-binary-unavailable.

**v3 resolution (new Task 18b):** Optional Marxan-binary integration test, skipif binary not available on PATH.

### M4 — `mip_sep_strategy="socp"` error message confusing

**Found by:** UX agent (§3).

The current text `"separation is combinatorial, not conic; 'socp' not applicable"` lands flat for a Phase 18 user who learned `mip_prob_strategy="socp"` was meaningful. Replace with: `"mip_sep_strategy='socp' is not valid — separation is a combinatorial constraint (greedy maximum independent set), not a conic/probabilistic one. Use 'drop' (default; gap reported on Solution.sep_shortfalls) or 'big_m' (deferred to a future phase)."`

### M5 — `MIPSolver` validation harmonisation

**Found by:** Adversarial (B3).

`mip_clump_strategy="socp"` is currently accepted at construction (no `__init__` validation). Phase 20 introduces strict `__init__` validation for `mip_sep_strategy`. Users get dual error models. Worth aligning Phase 19 in the same PR.

**v3 resolution (pinned in Task 13):** Add `mip_clump_strategy` validation alongside `mip_sep_strategy`. Phase 19 housekeeping; 5 LOC.

### M6 — Float `sepnum` silently truncates

**Found by:** Adversarial (B8).

`"2.7"` in spec.dat → `int(2.7) = 2`. No error. Phase 19's clumptype does the same.

**v3 resolution (pinned in Task 2):** `if not (df["sepnum"] == df["sepnum"].astype(int)).all(): raise ValueError("sepnum must be integer; got non-integer at feature_id [...]")`.

---

## LOW (defer or note)

- **L1** — `sepdistance = inf` accepted (B9). Add `validate()` warning when `sepdistance > planning-region diameter`.
- **L2** — Hide `sepdistance`/`sepnum` columns when all-default in feature_table (UX §4). Avoids cluttering the editor.
- **L3** — `help_content.py` should note the unit mismatch: `sep_shortfalls` is int count, `clump_shortfalls` is float amount, `prob_shortfalls` is float probability (UX §5).
- **L4** — `sepnum > n_pu` clamping at `ProblemCache.from_problem` (Adversarial C3, C4).
- **L5** — README "Marxan-classic features supported" paragraph covering Phases 18+19+20 (UX §6). v0.2.0 final is a publishable milestone.

---

## Patches landed in v3

- **`docs/plans/2026-05-16-phase20-design.md`** — risks R5/R6 updated with the round-2 findings; CR1 pairwise-matrix shape pinned.
- **`docs/plans/2026-05-16-phase20-implementation.md`** — Tasks 1, 2, 5, 7, 9, 10, 13, 15, 17 updated in-place; new Tasks 13b (zone-solver guard), 16b (mvbest sep columns), 18b (Marxan-binary cross-validation), 14 (apply_feature_overrides extension).

After v3 patches the plan resolves both round-1 + round-2 findings. Confidence to execute: 95 %.
