# Tier C — raptr space targets — engineering design review synthesis

**Date:** 2026-07-15
**Lenses:** Codebase-grounding (RAN the Task-1 code), Architect. (Scientific lens done earlier:
`...-review.md`; independent-redesign skipped — the science was the novel part, now settled.)

## Verdict

**Task 1 (`compute_space_held`) is nearly right — 2 fixable bugs; Tasks 2-3 (SA/greedy) need
redesign** per the findings below. The grounding agent confirmed the core maths computes raptr's
`1 − WSS/TSS` correctly on well-formed data, and verified the reuse of `get_pu_coordinates` /
`pu_id_to_index` / `groupby("species")`. But the solver integration as planned has real defects.

## Task 1 fixes (fold + implement now)

- **BUG-A (blocker):** `SpaceSpec(attribute_columns=["env"])` with the default `include_geographic
  =True` still calls `get_pu_coordinates` → `PUCoordinatesUnavailableError` on a coord-less problem,
  and builds a 3-D `[x,y,env]` space, not env-only. *Fix:* when `attribute_columns` is given,
  `include_geographic` is only added if explicitly true AND coords resolve; simplest — geographic is
  used only when `attribute_columns is None` OR `include_geographic` is explicitly set. Make the
  env-only test pass `include_geographic=False`. ("PASS (7 tests)" → 6 tests.)
- **BUG-B (latent correctness):** `w = amount[:len(occ)]` misaligns weights to demand points when
  `pu_vs_features` has an unknown PU id **not at the end** (grounding proved 0.136 vs 0.8). *Fix:*
  build a boolean `keep` mask (`keep = [int(p) in idx for p in rows.pu]`), then `occ = idx[...keep]`
  and `w = amount[keep]` — the defensive pattern from `separation.py:274-278`.

## Tasks 2-3 redesign (fold into the plan; build as Phase B)

- **SpaceState home + shape:** lives in **`solvers/space_state.py`** (a `ProblemCache` companion, like
  `ClumpState`/`SepState`), NOT `adequacy/`. Drop the "or fold into adequacy/" ambiguity and the
  "mirror SepState exactly" framing — the plan's stateless `selected`-passing design is valid but
  *different* from SepState's `cache`-passing/`apply_flip` design; state that explicitly.
- **HIGH — no DataFrame in the hot loop:** `delta_penalty` must run a **precomputed numpy kernel**
  over the affected features' demand points — never the Task-1 `compute_space_held` (which re-groups
  the DataFrame + re-z-scores every call). Precompute per-feature demand-point positions / weights /
  `TSS` and a `pu_to_space_feats` inverse index at `from_problem`/`from_selection` time (the cache's
  stated inverse-index discipline, `cache.py:8-14`).
- **HIGH — space is ADDITIVE:** a feature has both an amount target and a space target; both
  penalties apply. Do **NOT** exclude space features from `_det_spf` (unlike clump/sep, which
  *replace* the amount penalty). State this so a worker doesn't copy the clump/sep `_det_spf`
  exclusion.
- **HIGH — greedy needs a two-phase redesign, not a `_score_pu` term:** `_score_pu` returns `None`
  once amount targets are met (`if not unmet: return None`) → the loop breaks before adding any
  space-only PU, and mixing a space term into the incommensurable HEURTYPE scales distorts rankings.
  *Redesign:* Phase 1 = existing HEURTYPE greedy to meet amount targets; **Phase 2** = keep adding
  the PU with the largest marginal space-penalty reduction until `space_held_f ≥ space_target_f ∀f`
  or no candidate improves. Gate Phase 2 on `space_active`. Update the design's stale "greedy is a
  clean full-objective fit" line.
- **MEDIUM — Solution reporting + `supports_space()`:** every prior constraint added a post-hoc
  `Solution` pair (`sep_shortfalls`/`sep_penalty`, …) populated by `build_solution` for *all* solver
  paths, plus a `supports_*()` gate. Add `Solution.space_held`/`space_penalty` + `build_solution`
  population + `supports_space()` (MIP/zone declare non-support / report the gap).
- **MEDIUM — config as columns:** `space_target` and `space_spf` are **`features` columns** (matches
  `spf`/`target2`/`sepnum`); `SpaceSpec` (a structured dataclass) is a **solver constructor arg**
  (like `HeuristicSolver(heurtype=…)`), NOT `problem.parameters` (flat scalars) and NOT double-listed
  on `SolverConfig`.
- **SA wiring precision:** stateless SpaceState needs the delta at the **temp-estimation** site
  (`simulated_annealing.py:183-193`) AND the main trial (`:249-252`), plus the init penalty (`:172`)
  — but **no** `apply_flip`/accept-site work (nothing to commit). The plan omitted the temp-est site.

## Recommended sequencing (given the redesign surface)

Ship **Task 1 (`compute_space_held`, bug-fixed)** as the clean, verified, low-risk first piece.
Build the SA `SpaceState` (precomputed kernel + inverse index) and the two-phase greedy + Solution
reporting as **Phase B** with these findings incorporated — a well-grounded follow-on rather than a
rushed integration.
