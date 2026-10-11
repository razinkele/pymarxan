# Zone-target normalization, validation and MIP infeasibility diagnostic — design

Issue: razinkele/pymarxan#7 (follow-on from #2 / v0.36.0). Status: design approved in chat
2026-10-11 (option "MarZone semantics"); pending 4-lens review.

## Problem

`ZonalProblem.zone_targets` rows that name an unknown zone or feature, or that repeat a
`(zone, feature)` pair, get three different treatments today. On the #2 spec anchor with
assignment `(1, 2, 2)`:

| Case | `ZoneMIPSolver` | `compute_zone_objective` | `ZoneProblemCache` |
|---|---|---|---|
| Duplicate (zone, feature) row | raises `PulpError` (duplicate constraint name) | — | — |
| Unknown zone id | `[]` | 17.0 | 7.0 |
| Unknown feature id | `[]` | 8.0 | 7.0 |

SA minimises the cache objective but reports the DataFrame objective, so on these inputs
`Solution.objective` disagrees with what the search minimised. No solver calls `validate()`,
so a validate-only fix would leave this inconsistency for every caller that skips it.

Separately, `ZoneMIPSolver` returns `[]` with no explanation when the model is infeasible. Since
v0.36.0 overall targets are hard constraints and partial contribution tables default unlisted
pairs to 0, an unreachable target in a hand-built problem (which never passes through the loader's
partial-table warning) is easy to create and hard to find.

Finally, `zones/writers.py` imports the private `objective._compute_zone_achieved`.

## MarZone reference semantics (verified 2026-10-11)

`Marxan-source-code/marzone`, `zones.hpp`:

- `BuildZoneTarget` (≈ lines 95–158): for each row, `spec.LookupIndex(speciesid) == -1` →
  `continue // invalid species - skip`; `LookupIndex(zoneid) == -1` →
  `continue // ignore invalid zone ids`; otherwise
  `zoneTarget[spindex][zoneind].target = …` — plain assignment, so a repeated pair is
  **last row wins**. (`speciesid == -1` is MarZone's "all species" wildcard; pymarxan's reader
  already rejects it explicitly — unchanged, out of scope.)
- `PopulateZoneContributions` (≈ lines 605–640): identical skip-unknown / assignment (last-wins)
  semantics for the contribution table. pymarxan's `contribution_lookup()` is a dict
  comprehension (last-wins) and `contribution_matrix()` skips unknown ids — **already
  MarZone-consistent**; only validation reporting is missing there.

## Design

### 1. Shared normalizer — `normalized_zone_targets`

New method on `ZonalProblem` (`zones/model.py`, next to `contribution_lookup`):

```python
def normalized_zone_targets(self, *, warn: bool = True) -> pd.DataFrame | None:
    """zone_targets with MarZone load semantics (zones.hpp BuildZoneTarget).

    Drops rows whose zone id is not in ``zone_ids`` or whose feature id is not in
    ``features["id"]``, then keeps the LAST row of each (zone, feature) pair; surviving rows
    keep their input order. ``None`` when ``zone_targets`` is ``None``.
    Emits one UserWarning naming counts (+ first five offending rows) when ``warn`` and
    anything was dropped or collapsed; silent otherwise.
    """
```

- Columns are passed through unchanged (including `targettype`, if present); index is reset.
- Row order: the kept rows in their original relative order (`drop_duplicates(keep="last")`
  after the unknown-id filter). Deterministic, and the order the MIP uses for constraint names.
- Not cached on the instance (problems are mutable dataclasses; `copy_with` must not carry stale
  state). Each consumer calls it once per solve/build — O(rows), negligible.
- Warning text distinguishes the two cases, e.g.
  `"zone_targets: ignored 2 row(s) with unknown zone/feature ids (MarZone skips them): zone 9,
  feature 1; …"` and `"zone_targets: 1 duplicate (zone, feature) pair(s) resolved last-wins
  (MarZone): zone 1, feature 2"`.

**Consumers switched to it** (every read of `problem.zone_targets` that drives an objective,
constraint or met-flag):

| Site | Today | After |
|---|---|---|
| `mip_solver._zone_target_exprs` / `_zone_target_penalty` / `_add_zone_target_constraints` (`mip_solver.py` ≈366–430) | iterate raw rows | iterate normalized rows (built once in `solve`, passed down or recomputed with `warn=False`) |
| `objective.check_zone_targets`, `objective.compute_zone_shortfalls` | iterate raw rows | iterate normalized rows (`warn=False` — these run per solution; the solver already warned) |
| `cache.ZoneProblemCache.from_problem` zone-target matrix (`cache.py` ≈180–188) | silently skips unknown ids; last-wins via overwrite | reads normalized rows (warns once at cache build) |

The warning is emitted once per solve: by the MIP in `solve()`, and by the cache build for
SA / heuristic / II (all build a `ZoneProblemCache`). Objective helpers pass `warn=False`.

Effect: duplicate rows no longer raise `PulpError`; unknown-id rows contribute nothing in all
three components; the issue's table collapses to one answer per row.

### 2. `validate()` reporting

`ZonalProblem.validate()` (`zones/model.py:137`) adds, only when the required columns exist:

- `zone_targets`: unknown zone ids and unknown feature ids (one error, first five shown, same
  format as the existing unknown-contribution error), and duplicate `(zone, feature)` pairs
  (one error, first five shown, noting "resolved last-wins by the solvers").
- `zone_contributions`: duplicate `(feature, zone)` rows (one error, same format). Unknown ids
  are already reported.

These are **errors** (consistent with the existing unknown-contribution check): the data is
almost certainly a mistake. The solvers stay permissive under MarZone semantics; `validate()`
remains advisory (it returns a list, never raises).

### 3. Infeasibility diagnostic — `diagnose_zone_infeasibility`

New public function in `zones/objective.py`, exported from `pymarxan.zones`:

```python
def diagnose_zone_infeasibility(problem: ZonalProblem) -> list[str]:
    """Targets that no assignment can meet, each checked on its own (necessary conditions).

    Overall target f: UB_f = Σ_i max(0, max_{z ∈ allowed(i)} amount[i,f] × contribution[z,f]).
    Zone target (k, f): UB_kf = Σ_{i : k ∈ allowed(i)} amount[i,f] × w[k,f]   (w > 0 terms only).
    allowed(i) = {} for status 3, {zone_ids[0]} for status 2, all zones otherwise
    (mirrors ZoneMIPSolver's locked-in/out constraints). A target t is reported when
    UB < t × MISSLEVEL beyond the shared 1e-9 relative tolerance (``_meets``).
    """
```

- Amounts: the same matrix the MIP's overall/zone constraints use (implementation must read
  that site and use the identical source, e.g. `problem.build_pu_feature_matrix()`), so the
  bound is exactly the MIP's.
- Zone targets come from `normalized_zone_targets(warn=False)`.
- `max(0, …)` because a PU may be left unassigned (MIP: `Σ_z x[i,z] ≤ 1`), contributing 0;
  negative contributions are never forced on a non-locked PU. For a locked-in PU the single
  allowed zone's value is used as-is (it may be negative — it is forced).
- Message per target, e.g. `"overall target for feature 3 is 120.0 but at most 85.5 is
  reachable (contributions × amounts, locked-out PUs excluded)"` /
  `"zone target (zone 2, feature 1) is 40.0 but at most 12.0 is reachable"`.
- Returns `[]` when every target is individually reachable.

`ZoneMIPSolver.solve`, at its `return []` for infeasible/no-values status
(`mip_solver.py` ≈205), emits one `UserWarning`:

- if the diagnostic is non-empty: `"ZoneMIPSolver: no feasible solution. Unreachable
  targets: …"` (first five, then count);
- else: `"ZoneMIPSolver: no feasible solution (status …). Every target is reachable on its own,
  so targets compete for the same planning units, or the solver hit its time limit."`

Return value stays `[]` (no API change). Not proving joint infeasibility (no IIS) — out of scope.
`validate()` does **not** call the diagnostic: unreachable targets are legitimate for the
penalty-based solvers (SA/heuristic/II).

### 4. Public `compute_zone_achieved`

Rename `objective._compute_zone_achieved` → `compute_zone_achieved`; keep
`_compute_zone_achieved = compute_zone_achieved` as a private alias (internal callers and any
downstream code keep working); export from `pymarxan.zones.__all__`; switch `zones/writers.py`
and internal callers to the public name. Docstrings that reference the private name are updated.

## Testing

All under `tests/pymarxan/zones/`, TDD, using the existing MarZone anchor fixture
(`marzone_anchor.py`) where possible.

1. **Agreement test (the issue's table):** for duplicate / unknown-zone / unknown-feature
   variants of the anchor with assignment `(1, 2, 2)`: `compute_zone_objective` ==
   `ZoneProblemCache` full objective, and `ZoneMIPSolver` solves without raising; each equals the
   objective of the hand-normalized problem.
2. **MarZone semantics pinned:** last-wins (two rows, different targets → second governs);
   unknown ids dropped; the warning fires once with counts; no warning on clean input.
3. **validate():** each new error appears for the bad input and not for the anchor; existing
   validate tests unchanged.
4. **Diagnostic:** (a) feasible anchor → `[]`; (b) overall target above the bound because a
   PU is locked out → named; (c) partial contribution table leaving a feature at 0 → named;
   (d) locked-in PU forced into `zone_ids[0]` counts only there; (e) MIP infeasible-but-each-
   reachable case (two zone targets competing for one PU) → `[]` and the solver's warning uses
   the "compete" wording; (f) MIP warning names the target in case (b).
5. **Public alias:** `compute_zone_achieved is _compute_zone_achieved`; importable from
   `pymarxan.zones`.
6. **Regression anchors:** `tests/data/simple` MIP = 35.0 and the MarZone anchor results
   unchanged; `make check` green; zone per-flip bench non-regressing (cache build only changed).

## Compatibility

- **Behaviour change (CHANGELOG, breaking for hand-built problems):** duplicate zone-target rows
  no longer raise `PulpError` in `ZoneMIPSolver`; they resolve last-wins with a warning.
  Unknown-id rows no longer make the MIP return `[]` nor add a phantom penalty in
  `compute_zone_objective`.
- No change for loader-produced problems with clean files. Public API is additive
  (`normalized_zone_targets`, `diagnose_zone_infeasibility`, `compute_zone_achieved`).
- Version: minor bump (0.37.0).

## Out of scope

- MarZone `speciesid = -1` wildcard (reader keeps rejecting it).
- Joint-infeasibility explanation (IIS / conflict refiner).
- #6 part 2 (lock into a named zone) — separate issue; the diagnostic's `allowed(i)` must be
  updated when it lands.
