# Zone PU locks and zone restrictions (pulock + puzone) — design

Issue: razinkele/pymarxan#6 part 2 ("no way to lock a PU into a named zone").
Status: **rev 2** — rev 1 (`1eed927`) approved in chat 2026-10-11 (scope "pulock + puzone";
puzone PUs may NOT be unassigned); patched after the 4-lens review
(`2026-10-11-zone-pu-locks-review.md`, findings 1–16).

## Problem

Zone solvers support only pymarxan's status convention: status 2 forces a PU into the first
sorted zone, status 3 forces it out of every zone (`diagnostics._pu_zone_rules`, plus separate
parsers in `solver.py`, `heuristic.py`, `iterative_improvement.py`). There is no way to say
"this existing restoration site must be in the *restore* zone" (the ROWER need), nor to restrict
a PU to a subset of zones. MarZone provides both through two input files.

## MarZone reference semantics (`Marxan-source-code/marzone` @ `85082e3`)

- **`pulock.dat`** (`PULOCKNAME`; header `puid, zoneid`, read positionally, header optional).
  `pu.hpp:708-750` `LoadPuLock`: rows for a PU not in pu.dat are ignored with a warning; stored
  in a map, so a repeated PU keeps its **last** row. `pu.hpp:483-496` `PersistPuLock`: the
  surviving lock is then skipped if its zone is unknown (**last-wins first, then the zone
  filter**).
- **`puzone.dat`** (`PUZONENAME`; same format, one row per allowed zone). `pu.hpp:752-790`
  `LoadPuZone` (unknown PU ignored with a warning); `pu.hpp:498-526` `PersistPuZone`: unknown
  zones skipped silently; rows are not de-duplicated; a pulocked PU keeps its lock; a PU left
  with exactly one zone is **fatal** ("Do not use this file to lock planning units!").
- **Precedence and moves.** `reserve.hpp:100-103`: "pu lock takes precedence above all"; a
  puzone PU always takes one of its listed zones (`pu.hpp:282-311` `RtnValidZoneForPu`; MarZone
  has no unassigned state). MarZone's II candidate filter has an index bug — not copied.
- **Status.** MarZone does **not** read a pu.dat status column (`ReadPUData`,
  `pu.hpp:641-700`). pymarxan's status 2/3 locks are a pymarxan extension with no MarZone
  precedent.
- **Loading.** MarZone loads the files only when the parameter is set and exits if a named
  file cannot be opened.

## Design

### 1. Data model and I/O

- `ZonalProblem` gains `pu_locks: pd.DataFrame | None` and `pu_zones: pd.DataFrame | None`,
  `field(default=None, kw_only=True)` (verified to work with the dataclass inheritance), both
  with columns `pu, zone`. `copy_with` carries them.
- `zones/readers.py`: `read_pu_lock(path)`, `read_pu_zone(path)` read the first two columns
  **by position** (whitespace stripped from headers; a first line that is numeric is data, not
  a header), as integers, into columns `pu, zone`. `load_zone_project` reads `PULOCKNAME` /
  `PUZONENAME`: a **default** name (`pulock.dat` / `puzone.dat`) is loaded if present
  (consistency with the other zone files; documented pymarxan convenience); an **explicitly
  named** file that does not exist raises `FileNotFoundError` (MarZone exits).
- `zones/writers.py`: `write_pu_lock(df, path)`, `write_pu_zone(df, path)` emit the MarZone
  header `puid,zoneid` (round-trip through the readers).
- Exported from `pymarxan.zones`.

### 2. One per-PU rule, on the model

```python
class PUZoneRule(NamedTuple):
    allowed: frozenset[int]        # zones the PU may take
    may_be_unassigned: bool        # zone 0 allowed
    start: int | None              # initial zone for heuristics (None = solver's choice)

    @property
    def forced(self) -> bool:      # exactly one zone and must be assigned
        return len(self.allowed) == 1 and not self.may_be_unassigned
```

(Replaces v0.37's `(allowed, forced)`; built by keyword everywhere.)

New `ZonalProblem` methods (one parse of the tables, like `zone_target_lookup`):
- `pu_lock_lookup() -> dict[int, int]` — known PUs only; last row wins; then entries whose zone
  is unknown are dropped (the PU falls through to lower rules).
- `pu_zone_lookup() -> dict[int, frozenset[int]]` — known PUs; known zones only; distinct.
- `pu_zone_rules() -> list[PUZoneRule]` — `planning_units` order; **the only place** that
  interprets status / pulock / puzone. `diagnostics._pu_zone_rules` becomes a thin alias.
- `pu_lock_issues()` — row-level issues for `validate()` and the warning (below).

Rule table, first matching row wins:

| Condition | `allowed` | `may_be_unassigned` | `start` |
|---|---|---|---|
| PU in `pu_lock_lookup` | `{zone}` | False | zone |
| status 3 | `∅` | True | 0 |
| status 2 and PU in `pu_zone_lookup` | puzone set | False | `min(set)` |
| status 2 | `{min(zone_ids)}` | False | that zone |
| PU in `pu_zone_lookup` | puzone set | False | `min(set)` |
| free | all zones | True | status 1 → `min(zone_ids)`, else None |

Status 2 + puzone means "must be assigned, to one of the listed zones" so the forced zone can
never fall outside the puzone list. A single-zone puzone set is a lock (lenient vs MarZone's
fatal error; reported, below).

### 3. Validation and warnings

`ZonalProblem.validate()` errors (never raises), from `pu_lock_issues()`:
- missing `pu`/`zone` columns; rows naming an unknown PU or zone;
- a PU repeated in `pu_locks` ("last-wins"); duplicate `(pu, zone)` rows in `pu_zones`;
- real conflicts only: status 3 with a pulock or puzone rows; a pulock zone not in the same
  PU's puzone list; status 2 with a pulock is **not** an error (they agree);
- a PU whose puzone rows leave exactly one known zone (MarZone aborts; pymarxan locks).

Warnings: new base `ZoneInputWarning(UserWarning)`; `ZoneTargetWarning` and new
`ZoneLockWarning` subclass it (backward compatible). One dispatcher
`warn_zone_input_issues(problem, solver_name)` replaces the per-solver
`warn_zone_target_issues` call (the old name stays as a wrapper) and emits at most one warning
of each kind per solver call. All three classes exported.

### 4. Solvers (every solver reads `problem.pu_zone_rules()`)

Shared helper `rule_violations(rules, assignment) -> list[int]` (PU indices whose zone breaks
their rule) — used by tests on every solver's output.

- **MIP**: `x[i,z].upBound = 0` for `z ∉ allowed`; `assign_{pid}`: `Σ_z x[i,z] == 1` if
  `not may_be_unassigned`, else `<= 1`. (No test inspects constraint names.)
- **SA**: per-PU option array `opts[i] = sorted(allowed)` (zone 0 never proposed, as today);
  for free PUs `opts[i]` is the shared `zone_options` array so the seeded RNG stream is
  unchanged. Not swappable: `allowed = ∅` (assigned 0) and forced PUs (assigned their zone).
  One proposal helper used by **both** the temperature-estimation loop and the annealing loop.
  Initial assignment: `rule.start` if set, else a random option.
- **Heuristic**: initial assignment = `rule.start` if set, else 0 when `may_be_unassigned`;
  candidates = `sorted(allowed)`; forced / `∅` PUs not swappable.
- **II**: `solve()` seeds each swappable PU at `rule.start` if set, else `min(allowed)`;
  candidates in **all** passes (removal, addition, swap) = `sorted(allowed)` plus `0` iff
  `may_be_unassigned`. `improve()` first **repairs** the incoming assignment (forced → its zone,
  `∅` → 0, a zone outside `allowed` or 0-when-must-assign → `min(allowed)`), then runs the passes.
- **Diagnostic**: `_best(rule, values)` = max over `allowed`, combined with 0 when
  `may_be_unassigned`; forced uses its value as-is. Unchanged for today's statuses.

### 5. Performance

Default path: a seeded golden test — no locks/restrictions → identical assignment and objective
to v0.37 for SA, heuristic, II. Comparative bench: SA per-flip with ~50 % of PUs
puzone-restricted vs unrestricted ≤ 1.05×.

## Testing

1. **Rules table** — every row and precedence pair, including status 2 + puzone and
   `pu_locks` rows `(1, 2), (1, 99)` → PU1 falls through (last-wins then zone filter).
2. **I/O** — MarZone's `pulock_test1.dat` / `puzone_test1.dat` content verbatim (header
   `puid, zoneid` with spaces); headerless file; writers emit `puid,zoneid` and round-trip;
   `load_zone_project` default names; explicitly named missing file → `FileNotFoundError`.
3. **Anchors (verified by enumeration and prototype MIP):** pulock PU1 → 2 moves the MIP optimum
   from `(1, 2, 2)` at 7.0 to `(2, 1, 2)` at 8.0. **Must-assign variant:** anchor + PU4 (no
   features, zone costs (3, 2)): unrestricted optimum `(1, 2, 2, 0)` at 7.0; puzone PU4 ∈ {1, 2}
   → `(1, 2, 2, 2)` at 9.0.
4. **All four solvers** (+ `improve()` fed an out-of-rule assignment, and RUNMODE 3/5): every
   returned solution has `rule_violations == []` on the must-assign variant and a lock case.
5. **validate()** errors and one `ZoneLockWarning` per solver call; `ZoneInputWarning` catches
   both kinds.
6. **Regression** — the two `TestRules` tests in `test_zone_diagnostics.py` are rewritten for the
   new rule fields (listed change); all other zone suites unchanged; seeded golden test (§5);
   `tests/data/simple` MIP = 35.0; MarZone anchor `(1, 2, 2)` at 7.0 without locks.

## Compatibility

Additive: two optional fields, readers/writers, `ZoneInputWarning` / `ZoneLockWarning`,
`pu_*` methods. `PUZoneRule` is internal (not in `__all__`); its fields change. Behaviour
without pulock/puzone is unchanged (golden test). Version: minor bump (0.38.0).

## Out of scope

- MarZone's `CalcPenalties` pre-counts pulocked PUs' amounts (`marzone.cpp:769-782`); pymarxan's
  `spf × shortfall` objective has no such baseline, so locks do not change penalty scaling —
  the same approximation as today.
- The `-1` / multi-file variants (`ZONETARGET2NAME`, `ZONECONTRIB2/3NAME`) — separate issue.
