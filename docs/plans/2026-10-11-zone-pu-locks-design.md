# Zone PU locks and zone restrictions (pulock + puzone) — design

Issue: razinkele/pymarxan#6 part 2 ("no way to lock a PU into a named zone").
Status: **rev 3** — rev 1 (`1eed927`) approved in chat 2026-10-11 (scope "pulock + puzone";
puzone PUs may NOT be unassigned). Rev 2 (`369faa8`) absorbed review round 1; rev 3 absorbs
round 2 (`2026-10-11-zone-pu-locks-review.md`, R2-1 … R2-15).

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
  filter**); the PU then falls through to puzone.
- **`puzone.dat`** (`PUZONENAME`; same format, one row per allowed zone). `pu.hpp:752-790`
  `LoadPuZone` (unknown PU ignored with a warning); `pu.hpp:498-526` `PersistPuZone`: unknown
  zones skipped silently; rows not de-duplicated; a pulocked PU keeps its lock (a lock outside
  the puzone list is accepted — the lock wins); a PU left with exactly one *row* is **fatal**
  ("Do not use this file to lock planning units!"); a PU left with **no** known zone has
  `numZones == 0` and is unrestricted (`pu.hpp:298-305`).
- **Precedence and moves.** `reserve.hpp:100-103`: "pu lock takes precedence above all"; a
  puzone PU always takes one of its listed zones, drawn at random for the initial solution
  (`reserve.hpp:110-113`) and by `RtnValidZoneForPu` (`pu.hpp:282-311`); MarZone has no
  unassigned state. MarZone's II candidate filter has an index bug — not copied.
- **Status.** `spustuff.status` exists but is never set: `ReadPUData` (`pu.hpp:635-706`) does not
  read a status column. pymarxan's status 1/2/3 handling is a pymarxan extension.
- **Loading.** MarZone loads each file only when its parameter is set; a named file that cannot
  be opened, or is empty, is fatal. Header detection: `is_like_numerical_data` (a line made only
  of digits, signs, `.`, `e`/`E`, separators and whitespace is data).

## Design

### 1. Data model and I/O

- `ZonalProblem` gains `pu_locks: pd.DataFrame | None` and `pu_zones: pd.DataFrame | None`,
  `field(default=None, kw_only=True)` (verified to work with the dataclass inheritance), both
  with columns `pu, zone`. `copy_with` carries them.
- `zones/readers.py`: `read_pu_lock(path)`, `read_pu_zone(path)` — a dedicated parser (not
  `io.readers._read_dat`, which drops the first row of a headerless file):
  - encoding `utf-8-sig`; CRLF accepted; separators `,`, tab, or whitespace (same detection as
    the other zone readers);
  - the first line is a header unless it passes MarZone's `is_like_numerical_data` character
    test; the first two columns are taken **by position** into `pu, zone`;
  - a 0-byte file → `ValueError` naming the file; a header-only file → empty DataFrame;
  - a blank or non-integral cell → `ValueError` naming the file and line.
- `load_zone_project` reads a file **only when** `PULOCKNAME` / `PUZONENAME` is in the
  parameters (MarZone parity; no default names). A named file that does not exist →
  `FileNotFoundError`; a named empty file → the reader's `ValueError`. (The sibling zone files
  keep their default-name behaviour; this difference is documented.)
- `zones/writers.py`: `write_pu_lock(df, path)`, `write_pu_zone(df, path)` emit the MarZone
  header `puid,zoneid` and round-trip through the readers.
- Exported from `pymarxan.zones`: `read_pu_lock`, `read_pu_zone`, `write_pu_lock`,
  `write_pu_zone`, `ZoneInputWarning`, `ZoneLockWarning`.

### 2. One per-PU rule, on the model

In `zones/model.py` (with `rule_violations`; `diagnostics` re-exports `PUZoneRule`, and
`diagnostics._pu_zone_rules` becomes a thin alias of the method below):

```python
class PUZoneRule(NamedTuple):
    allowed: frozenset[int]        # zones the PU may take
    may_be_unassigned: bool        # zone 0 allowed
    start: int | None              # deterministic initial zone, or None (solver's choice)

    @property
    def forced(self) -> bool:      # exactly one zone and must be assigned
        return len(self.allowed) == 1 and not self.may_be_unassigned
```

New `ZonalProblem` methods (one parse of the tables, ids parsed with `_known_id` as in
`zone_target_lookup`: non-integral / NaN → unknown):
- `pu_lock_lookup() -> dict[int, int]` — rows for known PUs; last row per PU wins; then a lock
  whose zone is unknown is dropped (the PU falls through). `{}` if the table is `None` or lacks
  `pu`/`zone`.
- `pu_zone_lookup() -> dict[int, frozenset[int]]` — known PUs; known zones only; distinct;
  **a PU with no known zone is omitted** (falls through, MarZone `numZones == 0`). `{}` as above.
- `pu_zone_rules() -> list[PUZoneRule]` — `planning_units` order; **within `pymarxan.zones`,
  the only place that interprets status / pulock / puzone**. Invariant (asserted): no rule has
  `allowed == ∅` with `may_be_unassigned == False`.
- `pu_lock_issues() -> list[str]` — the issue list in §3.

Rule table, first matching row wins:

| Condition | `allowed` | `may_be_unassigned` | `start` |
|---|---|---|---|
| PU in `pu_lock_lookup` | `{zone}` | False | zone |
| status 3 | `∅` | True | 0 |
| status 2 and PU in `pu_zone_lookup` | puzone set | False | None (forced zone if the set has one) |
| status 2 | `{min(zone_ids)}` | False | that zone |
| PU in `pu_zone_lookup` | puzone set | False | None (that zone if the set has one) |
| free | all zones | True | `min(zone_ids)` if status 1, else None |

### 3. Validation and warnings

`pu_lock_issues()` produces one message per class (first five items shown), each stating what
the solvers do:

| Issue | Solvers follow | `validate()` error | `ZoneLockWarning` |
|---|---|---|---|
| `pu_locks` / `pu_zones` missing `pu`/`zone` columns | table ignored | yes | yes |
| rows naming an unknown PU, zone, or a non-integral id | row skipped | yes | yes |
| a PU repeated in `pu_locks` | last row (and, if its zone is unknown, "PU not locked") | yes | yes |
| duplicate `(pu, zone)` rows in `pu_zones` | collapsed | yes | yes |
| a PU whose puzone rows name only unknown zones | PU unrestricted | yes | yes |
| a PU (not pulocked) restricted to one distinct known zone | treated as a lock ("MarZone rejects single-zone puzone") | yes | yes |
| status 3 with a pulock | the lock wins | yes | yes |
| status 3 with puzone rows | status 3 wins | yes | yes |
| a pulock zone outside the same PU's puzone list | the lock wins (MarZone accepts) | no | yes |

Status 2 with a pulock (they agree) and status 2 with puzone rows (combined in the table) are
not issues.

Warnings: new base `ZoneInputWarning(UserWarning)`; `ZoneTargetWarning` (re-parented; backward
compatible) and new `ZoneLockWarning` subclass it. One module-level dispatcher
`warn_zone_input_issues(problem, solver_name)` in `zones/model.py` (not exported) emits at most
one warning of each kind; every zone solver calls it once at entry (`solve`, and
`ZoneIISolver.improve`). `warn_zone_target_issues` stays as a wrapper that emits
`ZoneTargetWarning` only. Both use one private emitter so the reported location is the caller of
`solve()`. "One per call" is per direct solver call; `RunModePipeline` RUNMODEs 2–5 call several
zone solvers per run and emit one per call.

### 4. Solvers (every solver reads `problem.pu_zone_rules()`)

`rule_violations(rules, assignment) -> list[int]` — PU indices whose zone breaks their rule.

Removed (replaced by the rules): the per-solver status parsers in `solver.py`, `heuristic.py`
(`_parse_status`), `iterative_improvement.py` (`_parse_locked`); the MIP's locked-in/out
constraint block; status wording in the `mip_solver` and `diagnostics` module docstrings.

- **MIP**: `x[i,z].upBound = 0` for `z ∉ allowed`; `assign_{pid}`: `Σ_z x[i,z] == 1` if
  `not may_be_unassigned`, else `<= 1`. The feasibility probe checks *any* variable for a value
  (the first PU's first-zone variable may be fixed to 0).
- **SA**: per-PU option array `opts[i] = sorted(allowed)` (zone 0 never proposed, as today); for
  free PUs `opts[i]` is the shared `zone_options` array, so the seeded RNG stream is unchanged.
  Not swappable: `allowed = ∅` (assigned 0) and forced PUs (assigned their zone). One proposal
  helper used by **both** the temperature-estimation loop and the annealing loop. Initial
  assignment: `rule.start` if set, else a random option (MarZone draws randomly).
- **Heuristic**: initial assignment = `rule.start` **only for PUs that must be assigned**, else
  (for those) `min(allowed)`; PUs that may be unassigned start at 0 as today (status 1 has no
  effect, as in v0.37). Candidates = `sorted(allowed)`; forced / `∅` PUs not swappable.
- **II**: `solve()` seeds each swappable PU at `rule.start` if set, else `min(allowed)` (v0.37
  seeds every swappable PU at `zone_ids[0]`; unchanged for free PUs). Candidate sets per pass,
  in this order:
  - removal: `[0]` if `may_be_unassigned`, else the PU is skipped;
  - addition (PUs at 0): `sorted(allowed)`; **also** must-assign PUs with `len(allowed) > 1`
    try every other allowed zone (no effect on free PUs, so v0.37 output is unchanged);
  - swap: `([0] if may_be_unassigned else []) + sorted(allowed)` — v0.37's `[0, *zone_ids]` order.
  `improve()` first **repairs** the incoming assignment (forced → its zone, `∅` → 0, a zone outside
  `allowed` or 0-when-must-assign → `min(allowed)`), then runs the passes.
- **Diagnostic**: `_best(rule, values)` = max over `allowed`, combined with 0 when
  `may_be_unassigned`; forced uses its value as-is. When `pu_locks` / `pu_zones` is set,
  unreachable-target messages and the "targets compete" fallback add "(given the PU zone
  restrictions)".
- **Analysis**: `compute_replacement_cost`'s status-3 variant also drops that PU's `pu_locks` /
  `pu_zones` rows (otherwise the lock wins and the cost is 0). Other status-based analyses
  (gap, representation, irreplaceability, Ferrier) do not see pulock/puzone — documented.

### 5. Performance

Default path: seeded golden tests — a fixture with status 0/1/2/3 PUs and no
pulock/puzone gives the same assignment and objective as v0.37 for SA, heuristic and II
ITIMPTYPE 1, 2, 3 (spf 1 and 10). Bench: SA seconds per proposal on
`make_zone_problem(n_pu=1000, n_feat=50, n_zones=3)` with ~50 % of PUs puzone-restricted to two
zones vs unrestricted, median of ≥ 5 reps, ≤ 1.10×.

## Testing

1. **Rules table** — every row and precedence pair; status 2 + puzone; `pu_locks` rows
   `(1, 2), (1, 99)` → PU1 falls through; puzone rows naming only unknown zones → free; partly
   unknown `{1, 99}` → `{1}` → lock, reported; invariant holds.
2. **I/O** — MarZone's fixtures byte-for-byte (`puid, zoneid\r\n2,1\r\n5,2\r\n30,1`, CRLF, no
   trailing newline): the reader keeps every row (incl. unknown PU 30), the lookup drops PU 30;
   headerless file; header-only file → empty; 0-byte and blank-cell files → `ValueError`; writers
   emit `puid,zoneid` and round-trip; `load_zone_project` loads only when the parameter is set;
   named missing → `FileNotFoundError`.
3. **Anchors (verified by enumeration and prototype MIP):** pulock PU1 → 2 moves the MIP optimum
   from `(1, 2, 2)` at 7.0 to `(2, 1, 2)` at 8.0. **Must-assign variant:** anchor + PU4 (no
   features, zone costs (3, 2)): unrestricted `(1, 2, 2, 0)` at 7.0; puzone PU4 ∈ {1, 2} →
   `(1, 2, 2, 2)` at 9.0. MIP with PU1 locked to a non-min zone (probe).
4. **All four solvers** (+ `improve()` fed an out-of-rule assignment; II ITIMPTYPE 1, 2, 3):
   every returned solution has `rule_violations == []` on the must-assign variant and a lock case.
5. **Issues** — each §3 row appears in `validate()` / the warning as tabled; one
   `ZoneLockWarning` per direct solver call; `ZoneInputWarning` catches both kinds;
   `warn_zone_target_issues` emits only `ZoneTargetWarning`.
6. **Diagnostic** — must-assign PU {1, 2} with values (−1, −2) → `_best = −1`; a lock-induced
   unreachable target is named and the message mentions the restrictions.
7. **Replacement cost** — a pulocked PU's replacement cost is not 0.
8. **Regression** — the two `TestRules` tests in `test_zone_diagnostics.py` are rewritten for the
   new rule fields; `ZONES_EXPORTS` gains the six names; seeded golden tests (§5);
   `tests/data/simple` MIP = 35.0; MarZone anchor `(1, 2, 2)` at 7.0 without locks.

## Compatibility

Additive: two optional fields, readers/writers, `ZoneInputWarning` / `ZoneLockWarning`, `pu_*`
methods. `PUZoneRule` is internal (not exported); its fields change. Behaviour without
pulock/puzone is unchanged for `solve()` (golden tests). Deliberate change: `ZoneIISolver.improve()`
now repairs out-of-rule incoming assignments (e.g. a status-2 PU not in its zone) before
improving. Version: minor bump (0.38.0).

## Out of scope

- MarZone's `CalcPenalties` pre-counts pulocked PUs' amounts when calibrating per-species penalty
  weights (`marzone.cpp:769-782`), so MarZone's heuristic solutions under locks can differ;
  pymarxan has no penalty calibration (existing approximation). Anchor tests compare only against
  the exact MIP or enumeration.
- The `-1` / multi-file variants (`ZONETARGET2NAME`, `ZONECONTRIB2/3NAME`) — separate issue.
