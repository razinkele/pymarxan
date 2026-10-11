# Zone PU Locks and Restrictions (pulock + puzone, #6) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Rev 2** — patched after the dry-run plan review (`2026-10-11-zone-pu-locks-plan-review.md`): stronger v0.37 golden + mutation check, strict-subset solver fixture, honest bench, corrected red lists, issue-message fixes, stricter reader.

**Goal:** Let zone problems lock a planning unit into a named zone (`pulock.dat`) or restrict it to a set of zones (`puzone.dat`), honoured identically by ZoneMIPSolver, ZoneSASolver, ZoneHeuristicSolver and ZoneIISolver.

**Architecture:** Two optional `ZonalProblem` tables (`pu_locks`, `pu_zones`) are parsed once by new model methods into one `PUZoneRule(allowed, may_be_unassigned, start)` per PU (`ZonalProblem.pu_zone_rules()`), which replaces every per-solver status parser. Every solver derives its fixed PUs, candidate zones and start zones from those rules; behaviour without the tables is pinned byte-identical to v0.37 by characterization tests recorded before any solver change.

**Tech Stack:** Python 3.12, numpy, pandas, PuLP (CBC), pytest.

**Spec:** `docs/plans/2026-10-11-zone-pu-locks-design.md` (rev 3). Review: `docs/plans/2026-10-11-zone-pu-locks-review.md` (rounds 1–2).

## Global Constraints

- Every file starts with `from __future__ import annotations`; full type hints; ruff (E, F, I, UP; line length 99) and mypy clean.
- Tests run ONLY with `/opt/micromamba/envs/shiny/bin/pytest` (never `.venv`). In a git worktree run mypy as `/home/razinka/marxan/.venv/bin/mypy` and non-pytest Python with `PYTHONPATH=src`.
- Rule precedence (first match wins): pulock → status 3 → status 2 with puzone → status 2 → puzone → free (spec §2 table, verbatim).
- A puzone PU may NOT be unassigned (zone 0); status 3 PUs are always 0; free PUs may be 0.
- MarZone semantics: pulock last row per PU wins, *then* an unknown zone drops the lock (PU falls through); a PU whose puzone rows name only unknown zones is unrestricted; files load only when `PULOCKNAME` / `PUZONENAME` is in the parameters.
- Ids in the lookups are parsed with `zones.model._known_id` (non-integral / NaN / unknown → row skipped).
- Without `pu_locks` / `pu_zones`, `solve()` output of SA, heuristic and II (ITIMPTYPE 1/2/3) is identical to v0.37 (characterization tests).
- `validate()` never raises; it returns `list[str]`.
- Anchors must not move: `tests/data/simple` MIP = 35.0; MarZone anchor `(1, 2, 2)` at 7.0 without locks.
- Bench assertions are comparative, never absolute wall-clock.
- Ruff the **test** files at every task handoff.
- Commit messages reference `(#6)` and end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Review Focus

1. **A MarZone `pulock.dat` saved on Windows (CRLF, UTF-8 BOM, `puid, zoneid` header with spaces)** — loads exactly like a Unix file. Test: Task 3 `test_reads_marzone_fixture_bytes`, `test_bom_and_crlf`.
2. **RUNMODE pipelines (heuristic → SA → II, `improve()`) with restrictions** — never return a solution that breaks a rule. Test: Task 6 `test_pipeline_respects_rules`.
3. **Every PU forced or locked out** — every solver returns exactly the forced assignment. Test: Task 6 `test_all_forced_every_solver`.
4. **Non-contiguous zone ids (`{5, 9}`)** — status 2 / status 1 use the smallest id; restrictions name real ids; solvers return those ids. Tests: Task 1 `test_non_contiguous_zone_ids`, Task 6 `test_non_contiguous_zone_ids_solvers`.
5. **A lock that makes a target unreachable** — the MIP warning names the target and mentions the PU zone restrictions. Test: Task 4 `test_lock_induced_infeasibility_message`.

---

## File Structure

| File | Responsibility | Change |
|---|---|---|
| `src/pymarxan/zones/model.py` | `pu_locks` / `pu_zones` fields; `PUZoneRule`, `rule_violations`, `fixed_zones`; lookups, rules, issues; `validate()`; warning classes + dispatcher | Modify |
| `src/pymarxan/zones/diagnostics.py` | Re-export `PUZoneRule`; `_pu_zone_rules` alias; `_best` for the new rule | Modify |
| `src/pymarxan/zones/readers.py` | `read_pu_lock`, `read_pu_zone`; loader | Modify |
| `src/pymarxan/zones/writers.py` | `write_pu_lock`, `write_pu_zone` | Modify |
| `src/pymarxan/zones/mip_solver.py` | Rules → bounds + `assign_{pid}` sense; probe; restriction note | Modify |
| `src/pymarxan/zones/solver.py` | SA: rules, per-PU options, one proposal helper | Modify |
| `src/pymarxan/zones/heuristic.py` | Rules; seeding; candidates | Modify |
| `src/pymarxan/zones/iterative_improvement.py` | Rules; seeding; per-pass candidates; `improve()` repair | Modify |
| `src/pymarxan/analysis/replacement_cost.py` | Variant drops the PU's lock/zone rows | Modify |
| `src/pymarxan/zones/__init__.py` | Exports | Modify |
| `tests/pymarxan/zones/test_zone_golden_v037.py` | Characterization of SA / heuristic / II | Create (Task 1) |
| `tests/pymarxan/zones/test_pu_zone_rules.py` | Rules, issues, warnings | Create (Tasks 1–2) |
| `tests/pymarxan/zones/test_pu_zone_io.py` | Readers / writers / loader | Create (Task 3) |
| `tests/pymarxan/zones/test_pu_zone_solvers.py` | MIP / SA / heuristic / II with rules | Create (Tasks 4–6) |
| `tests/pymarxan/zones/test_zone_diagnostics.py` | Rewrite the two `TestRules` tests | Modify (Task 1) |
| `tests/pymarxan/zones/test_package_exports.py` | `ZONES_EXPORTS` | Modify (Tasks 2–3) |
| `tests/pymarxan/analysis/test_replacement_cost_zone_locks.py` | Replacement cost with a lock | Create (Task 7) |
| `tests/benchmarks/bench_zone_pu_restrictions.py` | SA per-proposal cost | Create (Task 7) |
| `CHANGELOG.md`, `docs/VALIDATION.md` | Docs | Modify (Task 7) |

Shared test fixtures (each test file defines the helpers it uses; they are repeated verbatim where needed):

```python
import pandas as pd
from tests.pymarxan.zones.marzone_anchor import make_anchor_problem


def _with_pu4(**kw):
    """MarZone anchor + PU4 (no features, zone costs 3 / 2). Unrestricted MIP optimum
    (1, 2, 2, 0) at 7.0; with PU4 restricted to {1, 2} it is (1, 2, 2, 2) at 9.0."""
    p = make_anchor_problem(**kw)
    pu = pd.concat(
        [p.planning_units, pd.DataFrame({"id": [4], "cost": [0.0], "status": [0]})],
        ignore_index=True,
    )
    zc = pd.concat(
        [p.zone_costs, pd.DataFrame({"pu": [4, 4], "zone": [1, 2], "cost": [3.0, 2.0]})],
        ignore_index=True,
    )
    return p.copy_with(planning_units=pu, zone_costs=zc)


def _table(rows):
    return pd.DataFrame(rows, columns=["pu", "zone"])
```

---

### Task 1: Characterize v0.37 solver output; rules on the model

**Files:**
- Create: `tests/pymarxan/zones/test_zone_golden_v037.py` (Step 1 — BEFORE any source change)
- Modify: `src/pymarxan/zones/model.py`, `src/pymarxan/zones/diagnostics.py`
- Modify: `tests/pymarxan/zones/test_zone_diagnostics.py` (`class TestRules`)
- Test: `tests/pymarxan/zones/test_pu_zone_rules.py`

**Interfaces:**
- Produces (all in `pymarxan.zones.model`):
  - fields `ZonalProblem.pu_locks: pd.DataFrame | None` and `pu_zones: pd.DataFrame | None` (kw-only, default `None`; columns `pu, zone`);
  - `class PUZoneRule(NamedTuple)`: `allowed: frozenset[int]`, `may_be_unassigned: bool`, `start: int | None`; property `forced -> bool`;
  - `ZonalProblem.pu_lock_lookup() -> dict[int, int]`, `pu_zone_lookup() -> dict[int, frozenset[int]]`, `pu_zone_rules() -> list[PUZoneRule]`;
  - `rule_violations(rules: list[PUZoneRule], assignment: np.ndarray) -> list[int]`;
  - `fixed_zones(rules: list[PUZoneRule]) -> dict[int, int]` (PU index → its forced zone, or 0 when `allowed` is empty).
- `pymarxan.zones.diagnostics` keeps importing names `PUZoneRule` and `_pu_zone_rules` (alias of `problem.pu_zone_rules()`).

- [ ] **Step 1: Record v0.37 behaviour (characterization, before any change)**

Create `tests/pymarxan/zones/test_zone_golden_v037.py`:

```python
"""v0.37 output of the zone SA / heuristic / II on a status-0/1/2/3 fixture (#6 spec §5).

The GOLDEN values were recorded on the unmodified v0.37 code; #6 must not change them
(no pu_locks / pu_zones here). Regenerate only with a written reason:
    PYTHONPATH=src /opt/micromamba/envs/shiny/bin/python \
        tests/pymarxan/zones/test_zone_golden_v037.py   (PYTHONPATH=src:. for tests.benchmarks)
"""
from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd
import pytest

from pymarxan.solvers.base import SolverConfig
from pymarxan.zones import ZoneHeuristicSolver, ZoneIISolver, ZoneSASolver, load_zone_project

ZONES_DIR = Path(__file__).parent.parent.parent / "data" / "zones"
SOLVER_CASES = [
    ("sa", None, None), ("sa", 10.0, None),
    ("heuristic", None, None), ("heuristic", 10.0, None),
    ("ii", None, 1), ("ii", None, 2), ("ii", None, 3),
    ("ii", 10.0, 1), ("ii", 10.0, 2), ("ii", 10.0, 3),
]
CASES = [(fx, *case) for fx in ("small", "large") for case in SOLVER_CASES]
Golden = list[tuple[tuple[int, ...], float]]


def _small():
    """tests/data/zones (4 PUs, 2 zones), statuses 1/2/3/0 — converged SA."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p = load_zone_project(ZONES_DIR)
    return p.copy_with(
        planning_units=p.planning_units.assign(status=[1, 2, 3, 0]),
        parameters={**p.parameters, "NUMITNS": 3000, "NUMTEMP": 30},
    )


def _large():
    """30 PUs + one tie PU, 3 zones, statuses cycling 0/1/2/3/0/0, UNconverged SA
    (NUMITNS 400) so the output depends on the RNG stream; PU 31 (no features, zone costs
    5 / 0 / 7) gives II's swap pass a tie between zone 0 and zone 2 from its zone-1 seed,
    so candidate order matters."""
    from tests.benchmarks.conftest import make_zone_problem

    p = make_zone_problem(n_pu=30, n_feat=5, n_zones=3, density=0.4, seed=3)
    statuses = [[0, 1, 2, 3, 0, 0][i % 6] for i in range(30)]
    pu = p.planning_units.assign(status=statuses)
    tie = {c: [0] for c in pu.columns}
    tie.update({"id": [31], "status": [0]})
    pu = pd.concat([pu, pd.DataFrame(tie)], ignore_index=True)
    zc = pd.concat(
        [p.zone_costs, pd.DataFrame({"pu": [31, 31, 31], "zone": [1, 2, 3],
                                     "cost": [5.0, 0.0, 7.0]})],
        ignore_index=True,
    )
    return p.copy_with(
        planning_units=pu, zone_costs=zc,
        parameters={**p.parameters, "NUMITNS": 400, "NUMTEMP": 10, "BLM": 0.5},
    )


def _fixture(fx: str, spf: float | None, itimptype: int | None):
    p = _small() if fx == "small" else _large()
    if itimptype is not None:
        p = p.copy_with(parameters={**p.parameters, "ITIMPTYPE": itimptype})
    if spf is not None:
        p = p.copy_with(features=p.features.assign(spf=spf))
    return p


def _run(fx: str, kind: str, spf: float | None, itimptype: int | None) -> Golden:
    p = _fixture(fx, spf, itimptype)
    solver = {"sa": ZoneSASolver, "heuristic": ZoneHeuristicSolver, "ii": ZoneIISolver}[kind]()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sols = solver.solve(p, SolverConfig(num_solutions=2, seed=7))
    return [
        (tuple(int(z) for z in s.zone_assignment), round(float(s.objective), 6)) for s in sols
    ]


GOLDEN: dict[str, Golden] = {
    # PASTE the dict printed by running this file as a script on the unmodified code.
}


@pytest.mark.parametrize("fx,kind,spf,itimptype", CASES)
def test_matches_v037(fx, kind, spf, itimptype):
    assert _run(fx, kind, spf, itimptype) == GOLDEN[f"{fx}-{kind}-{spf}-{itimptype}"]


if __name__ == "__main__":
    print("GOLDEN = {")
    for fx, kind, spf, itimptype in CASES:
        print(f"    {f'{fx}-{kind}-{spf}-{itimptype}'!r}: {_run(fx, kind, spf, itimptype)!r},")
    print("}")
```

Run on the **unmodified** code: `PYTHONPATH=src:. /opt/micromamba/envs/shiny/bin/python tests/pymarxan/zones/test_zone_golden_v037.py`, paste the printed dict over the `GOLDEN` placeholder, then run `/opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/zones/test_zone_golden_v037.py -q` → 20 passed.

**Mutation check (do it, then revert; record the outcome in the commit message):** (a) in `src/pymarxan/zones/solver.py`'s annealing loop add `rng.integers(2)` immediately before the `new_zone` draw → at least one `large-sa-*` case must FAIL; (b) in `src/pymarxan/zones/iterative_improvement.py` change `all_options = [0, *zone_ids]` to `[*zone_ids, 0]` → at least one `large-ii-*` case must FAIL. Revert both (`git diff src` empty). If either mutation passes, the fixture does not discriminate — stop and report (BLOCKED) rather than continuing. Commit the test file alone:

```bash
git add tests/pymarxan/zones/test_zone_golden_v037.py
git commit -m "test(zones): characterize v0.37 SA/heuristic/II output on two status-0/1/2/3 fixtures (#6)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

- [ ] **Step 2: Write the failing rule tests**

Create `tests/pymarxan/zones/test_pu_zone_rules.py`:

```python
"""PU zone rules: pulock / puzone / status precedence (#6 spec §2)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from pymarxan.zones.model import PUZoneRule, fixed_zones, rule_violations
from tests.pymarxan.zones.marzone_anchor import make_anchor_problem

ALL3 = frozenset({1, 2, 3})


def _table(rows):
    return pd.DataFrame(rows, columns=["pu", "zone"])


def _three_zone(statuses=(0, 0, 0), **tables):
    p = make_anchor_problem()
    zones = pd.DataFrame({"id": [1, 2, 3], "name": ["protect", "restore", "use"]})
    zc = pd.concat(
        [p.zone_costs, pd.DataFrame({"pu": [1, 2, 3], "zone": [3, 3, 3], "cost": [2.0] * 3})],
        ignore_index=True,
    )
    p = p.copy_with(
        zones=zones, zone_costs=zc, planning_units=p.planning_units.assign(status=list(statuses)),
    )
    return p.copy_with(**tables)


def _rule(allowed, may, start):
    return PUZoneRule(allowed=frozenset(allowed), may_be_unassigned=may, start=start)


def test_free_and_status_rows():
    rules = _three_zone((1, 2, 3)).pu_zone_rules()
    assert rules == [_rule(ALL3, True, 1), _rule({1}, False, 1), _rule(set(), True, 0)]


def test_lock_and_last_row_wins():
    p = _three_zone(pu_locks=_table([(1, 3), (1, 2)]))
    assert p.pu_lock_lookup() == {1: 2}
    assert p.pu_zone_rules()[0] == _rule({2}, False, 2)


def test_last_row_wins_before_unknown_zone_filter():
    p = _three_zone(pu_locks=_table([(1, 2), (1, 99)]))
    assert p.pu_lock_lookup() == {}
    assert p.pu_zone_rules()[0] == _rule(ALL3, True, None)


def test_unknown_pu_and_odd_ids_skipped():
    p = _three_zone(pu_locks=_table([(9, 2), (np.nan, 2), (2.0, 3.0), (3, 2.5)]))
    assert p.pu_lock_lookup() == {2: 3}


def test_lock_beats_status_3_and_status_3_beats_puzone():
    p = _three_zone((3, 3, 0), pu_locks=_table([(1, 2)]), pu_zones=_table([(2, 1), (2, 2)]))
    rules = p.pu_zone_rules()
    assert rules[0] == _rule({2}, False, 2)
    assert rules[1] == _rule(set(), True, 0)


def test_status_2_within_puzone():
    p = _three_zone((2, 2, 2), pu_zones=_table([(1, 2), (1, 3), (2, 3)]))
    rules = p.pu_zone_rules()
    assert rules[0] == _rule({2, 3}, False, None)
    assert rules[1] == _rule({3}, False, 3)
    assert rules[2] == _rule({1}, False, 1)


def test_puzone_rows():
    p = _three_zone(pu_zones=_table([(1, 2), (1, 3), (1, 3), (2, 99), (2, 98), (3, 1), (3, 99)]))
    assert p.pu_zone_lookup() == {1: frozenset({2, 3}), 3: frozenset({1})}
    rules = p.pu_zone_rules()
    assert rules[0] == _rule({2, 3}, False, None)
    assert rules[1] == _rule(ALL3, True, None)     # all-unknown puzone -> free
    assert rules[2] == _rule({1}, False, 1)        # partly unknown -> single zone -> lock


def test_missing_columns_mean_no_table():
    p = _three_zone(pu_locks=pd.DataFrame({"pu": [1]}), pu_zones=pd.DataFrame({"zone": [1]}))
    assert p.pu_lock_lookup() == {} and p.pu_zone_lookup() == {}


def test_non_contiguous_zone_ids():
    p = make_anchor_problem()
    relabel = {1: 5, 2: 9}
    p = p.copy_with(
        zones=p.zones.assign(id=p.zones["id"].map(relabel)),
        zone_costs=p.zone_costs.assign(zone=p.zone_costs["zone"].map(relabel)),
        zone_contributions=p.zone_contributions.assign(
            zone=p.zone_contributions["zone"].map(relabel),
        ),
        zone_targets=p.zone_targets.assign(zone=p.zone_targets["zone"].map(relabel)),
        planning_units=p.planning_units.assign(status=[2, 1, 0]),
        pu_zones=_table([(3, 9)]),
    )
    assert p.pu_zone_rules() == [
        _rule({5}, False, 5), _rule({5, 9}, True, 5), _rule({9}, False, 9),
    ]


def test_precedence_pairs():
    p = _three_zone(
        (1, 2, 3),
        pu_locks=_table([(2, 3)]),
        pu_zones=_table([(1, 2), (1, 3), (2, 1), (2, 2), (3, 2)]),
    )
    assert p.pu_zone_rules() == [
        _rule({2, 3}, False, None),   # status 1 + puzone -> puzone (start dropped)
        _rule({3}, False, 3),         # status 2 + puzone + lock -> lock
        _rule(set(), True, 0),        # status 3 + puzone -> status 3
    ]


def test_invariant_no_unsatisfiable_rule():
    p = _three_zone((3, 2, 1), pu_locks=_table([(1, 99)]), pu_zones=_table([(2, 99), (3, 98)]))
    assert all(r.allowed or r.may_be_unassigned for r in p.pu_zone_rules())


def test_forced_property():
    assert _rule({2}, False, 2).forced
    assert not _rule({2}, True, None).forced
    assert not _rule({2, 3}, False, None).forced
    assert not _rule(set(), True, 0).forced


def test_rule_violations_and_fixed_zones():
    rules = [_rule({2}, False, 2), _rule(ALL3, True, None), _rule({2, 3}, False, None),
             _rule(set(), True, 0)]
    assert rule_violations(rules, np.array([2, 0, 3, 0])) == []
    assert rule_violations(rules, np.array([1, 0, 0, 2])) == [0, 2, 3]
    assert fixed_zones(rules) == {0: 2, 3: 0}


def test_copy_with_carries_tables():
    p = _three_zone(pu_locks=_table([(1, 2)]))
    q = p.copy_with(planning_units=p.planning_units.copy())
    assert q.pu_locks is not None and q.pu_lock_lookup() == {1: 2}
```

Rewrite `TestRules` in `tests/pymarxan/zones/test_zone_diagnostics.py`:

```python
class TestRules:
    def test_status_convention(self):
        p = _status(make_anchor_problem(), [0, 2, 3])
        assert _pu_zone_rules(p) == [
            PUZoneRule(allowed=frozenset({1, 2}), may_be_unassigned=True, start=None),
            PUZoneRule(allowed=frozenset({1}), may_be_unassigned=False, start=1),
            PUZoneRule(allowed=frozenset(), may_be_unassigned=True, start=0),
        ]

    def test_missing_status_column_means_free(self):
        p = make_anchor_problem()
        p = p.copy_with(planning_units=p.planning_units.drop(columns="status"))
        free = PUZoneRule(allowed=frozenset({1, 2}), may_be_unassigned=True, start=None)
        assert all(r == free for r in _pu_zone_rules(p))

    def test_best_for_must_assign_negative_values(self):
        from pymarxan.zones.diagnostics import _best
        rule = PUZoneRule(allowed=frozenset({1, 2}), may_be_unassigned=False, start=None)
        assert _best(rule, {1: -1.0, 2: -2.0}) == -1.0
```

- [ ] **Step 2b: Run the new tests to verify they fail**

Run: `/opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/zones/test_pu_zone_rules.py tests/pymarxan/zones/test_zone_diagnostics.py -q`
Expected: collection ERROR — `ImportError: cannot import name 'PUZoneRule' from 'pymarxan.zones.model'`.

- [ ] **Step 3: Implement in `src/pymarxan/zones/model.py`**

Imports: add `from typing import NamedTuple` and replace the `ConservationProblem` import with

```python
from pymarxan.models.problem import (
    STATUS_INITIAL_INCLUDE,
    STATUS_LOCKED_IN,
    STATUS_LOCKED_OUT,
    ConservationProblem,
)
```

Add above `@dataclass class ZonalProblem`:

```python
class PUZoneRule(NamedTuple):
    """Zones a planning unit may take (``ZonalProblem.pu_zone_rules``)."""

    allowed: frozenset[int]
    may_be_unassigned: bool
    start: int | None

    @property
    def forced(self) -> bool:
        """Exactly one allowed zone and the PU must be assigned."""
        return len(self.allowed) == 1 and not self.may_be_unassigned


def rule_violations(rules: list[PUZoneRule], assignment: np.ndarray) -> list[int]:
    """Indices of PUs whose zone in ``assignment`` breaks their rule."""
    bad: list[int] = []
    for i, (rule, zone) in enumerate(zip(rules, assignment, strict=True)):
        z = int(zone)
        if (z == 0 and not rule.may_be_unassigned) or (z != 0 and z not in rule.allowed):
            bad.append(i)
    return bad


def fixed_zones(rules: list[PUZoneRule]) -> dict[int, int]:
    """PU index -> the zone it must take (forced) or 0 (no allowed zone); the rest move."""
    fixed: dict[int, int] = {}
    for i, rule in enumerate(rules):
        if rule.forced:
            (fixed[i],) = rule.allowed
        elif not rule.allowed:
            fixed[i] = 0
    return fixed


def _only(zones: frozenset[int]) -> int | None:
    return next(iter(zones)) if len(zones) == 1 else None
```

Add the fields to `ZonalProblem` after `zone_boundary_costs`:

```python
    pu_locks: pd.DataFrame | None = field(default=None, kw_only=True)
    pu_zones: pd.DataFrame | None = field(default=None, kw_only=True)
```

Add the methods after `zone_target_issues`:

```python
    # ------------------------------------------------------------------
    # PU locks / zone restrictions (MarZone pu.hpp LoadPuLock / PersistPuLock / PersistPuZone)
    # ------------------------------------------------------------------

    def _pu_table_rows(self, table: pd.DataFrame | None) -> list[tuple[object, object]]:
        if table is None or not {"pu", "zone"}.issubset(table.columns):
            return []
        return list(zip(table["pu"].tolist(), table["zone"].tolist(), strict=True))

    def pu_lock_lookup(self) -> dict[int, int]:
        """PU id -> locked zone. Last row per known PU wins, THEN a lock naming an unknown
        zone is dropped and the PU falls through to lower rules (MarZone pu.hpp:483-496)."""
        pus = {int(p) for p in self.planning_units["id"].values}
        zones = {int(z) for z in self.zone_ids}
        last: dict[int, object] = {}
        for p, z in self._pu_table_rows(self.pu_locks):
            pid = _known_id(p, pus)
            if pid is not None:
                last[pid] = z
        out: dict[int, int] = {}
        for pid, z in last.items():
            zid = _known_id(z, zones)
            if zid is not None:
                out[pid] = zid
        return out

    def pu_zone_lookup(self) -> dict[int, frozenset[int]]:
        """PU id -> distinct known zones it is restricted to. A PU whose rows name only
        unknown zones is omitted (unrestricted, MarZone numZones == 0)."""
        pus = {int(p) for p in self.planning_units["id"].values}
        zones = {int(z) for z in self.zone_ids}
        acc: dict[int, set[int]] = {}
        for p, z in self._pu_table_rows(self.pu_zones):
            pid, zid = _known_id(p, pus), _known_id(z, zones)
            if pid is not None and zid is not None:
                acc.setdefault(pid, set()).add(zid)
        return {pid: frozenset(zs) for pid, zs in acc.items()}

    def pu_zone_rules(self) -> list[PUZoneRule]:
        """One rule per PU in ``planning_units`` order; within pymarxan.zones the only place
        that interprets status, pu_locks and pu_zones (spec §2 table, first match wins)."""
        every = frozenset(int(z) for z in self.zone_ids)
        first = min(every) if every else None
        locks = self.pu_lock_lookup()
        restrict = self.pu_zone_lookup()
        pu = self.planning_units
        statuses = (
            pu["status"].values.astype(int) if "status" in pu.columns
            else np.zeros(len(pu), dtype=int)
        )
        rules: list[PUZoneRule] = []
        for pid_raw, status in zip(pu["id"].values, statuses, strict=True):
            pid, s = int(pid_raw), int(status)
            if pid in locks:
                z = locks[pid]
                rule = PUZoneRule(allowed=frozenset({z}), may_be_unassigned=False, start=z)
            elif s == STATUS_LOCKED_OUT:
                rule = PUZoneRule(allowed=frozenset(), may_be_unassigned=True, start=0)
            elif pid in restrict:  # status 2 within puzone, or plain puzone
                zs = restrict[pid]
                rule = PUZoneRule(allowed=zs, may_be_unassigned=False, start=_only(zs))
            elif s == STATUS_LOCKED_IN and first is not None:
                rule = PUZoneRule(allowed=frozenset({first}), may_be_unassigned=False, start=first)
            else:
                start = first if s == STATUS_INITIAL_INCLUDE else None
                rule = PUZoneRule(allowed=every, may_be_unassigned=True, start=start)
            rules.append(rule)
        if any(not r.allowed and not r.may_be_unassigned for r in rules):
            raise AssertionError("pu_zone_rules: a PU with no zone that must be assigned")
        return rules
```

(The single `pid in restrict` branch covers both "status 2 and puzone" and "puzone" rows of the table: both give `(puzone set, False, only-zone-or-None)`.)

`src/pymarxan/zones/diagnostics.py`: delete the local `PUZoneRule` class and `_pu_zone_rules` body and the `STATUS_*` import; add `from pymarxan.zones.model import PUZoneRule, ZonalProblem` and

```python
def _pu_zone_rules(problem: ZonalProblem) -> list[PUZoneRule]:
    """Alias of ``ZonalProblem.pu_zone_rules`` (kept for the MIP and older imports)."""
    return problem.pu_zone_rules()


def _best(rule: PUZoneRule, value_in: dict[int, float]) -> float:
    """A PU's largest possible contribution: the best allowed zone's value, or 0 when the
    PU may stay unassigned and that is better; a forced PU's value as-is."""
    values = [value_in.get(z, 0.0) for z in rule.allowed]
    if rule.may_be_unassigned:
        values.append(0.0)
    return max(values) if values else 0.0
```

Replace the module docstring with:

```python
"""Single-target infeasibility diagnostics for ZoneMIPSolver.

``_feature_groups`` (the MIP constraints' amount source) lives here; which zones a PU may
take is ``ZonalProblem.pu_zone_rules`` (``_pu_zone_rules`` is a thin alias kept for the MIP
and older imports).
"""
```

Note: from this task on the MIP already honours pulocks and puzone allowed sets (it reads `rule.forced` / `rule.allowed` from the new rules); Task 4 adds only the must-assign equality and the message.

- [ ] **Step 4: Run tests to verify they pass**

Run: `/opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/zones -q`
Expected: all PASS (incl. the 10 golden tests).

- [ ] **Step 5: Lint, type-check, commit**

```bash
~/.local/bin/ruff check src/pymarxan/zones tests/pymarxan/zones
/home/razinka/marxan/.venv/bin/mypy src/pymarxan/zones --ignore-missing-imports
git add src/pymarxan/zones/model.py src/pymarxan/zones/diagnostics.py tests/pymarxan/zones/test_pu_zone_rules.py tests/pymarxan/zones/test_zone_diagnostics.py
git commit -m "feat(zones): pu_locks/pu_zones fields and one PUZoneRule per PU on the model (#6)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: Issues, `validate()`, warning classes and the solve-entry dispatcher

**Files:**
- Modify: `src/pymarxan/zones/model.py` (issues, `validate()`, warnings), `src/pymarxan/zones/{mip_solver,solver,heuristic,iterative_improvement}.py` (entry call), `src/pymarxan/zones/__init__.py`
- Modify: `tests/pymarxan/zones/test_package_exports.py`
- Test: `tests/pymarxan/zones/test_pu_zone_rules.py` (append)

**Interfaces:**
- Consumes: Task 1 lookups / rules.
- Produces (`pymarxan.zones.model`): `ZonalProblem.pu_lock_issues() -> list[str]`; `class ZoneInputWarning(UserWarning)`; `ZoneTargetWarning(ZoneInputWarning)` (re-parented); `class ZoneLockWarning(ZoneInputWarning)`; `warn_zone_input_issues(problem, solver_name) -> None` (module-level, not exported); `warn_zone_target_issues` kept, emits `ZoneTargetWarning` only. `ZoneInputWarning`, `ZoneLockWarning` exported from `pymarxan.zones`.

- [ ] **Step 1: Append the failing tests** to `tests/pymarxan/zones/test_pu_zone_rules.py` and merge the imports into its import block:

```python
import warnings

import pytest

from pymarxan.solvers.base import SolverConfig
from pymarxan.zones import (
    ZoneHeuristicSolver,
    ZoneIISolver,
    ZoneInputWarning,
    ZoneLockWarning,
    ZoneMIPSolver,
    ZoneSASolver,
    ZoneTargetWarning,
)
from pymarxan.zones.model import warn_zone_target_issues
```

```python
def _issues(**kw):
    return _three_zone(**kw).pu_lock_issues()


def test_issue_rows_and_validate():
    p = _three_zone(
        (3, 3, 0),
        pu_locks=_table([(1, 2), (9, 1), (2, 1), (2, 3)]),
        pu_zones=_table([(1, 3), (3, 2), (3, 2), (2, 99)]),
    )
    issues = p.pu_lock_issues()
    text = " | ".join(issues)
    assert "unknown PU or zone" in text and "PU 9, zone 1" in text
    assert "repeats 1 PU(s); the last row wins" in text and "PU 2 → zone 3" in text
    assert "duplicate (pu, zone)" in text and "PU 3, zone 2" in text
    assert "single zone (treated as a lock" in text and "PU 3 → zone 2" in text
    assert "status 3 and a pu_locks row (the lock wins)" in text
    assert "outside the PU's pu_zones list (the lock wins)" in text and "PU 1 → zone 2" in text
    errors = p.validate()
    assert any("repeats 1 PU(s)" in e for e in errors)
    assert not any("outside the PU's pu_zones list" in e for e in errors)  # warning only


def test_all_unknown_puzone_reported():
    assert any("name only unknown zones (PU unrestricted)" in m
               for m in _issues(pu_zones=_table([(1, 98), (1, 99)])))


def test_status_3_with_puzone_reported():
    assert any("status 3 and pu_zones rows (status 3 wins)" in m
               for m in _issues(statuses=(3, 0, 0), pu_zones=_table([(1, 2), (1, 3)])))


def test_missing_columns_reported():
    assert any("pu_locks missing columns ['zone'] (table ignored)" in m
               for m in _issues(pu_locks=pd.DataFrame({"pu": [1]})))


ERROR_ROWS = [
    ("unknown PU or zone", dict(pu_locks=_table([(9, 1)]))),
    ("repeats 1 PU(s)", dict(pu_locks=_table([(1, 1), (1, 2)]))),
    ("duplicate (pu, zone)", dict(pu_zones=_table([(1, 1), (1, 1), (1, 2)]))),
    ("name only unknown zones", dict(pu_zones=_table([(1, 99)]))),
    ("single zone (treated as a lock", dict(pu_zones=_table([(1, 2)]))),
    ("status 3 and a pu_locks row", dict(statuses=(3, 0, 0), pu_locks=_table([(1, 2)]))),
    ("status 3 and pu_zones rows",
     dict(statuses=(3, 0, 0), pu_zones=_table([(1, 1), (1, 2)]))),
    ("missing columns", dict(pu_locks=pd.DataFrame({"pu": [1]}))),
]


@pytest.mark.parametrize("needle,kw", ERROR_ROWS, ids=[r[0] for r in ERROR_ROWS])
def test_each_error_row_reaches_validate(needle, kw):
    kw = dict(kw)
    statuses = kw.pop("statuses", (0, 0, 0))
    assert any(needle in e for e in _three_zone(statuses, **kw).validate())


def test_repeated_lock_with_unknown_last_zone_says_not_locked():
    assert any("PU 1 → zone 99 (unknown zone, PU not locked)" in m
               for m in _issues(pu_locks=_table([(1, 2), (1, 99)])))


def test_status_3_single_zone_puzone_reported_only_as_status_3():
    issues = _issues(statuses=(3, 0, 0), pu_zones=_table([(1, 2)]))
    assert any("status 3 and pu_zones rows" in m for m in issues)
    assert not any("single zone" in m for m in issues)


def test_no_tables_no_scan():
    assert _three_zone((1, 2, 3)).pu_lock_issues() == []


def test_clean_problem_has_no_issues():
    p = _three_zone((2, 1, 0), pu_locks=_table([(1, 2)]), pu_zones=_table([(2, 1), (2, 2)]))
    assert p.pu_lock_issues() == []
    assert p.validate() == []


def test_warning_hierarchy():
    assert issubclass(ZoneTargetWarning, ZoneInputWarning)
    assert issubclass(ZoneLockWarning, ZoneInputWarning)
    assert not issubclass(ZoneLockWarning, ZoneTargetWarning)


SOLVERS = [ZoneMIPSolver, ZoneSASolver, ZoneHeuristicSolver, ZoneIISolver]


@pytest.mark.parametrize("cls", SOLVERS, ids=lambda c: c.__name__)
def test_each_solver_warns_once_about_locks(cls):
    p = _three_zone(pu_locks=_table([(1, 2), (9, 1)]))
    p = p.copy_with(parameters={**p.parameters, "NUMITNS": 500})
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        cls().solve(p, SolverConfig(num_solutions=2, seed=1))
    hits = [w for w in rec if issubclass(w.category, ZoneLockWarning)]
    assert len(hits) == 1
    assert str(hits[0].message).startswith(f"{cls.__name__}: PU zone restrictions:")


def test_ii_improve_warns_once_about_locks():
    p = _three_zone(pu_locks=_table([(1, 2), (9, 1)]))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        (sol,) = ZoneHeuristicSolver().solve(p, SolverConfig(num_solutions=1, seed=1))
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        ZoneIISolver().improve(p, sol)
    assert len([w for w in rec if issubclass(w.category, ZoneLockWarning)]) == 1


def test_target_wrapper_emits_only_target_warning():
    p = _three_zone(pu_locks=_table([(9, 1)])).copy_with(
        zone_targets=pd.DataFrame({"zone": [9], "feature": [1], "target": [1.0]}),
    )
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        warn_zone_target_issues(p, "ZoneSASolver")
    assert [w.category for w in rec] == [ZoneTargetWarning]
```

Add to `ZONES_EXPORTS` in `tests/pymarxan/zones/test_package_exports.py` (keep the dict's ordering style):

```python
    "ZoneInputWarning": "pymarxan.zones.model",
    "ZoneLockWarning": "pymarxan.zones.model",
```

- [ ] **Step 2: Run to verify failure**

Run: `/opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/zones/test_pu_zone_rules.py tests/pymarxan/zones/test_package_exports.py -q`
Expected: collection ERROR — `ImportError: cannot import name 'ZoneInputWarning'`.

- [ ] **Step 3: Implement**

In `ZonalProblem` (after `pu_zone_rules`):

```python
    def _pu_lock_issue_list(self) -> list[tuple[str, bool]]:
        """(message, is_validate_error) per issue class — spec §3 table."""
        out: list[tuple[str, bool]] = []
        if self.pu_locks is None and self.pu_zones is None:
            return out  # v0.37 problems: no scan, no status cast
        pu = self.planning_units
        status_of = (
            dict(zip((int(p) for p in pu["id"].values), pu["status"].values.astype(int),
                     strict=True))
            if "status" in pu.columns else {}
        )
        pus = {int(p) for p in self.planning_units["id"].values}
        zones = {int(z) for z in self.zone_ids}
        usable: dict[str, list[tuple[object, object]]] = {}
        for name, table in (("pu_locks", self.pu_locks), ("pu_zones", self.pu_zones)):
            if table is None:
                continue
            missing = sorted({"pu", "zone"} - set(table.columns))
            if missing:
                out.append((f"{name} missing columns {missing} (table ignored)", True))
                continue
            rows = self._pu_table_rows(table)
            usable[name] = rows
            bad = [(p, z) for p, z in rows
                   if _known_id(p, pus) is None or _known_id(z, zones) is None]
            if bad:
                out.append((
                    f"{name} has {len(bad)} row(s) naming an unknown PU or zone, or a "
                    "non-integral id (skipped): "
                    + _first_five([f"PU {p}, zone {z}" for p, z in bad]),
                    True,
                ))
        locks = self.pu_lock_lookup()
        restrict = self.pu_zone_lookup()
        if "pu_locks" in usable:
            per_pu: dict[int, list[object]] = {}
            for p, z in usable["pu_locks"]:
                pid = _known_id(p, pus)
                if pid is not None:
                    per_pu.setdefault(pid, []).append(z)
            repeated = [(pid, zs[-1]) for pid, zs in per_pu.items() if len(zs) > 1]
            if repeated:
                items = [
                    f"PU {pid} → zone {z}" + ("" if pid in locks
                                               else " (unknown zone, PU not locked)")
                    for pid, z in repeated
                ]
                out.append((
                    f"pu_locks repeats {len(repeated)} PU(s); the last row wins: "
                    + _first_five(items),
                    True,
                ))
        if "pu_zones" in usable:
            seen: set[tuple[int, int]] = set()
            dup_set: set[tuple[int, int]] = set()
            dups: list[tuple[int, int]] = []
            listed: set[int] = set()
            for p, z in usable["pu_zones"]:
                pid, zid = _known_id(p, pus), _known_id(z, zones)
                if pid is None:
                    continue
                listed.add(pid)
                if zid is None:
                    continue
                key = (pid, zid)
                if key in seen and key not in dup_set:
                    dup_set.add(key)
                    dups.append(key)
                seen.add(key)
            if dups:
                out.append((
                    f"pu_zones has {len(dups)} duplicate (pu, zone) row(s) (collapsed): "
                    + _first_five([f"PU {p}, zone {z}" for p, z in dups]),
                    True,
                ))
            all_unknown = sorted(pid for pid in listed if pid not in restrict
                                 and pid not in locks
                                 and status_of.get(pid) != STATUS_LOCKED_OUT)
            if all_unknown:
                out.append((
                    f"pu_zones rows for {len(all_unknown)} PU(s) name only unknown zones "
                    "(PU unrestricted): " + _first_five([f"PU {p}" for p in all_unknown]),
                    True,
                ))
            single = [(pid, next(iter(zs))) for pid, zs in restrict.items()
                      if len(zs) == 1 and pid not in locks
                      and status_of.get(pid) != STATUS_LOCKED_OUT]
            if single:
                out.append((
                    f"pu_zones restricts {len(single)} PU(s) to a single zone (treated as a "
                    "lock; MarZone rejects single-zone puzone — use pu_locks): "
                    + _first_five([f"PU {p} → zone {z}" for p, z in single]),
                    True,
                ))
        st3_lock = [p for p in locks if status_of.get(p) == STATUS_LOCKED_OUT]
        if st3_lock:
            out.append((
                f"{len(st3_lock)} PU(s) have status 3 and a pu_locks row (the lock wins): "
                + _first_five([f"PU {p}" for p in st3_lock]),
                True,
            ))
        st3_zone = [p for p in restrict
                    if status_of.get(p) == STATUS_LOCKED_OUT and p not in locks]
        if st3_zone:
            out.append((
                f"{len(st3_zone)} PU(s) have status 3 and pu_zones rows (status 3 wins): "
                + _first_five([f"PU {p}" for p in st3_zone]),
                True,
            ))
        outside = [(p, z) for p, z in locks.items() if p in restrict and z not in restrict[p]]
        if outside:
            out.append((
                f"{len(outside)} pu_locks zone(s) are outside the PU's pu_zones list "
                "(the lock wins): " + _first_five([f"PU {p} → zone {z}" for p, z in outside]),
                False,
            ))
        return out

    def pu_lock_issues(self) -> list[str]:
        """Messages for every pu_locks / pu_zones row the solvers skip, collapse or override."""
        return [msg for msg, _ in self._pu_lock_issue_list()]
```

In `validate()`, directly before `return errors`:

```python
        errors.extend(msg for msg, is_error in self._pu_lock_issue_list() if is_error)
```

Replace the tail of the module (from `class ZoneTargetWarning` to the end) with:

```python
class ZoneInputWarning(UserWarning):
    """Zone input rows the zone solvers skip, collapse or override (base class)."""


class ZoneTargetWarning(ZoneInputWarning):
    """``zone_targets`` rows the zone solvers skip (unknown ids) or collapse (duplicates)."""


class ZoneLockWarning(ZoneInputWarning):
    """``pu_locks`` / ``pu_zones`` rows the zone solvers skip, collapse or override."""


def _emit(message: str, category: type[UserWarning], stacklevel: int = 4) -> None:
    # _emit <- warn_* <- solve() <- caller: stacklevel 4 reports the caller of solve().
    # The warn_* helpers are meant to be called only from a solver entry point.
    warnings.warn(message, category, stacklevel=stacklevel)


def _zone_target_message(problem: ZonalProblem) -> str | None:
    unknown, dups = problem.zone_target_issues()
    if not unknown and not dups:
        return None
    parts: list[str] = []
    if unknown:
        parts.append(
            f"{len(unknown)} row(s) with unknown zone/feature ids (skipped, as MarZone "
            "does): " + _first_five([f"zone {z}, feature {f}" for _, z, f in unknown])
        )
    if dups:
        parts.append(
            f"{len(dups)} duplicate (zone, feature) pair(s) (last row wins): "
            + _first_five([f"zone {z}, feature {f}" for z, f in dups])
        )
    return "; and ".join(parts)


def warn_zone_target_issues(problem: ZonalProblem, solver_name: str) -> None:
    """Emit one ``ZoneTargetWarning`` when ``zone_targets`` has skipped/collapsed rows."""
    msg = _zone_target_message(problem)
    if msg:
        _emit(f"{solver_name}: zone_targets has {msg}", ZoneTargetWarning)


def warn_zone_input_issues(problem: ZonalProblem, solver_name: str) -> None:
    """At most one ``ZoneTargetWarning`` and one ``ZoneLockWarning`` per zone-solver call."""
    msg = _zone_target_message(problem)
    if msg:
        _emit(f"{solver_name}: zone_targets has {msg}", ZoneTargetWarning)
    issues = problem.pu_lock_issues()
    if issues:
        _emit(f"{solver_name}: PU zone restrictions: " + " | ".join(issues), ZoneLockWarning)
```

In `mip_solver.py`, `solver.py`, `heuristic.py`, `iterative_improvement.py`: replace the import of `warn_zone_target_issues` with `warn_zone_input_issues` and every call `warn_zone_target_issues(problem, "<Name>")` with `warn_zone_input_issues(problem, "<Name>")` (five call sites incl. `ZoneIISolver.improve`).

`__init__.py`: `from pymarxan.zones.model import ZonalProblem, ZoneInputWarning, ZoneLockWarning, ZoneTargetWarning`; add `"ZoneInputWarning",` and `"ZoneLockWarning",` to `__all__` (alphabetical among the `Zone*` names).

- [ ] **Step 4: Run tests**

Run: `/opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/zones -q`
Expected: all PASS (existing `ZoneTargetWarning` tests unchanged; golden tests unchanged).

- [ ] **Step 5: Lint, type-check, commit**

```bash
~/.local/bin/ruff check src/pymarxan/zones tests/pymarxan/zones
/home/razinka/marxan/.venv/bin/mypy src/pymarxan/zones --ignore-missing-imports
git add src/pymarxan/zones tests/pymarxan/zones/test_pu_zone_rules.py tests/pymarxan/zones/test_package_exports.py
git commit -m "feat(zones): pu_lock_issues, validate() errors, ZoneInputWarning/ZoneLockWarning, one dispatcher at every zone solver entry (#6)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: `pulock.dat` / `puzone.dat` readers, writers and loader

**Files:**
- Modify: `src/pymarxan/zones/readers.py`, `src/pymarxan/zones/writers.py`, `src/pymarxan/zones/__init__.py`
- Modify: `tests/pymarxan/zones/test_package_exports.py`
- Test: `tests/pymarxan/zones/test_pu_zone_io.py`

**Interfaces:**
- Produces: `read_pu_lock(path) -> pd.DataFrame`, `read_pu_zone(path) -> pd.DataFrame` (columns `pu, zone`, int); `write_pu_lock(df, path)`, `write_pu_zone(df, path)` (header `puid,zoneid`); `load_zone_project` fills `pu_locks` / `pu_zones` only when `PULOCKNAME` / `PUZONENAME` is in the parameters. All four functions exported from `pymarxan.zones`.

- [ ] **Step 1: Write the failing tests** — `tests/pymarxan/zones/test_pu_zone_io.py`:

```python
"""pulock.dat / puzone.dat I/O (#6 spec §1; MarZone pu.hpp:708-790)."""
from __future__ import annotations

import shutil
import warnings
from pathlib import Path

import pandas as pd
import pytest

from pymarxan.zones import (
    load_zone_project,
    read_pu_lock,
    read_pu_zone,
    write_pu_lock,
    write_pu_zone,
)

ZONES_DIR = Path(__file__).parent.parent.parent / "data" / "zones"
MARZONE_PULOCK = b"puid, zoneid\r\n2,1\r\n5,2\r\n30,1"


def _frame(rows):
    return pd.DataFrame(rows, columns=["pu", "zone"])


def test_reads_marzone_fixture_bytes(tmp_path):
    f = tmp_path / "pulock.dat"
    f.write_bytes(MARZONE_PULOCK)
    pd.testing.assert_frame_equal(read_pu_lock(f), _frame([(2, 1), (5, 2), (30, 1)]))


def test_bom_and_crlf(tmp_path):
    f = tmp_path / "puzone.dat"
    f.write_bytes(b"\xef\xbb\xbfpuid,zoneid\r\n1,1\r\n1,2\r\n")
    pd.testing.assert_frame_equal(read_pu_zone(f), _frame([(1, 1), (1, 2)]))


def test_headerless_tab_and_space(tmp_path):
    f = tmp_path / "pulock.dat"
    f.write_text("1\t2\n3 1\n")
    pd.testing.assert_frame_equal(read_pu_lock(f), _frame([(1, 2), (3, 1)]))


def test_header_only_is_empty(tmp_path):
    f = tmp_path / "pulock.dat"
    f.write_text("puid,zoneid\n")
    out = read_pu_lock(f)
    assert list(out.columns) == ["pu", "zone"] and len(out) == 0


@pytest.mark.parametrize("content", [b"", b"puid,zoneid\n1,\n", b"puid,zoneid\n1,2.5\n"])
def test_bad_files_raise(tmp_path, content):
    f = tmp_path / "pulock.dat"
    f.write_bytes(content)
    with pytest.raises(ValueError, match="pulock.dat"):
        read_pu_lock(f)


def test_writers_round_trip(tmp_path):
    df = _frame([(1, 2), (4, 1)])
    write_pu_lock(df, tmp_path / "a.dat")
    write_pu_zone(df, tmp_path / "b.dat")
    assert (tmp_path / "a.dat").read_text().splitlines()[0] == "puid,zoneid"
    pd.testing.assert_frame_equal(read_pu_lock(tmp_path / "a.dat"), df)
    pd.testing.assert_frame_equal(read_pu_zone(tmp_path / "b.dat"), df)


def _project(tmp_path, extra_lines: list[str]) -> Path:
    proj = tmp_path / "proj"
    shutil.copytree(ZONES_DIR, proj)
    with open(proj / "input.dat", "a") as fh:
        fh.write("\n" + "\n".join(extra_lines) + "\n")
    return proj


def _load(proj):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return load_zone_project(proj)


def test_loader_reads_only_named_files(tmp_path):
    proj = _project(tmp_path, [])
    (proj / "input" / "pulock.dat").write_text("puid,zoneid\n1,2\n")
    assert _load(proj).pu_locks is None  # not named in input.dat -> not loaded (MarZone)
    proj2 = _project(tmp_path / "b", ["PULOCKNAME pulock.dat", "PUZONENAME puzone.dat"])
    (proj2 / "input" / "pulock.dat").write_text("puid,zoneid\n1,2\n")
    (proj2 / "input" / "puzone.dat").write_text("puid,zoneid\n2,1\n2,2\n")
    p = _load(proj2)
    assert p.pu_lock_lookup() == {1: 2}
    assert p.pu_zone_lookup() == {2: frozenset({1, 2})}


def test_loader_drops_unknown_pus_from_marzone_file(tmp_path):
    proj = _project(tmp_path, ["PULOCKNAME pulock.dat"])
    (proj / "input" / "pulock.dat").write_bytes(MARZONE_PULOCK)
    assert _load(proj).pu_lock_lookup() == {2: 1}  # PUs 5 and 30 are not in pu.dat


def test_loader_named_empty_file_raises(tmp_path):
    proj = _project(tmp_path, ["PUZONENAME puzone.dat"])
    (proj / "input" / "puzone.dat").write_bytes(b"")
    with pytest.raises(ValueError, match="puzone.dat"):
        _load(proj)


def test_blank_middle_cell_raises(tmp_path):
    f = tmp_path / "pulock.dat"
    f.write_text("puid,zoneid\n1,,2\n")
    with pytest.raises(ValueError, match="pulock.dat"):
        read_pu_lock(f)


def test_loader_named_missing_file_raises(tmp_path):
    proj = _project(tmp_path, ["PULOCKNAME nope.dat"])
    with pytest.raises(FileNotFoundError, match="nope.dat"):
        _load(proj)
```

Add to `ZONES_EXPORTS`:

```python
    "read_pu_lock": "pymarxan.zones.readers",
    "read_pu_zone": "pymarxan.zones.readers",
    "write_pu_lock": "pymarxan.zones.writers",
    "write_pu_zone": "pymarxan.zones.writers",
```

`read_input_dat` keeps unknown keys (non-numeric values as strings), so `PULOCKNAME` / `PUZONENAME` reach `base.parameters`; no allow-list change is needed (verified in the plan dry run).

- [ ] **Step 2: Run to verify failure**

Run: `/opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/zones/test_pu_zone_io.py -q`
Expected: collection ERROR — `ImportError: cannot import name 'read_pu_lock'`.

- [ ] **Step 3: Implement**

`readers.py` — add `import re` and:

```python
# MarZone util.hpp is_like_numerical_data: a line made only of these is data, not a header.
_NUMERIC_LINE_CHARS = frozenset(" ,;:^*\"/\t'\\\n\r.+-Ee0123456789")
_WS = re.compile(r"\s+")


def _read_pu_zone_pairs(path: str | Path) -> pd.DataFrame:
    """Two integer columns read by position (MarZone LoadPuLock / LoadPuZone)."""
    path = Path(path)
    text = path.read_text(encoding="utf-8-sig")
    if text == "":
        raise ValueError(f"{path}: file is empty (MarZone: 'cannot be read or is empty')")
    lines = text.splitlines()
    start = 0 if set(lines[0]) <= _NUMERIC_LINE_CHARS else 1
    rows: list[tuple[int, int]] = []
    data = [ln for ln in lines[start:] if ln.strip()]
    sep = "," if data and "," in data[0] else ("\t" if data and "\t" in data[0] else None)
    for lineno, line in enumerate(lines[start:], start=start + 1):
        if not line.strip():
            continue
        raw = line.strip()
        cells = [c.strip() for c in raw.split(sep)] if sep else _WS.split(raw)
        if len(cells) < 2:
            raise ValueError(f"{path}: line {lineno} needs two values (puid, zoneid)")
        try:
            a, b = float(cells[0]), float(cells[1])
        except ValueError as exc:
            raise ValueError(f"{path}: line {lineno} has a non-numeric id") from exc
        if not (a.is_integer() and b.is_integer()):
            raise ValueError(f"{path}: line {lineno} has a non-integral id")
        rows.append((int(a), int(b)))
    return pd.DataFrame(rows, columns=["pu", "zone"], dtype=int)


def read_pu_lock(path: str | Path) -> pd.DataFrame:
    """Read ``pulock.dat`` (``puid, zoneid``) into columns ``pu, zone``."""
    return _read_pu_zone_pairs(path)


def read_pu_zone(path: str | Path) -> pd.DataFrame:
    """Read ``puzone.dat`` (``puid, zoneid``, one row per allowed zone) into ``pu, zone``."""
    return _read_pu_zone_pairs(path)
```

One delimiter per file, detected from the first data line (`,`, else tab, else whitespace), so a blank cell (`1,,2`, `1,`) fails the numeric parse. Note: `"1,"` gives cells `["1", ""]` → "non-numeric id" `ValueError` (the `match="pulock.dat"` test passes because the path is in every message). An empty DataFrame from a header-only file must still have int dtype — `pd.DataFrame([], columns=[...], dtype=int)` does that.

In `load_zone_project`, before `problem = ZonalProblem(...)`:

```python
    pu_locks = None
    pu_zones = None
    for key, reader, target in (
        ("PULOCKNAME", read_pu_lock, "pu_locks"),
        ("PUZONENAME", read_pu_zone, "pu_zones"),
    ):
        name = base.parameters.get(key)
        if name is None:
            continue  # MarZone loads these files only when named
        path = input_dir / str(name)
        if not path.exists():
            raise FileNotFoundError(f"{project_dir}: {key} names {path}, which does not exist")
        if target == "pu_locks":
            pu_locks = reader(path)
        else:
            pu_zones = reader(path)
```

and pass `pu_locks=pu_locks, pu_zones=pu_zones,` to the `ZonalProblem(...)` call.

`writers.py`:

```python
def _write_pu_zone_pairs(df: pd.DataFrame, path: str | Path) -> None:
    df[["pu", "zone"]].rename(columns={"pu": "puid", "zone": "zoneid"}).to_csv(
        path, index=False,
    )


def write_pu_lock(df: pd.DataFrame, path: str | Path) -> None:
    """Write ``pu_locks`` as MarZone ``pulock.dat`` (header ``puid,zoneid``)."""
    _write_pu_zone_pairs(df, path)


def write_pu_zone(df: pd.DataFrame, path: str | Path) -> None:
    """Write ``pu_zones`` as MarZone ``puzone.dat`` (header ``puid,zoneid``)."""
    _write_pu_zone_pairs(df, path)
```

`__init__.py`: import and export `read_pu_lock`, `read_pu_zone`, `write_pu_lock`, `write_pu_zone` (alphabetical in `__all__`).

- [ ] **Step 4: Run tests** — `/opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/zones -q` → all PASS.

- [ ] **Step 5: Lint, type-check, commit**

```bash
~/.local/bin/ruff check src/pymarxan/zones tests/pymarxan/zones
/home/razinka/marxan/.venv/bin/mypy src/pymarxan/zones --ignore-missing-imports
git add src/pymarxan/zones/readers.py src/pymarxan/zones/writers.py src/pymarxan/zones/__init__.py tests/pymarxan/zones/test_pu_zone_io.py tests/pymarxan/zones/test_package_exports.py
git commit -m "feat(zones): read/write MarZone pulock.dat and puzone.dat; load only when PULOCKNAME/PUZONENAME is set (#6)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: ZoneMIPSolver honours the rules

**Files:**
- Modify: `src/pymarxan/zones/mip_solver.py` (variable/lock block in `solve`; `has_values` probe; `_warn_no_solution`)
- Test: `tests/pymarxan/zones/test_pu_zone_solvers.py` (create)

**Interfaces:**
- Consumes: `problem.pu_zone_rules()` (via `_pu_zone_rules`), `PUZoneRule.allowed`, `.may_be_unassigned`.

- [ ] **Step 1: Write the failing tests** — create `tests/pymarxan/zones/test_pu_zone_solvers.py`:

```python
"""Zone solvers honour pu_locks / pu_zones / status rules (#6 spec §4)."""
from __future__ import annotations

import warnings

import pandas as pd
import pytest

from pymarxan.solvers.base import SolverConfig
from pymarxan.zones import ZoneMIPNoSolutionWarning, ZoneMIPSolver
from pymarxan.zones.model import rule_violations
from tests.pymarxan.zones.marzone_anchor import make_anchor_problem

CONFIG = SolverConfig(num_solutions=1, seed=1)


def _table(rows):
    return pd.DataFrame(rows, columns=["pu", "zone"])


def _with_pu4(**kw):
    p = make_anchor_problem(**kw)
    pu = pd.concat(
        [p.planning_units, pd.DataFrame({"id": [4], "cost": [0.0], "status": [0]})],
        ignore_index=True,
    )
    zc = pd.concat(
        [p.zone_costs, pd.DataFrame({"pu": [4, 4], "zone": [1, 2], "cost": [3.0, 2.0]})],
        ignore_index=True,
    )
    return p.copy_with(planning_units=pu, zone_costs=zc)


def _strict_subset():
    """3 zones; zone 3 is free and fully contributing, but every PU is restricted to {1, 2}.
    Feasible under the restriction ((1, 2, 2) at 7.0); any solver ignoring it drifts to 3."""
    p = make_anchor_problem(spf=10.0)
    zones = pd.DataFrame({"id": [1, 2, 3], "name": ["protect", "restore", "use"]})
    zc = pd.concat(
        [p.zone_costs, pd.DataFrame({"pu": [1, 2, 3], "zone": [3, 3, 3], "cost": [0.0] * 3})],
        ignore_index=True,
    )
    contrib = pd.concat(
        [p.zone_contributions,
         pd.DataFrame({"feature": [1], "zone": [3], "contribution": [1.0]})],
        ignore_index=True,
    )
    return p.copy_with(
        zones=zones, zone_costs=zc, zone_contributions=contrib,
        pu_zones=_table([(i, z) for i in (1, 2, 3) for z in (1, 2)]),
    )


def _mip(p):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return ZoneMIPSolver().solve(p, CONFIG)


def _assign(sols):
    return [tuple(int(z) for z in s.zone_assignment) for s in sols]


class TestMIP:
    def test_lock_moves_anchor_optimum(self):
        sols = _mip(make_anchor_problem().copy_with(pu_locks=_table([(1, 2)])))
        assert _assign(sols) == [(2, 1, 2)] and sols[0].objective == pytest.approx(8.0)

    def test_unrestricted_pu4_variant(self):
        sols = _mip(_with_pu4())
        assert _assign(sols) == [(1, 2, 2, 0)] and sols[0].objective == pytest.approx(7.0)

    def test_puzone_forbids_unassigned(self):
        sols = _mip(_with_pu4().copy_with(pu_zones=_table([(4, 1), (4, 2)])))
        assert _assign(sols) == [(1, 2, 2, 2)] and sols[0].objective == pytest.approx(9.0)

    def test_status_semantics_unchanged(self):
        p = make_anchor_problem().copy_with(
            planning_units=make_anchor_problem().planning_units.assign(status=[2, 0, 0]),
        )
        sols = _mip(p)
        assert sols[0].zone_assignment[0] == 1

    def test_strict_subset(self):
        p = _strict_subset()
        sols = _mip(p)
        assert _assign(sols) == [(1, 2, 2)]
        assert rule_violations(p.pu_zone_rules(), sols[0].zone_assignment) == []

    def test_rules_hold_on_output(self):
        p = _with_pu4().copy_with(pu_locks=_table([(1, 2)]), pu_zones=_table([(4, 1), (4, 2)]))
        sols = _mip(p)
        assert sols and rule_violations(p.pu_zone_rules(), sols[0].zone_assignment) == []

    def test_lock_induced_infeasibility_message(self):
        p = make_anchor_problem().copy_with(pu_locks=_table([(1, 2), (2, 2), (3, 2)]))
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            assert ZoneMIPSolver().solve(p, CONFIG) == []
        msgs = [str(w.message) for w in rec if issubclass(w.category, ZoneMIPNoSolutionWarning)]
        assert len(msgs) == 1
        assert "feature 1 is 15 but at most 12" in msgs[0]
        assert "given the PU zone restrictions" in msgs[0]
```

- [ ] **Step 2: Run to verify failure**

Run: `/opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/zones/test_pu_zone_solvers.py -q`
Expected: 3 FAIL — (`test_strict_subset` already passes: since Task 1 the MIP bounds disallowed zones) `test_puzone_forbids_unassigned` returns `(1, 2, 2, 0)`; `test_rules_hold_on_output` reports violation `[3]` (PU4 unassigned); `test_lock_induced_infeasibility_message` lacks the restriction phrase. Pins that already pass (since Task 1 the MIP reads locks/puzone through `rule.forced` / `rule.allowed`): `test_lock_moves_anchor_optimum`, `test_unrestricted_pu4_variant`, `test_status_semantics_unchanged`. The only new MIP behaviour in this task is `== 1` for must-assign PUs plus the message.

- [ ] **Step 3: Implement** in `mip_solver.py`. Replace the per-PU block inside `for k, pid in enumerate(pu_ids):` (from the variable loop through the `locked_out_` constraint) with:

```python
        for k, pid in enumerate(pu_ids):
            pid = int(pid)
            rule = rules[k]
            for zid in zone_ids:
                var = pulp.LpVariable(f"x_{pid}_{zid}", cat="Binary")
                if zid not in rule.allowed:
                    var.upBound = 0
                x[(pid, zid)] = var
            assigned = pulp.lpSum(x[(pid, zid)] for zid in zone_ids)
            if rule.may_be_unassigned:
                model += assigned <= 1, f"assign_{pid}"
            else:
                model += assigned == 1, f"assign_{pid}"
```

Replace the two probe lines (`first_pid = int(pu_ids[0])` and `has_values = pulp.value(x[(first_pid, zone_ids[0])]) is not None`) with the single line:

```python
        has_values = any(var.varValue is not None for var in x.values())
```

In `_warn_no_solution`, replace the `else:` clause of the `try` (the block that builds the "unreachable" and "compete" messages) with:

```python
        else:
            note = (
                " (given the PU zone restrictions)"
                if problem.pu_locks is not None or problem.pu_zones is not None
                else ""
            )
            if unreachable:
                more = f" (+{len(unreachable) - 5} more)" if len(unreachable) > 5 else ""
                msg = (
                    "ZoneMIPSolver: infeasible. Unreachable targets: "
                    + "; ".join(unreachable[:5]) + more + note
                )
            else:
                msg = (
                    f"ZoneMIPSolver: infeasible, but every target is reachable on its own{note} "
                    "— targets compete for the same planning units. ZoneSASolver treats "
                    "targets as penalties and will return a best-effort plan."
                )
```

(The diagnostic-failure message is unchanged.)

Update the class docstring's constraint list: replace the two status lines with
"x[i,z] fixed to 0 for zones outside a PU's rule; Σ_z x[i,z] == 1 for PUs that must be assigned, <= 1 otherwise (``ZonalProblem.pu_zone_rules``: status, pu_locks, pu_zones)".

- [ ] **Step 4: Run tests** — `/opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/zones -q` → all PASS (incl. existing locked-in/out MIP tests and `tests/data/simple` paths untouched).

- [ ] **Step 5: Lint, type-check, commit**

```bash
~/.local/bin/ruff check src/pymarxan/zones tests/pymarxan/zones
/home/razinka/marxan/.venv/bin/mypy src/pymarxan/zones --ignore-missing-imports
git add src/pymarxan/zones/mip_solver.py tests/pymarxan/zones/test_pu_zone_solvers.py
git commit -m "feat(zones): ZoneMIPSolver honours pu_locks/pu_zones via variable bounds and assign equality (#6)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: ZoneSASolver honours the rules

**Files:**
- Modify: `src/pymarxan/zones/solver.py` (anchors by content: the `locked: dict[int, int] = {}` status block, the initial-assignment loop, the temperature-estimation draw, the annealing-loop draw)
- Test: `tests/pymarxan/zones/test_pu_zone_solvers.py` (append)

**Interfaces:**
- Consumes: `problem.pu_zone_rules()`, `fixed_zones(rules)` (Task 1).

- [ ] **Step 1: Append failing tests** (merge `ZoneSASolver` into the `pymarxan.zones` import):

```python
def _fast(p, n=3000):
    return p.copy_with(parameters={**p.parameters, "NUMITNS": n, "NUMTEMP": 30})


def _solve(cls, p, n_sol=3, seed=3):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return cls().solve(p, SolverConfig(num_solutions=n_sol, seed=seed))


RESTRICTED = [
    ("lock", lambda: make_anchor_problem(spf=10.0).copy_with(pu_locks=_table([(1, 2)]))),
    ("must_assign", lambda: _with_pu4(spf=10.0).copy_with(pu_zones=_table([(4, 1), (4, 2)]))),
    ("strict_subset", _strict_subset),
]


class TestSA:
    @pytest.mark.parametrize("name,make", RESTRICTED, ids=[r[0] for r in RESTRICTED])
    def test_rules_hold(self, name, make):
        p = _fast(make())
        for s in _solve(ZoneSASolver, p):
            assert rule_violations(p.pu_zone_rules(), s.zone_assignment) == []
```

- [ ] **Step 2: Run to verify failure** — `/opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/zones/test_pu_zone_solvers.py::TestSA -q` → FAIL: `test_rules_hold[lock]` (PU1 leaves zone 2) and `test_rules_hold[strict_subset]` (PUs drift to zone 3). `test_rules_hold[must_assign]` already passes (zone SA never proposes zone 0) — kept as a pin.

- [ ] **Step 3: Implement** in `solver.py`. Import `from pymarxan.zones.model import ZonalProblem, fixed_zones, warn_zone_input_issues`. Replace the status-parsing block (`locked: dict[int, int] = {}` … through `swappable = np.array(...)`) with:

```python
        rules = problem.pu_zone_rules()
        locked = fixed_zones(rules)
        every = frozenset(zone_ids_list)
        opts: list[np.ndarray] = [
            zone_options if rule.allowed == every
            else np.array(sorted(rule.allowed), dtype=int)
            for rule in rules
        ]
        swappable = np.array(
            [i for i in range(n_pu) if i not in locked], dtype=int
        )
```

Add a module-level helper (after the imports):

```python
def _propose(
    rng: np.random.Generator, swappable: np.ndarray, opts: list[np.ndarray],
) -> tuple[int, int]:
    """A random swappable PU and a random zone from its options. Same two RNG draws as
    v0.37 for unrestricted PUs (whose options are the shared ``zone_options`` array)."""
    idx = int(swappable[rng.integers(len(swappable))])
    options = opts[idx]
    return idx, int(options[rng.integers(len(options))])
```

Initial assignment — replace the `for idx in swappable:` block with:

```python
            for idx in swappable:
                start = rules[idx].start
                if start is not None:
                    assignment[idx] = start
                else:
                    options = opts[idx]
                    assignment[idx] = options[rng.integers(len(options))]
```

Temperature loop — replace the two lines drawing `idx` and `new_zone` with `idx, new_zone = _propose(rng, swappable, opts)` followed by `old_zone = int(assignment[idx])` (keep the `if new_zone == old_zone: continue`). Do the same in the annealing loop. Remove the now-unused `initial_include`, `pu_id_to_idx` and `n_zone_options` locals and the old status loop (keep `n_swappable`: the all-locked branch still uses it). Put `_propose` two blank lines after the imports, before `_COOLING_FACTORIES`.

Note on byte-identity: v0.37 drew `rng.integers(n_swappable)` then `rng.integers(n_zone_options)`; `_propose` makes the same two draws with the same bounds for free PUs, and status-1 PUs start at `rules[idx].start == min(zone_ids)` exactly as `initial_include` did. The golden tests verify this.

- [ ] **Step 4: Run tests** — `/opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/zones -q` → all PASS, **including all `test_zone_golden_v037.py` SA cases**. If a golden SA case fails, the RNG draw order changed — fix the code, never the golden values.

- [ ] **Step 5: Lint, type-check, commit**

```bash
~/.local/bin/ruff check src/pymarxan/zones tests/pymarxan/zones
/home/razinka/marxan/.venv/bin/mypy src/pymarxan/zones --ignore-missing-imports
git add src/pymarxan/zones/solver.py tests/pymarxan/zones/test_pu_zone_solvers.py
git commit -m "feat(zones): ZoneSASolver draws moves from each PU's allowed zones; v0.37 RNG stream kept for free PUs (#6)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 6: Heuristic and II honour the rules; `improve()` repair

**Files:**
- Modify: `src/pymarxan/zones/heuristic.py` (`solve`, `_greedy_assign`; delete `_parse_status`)
- Modify: `src/pymarxan/zones/iterative_improvement.py` (`solve`, `improve`, `_improve`, the three passes; delete `_parse_locked`)
- Test: `tests/pymarxan/zones/test_pu_zone_solvers.py` (append)

**Interfaces:**
- Consumes: `problem.pu_zone_rules()`, `fixed_zones`, `rule_violations` (Task 1).
- Produces (private, changed signatures): `ZoneHeuristicSolver._greedy_assign(problem, assignment, locked, rules, blm)`; `ZoneIISolver._improve(problem, assignment, swappable, rules, blm, itimptype)`; `_removal_pass(problem, assignment, swappable, rules, blm)`; `_addition_pass(...)`, `_two_step(...)`, `_swap_pass(...)` all take `rules: list[PUZoneRule]` in place of `zone_ids`.

- [ ] **Step 1: Append failing tests** (merge `ZoneHeuristicSolver, ZoneIISolver` into the import; add `from pymarxan.solvers.run_mode import RunModePipeline` and `import numpy as np`):

```python
def _ii(p, itimptype):
    return p.copy_with(parameters={**p.parameters, "ITIMPTYPE": itimptype})


class TestHeuristicAndII:
    @pytest.mark.parametrize("name,make", RESTRICTED, ids=[r[0] for r in RESTRICTED])
    def test_heuristic_rules_hold(self, name, make):
        p = make()
        for s in _solve(ZoneHeuristicSolver, p, n_sol=1):
            assert rule_violations(p.pu_zone_rules(), s.zone_assignment) == []

    @pytest.mark.parametrize("itimptype", [1, 2, 3])
    @pytest.mark.parametrize("name,make", RESTRICTED, ids=[r[0] for r in RESTRICTED])
    def test_ii_rules_hold(self, name, make, itimptype):
        p = _ii(make(), itimptype)
        for s in _solve(ZoneIISolver, p, n_sol=1):
            assert rule_violations(p.pu_zone_rules(), s.zone_assignment) == []

    def test_ii_addition_moves_must_assign_pu_to_better_zone(self):
        p = _ii(_with_pu4(spf=10.0).copy_with(pu_zones=_table([(4, 1), (4, 2)])), 2)
        (s,) = _solve(ZoneIISolver, p, n_sol=1)
        assert int(s.zone_assignment[3]) == 2  # seeded at 1 (cost 3), zone 2 costs 2

    def test_improve_repairs_out_of_rule_input(self):
        p = _ii(_with_pu4(spf=10.0).copy_with(
            pu_locks=_table([(1, 2)]), pu_zones=_table([(4, 1), (4, 2)]),
        ), 3)
        (seed_sol,) = _solve(ZoneHeuristicSolver, make_anchor_problem(spf=10.0), n_sol=1)
        bad = seed_sol
        bad.zone_assignment = np.array([1, 2, 2, 0])  # PU1 not in its lock, PU4 unassigned
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = ZoneIISolver().improve(p, bad)
        assert rule_violations(p.pu_zone_rules(), out.zone_assignment) == []

    @pytest.mark.parametrize("runmode", [0, 1, 2, 3, 4, 5, 6])
    def test_pipeline_respects_rules(self, runmode):
        p = _fast(_with_pu4(spf=10.0).copy_with(
            pu_locks=_table([(1, 2)]), pu_zones=_table([(4, 1), (4, 2)]),
        ))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sols = RunModePipeline(runmode=runmode).solve(p, SolverConfig(num_solutions=2, seed=5))
        for s in sols:
            assert rule_violations(p.pu_zone_rules(), s.zone_assignment) == []

    @pytest.mark.parametrize(
        "cls", [ZoneMIPSolver, ZoneSASolver, ZoneIISolver], ids=lambda c: c.__name__,
    )
    def test_non_contiguous_zone_ids_solvers(self, cls):
        p = make_anchor_problem(spf=10.0)
        relabel = {1: 5, 2: 9}
        p = _ii(_fast(p.copy_with(
            zones=p.zones.assign(id=p.zones["id"].map(relabel)),
            zone_costs=p.zone_costs.assign(zone=p.zone_costs["zone"].map(relabel)),
            zone_contributions=p.zone_contributions.assign(
                zone=p.zone_contributions["zone"].map(relabel),
            ),
            zone_targets=p.zone_targets.assign(zone=p.zone_targets["zone"].map(relabel)),
            planning_units=p.planning_units.assign(status=[2, 1, 0]),
            pu_zones=_table([(3, 9)]),
        )), 3)
        sols = _solve(cls, p, n_sol=1)
        assert sols
        for s in sols:
            assert set(int(z) for z in s.zone_assignment) <= {0, 5, 9}
            assert rule_violations(p.pu_zone_rules(), s.zone_assignment) == []

    @pytest.mark.parametrize(
        "cls", [ZoneMIPSolver, ZoneSASolver, ZoneHeuristicSolver, ZoneIISolver],
        ids=lambda c: c.__name__,
    )
    def test_all_forced_every_solver(self, cls):
        base = _with_pu4(spf=10.0)
        p = _fast(base.copy_with(
            planning_units=base.planning_units.assign(status=[0, 3, 0, 2]),
            pu_locks=_table([(1, 2), (3, 2)]),
            features=base.features.assign(target=0.0),  # forced plan must be feasible
            zone_targets=None,
        ))
        p = _ii(p, 3)
        sols = _solve(cls, p, n_sol=1)
        assert len(sols) == 1  # never vacuous (the MIP must not return [])
        assert tuple(int(z) for z in sols[0].zone_assignment) == (2, 0, 2, 1)
```

(If `Solution.zone_assignment` is read-only on the dataclass, build `bad` with `dataclasses.replace(seed_sol, zone_assignment=np.array([1, 2, 2, 0]))` instead of assignment.)

- [ ] **Step 2: Run to verify failure** — `/opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/zones/test_pu_zone_solvers.py::TestHeuristicAndII -q` → FAIL. Record the exact count in the commit message. Must fail: every `strict_subset` case (heuristic, II 1/2/3), `test_ii_addition_moves_must_assign_pu_to_better_zone`, `test_improve_repairs_out_of_rule_input`, and the `test_all_forced_every_solver` cases for the heuristic and II. Some `lock` cases may pass on v0.37 by coincidence — the strict-subset and all-forced tests carry the proof.

- [ ] **Step 3: Implement**

`heuristic.py` — import `from pymarxan.zones.model import PUZoneRule, ZonalProblem, fixed_zones, warn_zone_input_issues`. In `solve`, replace `locked, initial_include = self._parse_status(...)` with:

```python
        rules = problem.pu_zone_rules()
        locked = fixed_zones(rules)
```

and the per-run seeding with:

```python
            assignment = np.zeros(n_pu, dtype=int)
            for idx, zid in locked.items():
                assignment[idx] = zid
            for idx, rule in enumerate(rules):
                if idx in locked or rule.may_be_unassigned:
                    continue  # free PUs start at 0 as in v0.37 (status 1 has no effect here)
                assignment[idx] = rule.start if rule.start is not None else min(rule.allowed)

            assignment = self._greedy_assign(problem, assignment, locked, rules, blm)
```

`_greedy_assign(self, problem, assignment, locked, rules: list[PUZoneRule], blm)`: replace `for zid in zone_ids:` with `for zid in sorted(rules[idx].allowed):`. Delete `_parse_status` and the unused `pu_id_to_idx` / `zone_ids_list` locals if they become unused.

`iterative_improvement.py` — import `PUZoneRule, fixed_zones` from `pymarxan.zones.model`. Add a module helper:

```python
def _seed(rule: PUZoneRule) -> int:
    return rule.start if rule.start is not None else min(rule.allowed)
```

`solve`: replace `locked = self._parse_locked(...)` with `rules = problem.pu_zone_rules()`, `locked = fixed_zones(rules)`; seeding of swappable PUs: `assignment[idx] = _seed(rules[idx])` (v0.37 used `zone_ids_list[0]`, which equals `_seed` for every free PU); call `self._improve(problem, assignment, swappable, rules, blm, itimptype)`.

In `solve` and `improve`, delete the now-unused `pu_id_to_idx` and `zone_ids_list` locals (ruff F841).

`improve`: same `rules`/`locked`/`swappable`; after copying the incoming assignment, repair it:

```python
        for idx, rule in enumerate(rules):
            zone = int(assignment[idx])
            if idx in locked:
                assignment[idx] = locked[idx]
            elif (zone == 0 and not rule.may_be_unassigned) or (
                zone != 0 and zone not in rule.allowed
            ):
                assignment[idx] = min(rule.allowed)
```

`_improve(self, problem, assignment, swappable, rules, blm, itimptype)` dispatches with `rules` instead of `zone_ids`.

`_removal_pass(self, problem, assignment, swappable, rules, blm)`: add at the top of the loop body `if not rules[idx].may_be_unassigned: continue`.

`_addition_pass(self, problem, assignment, swappable, rules, blm)`:

```python
        current_obj = compute_zone_objective(problem, assignment, blm)
        for idx in swappable:
            rule = rules[idx]
            old_zone = int(assignment[idx])
            if old_zone == 0:
                candidates = sorted(rule.allowed)
            elif not rule.may_be_unassigned and len(rule.allowed) > 1:
                candidates = [z for z in sorted(rule.allowed) if z != old_zone]
            else:
                continue
            best_delta = 0.0
            best_zone = old_zone
            for zid in candidates:
                assignment[idx] = zid
                delta = compute_zone_objective(problem, assignment, blm) - current_obj
                if delta < best_delta:
                    best_delta = delta
                    best_zone = zid
                assignment[idx] = old_zone
            if best_zone != old_zone:
                assignment[idx] = best_zone
                current_obj += best_delta
        return assignment
```

`_two_step(self, problem, assignment, swappable, rules, blm)`: pass `rules` to both passes.

`_swap_pass(self, problem, assignment, swappable, rules, blm)`: delete `all_options = [0, *zone_ids]`; inside the loop use

```python
                rule = rules[idx]
                options = ([0] if rule.may_be_unassigned else []) + sorted(rule.allowed)
                for zid in options:
```

Delete `_parse_locked`.

- [ ] **Step 4: Run tests** — `/opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/zones tests/pymarxan/solvers -q` → all PASS, **including every heuristic and II golden case**. If a golden case fails, the candidate order or seeding drifted from v0.37 — fix the code, never the golden values.

- [ ] **Step 5: Lint, type-check, commit**

```bash
~/.local/bin/ruff check src/pymarxan/zones tests/pymarxan/zones
/home/razinka/marxan/.venv/bin/mypy src/pymarxan/zones --ignore-missing-imports
git add src/pymarxan/zones/heuristic.py src/pymarxan/zones/iterative_improvement.py tests/pymarxan/zones/test_pu_zone_solvers.py
git commit -m "feat(zones): heuristic and II honour PU zone rules; II per-pass candidates; improve() repairs out-of-rule input (#6)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 7: Replacement cost, bench, docs, full gate

**Files:**
- Modify: `src/pymarxan/analysis/replacement_cost.py` (the `variant = problem.copy_with(planning_units=problem.planning_units.copy())` construction inside the loop)
- Create: `tests/pymarxan/analysis/test_replacement_cost_zone_locks.py`, `tests/benchmarks/bench_zone_pu_restrictions.py`
- Modify: `CHANGELOG.md` (`## [Unreleased]`), `docs/VALIDATION.md` (end of §4, before `## Comparing against the Marxan C++ binary`)

- [ ] **Step 1: Failing test** — `tests/pymarxan/analysis/test_replacement_cost_zone_locks.py`:

```python
"""Replacement cost's status-3 variant must not be overridden by a pulock (#6 R2-6)."""
from __future__ import annotations

import math
import warnings

import pandas as pd

from pymarxan.analysis.replacement_cost import compute_replacement_cost
from pymarxan.zones import ZoneMIPSolver
from tests.pymarxan.zones.marzone_anchor import make_anchor_problem


def test_locked_pu_replacement_cost_not_zero():
    p = make_anchor_problem().copy_with(
        pu_locks=pd.DataFrame({"pu": [1], "zone": [1]}),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scores = compute_replacement_cost(p, solver=ZoneMIPSolver())  # type: ignore[arg-type]
    # Without PU1, PUs 2-3 cannot meet overall 15 and zone-2 10 together -> infeasible.
    assert scores[1] == math.inf
```

Run: `/opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/analysis/test_replacement_cost_zone_locks.py -q` → FAIL (`0.0 != inf`: the lock wins over status 3).

- [ ] **Step 2: Implement** — in `compute_replacement_cost`, replace the variant construction with:

```python
        overrides: dict[str, object] = {"planning_units": problem.planning_units.copy()}
        for name in ("pu_locks", "pu_zones"):  # zone problems: the lock would beat status 3
            table = getattr(problem, name, None)
            if table is not None and "pu" in table.columns:
                overrides[name] = table[table["pu"] != pid].reset_index(drop=True)
        variant = problem.copy_with(**overrides)
```

(keep the following `.loc[...] = STATUS_LOCKED_OUT` and the parameters deepcopy). Add one docstring sentence: "For zone problems the PU's ``pu_locks`` / ``pu_zones`` rows are dropped in the variant; other status-based analyses do not consult them." Re-run the test → PASS.

- [ ] **Step 3: Bench** — `tests/benchmarks/bench_zone_pu_restrictions.py`:

```python
"""SA per-proposal cost with ~50 % of PUs puzone-restricted vs none (#6 spec §5)."""
from __future__ import annotations

import statistics
import time

import numpy as np
import pandas as pd
import pytest

from pymarxan.zones.cache import ZoneProblemCache
from pymarxan.zones.model import fixed_zones
from pymarxan.zones.solver import _propose
from tests.benchmarks.conftest import make_zone_problem

pytestmark = pytest.mark.bench
N_PU, N_FEAT, N_ZONES, N_MOVES = 1000, 50, 3, 20_000


def _opts(problem):
    rules = problem.pu_zone_rules()
    zone_options = np.array(sorted(problem.zone_ids), dtype=int)
    every = frozenset(zone_options.tolist())
    opts = [zone_options if r.allowed == every else np.array(sorted(r.allowed), dtype=int)
            for r in rules]
    swappable = np.array([i for i in range(len(rules)) if i not in fixed_zones(rules)])
    return opts, swappable, zone_options


def _per_evaluated_delta(problem) -> float:
    """Seconds per evaluated delta (no-op proposals excluded, so the restricted run's higher
    no-op share cannot hide proposal overhead)."""
    opts, swappable, _ = _opts(problem)
    cache = ZoneProblemCache.from_zone_problem(problem)
    assignment = np.array([o[0] for o in opts])
    held = cache.compute_held(assignment)
    samples = []
    for rep in range(5):
        rng = np.random.default_rng(rep)
        evaluated = 0
        t0 = time.perf_counter()
        for _ in range(N_MOVES):
            idx, new_zone = _propose(rng, swappable, opts)
            old_zone = int(assignment[idx])
            if new_zone != old_zone:
                cache.compute_delta_zone_objective(
                    idx, old_zone, new_zone, assignment, held=held, blm=1.0,
                )
                evaluated += 1
        samples.append((time.perf_counter() - t0) / evaluated)
    return statistics.median(samples)


def _propose_seconds(draw) -> float:
    samples = []
    for _ in range(5):
        t0 = time.perf_counter()
        for _ in range(N_MOVES):
            draw()
        samples.append((time.perf_counter() - t0) / N_MOVES)
    return statistics.median(samples)


def _restricted(base):
    ids = base.planning_units["id"].astype(int).tolist()[::2]
    rows = [(pid, z) for pid in ids for z in (1, 2)]
    return base.copy_with(pu_zones=pd.DataFrame(rows, columns=["pu", "zone"]))


def test_restrictions_cost_at_most_ten_percent_per_delta() -> None:
    base = make_zone_problem(n_pu=N_PU, n_feat=N_FEAT, n_zones=N_ZONES, density=0.3, seed=42)
    ratio = _per_evaluated_delta(_restricted(base)) / _per_evaluated_delta(base)
    assert ratio <= 1.10, f"restricted/unrestricted per-delta ratio {ratio:.3f}"


def test_propose_vs_v037_inline_draw() -> None:
    base = make_zone_problem(n_pu=N_PU, n_feat=N_FEAT, n_zones=N_ZONES, density=0.3, seed=42)
    opts, swappable, zone_options = _opts(base)
    n_sw, n_z = len(swappable), len(zone_options)
    rng_a, rng_b = np.random.default_rng(0), np.random.default_rng(0)

    def v037() -> tuple[int, int]:
        idx = int(swappable[rng_a.integers(n_sw)])
        return idx, int(zone_options[rng_a.integers(n_z)])

    ratio = _propose_seconds(lambda: _propose(rng_b, swappable, opts)) / _propose_seconds(v037)
    assert ratio <= 1.10, f"_propose / v0.37 inline draw {ratio:.3f}"
```

Run: `/opt/micromamba/envs/shiny/bin/pytest tests/benchmarks/bench_zone_pu_restrictions.py tests/benchmarks/bench_zone_sa.py tests/benchmarks/bench_zone_overall.py -m bench -q` → PASS (dry run: per-delta ratio ≈ 1.0, `_propose` vs v0.37 ≈ 1.02–1.08). Report both ratios in the commit message.

- [ ] **Step 4: Docs** — `CHANGELOG.md` under `## [Unreleased]`:

```markdown
### Added
- Zone problems: MarZone `pulock.dat` (lock a planning unit into a named zone) and
  `puzone.dat` (restrict it to a set of zones), as `ZonalProblem.pu_locks` / `pu_zones`,
  `read_pu_lock` / `read_pu_zone` / `write_pu_lock` / `write_pu_zone`, and loaded by
  `load_zone_project` when `PULOCKNAME` / `PUZONENAME` is set (#6). All four zone solvers
  honour them. Precedence: pulock → status 3 → status 2 (within puzone if given) → puzone.
  A puzone PU must take one of its listed zones.
- `ZonalProblem.pu_zone_rules()` / `pu_lock_lookup()` / `pu_zone_lookup()` /
  `pu_lock_issues()`; `validate()` reports unknown, repeated, duplicate and conflicting rows.
- `ZoneLockWarning` and the common base `ZoneInputWarning` (also the base of
  `ZoneTargetWarning`), one per zone-solver call.

### Changed
- `ZoneIISolver.improve()` first repairs an incoming assignment that breaks a PU's rule
  (e.g. a status-2 PU outside its zone) before improving it.
- `compute_replacement_cost` drops a PU's `pu_locks` / `pu_zones` rows in its lock-out variant.
  Other status edits (e.g. WDPA lock-in) do not override a `pu_locks` row.
- `ZoneLockWarning` is emitted once per zone-solver call; `RunModePipeline` RUNMODEs 2–5 call
  several zone solvers per run and warn once per stage.
- Zone projects have no `save_zone_project`; `write_pu_lock` / `write_pu_zone` are standalone
  writers.
```

`docs/VALIDATION.md` — append before `## Comparing against the Marxan C++ binary`:

```markdown
**PU locks and zone restrictions (since v0.38).** `pulock.dat` and `puzone.dat` follow MarZone
`LoadPuLock` / `PersistPuLock` / `LoadPuZone` / `PersistPuZone` (`pu.hpp:483-526`, `:708-790`):
the last pulock row per PU wins and a lock naming an unknown zone is then dropped; unknown puzone
zones are skipped and a PU left with none is unrestricted; a pulock overrides everything
(`reserve.hpp:100-103`); a puzone PU always takes one of its listed zones. Deliberate
differences: MarZone does not read a pu.dat status column, so pymarxan's status 2/3 locks have no
MarZone counterpart; a single-zone puzone, fatal in MarZone, is treated as a lock and reported;
MarZone's `CalcPenalties` pre-counts pulocked amounts when calibrating penalties
(`marzone.cpp:769-782`), which pymarxan's `spf × shortfall` objective does not model.
`tests/pymarxan/zones/test_pu_zone_solvers.py` pins the exact-MIP anchors (lock: `(2, 1, 2)` at
8.0; must-assign: `(1, 2, 2, 2)` at 9.0) and the rules on every solver's output.
```

- [ ] **Step 5: Full gate**

```bash
PATH="/opt/micromamba/envs/shiny/bin:$HOME/.local/bin:/home/razinka/marxan/.venv/bin:$PATH" make check
PYTHONPATH=src /opt/micromamba/envs/shiny/bin/python examples/validate_marxan_parity.py
```

Expected: lint + types clean; full suite green, coverage ≥ 75 %; parity MIP 35.0, round-trip True.

- [ ] **Step 6: Commit**

```bash
git add src/pymarxan/analysis/replacement_cost.py tests/pymarxan/analysis/test_replacement_cost_zone_locks.py tests/benchmarks/bench_zone_pu_restrictions.py CHANGELOG.md docs/VALIDATION.md
git commit -m "feat(zones): replacement cost drops PU lock rows; restriction bench; CHANGELOG + VALIDATION for pulock/puzone (#6)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

## Follow-ons (not in this plan)

- **ROWER_pymarxan_DST on the v0.38 pin bump:** derive `attainable_by_target` from
  `problem.pu_zone_rules()` (per-PU `allowed`) instead of status masks; add `pu_locks`,
  `pu_zones` (and `zone_boundary_costs`) to `problem_hash` so scenarios differing only in locks
  do not collide.
- `save_zone_project` writing every zone table incl. pulock/puzone (roadmap).
