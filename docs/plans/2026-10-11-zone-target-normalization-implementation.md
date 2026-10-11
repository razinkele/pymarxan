# Zone-Target Normalization (#7) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Rev 2** — patched after the plan review (`2026-10-11-zone-target-normalization-plan-review.md`):
inert-pair convention for met-flags (ROWER compatibility), robust id parsing, exported warning
classes, per-solver-call warning counts, mutation-killing tests, worktree-safe gate commands.

**Goal:** Make every zone component (MIP, DataFrame objective, SA cache, summary writer) treat `zone_targets` with MarZone load semantics, report bad rows in `validate()` and once per zone-solver call, explain an infeasible zone MIP, and publish `compute_zone_achieved`.

**Architecture:** One cheap accessor `ZonalProblem.zone_target_lookup(*, include_inert=False)` (insertion-ordered dict; unknown ids skipped; last row wins; `t <= 0` pairs dropped unless `include_inert`) replaces every raw row loop over `zone_targets`. Met-flags and the summary use `include_inert=True` and report inert pairs as met (the `check_overall_targets` convention); shortfalls, penalty, cache and MIP use effective pairs only. A shared `zone_target_issues()` feeds `validate()` and a solve-entry `ZoneTargetWarning`. New `zones/diagnostics.py` holds the MIP's amount source (`_feature_groups`, moved), the per-PU zone rule (`_pu_zone_rules`, shared with the MIP), `diagnose_zone_infeasibility` and `ZoneMIPNoSolutionWarning`.

**Tech Stack:** Python 3.12, numpy, pandas, PuLP (CBC), pytest.

**Spec:** `docs/plans/2026-10-11-zone-target-normalization-design.md` (rev 3). Reviews: `…-review.md` (design), `…-plan-review.md` (this plan).

## Global Constraints

- Every file starts with `from __future__ import annotations`; full type hints; ruff (E, F, I, UP; line length 99) and mypy clean.
- Tests run ONLY with `/opt/micromamba/envs/shiny/bin/pytest` (never `.venv`).
- **In a git worktree** the `.venv` lives in the main checkout: run mypy as `/home/razinka/marxan/.venv/bin/mypy`, and run any non-pytest Python against the worktree's code with `PYTHONPATH=src` (the editable install points at the main checkout; only pytest picks up `src/` via `pyproject`'s `pythonpath`).
- MarZone semantics: unknown zone/feature ids skipped; repeated `(zone, feature)` → last row wins; then pairs with `target <= 0` are inert — no penalty, no MIP constraint (`zones.hpp:131,142,148`; `reserve.hpp:241,472,846` @ `85082e3`).
- Met-flags (`check_zone_targets`, `metadata["zone_targets_met"]`, summary rows) cover every known pair including inert ones, which count as met — mirroring `check_overall_targets`.
- MIP return value on failure stays `[]` (no API change); a diagnostic failure must never turn it into an exception.
- `validate()` never raises; it returns `list[str]`.
- No new dataclass fields on `ZonalProblem`. New `ZoneProblemCache` field must have a default.
- Bench assertions are comparative, never absolute wall-clock.
- Ruff the **test** files at every task handoff, not only `src/`.
- Anchors must not move: `tests/data/simple` MIP = 35.0; MarZone anchor `(1, 2, 2)` at 7.0.
- Commit messages reference `(#7)` and end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Review Focus

1. **A problem whose `zone_targets` rows are all unknown or inert** — no penalty, no MIP constraint, objective and MIP solution identical to `zone_targets=None`; met-flags list only the known inert pairs, as met. Tests: Task 2 `test_all_rows_inert_add_no_penalty`, Task 3 `test_all_rows_inert_mip_matches_none`.
2. **`RunModePipeline` with bad rows** — one `ZoneTargetWarning` per zone-solver call (RUNMODE 1 → 1 per run, 2 → 2, 5 → 3), never an exception. Test: Task 3 `test_pipeline_warns_once_per_solver_call`.
3. **A target on a feature with no amounts at all** (the "ghost" feature) — the MIP warning names it ("at most 0"). Test: Task 4 `test_ghost_feature_named_in_mip_warning`.
4. **Duplicate `pu_vs_features` rows** for one PU — the diagnostic sums them exactly as the MIP does. Test: Task 4 `test_duplicate_puvspr_rows_summed`.
5. **Float, fractional, NaN and non-numeric ids** in `zone_targets` — `2.0` matches zone 2; `2.5`, NaN and `'x'` are unknown; `validate()` reports them and never raises. Tests: Task 1 `test_odd_ids`, `test_validate_never_raises_on_odd_ids`.

---

## File Structure

| File | Responsibility | Change |
|---|---|---|
| `src/pymarxan/zones/model.py` | Lookup, issue detection, `ZoneTargetWarning`, `warn_zone_target_issues`, `validate()` | Modify |
| `src/pymarxan/zones/objective.py` | Met-flags/shortfalls from the lookup; public `compute_zone_achieved` | Modify |
| `src/pymarxan/zones/cache.py` | Lookup-filled target matrix; negative-weight mask | Modify |
| `src/pymarxan/zones/writers.py` | Summary rows from the lookup | Modify |
| `src/pymarxan/zones/diagnostics.py` | `_feature_groups` (moved), `PUZoneRule`, `_pu_zone_rules`, `diagnose_zone_infeasibility`, `ZoneMIPNoSolutionWarning` | Create |
| `src/pymarxan/zones/mip_solver.py` | Lookup in 3 helpers; shared rules; failure warning; entry warning | Modify |
| `src/pymarxan/zones/{solver,heuristic,iterative_improvement}.py` | Entry warning | Modify |
| `src/pymarxan/zones/__init__.py` | Exports | Modify |
| `tests/pymarxan/zones/test_zone_target_lookup.py` | Task 1 | Create |
| `tests/pymarxan/zones/test_zone_target_agreement.py` | Tasks 2–3 | Create |
| `tests/pymarxan/zones/test_zone_diagnostics.py` | Task 4 | Create |
| `tests/pymarxan/zones/{test_package_exports,test_zone_mip_metadata,test_writers}.py` | Exports; raw-row helpers → lookup | Modify |
| `tests/benchmarks/bench_zone_target_lookup.py` | Task 5 | Create |
| `CHANGELOG.md`, `docs/VALIDATION.md` | Docs | Modify |

---

### Task 1: Lookup, issue detection, `ZoneTargetWarning`, `validate()`

**Files:**
- Modify: `src/pymarxan/zones/model.py` (imports; helpers above the class; methods after `get_contribution`, which ends ≈ line 78; `validate()` at line 137; module-level class/function at end of file)
- Modify: `src/pymarxan/zones/__init__.py` (export `ZoneTargetWarning`)
- Modify: `tests/pymarxan/zones/test_package_exports.py` (`ZONES_EXPORTS`)
- Test: `tests/pymarxan/zones/test_zone_target_lookup.py`

**Interfaces:**
- Produces:
  - `ZonalProblem.zone_target_lookup(self, *, include_inert: bool = False) -> dict[tuple[int, int], float]` — key `(zone_id, feature_id)`, raw target, insertion order = first appearance; `include_inert=True` also keeps known pairs whose surviving target is `<= 0` (incl. NaN targets).
  - `ZonalProblem.zone_target_issues(self) -> tuple[list[tuple[int, object, object]], list[tuple[int, int]]]` — `(unknown rows as (position, zone, feature), duplicate keys in first-duplicate order)`.
  - `class ZoneTargetWarning(UserWarning)` in `pymarxan.zones.model`, exported from `pymarxan.zones`.
  - `warn_zone_target_issues(problem: ZonalProblem, solver_name: str) -> None` in `pymarxan.zones.model`.

- [ ] **Step 1: Write the failing tests**

Create `tests/pymarxan/zones/test_zone_target_lookup.py`:

```python
"""zone_target_lookup / zone_target_issues / validate() / ZoneTargetWarning (#7, spec §1–2)."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from pymarxan.zones import ZoneTargetWarning
from pymarxan.zones.model import warn_zone_target_issues
from tests.pymarxan.zones.marzone_anchor import make_anchor_problem


def _with_targets(rows: list[tuple[object, object, float]]):
    p = make_anchor_problem()
    zt = pd.DataFrame(rows, columns=["zone", "feature", "target"])
    return p.copy_with(zone_targets=zt)


class TestLookup:
    def test_anchor(self):
        assert make_anchor_problem().zone_target_lookup() == {(2, 1): 10.0}

    def test_last_row_wins_both_orders(self):
        assert _with_targets([(2, 1, 25.0), (2, 1, 10.0)]).zone_target_lookup() == {(2, 1): 10.0}
        assert _with_targets([(2, 1, 10.0), (2, 1, 25.0)]).zone_target_lookup() == {(2, 1): 25.0}

    def test_unknown_ids_skipped(self):
        p = _with_targets([(9, 1, 10.0), (2, 99, 1.0), (2, 1, 10.0)])
        assert p.zone_target_lookup() == {(2, 1): 10.0}

    def test_positive_then_zero_switches_pair_off(self):
        p = _with_targets([(2, 1, 10.0), (2, 1, 0.0)])
        assert p.zone_target_lookup() == {}
        assert p.zone_target_lookup(include_inert=True) == {(2, 1): 0.0}

    def test_zero_then_positive_keeps_positive(self):
        assert _with_targets([(2, 1, 0.0), (2, 1, 10.0)]).zone_target_lookup() == {(2, 1): 10.0}

    def test_negative_is_inert(self):
        p = _with_targets([(1, 1, -5.0)])
        assert p.zone_target_lookup() == {}
        assert p.zone_target_lookup(include_inert=True) == {(1, 1): -5.0}

    def test_include_inert_never_includes_unknown_ids(self):
        p = _with_targets([(9, 1, 0.0), (2, 1, 0.0)])
        assert p.zone_target_lookup(include_inert=True) == {(2, 1): 0.0}

    def test_insertion_order_is_first_appearance(self):
        p = _with_targets([(2, 1, 1.0), (1, 1, 2.0), (2, 1, 3.0)])
        assert list(p.zone_target_lookup().items()) == [((2, 1), 3.0), ((1, 1), 2.0)]

    def test_odd_ids(self):
        p = _with_targets([(2.0, 1.0, 10.0), (np.nan, 1, 5.0), (1, np.nan, 5.0),
                           ("x", 1, 5.0), (2.5, 1, 5.0)])
        assert p.zone_target_lookup() == {(2, 1): 10.0}
        unknown, dups = p.zone_target_issues()
        assert [pos for pos, _, _ in unknown] == [1, 2, 3, 4]
        assert dups == []

    def test_none_and_missing_columns(self):
        p = make_anchor_problem()
        assert p.copy_with(zone_targets=None).zone_target_lookup() == {}
        bad = pd.DataFrame({"zone": [2], "feature": [1]})
        assert p.copy_with(zone_targets=bad).zone_target_lookup() == {}
        assert p.copy_with(zone_targets=bad).zone_target_issues() == ([], [])


class TestIssues:
    def test_clean_anchor(self):
        assert make_anchor_problem().zone_target_issues() == ([], [])

    def test_unknown_and_duplicates(self):
        p = _with_targets([(9, 1, 10.0), (2, 1, 25.0), (2, 99, 1.0), (2, 1, 10.0), (2, 1, 3.0)])
        unknown, dups = p.zone_target_issues()
        assert [(pos, z, f) for pos, z, f in unknown] == [(0, 9, 1), (2, 2, 99)]
        assert dups == [(2, 1)]


class TestValidate:
    def test_anchor_has_no_errors(self):
        assert make_anchor_problem().validate() == []

    def test_unknown_ids_reported(self):
        errors = _with_targets([(9, 1, 10.0), (2, 1, 10.0)]).validate()
        assert any("zone_targets" in e and "unknown" in e and "zone 9" in e for e in errors)

    def test_duplicates_reported(self):
        errors = _with_targets([(2, 1, 25.0), (2, 1, 10.0)]).validate()
        assert any("duplicate" in e and "last-wins" in e and "zone 2, feature 1" in e
                   for e in errors)

    def test_validate_never_raises_on_odd_ids(self):
        errors = _with_targets([("x", 1, 5.0), (2.5, 1, 5.0)]).validate()
        assert any("zone x" in e and "zone 2.5" in e for e in errors)

    def test_duplicate_contribution_rows_reported(self):
        p = make_anchor_problem()
        zc = pd.DataFrame({"feature": [1, 1, 1], "zone": [1, 2, 2],
                           "contribution": [1.0, 0.4, 0.5]})
        errors = p.copy_with(zone_contributions=zc).validate()
        assert any("zone_contributions" in e and "duplicate" in e and "feature 1, zone 2" in e
                   for e in errors)


class TestWarning:
    def test_silent_on_clean(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            warn_zone_target_issues(make_anchor_problem(), "ZoneSASolver")

    def test_one_warning_with_counts(self):
        p = _with_targets([(9, 1, 10.0), (2, 1, 25.0), (2, 1, 10.0)])
        with pytest.warns(ZoneTargetWarning) as rec:
            warn_zone_target_issues(p, "ZoneSASolver")
        assert len(rec) == 1
        msg = str(rec[0].message)
        assert msg.startswith("ZoneSASolver: zone_targets has")
        assert "1 row(s) with unknown" in msg and "1 duplicate" in msg

    def test_is_user_warning_subclass(self):
        assert issubclass(ZoneTargetWarning, UserWarning)
```

In `tests/pymarxan/zones/test_package_exports.py`, add to the `ZONES_EXPORTS` dict (insert after the `"ZoneSASolver"` entry):

```python
    "ZoneTargetWarning": "pymarxan.zones.model",
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `/opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/zones/test_zone_target_lookup.py tests/pymarxan/zones/test_package_exports.py -q`
Expected: collection ERROR — `ImportError: cannot import name 'ZoneTargetWarning' from 'pymarxan.zones'`.

- [ ] **Step 3: Implement in `src/pymarxan/zones/model.py`**

Top-of-file imports become:

```python
from __future__ import annotations

import warnings
from collections import Counter
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from pymarxan.models.problem import ConservationProblem
```

Add module-level helpers directly above `@dataclass class ZonalProblem`:

```python
def _first_five(items: list[str]) -> str:
    """'a; b; c; d; e …' — the first five items, with an ellipsis when more exist."""
    return "; ".join(items[:5]) + (" …" if len(items) > 5 else "")


def _known_id(value: object, known: set[int]) -> int | None:
    """The integer id ``value`` names when it is integral and in ``known``; else ``None``.

    Accepts ``2`` and ``2.0``; rejects NaN, inf, ``2.5`` and non-numeric values (never raises).
    """
    try:
        as_float = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if not as_float.is_integer():
        return None
    ident = int(as_float)
    return ident if ident in known else None
```

Add these methods to `ZonalProblem` directly after `get_contribution`:

```python
    # ------------------------------------------------------------------
    # Zone targets (MarZone zones.hpp BuildZoneTarget; reserve.hpp target > 0 gate)
    # ------------------------------------------------------------------

    def zone_target_lookup(
        self, *, include_inert: bool = False,
    ) -> dict[tuple[int, int], float]:
        """Zone targets keyed ``(zone_id, feature_id)`` under MarZone load semantics.

        Rows naming a zone not in ``zone_ids`` or a feature not in ``features["id"]``
        (or a non-integral id) are skipped; a repeated pair keeps its LAST row
        (``zones.hpp`` BuildZoneTarget). Pairs whose surviving target is not > 0 are
        inert (``reserve.hpp`` applies zone targets only when > 0) and are dropped
        unless ``include_inert`` — met-flags and the summary keep them, as met.
        Insertion order is each kept pair's first appearance. ``{}`` when
        ``zone_targets`` is ``None`` or lacks ``zone``/``feature``/``target``.
        Never warns. Last-wins on ``(zone, feature)`` is exact because the reader
        rejects occurrence targettypes 2/3; supporting them makes the key per field.
        """
        zt = self.zone_targets
        if zt is None or not {"zone", "feature", "target"}.issubset(zt.columns):
            return {}
        zones = {int(z) for z in self.zone_ids}
        feats = {int(f) for f in self.features["id"].values}
        out: dict[tuple[int, int], float] = {}
        for z, f, t in zip(
            zt["zone"].tolist(), zt["feature"].tolist(), zt["target"].tolist(), strict=True,
        ):
            zid, fid = _known_id(z, zones), _known_id(f, feats)
            if zid is not None and fid is not None:
                out[(zid, fid)] = float(t)
        if include_inert:
            return out
        return {key: t for key, t in out.items() if t > 0}

    def zone_target_issues(
        self,
    ) -> tuple[list[tuple[int, object, object]], list[tuple[int, int]]]:
        """(unknown rows as ``(position, zone, feature)``, duplicate ``(zone, feature)`` keys).

        The rows ``zone_target_lookup`` skips or collapses; shared by ``validate()`` and
        ``warn_zone_target_issues``. ``([], [])`` when ``zone_targets`` is ``None`` or
        lacks ``zone``/``feature``/``target``.
        """
        zt = self.zone_targets
        if zt is None or not {"zone", "feature", "target"}.issubset(zt.columns):
            return [], []
        zones = {int(z) for z in self.zone_ids}
        feats = {int(f) for f in self.features["id"].values}
        unknown: list[tuple[int, object, object]] = []
        seen: set[tuple[int, int]] = set()
        dups: list[tuple[int, int]] = []
        for pos, (z, f) in enumerate(
            zip(zt["zone"].tolist(), zt["feature"].tolist(), strict=True),
        ):
            zid, fid = _known_id(z, zones), _known_id(f, feats)
            if zid is None or fid is None:
                unknown.append((pos, z, f))
                continue
            key = (zid, fid)
            if key in seen and key not in dups:
                dups.append(key)
            seen.add(key)
        return unknown, dups
```

In `validate()`, inside the `zone_contributions` `else:` branch, after the existing `if unknown:` error, add:

```python
                pairs = list(zip(self.zone_contributions["feature"].tolist(),
                                 self.zone_contributions["zone"].tolist(), strict=True))
                dup_pairs = [p for p, n in Counter(pairs).items() if n > 1]
                if dup_pairs:
                    errors.append(
                        f"zone_contributions has {len(dup_pairs)} duplicate (feature, zone) "
                        "pair(s), resolved last-wins: "
                        + _first_five([f"feature {f}, zone {z}" for f, z in dup_pairs])
                    )
```

Replace the `zone_targets` block (`if self.zone_targets is not None:` with the `req = {"zone", "feature", "target"}` check) with:

```python
        if self.zone_targets is not None:
            req = {"zone", "feature", "target"}
            if not req.issubset(set(self.zone_targets.columns)):
                errors.append(
                    f"zone_targets missing columns: "
                    f"{sorted(req - set(self.zone_targets.columns))}"
                )
            else:
                unknown_rows, dup_keys = self.zone_target_issues()
                if unknown_rows:
                    errors.append(
                        f"zone_targets references {len(unknown_rows)} row(s) with unknown "
                        "zone/feature ids (skipped by the solvers): "
                        + _first_five([f"zone {z}, feature {f}" for _, z, f in unknown_rows])
                    )
                if dup_keys:
                    errors.append(
                        f"zone_targets has {len(dup_keys)} duplicate (zone, feature) "
                        "pair(s), resolved last-wins by the solvers: "
                        + _first_five([f"zone {z}, feature {f}" for z, f in dup_keys])
                    )
```

At the end of `model.py` add:

```python
class ZoneTargetWarning(UserWarning):
    """``zone_targets`` rows the zone solvers skip (unknown ids) or collapse (duplicates)."""


def warn_zone_target_issues(problem: ZonalProblem, solver_name: str) -> None:
    """Emit one ``ZoneTargetWarning`` when ``problem.zone_target_issues()`` is non-empty.

    Called once at the entry of every zone-solver call (``solve`` and
    ``ZoneIISolver.improve``); nothing downstream warns.
    """
    unknown, dups = problem.zone_target_issues()
    if not unknown and not dups:
        return
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
    warnings.warn(
        f"{solver_name}: zone_targets has " + "; and ".join(parts),
        ZoneTargetWarning,
        stacklevel=3,
    )
```

In `src/pymarxan/zones/__init__.py`, change `from pymarxan.zones.model import ZonalProblem` to `from pymarxan.zones.model import ZonalProblem, ZoneTargetWarning`, and add `"ZoneTargetWarning",` to `__all__` directly after `"ZoneSASolver",`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `/opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/zones/test_zone_target_lookup.py tests/pymarxan/zones/test_model.py tests/pymarxan/zones/test_package_exports.py -q`
Expected: all PASS.

- [ ] **Step 5: Lint, type-check, commit**

```bash
~/.local/bin/ruff check src/pymarxan/zones tests/pymarxan/zones
/home/razinka/marxan/.venv/bin/mypy src/pymarxan/zones --ignore-missing-imports
git add src/pymarxan/zones/model.py src/pymarxan/zones/__init__.py tests/pymarxan/zones/test_zone_target_lookup.py tests/pymarxan/zones/test_package_exports.py
git commit -m "feat(zones): zone_target_lookup/zone_target_issues, ZoneTargetWarning, validate() reports bad zone-target and duplicate contribution rows (#7)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: Objective, cache and writer read the lookup; public `compute_zone_achieved`; cache negative-weight mask

**Files:**
- Modify: `src/pymarxan/zones/objective.py` (`_compute_zone_achieved` at ≈238; `check_zone_targets`; `compute_zone_shortfalls`)
- Modify: `src/pymarxan/zones/cache.py` (field list ≈96-106; zone-target fill block ≈180-189; `_compute_zone_penalty`; `_penalty_delta`)
- Modify: `src/pymarxan/zones/writers.py` (imports ≈10-15; zone block ≈174-193)
- Modify: `src/pymarxan/zones/__init__.py`; `tests/pymarxan/zones/test_package_exports.py`; `tests/pymarxan/zones/test_zone_mip_metadata.py:22-23`; `tests/pymarxan/zones/test_writers.py:256-261`
- Test: `tests/pymarxan/zones/test_zone_target_agreement.py`

**Interfaces:**
- Consumes: `ZonalProblem.zone_target_lookup(*, include_inert=False)` (Task 1).
- Produces: `pymarxan.zones.objective.compute_zone_achieved(problem, zone_assignment) -> dict[tuple[int, int], float]` (public; `_compute_zone_achieved` kept as alias, exported from `pymarxan.zones`); `ZoneProblemCache.zone_target_active: np.ndarray | None` (default `None`; built only when `zone_target_weight` has a negative entry). `check_zone_targets` keys = `zone_target_lookup(include_inert=True)`; `compute_zone_shortfalls` keys = `zone_target_lookup()`.

- [ ] **Step 1: Write the failing tests**

Create `tests/pymarxan/zones/test_zone_target_agreement.py`:

```python
"""All zone components agree on bad zone_targets rows (#7, spec §1 + cache addendum).

The dup_* DataFrame/cache cases and the cache delta test are pins (those paths were already
last-wins / consistent); the RED cases are unknown ids in the objective, the writer, the
negative-weight cache mask and the public name.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from pymarxan.zones import compute_zone_achieved
from pymarxan.zones.cache import ZoneProblemCache
from pymarxan.zones.objective import (
    _compute_zone_achieved,
    build_zone_solution,
    check_zone_targets,
    compute_zone_objective,
    compute_zone_shortfalls,
)
from pymarxan.zones.writers import write_zone_summary
from tests.pymarxan.zones.marzone_anchor import OPTIMUM, make_anchor_problem

ASSIGN = np.array(OPTIMUM)
VARIANTS = {
    "dup_25_then_10": [(2, 1, 25.0), (2, 1, 10.0)],
    "dup_10_then_25": [(2, 1, 10.0), (2, 1, 25.0)],
    "dup_zone1_10_then_20": [(1, 1, 10.0), (1, 1, 20.0)],
    "unknown_zone": [(9, 1, 10.0), (2, 1, 10.0)],
    "unknown_feature": [(2, 99, 1.0), (2, 1, 10.0)],
    "all_three": [(9, 1, 10.0), (2, 99, 1.0), (2, 1, 25.0), (2, 1, 10.0)],
}
NEG_CONTRIB = pd.DataFrame({"feature": [1, 1], "zone": [1, 2], "contribution": [-0.5, 0.4]})


def _variant(rows, **kw):
    p = make_anchor_problem(**kw)
    return p.copy_with(zone_targets=pd.DataFrame(rows, columns=["zone", "feature", "target"]))


def _hand_normalized(p):
    lookup = p.zone_target_lookup()
    zt = pd.DataFrame(
        [(z, f, t) for (z, f), t in lookup.items()], columns=["zone", "feature", "target"],
    )
    return p.copy_with(zone_targets=zt if len(zt) else None)


def _cache_objective(p, assignment, blm=0.0):
    cache = ZoneProblemCache.from_zone_problem(p)
    return cache.compute_full_zone_objective(
        assignment, held=cache.compute_held(assignment), blm=blm,
    )


@pytest.mark.parametrize("name", sorted(VARIANTS))
def test_dataframe_and_cache_agree_with_hand_normalized(name):
    p = _variant(VARIANTS[name], spf=10.0)
    expected = compute_zone_objective(_hand_normalized(p), ASSIGN, 0.0)
    assert compute_zone_objective(p, ASSIGN, 0.0) == pytest.approx(expected)
    assert _cache_objective(p, ASSIGN) == pytest.approx(expected)


def test_unknown_zone_no_phantom_penalty():
    # Before #7: 17.0 (spf 1) — the unknown zone 9 row added a 10.0 shortfall.
    assert compute_zone_objective(_variant(VARIANTS["unknown_zone"]), ASSIGN, 0.0) == 7.0


def test_met_flags_and_shortfalls_keyed_by_lookup():
    p = _variant(VARIANTS["all_three"])
    assert set(check_zone_targets(p, ASSIGN)) == {(2, 1)}
    assert set(compute_zone_shortfalls(p, ASSIGN)) == {(2, 1)}
    sol = build_zone_solution(p, ASSIGN, 0.0, solver_name="t")
    assert set(sol.metadata["zone_targets_met"]) == {"z2_f1"}


def test_all_rows_inert_add_no_penalty():
    p = _variant([(9, 1, 10.0), (2, 1, 10.0), (2, 1, 0.0)])
    none = make_anchor_problem().copy_with(zone_targets=None)
    # Known inert pair stays in the met-flags, as met (check_overall_targets convention).
    assert check_zone_targets(p, ASSIGN) == {(2, 1): True}
    assert compute_zone_shortfalls(p, ASSIGN) == {}
    assert compute_zone_objective(p, ASSIGN, 0.0) == compute_zone_objective(none, ASSIGN, 0.0)
    assert _cache_objective(p, ASSIGN) == pytest.approx(_cache_objective(none, ASSIGN))


def test_inert_pair_met_even_when_weighted_achievement_is_negative():
    p = make_anchor_problem(zone_target_contrib=1).copy_with(
        zone_contributions=NEG_CONTRIB,
        zone_targets=pd.DataFrame({"zone": [1], "feature": [1], "target": [0.0]}),
    )
    assert check_zone_targets(p, ASSIGN) == {(1, 1): True}  # achieved -5, target inert


def test_cache_no_phantom_penalty_for_untargeted_pair_with_negative_weight():
    p = make_anchor_problem(zone_target_contrib=1).copy_with(zone_contributions=NEG_CONTRIB)
    df = compute_zone_objective(p, ASSIGN, 0.0)
    assert df == 21.0  # cost 7 + overall (15 - (-5 + 8)) = 12 + zone 2 (10 - 8) = 2
    assert _cache_objective(p, ASSIGN) == pytest.approx(df)  # was 26.0


def test_cache_mask_gate():
    assert ZoneProblemCache.from_zone_problem(make_anchor_problem()).zone_target_active is None
    pos = make_anchor_problem(zone_target_contrib=1)
    assert ZoneProblemCache.from_zone_problem(pos).zone_target_active is None
    neg = pos.copy_with(zone_contributions=NEG_CONTRIB)
    active = ZoneProblemCache.from_zone_problem(neg).zone_target_active
    assert active is not None
    assert not active[1, 0] and active[2, 0]  # zone 1 untargeted, zone 2 targeted


def test_cache_delta_matches_full_with_negative_weight():
    p = make_anchor_problem(zone_target_contrib=1).copy_with(zone_contributions=NEG_CONTRIB)
    cache = ZoneProblemCache.from_zone_problem(p)
    a = ASSIGN.copy()
    held = cache.compute_held(a)
    before = cache.compute_full_zone_objective(a, held=held, blm=0.0)
    delta = cache.compute_delta_zone_objective(0, int(a[0]), 2, a, held=held, blm=0.0)
    b = a.copy()
    b[0] = 2
    after = cache.compute_full_zone_objective(b, held=cache.compute_held(b), blm=0.0)
    assert before + delta == pytest.approx(after)


def test_writer_one_row_per_known_pair(tmp_path):
    p = _variant(VARIANTS["all_three"] + [(1, 1, 0.0)])
    sol = build_zone_solution(p, ASSIGN, 0.0, solver_name="t")
    out = tmp_path / "summary.csv"
    write_zone_summary(p, [sol], out)
    zone_rows = pd.read_csv(out).query("tier == 'zone'")
    assert list(zip(zone_rows["zone"], zone_rows["feature"], zone_rows["target"],
                    zone_rows["times_met"], strict=True)) == [(2, 1, 10.0, 1), (1, 1, 0.0, 1)]


def test_public_compute_zone_achieved():
    assert compute_zone_achieved is _compute_zone_achieved
    assert compute_zone_achieved(make_anchor_problem(), ASSIGN) == {(1, 1): 10.0, (2, 1): 20.0}


def test_no_warnings_from_objective_helpers():
    p = _variant(VARIANTS["all_three"])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        compute_zone_objective(p, ASSIGN, 0.0)
        _cache_objective(p, ASSIGN)
```

In `tests/pymarxan/zones/test_package_exports.py`, add to `ZONES_EXPORTS` (insert after the `"compute_overall_shortfalls"` entry):

```python
    "compute_zone_achieved": "pymarxan.zones.objective",
```

Switch the two in-tree raw-row helpers to the lookup:

- `tests/pymarxan/zones/test_zone_mip_metadata.py:22-23` — the helper body becomes
  `return {f"z{z}_f{f}" for z, f in problem.zone_target_lookup(include_inert=True)}`
  (delete the `zt = problem.zone_targets` line).
- `tests/pymarxan/zones/test_writers.py:256-261` — import `compute_zone_achieved` instead of `_compute_zone_achieved`, and replace the `for zid, fid, target in problem.zone_targets[[...]].itertuples(index=False,):` header with
  `for (zid, fid), target in problem.zone_target_lookup(include_inert=True).items():`
  (rename the later `_compute_zone_achieved(` call to `compute_zone_achieved(`).

- [ ] **Step 2: Run tests to verify they fail**

Run: `/opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/zones/test_zone_target_agreement.py tests/pymarxan/zones/test_package_exports.py -q`
Expected: collection ERROR — `ImportError: cannot import name 'compute_zone_achieved' from 'pymarxan.zones'`.

- [ ] **Step 3: Implement**

`objective.py` — rename the function `_compute_zone_achieved` to `compute_zone_achieved` (body unchanged) and directly after it add:

```python
_compute_zone_achieved = compute_zone_achieved  # pre-0.37 private name, kept for callers
```

Replace every internal call `_compute_zone_achieved(` in `objective.py` with `compute_zone_achieved(`, and in docstrings replace "``_compute_zone_achieved`` returns" with "``compute_zone_achieved`` returns".

`check_zone_targets` — docstring and body after the signature become:

```python
    """(zone id, feature id) -> Z_kf >= zone target × MISSLEVEL, for every known pair.

    Keys are ``ZonalProblem.zone_target_lookup(include_inert=True)``: inert pairs
    (surviving target <= 0) are reported as met, like ``check_overall_targets``.
    ``zone_achieved`` is the dict ``compute_zone_achieved`` returns; ``None`` recomputes
    it. The met test tolerates a relative 1e-9 to absorb float summation;
    ``compute_zone_shortfalls`` / ``compute_zone_penalty`` are exact.
    """
    lookup = problem.zone_target_lookup(include_inert=True)
    if not lookup:
        return {}
    misslevel = float(problem.parameters.get("MISSLEVEL", 1.0))
    if zone_achieved is None:
        zone_achieved = compute_zone_achieved(problem, zone_assignment)
    return {
        key: bool(not target > 0 or _meets(zone_achieved.get(key, 0.0), target * misslevel))
        for key, target in lookup.items()
    }
```

`compute_zone_shortfalls` — docstring and body after the signature become:

```python
    """(zone id, feature id) -> max(0, zone target × MISSLEVEL − Z_kf).

    For the effective zone targets (``ZonalProblem.zone_target_lookup``; inert pairs
    carry no shortfall and are absent).
    """
    lookup = problem.zone_target_lookup()
    if not lookup:
        return {}
    misslevel = float(problem.parameters.get("MISSLEVEL", 1.0))
    if zone_achieved is None:
        zone_achieved = compute_zone_achieved(problem, zone_assignment)
    return {
        key: max(0.0, target * misslevel - zone_achieved.get(key, 0.0))
        for key, target in lookup.items()
    }
```

`cache.py` — add as the last field (after `conn_weight: float`):

```python
    zone_target_active: np.ndarray | None = None
```

Document it in the class docstring after `zone_target_weight`:

```
    zone_target_active : np.ndarray | None
        (n_zones+1, n_feat) bool — True where a zone target exists. Built only when
        zone_target_weight has a negative entry (ZONETARGETCONTRIB=1 with a negative
        contribution) to mask phantom shortfalls of untargeted pairs; None otherwise.
```

Replace the zone-target fill block (the `zone_target_matrix = np.zeros(...)` line through the loop ending `zone_target_matrix[zcol, fcol] = float(zt_tg[k])`) with:

```python
        zone_target_matrix = np.zeros((n_zones + 1, n_feat), dtype=np.float64)
        for (zid, fid), target in problem.zone_target_lookup().items():
            zcol = zone_id_to_col.get(zid)
            fcol = feat_id_to_col.get(fid)
            if zcol is not None and fcol is not None:
                zone_target_matrix[zcol, fcol] = target
```

After both `zone_target_matrix *= misslevel` and the line that assigns `zone_target_weight` have run (place it after whichever comes later in `from_zone_problem`), add:

```python
        zone_target_active = (
            zone_target_matrix > 0 if bool(np.any(zone_target_weight < 0)) else None
        )
```

and pass `zone_target_active=zone_target_active,` as the last argument of the `return cls(...)` call.

`_compute_zone_penalty` body becomes:

```python
        shortfalls = np.maximum(
            0.0, self.zone_target_matrix - held_per_zone * self.zone_target_weight,
        )
        if self.zone_target_active is not None:
            shortfalls = shortfalls * self.zone_target_active
        return float(np.sum(shortfalls * self.feat_spf[np.newaxis, :]))
```

In `_penalty_delta`, in the `if old_col != 0:` block replace `delta += float(np.dot(self.feat_spf, new_shortfall - old_shortfall))` with:

```python
            diff = new_shortfall - old_shortfall
            if self.zone_target_active is not None:
                diff = diff * self.zone_target_active[old_col]
            delta += float(np.dot(self.feat_spf, diff))
```

and in the `if new_col != 0:` block with the same three lines using `self.zone_target_active[new_col]`.

`writers.py` — in the import block replace `_compute_zone_achieved,` with `compute_zone_achieved,` (keep the list sorted: it goes after `compute_overall_achieved,`). Replace the zone block (from `if problem.zone_targets is not None:` to the end of its loop) with:

```python
    lookup = problem.zone_target_lookup(include_inert=True)
    if lookup:
        zone_achieved = [compute_zone_achieved(problem, a) for a in assignments]
        zone_met = [check_zone_targets(problem, a) for a in assignments]
        for key, target in lookup.items():
            rows.append({
                "tier": "zone",
                "zone": key[0],
                "feature": key[1],
                "target": target,
                "mean_achieved": (
                    sum(a.get(key, 0.0) for a in zone_achieved) / total_runs
                    if total_runs else 0.0
                ),
                "times_met": sum(int(m.get(key, False)) for m in zone_met),
                "total_runs": total_runs,
            })
```

In `write_zone_summary`'s docstring, describe zone rows as: "one per known zone target (``ZonalProblem.zone_target_lookup(include_inert=True)``: unknown ids omitted, duplicates last-wins, inert targets <= 0 reported as met)". Wrap to ≤ 99 columns.

`__init__.py` — add `compute_zone_achieved` to the `from pymarxan.zones.objective import (...)` list and `"compute_zone_achieved",` to `__all__` directly after `"compute_overall_shortfalls",`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `/opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/zones/ -q`
Expected: all PASS.

- [ ] **Step 5: Lint, type-check, commit**

```bash
~/.local/bin/ruff check src/pymarxan/zones tests/pymarxan/zones
/home/razinka/marxan/.venv/bin/mypy src/pymarxan/zones --ignore-missing-imports
git add src/pymarxan/zones/objective.py src/pymarxan/zones/cache.py src/pymarxan/zones/writers.py src/pymarxan/zones/__init__.py tests/pymarxan/zones/test_zone_target_agreement.py tests/pymarxan/zones/test_package_exports.py tests/pymarxan/zones/test_zone_mip_metadata.py tests/pymarxan/zones/test_writers.py
git commit -m "fix(zones): objective, cache and summary writer read the zone-target lookup; cache masks phantom shortfalls under negative weights; public compute_zone_achieved (#7)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: MIP reads the lookup; solve-entry warning in all four zone solvers

**Files:**
- Modify: `src/pymarxan/zones/mip_solver.py` (`_zone_achieved_exprs`, `_build_penalty_expr`, `_add_zone_target_constraints`; entry warning after `raise_if_separation_active(problem, "ZoneMIPSolver")`)
- Modify: `src/pymarxan/zones/solver.py`, `heuristic.py`, `iterative_improvement.py` (after each `raise_if_separation_active(...)` line; top of `ZoneIISolver.improve`)
- Test: `tests/pymarxan/zones/test_zone_target_agreement.py` (append)

**Interfaces:**
- Consumes: `zone_target_lookup()`, `warn_zone_target_issues`, `ZoneTargetWarning` (Task 1).
- Produces: behaviour only.

- [ ] **Step 1: Append the failing tests** to `tests/pymarxan/zones/test_zone_target_agreement.py`, and merge these imports into the file's import block (ruff `I` orders them):

```python
from pymarxan.solvers.base import SolverConfig
from pymarxan.solvers.run_mode import RunModePipeline
from pymarxan.zones import (
    ZoneHeuristicSolver,
    ZoneIISolver,
    ZoneMIPSolver,
    ZoneSASolver,
    ZoneTargetWarning,
)
```

Tests to append:

```python
CONFIG = SolverConfig(num_solutions=1, seed=1)
SOLVERS = [ZoneMIPSolver, ZoneSASolver, ZoneHeuristicSolver, ZoneIISolver]
STAGES = {1: 1, 2: 2, 5: 3}  # zone-solver calls per run (run_mode.py docstring)


def _fast(p):
    return p.copy_with(parameters={**p.parameters, "NUMITNS": 1000})


def _mip(p):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return ZoneMIPSolver().solve(p, CONFIG)


def _zone_target_warnings(rec):
    return [w for w in rec if issubclass(w.category, ZoneTargetWarning)]


@pytest.mark.parametrize("name", sorted(VARIANTS))
def test_mip_matches_hand_normalized(name):
    p = _variant(VARIANTS[name])
    got, want = _mip(p), _mip(_hand_normalized(p))
    assert [list(s.zone_assignment) for s in got] == [list(s.zone_assignment) for s in want]
    assert [s.objective for s in got] == pytest.approx([s.objective for s in want])


def test_mip_duplicate_rows_no_longer_raise():
    sols = _mip(_variant(VARIANTS["dup_25_then_10"]))
    assert len(sols) == 1 and sols[0].objective == pytest.approx(7.0)


def test_mip_last_wins_is_discriminating():
    # Last row (zone-1 target 20) needs two PUs in zone 1: (1, 1, 0) at 11.0.
    # First-wins (10) would give the anchor optimum (1, 2, 2) at 7.0.
    sols = _mip(_variant(VARIANTS["dup_zone1_10_then_20"]))
    assert [list(s.zone_assignment) for s in sols] == [[1, 1, 0]]
    assert sols[0].objective == pytest.approx(11.0)


def test_all_rows_inert_mip_matches_none():
    p = _variant([(9, 1, 10.0), (2, 1, 10.0), (2, 1, 0.0)])
    none = make_anchor_problem().copy_with(zone_targets=None)
    got, want = _mip(p), _mip(none)
    assert [list(s.zone_assignment) for s in got] == [list(s.zone_assignment) for s in want]
    assert [s.objective for s in got] == pytest.approx([s.objective for s in want])


@pytest.mark.parametrize("cls", SOLVERS, ids=lambda c: c.__name__)
def test_each_solver_warns_exactly_once(cls):
    p = _fast(_variant(VARIANTS["all_three"], spf=10.0))
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        cls().solve(p, SolverConfig(num_solutions=2, seed=1))
    hits = _zone_target_warnings(rec)
    assert len(hits) == 1
    assert str(hits[0].message).startswith(f"{cls.__name__}: zone_targets has")


@pytest.mark.parametrize("cls", SOLVERS, ids=lambda c: c.__name__)
def test_each_solver_silent_on_clean_anchor(cls):
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        cls().solve(_fast(make_anchor_problem(spf=10.0)), CONFIG)
    assert not _zone_target_warnings(rec)


def test_ii_improve_warns_once():
    p = _fast(_variant(VARIANTS["all_three"], spf=10.0))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sol = ZoneHeuristicSolver().solve(p, CONFIG)[0]
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        ZoneIISolver().improve(p, sol)
    assert len(_zone_target_warnings(rec)) == 1


@pytest.mark.parametrize("runmode", sorted(STAGES))
def test_pipeline_warns_once_per_solver_call(runmode):
    p = _fast(_variant(VARIANTS["all_three"], spf=10.0))
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        sols = RunModePipeline(runmode=runmode).solve(p, SolverConfig(num_solutions=2, seed=1))
    assert len(sols) == 2
    assert len(_zone_target_warnings(rec)) == 2 * STAGES[runmode]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `/opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/zones/test_zone_target_agreement.py -q`
Expected: failures —
- `test_mip_matches_hand_normalized[*]`: the three `dup_*` variants raise `PulpError: overlapping constraint names: shortfall_…`; `unknown_zone`, `unknown_feature` and `all_three` return `[]` (phantom `0 >= target` constraint) instead of the hand-normalized solution;
- `test_mip_duplicate_rows_no_longer_raise`, `test_mip_last_wins_is_discriminating` and `test_each_solver_warns_exactly_once[ZoneMIPSolver]`: `PulpError`;
- `test_each_solver_warns_exactly_once[ZoneSASolver|ZoneHeuristicSolver|ZoneIISolver]` and `test_ii_improve_warns_once`: `assert 0 == 1`;
- `test_pipeline_warns_once_per_solver_call[*]`: `assert 0 == 2` / `0 == 4` / `0 == 6`;
- `test_all_rows_inert_mip_matches_none`: the unknown zone-9 row's phantom constraint makes the MIP return `[]`.
Record the exact count in the commit message.

- [ ] **Step 3: Implement**

`mip_solver.py` — change the model import to `from pymarxan.zones.model import ZonalProblem, warn_zone_target_issues`. In `solve`, directly after `raise_if_separation_active(problem, "ZoneMIPSolver")`:

```python
        warn_zone_target_issues(problem, "ZoneMIPSolver")
```

`_zone_achieved_exprs` — docstring and body after the signature become:

```python
    """Σ_i amount[i, f] × w[z, f] × x[i, z] for every effective (zone, feature) target.

    Targets come from ``zone_target_lookup``. Built once and shared by the penalty
    (slack) and the hard constraint. ``w`` is ``problem.zone_target_weight_matrix()``
    (raw by default, MarZone ``reserve.hpp:164``).
    """
    lookup = problem.zone_target_lookup()
    if not lookup:
        return {}
    weight = problem.zone_target_weight_matrix()
    zidx = problem.zone_index()
    fidx = problem.feature_index()
    groups = _feature_groups(problem)
    exprs: dict[tuple[int, int], pulp.LpAffineExpression] = {}
    for zid, fid in lookup:
        w = float(weight[zidx[zid], fidx[fid]])
        exprs[(zid, fid)] = pulp.lpSum(
            amt * w * x[(pid, zid)]
            for pid, amt in groups.get(fid, [])
            if (pid, zid) in x
        )
    return exprs
```

`_build_penalty_expr` — body after the docstring becomes:

```python
    lookup = problem.zone_target_lookup()
    if not lookup:
        return pulp.lpSum([]), {}
    spf_lookup = _spf_lookup(problem)
    misslevel = float(problem.parameters.get("MISSLEVEL", 1.0))
    expr = pulp.lpSum([])
    slack_vars = {}
    for (zid, fid), target in lookup.items():
        slack = pulp.LpVariable(f"slack_{zid}_{fid}", lowBound=0, cat="Continuous")
        model += (
            slack >= target * misslevel - exprs[(zid, fid)],
            f"shortfall_{zid}_{fid}",
        )
        slack_vars[(zid, fid)] = slack
        expr += spf_lookup.get(fid, 1.0) * slack
    return expr, slack_vars
```

`_add_zone_target_constraints` — body after the docstring becomes:

```python
    misslevel = float(problem.parameters.get("MISSLEVEL", 1.0))
    for (zid, fid), target in problem.zone_target_lookup().items():
        model += (exprs[(zid, fid)] >= target * misslevel, f"zone_target_{zid}_{fid}")
```

`solver.py`, `heuristic.py`, `iterative_improvement.py` — extend each file's `from pymarxan.zones.model import ZonalProblem` to `from pymarxan.zones.model import ZonalProblem, warn_zone_target_issues`, and directly after each `raise_if_separation_active(problem, "<Name>")` line add:

```python
        warn_zone_target_issues(problem, "<Name>")
```

with `<Name>` = `ZoneSASolver`, `ZoneHeuristicSolver`, `ZoneIISolver` respectively. In `ZoneIISolver.improve`, make the first statement after the docstring:

```python
        warn_zone_target_issues(problem, "ZoneIISolver")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `/opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/zones/ tests/pymarxan/solvers/ -q`
Expected: all PASS; `test_zone_target_agreement.py` completes in well under 5 s.

- [ ] **Step 5: Lint, type-check, commit**

```bash
~/.local/bin/ruff check src/pymarxan/zones tests/pymarxan/zones
/home/razinka/marxan/.venv/bin/mypy src/pymarxan/zones --ignore-missing-imports
git add src/pymarxan/zones/mip_solver.py src/pymarxan/zones/solver.py src/pymarxan/zones/heuristic.py src/pymarxan/zones/iterative_improvement.py tests/pymarxan/zones/test_zone_target_agreement.py
git commit -m "fix(zones): MIP reads the zone-target lookup (duplicates no longer raise); one ZoneTargetWarning per zone-solver call (#7)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: `zones/diagnostics.py` — shared PU zone rules, the diagnostic, the MIP failure warning

**Files:**
- Create: `src/pymarxan/zones/diagnostics.py`
- Modify: `src/pymarxan/zones/mip_solver.py` (anchors by content: the `def _feature_groups` function; the `pu_status = ...` line; the `for k, pid in enumerate(pu_ids):` block through the `locked_out_` constraint; the `return []` under `if model.status in infeasible_statuses or not has_values:`)
- Modify: `src/pymarxan/zones/__init__.py`, `tests/pymarxan/zones/test_package_exports.py`
- Test: `tests/pymarxan/zones/test_zone_diagnostics.py`

**Interfaces:**
- Consumes: `zone_target_lookup()` (Task 1); `objective._meets(achieved, required) -> bool`.
- Produces (all in `pymarxan.zones.diagnostics`):
  - `_feature_groups(problem) -> dict[int, list[tuple[int, float]]]` (moved verbatim from `mip_solver`, which imports it).
  - `class PUZoneRule(NamedTuple): allowed: frozenset[int]; forced: bool`.
  - `_pu_zone_rules(problem) -> list[PUZoneRule]` (planning_units order).
  - `diagnose_zone_infeasibility(problem: ZonalProblem) -> list[str]` — exported from `pymarxan.zones`.
  - `class ZoneMIPNoSolutionWarning(UserWarning)` — exported from `pymarxan.zones`.

- [ ] **Step 1: Write the failing tests**

Create `tests/pymarxan/zones/test_zone_diagnostics.py`:

```python
"""diagnose_zone_infeasibility + ZoneMIPSolver failure warnings (#7, spec §3)."""
from __future__ import annotations

import warnings

import pandas as pd
import pytest

from pymarxan.solvers.base import SolverConfig
from pymarxan.zones import ZoneMIPNoSolutionWarning, ZoneMIPSolver, diagnose_zone_infeasibility
from pymarxan.zones.diagnostics import PUZoneRule, _pu_zone_rules
from tests.pymarxan.zones.marzone_anchor import make_anchor_problem

CONFIG = SolverConfig(num_solutions=1, seed=1)


def _status(p, statuses):
    return p.copy_with(planning_units=p.planning_units.assign(status=statuses))


def _contrib(p, rows):
    return p.copy_with(
        zone_contributions=pd.DataFrame(rows, columns=["feature", "zone", "contribution"]),
    )


def _mip_warning(p) -> str:
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        assert ZoneMIPSolver().solve(p, CONFIG) == []
    msgs = [str(w.message) for w in rec if issubclass(w.category, ZoneMIPNoSolutionWarning)]
    assert len(msgs) == 1, msgs
    return msgs[0]


class TestRules:
    def test_status_convention(self):
        p = _status(make_anchor_problem(), [0, 2, 3])
        assert _pu_zone_rules(p) == [
            PUZoneRule(frozenset({1, 2}), False),
            PUZoneRule(frozenset({1}), True),
            PUZoneRule(frozenset(), False),
        ]

    def test_missing_status_column_means_free(self):
        p = make_anchor_problem()
        p = p.copy_with(planning_units=p.planning_units.drop(columns="status"))
        assert all(r == PUZoneRule(frozenset({1, 2}), False) for r in _pu_zone_rules(p))


class TestDiagnose:
    def test_anchor_reachable(self):
        assert diagnose_zone_infeasibility(make_anchor_problem()) == []

    def test_two_lockouts_make_overall_unreachable(self):
        msgs = diagnose_zone_infeasibility(_status(make_anchor_problem(), [3, 3, 0]))
        assert msgs == ["overall target for feature 1 is 15 but at most 10 is reachable"]

    def test_partial_contribution_table(self):
        p = _contrib(make_anchor_problem(), [(1, 2, 0.4)])  # zone 1 now contributes 0
        assert diagnose_zone_infeasibility(p) == [
            "overall target for feature 1 is 15 but at most 12 is reachable"
        ]

    def test_locked_in_forced_negative_counts(self):
        p = _status(_contrib(make_anchor_problem(), [(1, 1, -0.5), (1, 2, 0.4)]), [2, 0, 0])
        # PU1 forced into zone 1: -5; PUs 2-3 best = max(0, -5, 4) = 4 each -> 3
        assert diagnose_zone_infeasibility(p) == [
            "overall target for feature 1 is 15 but at most 3 is reachable"
        ]

    def test_free_pu_may_stay_unassigned_when_every_zone_is_negative(self):
        p = _contrib(make_anchor_problem(), [(1, 1, -0.5), (1, 2, -0.1)])
        assert diagnose_zone_infeasibility(p) == [
            "overall target for feature 1 is 15 but at most 0 is reachable"
        ]

    def test_zone_target_with_contrib_weight_and_forced_negative(self):
        p = make_anchor_problem(zone_target_contrib=1)
        p = _contrib(p, [(1, 1, -0.5), (1, 2, 0.4)])
        p = _status(p, [2, 0, 0])
        p = p.copy_with(
            zone_targets=pd.DataFrame({"zone": [1], "feature": [1], "target": [1.0]}),
            features=p.features.assign(target=0.0),
        )
        # zone 1 weight -0.5: PU1 forced -5; free PUs max(0, -5) = 0 -> -5
        assert diagnose_zone_infeasibility(p) == [
            "zone target (zone 1, feature 1) is 1 but at most -5 is reachable"
        ]

    def test_zero_targets_never_reported(self):
        p = _status(_contrib(make_anchor_problem(), [(1, 1, -0.5), (1, 2, 0.4)]), [2, 0, 0])
        p = p.copy_with(features=p.features.assign(target=0.0), zone_targets=None)
        assert diagnose_zone_infeasibility(p) == []

    def test_misslevel_scales_the_overall_target(self):
        p = _status(make_anchor_problem(misslevel=0.5), [3, 3, 0])
        assert diagnose_zone_infeasibility(p) == []  # 7.5 <= 10

    def test_misslevel_scales_zone_targets(self):
        p = _status(make_anchor_problem(misslevel=0.5), [3, 0, 0]).copy_with(
            zone_targets=pd.DataFrame({"zone": [2], "feature": [1], "target": [30.0]}),
        )
        assert diagnose_zone_infeasibility(p) == []  # 15 <= 20

    def test_duplicate_puvspr_rows_summed(self):
        p = _status(make_anchor_problem(), [3, 3, 0])
        puv = pd.DataFrame({"species": [1, 1, 1, 1], "pu": [1, 2, 3, 3],
                            "amount": [10.0, 10.0, 5.0, 5.0]})
        assert diagnose_zone_infeasibility(p.copy_with(pu_vs_features=puv)) == [
            "overall target for feature 1 is 15 but at most 10 is reachable"
        ]


class TestMIPWarning:
    def test_names_unreachable_target(self):
        msg = _mip_warning(_status(make_anchor_problem(), [3, 3, 0]))
        assert msg.startswith("ZoneMIPSolver: infeasible. Unreachable targets: ")
        assert "feature 1 is 15 but at most 10" in msg

    def test_compete_wording_when_each_target_reachable(self):
        p = _status(make_anchor_problem(), [3, 0, 0])
        assert diagnose_zone_infeasibility(p) == []
        msg = _mip_warning(p)
        assert "every target is reachable on its own" in msg
        assert "ZoneSASolver" in msg

    def test_ghost_feature_named_in_mip_warning(self):
        p = make_anchor_problem()
        ghost = pd.DataFrame({"id": [2], "name": ["ghost"], "target": [5.0], "spf": [1.0]})
        p = p.copy_with(features=pd.concat([p.features, ghost], ignore_index=True))
        msg = _mip_warning(p)
        assert "overall target for feature 2 is 5 but at most 0 is reachable" in msg

    def test_diagnostic_crash_still_returns_empty(self, monkeypatch):
        import pymarxan.zones.mip_solver as ms

        def boom(problem):
            raise RuntimeError("boom")

        monkeypatch.setattr(ms, "diagnose_zone_infeasibility", boom)
        msg = _mip_warning(_status(make_anchor_problem(), [3, 3, 0]))
        assert "diagnostic failed" in msg

    def test_feasible_solve_emits_no_failure_warning(self):
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            sols = ZoneMIPSolver().solve(make_anchor_problem(), CONFIG)
        assert sols and sols[0].objective == pytest.approx(7.0)
        assert not [w for w in rec if issubclass(w.category, ZoneMIPNoSolutionWarning)]
```

Add to `ZONES_EXPORTS` in `tests/pymarxan/zones/test_package_exports.py` (insert after the `"ZoneMIPSolver"` entry and after the `"compute_zone_shortfalls"` entry respectively):

```python
    "ZoneMIPNoSolutionWarning": "pymarxan.zones.diagnostics",
    "diagnose_zone_infeasibility": "pymarxan.zones.diagnostics",
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `/opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/zones/test_zone_diagnostics.py -q`
Expected: collection ERROR — `ImportError: cannot import name 'ZoneMIPNoSolutionWarning' from 'pymarxan.zones'`. (ruff `I001` on this test file is expected until Step 3 creates `diagnostics.py`.)

- [ ] **Step 3: Create `src/pymarxan/zones/diagnostics.py`**

```python
"""Single-target infeasibility diagnostics for ZoneMIPSolver, and the rules it shares with it.

``_feature_groups`` (the MIP constraints' amount source) and ``_pu_zone_rules`` (which zones
a PU may take) live here so the MIP and the diagnostic read one definition each; #6 part 2
(lock into a named zone) extends ``_pu_zone_rules`` only.
"""
from __future__ import annotations

from typing import NamedTuple

from pymarxan.models.problem import STATUS_LOCKED_IN, STATUS_LOCKED_OUT
from pymarxan.zones.model import ZonalProblem
from pymarxan.zones.objective import _meets


class ZoneMIPNoSolutionWarning(UserWarning):
    """ZoneMIPSolver returned ``[]``; the message says why (diagnostic, compete, time limit)."""


class PUZoneRule(NamedTuple):
    """Zones a PU may take; ``forced`` means it must take the single allowed zone."""

    allowed: frozenset[int]
    forced: bool


def _feature_groups(problem: ZonalProblem) -> dict[int, list[tuple[int, float]]]:
    """feature id -> [(pu id, amount), ...] from pu_vs_features."""
    puvspr = problem.pu_vs_features
    groups: dict[int, list[tuple[int, float]]] = {}
    for pid, fid, amt in zip(
        puvspr["pu"].values, puvspr["species"].values, puvspr["amount"].values, strict=True,
    ):
        groups.setdefault(int(fid), []).append((int(pid), float(amt)))
    return groups


def _pu_zone_rules(problem: ZonalProblem) -> list[PUZoneRule]:
    """Per PU in ``planning_units`` order: status 3 -> no zone; status 2 -> forced into
    ``min(zone_ids)``; otherwise free over every zone. No ``status`` column means free.

    This is pymarxan's status convention (ZoneMIPSolver / SA / heuristic / II), not a
    MarZone rule — MarZone locks PUs into named zones via its pulock file (#6 part 2).
    """
    every = frozenset(int(z) for z in problem.zone_ids)
    first = frozenset({min(every)}) if every else frozenset()
    pu = problem.planning_units
    if "status" not in pu.columns:
        return [PUZoneRule(every, False)] * len(pu)
    rules: list[PUZoneRule] = []
    for s in pu["status"].values.astype(int):
        if s == STATUS_LOCKED_OUT:
            rules.append(PUZoneRule(frozenset(), False))
        elif s == STATUS_LOCKED_IN:
            rules.append(PUZoneRule(first, True))
        else:
            rules.append(PUZoneRule(every, False))
    return rules


def _best(rule: PUZoneRule, value_in: dict[int, float]) -> float:
    """A PU's largest possible contribution: 0 if it can take no zone; its forced zone's
    value as-is (possibly negative); else max(0, best allowed) — it may stay unassigned."""
    if not rule.allowed:
        return 0.0
    if rule.forced:
        (zid,) = rule.allowed
        return value_in.get(zid, 0.0)
    return max(0.0, max((value_in.get(z, 0.0) for z in rule.allowed), default=0.0))


def _fmt(x: float) -> str:
    return f"{x:.6g}"


def diagnose_zone_infeasibility(problem: ZonalProblem) -> list[str]:
    """Targets no assignment can meet, each checked on its own (necessary conditions only).

    For PU ``i`` with per-zone value ``v_i(z)``, its best contribution is ``_best``;
    overall target ``f`` uses ``v_i(z) = amount[i, f] × contribution[z, f]`` and zone
    target ``(k, f)`` uses ``v_i(k) = amount[i, f] × w[k, f]`` (0 in other zones). A
    target ``t > 0`` is reported when ``Σ_i best_i < t × MISSLEVEL`` (1e-9 relative
    tolerance). Amounts come from ``_feature_groups`` — the MIP constraints' own source
    — so the bound is the MIP's. Zone targets are hard only in pymarxan's MIP (MarZone
    treats them as penalties). ``[]`` does not prove feasibility: targets may compete.
    """
    rule_of = dict(
        zip(
            (int(p) for p in problem.planning_units["id"].values),
            _pu_zone_rules(problem),
            strict=True,
        )
    )
    misslevel = float(problem.parameters.get("MISSLEVEL", 1.0))
    contrib = problem.contribution_matrix()
    weight = problem.zone_target_weight_matrix()
    zidx = problem.zone_index()
    fidx = problem.feature_index()
    groups = _feature_groups(problem)

    def per_pu(fid: int) -> dict[int, float]:
        summed: dict[int, float] = {}
        for pid, amt in groups.get(fid, []):
            if pid in rule_of:
                summed[pid] = summed.get(pid, 0.0) + amt
        return summed

    messages: list[str] = []
    for fid_raw, target in zip(
        problem.features["id"].values, problem.features["target"].values, strict=True,
    ):
        fid, t = int(fid_raw), float(target)
        if not t > 0:
            continue
        col = fidx[fid]
        ub = 0.0
        for pid, amt in per_pu(fid).items():
            rule = rule_of[pid]
            ub += _best(rule, {z: amt * float(contrib[zidx[z], col]) for z in rule.allowed})
        if not _meets(ub, t * misslevel):
            messages.append(
                f"overall target for feature {fid} is {_fmt(t * misslevel)} "
                f"but at most {_fmt(ub)} is reachable"
            )

    for (zid, fid), t in problem.zone_target_lookup().items():
        w = float(weight[zidx[zid], fidx[fid]])
        ub = 0.0
        for pid, amt in per_pu(fid).items():
            rule = rule_of[pid]
            ub += _best(rule, {zid: amt * w} if zid in rule.allowed else {})
        if not _meets(ub, t * misslevel):
            messages.append(
                f"zone target (zone {zid}, feature {fid}) is {_fmt(t * misslevel)} "
                f"but at most {_fmt(ub)} is reachable"
            )
    return messages
```

For a zone target, a free PU's dict holds only the target zone, so its best is `max(0, amt × w)`; a PU forced into a different zone gets `value_in.get(forced_zone, 0.0) = 0`.

- [ ] **Step 4: Wire the MIP to it**

In `mip_solver.py`:

1. Delete the `def _feature_groups(...)` function and add to the imports (one name per line; ruff sorts):
   `from pymarxan.zones.diagnostics import (ZoneMIPNoSolutionWarning, _feature_groups, _pu_zone_rules, diagnose_zone_infeasibility)`. Add `import warnings`. Then run `grep -rn "_feature_groups" src tests` and repoint any other importer to `pymarxan.zones.diagnostics`.
2. Delete the `pu_status = problem.planning_units["status"].values.astype(int)` line; add `rules = _pu_zone_rules(problem)` in its place. Replace the `for k, pid in enumerate(pu_ids):` block (through the `locked_out_` constraint) with:

```python
        for k, pid in enumerate(pu_ids):
            pid = int(pid)
            for zid in zone_ids:
                x[(pid, zid)] = pulp.LpVariable(f"x_{pid}_{zid}", cat="Binary")

            # Each PU in at most one zone
            model += (
                pulp.lpSum(x[(pid, zid)] for zid in zone_ids) <= 1,
                f"assign_{pid}",
            )

            rule = rules[k]
            if rule.forced:
                (forced_zone,) = rule.allowed
                model += x[(pid, forced_zone)] == 1, f"locked_in_{pid}"
            else:
                for zid in zone_ids:
                    if zid not in rule.allowed:
                        model += x[(pid, zid)] == 0, f"locked_out_{pid}_{zid}"
```

   Remove the now-unused `STATUS_LOCKED_IN` / `STATUS_LOCKED_OUT` import (ruff F401 flags it).
3. Replace the `return []` under `if model.status in infeasible_statuses or not has_values:` with:

```python
            _warn_no_solution(problem, model.status)
            return []
```

   and add this module-level function after the class:

```python
def _warn_no_solution(problem: ZonalProblem, status: int) -> None:
    """One ZoneMIPNoSolutionWarning explaining an empty result; the return stays []."""
    name = pulp.LpStatus.get(status, str(status))
    if status == pulp.constants.LpStatusInfeasible:
        try:
            unreachable = diagnose_zone_infeasibility(problem)
            failed = ""
        except Exception as exc:  # a diagnostic must never turn [] into a crash
            unreachable, failed = [], f" (diagnostic failed: {exc!r})"
        if unreachable:
            more = f" (+{len(unreachable) - 5} more)" if len(unreachable) > 5 else ""
            msg = (
                "ZoneMIPSolver: infeasible. Unreachable targets: "
                + "; ".join(unreachable[:5]) + more
            )
        else:
            msg = (
                "ZoneMIPSolver: infeasible, but every target is reachable on its own — "
                "targets compete for the same planning units. ZoneSASolver treats "
                f"targets as penalties and will return a best-effort plan.{failed}"
            )
    elif status == pulp.constants.LpStatusNotSolved:
        msg = (
            "ZoneMIPSolver: no feasible solution found within the time limit (status "
            f"{name}); raise MIP_TIME_LIMIT or use ZoneSASolver."
        )
    else:
        msg = f"ZoneMIPSolver: no solution (status {name})."
    warnings.warn(msg, ZoneMIPNoSolutionWarning, stacklevel=3)
```

   In the class docstring's constraint list add: "Infeasible / no incumbent → ``[]`` plus one ``ZoneMIPNoSolutionWarning`` (``diagnose_zone_infeasibility``)".

`__init__.py` — add `from pymarxan.zones.diagnostics import ZoneMIPNoSolutionWarning, diagnose_zone_infeasibility`; add `"ZoneMIPNoSolutionWarning",` to `__all__` directly after `"ZoneMIPSolver",` and `"diagnose_zone_infeasibility",` directly after `"compute_zone_shortfalls",`.

- [ ] **Step 5: Run tests to verify they pass**

Run: `/opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/zones/ -q`
Expected: all PASS — including the pre-existing `test_zone_mip_overall.py` `== []` cases (they now also emit a `ZoneMIPNoSolutionWarning`, which is not an error) and the locked-in/out MIP tests (identical constraints, built from the shared rule).

If CBC reports a status other than `LpStatusInfeasible` for an infeasible fixture, print `model.status` in a scratch run and extend `_warn_no_solution`'s first branch to accept it — do not loosen the test.

- [ ] **Step 6: Lint, type-check, commit**

```bash
~/.local/bin/ruff check src/pymarxan/zones tests/pymarxan/zones
/home/razinka/marxan/.venv/bin/mypy src/pymarxan/zones --ignore-missing-imports
git add src/pymarxan/zones/diagnostics.py src/pymarxan/zones/mip_solver.py src/pymarxan/zones/__init__.py tests/pymarxan/zones/test_zone_diagnostics.py tests/pymarxan/zones/test_package_exports.py
git commit -m "feat(zones): diagnose_zone_infeasibility and ZoneMIPNoSolutionWarning; MIP and diagnostic share the PU zone rules (#7)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: Comparative bench, docs, full gate

**Files:**
- Create: `tests/benchmarks/bench_zone_target_lookup.py`
- Modify: `CHANGELOG.md` (`## [Unreleased]`), `docs/VALIDATION.md` (end of §4, before `## Comparing against the Marxan C++ binary`)

**Interfaces:** consumes everything above; produces nothing new.

- [ ] **Step 1: Write the bench**

This replaces spec Testing #9's "after vs before ≤ 1.05×" (no in-tree baseline exists) with a lookup/objective ratio; per-flip non-regression is Step 2.

```python
"""Cost of zone_target_lookup relative to one compute_zone_objective call (#7).

Comparative, not absolute: the lookup runs inside every objective call (heuristic / II
evaluate one per candidate move), so it must stay a small fraction of that call. A pandas
filter + drop_duplicates normalizer measured ~95 % of the call (design review #2); the dict
lookup measured ~4 % in the plan dry run.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import pytest

from pymarxan.zones.objective import compute_zone_objective
from tests.benchmarks.conftest import make_zone_problem

pytestmark = pytest.mark.bench


def _best_of(fn, repeats: int = 5, inner: int = 50) -> float:
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        for _ in range(inner):
            fn()
        best = min(best, time.perf_counter() - t0)
    return best / inner


def test_lookup_is_a_small_fraction_of_the_objective() -> None:
    p = make_zone_problem(n_pu=200, n_feat=20, n_zones=3, density=0.3, seed=7)
    zt = p.zone_targets
    assert zt is not None and len(zt) > 0
    # Duplicates and an unknown id so the lookup does its full work.
    extra = pd.concat(
        [zt, zt.head(5), pd.DataFrame({"zone": [99], "feature": [1], "target": [1.0]})],
        ignore_index=True,
    )
    p = p.copy_with(zone_targets=extra)
    a = np.random.default_rng(0).integers(0, 4, size=200)
    lookup = _best_of(p.zone_target_lookup)
    objective = _best_of(lambda: compute_zone_objective(p, a, 1.0))
    assert lookup <= 0.15 * objective, (
        f"lookup {lookup * 1e6:.1f} µs vs objective {objective * 1e6:.1f} µs"
    )
```

Run: `/opt/micromamba/envs/shiny/bin/pytest tests/benchmarks/bench_zone_target_lookup.py -m bench -q`
Expected: PASS (dry run measured 0.04).

- [ ] **Step 2: Existing zone benches (per-flip non-regression)**

Run: `/opt/micromamba/envs/shiny/bin/pytest tests/benchmarks/bench_zone_sa.py tests/benchmarks/bench_zone_overall.py -m bench -q`
Expected: PASS (the mask is gated off for these fixtures).

- [ ] **Step 3: CHANGELOG** — under `## [Unreleased]` add:

```markdown
### Fixed
- Zone problems: `zone_targets` rows with unknown zone/feature ids or a repeated (zone, feature)
  pair are now handled identically by `ZoneMIPSolver`, `compute_zone_objective`, the zone SA
  cache and `write_zone_summary`, following MarZone (`zones.hpp` BuildZoneTarget): unknown ids
  are skipped, the last duplicate row wins, and a pair whose surviving target is <= 0 adds no
  penalty and no MIP constraint (#7).
- Zone SA cache: under `ZONETARGETCONTRIB = 1` with a negative contribution, untargeted
  (zone, feature) pairs no longer accrue a phantom zone-target penalty (#7).

### Added
- `ZonalProblem.zone_target_lookup()` / `zone_target_issues()`; `validate()` reports unknown
  or non-integral ids and duplicate pairs in `zone_targets`, and duplicate rows in
  `zone_contributions`.
- `ZoneTargetWarning`, emitted once per zone-solver call (`solve`, `ZoneIISolver.improve`)
  when such rows exist; `RunModePipeline` RUNMODEs 2–5 call two or three zone solvers per run.
- `pymarxan.zones.diagnose_zone_infeasibility` and `ZoneMIPNoSolutionWarning`: when
  `ZoneMIPSolver` returns `[]` it now warns with the unreachable targets, the "targets
  compete" explanation, or the time-limit hint. Filter the category in loops that expect
  infeasibility.
- `pymarxan.zones.compute_zone_achieved` (public name for `_compute_zone_achieved`, kept as
  an alias).

### Changed (breaking for hand-built zone problems)
- Duplicate zone-target rows no longer raise `PulpError` in `ZoneMIPSolver`; unknown-id rows
  no longer make it return `[]` or add a phantom penalty to `compute_zone_objective`.
- `check_zone_targets`, `metadata["zone_targets_met"]` and `write_zone_summary` cover the known
  pairs only (unknown ids absent; duplicates collapsed last-wins); inert pairs (target <= 0)
  are reported as met. `compute_zone_shortfalls` covers effective pairs (target > 0) only.
  Callers that index these results by raw `zone_targets` rows should iterate
  `zone_target_lookup(include_inert=True)` instead.
```

- [ ] **Step 4: VALIDATION.md** — append to §4, directly before `## Comparing against the Marxan C++ binary`:

```markdown
**Zone-target input semantics (since v0.37).** `zone_targets` rows are read the way MarZone's
`BuildZoneTarget` reads `zonetarget.dat`: a row naming an unknown feature or zone is skipped
(`zones.hpp:131`, `:142`), and a repeated (zone, feature) pair is resolved by plain overwrite,
so the last row wins (`zones.hpp:148`). A pair whose surviving target is not positive is
inert — MarZone applies zone targets only when `target > 0` (`reserve.hpp:241`, `:472`,
`:846`). The MIP, the DataFrame objective, the SA cache and the summary writer all read one
accessor (`ZonalProblem.zone_target_lookup`); `tests/pymarxan/zones/test_zone_target_agreement.py`
pins their agreement. `validate()` reports such rows, and each zone solver warns once
(`ZoneTargetWarning`).
```

- [ ] **Step 5: Full gate**

```bash
PATH="/opt/micromamba/envs/shiny/bin:$HOME/.local/bin:/home/razinka/marxan/.venv/bin:$PATH" make check
PYTHONPATH=src /opt/micromamba/envs/shiny/bin/python examples/validate_marxan_parity.py
```

Expected: lint + types clean; full suite green with coverage ≥ 75 %; the parity harness reports MIP 35.0 on `tests/data/simple` and round-trip True.

- [ ] **Step 6: Commit**

```bash
git add tests/benchmarks/bench_zone_target_lookup.py CHANGELOG.md docs/VALIDATION.md
git commit -m "test(zones): comparative lookup bench; docs: CHANGELOG + VALIDATION for zone-target semantics (#7)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

## Follow-ons (not in this plan)

- **ROWER_pymarxan_DST** on re-pin to v0.37: `_target_rows` keeps working (met-flags cover every
  known pair); optionally replace `attainable_by_target` with `diagnose_zone_infeasibility` /
  `_pu_zone_rules` and surface the diagnostic in the "infeasible" `Evaluation.message`.
- Loader silently ignores `ZONETARGET2NAME` / `ZONECONTRIB2NAME` / `ZONECONTRIB3NAME` — separate
  issue (design review #13).
- Elastic re-solve for exact shortfalls on the "compete" path (design review #14).
