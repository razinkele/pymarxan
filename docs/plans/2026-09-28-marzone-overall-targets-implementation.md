# MarZone Overall Targets Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make every zone solver enforce MarZone's overall (contribution-weighted) feature
targets, accumulate zone targets on raw amounts, and default unlisted contribution pairs the
way MarZone does — without moving the single-zone parity anchor.

**Architecture:** One contribution source on `ZonalProblem` feeds the DataFrame objective
module, the MIP builders, the `ZoneProblemCache` fast path and the writers. Two accumulators
(`ZoneHeld.per_zone` raw, `ZoneHeld.overall` contribution-weighted) mirror MarZone's
`zoneSpec` / `speciesAmounts`; a single `build_zone_solution` replaces the five hand-rolled
`Solution` builders. Semantic flips land in paired tasks (DataFrame path + cache + MIP together)
so the existing cache-equals-objective tests stay green after every task.

**Tech Stack:** Python 3.12, NumPy, pandas, PuLP (CBC); pytest under the `shiny` micromamba
env; ruff; mypy.

**Spec:** `docs/plans/2026-09-28-marzone-overall-targets-design.md` (revision 2, approved).
Review synthesis: `docs/plans/2026-09-28-marzone-overall-targets-review.md`.

## Global Constraints

- Run every test with `/opt/micromamba/envs/shiny/bin/pytest` (never `.venv`; see CLAUDE.md).
  Final gate before the branch is finished: `make check` with the shiny env first on `PATH`.
- `from __future__ import annotations` in every file; full type hints; ruff rules E, F, I, UP,
  line length 99; `make types` (mypy on `pymarxan` + `pymarxan_shiny`) must stay clean.
- Run `ruff check` on the **test** files of every task before committing (an E402 slipped two
  task reviews in earlier phases).
- Parity anchor: `tests/data/simple` exact optimum reserve `{2, 4, 6}` at cost **35.0**; SA 43,
  greedy 45 (`tests/test_examples.py`). Must not move.
- Do **not** bump the version in `pyproject.toml`; `scripts/release.sh 0.36.0` does that.
  This plan fills `## [Unreleased]` in `CHANGELOG.md` only.
- Pure NumPy, dense arrays in the cache; preserve per-flip delta computation (no full
  recomputation inside SA loops). No Numba/Cython.
- Every new pytest file basename must be repo-unique.
- Contribution semantics (verbatim from spec §3.1): default for an unlisted (feature, zone)
  pair is **1.0 when `zone_contributions` is `None`; 0.0 when a table is supplied**.
- Overall achieved `A_f = Σ_i amount[i, f] × contribution[f, z_i]` over `z_i > 0`; met when
  `A_f ≥ target_f × MISSLEVEL`; penalty `spf_f × max(0, target_f × MISSLEVEL − A_f)`.
- Zone achieved `Z_kf = Σ_{i: z_i = k} amount[i, f] × w[k, f]`, `w` all ones by default and
  `= contribution_matrix()` when `ZONETARGETCONTRIB == 1`; values outside {0, 1} rejected by
  `validate()` and by the cache.
- Objective: `zone_cost + BLM × standard_boundary + zone_boundary + overall_penalty +
  zone_penalty + connectivity`.
- **Index convention (all modules):** matrix row for zone id `z` = its rank in
  `sorted(zone_ids)` + 1 (row 0 = unassigned); column for feature id `f` = its position in
  `features["id"]`. Exposed as `ZonalProblem.zone_index()` / `feature_index()`.
- **Cache API convention:** public cache methods (`compute_held`, `update_held`,
  `compute_delta_zone_objective`) take zone **ids**; private `_penalty_delta` /
  `_overall_penalty_delta` take matrix **columns**.
- Every commit ends with the two trailer lines:
  `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>` and
  `Claude-Session: https://claude.ai/code/session_01X8aGAi384D5xybKRugtFXT`.

## Rulings recorded while writing this plan (spec gaps)

- **R1 — no `save_zone_project`.** `io.writers.save_project` writes only the base files, so
  the spec's `load_zone_project → save_project → load_zone_project` round trip cannot run as
  written. Task 11 tests the parameter round trip through `write_input_dat`/`read_input_dat`
  and a full-directory round trip using the existing zone writers in a test helper. A public
  `save_zone_project` is a follow-on (noted in CHANGELOG "Unreleased" as not included).
- **R2 — `targettype = 1` resolves in `load_zone_project`,** not in `read_zone_targets`,
  because it needs `pu_vs_features`. After resolution the row's `targettype` is set to 0 so a
  write → read cycle does not multiply twice (mirrors the idempotence of
  `io.readers._resolve_prop_targets`).
- **R3 — unlisted contribution pairs produce one summary message** in `validate()` (count
  listed / total, count unlisted, first unlisted pair, the 0.0 default), not one line per pair.
- **R4 — `Solution.targets_met` keeps its union type annotation;** only the comment changes.
  Zone solvers now populate int keys; narrowing the annotation is a follow-on.
- **R5 — `write_zone_summary` emits zone rows only for listed zone targets** (the spec's "zone
  block (zone, feature, zone_target, …)"), not for every zone × feature pair as today.

## Review Focus

Inputs the spec implies but no spec test exercises; each is pinned in the owning task:

1. `ZONETARGETCONTRIB` given as `2`, `"1"` or `1.5` — `validate()` returns an error naming the
   value and `ZoneProblemCache.from_zone_problem` raises `ValueError` (Task 1, Task 5).
2. `zonetarget.dat` with `targettype = 1` written back out and reloaded — the target is not
   multiplied a second time (Task 11).
3. A feature with `target > 0` and no `puvspr` rows — the MIP returns `[]`, the heuristics
   return a penalised assignment, nothing crashes (Task 6, Task 7).
4. A `zone_contributions` row naming a zone or feature that does not exist — `validate()`
   reports it instead of silently dropping it (Task 1).
5. `features` without an `spf` column — the cache and the objective module use 1.0 (Task 5).

## File structure

| File | Responsibility after this plan |
|---|---|
| `src/pymarxan/zones/model.py` | `ZonalProblem` + index maps, contribution lookup/matrix, zone-target weight matrix, `validate()` additions |
| `src/pymarxan/zones/objective.py` | DataFrame reference implementation of both target tiers, `compute_zone_objective`, shared `build_zone_solution` |
| `src/pymarxan/zones/cache.py` | `ZoneHeld` + `ZoneProblemCache` fast path (full and delta objective) |
| `src/pymarxan/zones/mip_solver.py` | zone-target expressions built once; overall hard constraints; uses the shared builder |
| `src/pymarxan/zones/solver.py`, `heuristic.py`, `iterative_improvement.py` | carry `ZoneHeld` (SA); shared builder; heuristic early exit removed |
| `src/pymarxan/zones/writers.py`, `readers.py`, `__init__.py` | summary rewrite; `targettype`; re-exports |
| `src/pymarxan/objectives/min_shortfall.py`, `max_coverage.py` | zone achieved routed through `compute_overall_achieved` |
| `src/pymarxan/solvers/base.py` | `targets_met` comment |
| `tests/pymarxan/zones/marzone_anchor.py` | spec §5 anchor problem factory + oracle (not a test module) |
| `tests/pymarxan/zones/test_zone_overall_targets.py` | objective-module tests for both tiers, builder |
| `tests/pymarxan/zones/test_zone_mip_overall.py` | the three MIP anchor behaviours |
| `tests/pymarxan/zones/test_zone_solvers_overall.py` | SA / heuristic / II anchor runs, cross-solver agreement |
| `tests/pymarxan/zones/test_marzone_accumulation.py` | `reserve.hpp:148-182` reimplementation cross-check |
| `tests/benchmarks/bench_zone_overall.py` | comparative per-flip bench (gate) |

---

### Task 1: Contribution source of truth on `ZonalProblem`

**Files:**
- Modify: `src/pymarxan/zones/model.py`
- Test: `tests/pymarxan/zones/test_model.py`

**Interfaces:**
- Consumes: nothing new.
- Produces (used by every later task):
  - `ZonalProblem.zone_index() -> dict[int, int]` (zone id → matrix row, 1-based; row 0 unassigned)
  - `ZonalProblem.feature_index() -> dict[int, int]` (feature id → column)
  - `ZonalProblem.contribution_default() -> float` (1.0 without a table, 0.0 with one)
  - `ZonalProblem.contribution_lookup() -> dict[tuple[int, int], float]` keyed `(feature, zone)`, listed pairs only
  - `ZonalProblem.get_contribution(feature_id, zone_id) -> float` (lookup + default)
  - `ZonalProblem.contribution_matrix() -> np.ndarray` shape `(n_zones + 1, n_feat)`, row 0 zero
  - `ZonalProblem.zone_target_contrib() -> int` (0 or 1; `ValueError` otherwise)
  - `ZonalProblem.zone_target_weight_matrix() -> np.ndarray` same shape; ones (rows 1..n) or the contribution matrix
  - `ZonalProblem.validate()` new messages (parameter domain, `target2`, unknown pairs, unlisted pairs)

- [ ] **Step 1: Write the failing tests**

Append to `tests/pymarxan/zones/test_model.py` (the file already defines `_make_base_data()`
and `_make_zone_data()` returning `zones, zone_costs, zone_contributions, zone_targets,
zone_boundary_costs` with contributions listed for all four (feature, zone) pairs):

```python
import numpy as np


class TestContributionSource:
    def _problem(self, **kw):
        pu, feat, puvspr = _make_base_data()
        zones, zc, zcontrib, zt, zbc = _make_zone_data()
        base = dict(
            planning_units=pu, features=feat, pu_vs_features=puvspr,
            zones=zones, zone_costs=zc,
        )
        base.update(kw)
        return ZonalProblem(**base)

    def test_zone_index_is_rank_in_sorted_ids_plus_one(self):
        zp = self._problem()
        assert zp.zone_index() == {1: 1, 2: 2}

    def test_feature_index_follows_features_order(self):
        zp = self._problem()
        assert zp.feature_index() == {1: 0, 2: 1}

    def test_default_is_one_without_table(self):
        zp = self._problem()
        assert zp.contribution_default() == 1.0
        assert zp.get_contribution(1, 2) == 1.0
        assert zp.contribution_lookup() == {}

    def test_default_is_zero_for_unlisted_pair_with_table(self):
        partial = pd.DataFrame({
            "feature": [1, 2], "zone": [1, 1], "contribution": [1.0, 0.8],
        })
        zp = self._problem(zone_contributions=partial)
        assert zp.contribution_default() == 0.0
        assert zp.get_contribution(1, 1) == 1.0
        assert zp.get_contribution(1, 2) == 0.0  # unlisted: MarZone zones.hpp:619
        assert zp.contribution_lookup() == {(1, 1): 1.0, (2, 1): 0.8}

    def test_contribution_matrix_layout(self):
        _, _, zcontrib, _, _ = _make_zone_data()
        zp = self._problem(zone_contributions=zcontrib)
        m = zp.contribution_matrix()
        assert m.shape == (3, 2)
        np.testing.assert_array_equal(m[0], [0.0, 0.0])
        np.testing.assert_array_equal(m[1], [1.0, 1.0])   # zone 1: f1=1.0, f2=1.0
        np.testing.assert_array_equal(m[2], [0.5, 0.3])   # zone 2: f1=0.5, f2=0.3

    def test_contribution_matrix_without_table_is_ones(self):
        m = self._problem().contribution_matrix()
        np.testing.assert_array_equal(m[1:], np.ones((2, 2)))
        np.testing.assert_array_equal(m[0], np.zeros(2))

    def test_zone_target_weight_default_is_ones(self):
        _, _, zcontrib, _, _ = _make_zone_data()
        zp = self._problem(zone_contributions=zcontrib)
        assert zp.zone_target_contrib() == 0
        w = zp.zone_target_weight_matrix()
        np.testing.assert_array_equal(w[1:], np.ones((2, 2)))
        np.testing.assert_array_equal(w[0], np.zeros(2))

    def test_zone_target_weight_is_contribution_when_flag_set(self):
        _, _, zcontrib, _, _ = _make_zone_data()
        zp = self._problem(
            zone_contributions=zcontrib, parameters={"ZONETARGETCONTRIB": 1},
        )
        assert zp.zone_target_contrib() == 1
        np.testing.assert_array_equal(
            zp.zone_target_weight_matrix(), zp.contribution_matrix(),
        )

    @pytest.mark.parametrize("bad", [2, "1", 1.5, -1])
    def test_zone_target_contrib_rejects_out_of_domain(self, bad):
        zp = self._problem(parameters={"ZONETARGETCONTRIB": bad})
        with pytest.raises(ValueError, match="ZONETARGETCONTRIB"):
            zp.zone_target_contrib()
        errors = zp.validate()
        assert any("ZONETARGETCONTRIB" in e for e in errors)

    def test_validate_reports_target2_in_zone_problem(self):
        pu, feat, puvspr = _make_base_data()
        feat = feat.copy()
        feat["target2"] = [5.0, 0.0]
        zp = self._problem(features=feat)
        assert any("target2" in e for e in zp.validate())

    def test_validate_reports_unknown_contribution_pair(self):
        bad = pd.DataFrame({
            "feature": [1, 9], "zone": [7, 1], "contribution": [1.0, 1.0],
        })
        zp = self._problem(zone_contributions=bad)
        msgs = [e for e in zp.validate() if "unknown" in e]
        assert len(msgs) == 1
        assert "zone 7" in msgs[0] and "feature 9" in msgs[0]

    def test_validate_summarises_unlisted_pairs_once(self):
        partial = pd.DataFrame({
            "feature": [1], "zone": [1], "contribution": [1.0],
        })
        zp = self._problem(zone_contributions=partial)
        msgs = [e for e in zp.validate() if "unlisted" in e]
        assert len(msgs) == 1
        assert "1 of 4" in msgs[0]
        assert "3 unlisted" in msgs[0]
        assert "default to 0.0" in msgs[0]

    def test_validate_clean_when_all_pairs_listed(self):
        zones, zc, zcontrib, zt, zbc = _make_zone_data()
        zp = self._problem(
            zone_contributions=zcontrib, zone_targets=zt, zone_boundary_costs=zbc,
        )
        assert zp.validate() == []
```

`pytest` and `pd` are already imported at the top of `test_model.py`; add `import numpy as np`
next to them (not mid-file, or ruff E402 fires).

- [ ] **Step 2: Run the tests to verify they fail**

Run: `/opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/zones/test_model.py -q -k TestContributionSource`
Expected: FAIL — `AttributeError: 'ZonalProblem' object has no attribute 'zone_index'` (and
siblings).

- [ ] **Step 3: Implement on `ZonalProblem`**

In `src/pymarxan/zones/model.py` add `import numpy as np` and replace `get_contribution` with
the block below (keep `get_zone_cost` as is):

```python
    # ------------------------------------------------------------------
    # Index conventions shared by objective.py, cache.py, mip_solver.py, writers.py
    # ------------------------------------------------------------------

    def zone_index(self) -> dict[int, int]:
        """Zone id -> matrix row: rank in ``sorted(zone_ids)`` + 1 (row 0 = unassigned)."""
        return {int(zid): k + 1 for k, zid in enumerate(sorted(self.zone_ids))}

    def feature_index(self) -> dict[int, int]:
        """Feature id -> matrix column, in ``features["id"]`` order."""
        return {int(fid): j for j, fid in enumerate(self.features["id"].values)}

    # ------------------------------------------------------------------
    # Contributions (MarZone zones.hpp:619-627 with a table, :651-668 without)
    # ------------------------------------------------------------------

    def contribution_default(self) -> float:
        """Contribution of an unlisted (feature, zone) pair.

        MarZone zero-fills the contribution array whenever a contribution file is
        supplied and only sets the listed pairs (``zones.hpp:619``); the all-ones
        default applies only when no file exists (``zones.hpp:651``).
        """
        return 1.0 if self.zone_contributions is None else 0.0

    def contribution_lookup(self) -> dict[tuple[int, int], float]:
        """Listed contributions keyed ``(feature_id, zone_id)``. Unlisted pairs are absent."""
        if self.zone_contributions is None:
            return {}
        zc = self.zone_contributions
        return {
            (int(f), int(z)): float(c)
            for f, z, c in zip(
                zc["feature"].values, zc["zone"].values, zc["contribution"].values,
                strict=True,
            )
        }

    def get_contribution(self, feature_id: int, zone_id: int) -> float:
        return self.contribution_lookup().get(
            (int(feature_id), int(zone_id)), self.contribution_default(),
        )

    def contribution_matrix(self) -> np.ndarray:
        """(n_zones + 1, n_feat) contribution per (zone row, feature column); row 0 is zero."""
        zidx = self.zone_index()
        fidx = self.feature_index()
        m = np.zeros((self.n_zones + 1, self.n_features), dtype=np.float64)
        m[1:, :] = self.contribution_default()
        for (fid, zid), c in self.contribution_lookup().items():
            row = zidx.get(zid)
            col = fidx.get(fid)
            if row is not None and col is not None:
                m[row, col] = c
        return m

    # ------------------------------------------------------------------
    # Zone-target weighting (pymarxan extension; MarZone always uses raw amounts)
    # ------------------------------------------------------------------

    def zone_target_contrib(self) -> int:
        """``ZONETARGETCONTRIB`` parameter as 0 or 1; ``ValueError`` for anything else."""
        raw = self.parameters.get("ZONETARGETCONTRIB", 0)
        if isinstance(raw, bool) or isinstance(raw, str) or raw not in (0, 1):
            raise ValueError(
                f"ZONETARGETCONTRIB must be 0 (raw zone targets, MarZone) or 1 "
                f"(contribution-weighted zone targets, pymarxan <= 0.35); got {raw!r}"
            )
        return int(raw)

    def zone_target_weight_matrix(self) -> np.ndarray:
        """Weight applied to amounts when accumulating zone targets.

        All ones (rows 1..n_zones) by default — MarZone ``reserve.hpp:164`` accumulates
        raw amounts — or the contribution matrix when ``ZONETARGETCONTRIB == 1``.
        """
        if self.zone_target_contrib() == 1:
            return self.contribution_matrix()
        w = np.zeros((self.n_zones + 1, self.n_features), dtype=np.float64)
        w[1:, :] = 1.0
        return w
```

Then extend `validate()`: after the existing `zone_contributions` column check, replace that
`if self.zone_contributions is not None:` block with:

```python
        if self.zone_contributions is not None:
            req = {"feature", "zone", "contribution"}
            if not req.issubset(set(self.zone_contributions.columns)):
                errors.append(
                    f"zone_contributions missing columns: "
                    f"{sorted(req - set(self.zone_contributions.columns))}"
                )
            else:
                feat_ids = [int(f) for f in self.features["id"].values]
                z_sorted = sorted(int(z) for z in self.zone_ids)
                lookup = self.contribution_lookup()
                unknown = [
                    (f, z) for (f, z) in lookup
                    if z not in self.zone_ids or f not in feat_ids
                ]
                if unknown:
                    f0, z0 = unknown[0]
                    errors.append(
                        f"zone_contributions references {len(unknown)} unknown "
                        f"(feature, zone) pair(s); first: feature {f0}, zone {z0}"
                    )
                total = len(feat_ids) * len(z_sorted)
                listed = len(lookup) - len(unknown)
                missing = [
                    (f, z) for f in feat_ids for z in z_sorted if (f, z) not in lookup
                ]
                if missing:
                    f0, z0 = missing[0]
                    errors.append(
                        f"zone_contributions lists {listed} of {total} (feature, zone) "
                        f"pairs; {len(missing)} unlisted pair(s) default to 0.0 "
                        f"(MarZone zones.hpp:619); first: feature {f0}, zone {z0}"
                    )
```

and append before `return errors`:

```python
        try:
            self.zone_target_contrib()
        except ValueError as exc:
            errors.append(str(exc))

        if "target2" in self.features.columns:
            t2 = self.features["target2"].fillna(0.0).astype(float)
            if (t2 > 0).any():
                errors.append(
                    "target2 (clumping) is not supported in zone problems: "
                    f"{int((t2 > 0).sum())} feature(s) have target2 > 0"
                )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `/opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/zones/test_model.py -q`
Expected: all PASS, including the pre-existing `test_get_contribution`,
`test_default_contribution_is_one`, `test_validate_valid`.

- [ ] **Step 5: Run the whole zones suite to confirm nothing else moved**

Run: `/opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/zones -q -m "not slow"`
Expected: PASS. (`test_writers.py` calls `get_contribution` on a table listing all pairs, so
the default change is invisible there.)

- [ ] **Step 6: Lint and commit**

```bash
~/.local/bin/ruff check src/pymarxan/zones/model.py tests/pymarxan/zones/test_model.py
git add src/pymarxan/zones/model.py tests/pymarxan/zones/test_model.py
git commit -m "feat(zones): contribution lookup/matrix, zone-target weight matrix, validate() domain checks (#2)

MarZone default for unlisted contribution pairs (zones.hpp:619 / :651); ZONETARGETCONTRIB
parameter domain; target2 rejected in zone problems; one summary message for partial tables.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01X8aGAi384D5xybKRugtFXT"
```

---

### Task 2: Anchor module and overall-target helpers in `objective.py` (additive)

**Files:**
- Create: `tests/pymarxan/zones/marzone_anchor.py`
- Create: `tests/pymarxan/zones/test_zone_overall_targets.py`
- Modify: `src/pymarxan/zones/objective.py`

**Interfaces:**
- Consumes: `ZonalProblem.contribution_matrix()`, `zone_index()`, `feature_index()` (Task 1).
- Produces:
  - `compute_overall_achieved(problem, zone_assignment, *, amounts=None) -> dict[int, float]`
  - `compute_overall_shortfalls(problem, zone_assignment) -> dict[int, float]` (features with target > 0)
  - `check_overall_targets(problem, zone_assignment) -> dict[int, bool]` (every feature id)
  - `compute_overall_penalty(problem, zone_assignment) -> float`
  - `compute_overall_shortfall(problem, zone_assignment) -> float`
  - `compute_zone_shortfalls(problem, zone_assignment) -> dict[tuple[int, int], float]` (issue #6 part 1)
  - test helpers `make_anchor_problem(...)`, `oracle(...)`, `all_assignments()`, constants
    `OPTIMUM`, `OPTIMUM_COST`, `RUNNER_UP_COST`.

Nothing is wired into `compute_zone_objective` yet (that is Task 5), so all existing tests keep
their current values.

- [ ] **Step 1: Create the anchor module (not a test file)**

`tests/pymarxan/zones/marzone_anchor.py`:

```python
"""Spec §5 anchor for MarZone overall + raw zone targets (hand-verified; review oracle).

Three PUs, two zones, one feature. Amount 10 in every PU. Contributions listed for every pair:
zone 1 -> 1.0, zone 2 -> 0.4. Zone costs: zone 1 = (5, 6, 7), zone 2 = (1, 1, 1). Overall
target 15. Zone target (zone 2, feature 1) = 10 raw. MISSLEVEL 1, BLM 0, no boundary.
Unique feasible optimum (1, 2, 2) at cost 7; runner-up 8. At spf = 1 the penalised minimum is
the infeasible (2, 2, 2) at 6.0 (threshold spf > 4/3); the heuristic tier therefore uses
spf = 10, where the penalised argmin equals the hard optimum.
"""
from __future__ import annotations

import itertools
from collections.abc import Sequence

import pandas as pd

from pymarxan.zones.model import ZonalProblem

AMOUNT = 10.0
CONTRIB: dict[int, float] = {1: 1.0, 2: 0.4}
ZONE_COSTS: dict[int, tuple[float, float, float]] = {1: (5.0, 6.0, 7.0), 2: (1.0, 1.0, 1.0)}
OVERALL_TARGET = 15.0
ZONE_TARGET = 10.0  # zone 2, feature 1, raw units
OPTIMUM = (1, 2, 2)
OPTIMUM_COST = 7.0
RUNNER_UP_COST = 8.0
HEURISTIC_SPF = 10.0


def make_anchor_problem(
    *,
    spf: float = 1.0,
    zone_target_contrib: int = 0,
    blm: float = 0.0,
    misslevel: float = 1.0,
) -> ZonalProblem:
    pu = pd.DataFrame({"id": [1, 2, 3], "cost": [0.0, 0.0, 0.0], "status": [0, 0, 0]})
    features = pd.DataFrame({
        "id": [1], "name": ["habitat"], "target": [OVERALL_TARGET], "spf": [spf],
    })
    puvspr = pd.DataFrame({
        "species": [1, 1, 1], "pu": [1, 2, 3], "amount": [AMOUNT, AMOUNT, AMOUNT],
    })
    zones = pd.DataFrame({"id": [1, 2], "name": ["protect", "restore"]})
    zone_costs = pd.DataFrame({
        "pu": [1, 2, 3, 1, 2, 3],
        "zone": [1, 1, 1, 2, 2, 2],
        "cost": [*ZONE_COSTS[1], *ZONE_COSTS[2]],
    })
    zone_contributions = pd.DataFrame({
        "feature": [1, 1], "zone": [1, 2], "contribution": [CONTRIB[1], CONTRIB[2]],
    })
    zone_targets = pd.DataFrame({"zone": [2], "feature": [1], "target": [ZONE_TARGET]})
    return ZonalProblem(
        planning_units=pu,
        features=features,
        pu_vs_features=puvspr,
        boundary=None,
        parameters={
            "BLM": blm,
            "MISSLEVEL": misslevel,
            "ZONETARGETCONTRIB": zone_target_contrib,
        },
        zones=zones,
        zone_costs=zone_costs,
        zone_contributions=zone_contributions,
        zone_targets=zone_targets,
        zone_boundary_costs=None,
    )


def oracle(
    assignment: Sequence[int],
    *,
    spf: float = 1.0,
    zone_target_contrib: int = 0,
) -> dict[str, float | bool]:
    """Spec §3 formulas written without pymarxan."""
    cost = sum(ZONE_COSTS[z][i] for i, z in enumerate(assignment) if z)
    overall = sum(AMOUNT * CONTRIB[z] for z in assignment if z)
    w = CONTRIB[2] if zone_target_contrib == 1 else 1.0
    zone2 = sum(AMOUNT * w for z in assignment if z == 2)
    penalty = spf * (max(0.0, OVERALL_TARGET - overall) + max(0.0, ZONE_TARGET - zone2))
    return {
        "cost": cost,
        "overall": overall,
        "zone2": zone2,
        "feasible": overall >= OVERALL_TARGET and zone2 >= ZONE_TARGET,
        "objective": cost + penalty,
    }


def all_assignments() -> list[tuple[int, int, int]]:
    return list(itertools.product([0, 1, 2], repeat=3))
```

- [ ] **Step 2: Write the failing tests**

`tests/pymarxan/zones/test_zone_overall_targets.py`:

```python
"""Overall (contribution-weighted) feature targets — objective module, spec §3.2."""
from __future__ import annotations

import numpy as np
import pytest

from pymarxan.zones.objective import (
    check_overall_targets,
    compute_overall_achieved,
    compute_overall_penalty,
    compute_overall_shortfall,
    compute_overall_shortfalls,
    compute_zone_shortfalls,
)
from tests.pymarxan.zones.marzone_anchor import (
    OPTIMUM,
    OPTIMUM_COST,
    all_assignments,
    make_anchor_problem,
    oracle,
)


class TestOracleSelfConsistency:
    """The oracle itself must reproduce the spec table before anything is compared to it."""

    def test_hard_optimum_is_unique(self):
        feasible = [(oracle(a)["cost"], a) for a in all_assignments() if oracle(a)["feasible"]]
        best = min(c for c, _ in feasible)
        assert best == OPTIMUM_COST
        assert [a for c, a in feasible if c == best] == [OPTIMUM]

    def test_spf_one_penalised_minimum_is_infeasible(self):
        best = min(all_assignments(), key=lambda a: oracle(a, spf=1.0)["objective"])
        assert best == (2, 2, 2)
        assert oracle(best, spf=1.0)["objective"] == 6.0

    def test_spf_ten_penalised_argmin_equals_hard_optimum(self):
        best = min(all_assignments(), key=lambda a: oracle(a, spf=10.0)["objective"])
        assert best == OPTIMUM
        assert oracle((2, 2, 2), spf=10.0)["objective"] == 33.0

    def test_contrib_weighted_zone_targets_infeasible_everywhere(self):
        assert not any(oracle(a, zone_target_contrib=1)["feasible"] for a in all_assignments())


class TestOverallAchieved:
    def test_matches_oracle_for_all_assignments(self):
        p = make_anchor_problem()
        for a in all_assignments():
            got = compute_overall_achieved(p, np.array(a))
            assert got == {1: pytest.approx(oracle(a)["overall"])}

    def test_unassigned_contributes_nothing(self):
        p = make_anchor_problem()
        assert compute_overall_achieved(p, np.array([0, 0, 0])) == {1: 0.0}

    def test_amounts_override_is_used(self):
        p = make_anchor_problem()
        amounts = np.full((3, 1), 2.0)
        got = compute_overall_achieved(p, np.array([1, 2, 0]), amounts=amounts)
        assert got == {1: pytest.approx(2.0 * 1.0 + 2.0 * 0.4)}

    def test_default_contribution_without_table_is_one(self):
        p = make_anchor_problem()
        p = p.copy_with(zone_contributions=None)
        assert compute_overall_achieved(p, np.array([2, 2, 2])) == {1: pytest.approx(30.0)}

    def test_default_contribution_with_partial_table_is_zero(self):
        p = make_anchor_problem()
        partial = p.zone_contributions[p.zone_contributions["zone"] == 1]
        p = p.copy_with(zone_contributions=partial)
        assert compute_overall_achieved(p, np.array([2, 2, 1])) == {1: pytest.approx(10.0)}


class TestOverallTargetsAndPenalty:
    def test_met_flags_follow_misslevel(self):
        p = make_anchor_problem()
        assert check_overall_targets(p, np.array([2, 2, 2])) == {1: False}   # 12 < 15
        assert check_overall_targets(p, np.array(OPTIMUM)) == {1: True}     # 18 >= 15
        p.parameters["MISSLEVEL"] = 0.8
        assert check_overall_targets(p, np.array([2, 2, 2])) == {1: True}    # 12 >= 12

    def test_inert_target_is_met(self):
        p = make_anchor_problem()
        p.features.loc[0, "target"] = 0.0
        assert check_overall_targets(p, np.array([0, 0, 0])) == {1: True}
        assert compute_overall_shortfalls(p, np.array([0, 0, 0])) == {}

    def test_penalty_is_spf_times_shortfall(self):
        p = make_anchor_problem(spf=10.0)
        assert compute_overall_penalty(p, np.array([2, 2, 2])) == pytest.approx(30.0)
        assert compute_overall_shortfall(p, np.array([2, 2, 2])) == pytest.approx(3.0)
        assert compute_overall_shortfalls(p, np.array([2, 2, 2])) == {1: pytest.approx(3.0)}
        assert compute_overall_penalty(p, np.array(OPTIMUM)) == 0.0

    def test_penalty_scales_with_misslevel(self):
        p = make_anchor_problem(spf=1.0, misslevel=0.5)
        # target 7.5; (0, 2, 0) achieves 4 -> shortfall 3.5
        assert compute_overall_penalty(p, np.array([0, 2, 0])) == pytest.approx(3.5)

    def test_spf_column_absent_defaults_to_one(self):
        p = make_anchor_problem()
        p = p.copy_with(features=p.features.drop(columns=["spf"]))
        assert compute_overall_penalty(p, np.array([2, 2, 2])) == pytest.approx(3.0)


class TestZoneShortfalls:
    def test_per_pair_shortfalls_on_anchor(self):
        p = make_anchor_problem()
        # Zone target (2, 1) = 10 raw; (2, 0, 0) holds 10 raw in zone 2 -> met.
        assert compute_zone_shortfalls(p, np.array([2, 0, 0])) == {(2, 1): 0.0}
        assert compute_zone_shortfalls(p, np.array([1, 1, 0])) == {(2, 1): 10.0}

    def test_empty_without_zone_targets(self):
        p = make_anchor_problem().copy_with(zone_targets=None)
        assert compute_zone_shortfalls(p, np.array([1, 1, 1])) == {}
```

- [ ] **Step 3: Run to verify failure**

Run: `/opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/zones/test_zone_overall_targets.py -q`
Expected: FAIL at import — `ImportError: cannot import name 'check_overall_targets'`.

- [ ] **Step 4: Implement the helpers in `objective.py`**

Add after `compute_standard_boundary` (before `_compute_zone_achieved`):

```python
# ----------------------------------------------------------------------
# Overall (contribution-weighted) feature targets — MarZone reserve.hpp:158-171,
# Watts et al. 2009 eq. 6.
# ----------------------------------------------------------------------


def _feature_arrays(problem: ZonalProblem) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(feature ids, targets × MISSLEVEL, spf) aligned with ``features`` order."""
    misslevel = float(problem.parameters.get("MISSLEVEL", 1.0))
    ids = problem.features["id"].values.astype(int)
    targets = problem.features["target"].values.astype(np.float64) * misslevel
    spf = (
        problem.features["spf"].values.astype(np.float64)
        if "spf" in problem.features.columns
        else np.ones(len(ids), dtype=np.float64)
    )
    return ids, targets, spf


def compute_overall_achieved(
    problem: ZonalProblem,
    zone_assignment: np.ndarray,
    *,
    amounts: np.ndarray | None = None,
) -> dict[int, float]:
    """A_f = Σ_i amount[i, f] × contribution[z_i, f] over PUs with z_i > 0.

    ``amounts`` (n_pu, n_feat) defaults to ``problem.build_pu_feature_matrix()``; the
    ``objectives`` package passes its (possibly probability-adjusted) effective amounts.
    """
    if amounts is None:
        amounts = problem.build_pu_feature_matrix()
    contrib = problem.contribution_matrix()
    zidx = problem.zone_index()
    rows = np.fromiter(
        (zidx.get(int(z), 0) for z in zone_assignment), dtype=np.int64, count=len(zone_assignment)
    )
    totals = (amounts * contrib[rows]).sum(axis=0)
    return {int(fid): float(totals[j]) for j, fid in enumerate(problem.features["id"].values)}


def compute_overall_shortfalls(
    problem: ZonalProblem,
    zone_assignment: np.ndarray,
) -> dict[int, float]:
    """feature id -> max(0, target × MISSLEVEL − A_f) for features with target > 0."""
    ids, targets, _ = _feature_arrays(problem)
    achieved = compute_overall_achieved(problem, zone_assignment)
    return {
        int(fid): max(0.0, float(t) - achieved.get(int(fid), 0.0))
        for fid, t in zip(ids, targets, strict=True)
        if t > 0
    }


def check_overall_targets(
    problem: ZonalProblem,
    zone_assignment: np.ndarray,
) -> dict[int, bool]:
    """feature id -> A_f >= target × MISSLEVEL (True for inert targets), every feature."""
    ids, targets, _ = _feature_arrays(problem)
    achieved = compute_overall_achieved(problem, zone_assignment)
    return {
        int(fid): bool(t <= 0 or achieved.get(int(fid), 0.0) >= float(t))
        for fid, t in zip(ids, targets, strict=True)
    }


def compute_overall_penalty(
    problem: ZonalProblem,
    zone_assignment: np.ndarray,
) -> float:
    """Σ_f spf_f × overall shortfall_f (approximation of MarZone's spf × penalty × proportion)."""
    ids, _, spf = _feature_arrays(problem)
    spf_of = {int(fid): float(s) for fid, s in zip(ids, spf, strict=True)}
    shortfalls = compute_overall_shortfalls(problem, zone_assignment)
    return float(sum(spf_of[fid] * sf for fid, sf in shortfalls.items()))


def compute_overall_shortfall(
    problem: ZonalProblem,
    zone_assignment: np.ndarray,
) -> float:
    """Unweighted total overall shortfall."""
    return float(sum(compute_overall_shortfalls(problem, zone_assignment).values()))
```

Add after `check_zone_targets`:

```python
def compute_zone_shortfalls(
    problem: ZonalProblem,
    zone_assignment: np.ndarray,
) -> dict[tuple[int, int], float]:
    """(zone id, feature id) -> max(0, zone target × MISSLEVEL − Z_kf) for listed zone targets."""
    if problem.zone_targets is None:
        return {}
    misslevel = float(problem.parameters.get("MISSLEVEL", 1.0))
    achieved = _compute_zone_achieved(problem, zone_assignment)
    zt = problem.zone_targets
    out: dict[tuple[int, int], float] = {}
    for zid, fid, target in zip(
        zt["zone"].values, zt["feature"].values, zt["target"].values, strict=True,
    ):
        key = (int(zid), int(fid))
        out[key] = max(0.0, float(target) * misslevel - achieved.get(key, 0.0))
    return out
```

and rewrite `compute_zone_penalty` / `compute_zone_shortfall` on top of it (behaviour unchanged):

```python
def compute_zone_penalty(
    problem: ZonalProblem,
    zone_assignment: np.ndarray,
) -> float:
    """Penalty for unmet zone targets: Σ spf_f × shortfall_kf."""
    shortfalls = compute_zone_shortfalls(problem, zone_assignment)
    if not shortfalls:
        return 0.0
    ids, _, spf = _feature_arrays(problem)
    spf_of = {int(fid): float(s) for fid, s in zip(ids, spf, strict=True)}
    return float(sum(spf_of.get(fid, 1.0) * sf for (_, fid), sf in shortfalls.items()))


def compute_zone_shortfall(
    problem: ZonalProblem,
    zone_assignment: np.ndarray,
) -> float:
    """Unweighted total shortfall across all zone targets."""
    return float(sum(compute_zone_shortfalls(problem, zone_assignment).values()))
```

- [ ] **Step 5: Run the new tests and the whole zones suite**

Run: `/opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/zones -q -m "not slow"`
Expected: PASS (new file green; `test_objective.py` unchanged because
`_compute_zone_achieved` is untouched in this task).

- [ ] **Step 6: Lint and commit**

```bash
~/.local/bin/ruff check src/pymarxan/zones/objective.py tests/pymarxan/zones/marzone_anchor.py tests/pymarxan/zones/test_zone_overall_targets.py
git add src/pymarxan/zones/objective.py tests/pymarxan/zones/marzone_anchor.py tests/pymarxan/zones/test_zone_overall_targets.py
git commit -m "feat(zones): overall-target helpers (achieved/met/penalty/shortfalls) + spec §5 anchor module (#2, #6)

Additive: compute_zone_objective is not yet wired (Task 5). compute_zone_shortfalls is the
public per-(zone, feature) shortfall helper from issue #6.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01X8aGAi384D5xybKRugtFXT"
```

---

### Task 3: Semantic flip A — raw zone targets and the MarZone contribution default, in objective, cache and MIP together

**Files:**
- Modify: `src/pymarxan/zones/objective.py` (`_compute_zone_achieved`)
- Modify: `src/pymarxan/zones/cache.py` (`from_zone_problem`, `compute_held_per_zone`, `update_held_per_zone`, `_compute_zone_penalty`, `_penalty_delta`, new field `zone_target_weight`)
- Modify: `src/pymarxan/zones/mip_solver.py` (`_zone_achieved_exprs` built once; `_build_penalty_expr`, `_add_zone_target_constraints`)
- Test: `tests/pymarxan/zones/test_zone_cache.py` (re-pins), `tests/pymarxan/zones/test_objective.py` (docstring + new tests), `tests/pymarxan/zones/test_zone_overall_targets.py` (weight-matrix tests)

**Interfaces:**
- Consumes: `zone_target_weight_matrix()`, `contribution_matrix()`, `zone_index()`, `feature_index()` (Task 1).
- Produces: `ZoneProblemCache.zone_target_weight: np.ndarray` (n_zones+1, n_feat);
  `ZoneProblemCache.contribution_matrix` now equals `problem.contribution_matrix()`;
  `held_per_zone` arrays are **raw** (no contribution multiply);
  `mip_solver._zone_achieved_exprs(problem, x, zone_ids) -> dict[tuple[int, int], pulp.LpAffineExpression]`;
  `_build_penalty_expr(problem, exprs, model)`, `_add_zone_target_constraints(problem, exprs, model)`.

Why together: `test_zone_cache.py::TestFullObjective` and `TestDeltaObjective` pin the cache
equal to `compute_zone_objective`; changing one side alone leaves the suite red.

- [ ] **Step 1: Re-pin the cache tests and add the mode tests (failing first)**

In `tests/pymarxan/zones/test_zone_cache.py::TestHeldPerZone::test_mixed_assignment` replace
the zone-2 block:

```python
        # Zone2: PU1(8,7) — raw amounts (MarZone reserve.hpp:164); contributions no longer apply
        # to zone targets. Old pins were 4.0 / 2.1 (8×0.5, 7×0.3).
        assert held[col2, f0] == pytest.approx(8.0)
        assert held[col2, f1] == pytest.approx(7.0)
```

Append to `test_zone_cache.py`:

```python
class TestZoneTargetWeight:
    def test_weight_is_ones_by_default(self, cache):
        np.testing.assert_array_equal(cache.zone_target_weight[1:], np.ones((2, 2)))
        np.testing.assert_array_equal(cache.zone_target_weight[0], np.zeros(2))

    def test_weight_is_contribution_matrix_when_flag_set(self, zone_problem):
        zone_problem.parameters["ZONETARGETCONTRIB"] = 1
        c = ZoneProblemCache.from_zone_problem(zone_problem)
        np.testing.assert_array_equal(c.zone_target_weight, c.contribution_matrix)
        np.testing.assert_array_equal(c.contribution_matrix, zone_problem.contribution_matrix())

    def test_cache_rejects_out_of_domain_flag(self, zone_problem):
        zone_problem.parameters["ZONETARGETCONTRIB"] = 2
        with pytest.raises(ValueError, match="ZONETARGETCONTRIB"):
            ZoneProblemCache.from_zone_problem(zone_problem)

    def test_full_objective_matches_reference_in_weighted_mode(self, zone_problem):
        zone_problem.parameters["ZONETARGETCONTRIB"] = 1
        c = ZoneProblemCache.from_zone_problem(zone_problem)
        rng = np.random.default_rng(7)
        for _ in range(10):
            assignment = rng.integers(0, 3, size=4)
            held = c.compute_held_per_zone(assignment)
            assert c.compute_full_zone_objective(assignment, held, 1.0) == pytest.approx(
                compute_zone_objective(zone_problem, assignment, 1.0), abs=1e-10,
            )

    def test_delta_matches_reference_in_weighted_mode(self, zone_problem):
        zone_problem.parameters["ZONETARGETCONTRIB"] = 1
        c = ZoneProblemCache.from_zone_problem(zone_problem)
        rng = np.random.default_rng(11)
        for _ in range(20):
            assignment = rng.integers(0, 3, size=4)
            idx = int(rng.integers(4))
            old_zone = int(assignment[idx])
            new_zone = int(rng.choice([z for z in (0, 1, 2) if z != old_zone]))
            held = c.compute_held_per_zone(assignment)
            before = c.compute_full_zone_objective(assignment, held, 1.0)
            delta = c.compute_delta_zone_objective(idx, old_zone, new_zone, assignment, held, 1.0)
            after_assignment = assignment.copy()
            after_assignment[idx] = new_zone
            after = c.compute_full_zone_objective(
                after_assignment, c.compute_held_per_zone(after_assignment), 1.0,
            )
            assert delta == pytest.approx(after - before, abs=1e-10)
```

In `tests/pymarxan/zones/test_objective.py::TestComputeZonePenalty::test_zone_penalty_zero_when_all_met`
replace the docstring with the raw numbers:

```python
        """Mixed assignment meeting all zone targets => penalty should be zero.

        Zone targets accumulate raw amounts (MarZone reserve.hpp:164).
        PU1,PU2 in zone 1 meets Z1 targets (F1: 18>=10, F2: 12>=8).
        PU3,PU4 in zone 2 meets Z2 targets (F1: 15>=5, F2: 13>=3).
        """
```

Append to `tests/pymarxan/zones/test_objective.py`:

```python
class TestZoneAchievedWeighting:
    """Spec §3.3: raw by default, contribution-weighted under ZONETARGETCONTRIB=1."""

    def test_raw_by_default(self):
        from pymarxan.zones.objective import _compute_zone_achieved
        problem = load_zone_project(DATA_DIR)
        achieved = _compute_zone_achieved(problem, np.array([1, 2, 1, 0]))
        assert achieved[(2, 1)] == 8.0   # PU2 feature 1, raw (was 4.0 = 8 × 0.5)
        assert achieved[(2, 2)] == 7.0   # PU2 feature 2, raw (was 2.1 = 7 × 0.3)

    def test_weighted_when_flag_set(self):
        from pymarxan.zones.objective import _compute_zone_achieved
        problem = load_zone_project(DATA_DIR)
        problem.parameters["ZONETARGETCONTRIB"] = 1
        achieved = _compute_zone_achieved(problem, np.array([1, 2, 1, 0]))
        assert achieved[(2, 1)] == 4.0
        assert achieved[(2, 2)] == pytest.approx(2.1)

    def test_partial_table_zero_weight_only_in_weighted_mode(self):
        from pymarxan.zones.objective import _compute_zone_achieved
        problem = load_zone_project(DATA_DIR)
        partial = problem.zone_contributions[problem.zone_contributions["zone"] == 1]
        problem = problem.copy_with(zone_contributions=partial)
        raw = _compute_zone_achieved(problem, np.array([2, 2, 2, 2]))
        assert raw[(2, 1)] == 29.0
        problem.parameters["ZONETARGETCONTRIB"] = 1
        weighted = _compute_zone_achieved(problem, np.array([2, 2, 2, 2]))
        assert weighted.get((2, 1), 0.0) == 0.0
```

(`test_objective.py` imports `pytest`? Check the top of the file; add `import pytest` there if
missing.)

Append to `tests/pymarxan/zones/test_zone_overall_targets.py`:

```python
class TestZoneTargetsRawOnAnchor:
    def test_zone_target_met_on_raw_amounts(self):
        from pymarxan.zones.objective import check_zone_targets
        p = make_anchor_problem()
        assert check_zone_targets(p, np.array([2, 0, 0])) == {(2, 1): True}     # 10 raw >= 10

    def test_zone_target_weighted_under_flag(self):
        from pymarxan.zones.objective import check_zone_targets
        p = make_anchor_problem(zone_target_contrib=1)
        assert check_zone_targets(p, np.array([2, 2, 0])) == {(2, 1): False}    # 8 < 10
        assert check_zone_targets(p, np.array([2, 2, 2])) == {(2, 1): True}     # 12 >= 10
```

- [ ] **Step 2: Run to verify failure**

Run: `/opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/zones/test_zone_cache.py tests/pymarxan/zones/test_objective.py tests/pymarxan/zones/test_zone_overall_targets.py -q`
Expected: FAIL — `test_mixed_assignment` (8.0 vs 4.0), `TestZoneTargetWeight` (no attribute
`zone_target_weight`), `TestZoneAchievedWeighting.test_raw_by_default`,
`TestZoneTargetsRawOnAnchor.test_zone_target_met_on_raw_amounts`.

- [ ] **Step 3: Flip `_compute_zone_achieved` in `objective.py`**

Replace the whole function:

```python
def _compute_zone_achieved(
    problem: ZonalProblem,
    zone_assignment: np.ndarray,
) -> dict[tuple[int, int], float]:
    """Achieved amount per (zone id, feature id): Σ_{i: z_i = k} amount[i, f] × w[k, f].

    ``w`` is ``problem.zone_target_weight_matrix()`` — all ones by default (MarZone
    ``reserve.hpp:164`` accumulates raw amounts for zone targets) or the contribution
    matrix under ``ZONETARGETCONTRIB = 1``.
    """
    weight = problem.zone_target_weight_matrix()
    zidx = problem.zone_index()
    fidx = problem.feature_index()
    amounts = problem.build_pu_feature_matrix()
    achieved: dict[tuple[int, int], float] = {}
    for zid, row in zidx.items():
        in_zone = zone_assignment == zid
        if not in_zone.any():
            continue
        totals = amounts[in_zone].sum(axis=0) * weight[row]
        for fid, col in fidx.items():
            if totals[col] != 0.0:
                achieved[(zid, fid)] = float(totals[col])
    return achieved
```

- [ ] **Step 4: Flip the cache**

In `cache.py`:

1. Add the field `zone_target_weight: np.ndarray` after `zone_target_matrix` in the dataclass
   and document it in the class docstring:
   `(n_zones+1, n_feat) float64 — weight applied to raw held amounts when scoring zone
   targets; ones by default, the contribution matrix under ZONETARGETCONTRIB=1.` Update the
   `contribution_matrix` doc line to `Row 0 (unassigned) is always 0; unlisted pairs use
   ZonalProblem.contribution_default().`
2. In `from_zone_problem`, replace the feature/zone index loops with
   `feat_id_to_col = problem.feature_index()` and `zone_id_to_col = problem.zone_index()`,
   replace the whole "Contribution matrix" block with
   `contribution_matrix = problem.contribution_matrix()`, add
   `zone_target_weight = problem.zone_target_weight_matrix()` right after it, and pass
   `zone_target_weight=zone_target_weight` to `cls(...)`. Guard spf:
   ```python
   feat_spf = (
       np.asarray(feat_df["spf"].values, dtype=np.float64)
       if "spf" in feat_df.columns
       else np.ones(n_feat, dtype=np.float64)
   )
   ```
3. `compute_held_per_zone`: the docstring becomes "held[z_col, f] = Σ raw amount (no
   contribution; MarZone reserve.hpp:164)" and the accumulation line becomes
   `held[zcol] += self.pu_feat_matrix[i]`.
4. `update_held_per_zone`: `held[old_col] -= amounts` and `held[new_col] += amounts`.
5. `_compute_zone_penalty`:
   ```python
   shortfalls = np.maximum(0.0, self.zone_target_matrix - held_per_zone * self.zone_target_weight)
   ```
6. `_penalty_delta`: replace `self.contribution_matrix[old_col]` with
   `self.zone_target_weight[old_col]` and `self.contribution_matrix[new_col]` with
   `self.zone_target_weight[new_col]`, and the two `old_held` / `new_held` reads with
   `held_per_zone[old_col] * self.zone_target_weight[old_col]` and
   `held_per_zone[new_col] * self.zone_target_weight[new_col]`. Rename the local variables
   `old_contrib` / `new_contrib` to `old_w` / `new_w`.

- [ ] **Step 5: Flip the MIP zone-target expressions**

In `mip_solver.py` replace `_build_penalty_expr` and `_add_zone_target_constraints` with:

```python
def _feature_groups(problem: ZonalProblem) -> dict[int, list[tuple[int, float]]]:
    """feature id -> [(pu id, amount), ...] from pu_vs_features."""
    puvspr = problem.pu_vs_features
    groups: dict[int, list[tuple[int, float]]] = {}
    for pid, fid, amt in zip(
        puvspr["pu"].values, puvspr["species"].values, puvspr["amount"].values, strict=True,
    ):
        groups.setdefault(int(fid), []).append((int(pid), float(amt)))
    return groups


def _spf_lookup(problem: ZonalProblem) -> dict[int, float]:
    feat_ids = problem.features["id"].values
    feat_spf = (
        problem.features["spf"].values
        if "spf" in problem.features.columns
        else np.ones(len(feat_ids))
    )
    return {int(f): float(s) for f, s in zip(feat_ids, feat_spf, strict=True)}


def _zone_achieved_exprs(
    problem: ZonalProblem,
    x: dict[tuple[int, int], pulp.LpVariable],
    zone_ids: list[int],
) -> dict[tuple[int, int], pulp.LpAffineExpression]:
    """Σ_i amount[i, f] × w[z, f] × x[i, z] for every listed (zone, feature) target.

    Built once and shared by the penalty (slack) and the hard constraint. ``w`` is
    ``problem.zone_target_weight_matrix()`` (raw by default, MarZone ``reserve.hpp:164``).
    """
    if problem.zone_targets is None:
        return {}
    weight = problem.zone_target_weight_matrix()
    zidx = problem.zone_index()
    fidx = problem.feature_index()
    groups = _feature_groups(problem)
    exprs: dict[tuple[int, int], pulp.LpAffineExpression] = {}
    zt = problem.zone_targets
    for zid, fid in zip(zt["zone"].values, zt["feature"].values, strict=True):
        zid, fid = int(zid), int(fid)
        w = float(weight[zidx[zid], fidx[fid]]) if zid in zidx and fid in fidx else 0.0
        exprs[(zid, fid)] = pulp.lpSum(
            amt * w * x[(pid, zid)]
            for pid, amt in groups.get(fid, [])
            if (pid, zid) in x
        )
    return exprs


def _build_penalty_expr(
    problem: ZonalProblem,
    exprs: dict[tuple[int, int], pulp.LpAffineExpression],
    model: pulp.LpProblem,
) -> tuple[pulp.LpAffineExpression, dict]:
    """Build penalty expression for unmet zone targets (SPF * slack)."""
    if problem.zone_targets is None:
        return pulp.lpSum([]), {}
    spf_lookup = _spf_lookup(problem)
    misslevel = float(problem.parameters.get("MISSLEVEL", 1.0))
    zt = problem.zone_targets
    expr = pulp.lpSum([])
    slack_vars = {}
    for zid, fid, target in zip(
        zt["zone"].values, zt["feature"].values, zt["target"].values, strict=True,
    ):
        zid, fid = int(zid), int(fid)
        slack = pulp.LpVariable(f"slack_{zid}_{fid}", lowBound=0, cat="Continuous")
        model += (
            slack >= float(target) * misslevel - exprs[(zid, fid)],
            f"shortfall_{zid}_{fid}",
        )
        slack_vars[(zid, fid)] = slack
        expr += spf_lookup.get(fid, 1.0) * slack
    return expr, slack_vars


def _add_zone_target_constraints(
    problem: ZonalProblem,
    exprs: dict[tuple[int, int], pulp.LpAffineExpression],
    model: pulp.LpProblem,
) -> None:
    """Add hard zone-specific feature target constraints."""
    if problem.zone_targets is None:
        return
    misslevel = float(problem.parameters.get("MISSLEVEL", 1.0))
    zt = problem.zone_targets
    for zid, fid, target in zip(
        zt["zone"].values, zt["feature"].values, zt["target"].values, strict=True,
    ):
        zid, fid = int(zid), int(fid)
        model += (exprs[(zid, fid)] >= float(target) * misslevel, f"zone_target_{zid}_{fid}")
```

and in `solve` replace the two call sites:

```python
        # 4. Zone-target penalty (slack) and hard constraints share one achieved expression
        zone_exprs = _zone_achieved_exprs(problem, x, zone_ids)
        penalty_expr, penalty_vars = _build_penalty_expr(problem, zone_exprs, model)
        ...
        # --- Feature target constraints ---
        _add_zone_target_constraints(problem, zone_exprs, model)
```

Run `grep -rn "_build_penalty_expr\|_add_zone_target_constraints" tests/` — expected: no
matches (the helpers are private). If a test calls them, update its call to the new signature.

- [ ] **Step 6: Run the zones suite (incl. slow) and the parity harness**

Run: `/opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/zones tests/test_examples.py -q`
Expected: PASS. The zones fixture stays feasible under raw zone targets (witness assignment
(1, 1, 2, 1): zone 1 raw f1 = 23 ≥ 10, f2 = 16 ≥ 8; zone 2 raw f1 = 6 ≥ 5, f2 = 9 ≥ 3).

- [ ] **Step 7: Lint and commit**

```bash
~/.local/bin/ruff check src/pymarxan/zones tests/pymarxan/zones
git add src/pymarxan/zones/objective.py src/pymarxan/zones/cache.py src/pymarxan/zones/mip_solver.py tests/pymarxan/zones/test_zone_cache.py tests/pymarxan/zones/test_objective.py tests/pymarxan/zones/test_zone_overall_targets.py
git commit -m "feat(zones)!: zone targets accumulate raw amounts; contribution default per MarZone (#2)

Objective, cache and MIP flip together (the cache-equals-objective tests gate it).
ZONETARGETCONTRIB=1 restores contribution-weighted zone targets via a weight matrix.
Re-pins: test_zone_cache mixed_assignment zone-2 held 4.0/2.1 -> 8.0/7.0 (8x0.5, 7x0.3 ->
raw 8, 7). test_objective docstring: zone-2 amounts 5.5/3.9 -> 15/13 raw.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01X8aGAi384D5xybKRugtFXT"
```

---

### Task 4: Semantic flip B — the overall-target term in `compute_zone_objective` and the cache (`ZoneHeld`), SA loop carries it

**Files:**
- Modify: `src/pymarxan/zones/objective.py` (`compute_zone_objective`)
- Modify: `src/pymarxan/zones/cache.py` (new `ZoneHeld`; `compute_held` / `update_held` replace `compute_held_per_zone` / `update_held_per_zone`; kw-only `held`; overall vector, gate, delta)
- Modify: `src/pymarxan/zones/solver.py` (SA loop, lines 169–248)
- Test: `tests/pymarxan/zones/test_zone_cache.py` (call-site sweep + new tests), `tests/pymarxan/zones/test_zone_connectivity.py:113-152` (sweep), `tests/pymarxan/zones/test_zone_overall_targets.py` (objective-vs-oracle)

**Interfaces:**
- Consumes: Task 2 helpers; Task 3 cache fields.
- Produces:
  - `ZoneHeld` dataclass: `per_zone: np.ndarray` (n_zones+1, n_feat) raw; `overall: np.ndarray` (n_feat,) contribution-weighted; `copy() -> ZoneHeld`.
  - `ZoneProblemCache.compute_held(assignment) -> ZoneHeld`
  - `ZoneProblemCache.update_held(held, idx, old_zone, new_zone) -> None` (zone **ids**)
  - `ZoneProblemCache.compute_full_zone_objective(assignment, *, held, blm) -> float`
  - `ZoneProblemCache.compute_delta_zone_objective(idx, old_zone, new_zone, assignment, *, held, blm) -> float`
  - fields `overall_target_vector: np.ndarray` (n_feat,), `has_overall_targets: bool`, `contrib_differs: np.ndarray` bool (n_zones+1, n_zones+1)
  - `compute_zone_objective` now includes `compute_overall_penalty`.
- Removed: `compute_held_per_zone`, `update_held_per_zone` (stale callers fail loudly, spec §4.4).

- [ ] **Step 1: Sweep the existing cache call sites in the tests to the new API (they will fail until Step 4)**

In `tests/pymarxan/zones/test_zone_cache.py` and `tests/pymarxan/zones/test_zone_connectivity.py`
apply, by hand (not blind sed — the `held[...]` reads must become `held.per_zone[...]`):

| Old | New |
|---|---|
| `cache.compute_held_per_zone(a)` | `cache.compute_held(a)` |
| `held[col, f]` / `held.shape` / `assert_array_almost_equal(held, np.zeros(...))` | `held.per_zone[col, f]` / `held.per_zone.shape` / `assert_array_almost_equal(held.per_zone, ...)` |
| `cache.update_held_per_zone(held, idx, old, new)` | `cache.update_held(held, idx, old, new)` |
| `assert_array_almost_equal(held, held_expected)` | `assert_array_almost_equal(held.per_zone, held_expected.per_zone)` **and** `assert_array_almost_equal(held.overall, held_expected.overall)` |
| `cache.compute_full_zone_objective(a, held, blm)` | `cache.compute_full_zone_objective(a, held=held, blm=blm)` |
| `cache.compute_delta_zone_objective(idx, o, n, a, held, blm)` | `cache.compute_delta_zone_objective(idx, o, n, a, held=held, blm=blm)` |

Confirm the sweep: `grep -rn "held_per_zone" tests/ src/` must return only `solver.py` (fixed in
Step 5) and the cache's private `_penalty_delta` parameter name.

- [ ] **Step 2: Add the new failing tests**

Append to `tests/pymarxan/zones/test_zone_cache.py`:

```python
from tests.pymarxan.zones.marzone_anchor import (
    all_assignments,
    make_anchor_problem,
    oracle,
)


class TestZoneHeld:
    @pytest.mark.parametrize("flag", [0, 1])
    def test_overall_equals_contribution_weighted_per_zone(self, zone_problem, flag):
        zone_problem.parameters["ZONETARGETCONTRIB"] = flag
        c = ZoneProblemCache.from_zone_problem(zone_problem)
        rng = np.random.default_rng(3)
        for _ in range(10):
            a = rng.integers(0, 3, size=4)
            held = c.compute_held(a)
            np.testing.assert_allclose(
                (c.contribution_matrix * held.per_zone).sum(axis=0), held.overall, atol=1e-12,
            )

    @pytest.mark.parametrize("flag", [0, 1])
    def test_per_zone_is_raw_in_both_modes(self, zone_problem, flag):
        zone_problem.parameters["ZONETARGETCONTRIB"] = flag
        c = ZoneProblemCache.from_zone_problem(zone_problem)
        held = c.compute_held(np.array([1, 2, 1, 0]))
        assert held.per_zone[c.zone_id_to_col[2], c.feat_id_to_col[1]] == pytest.approx(8.0)

    def test_overall_is_contribution_weighted(self, cache):
        held = cache.compute_held(np.array([1, 2, 1, 0]))
        # f1: PU0 10×1.0 + PU2 6×1.0 + PU1 8×0.5 = 20; f2: 5 + 9 + 7×0.3 = 16.1
        assert held.overall[cache.feat_id_to_col[1]] == pytest.approx(20.0)
        assert held.overall[cache.feat_id_to_col[2]] == pytest.approx(16.1)

    def test_update_keeps_invariant_through_a_walk(self, cache):
        rng = np.random.default_rng(5)
        a = rng.integers(0, 3, size=4)
        held = cache.compute_held(a)
        for _ in range(50):
            idx = int(rng.integers(4))
            new = int(rng.integers(0, 3))
            cache.update_held(held, idx, int(a[idx]), new)
            a[idx] = new
        fresh = cache.compute_held(a)
        np.testing.assert_allclose(held.per_zone, fresh.per_zone, atol=1e-12)
        np.testing.assert_allclose(held.overall, fresh.overall, atol=1e-12)


class TestOverallTermInCache:
    def test_overall_target_vector_applies_misslevel_and_zeroes_inert(self, zone_problem):
        zone_problem.parameters["MISSLEVEL"] = 0.5
        zone_problem.features.loc[1, "target"] = 0.0
        c = ZoneProblemCache.from_zone_problem(zone_problem)
        np.testing.assert_array_equal(c.overall_target_vector, [10.0, 0.0])
        assert c.has_overall_targets

    def test_no_overall_targets_flag(self, zone_problem):
        zone_problem.features["target"] = 0.0
        c = ZoneProblemCache.from_zone_problem(zone_problem)
        assert not c.has_overall_targets
        held = c.compute_held(np.array([0, 0, 0, 0]))
        assert c._compute_overall_penalty(held.overall) == 0.0

    def test_contrib_differs_gate(self, zone_problem):
        c = ZoneProblemCache.from_zone_problem(zone_problem)
        assert c.contrib_differs[1, 2]          # fixture contributions differ (1.0 vs 0.5/0.3)
        assert c.contrib_differs[0, 1]
        assert not c.contrib_differs[1, 1]
        plain = ZoneProblemCache.from_zone_problem(
            zone_problem.copy_with(zone_contributions=None),
        )
        assert not plain.contrib_differs[1, 2]  # default contributions: zone-to-zone moves free
        assert plain.contrib_differs[0, 2]      # assigning an unassigned PU still changes A_f

    def test_gated_move_has_zero_overall_delta(self, zone_problem):
        plain = ZoneProblemCache.from_zone_problem(
            zone_problem.copy_with(zone_contributions=None),
        )
        held = plain.compute_held(np.array([1, 1, 2, 2]))
        assert plain._overall_penalty_delta(0, 1, 2, held.overall) == 0.0

    @pytest.mark.parametrize("flag", [0, 1])
    def test_delta_matches_full_recomputation_random_flips(self, zone_problem, flag):
        zone_problem.parameters["ZONETARGETCONTRIB"] = flag
        c = ZoneProblemCache.from_zone_problem(zone_problem)
        rng = np.random.default_rng(13)
        for _ in range(40):
            a = rng.integers(0, 3, size=4)
            idx = int(rng.integers(4))
            old = int(a[idx])
            new = int(rng.choice([z for z in (0, 1, 2) if z != old]))
            held = c.compute_held(a)
            before = c.compute_full_zone_objective(a, held=held, blm=1.0)
            delta = c.compute_delta_zone_objective(idx, old, new, a, held=held, blm=1.0)
            b = a.copy()
            b[idx] = new
            after = c.compute_full_zone_objective(b, held=c.compute_held(b), blm=1.0)
            assert delta == pytest.approx(after - before, abs=1e-10)

    def test_spf_column_absent_defaults_to_one(self):
        p = make_anchor_problem()
        p = p.copy_with(features=p.features.drop(columns=["spf"]))
        c = ZoneProblemCache.from_zone_problem(p)
        np.testing.assert_array_equal(c.feat_spf, [1.0])


class TestCacheAgainstAnchorOracle:
    @pytest.mark.parametrize("spf", [1.0, 10.0])
    @pytest.mark.parametrize("flag", [0, 1])
    def test_full_objective_equals_oracle_for_all_27(self, spf, flag):
        p = make_anchor_problem(spf=spf, zone_target_contrib=flag)
        c = ZoneProblemCache.from_zone_problem(p)
        for a in all_assignments():
            arr = np.array(a)
            got = c.compute_full_zone_objective(arr, held=c.compute_held(arr), blm=0.0)
            assert got == pytest.approx(oracle(a, spf=spf, zone_target_contrib=flag)["objective"])
```

Append to `tests/pymarxan/zones/test_zone_overall_targets.py`:

```python
class TestZoneObjectiveIncludesBothTiers:
    @pytest.mark.parametrize("spf", [1.0, 10.0])
    @pytest.mark.parametrize("flag", [0, 1])
    def test_compute_zone_objective_equals_oracle(self, spf, flag):
        from pymarxan.zones.objective import compute_zone_objective
        p = make_anchor_problem(spf=spf, zone_target_contrib=flag)
        for a in all_assignments():
            got = compute_zone_objective(p, np.array(a), 0.0)
            assert got == pytest.approx(oracle(a, spf=spf, zone_target_contrib=flag)["objective"])

    def test_spf_one_penalised_minimum_is_the_infeasible_corner(self):
        from pymarxan.zones.objective import compute_zone_objective
        p = make_anchor_problem(spf=1.0)
        assert compute_zone_objective(p, np.array([2, 2, 2]), 0.0) == pytest.approx(6.0)
        assert check_overall_targets(p, np.array([2, 2, 2])) == {1: False}
```

- [ ] **Step 3: Run to verify failure**

Run: `/opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/zones/test_zone_cache.py tests/pymarxan/zones/test_zone_connectivity.py tests/pymarxan/zones/test_zone_overall_targets.py -q`
Expected: FAIL — `AttributeError: 'ZoneProblemCache' object has no attribute 'compute_held'`;
`test_compute_zone_objective_equals_oracle` fails on assignments with an unmet overall target.

- [ ] **Step 4: Implement**

`objective.py` — `compute_zone_objective` becomes:

```python
def compute_zone_objective(
    problem: ZonalProblem,
    zone_assignment: np.ndarray,
    blm: float,
) -> float:
    """Full MarZone-style objective.

    zone_cost + BLM × standard_boundary + zone_boundary + overall_penalty + zone_penalty
    + connectivity. Reference implementation (DataFrame-based); ``ZoneProblemCache`` must
    agree to 1e-10.
    """
    cost = compute_zone_cost(problem, zone_assignment)
    std_boundary = compute_standard_boundary(problem, zone_assignment)
    zone_boundary = compute_zone_boundary(problem, zone_assignment)
    overall_penalty = compute_overall_penalty(problem, zone_assignment)
    zone_penalty = compute_zone_penalty(problem, zone_assignment)
    connectivity = compute_zone_connectivity(problem, zone_assignment)
    return (
        cost + blm * std_boundary + zone_boundary + overall_penalty + zone_penalty + connectivity
    )
```

`cache.py`:

1. Module docstring: add "Two accumulators (``ZoneHeld``) mirror MarZone's ``zoneSpec`` (raw,
   per zone; ``reserve.hpp:164``) and ``speciesAmounts`` (contribution-weighted; ``:170``)."
2. Before `ZoneProblemCache` add:

```python
@dataclass
class ZoneHeld:
    """Held amounts carried through a solver run.

    per_zone : (n_zones+1, n_feat) raw amounts per (zone row, feature) — zone targets.
    overall  : (n_feat,) Σ_z contribution[z, f] × per_zone[z, f] — overall targets.
    """

    per_zone: np.ndarray
    overall: np.ndarray

    def copy(self) -> ZoneHeld:
        return ZoneHeld(self.per_zone.copy(), self.overall.copy())
```

3. Add fields after `zone_target_weight`: `overall_target_vector: np.ndarray`,
   `has_overall_targets: bool`, `contrib_differs: np.ndarray`, with docstring lines:
   `overall_target_vector : (n_feat,) features.target × MISSLEVEL, 0 where target <= 0.`
   `has_overall_targets : bool — any overall_target_vector > 0 (skips the term entirely).`
   `contrib_differs : (n_zones+1, n_zones+1) bool — contrib_differs[a, b] is True when any
   feature's contribution differs between zone rows a and b (MarZone reserve.hpp:393 gate).`
4. In `from_zone_problem`, after `zone_target_weight`:

```python
        # --- Overall (contribution-weighted) targets ---
        misslevel = float(problem.parameters.get("MISSLEVEL", 1.0))
        raw_targets = (
            np.asarray(feat_df["target"].values, dtype=np.float64)
            if "target" in feat_df.columns
            else np.zeros(n_feat, dtype=np.float64)
        )
        overall_target_vector = np.where(raw_targets > 0, raw_targets * misslevel, 0.0)
        has_overall_targets = bool(np.any(overall_target_vector > 0))
        contrib_differs = np.zeros((n_zones + 1, n_zones + 1), dtype=bool)
        for a in range(n_zones + 1):
            for b in range(n_zones + 1):
                contrib_differs[a, b] = bool(
                    np.any(contribution_matrix[a] != contribution_matrix[b])
                )
```

   (The existing `misslevel` line for the zone target matrix stays; reuse the variable.) Pass
   the three new fields to `cls(...)`.
5. Replace `compute_held_per_zone` / `update_held_per_zone` with:

```python
    def _zone_col(self, zone_id: int) -> int:
        return 0 if zone_id == 0 else self.zone_id_to_col.get(zone_id, 0)

    def compute_held(self, assignment: np.ndarray) -> ZoneHeld:
        """Both accumulators from scratch: raw per zone, contribution-weighted overall."""
        per_zone = np.zeros((self.n_zones + 1, self.n_feat), dtype=np.float64)
        overall = np.zeros(self.n_feat, dtype=np.float64)
        for i in range(self.n_pu):
            zcol = self._zone_col(int(assignment[i]))
            if zcol == 0:
                continue
            per_zone[zcol] += self.pu_feat_matrix[i]
            overall += self.pu_feat_matrix[i] * self.contribution_matrix[zcol]
        return ZoneHeld(per_zone, overall)

    def update_held(
        self,
        held: ZoneHeld,
        idx: int,
        old_zone: int,
        new_zone: int,
    ) -> None:
        """In-place update of both accumulators after PU ``idx`` moves old_zone -> new_zone."""
        amounts = self.pu_feat_matrix[idx]
        old_col = self._zone_col(old_zone)
        new_col = self._zone_col(new_zone)
        if old_col != 0:
            held.per_zone[old_col] -= amounts
            held.overall -= amounts * self.contribution_matrix[old_col]
        if new_col != 0:
            held.per_zone[new_col] += amounts
            held.overall += amounts * self.contribution_matrix[new_col]
```

6. `compute_full_zone_objective(self, assignment, *, held: ZoneHeld, blm: float)`: docstring
   objective line becomes `zone_cost + BLM * standard_boundary + zone_boundary +
   overall_penalty + zone_penalty + connectivity`; body uses
   `zone_penalty = self._compute_zone_penalty(held.per_zone)` and adds
   `overall_penalty = self._compute_overall_penalty(held.overall)` to the returned sum.
7. `compute_delta_zone_objective(self, idx, old_zone, new_zone, assignment, *, held: ZoneHeld, blm: float)`:
   `penalty_delta = self._penalty_delta(idx, old_col, new_col, held.per_zone) +
   self._overall_penalty_delta(idx, old_col, new_col, held.overall)`.
8. Add the two private helpers after `_compute_zone_penalty`:

```python
    def _compute_overall_penalty(self, overall: np.ndarray) -> float:
        """Σ_f spf[f] × max(0, overall_target[f] − overall[f])."""
        if not self.has_overall_targets:
            return 0.0
        shortfalls = np.maximum(0.0, self.overall_target_vector - overall)
        return float(np.dot(self.feat_spf, shortfalls))

    def _overall_penalty_delta(
        self,
        idx: int,
        old_col: int,
        new_col: int,
        overall: np.ndarray,
    ) -> float:
        """Overall-penalty change for moving PU idx old_col -> new_col.

        Dense row op on the PU's row, skipped when no overall target exists or when no
        feature's contribution differs between the two zones (``contrib_differs`` gate,
        MarZone ``reserve.hpp:393``): under default contributions every zone-to-zone move
        leaves ``overall`` unchanged.
        """
        if not self.has_overall_targets or not self.contrib_differs[old_col, new_col]:
            return 0.0
        change = self.pu_feat_matrix[idx] * (
            self.contribution_matrix[new_col] - self.contribution_matrix[old_col]
        )
        before = np.maximum(0.0, self.overall_target_vector - overall)
        after = np.maximum(0.0, self.overall_target_vector - (overall + change))
        return float(np.dot(self.feat_spf, after - before))
```

- [ ] **Step 5: Carry `ZoneHeld` through the SA loop**

In `solver.py` replace lines 169–173 with

```python
            # Both accumulators (raw per zone, contribution-weighted overall) from the cache
            held = cache.compute_held(assignment)
            current_obj = cache.compute_full_zone_objective(assignment, held=held, blm=blm)
```

and the two delta calls with
`cache.compute_delta_zone_objective(idx, old_zone, new_zone, assignment, held=held, blm=blm)`,
and the update with `cache.update_held(held, idx, old_zone, new_zone)`.

- [ ] **Step 6: Run the zones suite (incl. slow) and the parity harness**

Run: `/opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/zones tests/test_examples.py -q`
Expected: PASS. Confirm `grep -rn "compute_held_per_zone\|update_held_per_zone" src tests`
returns nothing.

- [ ] **Step 7: Lint and commit**

```bash
~/.local/bin/ruff check src/pymarxan/zones tests/pymarxan/zones
git add src/pymarxan/zones/objective.py src/pymarxan/zones/cache.py src/pymarxan/zones/solver.py tests/pymarxan/zones/test_zone_cache.py tests/pymarxan/zones/test_zone_connectivity.py tests/pymarxan/zones/test_zone_overall_targets.py
git commit -m "feat(zones): overall feature targets in the objective and the SA cache (ZoneHeld) (#2)

compute_zone_objective adds the overall penalty; ZoneProblemCache carries raw per-zone and
contribution-weighted overall accumulators in one ZoneHeld state (required keyword argument);
overall delta is a dense row op gated by contrib_differs[old, new] (reserve.hpp:393).
compute_held_per_zone/update_held_per_zone removed.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01X8aGAi384D5xybKRugtFXT"
```

---

### Task 5: One shared `build_zone_solution`; all four solvers and `ZoneIISolver.improve` use it

**Files:**
- Modify: `src/pymarxan/zones/objective.py` (add `build_zone_solution`)
- Modify: `src/pymarxan/zones/solver.py` (both `Solution(...)` sites), `heuristic.py` (`_build_zone_solution` removed), `iterative_improvement.py` (`_build_zone_solution` removed), `mip_solver.py` (`_build_zone_solution` removed)
- Modify: `src/pymarxan/solvers/base.py:20-21` (comment)
- Create: `tests/pymarxan/zones/test_zone_solvers_overall.py`
- Test: `tests/pymarxan/zones/test_zone_overall_targets.py` (builder fields)

**Interfaces:**
- Consumes: Task 2 helpers, `compute_zone_objective` (Task 4).
- Produces: `build_zone_solution(problem, zone_assignment, blm, *, solver_name: str, run: int | None = None) -> Solution` in `objective.py`:
  - `targets_met = check_overall_targets(...)` (feature ids → bool)
  - `penalty = overall_penalty + zone_penalty`; `shortfall = overall + zone shortfall`
  - `objective = compute_zone_objective(...)`
  - `metadata`: `solver`, `zone_boundary_cost` (rounded 4), `zone_targets_met` (`"z{z}_f{f}"` → bool), `overall_penalty`, `zone_penalty`, and `run` when given.

- [ ] **Step 1: Write the failing tests**

Append to `tests/pymarxan/zones/test_zone_overall_targets.py`:

```python
class TestBuildZoneSolution:
    def test_fields_on_anchor(self):
        from pymarxan.zones.objective import build_zone_solution, compute_zone_objective
        p = make_anchor_problem(spf=10.0)
        sol = build_zone_solution(p, np.array([2, 2, 2]), 0.0, solver_name="test", run=3)
        assert sol.cost == 3.0
        assert sol.objective == compute_zone_objective(p, np.array([2, 2, 2]), 0.0)
        assert sol.objective == pytest.approx(33.0)
        assert sol.targets_met == {1: False}
        assert sol.all_targets_met is False
        assert sol.penalty == pytest.approx(30.0)
        assert sol.shortfall == pytest.approx(3.0)
        assert sol.metadata["zone_targets_met"] == {"z2_f1": True}
        assert sol.metadata["overall_penalty"] == pytest.approx(30.0)
        assert sol.metadata["zone_penalty"] == 0.0
        assert sol.metadata["solver"] == "test"
        assert sol.metadata["run"] == 3
        assert sol.metadata["zone_boundary_cost"] == 0.0
        np.testing.assert_array_equal(sol.selected, [True, True, True])
        np.testing.assert_array_equal(sol.zone_assignment, [2, 2, 2])

    def test_run_omitted_when_none(self):
        from pymarxan.zones.objective import build_zone_solution
        p = make_anchor_problem()
        sol = build_zone_solution(p, np.array(OPTIMUM), 0.0, solver_name="test")
        assert "run" not in sol.metadata
        assert sol.penalty == 0.0 and sol.shortfall == 0.0

    def test_assignment_is_copied(self):
        from pymarxan.zones.objective import build_zone_solution
        p = make_anchor_problem()
        a = np.array(OPTIMUM)
        sol = build_zone_solution(p, a, 0.0, solver_name="test")
        a[0] = 0
        assert sol.zone_assignment is not None
        assert int(sol.zone_assignment[0]) == 1
```

`tests/pymarxan/zones/test_zone_solvers_overall.py` (only the agreement class in this task;
Tasks 7 and 8 append the anchor classes):

```python
"""Zone SA / heuristic / II on the spec §5 anchor, and cross-solver agreement on the fixture."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pymarxan.solvers.base import SolverConfig
from pymarxan.zones.heuristic import ZoneHeuristicSolver
from pymarxan.zones.iterative_improvement import ZoneIISolver
from pymarxan.zones.mip_solver import ZoneMIPSolver
from pymarxan.zones.objective import (
    check_overall_targets,
    check_zone_targets,
    compute_zone_objective,
)
from pymarxan.zones.readers import load_zone_project
from pymarxan.zones.solver import ZoneSASolver
from tests.pymarxan.zones.marzone_anchor import (
    HEURISTIC_SPF,
    OPTIMUM_COST,
    make_anchor_problem,
)

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "zones"


def _fixture():
    p = load_zone_project(DATA_DIR)      # BLM 1.0, bound.dat and zoneboundcost.dat present
    p.parameters["NUMITNS"] = 500
    p.parameters["NUMTEMP"] = 20
    p.parameters["ITIMPTYPE"] = 3
    assert float(p.parameters["BLM"]) > 0 and p.boundary is not None
    return p


SOLVERS = [ZoneMIPSolver(), ZoneSASolver(), ZoneHeuristicSolver(), ZoneIISolver()]


class TestCrossSolverAgreement:
    @pytest.mark.parametrize("solver", SOLVERS, ids=lambda s: s.name())
    def test_objective_equals_reference_with_blm_and_boundary(self, solver):
        p = _fixture()
        sol = solver.solve(p, SolverConfig(num_solutions=1, seed=42))[0]
        blm = float(p.parameters["BLM"])
        assert sol.objective == pytest.approx(
            compute_zone_objective(p, sol.zone_assignment, blm), abs=1e-9,
        )

    @pytest.mark.parametrize("solver", SOLVERS, ids=lambda s: s.name())
    def test_targets_met_is_feature_keyed_and_zone_keys_are_strings(self, solver):
        p = _fixture()
        sol = solver.solve(p, SolverConfig(num_solutions=1, seed=42))[0]
        assert set(sol.targets_met) == {1, 2}
        assert sol.targets_met == check_overall_targets(p, sol.zone_assignment)
        assert set(sol.metadata["zone_targets_met"]) == {"z1_f1", "z1_f2", "z2_f1", "z2_f2"}
        expected = {
            f"z{z}_f{f}": v for (z, f), v in check_zone_targets(p, sol.zone_assignment).items()
        }
        assert sol.metadata["zone_targets_met"] == expected
        assert sol.metadata["solver"] == solver.name()
        assert sol.penalty == pytest.approx(
            sol.metadata["overall_penalty"] + sol.metadata["zone_penalty"],
        )

    def test_ii_improve_uses_shared_builder(self):
        p = _fixture()
        start = ZoneHeuristicSolver().solve(p, SolverConfig(num_solutions=1))[0]
        improved = ZoneIISolver().improve(p, start)
        assert set(improved.targets_met) == {1, 2}
        assert "overall_penalty" in improved.metadata
        assert improved.objective == pytest.approx(
            compute_zone_objective(p, improved.zone_assignment, float(p.parameters["BLM"])),
        )
```

- [ ] **Step 2: Run to verify failure**

Run: `/opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/zones/test_zone_overall_targets.py tests/pymarxan/zones/test_zone_solvers_overall.py -q`
Expected: FAIL — `ImportError: cannot import name 'build_zone_solution'`; agreement tests
fail on SA (`targets_met == {}`) and heuristic/II (tuple keys, no `overall_penalty`).

- [ ] **Step 3: Implement the builder**

In `objective.py` add `from typing import Any` and `from pymarxan.solvers.base import Solution`
to the imports, and append:

```python
def build_zone_solution(
    problem: ZonalProblem,
    zone_assignment: np.ndarray,
    blm: float,
    *,
    solver_name: str,
    run: int | None = None,
) -> Solution:
    """The one Solution builder for every zone solver.

    ``targets_met`` holds the overall (contribution-weighted) feature targets, feature-keyed,
    so ``Solution.all_targets_met`` means "overall targets met" for zone runs. Zone targets
    live in ``metadata["zone_targets_met"]`` as ``"z{zone}_f{feature}" -> bool``.
    ``objective`` equals ``compute_zone_objective`` by construction.
    """
    assignment = np.asarray(zone_assignment, dtype=int)
    cost = compute_zone_cost(problem, assignment)
    std_boundary = compute_standard_boundary(problem, assignment)
    zone_boundary = compute_zone_boundary(problem, assignment)
    overall_penalty = compute_overall_penalty(problem, assignment)
    zone_penalty = compute_zone_penalty(problem, assignment)
    zone_targets_met = check_zone_targets(problem, assignment)
    metadata: dict[str, Any] = {
        "solver": solver_name,
        "zone_boundary_cost": round(zone_boundary, 4),
        "zone_targets_met": {
            f"z{z}_f{f}": bool(v) for (z, f), v in zone_targets_met.items()
        },
        "overall_penalty": overall_penalty,
        "zone_penalty": zone_penalty,
    }
    if run is not None:
        metadata["run"] = run
    return Solution(
        selected=assignment > 0,
        cost=cost,
        boundary=std_boundary,
        objective=compute_zone_objective(problem, assignment, blm),
        targets_met=check_overall_targets(problem, assignment),
        penalty=overall_penalty + zone_penalty,
        shortfall=(
            compute_overall_shortfall(problem, assignment)
            + compute_zone_shortfall(problem, assignment)
        ),
        zone_assignment=assignment.copy(),
        metadata=metadata,
    )
```

(No import cycle: `pymarxan.solvers.__init__` does not import `pymarxan.zones` at module
level — `run_mode.py` imports it lazily — verified by importing `pymarxan.solvers.base` in a
fresh interpreter and checking `sys.modules`.)

- [ ] **Step 4: Route every solver through it**

- `solver.py`: delete the imports of `check_zone_targets`, `compute_standard_boundary`,
  `compute_zone_boundary`, `compute_zone_cost`, `compute_zone_penalty`,
  `compute_zone_shortfall`; import `build_zone_solution`. All-locked branch body of the
  `for run_idx` loop becomes
  `solutions.append(build_zone_solution(problem, assignment, blm, solver_name=self.name(), run=run_idx + 1))`
  (drop the local `blm_val`). Main loop: replace lines 254–281 with
  `solutions.append(build_zone_solution(problem, best_assignment, blm, solver_name=self.name(), run=run_idx + 1))`.
- `heuristic.py`: delete the static `_build_zone_solution`; `solve` appends
  `build_zone_solution(problem, assignment, blm, solver_name=self.name(), run=run_idx + 1)`;
  trim the import list to `build_zone_solution`, `check_zone_targets`,
  `compute_zone_objective` (the early exit still uses `check_zone_targets` until Task 7).
- `iterative_improvement.py`: delete the static `_build_zone_solution`; `solve` appends
  `build_zone_solution(problem, assignment, blm, solver_name=self.name(), run=run_idx + 1)`;
  `improve` returns `build_zone_solution(problem, assignment, blm, solver_name=self.name(), run=1)`;
  imports become `build_zone_solution`, `compute_zone_objective`.
- `mip_solver.py`: delete the module-level `_build_zone_solution`; in `solve` replace the
  `sol = _build_zone_solution(problem, zone_assignment, blm)` line with
  `sol = build_zone_solution(problem, zone_assignment, blm, solver_name=self.name())` and
  drop `"solver": self.name()` from the following `metadata.update` (keep `status` and
  `mip_backend`). Remove the now-unused imports (`check_zone_targets`, `compute_zone_cost`,
  `compute_standard_boundary`, `compute_zone_objective`, `compute_zone_penalty`,
  `compute_zone_shortfall`, `Solution` if unused) — ruff F401 will list them.
- `solvers/base.py:20-21`: replace the comment with
  `# Feature ID -> met. Zone solvers report the overall (contribution-weighted) feature`
  `# targets here and the per-zone targets in metadata["zone_targets_met"].`

- [ ] **Step 5: Run the zones suite (incl. slow), the objectives tests and the parity harness**

Run: `/opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/zones tests/pymarxan/objectives tests/test_examples.py tests/test_integration_phase3.py -q`
Expected: PASS. `test_zone_mip_metadata.py` still finds `status`, `mip_backend` and the
`z{z}_f{f}` keys; `test_zone_heuristic_ii.py::test_targets_met_populated` still sees a dict.

- [ ] **Step 6: Lint, types, commit**

```bash
~/.local/bin/ruff check src/pymarxan/zones src/pymarxan/solvers/base.py tests/pymarxan/zones
PATH="/opt/micromamba/envs/shiny/bin:$HOME/.local/bin:$PWD/.venv/bin:$PATH" make types
git add src/pymarxan/zones src/pymarxan/solvers/base.py tests/pymarxan/zones/test_zone_overall_targets.py tests/pymarxan/zones/test_zone_solvers_overall.py
git commit -m "refactor(zones): one build_zone_solution for MIP, SA, heuristic and II (#2)

targets_met is feature-keyed (overall targets) in all four solvers; zone targets in
metadata[\"zone_targets_met\"]; penalty/shortfall sum both tiers; objective equals
compute_zone_objective by construction (review H3/M5).

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01X8aGAi384D5xybKRugtFXT"
```

---

### Task 6: MIP hard constraints for overall targets; the three anchor behaviours

**Files:**
- Modify: `src/pymarxan/zones/mip_solver.py` (`_add_overall_target_constraints`, `solve`)
- Create: `tests/pymarxan/zones/test_zone_mip_overall.py`

**Interfaces:**
- Consumes: `contribution_lookup()`, `contribution_default()` (Task 1); `_feature_groups` (Task 3); `build_zone_solution` (Task 5).
- Produces: `_add_overall_target_constraints(problem, x, zone_ids, model) -> None`.

- [ ] **Step 1: Write the failing tests**

```python
"""ZoneMIPSolver on the spec §5 anchor: overall targets are hard constraints (Watts eq. 6)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pymarxan.solvers.base import SolverConfig
from pymarxan.zones.mip_solver import ZoneMIPSolver
from pymarxan.zones.objective import check_overall_targets, check_zone_targets
from pymarxan.zones.readers import load_zone_project
from tests.pymarxan.zones.marzone_anchor import OPTIMUM, OPTIMUM_COST, make_anchor_problem

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "zones"
CONFIG = SolverConfig(num_solutions=1)


class TestAnchorBehaviours:
    def test_new_semantics_find_the_hard_optimum(self):
        sol = ZoneMIPSolver().solve(make_anchor_problem(spf=1.0), CONFIG)[0]
        assert tuple(int(z) for z in sol.zone_assignment) == OPTIMUM
        assert sol.cost == pytest.approx(OPTIMUM_COST)
        assert sol.objective == pytest.approx(OPTIMUM_COST)
        assert sol.targets_met == {1: True}
        assert sol.metadata["zone_targets_met"] == {"z2_f1": True}
        assert sol.penalty == 0.0
        assert sol.metadata["overall_penalty"] == 0.0

    def test_old_answer_is_no_longer_returned(self):
        """v0.35 returned (2, 2, 2) at cost 3 with targets_met {1: True} (spec §1)."""
        sol = ZoneMIPSolver().solve(make_anchor_problem(), CONFIG)[0]
        assert tuple(int(z) for z in sol.zone_assignment) != (2, 2, 2)
        assert sol.cost > 3.0

    def test_contribution_weighted_zone_targets_are_infeasible(self):
        assert ZoneMIPSolver().solve(make_anchor_problem(zone_target_contrib=1), CONFIG) == []

    def test_misslevel_relaxes_the_overall_target(self):
        # MISSLEVEL 0.8: overall target 12, zone target 8 -> (2, 2, 2) at cost 3 is feasible.
        sol = ZoneMIPSolver().solve(make_anchor_problem(misslevel=0.8), CONFIG)[0]
        assert tuple(int(z) for z in sol.zone_assignment) == (2, 2, 2)
        assert sol.cost == pytest.approx(3.0)
        assert sol.targets_met == {1: True}

    def test_spf_does_not_change_the_hard_optimum(self):
        sol = ZoneMIPSolver().solve(make_anchor_problem(spf=10.0), CONFIG)[0]
        assert tuple(int(z) for z in sol.zone_assignment) == OPTIMUM

    def test_feature_with_target_but_no_amounts_is_infeasible(self):
        p = make_anchor_problem()
        features = pd.concat([
            p.features,
            pd.DataFrame({"id": [2], "name": ["ghost"], "target": [5.0], "spf": [1.0]}),
        ], ignore_index=True)
        p = p.copy_with(features=features)
        assert ZoneMIPSolver().solve(p, CONFIG) == []

    def test_inert_overall_target_adds_no_constraint(self):
        p = make_anchor_problem()
        p.features.loc[0, "target"] = 0.0
        sol = ZoneMIPSolver().solve(p, CONFIG)[0]
        # Only the raw zone target (zone 2 >= 10) binds: one PU in zone 2 at cost 1.
        assert sol.cost == pytest.approx(1.0)
        assert sol.targets_met == {1: True}


class TestFixtureUnderNewSemantics:
    def test_fixture_is_feasible_and_both_tiers_met(self):
        p = load_zone_project(DATA_DIR)
        sols = ZoneMIPSolver().solve(p, CONFIG)
        assert len(sols) == 1
        sol = sols[0]
        assert sol.all_targets_met
        assert all(sol.metadata["zone_targets_met"].values())
        assert check_overall_targets(p, sol.zone_assignment) == {1: True, 2: True}
        assert all(check_zone_targets(p, sol.zone_assignment).values())

    def test_witness_assignment_is_feasible_by_hand(self):
        """(1, 1, 2, 1): overall f1 = 23 + 3 = 26 >= 20, f2 = 16 + 2.7 = 18.7 >= 15;
        zone 1 raw (23, 16) >= (10, 8); zone 2 raw (6, 9) >= (5, 3)."""
        p = load_zone_project(DATA_DIR)
        a = np.array([1, 1, 2, 1])
        assert check_overall_targets(p, a) == {1: True, 2: True}
        assert all(check_zone_targets(p, a).values())
```

- [ ] **Step 2: Run to verify failure**

Run: `/opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/zones/test_zone_mip_overall.py -q`
Expected: FAIL — `test_new_semantics_find_the_hard_optimum` gets (2, 2, 2) at cost 3;
`test_contribution_weighted_zone_targets_are_infeasible` gets a solution;
`test_feature_with_target_but_no_amounts_is_infeasible` gets a solution.

- [ ] **Step 3: Implement**

Add to `mip_solver.py` after `_add_zone_target_constraints`:

```python
def _add_overall_target_constraints(
    problem: ZonalProblem,
    x: dict[tuple[int, int], pulp.LpVariable],
    zone_ids: list[int],
    model: pulp.LpProblem,
) -> None:
    """Hard constraints Σ_i Σ_z amount[i, f] × contribution[f, z] × x[i, z] >= target_f × MISSLEVEL.

    Watts et al. 2009 eq. 6; MarZone ``reserve.hpp:158-171``. One constraint per feature with
    ``target > 0``; a feature with no amounts anywhere makes the model infeasible (the solver
    then returns ``[]``), which is the honest answer rather than a silently met target.
    """
    lookup = problem.contribution_lookup()
    default = problem.contribution_default()
    misslevel = float(problem.parameters.get("MISSLEVEL", 1.0))
    groups = _feature_groups(problem)
    for fid, target in zip(
        problem.features["id"].values, problem.features["target"].values, strict=True,
    ):
        fid = int(fid)
        t = float(target)
        if t <= 0:
            continue
        terms = []
        for pid, amt in groups.get(fid, []):
            for zid in zone_ids:
                c = lookup.get((fid, zid), default)
                if c != 0.0 and (pid, zid) in x:
                    terms.append(amt * c * x[(pid, zid)])
        model += (pulp.lpSum(terms) >= t * misslevel, f"overall_target_{fid}")
```

In `solve`, right after `_add_zone_target_constraints(problem, zone_exprs, model)`:

```python
        _add_overall_target_constraints(problem, x, zone_ids, model)
```

Update the class docstring "Constraints" list: add
`overall feature targets on contribution-weighted amounts (hard; Watts eq. 6)` and change
`zone-specific feature targets with contributions` to
`zone-specific feature targets on raw amounts (hard + slack; weighted under ZONETARGETCONTRIB=1)`.

- [ ] **Step 4: Run the tests**

Run: `/opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/zones -q`
Expected: PASS (all of `test_zone_mip.py`, `test_zone_mip_metadata.py` included).

- [ ] **Step 5: Lint and commit**

```bash
~/.local/bin/ruff check src/pymarxan/zones/mip_solver.py tests/pymarxan/zones/test_zone_mip_overall.py
git add src/pymarxan/zones/mip_solver.py tests/pymarxan/zones/test_zone_mip_overall.py
git commit -m "feat(zones): ZoneMIPSolver enforces overall feature targets as hard constraints (#2)

Spec §5 anchor: (1, 2, 2) at cost 7 (was (2, 2, 2) at 3 with the target ignored);
ZONETARGETCONTRIB=1 infeasible -> []; MISSLEVEL relaxes; zones fixture stays feasible.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01X8aGAi384D5xybKRugtFXT"
```

---

### Task 7: Heuristic — remove the zone-target early exit; anchor run

**Files:**
- Modify: `src/pymarxan/zones/heuristic.py` (`_greedy_assign` lines 116–119, class docstring, imports)
- Test: `tests/pymarxan/zones/test_zone_solvers_overall.py`

**Interfaces:**
- Consumes: `build_zone_solution` (Task 5), `compute_zone_objective` with both tiers (Task 4).
- Produces: no new API; `_greedy_assign` ends only when no move improves the objective.

- [ ] **Step 1: Write the failing tests**

Append to `tests/pymarxan/zones/test_zone_solvers_overall.py`:

```python
class TestHeuristicOnAnchor:
    def test_greedy_meets_both_tiers_and_lands_at_or_above_optimum(self):
        """Hand trace at spf = 10 from (0, 0, 0): PU1->z2 (obj 111), PU2->z1 (17),
        PU3->z2 (8); no further improving move. v0.35 stopped at (2, 0, 0) cost 1 because
        the zone target was already met (review H2)."""
        p = make_anchor_problem(spf=HEURISTIC_SPF)
        sol = ZoneHeuristicSolver().solve(p, SolverConfig(num_solutions=1))[0]
        assert sol.targets_met == {1: True}
        assert sol.metadata["zone_targets_met"] == {"z2_f1": True}
        assert sol.cost >= OPTIMUM_COST
        assert tuple(int(z) for z in sol.zone_assignment) == (2, 1, 2)
        assert sol.cost == pytest.approx(8.0)
        assert sol.penalty == 0.0

    def test_greedy_no_longer_stops_when_only_zone_targets_are_met(self):
        p = make_anchor_problem(spf=HEURISTIC_SPF)
        sol = ZoneHeuristicSolver().solve(p, SolverConfig(num_solutions=1))[0]
        assert tuple(int(z) for z in sol.zone_assignment) != (2, 0, 0)

    def test_feature_without_amounts_is_penalised_not_fatal(self):
        import pandas as pd
        p = make_anchor_problem(spf=HEURISTIC_SPF)
        features = pd.concat([
            p.features,
            pd.DataFrame({"id": [2], "name": ["ghost"], "target": [5.0], "spf": [1.0]}),
        ], ignore_index=True)
        p = p.copy_with(features=features)
        sol = ZoneHeuristicSolver().solve(p, SolverConfig(num_solutions=1))[0]
        assert sol.targets_met == {1: True, 2: False}
        assert sol.penalty == pytest.approx(5.0)
        assert sol.all_targets_met is False
```

- [ ] **Step 2: Run to verify failure**

Run: `/opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/zones/test_zone_solvers_overall.py -q -k Heuristic`
Expected: FAIL — the first two tests get assignment (2, 0, 0), cost 1, `targets_met {1: False}`.

- [ ] **Step 3: Remove the early exit**

In `heuristic.py::_greedy_assign` delete

```python
            # Check if all targets met
            targets = check_zone_targets(problem, assignment)
            if targets and all(targets.values()):
                break
```

drop `check_zone_targets` from the imports, and change the class docstring's last sentence to
"Stops when no candidate move improves the objective (both target tiers are part of the
objective, so meeting the zone targets alone no longer ends the search)."

- [ ] **Step 4: Run the tests**

Run: `/opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/zones/test_zone_solvers_overall.py tests/pymarxan/zones/test_zone_heuristic_ii.py -q`
Expected: PASS.

- [ ] **Step 5: Lint and commit**

```bash
~/.local/bin/ruff check src/pymarxan/zones/heuristic.py tests/pymarxan/zones/test_zone_solvers_overall.py
git add src/pymarxan/zones/heuristic.py tests/pymarxan/zones/test_zone_solvers_overall.py
git commit -m "fix(zones): greedy zone heuristic no longer exits when only zone targets are met (#2)

The loop already ends when no move improves the objective; the early exit ignored the
overall tier (review H2). Anchor: (2, 1, 2) at cost 8 with both tiers met.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01X8aGAi384D5xybKRugtFXT"
```

---

### Task 8: II and SA on the anchor

**Files:**
- Test: `tests/pymarxan/zones/test_zone_solvers_overall.py`
- Modify (docstring only): `src/pymarxan/zones/iterative_improvement.py` class docstring

**Interfaces:**
- Consumes: Tasks 4, 5, 7. No production code change beyond a docstring; these tests pin
  behaviour that the earlier tasks produced.

- [ ] **Step 1: Write the tests**

Append to `tests/pymarxan/zones/test_zone_solvers_overall.py`:

```python
class TestIIOnAnchor:
    def test_swap_pass_meets_both_tiers(self):
        """ITIMPTYPE 3 from the all-first-zone start (1, 1, 1): PU1->z2 (obj 14),
        PU2->z2 (9); second sweep finds nothing. Ends at (2, 2, 1), cost 9."""
        p = make_anchor_problem(spf=HEURISTIC_SPF)
        p.parameters["ITIMPTYPE"] = 3
        sol = ZoneIISolver().solve(p, SolverConfig(num_solutions=1))[0]
        assert sol.targets_met == {1: True}
        assert sol.metadata["zone_targets_met"] == {"z2_f1": True}
        assert sol.cost >= OPTIMUM_COST
        assert tuple(int(z) for z in sol.zone_assignment) == (2, 2, 1)
        assert sol.cost == pytest.approx(9.0)

    def test_itimptype_zero_returns_the_start_unchanged(self):
        p = make_anchor_problem(spf=HEURISTIC_SPF)
        p.parameters["ITIMPTYPE"] = 0
        sol = ZoneIISolver().solve(p, SolverConfig(num_solutions=1))[0]
        assert tuple(int(z) for z in sol.zone_assignment) == (1, 1, 1)
        assert sol.targets_met == {1: True}                       # 30 >= 15
        assert sol.metadata["zone_targets_met"] == {"z2_f1": False}
        assert sol.penalty == pytest.approx(HEURISTIC_SPF * 10.0)

    def test_improve_from_infeasible_corner(self):
        from pymarxan.zones.objective import build_zone_solution
        p = make_anchor_problem(spf=HEURISTIC_SPF)
        p.parameters["ITIMPTYPE"] = 3
        start = build_zone_solution(p, np.array([2, 2, 2]), 0.0, solver_name="seed")
        improved = ZoneIISolver().improve(p, start)
        assert improved.targets_met == {1: True}
        assert improved.objective < start.objective


class TestSAOnAnchor:
    def test_sa_reaches_the_optimum_with_both_tiers_met(self):
        p = make_anchor_problem(spf=HEURISTIC_SPF)
        p.parameters["NUMITNS"] = 3000
        p.parameters["NUMTEMP"] = 100
        sols = ZoneSASolver().solve(p, SolverConfig(num_solutions=3, seed=42))
        for sol in sols:
            assert sol.targets_met == {1: True}
            assert sol.metadata["zone_targets_met"] == {"z2_f1": True}
            assert sol.cost >= OPTIMUM_COST
            assert sol.objective == pytest.approx(
                compute_zone_objective(p, sol.zone_assignment, 0.0),
            )
        assert min(s.cost for s in sols) == pytest.approx(OPTIMUM_COST)

    def test_sa_all_locked_in_uses_shared_builder(self):
        p = make_anchor_problem(spf=HEURISTIC_SPF)
        p.planning_units["status"] = 2                     # all locked into zone 1
        p.parameters["NUMITNS"] = 100
        sol = ZoneSASolver().solve(p, SolverConfig(num_solutions=1, seed=1))[0]
        assert tuple(int(z) for z in sol.zone_assignment) == (1, 1, 1)
        assert sol.targets_met == {1: True}
        assert "overall_penalty" in sol.metadata
        assert sol.objective == pytest.approx(compute_zone_objective(p, sol.zone_assignment, 0.0))
```

- [ ] **Step 2: Run them (they should pass already; if any fails, that is a real regression from Tasks 4–7 — fix there, do not edit the expected values)**

Run: `/opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/zones/test_zone_solvers_overall.py -q`
Expected: PASS. If `test_swap_pass_meets_both_tiers` ends elsewhere than (2, 2, 1), report the
assignment and objective trace in the task report instead of changing the pin.

- [ ] **Step 3: Docstring**

In `iterative_improvement.py` add to the class docstring after the mode list:
"Both target tiers are part of the objective; mode 0 returns the start assignment unchanged
(anchor tests set ITIMPTYPE 3)."

- [ ] **Step 4: Lint and commit**

```bash
~/.local/bin/ruff check src/pymarxan/zones/iterative_improvement.py tests/pymarxan/zones/test_zone_solvers_overall.py
git add src/pymarxan/zones/iterative_improvement.py tests/pymarxan/zones/test_zone_solvers_overall.py
git commit -m "test(zones): II (ITIMPTYPE 3 / 0) and SA on the MarZone anchor (#2)

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01X8aGAi384D5xybKRugtFXT"
```

---

### Task 9: Route the `objectives/` zone achieved through `compute_overall_achieved`

**Files:**
- Modify: `src/pymarxan/objectives/min_shortfall.py:199-224`, `src/pymarxan/objectives/max_coverage.py` (`_compute_zone_achieved`)
- Test: `tests/pymarxan/objectives/test_objectives.py`

**Interfaces:**
- Consumes: `compute_overall_achieved(problem, assignment, *, amounts=...)` (Task 2).
- Produces: `MinShortfallObjective.compute_zone_score` and `MaxCoverageObjective.compute_zone_score`
  now apply contributions (today the lookup is keyed backwards on a DataFrame and always uses 1.0).

- [ ] **Step 1: Write the failing tests**

Append to `tests/pymarxan/objectives/test_objectives.py`:

```python
class TestZoneScoresApplyContributions:
    """Review M4: the objectives' zone achieved keyed the lookup backwards and used 1.0."""

    def _anchor(self):
        from tests.pymarxan.zones.marzone_anchor import make_anchor_problem
        p = make_anchor_problem()
        amounts = p.build_pu_feature_matrix()
        pu_index = p.pu_id_to_index()
        return p, amounts, pu_index

    def test_min_shortfall_zone_score_uses_contribution_weighted_amounts(self):
        from pymarxan.objectives.min_shortfall import MinShortfallObjective
        p, amounts, pu_index = self._anchor()
        # (2, 2, 2): A_f = 12 < 15 -> shortfall 3 (was 0: 30 raw)
        score = MinShortfallObjective().compute_zone_score(
            p, np.array([2, 2, 2]), amounts, pu_index,
        )
        assert score == pytest.approx(3.0)

    def test_max_coverage_zone_score_uses_contribution_weighted_amounts(self):
        from pymarxan.objectives.max_coverage import MaxCoverageObjective
        p, amounts, pu_index = self._anchor()
        # coverage min(A_f, target) = min(12, 15) = 12, negated
        score = MaxCoverageObjective().compute_zone_score(
            p, np.array([2, 2, 2]), amounts, pu_index,
        )
        assert score == pytest.approx(-12.0)

    def test_effective_amounts_are_honoured(self):
        from pymarxan.objectives.min_shortfall import MinShortfallObjective
        p, amounts, pu_index = self._anchor()
        halved = amounts * 0.5   # e.g. probability-adjusted
        score = MinShortfallObjective().compute_zone_score(
            p, np.array([1, 1, 1]), halved, pu_index,
        )
        assert score == pytest.approx(0.0)      # 15 >= 15
        score = MinShortfallObjective().compute_zone_score(
            p, np.array([1, 1, 0]), halved, pu_index,
        )
        assert score == pytest.approx(5.0)      # 10 < 15
```

(`np` and `pytest` are imported at the top of that file; verify, and add `import numpy as np`
at the top if not.) `pu_id_to_index` is a method on `ConservationProblem`
(`models/problem.py:87`); if it is a property there, drop the parentheses.

- [ ] **Step 2: Run to verify failure**

Run: `/opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/objectives/test_objectives.py -q -k ZoneScores`
Expected: FAIL — min-shortfall score 0.0, max-coverage −15.0.

- [ ] **Step 3: Implement**

In both files replace the body of the static `_compute_zone_achieved` with:

```python
        """Contribution-weighted achieved amount per feature (MarZone reserve.hpp:158-171).

        Delegates to ``pymarxan.zones.objective.compute_overall_achieved`` with the caller's
        (possibly probability-adjusted) ``effective_amounts``; ``pu_index`` is kept for
        interface symmetry with ``_compute_achieved``.
        """
        from pymarxan.zones.objective import compute_overall_achieved

        return compute_overall_achieved(problem, assignment, amounts=effective_amounts)
```

(Local import: `objectives` must not import `zones` at module level — `zones.objective`
imports `solvers.base`, and `objectives` is imported by `solvers.mip_solver`.)

- [ ] **Step 4: Run the objectives tests**

Run: `/opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/objectives -q`
Expected: PASS.

- [ ] **Step 5: Lint and commit**

```bash
~/.local/bin/ruff check src/pymarxan/objectives tests/pymarxan/objectives/test_objectives.py
git add src/pymarxan/objectives/min_shortfall.py src/pymarxan/objectives/max_coverage.py tests/pymarxan/objectives/test_objectives.py
git commit -m "fix(objectives): zone achieved uses the shared contribution-weighted accumulator (#2)

The (z, fid)-keyed .get on a DataFrame silently returned 1.0 for every pair.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01X8aGAi384D5xybKRugtFXT"
```

---

### Task 10: `write_zone_summary` rewritten on the shared helpers

**Files:**
- Modify: `src/pymarxan/zones/writers.py:111-186`
- Test: `tests/pymarxan/zones/test_writers.py` (replace the three tests that call `write_zone_summary`: the column/row-count test around line 255, `test_summary_times_met`, `test_summary_multiple_solutions`)

**Interfaces:**
- Consumes: `compute_overall_achieved`, `check_overall_targets`, `_compute_zone_achieved`, `check_zone_targets` (Tasks 2–3).
- Produces: CSV columns `tier, zone, feature, target, mean_achieved, times_met, total_runs`;
  `tier == "overall"` rows (zone 0, one per feature) then `tier == "zone"` rows (one per listed
  zone target). Ruling R5.

- [ ] **Step 1: Replace the three existing summary tests with these (failing first)**

```python
class TestWriteZoneSummary:
    """Rows equal the objective module's numbers (review H5: the old writer matched neither tier)."""

    def _problem_and_solutions(self):
        problem = load_zone_project(DATA_DIR)
        sols = [
            _make_solution([True, True, True, True], [1, 1, 2, 1]),
            _make_solution([True, True, True, True], [2, 2, 2, 2]),
        ]
        return problem, sols

    def test_columns_and_row_groups(self, tmp_path: Path) -> None:
        problem, sols = self._problem_and_solutions()
        out = tmp_path / "zone_summary.csv"
        write_zone_summary(problem, sols, out)
        df = pd.read_csv(out)
        assert list(df.columns) == [
            "tier", "zone", "feature", "target", "mean_achieved", "times_met", "total_runs",
        ]
        assert (df["tier"] == "overall").sum() == 2          # one per feature
        assert (df["tier"] == "zone").sum() == 4             # one per listed zone target
        assert (df["total_runs"] == 2).all()
        assert (df.loc[df["tier"] == "overall", "zone"] == 0).all()

    def test_overall_rows_match_objective_module(self, tmp_path: Path) -> None:
        from pymarxan.zones.objective import check_overall_targets, compute_overall_achieved
        problem, sols = self._problem_and_solutions()
        out = tmp_path / "zone_summary.csv"
        write_zone_summary(problem, sols, out)
        df = pd.read_csv(out)
        for fid in (1, 2):
            row = df[(df["tier"] == "overall") & (df["feature"] == fid)].iloc[0]
            achieved = [compute_overall_achieved(problem, s.zone_assignment)[fid] for s in sols]
            met = [check_overall_targets(problem, s.zone_assignment)[fid] for s in sols]
            assert row["mean_achieved"] == pytest.approx(sum(achieved) / 2)
            assert row["times_met"] == sum(met)
            assert row["target"] == float(problem.features.set_index("id").loc[fid, "target"])

    def test_zone_rows_match_objective_module(self, tmp_path: Path) -> None:
        from pymarxan.zones.objective import _compute_zone_achieved, check_zone_targets
        problem, sols = self._problem_and_solutions()
        out = tmp_path / "zone_summary.csv"
        write_zone_summary(problem, sols, out)
        df = pd.read_csv(out)
        for zid, fid, target in problem.zone_targets[["zone", "feature", "target"]].itertuples(
            index=False,
        ):
            row = df[(df["tier"] == "zone") & (df["zone"] == zid) & (df["feature"] == fid)].iloc[0]
            achieved = [
                _compute_zone_achieved(problem, s.zone_assignment).get((zid, fid), 0.0)
                for s in sols
            ]
            met = [check_zone_targets(problem, s.zone_assignment)[(zid, fid)] for s in sols]
            assert row["mean_achieved"] == pytest.approx(sum(achieved) / 2)
            assert row["times_met"] == sum(met)
            assert row["target"] == float(target)

    def test_zone_rows_are_raw_by_default(self, tmp_path: Path) -> None:
        problem, sols = self._problem_and_solutions()
        out = tmp_path / "zone_summary.csv"
        write_zone_summary(problem, sols, out)
        df = pd.read_csv(out)
        row = df[(df["tier"] == "zone") & (df["zone"] == 2) & (df["feature"] == 1)].iloc[0]
        # run 1: PU3 raw 6; run 2: all four raw 29 -> mean 17.5 (weighted would be 8.75)
        assert row["mean_achieved"] == pytest.approx(17.5)

    def test_no_zone_targets_gives_overall_rows_only(self, tmp_path: Path) -> None:
        problem, sols = self._problem_and_solutions()
        problem = problem.copy_with(zone_targets=None)
        out = tmp_path / "zone_summary.csv"
        write_zone_summary(problem, sols, out)
        df = pd.read_csv(out)
        assert set(df["tier"]) == {"overall"}

    def test_selected_only_solution_counts_as_first_zone(self, tmp_path: Path) -> None:
        problem, _ = self._problem_and_solutions()
        sol = _make_solution([True, True, False, False], None)
        out = tmp_path / "zone_summary.csv"
        write_zone_summary(problem, [sol], out)
        df = pd.read_csv(out)
        row = df[(df["tier"] == "zone") & (df["zone"] == 1) & (df["feature"] == 1)].iloc[0]
        assert row["mean_achieved"] == pytest.approx(18.0)   # PU1 10 + PU2 8, raw
```

`pytest`, `pd`, `Path`, `DATA_DIR`, `_make_solution` and `load_zone_project` already exist in
`test_writers.py` (check the import block; add `from pymarxan.zones.readers import
load_zone_project` and `import pytest` at the top if absent).

- [ ] **Step 2: Run to verify failure**

Run: `/opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/zones/test_writers.py -q -k WriteZoneSummary`
Expected: FAIL — column set mismatch (`tier` missing).

- [ ] **Step 3: Rewrite the writer**

Add to the imports of `writers.py` (module level, outside `TYPE_CHECKING`):

```python
from pymarxan.zones.objective import (
    _compute_zone_achieved,
    check_overall_targets,
    check_zone_targets,
    compute_overall_achieved,
)
```

and replace `write_zone_summary` entirely:

```python
_SUMMARY_COLUMNS = [
    "tier", "zone", "feature", "target", "mean_achieved", "times_met", "total_runs",
]


def write_zone_summary(
    problem: ZonalProblem,
    solutions: list[Solution],
    path: str | Path,
) -> None:
    """Target-achievement summary across runs, computed by the objective module.

    Two row groups share one CSV (columns ``tier, zone, feature, target, mean_achieved,
    times_met, total_runs``):

    - ``tier == "overall"`` — one row per feature, ``zone`` 0: ``target`` is
      ``features.target``; ``mean_achieved`` the contribution-weighted amount
      (``compute_overall_achieved``); ``times_met`` counts runs where
      ``check_overall_targets`` is True.
    - ``tier == "zone"`` — one row per listed zone target: raw amounts by default
      (contribution-weighted under ``ZONETARGETCONTRIB = 1``), met per ``check_zone_targets``.

    Solutions without ``zone_assignment`` are read as "selected PUs in the first zone".
    """
    zone_ids = sorted(int(z) for z in problem.zones["id"].tolist())
    first_zone = zone_ids[0] if zone_ids else 1
    assignments = [
        np.asarray(sol.zone_assignment, dtype=int)
        if sol.zone_assignment is not None
        else np.where(np.asarray(sol.selected, dtype=bool), first_zone, 0)
        for sol in solutions
    ]
    total_runs = len(assignments)

    feat_ids = [int(f) for f in problem.features["id"].tolist()]
    feat_targets = dict(
        zip(feat_ids, problem.features["target"].astype(float).tolist(), strict=True)
    )
    overall_achieved = [compute_overall_achieved(problem, a) for a in assignments]
    overall_met = [check_overall_targets(problem, a) for a in assignments]

    rows: list[dict[str, object]] = []
    for fid in feat_ids:
        rows.append({
            "tier": "overall",
            "zone": 0,
            "feature": fid,
            "target": feat_targets[fid],
            "mean_achieved": (
                sum(a.get(fid, 0.0) for a in overall_achieved) / total_runs
                if total_runs else 0.0
            ),
            "times_met": sum(int(m.get(fid, False)) for m in overall_met),
            "total_runs": total_runs,
        })

    if problem.zone_targets is not None:
        zone_achieved = [_compute_zone_achieved(problem, a) for a in assignments]
        zone_met = [check_zone_targets(problem, a) for a in assignments]
        zt = problem.zone_targets
        for zid, fid, target in zip(
            zt["zone"].values, zt["feature"].values, zt["target"].values, strict=True,
        ):
            key = (int(zid), int(fid))
            rows.append({
                "tier": "zone",
                "zone": key[0],
                "feature": key[1],
                "target": float(target),
                "mean_achieved": (
                    sum(a.get(key, 0.0) for a in zone_achieved) / total_runs
                    if total_runs else 0.0
                ),
                "times_met": sum(int(m.get(key, False)) for m in zone_met),
                "total_runs": total_runs,
            })

    pd.DataFrame(rows, columns=_SUMMARY_COLUMNS).to_csv(path, index=False)
```

The `TYPE_CHECKING` import of `ZonalProblem` can stay; `Solution` too.

- [ ] **Step 4: Run the writer tests and the Shiny modules that consume the summary**

Run: `/opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/zones/test_writers.py -q` then
`grep -rn "write_zone_summary\|zone_summary" src/pymarxan_shiny src/pymarxan_app tests --include=*.py`
Expected: PASS; if any Shiny module reads the old column set (`zone, feature, target,
times_met, total_runs` with a row per zone × feature), update it to filter `tier == "zone"`
and run its tests.

- [ ] **Step 5: Lint and commit**

```bash
~/.local/bin/ruff check src/pymarxan/zones/writers.py tests/pymarxan/zones/test_writers.py
git add src/pymarxan/zones/writers.py tests/pymarxan/zones/test_writers.py
git commit -m "fix(zones): write_zone_summary reports both target tiers from the objective module (#2)

The old writer scored per-zone contribution-weighted amounts against features.target,
matching neither tier (review H5, pre-existing). New columns: tier, zone, feature, target,
mean_achieved, times_met, total_runs.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01X8aGAi384D5xybKRugtFXT"
```

---

### Task 11: `zonetarget.dat` `targettype` column; `ZONETARGETCONTRIB` round trip

**Files:**
- Modify: `src/pymarxan/zones/readers.py` (`read_zone_targets`, new `resolve_zone_target_types`, `load_zone_project`)
- Test: `tests/pymarxan/zones/test_readers.py`

**Interfaces:**
- Consumes: `write_zone_targets` (existing), `io.writers.write_input_dat` / `save_project`, `io.readers.read_input_dat`.
- Produces: `read_zone_targets(path)` keeps an optional int `targettype` column (0/1 pass, 2/3
  raise); `resolve_zone_target_types(zone_targets, pu_vs_features) -> pd.DataFrame` (type 1 →
  target × feature total raw amount, row becomes type 0); `load_zone_project` applies it.
  Rulings R1 and R2.

- [ ] **Step 1: Write the failing tests**

Append to `tests/pymarxan/zones/test_readers.py` (it already imports `Path`, `pd`, `pytest`,
`DATA_DIR`/`INPUT_DIR` — check the header and add any of these that are missing at the top):

```python
from pymarxan.io.readers import read_input_dat
from pymarxan.io.writers import save_project, write_input_dat
from pymarxan.zones.readers import (
    load_zone_project,
    read_zone_targets,
    resolve_zone_target_types,
)
from pymarxan.zones.writers import (
    write_zone_boundary_costs,
    write_zone_contributions,
    write_zone_costs,
    write_zone_targets,
    write_zones,
)


def _write_zonetarget(tmp_path: Path, rows: str) -> Path:
    path = tmp_path / "zonetarget.dat"
    path.write_text("zone,feature,target,targettype\n" + rows)
    return path


class TestZoneTargetType:
    def test_type_zero_is_unchanged_and_column_kept(self, tmp_path: Path):
        df = read_zone_targets(_write_zonetarget(tmp_path, "1,1,10.0,0\n2,2,3.0,0\n"))
        assert list(df["targettype"]) == [0, 0]
        assert list(df["target"]) == [10.0, 3.0]

    def test_type_one_resolves_to_fraction_of_feature_total(self, tmp_path: Path):
        problem = load_zone_project(DATA_DIR)          # feature 1 total raw amount = 29
        df = read_zone_targets(_write_zonetarget(tmp_path, "1,1,0.5,1\n2,2,3.0,0\n"))
        out = resolve_zone_target_types(df, problem.pu_vs_features)
        assert out.loc[0, "target"] == pytest.approx(14.5)
        assert out.loc[0, "targettype"] == 0                # resolved rows become type 0
        assert out.loc[1, "target"] == 3.0
        assert list(df["target"]) == [0.5, 3.0]             # input not mutated

    def test_resolution_is_idempotent_across_write_and_read(self, tmp_path: Path):
        problem = load_zone_project(DATA_DIR)
        df = read_zone_targets(_write_zonetarget(tmp_path, "1,1,0.5,1\n"))
        once = resolve_zone_target_types(df, problem.pu_vs_features)
        write_zone_targets(once, tmp_path / "again.dat")
        twice = resolve_zone_target_types(
            read_zone_targets(tmp_path / "again.dat"), problem.pu_vs_features,
        )
        assert twice.loc[0, "target"] == pytest.approx(14.5)

    def test_missing_column_passes_through(self):
        problem = load_zone_project(DATA_DIR)
        out = resolve_zone_target_types(problem.zone_targets, problem.pu_vs_features)
        pd.testing.assert_frame_equal(out, problem.zone_targets)

    @pytest.mark.parametrize("ttype", [2, 3])
    def test_occurrence_types_are_rejected_naming_the_row(self, tmp_path: Path, ttype: int):
        path = _write_zonetarget(tmp_path, f"1,1,10.0,0\n2,1,4,{ttype}\n")
        with pytest.raises(ValueError, match=r"row 2 .*zone 2.*feature 1.*targettype " + str(ttype)):
            read_zone_targets(path)

    def test_unknown_type_is_rejected(self, tmp_path: Path):
        with pytest.raises(ValueError, match="targettype 7"):
            read_zone_targets(_write_zonetarget(tmp_path, "1,1,10.0,7\n"))

    def test_load_zone_project_applies_resolution(self, tmp_path: Path):
        _copy_zone_project(tmp_path)
        (tmp_path / "input" / "zonetarget.dat").write_text(
            "zone,feature,target,targettype\n1,1,0.5,1\n2,2,3.0,0\n"
        )
        problem = load_zone_project(tmp_path)
        assert problem.zone_targets.loc[0, "target"] == pytest.approx(14.5)


def _copy_zone_project(dest: Path) -> None:
    """Write the zones fixture into ``dest`` with the existing writers (ruling R1: there is no
    save_zone_project yet)."""
    problem = load_zone_project(DATA_DIR)
    save_project(problem, dest)                                   # base files + input.dat
    input_dir = dest / "input"
    write_zones(problem.zones, input_dir / "zones.dat")
    write_zone_costs(problem.zone_costs, input_dir / "zonecost.dat")
    write_zone_contributions(problem.zone_contributions, input_dir / "zonecontrib.dat")
    write_zone_targets(problem.zone_targets, input_dir / "zonetarget.dat")
    write_zone_boundary_costs(problem.zone_boundary_costs, input_dir / "zoneboundcost.dat")


class TestZoneTargetContribRoundTrip:
    def test_input_dat_round_trips_as_int(self, tmp_path: Path):
        write_input_dat({"ZONETARGETCONTRIB": 1, "BLM": 1.0}, tmp_path / "input.dat")
        params = read_input_dat(tmp_path / "input.dat")
        assert params["ZONETARGETCONTRIB"] == 1
        assert isinstance(params["ZONETARGETCONTRIB"], int)

    def test_full_project_round_trip_keeps_the_parameter(self, tmp_path: Path):
        problem = load_zone_project(DATA_DIR)
        problem.parameters["ZONETARGETCONTRIB"] = 1
        save_project(problem, tmp_path)
        input_dir = tmp_path / "input"
        write_zones(problem.zones, input_dir / "zones.dat")
        write_zone_costs(problem.zone_costs, input_dir / "zonecost.dat")
        write_zone_contributions(problem.zone_contributions, input_dir / "zonecontrib.dat")
        write_zone_targets(problem.zone_targets, input_dir / "zonetarget.dat")
        write_zone_boundary_costs(problem.zone_boundary_costs, input_dir / "zoneboundcost.dat")
        again = load_zone_project(tmp_path)
        assert again.zone_target_contrib() == 1
        assert again.validate() == []

    def test_out_of_domain_value_loads_but_fails_validation(self, tmp_path: Path):
        _copy_zone_project(tmp_path)
        text = (tmp_path / "input.dat").read_text()
        (tmp_path / "input.dat").write_text(text + "ZONETARGETCONTRIB 2\n")
        problem = load_zone_project(tmp_path)
        assert any("ZONETARGETCONTRIB" in e for e in problem.validate())
```

(`test_full_project_round_trip_keeps_the_parameter` inlines the writer calls instead of using
`_copy_zone_project` because it mutates `parameters` before saving.)

- [ ] **Step 2: Run to verify failure**

Run: `/opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/zones/test_readers.py -q`
Expected: FAIL at import — `cannot import name 'resolve_zone_target_types'`.

- [ ] **Step 3: Implement**

In `readers.py`:

```python
_ZONE_TARGET_TYPES_SUPPORTED = {0, 1}
_ZONE_TARGET_TYPES_KNOWN = {0, 1, 2, 3}


def read_zone_targets(path: str | Path) -> pd.DataFrame:
    """Read ``zonetarget.dat`` (``zone, feature, target[, targettype]``).

    ``targettype`` follows MarZone ``zones.hpp:96-160``: 0 = amount (default), 1 = proportion
    of the feature's total raw amount (resolved by :func:`resolve_zone_target_types`),
    2/3 = occurrence targets, which pymarxan does not support and rejects by row.
    """
    df = _read_dat(path)
    df["zone"] = df["zone"].astype(int)
    df["feature"] = df["feature"].astype(int)
    df["target"] = df["target"].astype(float)
    if "targettype" in df.columns:
        df["targettype"] = df["targettype"].fillna(0).astype(int)
        for pos, (zid, fid, ttype) in enumerate(
            zip(df["zone"].values, df["feature"].values, df["targettype"].values, strict=True),
            start=1,
        ):
            if int(ttype) not in _ZONE_TARGET_TYPES_KNOWN:
                raise ValueError(
                    f"{path}: row {pos} (zone {int(zid)}, feature {int(fid)}) has "
                    f"targettype {int(ttype)}; known types are 0-3"
                )
            if int(ttype) not in _ZONE_TARGET_TYPES_SUPPORTED:
                raise ValueError(
                    f"{path}: row {pos} (zone {int(zid)}, feature {int(fid)}) has "
                    f"targettype {int(ttype)}: occurrence targets (MarZone types 2/3) are "
                    "not supported"
                )
    return df


def resolve_zone_target_types(
    zone_targets: pd.DataFrame,
    pu_vs_features: pd.DataFrame,
) -> pd.DataFrame:
    """Resolve ``targettype == 1`` rows: target × the feature's total raw amount.

    MarZone ``zones.hpp:149-150``. Resolved rows are rewritten as type 0 so that a
    write → read cycle does not multiply again (same idempotence contract as
    ``io.readers._resolve_prop_targets``). Frames without the column pass through.
    """
    if "targettype" not in zone_targets.columns:
        return zone_targets
    df = zone_targets.copy()
    totals = pu_vs_features.groupby("species")["amount"].sum()
    is_prop = df["targettype"] == 1
    if is_prop.any():
        feature_totals = df.loc[is_prop, "feature"].map(totals).fillna(0.0)
        df.loc[is_prop, "target"] = df.loc[is_prop, "target"] * feature_totals
        df.loc[is_prop, "targettype"] = 0
    return df
```

and in `load_zone_project` replace `zone_targets = read_zone_targets(ztarget_path)` with

```python
        zone_targets = resolve_zone_target_types(
            read_zone_targets(ztarget_path), base.pu_vs_features,
        )
```

- [ ] **Step 4: Run the reader and writer tests**

Run: `/opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/zones/test_readers.py tests/pymarxan/zones/test_writers.py -q`
Expected: PASS (`write_zone_targets` writes whatever columns the frame has, so `targettype`
survives the write).

- [ ] **Step 5: Lint and commit**

```bash
~/.local/bin/ruff check src/pymarxan/zones/readers.py tests/pymarxan/zones/test_readers.py
git add src/pymarxan/zones/readers.py tests/pymarxan/zones/test_readers.py
git commit -m "feat(zones): read zonetarget.dat targettype (0/1), reject occurrence types 2/3 (#2)

Type 1 resolves to a fraction of the feature's total raw amount in load_zone_project
(zones.hpp:149-150) and is rewritten as type 0 so a save/load cycle is idempotent.
ZONETARGETCONTRIB round-trips through input.dat as an int.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01X8aGAi384D5xybKRugtFXT"
```

---

### Task 12: MarZone accumulation cross-check, comparative per-flip bench, stale-site sweep

**Files:**
- Create: `tests/pymarxan/zones/test_marzone_accumulation.py`
- Create: `tests/benchmarks/bench_zone_overall.py`
- Modify: `tests/benchmarks/bench_zone_sa.py` (docstring only: note the comparative bench)

**Interfaces:**
- Consumes: `compute_overall_achieved`, `_compute_zone_achieved`, `ZoneProblemCache.compute_held` / `compute_delta_zone_objective`, `make_zone_problem` (`tests/benchmarks/conftest.py`).
- Produces: no API. The sweep asserts that no stale contribution lookup survives in `src/`.

- [ ] **Step 1: Write the accumulation cross-check**

```python
"""Cross-check against a reimplementation of MarZone reserve.hpp:148-182 on tests/data/zones.

MarZone accumulates two arrays per reserve: zoneSpec[(zone, species)] += raw amount (:164) and
speciesAmounts[species] += amount × GetZoneContrib(species, zone) (:170). Contributions come
from zones.hpp:619 (0 for an unlisted pair when a contribution file is supplied). PUs in
pymarxan's zone 0 are skipped by construction (MarZone has no unassigned state; spec §8).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pymarxan.zones.cache import ZoneProblemCache
from pymarxan.zones.model import ZonalProblem
from pymarxan.zones.objective import _compute_zone_achieved, compute_overall_achieved
from pymarxan.zones.readers import load_zone_project

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "zones"
ASSIGNMENTS = [(1, 2, 1, 2), (1, 1, 2, 0), (2, 2, 2, 2), (0, 0, 0, 0)]


def marzone_accumulate(
    problem: ZonalProblem, solution: np.ndarray,
) -> tuple[dict[tuple[int, int], float], dict[int, float]]:
    contrib: dict[tuple[int, int], float] = {}
    if problem.zone_contributions is not None:
        for r in problem.zone_contributions.itertuples():
            contrib[(int(r.feature), int(r.zone))] = float(r.contribution)
    default = 1.0 if problem.zone_contributions is None else 0.0     # zones.hpp:619 / :651
    pu_ids = problem.planning_units["id"].tolist()
    zone_spec: dict[tuple[int, int], float] = {}
    species_amounts: dict[int, float] = {}
    for r in problem.pu_vs_features.itertuples():
        z = int(solution[pu_ids.index(int(r.pu))])
        if z == 0:
            continue
        isp, amt = int(r.species), float(r.amount)
        weighted = amt * contrib.get((isp, z), default)
        if amt:
            zone_spec[(z, isp)] = zone_spec.get((z, isp), 0.0) + amt                    # :164
        if weighted:
            species_amounts[isp] = species_amounts.get(isp, 0.0) + weighted            # :170
    return zone_spec, species_amounts


@pytest.fixture(params=["with_table", "without_table"])
def problem(request):
    p = load_zone_project(DATA_DIR)
    if request.param == "without_table":
        p = p.copy_with(zone_contributions=None)
    return p


@pytest.mark.parametrize("assignment", ASSIGNMENTS)
def test_overall_achieved_matches_marzone_species_amounts(problem, assignment):
    a = np.array(assignment)
    _, species_amounts = marzone_accumulate(problem, a)
    got = compute_overall_achieved(problem, a)
    for fid in problem.features["id"]:
        assert got[int(fid)] == pytest.approx(species_amounts.get(int(fid), 0.0))


@pytest.mark.parametrize("assignment", ASSIGNMENTS)
def test_zone_achieved_matches_marzone_zone_spec_raw(problem, assignment):
    a = np.array(assignment)
    zone_spec, _ = marzone_accumulate(problem, a)
    got = _compute_zone_achieved(problem, a)
    assert {k: pytest.approx(v) for k, v in zone_spec.items()} == got


@pytest.mark.parametrize("assignment", ASSIGNMENTS)
def test_cache_held_matches_both_marzone_arrays(problem, assignment):
    a = np.array(assignment)
    zone_spec, species_amounts = marzone_accumulate(problem, a)
    cache = ZoneProblemCache.from_zone_problem(problem)
    held = cache.compute_held(a)
    for (z, f), v in zone_spec.items():
        assert held.per_zone[cache.zone_id_to_col[z], cache.feat_id_to_col[f]] == pytest.approx(v)
    for f, v in species_amounts.items():
        assert held.overall[cache.feat_id_to_col[f]] == pytest.approx(v)
    assert held.per_zone.sum() == pytest.approx(sum(zone_spec.values()))
```

- [ ] **Step 2: Write the comparative bench**

`tests/benchmarks/bench_zone_overall.py`:

```python
"""Per-flip cost of the overall-target term (spec §6; review M1).

Comparative, not absolute: under default contributions the contrib_differs gate must make
every zone-to-zone move skip the overall term, so populated overall targets may cost at most
5 % more per flip than zeroed ones. Absolute budgets are machine-relative and live in
bench_zone_sa.py.
"""
from __future__ import annotations

import time

import numpy as np
import pytest

from pymarxan.zones.cache import ZoneProblemCache
from tests.benchmarks.conftest import make_zone_problem

pytestmark = pytest.mark.bench

N_PU, N_FEAT, N_ZONES, N_MOVES = 1000, 50, 3, 20_000


def _per_flip_seconds(problem, moves, assignment) -> float:
    cache = ZoneProblemCache.from_zone_problem(problem)
    held = cache.compute_held(assignment)
    best = float("inf")
    for _ in range(3):
        t0 = time.perf_counter()
        for idx, new_zone in moves:
            cache.compute_delta_zone_objective(
                idx, int(assignment[idx]), new_zone, assignment, held=held, blm=1.0,
            )
        best = min(best, time.perf_counter() - t0)
    return best / len(moves)


def test_overall_term_is_free_under_default_contributions() -> None:
    populated = make_zone_problem(
        n_pu=N_PU, n_feat=N_FEAT, n_zones=N_ZONES, density=0.3, seed=42,
    )
    assert populated.zone_contributions is None
    assert (populated.features["target"] > 0).all()
    zeroed = populated.copy_with(features=populated.features.assign(target=0.0))

    rng = np.random.default_rng(0)
    assignment = rng.integers(1, N_ZONES + 1, size=N_PU)
    moves = [
        (int(rng.integers(N_PU)), int(rng.integers(1, N_ZONES + 1))) for _ in range(N_MOVES)
    ]

    with_targets = _per_flip_seconds(populated, moves, assignment.copy())
    without = _per_flip_seconds(zeroed, moves, assignment.copy())
    ratio = with_targets / without
    assert ratio <= 1.05, (
        f"overall term not gated: {with_targets * 1e6:.2f} µs vs {without * 1e6:.2f} µs "
        f"per flip (ratio {ratio:.3f})"
    )


def test_overall_term_costs_bounded_when_contributions_differ() -> None:
    """Informational upper bound: with a contribution table every move pays the row op."""
    base = make_zone_problem(n_pu=N_PU, n_feat=N_FEAT, n_zones=N_ZONES, density=0.3, seed=42)
    rng = np.random.default_rng(1)
    table = [
        {"feature": f, "zone": z, "contribution": float(rng.uniform(0.2, 1.0))}
        for f in range(1, N_FEAT + 1)
        for z in range(1, N_ZONES + 1)
    ]
    import pandas as pd
    weighted = base.copy_with(zone_contributions=pd.DataFrame(table))
    assignment = rng.integers(1, N_ZONES + 1, size=N_PU)
    moves = [
        (int(rng.integers(N_PU)), int(rng.integers(1, N_ZONES + 1))) for _ in range(N_MOVES)
    ]
    plain = _per_flip_seconds(base, moves, assignment.copy())
    paid = _per_flip_seconds(weighted, moves, assignment.copy())
    # Review M1 measured +109 % ungated; the dense row op should stay well under 3×.
    assert paid / plain <= 3.0, f"ratio {paid / plain:.2f}"
```

Move the `import pandas as pd` to the module imports (ruff E402 otherwise). Add one line to
the `bench_zone_sa.py` module docstring: "See bench_zone_overall.py for the comparative
overall-target gate."

- [ ] **Step 3: Run both**

Run: `/opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/zones/test_marzone_accumulation.py -q`
Expected: PASS (all 24 cases).
Run: `/opt/micromamba/envs/shiny/bin/pytest tests/benchmarks/bench_zone_overall.py -q -m bench -p no:cacheprovider`
Expected: PASS; record both ratios in the task report. (`make test` excludes `bench`, so this
is run deliberately here and by the grounding reviewer, not in CI.)

- [ ] **Step 4: Stale-site sweep**

Run:
```bash
grep -rn "contrib_lookup\|zone_contributions.get\|zone_contributions\[" src/pymarxan --include=*.py
grep -rn "get_contribution(" src/pymarxan --include=*.py
```
Expected: the first returns only `zones/model.py` (`contribution_lookup`) and
`zones/model.py::validate`; the second returns only the definition in `zones/model.py`. Any
other hit is a contribution site the earlier tasks missed — route it through
`contribution_lookup()` / `contribution_matrix()` in this task and add a one-line test in
`test_marzone_accumulation.py` that exercises it.

- [ ] **Step 5: Lint and commit**

```bash
~/.local/bin/ruff check tests/pymarxan/zones/test_marzone_accumulation.py tests/benchmarks/bench_zone_overall.py tests/benchmarks/bench_zone_sa.py
git add tests/pymarxan/zones/test_marzone_accumulation.py tests/benchmarks/bench_zone_overall.py tests/benchmarks/bench_zone_sa.py
git commit -m "test(zones): MarZone reserve.hpp accumulation cross-check; comparative per-flip bench for the overall term (#2)

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01X8aGAi384D5xybKRugtFXT"
```

---

### Task 13: Re-exports, VALIDATION.md, CHANGELOG, full gate

**Files:**
- Modify: `src/pymarxan/zones/__init__.py`, `tests/pymarxan/zones/test_package_exports.py`
- Modify: `docs/VALIDATION.md` (new subsection before "## Comparing against the Marxan C++ binary"; reference added)
- Modify: `CHANGELOG.md` (`## [Unreleased]`)

**Interfaces:**
- Produces: `pymarxan.zones` re-exports `ZoneHeld`, `build_zone_solution`, `check_overall_targets`,
  `compute_overall_achieved`, `compute_overall_shortfalls`, `compute_zone_shortfalls`,
  `resolve_zone_target_types`.

- [ ] **Step 1: Extend the exports test (failing first)**

In `tests/pymarxan/zones/test_package_exports.py` add to `ZONES_EXPORTS`:

```python
    "ZoneHeld": "pymarxan.zones.cache",
    "build_zone_solution": "pymarxan.zones.objective",
    "check_overall_targets": "pymarxan.zones.objective",
    "compute_overall_achieved": "pymarxan.zones.objective",
    "compute_overall_shortfalls": "pymarxan.zones.objective",
    "compute_zone_shortfalls": "pymarxan.zones.objective",
    "resolve_zone_target_types": "pymarxan.zones.readers",
```

Run: `/opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/zones/test_package_exports.py -q`
Expected: FAIL on the seven new names.

- [ ] **Step 2: Add the re-exports**

In `src/pymarxan/zones/__init__.py` add `from pymarxan.zones.cache import ZoneHeld`, extend the
`objective` import with `build_zone_solution, check_overall_targets, compute_overall_achieved,
compute_overall_shortfalls, compute_zone_shortfalls`, the `readers` import with
`resolve_zone_target_types`, and add all seven names to `__all__` (keep it sorted as it is).
Run the exports test again — PASS.

- [ ] **Step 3: VALIDATION.md**

Insert before `## Comparing against the Marxan C++ binary`:

```markdown
### 4. Marxan with Zones: two target tiers

Multi-zone runs are checked against a second hand-verified anchor
(`tests/pymarxan/zones/marzone_anchor.py`, spec
`docs/plans/2026-09-28-marzone-overall-targets-design.md` §5). Three planning
units, two zones, one feature with amount 10 everywhere; contributions zone 1 →
1.0, zone 2 → 0.4; zone costs (5, 6, 7) and (1, 1, 1); overall target 15; a raw
zone target of 10 in zone 2. Enumerating all 27 assignments gives a unique
feasible optimum **(1, 2, 2) at cost 7** (runner-up 8). `ZoneMIPSolver` must
return it; the SA, greedy and iterative-improvement zone solvers must meet both
tiers at cost ≥ 7 (the heuristic tier of the anchor uses SPF 10, because at SPF 1
the penalised minimum is the infeasible (2, 2, 2) at 6.0). Before v0.36 every
solver returned (2, 2, 2) at cost 3 with the overall target reported as met.

The semantics follow the MarZone C++ source
(https://github.com/Marxan-source-code/marzone): overall feature targets on
contribution-weighted amounts summed over zones (`reserve.hpp:158-171`; Watts et
al. 2009 eq. 6); zone targets on raw amounts (`reserve.hpp:164`; eq. 7); unlisted
contribution pairs default to 0 when a contribution file is supplied and to 1
without one (`zones.hpp:619-627`, `:651-668`). A reimplementation of the
accumulation loop is cross-checked against pymarxan on the zones fixture
(`tests/pymarxan/zones/test_marzone_accumulation.py`).

Three named deviations, all deliberate:

- **MISSLEVEL.** pymarxan scales the met test, the penalty and the MIP constraint
  of both tiers by MISSLEVEL; MarZone applies it only when counting missing
  features (`CountMissing`, `reserve.hpp:193-275`).
- **Penalty form.** pymarxan penalises `SPF × absolute shortfall` per tier; MarZone
  uses `SPF × penalty_f × Σ(shortfall / target)` with a greedy cost-to-meet baseline
  (`reserve.hpp:751-775`, `:816-872`). The two agree on feasibility, not on the
  penalised value of infeasible assignments.
- **Unassigned versus the "available" zone.** MarZone has no unassigned state; its
  reduction to classic Marxan holds only when the available zone has zero cost,
  zero contribution for every feature, no zone targets and matching boundary
  treatment (Watts et al. 2009). pymarxan's zone 0 satisfies the first three by
  construction, so a MarZone project that lists such an available zone reproduces;
  one **without** a contribution file (available zone contributing 1) does not.
```

Add to `## References`:

```markdown
- Watts, M. E., Ball, I. R., Stewart, R. S., Klein, C. J., Wilson, K.,
  Steinback, C., Lourival, R., Kircher, L., & Possingham, H. P. (2009). Marxan
  with Zones: Software for optimal conservation based land- and sea-use zoning.
  *Environmental Modelling & Software, 24*(12), 1513–1521.
  https://doi.org/10.1016/j.envsoft.2009.06.005
```

- [ ] **Step 4: CHANGELOG `## [Unreleased]`**

```markdown
## [Unreleased]

### Changed
- **BREAKING (zone targets only): zone targets now accumulate raw amounts; unlisted
  contribution pairs default to 0 when a contribution table is supplied; set
  `ZONETARGETCONTRIB 1` to restore v0.35 zone-target behaviour.** Both follow the
  Marxan with Zones source (`reserve.hpp:164`, `zones.hpp:619`). Projects whose
  `zonecontrib.dat` lists every (feature, zone) pair and that never paired
  contributions with zone targets are unaffected. `ZonalProblem.validate()` now
  reports partially listed contribution tables in one summary line. (#2)
- All four zone solvers (`ZoneMIPSolver`, `ZoneSASolver`, `ZoneHeuristicSolver`,
  `ZoneIISolver`) build their `Solution` through one shared
  `pymarxan.zones.build_zone_solution`: `targets_met` is feature-keyed and reports
  the overall targets (so `Solution.all_targets_met` means "overall targets met"
  for zone runs), per-zone targets live in `metadata["zone_targets_met"]` as
  `"z{zone}_f{feature}"`, `penalty`/`shortfall` sum both tiers, and
  `metadata["overall_penalty"]` / `metadata["zone_penalty"]` split them. The
  heuristic and II solvers previously put tuple-keyed zone results in
  `targets_met` and no metadata.
- `ZoneProblemCache`: `compute_held_per_zone` / `update_held_per_zone` are replaced by
  `compute_held` / `update_held` returning and mutating a `ZoneHeld` (raw per-zone
  and contribution-weighted overall accumulators); `compute_full_zone_objective` and
  `compute_delta_zone_objective` take `held=` and `blm=` as keyword arguments.
- `write_zone_summary` now writes two row groups computed by the objective module
  (`tier` = `overall` per feature, `zone` per listed zone target; columns `tier, zone,
  feature, target, mean_achieved, times_met, total_runs`). The previous writer scored
  per-zone contribution-weighted amounts against the overall target, which matched
  neither tier.
- `ZoneHeuristicSolver` no longer stops as soon as the zone targets are met (it ended
  with the overall target unmet); it stops when no move improves the objective.

### Added
- **Overall feature targets in Marxan with Zones** (Watts et al. 2009 eq. 6;
  `reserve.hpp:158-171`): every zone solver enforces `features.target` on the
  contribution-weighted amount summed over zones — a hard constraint in the MIP,
  `SPF × shortfall` in the heuristics. New helpers `compute_overall_achieved`,
  `check_overall_targets`, `compute_overall_shortfalls`, `compute_zone_shortfalls`
  (issue #6, part 1), all exported from `pymarxan.zones`. Hand-verified anchor and a
  `reserve.hpp` accumulation cross-check documented in `docs/VALIDATION.md` §4. (#2)
- `ZonalProblem.contribution_lookup()`, `contribution_matrix()`,
  `zone_target_weight_matrix()`, `zone_index()`, `feature_index()` — the single
  contribution source for the objective, MIP, cache, writers and the `objectives`
  zone methods (which previously keyed the lookup backwards and used 1.0).
- `zonetarget.dat` optional `targettype` column: 0 as before, 1 = fraction of the
  feature's total raw amount (resolved in `load_zone_project`, rewritten as 0 so a
  save/load cycle is idempotent), 2/3 (occurrence targets) rejected naming the row.
- `ZonalProblem.validate()` rejects `ZONETARGETCONTRIB` outside {0, 1}, `target2 > 0`
  in zone problems, and contribution rows naming unknown zones or features.
- Comparative per-flip bench (`bench` marker): the overall term is gated by
  `contrib_differs[old, new]` (`reserve.hpp:393`), so under default contributions it
  costs ≤ 5 % per flip.

### Not included (follow-ons)
- `save_zone_project` (the symmetric partner of `load_zone_project`); occurrence
  targets and `zonetarget2.dat`; MarZone's proportional penalty form; `target2` in
  zone problems; lock-into-a-named-zone (#6 part 2).
```

- [ ] **Step 5: Full gate**

```bash
PATH="/opt/micromamba/envs/shiny/bin:$HOME/.local/bin:$PWD/.venv/bin:$PATH" make check
/opt/micromamba/envs/shiny/bin/python examples/validate_marxan_parity.py
```
Expected: lint clean, mypy clean, full suite green (coverage ≥ 75 %); the parity harness
prints 35.0 / 43.0 / 45.0. If `test_solutions_are_different` alone fails, rerun it once
(known stochastic flake).

- [ ] **Step 6: Commit**

```bash
git add src/pymarxan/zones/__init__.py tests/pymarxan/zones/test_package_exports.py docs/VALIDATION.md CHANGELOG.md
git commit -m "docs(zones): VALIDATION §4 MarZone two-tier anchor + deviations; CHANGELOG for 0.36.0; zones re-exports (#2)

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01X8aGAi384D5xybKRugtFXT"
```

---

## Self-review record

- **Spec coverage.** §3.1 → Task 1; §3.2 → Tasks 2, 4, 6; §3.3 → Tasks 3, 11; §3.4 → Tasks 2, 4
  (MISSLEVEL tests), 13 (doc); §3.5 → Task 5; §4.1 → 1; §4.2 → 2, 3, 4, 5; §4.3 → 3, 6; §4.4 →
  3, 4; §4.5 → 4, 5, 7, 8; §4.6 → 10; §4.7 → 11; §4.8 → 13; §5 → 2 (module + oracle), 4
  (cache vs oracle), 6, 7, 8; §6 test list → every bullet has a task (bench → 12; parity
  harness → 13; re-pins → 3 with old/new values in the commit message).
- **Placeholders.** None: every step carries code or an exact command; no task defers to
  another task's text for its own code.
- **Type consistency.** `build_zone_solution(problem, zone_assignment, blm, *, solver_name,
  run=None)` used identically in Tasks 5–8; `compute_held` / `update_held` /
  `held=` / `blm=` identical in Tasks 4, 8, 12; `contribution_lookup()` keyed
  `(feature, zone)` everywhere; `zone_index()` / `feature_index()` used by objective (2, 3),
  cache (3), MIP (3), writers (10); `_feature_groups` defined in Task 3 and used in Task 6.
- **Review Focus.** All five lines have a pinned test in their owning task (1 → Task 1
  `test_zone_target_contrib_rejects_out_of_domain` + Task 3 `test_cache_rejects_out_of_domain_flag`;
  2 → Task 11 `test_resolution_is_idempotent_across_write_and_read`; 3 → Task 6
  `test_feature_with_target_but_no_amounts_is_infeasible` + Task 7
  `test_feature_without_amounts_is_penalised_not_fatal`; 4 → Task 1
  `test_validate_reports_unknown_contribution_pair`; 5 → Task 2
  `test_spf_column_absent_defaults_to_one` + Task 4 same name in the cache tests).
