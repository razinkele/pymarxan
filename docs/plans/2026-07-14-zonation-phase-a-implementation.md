# Zonation rank-removal (Phase A) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `pymarxan.zonation` — a Zonation-style rank-removal engine that ranks every planning unit by iterative backward removal (CAZ + ABF rules), producing a continuous priority map and per-feature performance curves.

**Architecture:** A pure-NumPy removal loop over the dense feature matrix. Start with the whole landscape; each iteration remove the cell(s) with the smallest weighted marginal loss (`max` over features for CAZ, `sum` for ABF), dividing by cost; the removal order is the priority ranking. Returns a `ZonationResult`. No new dependencies, no UI (later phases add a Solver adapter, smoothing, Shiny).

**Tech Stack:** Python 3.12+, NumPy, pandas.

**Design spec:** `docs/plans/2026-07-14-zonation-phase-a-design.md` (read it first).

## Global Constraints

- Python 3.12+, `from __future__ import annotations` at the top of every new file, full type hints.
- Zero new third-party dependencies (pure Python / NumPy / pandas).
- Tests **must** run under the `shiny` micromamba env: `/opt/micromamba/envs/shiny/bin/pytest` (`.venv` lacks rasterio/ipyleaflet). See the `marxan-testing` skill.
- Domain models are dataclasses; test layout mirrors `src/` under `tests/pymarxan/`.
- Lint: ruff (E, F, I, UP; line length 99). Types: mypy clean. Coverage threshold 75%.
- The bar before done: `make check` green (lint + types + full suite).
- Commit after each task (TDD: failing test → implementation → passing test → commit).
- **Removal math:** CAZ `δ_i = max_j(w_j·q_ij/Q_j)` (an exact transcription of Moilanen 2007 Eq. 1a), ABF `δ_i = Σ_j(w_j·q_ij/Q_j)` (the proportional / remaining-sum member of Zonation's additive-benefit family — `1/R_j` marginal benefit, NOT a strictly *linear* benefit, and NOT the paper's general power form). Both over the *remaining* feature sum `Q_j`, divided by cost `c_i`. Cells removed last = highest priority; `priority_rank[k-th removed] = (k+1)/n`.

## File Structure

- `src/pymarxan/zonation/__init__.py` — exports `rank_removal`, `ZonationResult`.
- `src/pymarxan/zonation/result.py` — `ZonationResult` dataclass.
- `src/pymarxan/zonation/rank_removal.py` — `rank_removal(...)` (the CAZ/ABF loop).
- `tests/pymarxan/zonation/__init__.py` — empty.
- `tests/pymarxan/zonation/test_result.py`, `test_rank_removal.py`.

**Hand-computed oracles** (used in the tests, all derived in the design spec):
- **P1** (CAZ order): 3 PUs, 2 features, uniform cost. `q = [[10,0],[0,10],[5,5]]` (PU ids 1,2,3). CAZ removal order = `[3, 1, 2]`; ranks PU3=1/3, PU1=2/3, PU2=1.0.
- **P2** (CAZ vs ABF divergence): 3 PUs, 2 features, uniform cost. `q = [[4,4],[5,1],[1,5]]` (PU ids 1,2,3; totals feat1=10, feat2=10). **CAZ** order = `[1, 2, 3]` (generalist PU1 removed first, rank 1/3). **ABF** order = `[2, 3, 1]` (generalist PU1 removed *last*, rank 1.0). Same cell, opposite priority — the CAZ-rarity vs ABF-richness divergence.

---

### Task 1: `ZonationResult` dataclass

**Files:**
- Create: `src/pymarxan/zonation/__init__.py`
- Create: `src/pymarxan/zonation/result.py`
- Create: `tests/pymarxan/zonation/__init__.py`
- Test: `tests/pymarxan/zonation/test_result.py`

**Interfaces:**
- Produces:
  - `ZonationResult(priority_rank: dict[int, float], removal_order: list[int], performance_curves: pd.DataFrame, rule: str)`.
  - `.top_fraction(f: float) -> set[int]` — the `ceil(f·n)` highest-ranked PU ids; raises `ValueError` if `f` not in `(0, 1]`.
  - `.to_dataframe() -> pd.DataFrame` — columns `pu_id`, `priority_rank`, `removal_position`.

- [ ] **Step 1: Write the failing tests**

Create `tests/pymarxan/zonation/__init__.py` (empty) and `tests/pymarxan/zonation/test_result.py`:

```python
"""Tests for the ZonationResult container."""
from __future__ import annotations

import pandas as pd
import pytest

from pymarxan.zonation.result import ZonationResult


def _result() -> ZonationResult:
    # removal_order [3, 1, 2] → ranks 3=1/3, 1=2/3, 2=1.0
    return ZonationResult(
        priority_rank={3: 1 / 3, 1: 2 / 3, 2: 1.0},
        removal_order=[3, 1, 2],
        performance_curves=pd.DataFrame(
            {"prop_landscape_remaining": [1.0, 0.5, 0.0], "feat_1": [1.0, 0.5, 0.0]}
        ),
        rule="caz",
    )


def test_top_fraction_returns_highest_ranked():
    res = _result()
    # top 1/3 of 3 PUs = 1 cell = the rank-1.0 PU (id 2)
    assert res.top_fraction(1 / 3) == {2}
    # top 2/3 = 2 cells = ids 2 and 1
    assert res.top_fraction(2 / 3) == {2, 1}
    # top 1.0 = all
    assert res.top_fraction(1.0) == {1, 2, 3}


def test_top_fraction_rejects_out_of_range():
    res = _result()
    with pytest.raises(ValueError):
        res.top_fraction(0.0)
    with pytest.raises(ValueError):
        res.top_fraction(1.5)


def test_to_dataframe_columns_and_positions():
    res = _result()
    df = res.to_dataframe().set_index("pu_id")
    assert list(res.to_dataframe().columns) == [
        "pu_id",
        "priority_rank",
        "removal_position",
    ]
    # removal_position is the 0-indexed slot in removal_order
    assert df.loc[3, "removal_position"] == 0
    assert df.loc[2, "removal_position"] == 2
    assert df.loc[2, "priority_rank"] == pytest.approx(1.0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `/opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/zonation/test_result.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pymarxan.zonation'`.

- [ ] **Step 3: Implement the result + package**

Create `src/pymarxan/zonation/__init__.py`:

```python
"""Zonation-style rank-removal prioritization for pymarxan.

Ranks every planning unit by iterative backward removal (Moilanen et al. 2005):
the whole landscape is stripped one cell at a time, removing the least-valuable
cell each step; the removal order is a continuous 0-1 priority map. See
``docs/plans/2026-07-14-zonation-phase-a-design.md``.
"""
from __future__ import annotations

from pymarxan.zonation.result import ZonationResult

__all__ = ["ZonationResult"]
```

Create `src/pymarxan/zonation/result.py`:

```python
"""Result container for a Zonation rank-removal run."""
from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd


@dataclass
class ZonationResult:
    """Priority ranking of every planning unit, with performance curves.

    ``priority_rank`` maps PU id -> rank in (0, 1], where 1.0 is the
    highest-priority cell (removed last). Ranks are unique by construction (each
    removal position gets a distinct ``(k+1)/n``), so ``top_fraction`` is
    deterministic. ``removal_order`` lists PU ids first-removed (lowest priority)
    first. ``performance_curves`` is wide form: ``prop_landscape_remaining`` and
    ``prop_cost_remaining`` columns plus one ``feat_<id>`` column per feature
    (retained proportion), one row per recorded step.
    """

    priority_rank: dict[int, float]
    removal_order: list[int]
    performance_curves: pd.DataFrame
    rule: str

    def top_fraction(self, f: float) -> set[int]:
        """Return the PU ids in the top ``f`` share by priority rank."""
        if not 0.0 < f <= 1.0:
            raise ValueError(f"f must be in (0, 1], got {f}")
        n = len(self.priority_rank)
        k = math.ceil(f * n)
        ordered = sorted(
            self.priority_rank, key=lambda pu: self.priority_rank[pu], reverse=True
        )
        return set(ordered[:k])

    def to_dataframe(self) -> pd.DataFrame:
        position = {pu: i for i, pu in enumerate(self.removal_order)}
        pus = list(self.priority_rank)
        return pd.DataFrame(
            {
                "pu_id": pus,
                "priority_rank": [self.priority_rank[p] for p in pus],
                "removal_position": [position[p] for p in pus],
            }
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `/opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/zonation/test_result.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/pymarxan/zonation/__init__.py src/pymarxan/zonation/result.py tests/pymarxan/zonation/
git commit -m "feat(zonation): ZonationResult container (priority rank, curves, top_fraction)"
```

---

### Task 2: `rank_removal` engine (CAZ + ABF)

**Files:**
- Create: `src/pymarxan/zonation/rank_removal.py`
- Modify: `src/pymarxan/zonation/__init__.py` (export `rank_removal`)
- Test: `tests/pymarxan/zonation/test_rank_removal.py`

**Interfaces:**
- Consumes: `ConservationProblem` (`build_pu_feature_matrix`, `planning_units`, `features`), `ZonationResult` (Task 1), status constants `STATUS_LOCKED_IN`/`STATUS_LOCKED_OUT`.
- Produces: `rank_removal(problem, *, rule="caz", weights=None, warp=1, use_cost=True) -> ZonationResult`.

- [ ] **Step 1: Write the failing tests**

Create `tests/pymarxan/zonation/test_rank_removal.py`:

```python
"""Tests for the Zonation CAZ/ABF rank-removal engine."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pymarxan.models.problem import ConservationProblem
from pymarxan.zonation.rank_removal import rank_removal


def _problem(q_rows, cost=None, status=None, feat_ids=(1, 2)) -> ConservationProblem:
    """Build a problem from a dense per-PU feature matrix (list of rows)."""
    n_pu = len(q_rows)
    n_feat = len(q_rows[0])
    pu_ids = list(range(1, n_pu + 1))
    planning_units = pd.DataFrame(
        {
            "id": pu_ids,
            "cost": [1.0] * n_pu if cost is None else list(cost),
            "status": [0] * n_pu if status is None else list(status),
        }
    )
    features = pd.DataFrame(
        {
            "id": list(feat_ids),
            "name": [f"f{j}" for j in feat_ids],
            "target": [1.0] * n_feat,
            "spf": [1.0] * n_feat,
        }
    )
    rows = []
    for pu, row in zip(pu_ids, q_rows):
        for fid, amt in zip(feat_ids, row):
            if amt:
                rows.append({"species": fid, "pu": pu, "amount": float(amt)})
    pu_vs_features = pd.DataFrame(rows, columns=["species", "pu", "amount"])
    return ConservationProblem(planning_units, features, pu_vs_features)


# P1: CAZ removal order is [3, 1, 2] (generalist PU3 removed first).
P1 = [[10, 0], [0, 10], [5, 5]]
# P2: CAZ [1,2,3] vs ABF [2,3,1] — generalist PU1 flips lowest<->highest.
P2 = [[4, 4], [5, 1], [1, 5]]


def test_caz_hand_computed_order():
    res = rank_removal(_problem(P1), rule="caz")
    assert res.removal_order == [3, 1, 2]
    assert res.priority_rank[3] == pytest.approx(1 / 3)
    assert res.priority_rank[1] == pytest.approx(2 / 3)
    assert res.priority_rank[2] == pytest.approx(1.0)
    assert res.rule == "caz"


def test_abf_diverges_from_caz():
    caz = rank_removal(_problem(P2), rule="caz")
    abf = rank_removal(_problem(P2), rule="abf")
    assert caz.removal_order == [1, 2, 3]
    assert abf.removal_order == [2, 3, 1]
    # the generalist PU1 is lowest priority under CAZ, highest under ABF
    assert caz.priority_rank[1] == pytest.approx(1 / 3)
    assert abf.priority_rank[1] == pytest.approx(1.0)


def test_caz_rarity_sole_occurrence_removed_last():
    # feature 2 occurs only in PU2 (status 0) → PU2 removed last (rank 1.0).
    q = [[10, 0], [1, 5], [10, 0]]  # only PU2 holds feature 2
    res = rank_removal(_problem(q), rule="caz")
    assert res.removal_order[-1] == 2
    assert res.priority_rank[2] == pytest.approx(1.0)


def test_locks_respected():
    # PU1 locked-in (2), PU2 normal (0), PU3 locked-out (3).
    res = rank_removal(_problem(P1, status=[2, 0, 3]), rule="caz")
    assert res.removal_order[0] == 3       # locked-out removed first
    assert res.removal_order[-1] == 1      # locked-in removed last
    assert res.priority_rank[1] == pytest.approx(1.0)


def test_cost_changes_order():
    # PU1=[10,0] cost 1, PU2=[0,10] cost 10 → equal biological delta.
    q = [[10, 0], [0, 10]]
    with_cost = rank_removal(_problem(q, cost=[1.0, 10.0]), rule="caz", use_cost=True)
    without = rank_removal(_problem(q, cost=[1.0, 10.0]), rule="caz", use_cost=False)
    assert with_cost.removal_order[0] == 2   # expensive cell removed first
    assert without.removal_order[0] == 1     # tie → lowest PU index first


def test_performance_curves_bounded_and_monotone():
    res = rank_removal(_problem(P1), rule="caz")
    pc = res.performance_curves
    assert pc["prop_landscape_remaining"].iloc[0] == pytest.approx(1.0)
    assert pc["prop_landscape_remaining"].iloc[-1] == pytest.approx(0.0)
    # cost axis present; == landscape axis under uniform cost
    assert pc["prop_cost_remaining"].iloc[0] == pytest.approx(1.0)
    assert pc["prop_cost_remaining"].iloc[-1] == pytest.approx(0.0)
    for col in ["feat_1", "feat_2"]:
        vals = pc[col].to_numpy()
        assert np.all((vals >= -1e-9) & (vals <= 1 + 1e-9))
        assert np.all(np.diff(vals) <= 1e-9)   # non-increasing
        assert vals[-1] == pytest.approx(0.0)  # empty landscape → 0 retained


def test_warp_matches_exact_on_tie_free_problem():
    # NB: exact order-equality is coincidental on P1 (the batch happens to be
    # order-preserving here); warp only guarantees coarse-bucket agreement.
    exact = rank_removal(_problem(P1), rule="caz", warp=1)
    warped = rank_removal(_problem(P1), rule="caz", warp=2)
    assert warped.removal_order == exact.removal_order


def test_zero_distribution_feature_is_inert():
    # feature 3 occurs in no PU (T_3 = 0) → excluded from delta, retained 1.0.
    q = [[10, 0, 0], [0, 10, 0], [5, 5, 0]]
    res = rank_removal(_problem(q, feat_ids=(1, 2, 3)), rule="caz")
    assert res.removal_order == [3, 1, 2]           # same as P1, feat3 inert
    assert res.performance_curves["feat_3"].iloc[0] == pytest.approx(1.0)
    assert res.performance_curves["feat_3"].iloc[-1] == pytest.approx(1.0)


def test_feature_extinction_midrun_uses_guard():
    # feature 2 lives only in a locked-out cell (PU1). It is stripped first, so
    # feature 2 goes extinct while two NORMAL cells remain — the only way to get
    # multi-cell extinction (CAZ otherwise protects a feature to its last cell).
    # Without the Q_safe guard both normal deltas become NaN (0/0) and removal
    # falls back to PU-index order [1,2,3]; with the guard it is value order.
    q = [[0, 5], [10, 0], [8, 0]]
    res = rank_removal(_problem(q, status=[3, 0, 0]), rule="caz")
    # PU3 (feat1=8, lower value) removed before PU2 (feat1=10) — by value not index
    assert res.removal_order == [1, 3, 2]
    assert all(np.isfinite(v) for v in res.priority_rank.values())
    assert np.all(np.isfinite(res.performance_curves.to_numpy()))


def test_invalid_rule_raises():
    with pytest.raises(ValueError, match="rule"):
        rank_removal(_problem(P1), rule="bogus")


def test_zero_cost_raises_when_use_cost():
    with pytest.raises(ValueError, match="cost"):
        rank_removal(_problem(P1, cost=[1.0, 0.0, 1.0]), rule="caz", use_cost=True)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `/opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/zonation/test_rank_removal.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pymarxan.zonation.rank_removal'`.

- [ ] **Step 3: Implement the engine**

Create `src/pymarxan/zonation/rank_removal.py`:

```python
"""Zonation CAZ/ABF rank-removal engine (Moilanen et al. 2005; Moilanen 2007).

Distinct from ``pymarxan.analysis.rank_importance`` (Jung et al. 2021), which
ranks only the *selected* PUs of an existing solution by Marxan-objective
increase; this ranks *every* PU from the whole landscape by proportional
biological loss.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from pymarxan.models.problem import (
    STATUS_LOCKED_IN,
    STATUS_LOCKED_OUT,
    ConservationProblem,
)
from pymarxan.zonation.result import ZonationResult


def rank_removal(
    problem: ConservationProblem,
    *,
    rule: str = "caz",
    weights: dict[int, float] | None = None,
    warp: int = 1,
    use_cost: bool = True,
) -> ZonationResult:
    """Rank every planning unit by iterative backward removal.

    Each step removes the cell(s) with the smallest weighted marginal loss
    ``delta_i`` — ``max_j`` over features for ``rule="caz"`` (core-area,
    favors rarity; an exact transcription of Moilanen 2007 Eq. 1a), ``sum_j``
    for ``rule="abf"`` (additive benefit, favors richness) — of
    ``w_j * q_ij / Q_j`` (``Q_j`` = remaining total of feature ``j``), divided
    by cost. ABF here is the proportional / remaining-sum member of Zonation's
    additive-benefit family (marginal benefit ``1/R_j``); it is NOT a strictly
    *linear* benefit (which would use the fixed original total and be static),
    and the concave power-benefit generalization is a future extension.
    Locked-out cells are removed first, locked-in last; the removal order is the
    priority ranking (last removed = rank 1.0).

    The O(n^2 * n_feat) recompute is inherent — removing a cell shifts every
    ``Q_j``, so the Marxan per-flip delta model does not apply (only ``Q_j -=
    q_ij`` is incremental). ``warp`` is the scaling knob; this suits vector PUs
    (hundreds to low-thousands), not million-cell rasters.
    """
    if rule not in ("caz", "abf"):
        raise ValueError(f"rule must be 'caz' or 'abf', got {rule!r}")

    q = problem.build_pu_feature_matrix()  # (n_pu, n_feat), rows = PU order
    n_pu, n_feat = q.shape
    pu_ids = problem.planning_units["id"].to_numpy()
    feat_ids = problem.features["id"].to_numpy()
    status = problem.planning_units["status"].to_numpy()

    w = np.ones(n_feat, dtype=float)
    if weights:
        for j, fid in enumerate(feat_ids):
            if int(fid) in weights:
                w[j] = float(weights[int(fid)])

    if use_cost:
        c = problem.planning_units["cost"].to_numpy().astype(float)
        if np.any(c <= 0):
            raise ValueError("use_cost=True requires every planning-unit cost > 0")
    else:
        c = np.ones(n_pu, dtype=float)

    warp = max(1, min(int(warp), max(n_pu, 1)))

    remaining = np.ones(n_pu, dtype=bool)
    Q = q.sum(axis=0)          # remaining totals per feature
    T = Q.copy()               # original totals (for curves)
    T_safe = np.where(T > 0, T, 1.0)
    cost_total = float(c.sum()) if c.sum() > 0 else 1.0

    removal_order: list[int] = []
    curve_rows: list[dict] = []

    def record_curve() -> None:
        retained = np.where(T > 0, Q / T_safe, 1.0)
        row: dict = {
            "prop_landscape_remaining": remaining.sum() / n_pu,
            "prop_cost_remaining": float(c[remaining].sum()) / cost_total,
        }
        for j, fid in enumerate(feat_ids):
            row[f"feat_{int(fid)}"] = float(retained[j])
        curve_rows.append(row)

    record_curve()

    def candidate_indices() -> np.ndarray:
        locked_out = remaining & (status == STATUS_LOCKED_OUT)
        if locked_out.any():
            return np.flatnonzero(locked_out)
        normal = remaining & (status != STATUS_LOCKED_OUT) & (status != STATUS_LOCKED_IN)
        if normal.any():
            return np.flatnonzero(normal)
        return np.flatnonzero(remaining & (status == STATUS_LOCKED_IN))

    while remaining.any():
        cand = candidate_indices()  # ascending PU-index order
        # w_j * q_ij / Q_j on the candidate slice; extinct features (Q_j == 0)
        # contribute 0 (Q_safe avoids the divide; the mask covers any residue).
        Q_safe = np.where(Q > 0, Q, 1.0)
        r = q[cand] * (w / Q_safe)
        r[:, Q <= 0] = 0.0
        if n_feat == 0:
            delta = np.zeros(cand.size)
        elif rule == "caz":
            delta = r.max(axis=1)
        else:  # abf
            delta = r.sum(axis=1)
        delta = delta / c[cand]
        # stable sort → ties broken by PU index (cand is ascending)
        order = np.argsort(delta, kind="stable")
        k = min(warp, cand.size)
        for idx in cand[order[:k]]:
            removal_order.append(int(pu_ids[idx]))
            remaining[idx] = False
            Q -= q[idx]
        record_curve()

    priority_rank = {
        pu: (position + 1) / n_pu for position, pu in enumerate(removal_order)
    }
    return ZonationResult(
        priority_rank=priority_rank,
        removal_order=removal_order,
        performance_curves=pd.DataFrame(curve_rows),
        rule=rule,
    )
```

Update `src/pymarxan/zonation/__init__.py`:

```python
"""Zonation-style rank-removal prioritization for pymarxan.

Ranks every planning unit by iterative backward removal (Moilanen et al. 2005):
the whole landscape is stripped one cell at a time, removing the least-valuable
cell each step; the removal order is a continuous 0-1 priority map. See
``docs/plans/2026-07-14-zonation-phase-a-design.md``.
"""
from __future__ import annotations

from pymarxan.zonation.rank_removal import rank_removal
from pymarxan.zonation.result import ZonationResult

__all__ = ["ZonationResult", "rank_removal"]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `/opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/zonation/test_rank_removal.py -v`
Expected: PASS (11 tests).

- [ ] **Step 5: Commit**

```bash
git add src/pymarxan/zonation/rank_removal.py src/pymarxan/zonation/__init__.py tests/pymarxan/zonation/test_rank_removal.py
git commit -m "feat(zonation): rank_removal engine — CAZ + ABF, cost/status/warp, performance curves"
```

---

### Task 3: CHANGELOG + full-suite green

**Files:**
- Modify: `CHANGELOG.md` (`## [Unreleased]` → `### Added`)

**Interfaces:** none (documentation + verification task).

- [ ] **Step 1: Add the CHANGELOG entry**

Under `## [Unreleased]` in `CHANGELOG.md` (add the `## [Unreleased]` and `### Added` headers if the section is empty):

```markdown
### Added

- **Zonation rank-removal prioritization (`pymarxan.zonation`, Phase A).** A
  Zonation-style engine (Moilanen et al. 2005; Moilanen 2007) that ranks every
  planning unit by iterative backward removal — ``rank_removal(problem, rule=...)``
  with core-area (CAZ, ``max`` over features → favors rarity) and additive-benefit
  (ABF, ``sum`` → favors richness) rules, cost- and status-aware, with a warp
  factor. Returns a ``ZonationResult`` with a continuous 0-1 priority map and
  per-feature performance curves — the priority-rank paradigm Marxan's min-set
  cannot express. +14 tests (hand-computed CAZ order; CAZ-vs-ABF divergence).
```

(Confirm the count with
`/opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/zonation -q`: 3 result + 11 engine = 14.)

- [ ] **Step 2: Run the full check**

Run: `PATH="/opt/micromamba/envs/shiny/bin:$HOME/.local/bin:$PWD/.venv/bin:$PATH" make check`
Expected: `make check` green — 0 ruff, 0 mypy, full suite passes (previous count + 12). If `test_solutions_are_different` fails, rerun once (known SA flake; see the `marxan-testing` skill).

Note: the `shiny` env activation path in CLAUDE.md (`/opt/micromamba/etc/profile.d/micromamba.sh`) may not exist on this machine; the `PATH=...` prefix above is the working invocation (shiny env first for the rasterio-capable pytest).

- [ ] **Step 3: Commit**

```bash
git add CHANGELOG.md
git commit -m "docs(zonation): CHANGELOG entry for Phase A rank-removal"
```

---

## Post-plan notes

- **Multi-agent design review:** Per CLAUDE.md, run the `multi-agent-design-review` skill on the spec before executing — the CAZ/ABF marginal-loss math and the cost-division detail are exactly what the scientific lens should confirm against the Zonation manual.
- **Parity note:** this adds a new solver-family engine but does **not** touch the Marxan min-set solvers or objective math, so the 35.0 ground-truth anchor is unaffected. A quick `marxan-parity-check` run after `make check` confirms nothing regressed.
- **Deferred (later phases, own specs):** Phase B `ZonationSolver` (Solver ABC adapter + registry), Phase C distribution smoothing (reuse `connectivity.smoothing`), Phase D Shiny panel.
- **Not in this plan:** version bump / release (see `release-pymarxan`), README blurb.
- **Scientific citations:** Moilanen et al. 2005 (`10.1098/rspb.2005.3164`), Moilanen 2007 (`10.1016/j.biocon.2006.09.008`), Lehtomäki & Moilanen 2013 (`10.1016/j.envsoft.2013.05.001`) — all scite-verified; CAZ/ABF formulas taken from the open Moilanen 2007 PDF.
