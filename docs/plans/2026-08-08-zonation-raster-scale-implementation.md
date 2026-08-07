# Zonation Raster-Scale rank_removal Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rewrite `pymarxan.zonation.rank_removal` internals as sparse + dirty-set incremental so ~1M-cell rasters rank in minutes, with output identical in semantics to the current dense engine (FP contract in the design spec §7).

**Architecture:** One-file engine rewrite (`rank_removal.py`) guarded by a characterization suite whose oracle is a verbatim copy of the current dense engine. CSR/CSC from `build_pu_feature_csr()`; per batch: rescore dirty∩candidates only, argpartition selection with exact PU-index tie-break, sequential `Q` updates, CSC-based dirty marking, incremental curve bookkeeping. Two new guards (negative amounts/weights raise; smoothing capped at 50k PUs) and a warp advisory helper.

**Tech Stack:** numpy, scipy.sparse (both already deps), pandas, pytest.

**Spec:** `docs/plans/2026-08-08-zonation-raster-scale-design.md` — read it first; §4 (algorithm), §7 (FP/equivalence contract), §9 (test matrix) are normative.

## Global Constraints

- Tests run under the shiny micromamba env: `/opt/micromamba/envs/shiny/bin/pytest` (NEVER bare `pytest` / `.venv` — see `marxan-testing` skill).
- Python 3.12+, `from __future__ import annotations` in every file, full type hints (mypy clean), ruff clean (E, F, I, UP; line length 99).
- Public API frozen: `rank_removal` signature, `ZonationResult`, `ZonationSolver` unchanged. Only new behavior: `ValueError` on negative amounts/weights, `ValueError` on smoothing with n_pu > 50_000, `UserWarning` warp advisory.
- The parity anchor (`tests/data/simple` cost 35.0) must stay green — no Marxan solver is touched; `make check` proves it at the end.
- Work on branch `feat/zonation-raster-scale` (already created; spec committed).
- `test_solutions_are_different` is a known stochastic flake — rerun before treating as real.

---

### Task 1: Characterization harness — oracle + equivalence suite

The oracle is the CURRENT engine, copied verbatim. All tests in this task must PASS against the current `src` implementation — they pin behavior before the rewrite. If any fails, STOP: the oracle copy or a builder is wrong, not `src`.

**Files:**
- Create: `tests/pymarxan/zonation/test_rank_removal_scale.py`

**Interfaces:**
- Produces: `_dense_rank_removal(problem, *, rule, weights, warp, use_cost) -> ZonationResult` (test-local oracle), `_random_problem(seed, n_pu, n_feat, *, integer, statuses, costs) -> ConservationProblem`, `_assert_equal_results(a, b, *, check_curves=True) -> None`. Tasks 3–4 reuse all three.

- [ ] **Step 1: Write the test module** with the oracle, builders, and the equivalence/tie/float tests:

```python
"""Characterization + equivalence suite for the sparse rank_removal rewrite.

The oracle ``_dense_rank_removal`` is a verbatim copy of the pre-rewrite dense
engine (v0.13.0 lineage), INCLUDING its smoothing branch. Every test here passed
against that engine before the rewrite; the sparse engine must keep them green
per the FP contract in docs/plans/2026-08-08-zonation-raster-scale-design.md §7:
integer amounts -> exact removal order for BOTH rules (the chunked-dense rescore
kernel reuses the dense engine's per-row expressions, so the only FP boundaries
are the initial-Q summation order — float amounts only — and the incremental
cost curve — float costs only). Float/smoothed fixtures pin fixed seeds and
assert exact order; a near-tie flip after a numpy/scipy upgrade fails
deterministically -> change the seed, never loosen to allclose on order.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pymarxan.models.problem import (
    STATUS_LOCKED_IN,
    STATUS_LOCKED_OUT,
    ConservationProblem,
)
from pymarxan.zonation.rank_removal import rank_removal
from pymarxan.zonation.result import ZonationResult
from pymarxan.zonation.smoothing import SmoothingSpec


# --------------------------------------------------------------------------
# Oracle: verbatim copy of the dense engine, smoothing branch included
# (review finding #5: the smoothing path changes materially in the rewrite,
# so it needs oracle coverage too).
# --------------------------------------------------------------------------
def _dense_rank_removal(
    problem: ConservationProblem,
    *,
    rule: str = "caz",
    weights: dict[int, float] | None = None,
    warp: int = 1,
    use_cost: bool = True,
    smoothing: SmoothingSpec | None = None,
) -> ZonationResult:
    q = problem.build_pu_feature_matrix()
    if smoothing is not None:
        q = smoothing.apply(q)
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
    else:
        c = np.ones(n_pu, dtype=float)

    warp = max(1, min(int(warp), max(n_pu, 1)))
    remaining = np.ones(n_pu, dtype=bool)
    Q = q.sum(axis=0)
    T = Q.copy()
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
        cand = candidate_indices()
        Q_safe = np.where(Q > 0, Q, 1.0)
        r = q[cand] * (w / Q_safe)
        r[:, Q <= 0] = 0.0
        if n_feat == 0:
            delta = np.zeros(cand.size)
        elif rule == "caz":
            delta = r.max(axis=1)
        else:
            delta = r.sum(axis=1)
        delta = delta / c[cand]
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


# --------------------------------------------------------------------------
# Builders
# --------------------------------------------------------------------------
def _random_problem(
    seed: int,
    n_pu: int = 80,
    n_feat: int = 6,
    *,
    integer: bool = True,
    statuses: bool = False,
    costs: bool = True,
) -> ConservationProblem:
    """Random sparse problem; integer amounts by default (exactness contract)."""
    rng = np.random.default_rng(seed)
    pu_ids = list(range(1, n_pu + 1))
    amounts = rng.integers(0, 6, size=(n_pu, n_feat)).astype(float)
    if not integer:
        amounts *= rng.uniform(0.5, 1.5, size=amounts.shape)
    amounts[rng.random(amounts.shape) < 0.55] = 0.0  # ~55% sparse
    status = np.zeros(n_pu, dtype=int)
    if statuses:
        status[rng.random(n_pu) < 0.1] = STATUS_LOCKED_IN
        status[rng.random(n_pu) < 0.1] = STATUS_LOCKED_OUT
    cost = rng.integers(1, 6, size=n_pu).astype(float) if costs else np.ones(n_pu)
    if not integer and costs:
        cost = cost * rng.uniform(0.5, 1.5, size=n_pu)  # float-cost regime (§7 site 2)
    planning_units = pd.DataFrame({"id": pu_ids, "cost": cost, "status": status})
    feat_ids = list(range(1, n_feat + 1))
    features = pd.DataFrame(
        {
            "id": feat_ids,
            "name": [f"f{j}" for j in feat_ids],
            "target": [1.0] * n_feat,
            "spf": [1.0] * n_feat,
        }
    )
    rows = [
        {"species": fid, "pu": pu, "amount": amounts[i, j]}
        for i, pu in enumerate(pu_ids)
        for j, fid in enumerate(feat_ids)
        if amounts[i, j]
    ]
    pu_vs_features = pd.DataFrame(rows, columns=["species", "pu", "amount"])
    return ConservationProblem(planning_units, features, pu_vs_features)


def _assert_equal_results(
    a: ZonationResult, b: ZonationResult, *, check_curves: bool = True
) -> None:
    assert a.removal_order == b.removal_order
    assert a.priority_rank == b.priority_rank
    assert a.rule == b.rule
    if check_curves:
        pd.testing.assert_frame_equal(
            a.performance_curves, b.performance_curves, check_exact=True
        )


# --------------------------------------------------------------------------
# Equivalence matrix (design §9.1) — exact per the §7 contract
# --------------------------------------------------------------------------
@pytest.mark.parametrize("rule", ["caz", "abf"])
@pytest.mark.parametrize("warp", [1, 3, 7, 80])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_equivalence_random_integer(rule: str, warp: int, seed: int) -> None:
    p = _random_problem(seed)
    _assert_equal_results(
        rank_removal(p, rule=rule, warp=warp),
        _dense_rank_removal(p, rule=rule, warp=warp),
    )


@pytest.mark.parametrize("rule", ["caz", "abf"])
def test_equivalence_locks_and_costs(rule: str) -> None:
    p = _random_problem(7, statuses=True)
    for warp in (1, 4):
        for use_cost in (True, False):
            _assert_equal_results(
                rank_removal(p, rule=rule, warp=warp, use_cost=use_cost),
                _dense_rank_removal(p, rule=rule, warp=warp, use_cost=use_cost),
            )


@pytest.mark.parametrize("rule", ["caz", "abf"])
def test_equivalence_weights_and_extinction(rule: str) -> None:
    # Feature 1 held ONLY by locked-out PUs -> goes extinct in the first phase.
    pu = pd.DataFrame(
        {"id": [1, 2, 3, 4, 5, 6], "cost": [1.0] * 6, "status": [3, 3, 0, 0, 0, 2]}
    )
    feats = pd.DataFrame(
        {"id": [1, 2], "name": ["a", "b"], "target": [1.0, 1.0], "spf": [1.0, 1.0]}
    )
    pvf = pd.DataFrame(
        [
            {"species": 1, "pu": 1, "amount": 4.0},
            {"species": 1, "pu": 2, "amount": 2.0},
            {"species": 2, "pu": 3, "amount": 5.0},
            {"species": 2, "pu": 4, "amount": 1.0},
            {"species": 2, "pu": 5, "amount": 3.0},
            {"species": 2, "pu": 6, "amount": 2.0},
        ]
    )
    p = ConservationProblem(pu, feats, pvf)
    _assert_equal_results(
        rank_removal(p, rule=rule, weights={2: 3.5}),
        _dense_rank_removal(p, rule=rule, weights={2: 3.5}),
    )


def test_equivalence_edge_shapes() -> None:
    # n_feat == 0 -> pure index order; and a PU with no features at all.
    pu = pd.DataFrame({"id": [1, 2, 3], "cost": [1.0, 2.0, 3.0], "status": [0, 0, 0]})
    feats0 = pd.DataFrame({"id": [], "name": [], "target": [], "spf": []})
    pvf0 = pd.DataFrame(columns=["species", "pu", "amount"])
    p0 = ConservationProblem(pu, feats0, pvf0)
    _assert_equal_results(rank_removal(p0), _dense_rank_removal(p0))

    feats1 = pd.DataFrame({"id": [1], "name": ["a"], "target": [1.0], "spf": [1.0]})
    pvf1 = pd.DataFrame([{"species": 1, "pu": 2, "amount": 3.0}])  # PU 1,3 featureless
    p1 = ConservationProblem(pu, feats1, pvf1)
    for rule in ("caz", "abf"):
        _assert_equal_results(
            rank_removal(p1, rule=rule), _dense_rank_removal(p1, rule=rule)
        )


def test_equivalence_duplicate_pvf_rows() -> None:
    pu = pd.DataFrame({"id": [1, 2], "cost": [1.0, 1.0], "status": [0, 0]})
    feats = pd.DataFrame({"id": [1], "name": ["a"], "target": [1.0], "spf": [1.0]})
    pvf = pd.DataFrame(
        [
            {"species": 1, "pu": 1, "amount": 2.0},
            {"species": 1, "pu": 1, "amount": 3.0},  # duplicate (pu, species): sums
            {"species": 1, "pu": 2, "amount": 4.0},
        ]
    )
    p = ConservationProblem(pu, feats, pvf)
    _assert_equal_results(rank_removal(p), _dense_rank_removal(p))


def test_tie_break_pinned() -> None:
    # All amounts and costs equal -> every delta ties -> pure PU-index order,
    # in every batch, for both rules (design §9.3: guards the argpartition
    # boundary logic).
    q_rows = [[2.0, 2.0]] * 9
    pu = pd.DataFrame({"id": list(range(1, 10)), "cost": [1.0] * 9, "status": [0] * 9})
    feats = pd.DataFrame(
        {"id": [1, 2], "name": ["a", "b"], "target": [1.0, 1.0], "spf": [1.0, 1.0]}
    )
    rows = [
        {"species": fid, "pu": i + 1, "amount": q_rows[i][j]}
        for i in range(9)
        for j, fid in enumerate([1, 2])
    ]
    p = ConservationProblem(pu, feats, pd.DataFrame(rows))
    for rule in ("caz", "abf"):
        for warp in (1, 4, 9):
            res = rank_removal(p, rule=rule, warp=warp)
            _assert_equal_results(res, _dense_rank_removal(p, rule=rule, warp=warp))
            assert res.removal_order == list(range(1, 10))


@pytest.mark.parametrize("rule", ["caz", "abf"])
def test_float_amounts_fixed_seed_exact(rule: str) -> None:
    # Float amounts AND float costs. The only order-affecting FP boundary is
    # the initial-Q summation order (design §7); on this fixed seed no near-tie
    # flips, so order equality is exact and deterministic. If a numpy/scipy
    # upgrade ever flips it: change the seed, never loosen (a rank-tolerance
    # band would be unprincipled — a mid-run flip cascades arbitrarily,
    # review finding #13). atol covers exact-zero terminal curve rows against
    # the float-cost accumulator drift (finding #7).
    p = _random_problem(11, integer=False)
    a = rank_removal(p, rule=rule, warp=2)
    b = _dense_rank_removal(p, rule=rule, warp=2)
    assert a.removal_order == b.removal_order
    assert a.priority_rank == b.priority_rank
    np.testing.assert_allclose(
        a.performance_curves.to_numpy(),
        b.performance_curves.to_numpy(),
        rtol=1e-9,
        atol=1e-12,
    )


def test_equivalence_stored_zero_row() -> None:
    # Explicit amount=0.0 pvf rows become stored zeros in the CSR;
    # eliminate_zeros() strips them so they can't mark features dirty
    # (review finding #14 — this was dead-path before this fixture).
    pu = pd.DataFrame({"id": [1, 2, 3], "cost": [1.0, 1.0, 1.0], "status": [0, 0, 0]})
    feats = pd.DataFrame(
        {"id": [1, 2], "name": ["a", "b"], "target": [1.0, 1.0], "spf": [1.0, 1.0]}
    )
    pvf = pd.DataFrame(
        [
            {"species": 1, "pu": 1, "amount": 2.0},
            {"species": 1, "pu": 2, "amount": 0.0},  # explicit stored zero
            {"species": 2, "pu": 2, "amount": 3.0},
            {"species": 2, "pu": 3, "amount": 1.0},
        ]
    )
    p = ConservationProblem(pu, feats, pvf)
    for rule in ("caz", "abf"):
        _assert_equal_results(
            rank_removal(p, rule=rule), _dense_rank_removal(p, rule=rule)
        )


@pytest.mark.parametrize("rule", ["caz", "abf"])
def test_equivalence_wide_feature_matrix(rule: str) -> None:
    # Production-width regime (10-100 features; review finding #8). Integer
    # amounts stay EXACT for both rules under the chunked-dense kernel.
    p = _random_problem(5, n_pu=60, n_feat=25)
    for warp in (1, 6):
        _assert_equal_results(
            rank_removal(p, rule=rule, warp=warp),
            _dense_rank_removal(p, rule=rule, warp=warp),
        )


@pytest.mark.parametrize("rule", ["caz", "abf"])
def test_equivalence_smoothing_path(rule: str) -> None:
    # The smoothing path (dense -> smoothing.apply -> csr -> sparse engine)
    # needs oracle coverage too (review finding #5). Smoothed matrices are
    # float -> fixed-seed exact per the §7 caveat (flip => change seed).
    rng = np.random.default_rng(17)
    p = _random_problem(17, n_pu=18, n_feat=4)
    spec = SmoothingSpec(alpha=0.5, coords=rng.uniform(0, 10, size=(18, 2)))
    _assert_equal_results(
        rank_removal(p, rule=rule, smoothing=spec),
        _dense_rank_removal(p, rule=rule, smoothing=spec),
    )
```

- [ ] **Step 2: Run the suite — every test must PASS against the current dense src engine**

Run: `/opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/zonation/test_rank_removal_scale.py -v`
Expected: ALL PASS (the sparse `rank_removal` doesn't exist yet — `rank_removal` IS the dense engine, and the oracle is its copy). If anything fails, fix the oracle/builder — do NOT touch `src`.

- [ ] **Step 3: Commit**

```bash
git add tests/pymarxan/zonation/test_rank_removal_scale.py
git commit -m "test(zonation): characterization oracle + equivalence suite for rank_removal"
```

---

### Task 2: Guards — negative-input validation, smoothing cap, warp advisory

TDD, failing-first. All three land in the still-dense engine so the Task 3 rewrite diff is pure algorithm. The advisory is a module-level helper tested directly (S3b `_warn_if_large_mip` precedent) — an integration run at 50k PUs would take minutes on the dense engine.

**Files:**
- Modify: `src/pymarxan/zonation/rank_removal.py`
- Test: `tests/pymarxan/zonation/test_rank_removal_scale.py` (append)

**Interfaces:**
- Produces: module constants `_SMOOTHING_MAX_PU = 50_000`, `_WARP_ADVISORY_MIN_PU = 50_000`; helpers `_warn_if_small_warp(n_pu: int, warp: int) -> None` and `_validate_inputs(problem: ConservationProblem, weights: dict[int, float] | None) -> None`. Task 3 keeps all of them and the call sites verbatim.

- [ ] **Step 1: Append failing tests**

```python
# --- Task 2: guards -------------------------------------------------------
# HIGH review finding #1: `from pymarxan.zonation import rank_removal` binds
# the FUNCTION (zonation/__init__.py re-exports it, shadowing the submodule of
# the same name — even `import pymarxan.zonation.rank_removal as m` binds the
# function). importlib is the only way to get the module object.
import importlib

rr_module = importlib.import_module("pymarxan.zonation.rank_removal")


def _pvf_problem(pvf_rows: list[dict]) -> ConservationProblem:
    pu = pd.DataFrame({"id": [1, 2], "cost": [1.0, 1.0], "status": [0, 0]})
    feats = pd.DataFrame({"id": [1], "name": ["a"], "target": [1.0], "spf": [1.0]})
    return ConservationProblem(pu, feats, pd.DataFrame(pvf_rows))


def test_negative_amount_raises() -> None:
    p = _pvf_problem(
        [
            {"species": 1, "pu": 1, "amount": 2.0},
            {"species": 1, "pu": 2, "amount": -1.0},
        ]
    )
    with pytest.raises(ValueError, match="amounts must be >= 0"):
        rank_removal(p)


def test_negative_amount_cancelling_duplicate_raises() -> None:
    # Raw-input validation (review finding #4): a -5 that a +5 duplicate sums
    # to zero in the matrix must STILL raise — the contract is on raw amounts.
    p = _pvf_problem(
        [
            {"species": 1, "pu": 1, "amount": 5.0},
            {"species": 1, "pu": 1, "amount": -5.0},
            {"species": 1, "pu": 2, "amount": 1.0},
        ]
    )
    with pytest.raises(ValueError, match="amounts must be >= 0"):
        rank_removal(p)


def test_nan_amount_raises() -> None:
    # NaN passes every `< 0` guard and would stall the sparse selection loop
    # (review finding #2, correctness-critical) — reject up front.
    p = _pvf_problem(
        [
            {"species": 1, "pu": 1, "amount": 2.0},
            {"species": 1, "pu": 2, "amount": float("nan")},
        ]
    )
    with pytest.raises(ValueError, match="amounts must be finite"):
        rank_removal(p)


def test_nan_cost_raises() -> None:
    p = _random_problem(0)
    p.planning_units.loc[0, "cost"] = float("nan")
    with pytest.raises(ValueError, match="costs must be finite"):
        rank_removal(p)


def test_negative_or_nan_weight_raises() -> None:
    p = _random_problem(0)
    with pytest.raises(ValueError, match="weights must be >= 0"):
        rank_removal(p, weights={1: -2.0})
    with pytest.raises(ValueError, match="weights must be finite"):
        rank_removal(p, weights={1: float("nan")})


def test_zero_pu_raises() -> None:
    # Review finding #12: dense returned a NaN curve row; sparse would
    # ZeroDivisionError — a clear ValueError beats both.
    pu = pd.DataFrame({"id": [], "cost": [], "status": []})
    feats = pd.DataFrame({"id": [], "name": [], "target": [], "spf": []})
    pvf = pd.DataFrame(columns=["species", "pu", "amount"])
    p = ConservationProblem(pu, feats, pvf)
    with pytest.raises(ValueError, match="at least one planning unit"):
        rank_removal(p)


def test_smoothing_capped_at_vector_scale(monkeypatch: pytest.MonkeyPatch) -> None:
    # Guard reads n_planning_units and must fire BEFORE any kernel/matrix work.
    p = _random_problem(0)
    monkeypatch.setattr(
        type(p), "n_planning_units", property(lambda self: 50_001)
    )
    spec = SmoothingSpec(alpha=1.0, coords=np.zeros((80, 2)))
    with pytest.raises(ValueError, match="vector-scale"):
        rank_removal(p, smoothing=spec)


def test_warp_advisory_helper() -> None:
    with pytest.warns(UserWarning, match="warp"):
        rr_module._warn_if_small_warp(1_000_000, 50)
    import warnings as _warnings

    for n_pu, warp in ((1_000_000, 1000), (10_000, 1), (50_000, 1)):
        with _warnings.catch_warnings():
            _warnings.simplefilter("error")
            rr_module._warn_if_small_warp(n_pu, warp)  # must not warn


def test_warp_advisory_called_from_rank_removal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[int, int]] = []
    monkeypatch.setattr(
        rr_module, "_warn_if_small_warp", lambda n_pu, warp: calls.append((n_pu, warp))
    )
    p = _random_problem(0, n_pu=12)
    rank_removal(p, warp=3)
    assert calls == [(12, 3)]
```

Note on `test_smoothing_capped_at_vector_scale`: `n_planning_units` IS a `@property` (`problem.py:66`) and the grounding review verified this exact monkeypatch works verbatim, including undo. The smoothing-cap check reads it BEFORE any matrix build, but `_validate_inputs` runs first and also reads it — the property fake (50_001) satisfies both since the fixture has real pvf rows.

- [ ] **Step 2: Run to verify the five tests fail**

Run: `/opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/zonation/test_rank_removal_scale.py -v -k "negative or nan or zero_pu or smoothing_capped or advisory"`
Expected: 9 FAIL (`AttributeError: _warn_if_small_warp` / no raise / no warn). If any
of them errors during COLLECTION instead, fix the test file first.

- [ ] **Step 3: Implement the guards in `rank_removal.py`**

Add after the imports:

```python
import warnings

_SMOOTHING_MAX_PU = 50_000
_WARP_ADVISORY_MIN_PU = 50_000


def _warn_if_small_warp(n_pu: int, warp: int) -> None:
    """Advise (warn-and-proceed, S3b precedent) when warp is too small to scale.

    warp=1 selection alone is O(n^2) at raster scale regardless of sparse
    rescoring. Large warp (10-100) is documented Zonation practice as a
    computation-time vs solution-refinement trade-off; ``warp ~ n_pu/1000`` is
    pymarxan performance advice for million-cell grids, not a Zonation norm.
    Silence with ``warnings.filterwarnings`` when a small warp is deliberate.
    """
    if n_pu > _WARP_ADVISORY_MIN_PU and warp < n_pu // 10_000:
        warnings.warn(
            f"rank_removal with n_pu={n_pu} and warp={warp} will be slow: "
            f"larger warp trades solution refinement for speed (documented "
            f"Zonation practice: 10-100); consider warp≈{n_pu // 1000}. "
            "Silence via warnings.filterwarnings if deliberate.",
            stacklevel=3,
        )
```

In `rank_removal`, immediately after the `rule` check:

```python
    n_pu_total = problem.n_planning_units
    if smoothing is not None and n_pu_total > _SMOOTHING_MAX_PU:
        raise ValueError(
            f"smoothing builds a dense {n_pu_total}x{n_pu_total} kernel and is "
            f"vector-scale only (n_pu <= {_SMOOTHING_MAX_PU}); raster-scale "
            "distribution smoothing (grid convolution) is a planned follow-up."
        )
```

After the existing `warp = max(1, min(int(warp), max(n_pu, 1)))` line:

```python
    _warn_if_small_warp(n_pu, warp)
```

Add a module-level validation helper (raw inputs — review findings #2/#4: NaN
passes every `< 0` guard, and raw-input checking keeps the contract identical
across the plain and smoothing paths and across the dense/sparse engine eras):

```python
def _validate_inputs(
    problem: ConservationProblem, weights: dict[int, float] | None
) -> None:
    """Raise ValueError for inputs the engine cannot rank meaningfully.

    Raw (pre-duplicate-sum, pre-smoothing) amounts are checked so the plain and
    smoothing paths enforce one contract. NaNs must be rejected up front: they
    pass every ``< 0`` comparison and would stall the removal loop. Negative
    weights are a real Zonation v3+ workflow (opportunity-cost features;
    Moilanen et al. 2011, doi:10.1890/10-1865.1) but are not yet supported.
    """
    if problem.n_planning_units == 0:
        raise ValueError("rank_removal requires at least one planning unit")
    amt = problem.pu_vs_features["amount"].to_numpy(dtype=float)
    if amt.size and not np.isfinite(amt).all():
        raise ValueError("feature amounts must be finite for rank_removal")
    if amt.size and (amt < 0).any():
        raise ValueError("feature amounts must be >= 0 for rank_removal")
    if weights:
        wv = np.asarray(list(weights.values()), dtype=float)
        if not np.isfinite(wv).all():
            raise ValueError("feature weights must be finite for rank_removal")
        if (wv < 0).any():
            raise ValueError(
                "feature weights must be >= 0 for rank_removal (negative "
                "weights, used by Zonation v3+ for opportunity-cost features, "
                "are not yet supported)"
            )
```

Call it in `rank_removal` immediately after the `rule` check (BEFORE the
smoothing guard — the 0-PU/NaN checks must win over everything). Finally, in
the `use_cost` branch, extend the existing cost check:

```python
        c = problem.planning_units["cost"].to_numpy().astype(float)
        if not np.isfinite(c).all():
            raise ValueError("planning-unit costs must be finite for rank_removal")
        if np.any(c <= 0):
            raise ValueError("use_cost=True requires every planning-unit cost > 0")
```

(Identical code survives Task 3 unchanged — the validation never moves.)

- [ ] **Step 4: Run the whole scale test file + existing zonation tests — all green**

Run: `/opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/zonation/ -v`
Expected: ALL PASS (guards added, equivalence untouched — the oracle has no guards, but no equivalence fixture uses negative values or >50k PUs).

- [ ] **Step 5: Commit**

```bash
git add src/pymarxan/zonation/rank_removal.py tests/pymarxan/zonation/test_rank_removal_scale.py
git commit -m "feat(zonation): input validation + smoothing cap + warp advisory for rank_removal"
```

---

### Task 3: The sparse + dirty-set engine rewrite

**Files:**
- Modify: `src/pymarxan/zonation/rank_removal.py` (full internals rewrite)
- Test: `tests/pymarxan/zonation/test_rank_removal_scale.py` (append no-dense test)

**Interfaces:**
- Consumes: `ConservationProblem.build_pu_feature_csr()` (scipy CSR, rows = PU order, cols = feature order, canonical/sorted, `toarray()` == dense builder), guards from Task 2.
- Produces: same public `rank_removal`; private keyword `_force_full_rescore: bool = False` (test-only, Task 4).

- [ ] **Step 1: Append the failing no-dense test**

```python
# --- Task 3: the sparse engine must never densify -------------------------
def test_no_dense_matrix_without_smoothing(monkeypatch: pytest.MonkeyPatch) -> None:
    p = _random_problem(3)
    def _boom(self: ConservationProblem) -> None:
        raise AssertionError("dense build_pu_feature_matrix called in sparse path")
    monkeypatch.setattr(ConservationProblem, "build_pu_feature_matrix", _boom)
    res = rank_removal(p, rule="caz", warp=2)
    assert len(res.removal_order) == p.n_planning_units
```

- [ ] **Step 2: Run to verify it fails**

Run: `/opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/zonation/test_rank_removal_scale.py::test_no_dense_matrix_without_smoothing -v`
Expected: FAIL with the AssertionError (current engine densifies).

- [ ] **Step 3: Rewrite the engine.** Replace the body of `rank_removal` below the guards; keep the module docstring's science paragraphs, replace its scaling paragraph. Full new `rank_removal.py` — the Task-2 helpers `_warn_if_small_warp` (shown) and `_validate_inputs` (defined in Task 2, retained verbatim between the two shown functions) both carry over unchanged:

```python
"""Zonation CAZ/ABF rank-removal engine (Moilanen et al. 2005; Moilanen 2007).

Distinct from ``pymarxan.analysis.rank_importance`` (Jung et al. 2021), which
ranks only the *selected* PUs of an existing solution by Marxan-objective
increase; this ranks *every* PU from the whole landscape by proportional
biological loss.

Note: real Zonation additionally restricts removal candidates to *edge* cells
(8-neighbour adjacency to already-removed area) — both an ecological choice and
a major speedup; this engine considers all remaining cells, a deliberate
v0.13-era difference, unchanged here.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from pymarxan.models.problem import (
    STATUS_LOCKED_IN,
    STATUS_LOCKED_OUT,
    ConservationProblem,
)
from pymarxan.zonation.result import ZonationResult
from pymarxan.zonation.smoothing import SmoothingSpec

_SMOOTHING_MAX_PU = 50_000
_WARP_ADVISORY_MIN_PU = 50_000
_RESCORE_CHUNK = 32_768


def _warn_if_small_warp(n_pu: int, warp: int) -> None:
    """Advise (warn-and-proceed, S3b precedent) when warp is too small to scale.

    warp=1 selection alone is O(n^2) at raster scale regardless of sparse
    rescoring. Large warp (10-100) is documented Zonation practice as a
    computation-time vs solution-refinement trade-off; ``warp ~ n_pu/1000`` is
    pymarxan performance advice for million-cell grids, not a Zonation norm.
    Silence with ``warnings.filterwarnings`` when a small warp is deliberate.
    """
    if n_pu > _WARP_ADVISORY_MIN_PU and warp < n_pu // 10_000:
        warnings.warn(
            f"rank_removal with n_pu={n_pu} and warp={warp} will be slow: "
            f"larger warp trades solution refinement for speed (documented "
            f"Zonation practice: 10-100); consider warp≈{n_pu // 1000}. "
            "Silence via warnings.filterwarnings if deliberate.",
            stacklevel=3,
        )


def rank_removal(
    problem: ConservationProblem,
    *,
    rule: str = "caz",
    weights: dict[int, float] | None = None,
    warp: int = 1,
    use_cost: bool = True,
    smoothing: SmoothingSpec | None = None,
    _force_full_rescore: bool = False,
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

    Scaling: the engine is sparse and incremental. Per batch it rescores only
    cells whose features' remaining totals changed (dirty set) — via chunked
    dense row buffers that reuse the reference engine's exact expressions, so
    per-row scores are bitwise-identical to the pre-rewrite engine given
    identical remaining totals — selects the ``warp`` smallest (ties by PU
    index) via partition, and updates totals, cost and curves incrementally.
    Init is O(nnz); million-cell rasters rank in minutes at raster-appropriate
    ``warp`` (an advisory warns; silence via ``warnings.filterwarnings``).
    Equivalence to the reference engine: exact removal order for BOTH rules on
    integer amounts (while sums stay below 2**53); float amounts (including
    smoothed matrices) can differ only via initial-total summation order (a few
    ULPs), which can flip exact float near-ties; float costs affect curve
    values only. ``ValueError`` on invalid input: negative or non-finite
    amounts/weights (negative weights — a Zonation v3+ opportunity-cost
    workflow — are not yet supported), non-finite costs, zero planning units.
    Smoothing stays vector-scale (n_pu <= 50_000).
    ``_force_full_rescore`` is test-only: it disables the dirty-set shortcut.
    """
    if rule not in ("caz", "abf"):
        raise ValueError(f"rule must be 'caz' or 'abf', got {rule!r}")
    _validate_inputs(problem, weights)

    n_pu_total = problem.n_planning_units
    if smoothing is not None and n_pu_total > _SMOOTHING_MAX_PU:
        raise ValueError(
            f"smoothing builds a dense {n_pu_total}x{n_pu_total} kernel and is "
            f"vector-scale only (n_pu <= {_SMOOTHING_MAX_PU}); raster-scale "
            "distribution smoothing (grid convolution) is a planned follow-up."
        )

    from scipy.sparse import csr_matrix

    if smoothing is not None:
        q = csr_matrix(smoothing.apply(problem.build_pu_feature_matrix()))
    else:
        q = problem.build_pu_feature_csr()
    q.eliminate_zeros()  # freshly built and ours: stored zeros must not mark dirty
    n_pu, n_feat = q.shape
    csc = q.tocsc()
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
        if not np.isfinite(c).all():
            raise ValueError("planning-unit costs must be finite for rank_removal")
        if np.any(c <= 0):
            raise ValueError("use_cost=True requires every planning-unit cost > 0")
    else:
        c = np.ones(n_pu, dtype=float)

    warp = max(1, min(int(warp), max(n_pu, 1)))
    _warn_if_small_warp(n_pu, warp)

    remaining = np.ones(n_pu, dtype=bool)
    n_remaining = n_pu
    Q = np.asarray(q.sum(axis=0)).ravel()
    T = Q.copy()
    T_safe = np.where(T > 0, T, 1.0)
    cost_total = float(c.sum()) if c.sum() > 0 else 1.0
    cost_remaining = float(c.sum())
    delta = np.zeros(n_pu, dtype=float)
    dirty = np.ones(n_pu, dtype=bool)

    removal_order: list[int] = []
    curve_rows: list[dict] = []

    def record_curve() -> None:
        retained = np.where(T > 0, Q / T_safe, 1.0)
        row: dict = {
            "prop_landscape_remaining": n_remaining / n_pu,
            # max(): float-cost sequential drift can leave a tiny negative
            # residual at run end (design §7 site 2); exact for integer costs.
            "prop_cost_remaining": max(cost_remaining, 0.0) / cost_total,
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

    def rescore(rows: np.ndarray) -> None:
        """Recompute delta for the given rows from current Q.

        Chunked-dense kernel (design-review #9): reuse the reference engine's
        exact expressions on (chunk, n_feat) row buffers, so per-row scores are
        bitwise-identical to the dense engine given identical Q — numpy's
        per-row reduction depends only on the row, never on chunk shape.
        """
        if rows.size == 0:
            return
        if n_feat == 0:
            delta[rows] = 0.0
            dirty[rows] = False
            return
        Q_safe = np.where(Q > 0, Q, 1.0)
        fac = w / Q_safe
        dead = Q <= 0
        for s in range(0, rows.size, _RESCORE_CHUNK):
            chunk = rows[s : s + _RESCORE_CHUNK]
            r = q[chunk].toarray() * fac
            r[:, dead] = 0.0
            out = r.max(axis=1) if rule == "caz" else r.sum(axis=1)
            delta[chunk] = out / c[chunk]
        dirty[rows] = False

    indptr, indices, data = q.indptr, q.indices, q.data

    while n_remaining > 0:
        cand = candidate_indices()  # ascending PU-index order
        stale = cand if _force_full_rescore else cand[dirty[cand]]
        rescore(stale)
        d = delta[cand]
        k = min(warp, cand.size)
        if k == cand.size:
            sel = np.argsort(d, kind="stable")
        else:
            part = np.argpartition(d, k - 1)
            v = d[part[k - 1]]
            below = np.flatnonzero(d < v)
            ties = np.flatnonzero(d == v)
            sel = np.concatenate([below, ties[: k - below.size]])
            sel = sel[np.argsort(d[sel], kind="stable")]  # emission: (delta, index)
        removed = cand[sel]
        changed_parts: list[np.ndarray] = []
        for idx in removed:
            removal_order.append(int(pu_ids[idx]))
            remaining[idx] = False
            s, e = indptr[idx], indptr[idx + 1]
            Q[indices[s:e]] -= data[s:e]  # sequential, matching the dense engine
            changed_parts.append(indices[s:e])
            cost_remaining -= float(c[idx])
        n_remaining -= removed.size
        if changed_parts:
            changed = np.unique(np.concatenate(changed_parts))
            holders = np.concatenate(
                [csc.indices[csc.indptr[j] : csc.indptr[j + 1]] for j in changed]
            ) if changed.size else np.zeros(0, dtype=np.intp)
            dirty[holders] = True
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

Implementation notes (why each subtle line is the way it is — keep these true):
- The rescore kernel MUST mirror the dense engine's expressions exactly
  (`toarray()` row buffer × precomputed `w/Q_safe`, then `r[:, Q<=0]=0`, then
  `max`/`sum(axis=1)`, then `/c`) — that is what makes per-row scores
  bitwise-identical given identical `Q` and lets equivalence tests assert exact
  order on integer amounts for BOTH rules. Do not "optimize" it back to
  reduceat over CSR data: the review empirically reproduced integer-ABF order
  flips from reduceat's different summation association.
- Featureless rows densify to all-zero rows → `max`/`sum` give 0.0, matching the
  dense engine; `n_feat == 0` short-circuits (a `(m, 0)` reduction would raise).
- `Q[indices[s:e]] -= data[s:e]` per removed cell IN EMISSION ORDER reproduces the
  dense engine's sequential `Q -= q[idx]` trajectory (subtracting absent features'
  zeros is a bitwise no-op).
- Selection: `below` and `ties` come from `flatnonzero` (ascending local index ==
  ascending PU index since `cand` is ascending); the final stable argsort makes
  emission order (delta, PU index), matching the dense stable argsort exactly.
- `dirty` may include already-removed cells — harmless: rescoring keys off
  `cand`, which is filtered by `remaining`.

- [ ] **Step 4: Run the ENTIRE zonation test dir + solver + smoothing tests**

Run: `/opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/zonation/ tests/pymarxan/solvers/test_zonation_solver.py -v`
(Adjust the solver test path if it differs — find it with `grep -rl ZonationSolver tests/`.)
Expected: ALL PASS, including every Task-1 characterization test and the no-dense test. If an equivalence test fails, debug the sparse engine against the oracle on the failing seed (print both removal orders; the first divergence pinpoints the bug) — the oracle is ground truth, never adjust it.

- [ ] **Step 5: Commit**

```bash
git add src/pymarxan/zonation/rank_removal.py tests/pymarxan/zonation/test_rank_removal_scale.py
git commit -m "feat(zonation): sparse + dirty-set incremental rank_removal engine"
```

---

### Task 4: Dirty-set internal check + scale bench

**Files:**
- Test: `tests/pymarxan/zonation/test_rank_removal_scale.py` (append)
- Create: `tests/benchmarks/bench_zonation.py`

**Interfaces:**
- Consumes: `_force_full_rescore` kwarg (Task 3), `_random_problem`/`_assert_equal_results` (Task 1).

- [ ] **Step 1: Append the dirty-vs-full test**

```python
# --- Task 4: dirty-set shortcut === full rescore --------------------------
@pytest.mark.parametrize("integer", [True, False])
@pytest.mark.parametrize("rule", ["caz", "abf"])
def test_dirty_set_equals_full_rescore(rule: str, integer: bool) -> None:
    # Bit-identical by construction (design §4.4): a clean cell's inputs are
    # unchanged, so skipping its rescore cannot change any value. The FLOAT
    # case is where this test has power the dense oracle doesn't (review
    # finding #6): dense-vs-sparse is only ULP-close there, but dirty-vs-full
    # within the sparse engine must stay exactly equal.
    p = _random_problem(21, n_pu=120, n_feat=8, statuses=True, integer=integer)
    for warp in (1, 5):
        _assert_equal_results(
            rank_removal(p, rule=rule, warp=warp),
            rank_removal(p, rule=rule, warp=warp, _force_full_rescore=True),
        )
```

- [ ] **Step 2: Run it**

Run: `/opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/zonation/test_rank_removal_scale.py::test_dirty_set_equals_full_rescore -v`
Expected: PASS. (If it fails, the dirty-marking misses an invalidation path — check that EVERY removed cell's features mark ALL their holders.)

- [ ] **Step 3: Create the bench** (excluded from CI; the perf budget is the point):

```python
"""Perf budget: raster-scale rank_removal (design §10). Run deliberately:
/opt/micromamba/envs/shiny/bin/pytest tests/benchmarks/bench_zonation.py -m bench -v
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import pytest

from pymarxan.models.problem import ConservationProblem
from pymarxan.zonation.rank_removal import rank_removal

pytestmark = pytest.mark.bench


def _grid_problem(side: int, n_feat: int = 8, seed: int = 0) -> ConservationProblem:
    """side x side grid; each feature occupies a random compact block (locality)."""
    rng = np.random.default_rng(seed)
    n_pu = side * side
    pu_ids = np.arange(1, n_pu + 1)
    planning_units = pd.DataFrame(
        {"id": pu_ids, "cost": np.ones(n_pu), "status": np.zeros(n_pu, dtype=int)}
    )
    feat_ids = list(range(1, n_feat + 1))
    features = pd.DataFrame(
        {
            "id": feat_ids,
            "name": [f"f{j}" for j in feat_ids],
            "target": [1.0] * n_feat,
            "spf": [1.0] * n_feat,
        }
    )
    frames = []
    for fid in feat_ids:
        r0, c0 = rng.integers(0, side // 2, size=2)
        h, w_ = rng.integers(side // 4, side // 2, size=2)
        rr, cc = np.meshgrid(
            np.arange(r0, min(r0 + h, side)), np.arange(c0, min(c0 + w_, side)),
            indexing="ij",
        )
        cells = (rr * side + cc).ravel() + 1
        amounts = rng.integers(1, 5, size=cells.size).astype(float)
        frames.append(
            pd.DataFrame({"species": fid, "pu": cells, "amount": amounts})
        )
    return ConservationProblem(planning_units, features, pd.concat(frames))


def test_rank_removal_scale_budget() -> None:
    p = _grid_problem(300)  # 90_000 cells
    t0 = time.perf_counter()
    res = rank_removal(p, rule="caz", warp=90)  # n // 1000 batches
    elapsed = time.perf_counter() - t0
    assert len(res.removal_order) == 90_000
    # Generous budget for slow machines; the dense engine takes >> this.
    assert elapsed < 60.0, f"raster-scale rank_removal took {elapsed:.1f}s"
```

- [ ] **Step 4: Run the bench deliberately (not in CI)**

Run: `/opt/micromamba/envs/shiny/bin/pytest tests/benchmarks/bench_zonation.py -m bench -v`
Expected: PASS well under budget (target: seconds). Record the elapsed time in the commit message.

- [ ] **Step 5: Commit**

```bash
git add tests/pymarxan/zonation/test_rank_removal_scale.py tests/benchmarks/bench_zonation.py
git commit -m "test(zonation): dirty-set invariance check + raster-scale perf bench"
```

---

### Task 5: Docs, CHANGELOG, full gate

**Files:**
- Modify: `CHANGELOG.md` (`[Unreleased]` section)
- Modify: `.github/copilot-instructions.md` (only if its solver matrix calls zonation vector-scale — check first)

- [ ] **Step 1: CHANGELOG entry** under `## [Unreleased]` (create the `### Changed` subsection if absent):

```markdown
### Changed
- `zonation.rank_removal` rewritten sparse + incremental (dirty-set rescoring via
  chunked dense row buffers, partition selection, incremental curves): million-cell
  rasters now rank in minutes at raster-appropriate `warp` (an advisory warns when
  warp is small; silence via `warnings.filterwarnings`). Results match the previous
  engine exactly on integer amounts for both rules; float amounts can differ only
  at exact float near-ties (initial-total summation order), float costs affect
  curve values only. Invalid inputs now raise `ValueError`: negative or non-finite
  feature amounts, negative or non-finite weights (negative weights — a Zonation
  v3+ opportunity-cost workflow — not yet supported), non-finite costs, 0-PU
  problems. `smoothing` is capped at 50k PUs pending grid-convolution smoothing.
```

- [ ] **Step 2: Check copilot-instructions**

Run: `grep -n -i "zonation" .github/copilot-instructions.md`
If any line claims vector-scale-only, update it to mention the sparse engine + warp knob; otherwise leave untouched.

- [ ] **Step 3: Full gate**

Run: `source /opt/micromamba/etc/profile.d/micromamba.sh && micromamba activate shiny && make check`
Expected: lint + types + full suite green (~1826 + ~20 new). `test_solutions_are_different` may flake — rerun once before investigating.

- [ ] **Step 4: Commit**

```bash
git add CHANGELOG.md .github/copilot-instructions.md
git commit -m "docs(zonation): changelog + doc updates for raster-scale rank_removal"
```

---

## Self-review checklist (done at write time)

- Spec coverage: §3→T3, §4→T3, §5→T2, §6→T2, §7→T1 (contract encoded in assertions), §9.1–9.3→T1, §9.4→T4, §9.5→T2, §9.6→T3 no-dense, §9.7→T4 bench, §9.8→T5 make check, §11→T5. `eliminate_zeros` question (§12) resolved in T3 (own fresh matrix).
- No placeholders; all code complete.
- Type consistency: `_warn_if_small_warp(n_pu, warp)` used identically in T2/T3; `_force_full_rescore` defined T3, consumed T4; builders defined T1, consumed T3/T4.
- (Resolved by design review 2026-08-08, synthesis in `2026-08-08-zonation-raster-scale-review.md`: rescore kernel switched to chunked-dense — integer-ABF now exact, dissolving the earlier §9.2 deviation; raw-input + finiteness validation; oracle smoothing branch restored; `importlib` module-binding fix in Task 2; float test asserts exact order on a fixed seed. This plan reflects the post-review state.)
