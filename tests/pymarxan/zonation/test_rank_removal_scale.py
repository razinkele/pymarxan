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

import importlib

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


# --- Task 2: guards -------------------------------------------------------
# HIGH review finding #1: `from pymarxan.zonation import rank_removal` binds
# the FUNCTION (zonation/__init__.py re-exports it, shadowing the submodule of
# the same name — even `import pymarxan.zonation.rank_removal as m` binds the
# function). importlib is the only way to get the module object.

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

    for n_pu, warp in ((1_000_000, 1000), (10_000, 1), (50_000, 1), (1_000_000, 1)):
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


# --- Task 3: the sparse engine must never densify -------------------------
def test_no_dense_matrix_without_smoothing(monkeypatch: pytest.MonkeyPatch) -> None:
    p = _random_problem(3)

    def _boom(self: ConservationProblem) -> None:
        raise AssertionError("dense build_pu_feature_matrix called in sparse path")

    monkeypatch.setattr(ConservationProblem, "build_pu_feature_matrix", _boom)
    res = rank_removal(p, rule="caz", warp=2)
    assert len(res.removal_order) == p.n_planning_units


def test_subnormal_amount_raises_instead_of_hanging() -> None:
    # A subnormal amount (5e-324, the smallest positive double) is finite and
    # >= 0 so _validate_inputs admits it, but it makes Q_j subnormal too, so
    # fac = w / Q_safe overflows to +inf. Rows that do NOT hold feature 1
    # (PUs 2-4) then compute 0.0 * inf = NaN there; once PU 1 (the only
    # feature-1 holder) is removed in batch 1, PUs 2-4 are never marked dirty
    # again (they hold nothing feature 1's removal touches), so their NaN
    # delta is permanently stale. In batch 2 every candidate delta is NaN, so
    # both `below` and `ties` (NaN < / == NaN is always False) are empty and
    # `removed.size == 0` forever -- a silent non-terminating loop without the
    # guard. The dense oracle does NOT hang on this input, but only because
    # a full argsort tie-breaks NaN-vs-NaN by raw PU index, silently returning
    # a garbage order ([1, 2, 3, 4], unrelated to the undefined deltas) rather
    # than reporting anything is wrong -- a loud RuntimeError here is strictly
    # better than either engine's alternative.
    pu = pd.DataFrame({"id": [1, 2, 3, 4], "cost": [1.0] * 4, "status": [0] * 4})
    feats = pd.DataFrame(
        {"id": [1, 2], "name": ["a", "b"], "target": [1.0, 1.0], "spf": [1.0, 1.0]}
    )
    pvf = pd.DataFrame(
        [
            {"species": 1, "pu": 1, "amount": 5e-324},
            {"species": 2, "pu": 2, "amount": 1.0},
            {"species": 2, "pu": 3, "amount": 1.0},
            {"species": 2, "pu": 4, "amount": 1.0},
        ]
    )
    p = ConservationProblem(pu, feats, pvf)
    for rule in ("caz", "abf"):
        with pytest.raises(RuntimeError, match="no progress"):
            rank_removal(p, rule=rule, warp=1)


@pytest.mark.parametrize("rule", ["caz", "abf"])
def test_cost_curve_never_negative_float_costs(rule: str) -> None:
    # The max(cost_remaining, 0.0) clamp is load-bearing for float costs:
    # sequential subtraction can end at a tiny negative residual (design §7
    # site 2). Pin non-negativity across several float-cost seeds.
    for seed in range(6):
        p = _random_problem(seed, integer=False)
        res = rank_removal(p, rule=rule, warp=3)
        assert (res.performance_curves["prop_cost_remaining"] >= 0).all()


# --- CELF phase, Task 1: curve_every + array curves -----------------------
def test_curve_every_validation() -> None:
    p = _random_problem(0)
    for bad in (0, -3, 2.5, "5", None):
        with pytest.raises(ValueError, match="curve_every"):
            rank_removal(p, curve_every=bad)  # type: ignore[arg-type]
    # np.integer must be ACCEPTED (design-review #5): the likeliest caller type
    # for a raster-scale memory knob is numpy-derived (n_pu // 1000 etc.).
    res = rank_removal(p, curve_every=np.int64(7))  # type: ignore[arg-type]
    assert len(res.removal_order) == p.n_planning_units


@pytest.mark.parametrize("force_batch", [False, True])
@pytest.mark.parametrize("rule", ["caz", "abf"])
def test_curve_every_thins_rows_exactly(rule: str, force_batch: bool) -> None:
    # Thinned rows must be a bitwise row-subset of the curve_every=1 run:
    # initial row, every k-th removal, and always the final state (design §6).
    p = _random_problem(3, n_pu=40)
    full = rank_removal(p, rule=rule, warp=1, _force_batch=force_batch)
    thin = rank_removal(p, rule=rule, warp=1, curve_every=7, _force_batch=force_batch)
    assert full.removal_order == thin.removal_order
    fc = full.performance_curves.to_numpy()
    tc = thin.performance_curves.to_numpy()
    k, n = 7, 40
    idxs = [0] + list(range(k, n + 1, k))
    if idxs[-1] != n:
        idxs.append(n)
    np.testing.assert_array_equal(tc, fc[idxs])
    assert list(thin.performance_curves.columns) == list(full.performance_curves.columns)


# --- CELF phase, Task 3: heap-vs-batch bitwise equivalence ----------------
@pytest.mark.parametrize("rule", ["caz", "abf"])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_heap_equals_batch_integer(rule: str, seed: int) -> None:
    p = _random_problem(seed)
    _assert_equal_results(
        rank_removal(p, rule=rule, warp=1),
        rank_removal(p, rule=rule, warp=1, _force_batch=True),
    )


@pytest.mark.parametrize("rule", ["caz", "abf"])
def test_heap_equals_batch_families(rule: str) -> None:
    # Locks, float amounts+costs, wide features, locks+float — design §8.1.
    fixtures = [
        _random_problem(7, statuses=True),
        _random_problem(11, integer=False),
        _random_problem(5, n_pu=60, n_feat=25),
        _random_problem(21, n_pu=120, n_feat=8, statuses=True, integer=False),
    ]
    for p in fixtures:
        _assert_equal_results(
            rank_removal(p, rule=rule, warp=1),
            rank_removal(p, rule=rule, warp=1, _force_batch=True),
        )


@pytest.mark.parametrize("rule", ["caz", "abf"])
def test_heap_equals_batch_smoothing(rule: str) -> None:
    rng = np.random.default_rng(17)
    p = _random_problem(17, n_pu=18, n_feat=4)
    spec = SmoothingSpec(alpha=0.5, coords=rng.uniform(0, 10, size=(18, 2)))
    _assert_equal_results(
        rank_removal(p, rule=rule, warp=1, smoothing=spec),
        rank_removal(p, rule=rule, warp=1, smoothing=spec, _force_batch=True),
    )


def test_heap_equals_batch_edge_shapes() -> None:
    pu = pd.DataFrame({"id": [1, 2, 3], "cost": [1.0, 2.0, 3.0], "status": [0, 0, 0]})
    feats0 = pd.DataFrame({"id": [], "name": [], "target": [], "spf": []})
    pvf0 = pd.DataFrame(columns=["species", "pu", "amount"])
    p0 = ConservationProblem(pu, feats0, pvf0)
    _assert_equal_results(
        rank_removal(p0, warp=1), rank_removal(p0, warp=1, _force_batch=True)
    )


@pytest.mark.parametrize("rule", ["caz", "abf"])
def test_heap_equals_batch_float_sweep(rule: str) -> None:
    # GENERAL heap-vs-batch coverage. Review #2 measured 0 extinction crossings
    # in 120 such runs — the constructed Task-4 fixtures, NOT this sweep, are
    # the §5 repair's net.
    for seed in range(30):
        p = _random_problem(seed, n_pu=50, n_feat=5, integer=False)
        _assert_equal_results(
            rank_removal(p, rule=rule, warp=1),
            rank_removal(p, rule=rule, warp=1, _force_batch=True),
        )


@pytest.mark.parametrize("rule", ["caz", "abf"])
def test_heap_equals_batch_weights(rule: str) -> None:
    # w != 1 is a distinct rescore input never covered by other families;
    # weight-0 features meet the dead-mask logic (review #6).
    p = _random_problem(13, n_pu=40, n_feat=4)
    w = {1: 3.5, 2: 0.0}
    _assert_equal_results(
        rank_removal(p, rule=rule, warp=1, weights=w),
        rank_removal(p, rule=rule, warp=1, weights=w, _force_batch=True),
    )


def test_heap_equals_batch_all_ties() -> None:
    # Sustained equal-key regime: every delta bitwise-equal each step — pure
    # (score, index) tuple-order selection with maximal stale-duplicate
    # traffic; the tie-break-through-staleness argument's stress case
    # (review #10). Expected order is directly assertable: ascending PU index.
    n = 30
    pu = pd.DataFrame(
        {"id": list(range(1, n + 1)), "cost": [1.0] * n, "status": [0] * n}
    )
    feats = pd.DataFrame({"id": [1], "name": ["a"], "target": [1.0], "spf": [1.0]})
    pvf = pd.DataFrame(
        [{"species": 1, "pu": i, "amount": 2.0} for i in range(1, n + 1)]
    )
    p = ConservationProblem(pu, feats, pvf)
    for rule in ("caz", "abf"):
        heap_res = rank_removal(p, rule=rule, warp=1, use_cost=False)
        _assert_equal_results(
            heap_res,
            rank_removal(p, rule=rule, warp=1, use_cost=False, _force_batch=True),
        )
        assert heap_res.removal_order == list(range(1, n + 1))


def test_heap_equals_batch_small_families() -> None:
    # Stored-zero, duplicate-(pu,species), and featureless-PU fixtures — cheap
    # direct coverage beyond the transitive oracle route (reviews #6/#10).
    pu3 = pd.DataFrame({"id": [1, 2, 3], "cost": [1.0] * 3, "status": [0] * 3})
    feats2 = pd.DataFrame(
        {"id": [1, 2], "name": ["a", "b"], "target": [1.0, 1.0], "spf": [1.0, 1.0]}
    )
    feats1 = pd.DataFrame({"id": [1], "name": ["a"], "target": [1.0], "spf": [1.0]})
    fixtures = [
        ConservationProblem(  # stored zero
            pu3,
            feats2,
            pd.DataFrame(
                [
                    {"species": 1, "pu": 1, "amount": 2.0},
                    {"species": 1, "pu": 2, "amount": 0.0},
                    {"species": 2, "pu": 2, "amount": 3.0},
                    {"species": 2, "pu": 3, "amount": 1.0},
                ]
            ),
        ),
        ConservationProblem(  # duplicate (pu, species) rows
            pu3.iloc[:2].reset_index(drop=True),
            feats1,
            pd.DataFrame(
                [
                    {"species": 1, "pu": 1, "amount": 2.0},
                    {"species": 1, "pu": 1, "amount": 3.0},
                    {"species": 1, "pu": 2, "amount": 4.0},
                ]
            ),
        ),
        ConservationProblem(  # featureless PUs 1 and 3
            pu3, feats1, pd.DataFrame([{"species": 1, "pu": 2, "amount": 3.0}])
        ),
    ]
    for p in fixtures:
        for rule in ("caz", "abf"):
            _assert_equal_results(
                rank_removal(p, rule=rule, warp=1),
                rank_removal(p, rule=rule, warp=1, _force_batch=True),
            )


# --- CELF phase, Task 4: extinction construction + pinning ----------------
def _extinction_problem(statuses: list[int]) -> ConservationProblem:
    """Detector-verified FP-residue extinction fixture (review #2, verified).

    Levers (both are load-bearing — the review's original 'huge costs on big
    holders' reasoning was backwards, a near-zero-amount holder at cost 1 is
    the global argmin and leaves FIRST):
    - the residue carrier PU3 gets cost 1e-15 so its score (1.67e-2) stays
      ABOVE the big holders' (5e-4, 1e-3) until the crossing;
    - detector PU6 (sole holder of f3, score 1/200 = 5e-3) sits strictly
      inside the carrier's stale-key/post-crossing-true-score gap (0, 1.67e-2),
      so a missing, broken, or phase-inverted repair provably FLIPS the order
      to [1,2,6,3,5,4] — this test VERIFIES the repair, not merely exercises it
      (execution-verified against a deliberately repair-less heap).
    """
    pu = pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5, 6],
            "cost": [1000.0, 1000.0, 1e-15, 1.0, 1.0, 200.0],
            "status": statuses,
        }
    )
    feats = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "name": ["a", "b", "c"],
            "target": [1.0] * 3,
            "spf": [1.0] * 3,
        }
    )
    pvf = pd.DataFrame(
        [
            {"species": 1, "pu": 1, "amount": 0.3},
            {"species": 1, "pu": 2, "amount": 0.3},
            {"species": 1, "pu": 3, "amount": 1e-17},
            {"species": 2, "pu": 4, "amount": 5.0},
            {"species": 2, "pu": 5, "amount": 3.0},
            {"species": 3, "pu": 6, "amount": 1.0},
        ]
    )
    return ConservationProblem(pu, feats, pvf)


@pytest.mark.parametrize("rule", ["caz", "abf"])
def test_fp_residue_extinction_heap_equals_batch(rule: str) -> None:
    # Crossing INSIDE the normal phase -> the repair-push path itself runs.
    assert (0.3 + 0.3 + 1e-17) - 0.3 - 0.3 <= 0.0  # the arithmetic premise
    p = _extinction_problem([0, 0, 0, 0, 0, 0])
    heap_res = rank_removal(p, rule=rule, warp=1)
    _assert_equal_results(
        heap_res, rank_removal(p, rule=rule, warp=1, _force_batch=True)
    )
    order = heap_res.removal_order
    # Big holders precede the carrier (crossing happens while PU3 remains),
    # and the detector-sensitive order is pinned exactly.
    assert max(order.index(1), order.index(2)) < order.index(3)
    assert order == [1, 2, 3, 6, 5, 4]


@pytest.mark.parametrize("rule", ["caz", "abf"])
def test_fp_residue_extinction_cross_phase(rule: str) -> None:
    # Big holders locked OUT (review #3): the crossing happens in the
    # locked-out phase while the carrier sits in the normal phase — exercising
    # the repair-push-skipped + dirty-carry + phase-init-rescore path that
    # design §5 declares safe.
    p = _extinction_problem([3, 3, 0, 0, 0, 0])
    _assert_equal_results(
        rank_removal(p, rule=rule, warp=1),
        rank_removal(p, rule=rule, warp=1, _force_batch=True),
    )


@pytest.mark.parametrize("rule", ["caz", "abf"])
def test_phase_mask_blocks_out_of_phase_repair_push(rule: str) -> None:
    # Crossing happens in the locked-out phase with phase_left > 0; the crossed
    # feature's surviving holder (PU3) is a NORMAL-phase cell. An out-of-phase
    # repair-push would enter PU3 into the locked-out phase's heap and select
    # it early ([1,3,2,4] instead of [1,2,3,4]) — this pins `& phase_mask[col]`.
    pu = pd.DataFrame(
        {"id": [1, 2, 3, 4], "cost": [1e21, 1.0, 1.0, 1.0], "status": [3, 3, 0, 0]}
    )
    feats = pd.DataFrame(
        {"id": [1, 2], "name": ["a", "b"], "target": [1.0, 1.0], "spf": [1.0, 1.0]}
    )
    pvf = pd.DataFrame(
        [
            {"species": 1, "pu": 1, "amount": 1.0},
            {"species": 1, "pu": 3, "amount": 1e-17},
            {"species": 2, "pu": 2, "amount": 1.0},
            {"species": 2, "pu": 3, "amount": 0.5},
            {"species": 2, "pu": 4, "amount": 1e20},
        ]
    )
    p = ConservationProblem(pu, feats, pvf)
    heap_res = rank_removal(p, rule=rule, warp=1)
    _assert_equal_results(
        heap_res, rank_removal(p, rule=rule, warp=1, _force_batch=True)
    )
    assert heap_res.removal_order == [1, 2, 3, 4]


def test_subnormal_raises_on_both_paths() -> None:
    # The NaN-poisoned no-progress input must raise on the heap path (at init
    # or repush) AND still on the forced batch path.
    pu = pd.DataFrame(
        {"id": [1, 2, 3, 4], "cost": [1.0] * 4, "status": [0] * 4}
    )
    feats = pd.DataFrame(
        {"id": [1, 2], "name": ["a", "b"], "target": [1.0, 1.0], "spf": [1.0, 1.0]}
    )
    pvf = pd.DataFrame(
        [
            {"species": 1, "pu": 1, "amount": 5e-324},
            {"species": 2, "pu": 2, "amount": 1.0},
            {"species": 2, "pu": 3, "amount": 2.0},
            {"species": 2, "pu": 4, "amount": 3.0},
        ]
    )
    p = ConservationProblem(pu, feats, pvf)
    for kwargs in ({}, {"_force_batch": True}):
        with pytest.raises(RuntimeError, match="no progress"):
            rank_removal(p, warp=1, **kwargs)


def test_curve_every_with_warp_batches() -> None:
    # warp>1: records land on batch boundaries whose cumulative removal count
    # is a multiple of curve_every; always the final row (design §6).
    p = _random_problem(9, n_pu=30)
    full = rank_removal(p, warp=5)               # rows at n_removed 0,5,...,30
    thin = rank_removal(p, warp=5, curve_every=10)
    fc = full.performance_curves.to_numpy()
    tc = thin.performance_curves.to_numpy()
    np.testing.assert_array_equal(tc, fc[[0, 2, 4, 6]])


def test_curve_every_misaligned_warp_records_sparse_rows() -> None:
    # warp=3 never lands on a multiple of 10 (batch ends at 3,6,9,...), so
    # only the initial and final rows are recorded — the docstring's
    # "choose curve_every a multiple of warp" advice exists for this reason.
    p = _random_problem(9, n_pu=30)
    res = rank_removal(p, warp=3, curve_every=10)
    full = rank_removal(p, warp=3)
    fc = full.performance_curves.to_numpy()
    tc = res.performance_curves.to_numpy()
    np.testing.assert_array_equal(tc, fc[[0, 10]])  # initial + final only
