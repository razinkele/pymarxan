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
