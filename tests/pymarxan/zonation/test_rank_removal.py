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


def test_smoothing_changes_ranking():
    from pymarxan.zonation.smoothing import SmoothingSpec

    # feature peaked on PU1; without smoothing PU2/PU3 hold none (removed by
    # index -> [2,3,1]); with smoothing PU2 (near) inherits value, out-ranks PU3.
    # NB: [3,2,1] relies on _problem's uniform cost (delta/cost tie-breaking).
    problem = _problem([[10], [0], [0]], feat_ids=(1,))
    coords = np.array([[0.0], [1.0], [2.0]])
    plain = rank_removal(problem, rule="caz")
    smoothed = rank_removal(
        problem, rule="caz", smoothing=SmoothingSpec(alpha=1.0, coords=coords)
    )
    assert plain.removal_order == [2, 3, 1]
    assert smoothed.removal_order == [3, 2, 1]  # near-neighbor PU2 now out-ranks PU3
