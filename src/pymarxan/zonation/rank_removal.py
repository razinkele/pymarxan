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
