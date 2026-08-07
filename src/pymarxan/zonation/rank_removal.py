"""Zonation CAZ/ABF rank-removal engine (Moilanen et al. 2005; Moilanen 2007).

Distinct from ``pymarxan.analysis.rank_importance`` (Jung et al. 2021), which
ranks only the *selected* PUs of an existing solution by Marxan-objective
increase; this ranks *every* PU from the whole landscape by proportional
biological loss.
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


def rank_removal(
    problem: ConservationProblem,
    *,
    rule: str = "caz",
    weights: dict[int, float] | None = None,
    warp: int = 1,
    use_cost: bool = True,
    smoothing: SmoothingSpec | None = None,
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

    _validate_inputs(problem, weights)

    n_pu_total = problem.n_planning_units
    if smoothing is not None and n_pu_total > _SMOOTHING_MAX_PU:
        raise ValueError(
            f"smoothing builds a dense {n_pu_total}x{n_pu_total} kernel and is "
            f"vector-scale only (n_pu <= {_SMOOTHING_MAX_PU}); raster-scale "
            "distribution smoothing (grid convolution) is a planned follow-up."
        )

    q = problem.build_pu_feature_matrix()  # (n_pu, n_feat), rows = PU order
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
        if not np.isfinite(c).all():
            raise ValueError("planning-unit costs must be finite for rank_removal")
        if np.any(c <= 0):
            raise ValueError("use_cost=True requires every planning-unit cost > 0")
    else:
        c = np.ones(n_pu, dtype=float)

    warp = max(1, min(int(warp), max(n_pu, 1)))
    _warn_if_small_warp(n_pu, warp)

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
