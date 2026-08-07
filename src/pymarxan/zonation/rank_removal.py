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
        if removed.size == 0:
            # NaN-poisoned scores (e.g. subnormal amounts overflowing w/Q_safe
            # to inf, then 0.0*inf -> NaN) make both the `below` and `ties`
            # masks empty in the argpartition branch, which would otherwise
            # spin here forever with n_remaining unchanged. Fail loudly
            # instead of hanging; the dense engine "terminates" on such input
            # only by producing NaN-poisoned garbage ordering, so a RuntimeError
            # here is strictly better than either engine's alternative.
            raise RuntimeError(
                "rank_removal made no progress: non-finite scores (extreme "
                "amounts/weights can overflow w/Q); cannot rank this input"
            )
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
            holders = (
                np.concatenate(
                    [csc.indices[csc.indptr[j] : csc.indptr[j + 1]] for j in changed]
                )
                if changed.size
                else np.zeros(0, dtype=np.intp)
            )
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
