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

import heapq
import operator
import warnings

import numpy as np
import pandas as pd

from pymarxan.models.problem import (
    STATUS_LOCKED_IN,
    STATUS_LOCKED_OUT,
    ConservationProblem,
)
from pymarxan.zonation.result import ZonationResult
from pymarxan.zonation.smoothing import GridSmoothingSpec, SmoothingSpec

_SMOOTHING_MAX_PU = 50_000
_WARP_ADVISORY_MIN_PU = 50_000
_RESCORE_CHUNK = 32_768
_NO_PROGRESS_MSG = (
    "rank_removal made no progress: non-finite scores (extreme "
    "amounts/weights can overflow w/Q); cannot rank this input"
)


def _warn_if_small_warp(n_pu: int, warp: int) -> None:
    """Advise (warn-and-proceed, S3b precedent) when warp is too small to scale.

    warp=1 routes to the exact lazy-heap path and is fast; the advisory covers
    2 <= warp < n_pu // 10_000 at raster scale, where batch selection pays an
    O(candidates) partition per small batch. Large warp (10-100) is documented
    Zonation practice as a computation-time vs solution-refinement trade-off;
    ``warp ~ n_pu/1000`` is pymarxan performance advice for batch mode.
    Silence with ``warnings.filterwarnings`` when a small warp is deliberate.
    """
    if warp > 1 and n_pu > _WARP_ADVISORY_MIN_PU and warp < n_pu // 10_000:
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
    smoothing: SmoothingSpec | GridSmoothingSpec | None = None,
    curve_every: int = 1,
    _force_full_rescore: bool = False,
    _force_batch: bool = False,
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
    Init is O(nnz). ``warp=1`` selects via lazy-greedy (accelerated greedy,
    Minoux 1978; popularized as CELF, Leskovec et al. 2007) mirrored to
    minimization: removal only increases remaining cells' scores, so cached
    keys are lower bounds on a min-heap and a popped fresh top is the true
    argmin. The warp=1 trajectory is bitwise-identical to batch selection for
    every input whose scores never evaluate to NaN (float amounts and +inf
    regimes included; NaN-producing runs raise ``RuntimeError`` on the heap
    path where batch selection may return a NaN-ordered tail) — exact with
    respect to the greedy removal sequence; the ranking itself remains a
    heuristic prioritization, not a provably optimal reserve. Single-cell
    removal is thereby feasible at raster scale and faster than batch selection
    at warp=1 (measured: 90k cells, heap 102.5s vs batch 138.4s; pass
    ``curve_every`` to keep curve memory bounded). ``warp>1`` uses
    batch selection; an advisory warns for small ``warp>1`` at raster scale
    (silence via ``warnings.filterwarnings``). ``curve_every=k`` records the
    initial state, every k-th removal (when ``warp>1``, a row lands only where
    a batch boundary coincides with a multiple of k — choose ``curve_every`` a
    multiple of ``warp`` for evenly spaced rows), and always the final state.
    Landscape-spanning features degrade either path
    toward O(n^2) holder-marking.
    Equivalence to the reference engine: exact removal order for BOTH rules on
    integer amounts (while sums stay below 2**53); float amounts (including
    smoothed matrices) can differ only via initial-total summation order (a few
    ULPs), which can flip exact float near-ties; float costs affect curve
    values only. ``ValueError`` on invalid input: negative or non-finite
    amounts/weights (negative weights — a Zonation v3+ opportunity-cost
    workflow — are not yet supported), non-finite costs (when ``use_cost=True``),
    zero planning units. Raises ``RuntimeError`` if any score evaluates to NaN
    (e.g. subnormal amounts overflowing ``w/Q``): immediately on the warp=1
    heap path, or when removal can make no progress on the batch path.
    Dense-kernel ``SmoothingSpec`` smoothing stays vector-scale (n_pu <= 50_000);
    ``GridSmoothingSpec`` (grid problems) has no cap — truncated 2-D
    convolution, mass-conserving.
    ``_force_full_rescore`` and ``_force_batch`` are test-only: the first
    disables the dirty-set shortcut (and forces batch selection), the second
    forces batch selection at ``warp=1``.
    """
    if rule not in ("caz", "abf"):
        raise ValueError(f"rule must be 'caz' or 'abf', got {rule!r}")
    try:
        curve_every = operator.index(curve_every)
    except TypeError:
        raise ValueError(
            f"curve_every must be an integer >= 1, got {curve_every!r}"
        ) from None
    if curve_every < 1:
        raise ValueError(
            f"curve_every must be an integer >= 1, got {curve_every!r}"
        )
    _validate_inputs(problem, weights)

    n_pu_total = problem.n_planning_units
    if isinstance(smoothing, GridSmoothingSpec) and problem.grid is None:
        raise ValueError(
            "GridSmoothingSpec requires a grid problem (problem.grid is None); "
            "use SmoothingSpec for vector problems"
        )
    if isinstance(smoothing, SmoothingSpec) and n_pu_total > _SMOOTHING_MAX_PU:
        # Positive isinstance (review #5): a future third spec type must not
        # silently inherit the dense cap or the dense apply path.
        raise ValueError(
            f"smoothing builds a dense {n_pu_total}x{n_pu_total} kernel and is "
            f"vector-scale only (n_pu <= {_SMOOTHING_MAX_PU}); use "
            "GridSmoothingSpec for grid problems at raster scale "
            "(problems constructed with grid=GridGeometry(...))."
        )

    from scipy.sparse import csr_matrix

    if isinstance(smoothing, GridSmoothingSpec):
        assert problem.grid is not None  # guarded above
        base = problem.build_pu_feature_csr().tocsc()
        n_rows = base.shape[0]
        data_parts: list[np.ndarray] = []
        idx_parts: list[np.ndarray] = []
        indptr = [0]
        for j in range(base.shape[1]):
            col = np.zeros(n_rows)
            sl = slice(base.indptr[j], base.indptr[j + 1])
            col[base.indices[sl]] = base.data[sl]
            sc = smoothing.apply(col, problem.grid)
            nz = np.flatnonzero(sc)
            data_parts.append(sc[nz])
            idx_parts.append(nz)
            indptr.append(indptr[-1] + nz.size)
        from scipy.sparse import csc_matrix

        q = csc_matrix(
            (
                np.concatenate(data_parts) if data_parts else np.zeros(0),
                np.concatenate(idx_parts) if idx_parts else np.zeros(0, dtype=np.intp),
                np.asarray(indptr, dtype=np.intp),
            ),
            shape=base.shape,
        ).tocsr()
    elif isinstance(smoothing, SmoothingSpec):
        q = csr_matrix(smoothing.apply(problem.build_pu_feature_matrix()))
    elif smoothing is not None:
        raise TypeError(
            f"unsupported smoothing spec: {type(smoothing).__name__}"
        )
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
    curve_cols = ["prop_landscape_remaining", "prop_cost_remaining"] + [
        f"feat_{int(fid)}" for fid in feat_ids
    ]
    curves = np.empty((3 + n_pu // curve_every, 2 + n_feat), dtype=float)
    n_curve_rows = 0
    last_recorded_at = -1  # n_removed value of the most recent recorded row

    def record_curve() -> None:
        nonlocal n_curve_rows, last_recorded_at
        curves[n_curve_rows, 0] = n_remaining / n_pu
        # max(): float-cost sequential drift can leave a tiny negative
        # residual at run end (design §7 site 2); exact for integer costs.
        curves[n_curve_rows, 1] = max(cost_remaining, 0.0) / cost_total
        curves[n_curve_rows, 2:] = np.where(T > 0, Q / T_safe, 1.0)
        n_curve_rows += 1
        last_recorded_at = n_pu - n_remaining

    record_curve()

    def candidate_indices() -> np.ndarray:
        locked_out = remaining & (status == STATUS_LOCKED_OUT)
        if locked_out.any():
            return np.flatnonzero(locked_out)
        normal = remaining & (status != STATUS_LOCKED_OUT) & (status != STATUS_LOCKED_IN)
        if normal.any():
            return np.flatnonzero(normal)
        return np.flatnonzero(remaining & (status == STATUS_LOCKED_IN))

    indptr, indices, data = q.indptr, q.indices, q.data

    def remove_cell(idx: int) -> tuple[np.ndarray, np.ndarray]:
        """Shared per-removal bookkeeping for both selection paths.

        Returns ``(cols, crossed)``: the removed row's feature columns, and the
        subset whose remaining total crossed from >0 to <=0 with this removal
        (FP-residue extinction — float amounts only; consumed by the heap
        path's invariant repair, design §5).
        """
        nonlocal n_remaining, cost_remaining
        removal_order.append(int(pu_ids[idx]))
        remaining[idx] = False
        s, e = indptr[idx], indptr[idx + 1]
        cols = indices[s:e]
        prev_pos = Q[cols] > 0
        Q[cols] -= data[s:e]  # sequential, matching the dense engine
        crossed = cols[prev_pos & (Q[cols] <= 0)]
        n_remaining -= 1
        cost_remaining -= float(c[idx])
        return cols, crossed

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

    use_heap = warp == 1 and not _force_batch and not _force_full_rescore

    if use_heap:
        while n_remaining > 0:
            cand = candidate_indices()  # one lock-phase at a time
            phase_mask = np.zeros(n_pu, dtype=bool)
            phase_mask[cand] = True
            rescore(cand[dirty[cand]])
            # NaN-only guard (design §4.3 / review #4): +inf keys are totally
            # ordered and must NOT raise (the batch path completes on all-inf
            # regimes); only NaN corrupts heapq ordering.
            if cand.size and np.isnan(delta[cand]).any():
                raise RuntimeError(_NO_PROGRESS_MSG)
            heap = [(float(delta[i]), int(i)) for i in cand]
            heapq.heapify(heap)
            phase_left = cand.size
            while phase_left > 0:
                if not heap:
                    raise RuntimeError(_NO_PROGRESS_MSG)
                s_val, i = heapq.heappop(heap)
                if not remaining[i]:
                    continue  # lazy deletion: cell already removed
                if dirty[i]:
                    # Buffered dirty rescore (review #1, the CRITICAL perf fix:
                    # single-row rescores cost ~72.5us in scipy slice overhead
                    # vs 1.5us/row vectorized; the per-pop variant measured
                    # ~18x SLOWER than the batch path). Drain the contiguous
                    # removed/dirty prefix of the heap, then rescore the
                    # deduplicated dirty buffer in ONE vectorized call.
                    buf = [i]
                    while heap:
                        s2, i2 = heap[0]
                        if not remaining[i2]:
                            heapq.heappop(heap)
                            continue
                        if dirty[i2]:
                            heapq.heappop(heap)
                            buf.append(i2)
                            continue
                        break
                    rows = np.unique(np.asarray(buf, dtype=np.intp))
                    rescore(rows)
                    if np.isnan(delta[rows]).any():
                        raise RuntimeError(_NO_PROGRESS_MSG)
                    for h in rows:
                        heapq.heappush(heap, (float(delta[h]), int(h)))
                    continue
                if s_val != delta[i]:
                    # Superseded duplicate. Safe ONLY because delta[] is written
                    # solely by rescore(), and every heap-path rescore is
                    # followed by a push/heapify of the rescored cells — a
                    # mismatched key always has a fresher sibling in the heap.
                    # (Rescoring without pushing would silently break argmin.)
                    continue
                # Fresh top == true global argmin (design §3), ties by index
                # via tuple order — accept.
                assert not dirty[i]
                cols, crossed = remove_cell(i)
                if cols.size:
                    holders = np.concatenate(
                        [csc.indices[csc.indptr[j] : csc.indptr[j + 1]] for j in cols]
                    )
                    dirty[holders] = True
                for j in crossed:
                    # FP-residue extinction repair (design §5): holders' true
                    # scores just DROPPED, so cached keys are no longer lower
                    # bounds — rescore and re-push, current phase only
                    # (phase_mask is load-bearing: an out-of-phase push would
                    # let a locked-in cell be selected early).
                    col = csc.indices[csc.indptr[j] : csc.indptr[j + 1]]
                    repair = col[remaining[col] & phase_mask[col]]
                    if repair.size:
                        rescore(repair)
                        if np.isnan(delta[repair]).any():
                            raise RuntimeError(_NO_PROGRESS_MSG)
                        for h in repair:
                            heapq.heappush(heap, (float(delta[h]), int(h)))
                phase_left -= 1
                n_removed = n_pu - n_remaining
                if n_removed % curve_every == 0:
                    record_curve()
    else:
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
                raise RuntimeError(_NO_PROGRESS_MSG)
            changed_parts: list[np.ndarray] = []
            for idx in removed:
                cols, _crossed = remove_cell(int(idx))
                changed_parts.append(cols)
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
            n_removed = n_pu - n_remaining
            if n_removed % curve_every == 0:
                record_curve()

    if last_recorded_at != n_pu:
        record_curve()

    priority_rank = {
        pu: (position + 1) / n_pu for position, pu in enumerate(removal_order)
    }
    return ZonationResult(
        priority_rank=priority_rank,
        removal_order=removal_order,
        performance_curves=pd.DataFrame(curves[:n_curve_rows], columns=curve_cols),
        rule=rule,
    )
