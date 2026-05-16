"""Cohon Pareto-frontier filter on a BLM calibration sweep (Phase 25).

``calibrate_blm`` produces a sweep across BLM values; many of those
points are dominated (some other BLM gives a strictly better cost AND
boundary). This module filters to the non-dominated set — the Pareto
frontier — so users see only the meaningful trade-off curve.

Named after Cohon (1978) *Multiobjective Programming and Planning*, the
foundational reference for Pareto-optimal trade-off analysis in
multi-objective optimisation.
"""
from __future__ import annotations

from pymarxan.calibration.blm import BLMResult


def pareto_frontier(result: BLMResult) -> BLMResult:
    """Drop dominated points from a BLM sweep.

    Point ``i`` is dominated iff some other point ``j`` has
    ``cost_j ≤ cost_i`` AND ``boundary_j ≤ boundary_i`` AND at least
    one of those inequalities is strict.

    Parameters
    ----------
    result
        Output of :func:`pymarxan.calibration.blm.calibrate_blm`.

    Returns
    -------
    BLMResult
        New :class:`BLMResult` containing only Pareto-optimal points.
        Input is not mutated.
    """
    n = len(result.costs)
    if n == 0:
        return BLMResult(
            blm_values=[], costs=[], boundaries=[],
            objectives=[], solutions=[],
        )

    keep_idx: list[int] = []
    for i in range(n):
        ci = result.costs[i]
        bi = result.boundaries[i]
        dominated = False
        for j in range(n):
            if i == j:
                continue
            cj = result.costs[j]
            bj = result.boundaries[j]
            if cj <= ci and bj <= bi and (cj < ci or bj < bi):
                dominated = True
                break
        if not dominated:
            keep_idx.append(i)

    # Deduplicate points that have identical (cost, boundary) — they all
    # survive the strict-dominance check but the user only wants one.
    seen: set[tuple[float, float]] = set()
    deduped: list[int] = []
    for i in keep_idx:
        key = (result.costs[i], result.boundaries[i])
        if key not in seen:
            deduped.append(i)
            seen.add(key)

    return BLMResult(
        blm_values=[result.blm_values[i] for i in deduped],
        costs=[result.costs[i] for i in deduped],
        boundaries=[result.boundaries[i] for i in deduped],
        objectives=[result.objectives[i] for i in deduped],
        solutions=[result.solutions[i] for i in deduped],
    )
