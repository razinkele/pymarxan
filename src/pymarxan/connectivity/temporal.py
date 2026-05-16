"""Temporal connectivity summaries (Phase 24).

Take a stack of per-timestep connectivity matrices and reduce to a
single ``(n, n)`` summary suitable for downstream Marxan workflows.

The mean / max reductions are useful when the user has, e.g.,
seasonal larval-dispersal matrices and wants either an average-year or
worst-case picture. The weighted reduction lets users emphasise
particular timesteps (recent years, climate-projection-based weights,
etc.).
"""
from __future__ import annotations

import numpy as np

_VALID_REDUCTIONS = ("mean", "max", "weighted")


def compute_temporal_connectivity(
    stack: np.ndarray,
    *,
    reduction: str = "mean",
    weights: np.ndarray | None = None,
) -> np.ndarray:
    """Reduce a stack of per-timestep connectivity matrices to an
    ``(n, n)`` summary.

    Parameters
    ----------
    stack
        Shape ``(T, n, n)`` — one matrix per timestep.
    reduction
        ``"mean"``, ``"max"``, or ``"weighted"``. Default ``"mean"``.
    weights
        Required when ``reduction="weighted"``; shape ``(T,)``. Need not
        sum to 1 (the function normalises internally).

    Returns
    -------
    np.ndarray
        Shape ``(n, n)``.
    """
    if reduction not in _VALID_REDUCTIONS:
        raise ValueError(
            f"reduction must be one of {_VALID_REDUCTIONS}, got {reduction!r}."
        )
    if stack.ndim != 3:
        raise ValueError(
            f"stack must be 3-D (T, n, n); got shape {stack.shape}."
        )
    if reduction == "mean":
        mean_result: np.ndarray = stack.mean(axis=0)
        return mean_result
    if reduction == "max":
        max_result: np.ndarray = stack.max(axis=0)
        return max_result
    # weighted
    if weights is None:
        raise ValueError(
            "reduction='weighted' requires the weights parameter "
            "(shape (T,))."
        )
    w = np.asarray(weights, dtype=np.float64)
    if w.shape != (stack.shape[0],):
        raise ValueError(
            f"weights shape {w.shape} doesn't match stack timesteps "
            f"{stack.shape[0]}."
        )
    total = w.sum()
    if total <= 0:
        raise ValueError("weights must sum to a positive value.")
    w_normalised = w / total
    # (T, n, n) * (T, 1, 1) → sum over T axis.
    weighted: np.ndarray = (stack * w_normalised[:, np.newaxis, np.newaxis]).sum(axis=0)
    return weighted
