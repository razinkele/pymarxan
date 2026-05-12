"""Distance decay functions for connectivity analysis."""
from __future__ import annotations

import numpy as np
import pandas as pd


def negative_exponential(distance: np.ndarray, alpha: float) -> np.ndarray:
    """Apply negative exponential decay: exp(-alpha * distance).

    Parameters
    ----------
    distance : np.ndarray
        Distance values (must be >= 0).
    alpha : float
        Decay rate (must be > 0).
    """
    if alpha <= 0:
        raise ValueError("alpha must be positive")
    return np.exp(-alpha * distance)


def inverse_power(distance: np.ndarray, beta: float) -> np.ndarray:
    """Apply inverse power decay: 1 / (1 + distance^beta).

    Parameters
    ----------
    distance : np.ndarray
        Distance values (must be >= 0).
    beta : float
        Power exponent (must be > 0).
    """
    if beta <= 0:
        raise ValueError("beta must be positive")
    result: np.ndarray = 1.0 / (1.0 + np.power(distance, beta))
    return result


def threshold(distance: np.ndarray, max_distance: float) -> np.ndarray:
    """Apply threshold decay: 1 if distance <= max_distance, else 0.

    Parameters
    ----------
    distance : np.ndarray
        Distance values.
    max_distance : float
        Maximum distance for connectivity (must be > 0).
    """
    if max_distance <= 0:
        raise ValueError("max_distance must be positive")
    return np.where(distance <= max_distance, 1.0, 0.0)


def apply_decay(
    edges: pd.DataFrame,
    decay_type: str,
    **params,
) -> pd.DataFrame:
    """Convert distance edge list to connectivity strength edge list.

    Accepts sparse edge lists (NOT dense matrices) to handle 50k+ PUs.

    Parameters
    ----------
    edges : pd.DataFrame
        DataFrame with columns: id1, id2, distance.
    decay_type : str
        One of "exponential", "power", "threshold".
    **params
        Decay function parameters (alpha for exponential, beta for power,
        max_distance for threshold).

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: id1, id2, value (connectivity strength).
    """
    required_cols = {"id1", "id2", "distance"}
    missing = required_cols - set(edges.columns)
    if missing:
        raise ValueError(f"edges missing columns: {sorted(missing)}")

    distances = edges["distance"].values.astype(np.float64)

    decay_functions = {
        "exponential": lambda d: negative_exponential(d, params.get("alpha", 1.0)),
        "power": lambda d: inverse_power(d, params.get("beta", 1.0)),
        "threshold": lambda d: threshold(d, params.get("max_distance", 1.0)),
    }

    if decay_type not in decay_functions:
        raise ValueError(
            f"Unknown decay_type '{decay_type}'. "
            f"Must be one of: {sorted(decay_functions.keys())}"
        )

    values = decay_functions[decay_type](distances)

    return pd.DataFrame({
        "id1": edges["id1"].values,
        "id2": edges["id2"].values,
        "value": values,
    })
