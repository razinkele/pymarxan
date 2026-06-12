"""Distribution smoothing via a dispersal kernel.

Spread each planning unit's feature amount to nearby units using a
negative-exponential dispersal kernel, so a solver values *being near*
abundance, not only holding it. This is the planning-unit (vector)
analogue of Zonation's distribution smoothing.
"""
from __future__ import annotations

import numpy as np

from pymarxan.connectivity.decay import negative_exponential


def distance_matrix_from_points(coords: np.ndarray) -> np.ndarray:
    """Euclidean pairwise distance matrix from an ``(n, 2)`` coordinate array."""
    from scipy.spatial.distance import cdist

    coords = np.asarray(coords, dtype=float)
    return cdist(coords, coords)


def smooth_distribution(
    amounts: np.ndarray,
    distances: np.ndarray,
    alpha: float,
    *,
    normalize: bool = True,
) -> np.ndarray:
    """Smooth a per-unit feature distribution with a dispersal kernel.

    The kernel is ``K_ij = exp(-alpha * distance_ij)`` (so the diagonal is
    1). With ``normalize=True`` the kernel is column-normalised so total
    amount is conserved — each source unit's amount is redistributed across
    units in proportion to the kernel. With ``normalize=False`` the result
    is the raw ``K @ amounts`` accumulation (total grows).

    Args:
        amounts: Length-``n`` array of per-unit amounts for one feature.
        distances: ``(n, n)`` pairwise distance matrix.
        alpha: Decay rate (> 0); larger = more local.
        normalize: Conserve total amount (default True).

    Returns:
        Length-``n`` array of smoothed amounts.
    """
    # negative_exponential raises ValueError if alpha <= 0.
    kernel = negative_exponential(np.asarray(distances, dtype=float), alpha)
    if normalize:
        col_sums = kernel.sum(axis=0)
        kernel = kernel / col_sums
    result: np.ndarray = kernel @ np.asarray(amounts, dtype=float)
    return result
