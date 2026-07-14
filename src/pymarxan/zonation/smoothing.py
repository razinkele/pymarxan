"""Distribution-smoothing config for Zonation rank-removal (Phase C)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pymarxan.connectivity.smoothing import (
    distance_matrix_from_points,
    smooth_distribution,
)


@dataclass(eq=False)
class SmoothingSpec:
    """Distribution-smoothing configuration for :func:`rank_removal`.

    Spreads each feature's amount to nearby planning units via a dispersal
    kernel before ranking (Zonation's distribution smoothing), reusing
    ``connectivity.smoothing``. Provide exactly one of ``coords`` (PU
    coordinates; distances computed via ``distance_matrix_from_points``) or a
    precomputed ``distances`` matrix. ``eq=False`` because numpy-array fields
    make the auto-generated ``__eq__`` raise on comparison.

    ``coords`` / ``distances`` rows MUST be aligned to
    ``problem.planning_units`` positional order — the same order
    ``build_pu_feature_matrix`` uses for the matrix rows (and that cost/status
    follow). Only the row *count* is validated, so a correctly-sized but
    mis-ordered array silently produces a wrong ranking. Smoothing is
    status-blind: the kernel spreads amount into and out of locked cells (locks
    are enforced later, by the solver).
    """

    alpha: float
    coords: np.ndarray | None = None
    distances: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.alpha <= 0:
            raise ValueError(f"alpha must be > 0, got {self.alpha}")
        if (self.coords is None) == (self.distances is None):
            raise ValueError("provide exactly one of 'coords' or 'distances'")
        if self.coords is not None and np.asarray(self.coords).ndim != 2:
            raise ValueError("coords must be 2-D (n_pu, d)")

    def resolve_distances(self, n_pu: int) -> np.ndarray:
        if self.distances is not None:
            d = np.asarray(self.distances, dtype=float)
            if d.shape != (n_pu, n_pu):
                raise ValueError(f"distances must be ({n_pu}, {n_pu}), got {d.shape}")
            return d
        coords = np.asarray(self.coords, dtype=float)
        if coords.shape[0] != n_pu:
            raise ValueError(f"coords must have {n_pu} rows, got {coords.shape[0]}")
        return distance_matrix_from_points(coords)

    def apply(self, q: np.ndarray) -> np.ndarray:
        """Return ``q`` with every feature column smoothed (one kernel build)."""
        distances = self.resolve_distances(q.shape[0])
        return smooth_distribution(np.asarray(q, dtype=float), distances, self.alpha)
