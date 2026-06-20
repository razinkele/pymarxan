"""Climate-refugia scoring for spatial conservation planning.

Climate refugia are areas that remain relatively buffered from climate change
and so let species persist (Keppel et al. 2015, doi:10.1890/140055). For
planning, the influential operational framing is "resilient **and** connected"
(Anderson et al. 2023, doi:10.1073/pnas.2204434119): good refugia are both
*climatically stable in place* and *accessible / part of movement corridors*.

``refugia_score`` composes two layers pymarxan already produces into a single
[0, 1] priority surface that feeds Marxan as a feature/cost:

- **stability** from :func:`pymarxan.connectivity.velocity.climate_velocity`
  (low velocity → high stability), and
- **connectivity** from any current/quality layer, e.g.
  :func:`pymarxan.connectivity.omniscape.omniscape` cumulative or normalised
  current.

Both inputs are min–max normalised within the raster (relative prioritisation),
so the score is a *ranking* surface, not an absolute index. Non-finite
velocities (flat climate) are treated as the worst (stability 0).
"""
from __future__ import annotations

import numpy as np


def _minmax(x: np.ndarray) -> np.ndarray:
    """Min–max normalise to [0, 1]; a constant array maps to all-zeros."""
    lo = np.nanmin(x)
    hi = np.nanmax(x)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros_like(x)
    return np.asarray((x - lo) / (hi - lo), dtype=float)


def refugia_score(
    velocity: np.ndarray,
    connectivity: np.ndarray | None = None,
    *,
    velocity_weight: float = 0.5,
    connectivity_weight: float = 0.5,
    method: str = "weighted",
) -> np.ndarray:
    """Combine climate stability (inverse velocity) and connectivity into a
    [0, 1] refugium-priority raster (higher = better refugium).

    Args:
        velocity: Climate-velocity raster (e.g. from ``climate_velocity``).
            Lower = more stable. ``inf``/``nan`` cells are treated as the worst.
        connectivity: Optional connectivity/quality raster (higher = better
            connected). If ``None``, the score is stability-only.
        velocity_weight, connectivity_weight: Weights for ``method="weighted"``
            (normalised to sum to 1).
        method: ``"weighted"`` (weighted mean) or ``"geometric"`` (geometric
            mean — a cell scores 0 if *either* component is 0).

    Returns:
        ``[0, 1]`` refugium score, same shape as ``velocity``.
    """
    if method not in ("weighted", "geometric"):
        raise ValueError(f"unknown method {method!r}; use 'weighted' or 'geometric'")

    v = np.asarray(velocity, dtype=float)
    # Treat non-finite velocity (flat climate → infinite velocity) as the worst.
    finite = np.isfinite(v)
    worst = float(v[finite].max()) if finite.any() else 1.0
    v_filled = np.where(finite, v, worst)
    stability = np.asarray(1.0 - _minmax(v_filled), dtype=float)  # low vel → high stability

    if connectivity is None:
        # Velocity-only refugia: the score is the stability surface itself.
        return stability

    conn = np.asarray(connectivity, dtype=float)
    if conn.shape != v.shape:
        raise ValueError("connectivity must match velocity shape")
    conn = _minmax(conn)

    if method == "weighted":
        total = velocity_weight + connectivity_weight
        if total <= 0:
            raise ValueError("weights must sum to a positive value")
        weighted = (velocity_weight * stability + connectivity_weight * conn) / total
        return np.asarray(weighted, dtype=float)
    return np.asarray(np.sqrt(np.clip(stability * conn, 0.0, None)), dtype=float)
