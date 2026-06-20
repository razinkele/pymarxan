"""Climate velocity — the speed a species must move to track its climate niche.

Climate velocity (Burrows et al. 2014, doi:10.1038/nature12976) is the local
**temporal** rate of climate change divided by the local **spatial** climate
gradient:

    velocity = |dC/dt| / |∇C|         (e.g. (°C/yr) / (°C/km) = km/yr)

It is a standard climate-adaptation layer for spatial conservation planning —
high-velocity cells are where climate "outruns" a species in place, so it feeds
Marxan as a cost, feature, or boundary layer (e.g. prioritise low-velocity
climate refugia, or treat velocity as a risk to avoid).

The spatial gradient uses the Horn (1981) 3×3 finite-difference method (the
same technique Burrows 2014 and the VoCC R package use). Inputs are plain
2D NumPy rasters — georeferencing/sampling to planning units is left to the
existing spatial raster pipeline.

Raster convention: ``climate[row, col]`` with ``row`` increasing **southward**
(top row = north), the usual image/raster layout. Edge cells use replicate
padding, so the border gradient is a one-sided approximation rather than NaN.

References
----------
- Burrows, M. T., Schoeman, D. S., Richardson, A. J., et al. (2014).
  Geographical limits to species-range shifts are suggested by climate
  velocity. *Nature, 507*(7493), 492–495. https://doi.org/10.1038/nature12976
"""
from __future__ import annotations

import numpy as np


def spatial_gradient(
    climate: np.ndarray, cell_size: float = 1.0
) -> tuple[np.ndarray, np.ndarray]:
    """Spatial gradient of a climate raster via the Horn (1981) 3×3 method.

    Parameters
    ----------
    climate
        2D array of a climate variable (e.g. mean temperature) per grid cell.
    cell_size
        Edge length of a (square) cell in distance units; the gradient is
        returned per that distance unit.

    Returns
    -------
    (magnitude, direction)
        ``magnitude`` is the gradient steepness (climate units per distance
        unit). ``direction`` is the gradient bearing in degrees (math
        convention, ``atan2(NS, WE)``), pointing toward increasing climate.
        Both have the same shape as ``climate``.
    """
    climate = np.asarray(climate, dtype=float)
    if climate.ndim != 2:
        raise ValueError("climate must be a 2D array")
    if cell_size <= 0:
        raise ValueError("cell_size must be positive")

    p = np.pad(climate, 1, mode="edge")
    # 3×3 neighbours (row increases southward)
    nw, n, ne = p[:-2, :-2], p[:-2, 1:-1], p[:-2, 2:]
    w, e = p[1:-1, :-2], p[1:-1, 2:]
    sw, s, se = p[2:, :-2], p[2:, 1:-1], p[2:, 2:]

    grad_we = ((ne + 2.0 * e + se) - (nw + 2.0 * w + sw)) / (8.0 * cell_size)
    grad_ns = ((nw + 2.0 * n + ne) - (sw + 2.0 * s + se)) / (8.0 * cell_size)

    magnitude = np.hypot(grad_we, grad_ns)
    direction = np.degrees(np.arctan2(grad_ns, grad_we))
    return magnitude, direction


def climate_velocity(
    climate: np.ndarray,
    temporal_trend: float | np.ndarray,
    cell_size: float = 1.0,
    max_velocity: float | None = None,
) -> np.ndarray:
    """Local climate velocity raster: ``|temporal_trend| / |spatial gradient|``.

    Parameters
    ----------
    climate
        2D climate raster (see :func:`spatial_gradient`).
    temporal_trend
        Rate of climate change over time — a scalar (uniform) or a 2D array
        matching ``climate`` (per-cell trend, e.g. °C/yr). Its sign is ignored;
        the result is a non-negative speed magnitude.
    cell_size
        Cell edge length in distance units; sets the velocity's distance unit.
    max_velocity
        Optional cap. Flat-climate cells (zero spatial gradient) otherwise have
        infinite velocity (a species cannot track climate by moving); pass a cap
        to clamp them for use as a finite Marxan layer.

    Returns
    -------
    np.ndarray
        Velocity magnitude per cell (distance units per time unit), same shape
        as ``climate``; ``inf`` where the spatial gradient is zero (unless
        ``max_velocity`` is given).
    """
    magnitude, _ = spatial_gradient(climate, cell_size)
    trend = np.abs(np.asarray(temporal_trend, dtype=float))

    with np.errstate(divide="ignore", invalid="ignore"):
        velocity = trend / magnitude
    velocity = np.where(magnitude == 0.0, np.inf, velocity)

    if max_velocity is not None:
        velocity = np.minimum(velocity, max_velocity)
    return velocity
