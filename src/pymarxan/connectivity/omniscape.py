"""Omniscape-style omnidirectional current-flow connectivity.

Circuit theory (:mod:`pymarxan.connectivity.circuit`) gives the effective
resistance between *specified* planning-unit pairs. **Omniscape**
(McRae et al. 2016; Landau et al. 2021, *Land* 10(3):301) instead maps
*omnidirectional* current density across a whole landscape: with a moving
window, every cell within a `radius` of a focal target injects current into
that target, and the resulting current through each cell is accumulated. The
result highlights diffuse movement areas and — once normalised against a
flat-resistance null — channelling "pinch points".

This implementation reuses the grounded grid-Laplacian of ``circuit.py`` (edge
conductance = 1/mean of the two cells' resistance, 4-neighbour grid). It is a
faithful *core* of Omniscape with these documented simplifications: a single
focal cell per window (block size 1), a circular window of the given radius,
and a uniform default source strength. Pure NumPy + ``scipy.sparse``; works on
small/medium rasters (one sparse solve per window).

- **cumulative_current** — summed current density per cell.
- **flow_potential** — the same algorithm on a *flat* (uniform) resistance
  surface: the null expectation of movement.
- **normalized_current** — ``cumulative / flow_potential``; >1 marks
  channelling (pinch points), <1 marks diffusion.

References
----------
- Landau, V. A., Shah, V. B., Anantharaman, R., & Hall, K. R. (2021).
  Omniscape.jl. *Journal of Open Source Software, 6*(57), 2829.
  https://doi.org/10.21105/joss.02829
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class OmniscapeResult:
    """Per-cell Omniscape rasters (all same shape as the input resistance)."""

    cumulative_current: np.ndarray
    flow_potential: np.ndarray
    normalized_current: np.ndarray


def _cumulative_current(
    resistance: np.ndarray, radius: float, source_strength: np.ndarray
) -> np.ndarray:
    from scipy.sparse import csc_matrix
    from scipy.sparse.linalg import spsolve

    n_rows, n_cols = resistance.shape
    cumulative = np.zeros((n_rows, n_cols))
    r2 = radius * radius

    for tr in range(n_rows):
        for tc in range(n_cols):
            # Circular window of cells within `radius` of the focal target.
            cells: list[tuple[int, int]] = []
            for r in range(max(0, int(tr - radius)), min(n_rows, int(tr + radius) + 1)):
                for c in range(max(0, int(tc - radius)), min(n_cols, int(tc + radius) + 1)):
                    if (r - tr) ** 2 + (c - tc) ** 2 <= r2:
                        cells.append((r, c))
            m = len(cells)
            if m < 2:
                continue
            local = {cell: i for i, cell in enumerate(cells)}
            ti = local[(tr, tc)]

            # Local grid Laplacian (4-neighbour, conductance = 1/mean R).
            rows_idx: list[int] = []
            cols_idx: list[int] = []
            data: list[float] = []
            diag = np.zeros(m)
            conduct: dict[tuple[int, int], float] = {}
            for (r, c), a in local.items():
                for dr, dc in ((0, 1), (1, 0)):
                    nb = (r + dr, c + dc)
                    b = local.get(nb)
                    if b is None:
                        continue
                    g = 1.0 / (0.5 * (resistance[r, c] + resistance[nb]))
                    rows_idx += [a, b]
                    cols_idx += [b, a]
                    data += [-g, -g]
                    diag[a] += g
                    diag[b] += g
                    conduct[(a, b)] = g
                    conduct[(b, a)] = g
            rows_idx += list(range(m))
            cols_idx += list(range(m))
            data += diag.tolist()
            lap = csc_matrix((data, (rows_idx, cols_idx)), shape=(m, m))

            # Inject source strength at every non-target cell; ground the
            # target (drop its row/col). Solve for node potentials.
            inj = np.array([source_strength[r, c] for (r, c) in cells])
            inj[ti] = 0.0
            keep = [i for i in range(m) if i != ti]
            lap_red = lap[keep, :][:, keep]
            v = np.zeros(m)
            v[keep] = spsolve(lap_red.tocsc(), inj[keep])

            # Node current density I_i = 0.5 Σ_j g_ij |v_i - v_j|; accumulate.
            for (a, b), g in conduct.items():
                if a < b:
                    flow = g * abs(v[a] - v[b])
                    r_a, c_a = cells[a]
                    r_b, c_b = cells[b]
                    cumulative[r_a, c_a] += 0.5 * flow
                    cumulative[r_b, c_b] += 0.5 * flow
    return cumulative


def omniscape(
    resistance: np.ndarray,
    radius: float,
    source_strength: np.ndarray | None = None,
) -> OmniscapeResult:
    """Omnidirectional cumulative current, flow potential, and normalised
    current for a resistance raster.

    Args:
        resistance: 2-D positive, finite resistance surface (higher = poorer
            movement).
        radius: Moving-window radius in cells (number of cells, > 0).
        source_strength: Optional per-cell source weight (same shape); defaults
            to uniform 1.

    Returns:
        :class:`OmniscapeResult` with ``cumulative_current``,
        ``flow_potential`` (flat-resistance null), and ``normalized_current``.
    """
    resistance = np.asarray(resistance, dtype=float)
    if resistance.ndim != 2:
        raise ValueError(f"resistance must be 2-D; got shape {resistance.shape}.")
    if not np.all(np.isfinite(resistance)) or not np.all(resistance > 0):
        raise ValueError("resistance values must be positive and finite.")
    if radius <= 0:
        raise ValueError("radius must be positive.")

    if source_strength is None:
        source_strength = np.ones_like(resistance)
    else:
        source_strength = np.asarray(source_strength, dtype=float)
        if source_strength.shape != resistance.shape:
            raise ValueError("source_strength must match resistance shape.")

    cumulative = _cumulative_current(resistance, radius, source_strength)
    flat = _cumulative_current(np.ones_like(resistance), radius, source_strength)

    with np.errstate(divide="ignore", invalid="ignore"):
        normalized = np.where(flat > 0, cumulative / flat, 0.0)

    return OmniscapeResult(
        cumulative_current=cumulative,
        flow_potential=flat,
        normalized_current=normalized,
    )
