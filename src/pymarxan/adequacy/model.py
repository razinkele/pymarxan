"""raptr-style space/adequacy: attribute space + demand points (Hanson et al. 2018).

See ``docs/plans/2026-07-15-tierc-raptr-space-targets-{design,review}.md``. The space-held
measure (``compute_space_held``) is ``1 - WSS/TSS`` verified against raptr's source.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pymarxan.models.problem import ConservationProblem
from pymarxan.solvers.separation import get_pu_coordinates


@dataclass
class SpaceSpec:
    """Attribute-space configuration for space/adequacy targets.

    Default: geographic PU centroids (via ``get_pu_coordinates``), z-scored per dimension.
    ``attribute_columns`` uses those ``planning_units`` columns (set ``include_geographic=False``
    for an attribute-only space). Demand points default to the occupied PUs (amount>0), weighted
    by amount — a documented deviation from raptr's KDE-sampled, density-weighted default.
    """

    attribute_columns: list[str] | None = None
    include_geographic: bool = True
    zscore: bool = True


def pu_attribute_space(problem: ConservationProblem, spec: SpaceSpec) -> np.ndarray:
    """(n_pu, n_dim) attribute positions, z-scored per dimension when ``spec.zscore``.

    Geographic centroids (when ``include_geographic``) and/or ``attribute_columns``, in that
    order; raises if neither is configured. A single dim-collection loop — no duplicate
    geographic append (design-review BUG-A).
    """
    dims: list[np.ndarray] = []
    if spec.include_geographic:
        dims.append(np.asarray(get_pu_coordinates(problem), dtype=float))  # (n_pu, 2)
    if spec.attribute_columns:
        dims.append(problem.planning_units[spec.attribute_columns].to_numpy(dtype=float))
    if not dims:
        raise ValueError(
            "SpaceSpec has no attribute dimensions "
            "(set include_geographic=True or provide attribute_columns)."
        )
    pos = np.column_stack(dims) if len(dims) > 1 else dims[0]
    if spec.zscore:
        mu = pos.mean(axis=0)
        sd = pos.std(axis=0)
        sd[sd == 0] = 1.0
        pos = (pos - mu) / sd
    return pos
