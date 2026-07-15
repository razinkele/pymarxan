"""restoptr-style MESH — effective mesh size (Jaeger 2000) of a habitat map on a raster grid.

MESH = ``Σ A_i² / A_total`` over habitat patches (connected components), with ``A_total`` the
**total landscape area** (every valid cell) — verified against ``landscapemetrics::lsm_c_mesh``
(the function restoptr itself calls). The flagship landscape index of restoptr-style restoration
planning (Justeau-Allaire et al. 2021).

Patch labeling uses ``scipy.ndimage.label`` (the raster connected-components tool) — deliberately
distinct from ``constraints/contiguity.count_connected_components``, a vector-PU BFS over a
boundary DataFrame (count-only, rook-only). See ``restoration/__init__.py`` for the
``restoration`` vs ``connectivity`` subpackage split.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from pymarxan.models.grid import GridGeometry

_STRUCTURES = {
    "rook": np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=int),  # 4-connectivity
    "queen": np.ones((3, 3), dtype=int),  # 8-connectivity
}


@dataclass(eq=False)  # numpy patch_areas field breaks the auto __eq__ (cf. GridGeometry)
class MeshResult:
    """Effective mesh size of a habitat map. ``mesh`` is m_eff in ``cell_area`` units.

    To reproduce restoptr / landscapemetrics numeric MESH values (reported in hectares), pass
    ``cell_area`` in hectares (or divide ``mesh`` by 1e4 for m² cells).
    """

    mesh: float
    n_patches: int
    patch_areas: np.ndarray  # (n_patches,) area per patch, descending
    total_area: float  # A_total = n_valid_cells · cell_area

    @property
    def coherence(self) -> float:
        """Jaeger's degree of coherence ``C = Σ(A_i/A_total)² = mesh / A_total`` ∈ [0, 1]."""
        return self.mesh / self.total_area if self.total_area > 0 else 0.0

    @property
    def division(self) -> float:
        """Jaeger's degree of landscape division ``D = 1 − C`` ∈ [0, 1]."""
        return 1.0 - self.coherence


def compute_mesh(
    grid: GridGeometry,
    habitat_mask: np.ndarray,
    *,
    connectivity: str = "rook",
    cell_area: float | None = None,
) -> MeshResult:
    """MESH (effective mesh size, Jaeger 2000): ``Σ A_i² / A_total`` over habitat patches
    (connected components), with ``A_total`` = total landscape area (every valid cell).

    Parameters
    ----------
    grid:
        Raster grid descriptor. ``A_total = grid.n_pu · cell_area``.
    habitat_mask:
        ``(grid.n_pu,)`` bool in the grid's row-major valid-cell (== PU) order — habitat/not.
    connectivity:
        ``"rook"`` (4-neighbour, restoptr's default) or ``"queen"`` (8-neighbour).
    cell_area:
        Area of one cell; defaults to ``grid.cell_width · grid.cell_height``. Must be > 0.
    """
    if connectivity not in _STRUCTURES:
        msg = f"connectivity must be 'rook' or 'queen', got {connectivity!r}"
        raise ValueError(msg)
    habitat_mask = np.asarray(habitat_mask).astype(bool)
    if habitat_mask.shape != (grid.n_pu,):
        msg = f"habitat_mask must have length {grid.n_pu}, got {habitat_mask.shape}"
        raise ValueError(msg)

    area = float(cell_area) if cell_area is not None else grid.cell_width * grid.cell_height
    if area <= 0:
        msg = f"cell_area must be > 0, got {area}"
        raise ValueError(msg)
    total_area = grid.n_pu * area

    hab2d = np.zeros(grid.shape, dtype=bool)
    hab2d[grid.mask] = habitat_mask
    labels, n = ndimage.label(hab2d, structure=_STRUCTURES[connectivity])
    counts = np.bincount(labels.ravel())[1:] if n > 0 else np.array([], dtype=np.int64)
    patch_areas = np.sort(counts.astype(float) * area)[::-1]

    mesh = float((patch_areas**2).sum() / total_area) if total_area > 0 else 0.0
    return MeshResult(
        mesh=mesh,
        n_patches=int(patch_areas.size),
        patch_areas=patch_areas,
        total_area=float(total_area),
    )
