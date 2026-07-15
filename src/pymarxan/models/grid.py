"""Grid-geometry descriptor for raster-grid planning units (S1)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(eq=False)
class GridGeometry:
    """A north-up, axis-aligned grid whose valid cells are planning units.

    ``mask`` is an ``(nrows, ncols)`` boolean array; ``True`` marks a valid cell
    (a planning unit). The valid cells in row-major order (row 0 = top) ARE the
    planning-unit order: PU row ``i`` <-> the ``i``-th valid cell <-> row ``i``
    of ``build_pu_feature_matrix``. Pure numpy/pandas (no shapely/rasterio).
    ``eq=False`` because the numpy ``mask`` breaks the auto ``__eq__``. Treat as
    immutable — do not mutate ``mask`` after construction.

    The analytic, polygon-free counterpart to two sibling modules it should not be
    confused with: ``spatial/grid.py`` (``generate_planning_grid`` — a materialized
    vector square grid) and ``models/geometry.py`` (``generate_grid`` — synthetic
    bounding boxes for the map).
    """

    x_min: float
    y_max: float
    cell_width: float
    cell_height: float
    mask: np.ndarray
    crs: str | None = None

    def __post_init__(self) -> None:
        mask = np.asarray(self.mask)
        if mask.ndim != 2 or mask.dtype != bool:
            raise ValueError("mask must be a 2-D boolean array")
        if self.cell_width <= 0 or self.cell_height <= 0:
            raise ValueError("cell_width and cell_height must be > 0")
        if not mask.any():
            raise ValueError("mask has no valid cells (all False)")
        self.mask = mask

    @property
    def shape(self) -> tuple[int, int]:
        return (self.mask.shape[0], self.mask.shape[1])

    @property
    def n_pu(self) -> int:
        return int(self.mask.sum())

    def valid_cells(self) -> list[tuple[int, int]]:
        """Valid ``(row, col)`` cells in row-major (top-down) order = PU order."""
        rows, cols = np.nonzero(self.mask)
        return list(zip(rows.tolist(), cols.tolist()))

    def cell_centroids(self) -> np.ndarray:
        """``(n_pu, 2)`` array of ``(x, y)`` cell centroids in PU order."""
        cells = self.valid_cells()
        out = np.empty((len(cells), 2), dtype=float)
        for i, (r, c) in enumerate(cells):
            out[i, 0] = self.x_min + (c + 0.5) * self.cell_width
            out[i, 1] = self.y_max - (r + 0.5) * self.cell_height
        return out

    def cell_bounds(self) -> list[tuple[float, float, float, float]]:
        """``(minx, miny, maxx, maxy)`` per PU in PU order."""
        out: list[tuple[float, float, float, float]] = []
        for r, c in self.valid_cells():
            minx = self.x_min + c * self.cell_width
            maxx = minx + self.cell_width
            maxy = self.y_max - r * self.cell_height
            miny = maxy - self.cell_height
            out.append((minx, miny, maxx, maxy))
        return out

    def build_boundary(self, pu_ids: np.ndarray | None = None) -> pd.DataFrame:
        """Analytic rook-adjacency boundary (columns ``id1``, ``id2``,
        ``boundary``), matching ``spatial.boundary.compute_boundary`` on the
        equivalent vector grid. ``pu_ids`` aligns to valid-cell order (defaults
        to ``1..n_pu``)."""
        cells = self.valid_cells()
        n = len(cells)
        if pu_ids is None:
            pu_ids = np.arange(1, n + 1)
        pu_ids = np.asarray(pu_ids)
        if len(pu_ids) != n:
            raise ValueError(f"pu_ids must have {n} entries, got {len(pu_ids)}")
        if len(set(pu_ids.tolist())) != n:
            raise ValueError("pu_ids must be unique")
        cell_to_id = {cell: int(pu_ids[i]) for i, cell in enumerate(cells)}
        shared: dict[int, float] = {int(pid): 0.0 for pid in pu_ids}
        rows: list[dict] = []
        for (r, c), pid in cell_to_id.items():
            # right neighbor shares a vertical edge (length cell_height);
            # down neighbor shares a horizontal edge (length cell_width).
            for nbr, edge in (((r, c + 1), self.cell_height), ((r + 1, c), self.cell_width)):
                nid = cell_to_id.get(nbr)
                if nid is not None:
                    rows.append({"id1": pid, "id2": nid, "boundary": edge})
                    shared[pid] += edge
                    shared[nid] += edge
        perimeter = 2.0 * (self.cell_width + self.cell_height)
        for pid in cell_to_id.values():
            self_b = perimeter - shared[pid]
            if self_b > 1e-10:
                rows.append({"id1": pid, "id2": pid, "boundary": self_b})
        return pd.DataFrame(rows, columns=["id1", "id2", "boundary"])
