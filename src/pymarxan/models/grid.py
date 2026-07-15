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
        to ``1..n_pu``). Vectorized (numpy), O(n_pu)."""
        mask = self.mask
        flat_valid = np.flatnonzero(mask.reshape(-1))  # row-major (C-order) == PU order
        n = int(flat_valid.size)
        if pu_ids is None:
            pu_ids = np.arange(1, n + 1)
        pu_ids = np.asarray(pu_ids)
        if len(pu_ids) != n:
            raise ValueError(f"pu_ids must have {n} entries, got {len(pu_ids)}")
        if len(np.unique(pu_ids)) != n:
            raise ValueError("pu_ids must be unique")

        # id grid: pu_ids scattered at valid cells (row-major); invalid cells unused.
        id_grid = np.zeros(mask.shape, dtype=np.int64)
        id_grid.reshape(-1)[flat_valid] = pu_ids  # id_grid is C-contiguous -> writable view
        id_at_valid = id_grid.reshape(-1)[flat_valid]  # pu_ids as int64, in PU order

        # Right edges (valid cell + valid right neighbor share a vertical edge = cell_height).
        both_r = mask[:, :-1] & mask[:, 1:]
        r_id1 = id_grid[:, :-1][both_r]
        r_id2 = id_grid[:, 1:][both_r]

        # Down edges (valid cell + valid down neighbor share a horizontal edge = cell_width).
        both_d = mask[:-1, :] & mask[1:, :]
        d_id1 = id_grid[:-1, :][both_d]
        d_id2 = id_grid[1:, :][both_d]

        # Self-boundary = exposed sides = perimeter - shared, per valid cell.
        has_left = np.zeros_like(mask)
        has_left[:, 1:] = mask[:, :-1]
        has_right = np.zeros_like(mask)
        has_right[:, :-1] = mask[:, 1:]
        has_up = np.zeros_like(mask)
        has_up[1:, :] = mask[:-1, :]
        has_down = np.zeros_like(mask)
        has_down[:-1, :] = mask[1:, :]
        self_grid = (
            (2 - has_left.astype(np.int64) - has_right.astype(np.int64)) * self.cell_height
            + (2 - has_up.astype(np.int64) - has_down.astype(np.int64)) * self.cell_width
        )
        self_vals = self_grid.reshape(-1)[flat_valid]
        keep = self_vals > 1e-10
        s_ids = id_at_valid[keep]
        s_vals = self_vals[keep]

        id1 = np.concatenate([r_id1, d_id1, s_ids])
        id2 = np.concatenate([r_id2, d_id2, s_ids])
        boundary = np.concatenate([
            np.full(r_id1.size, self.cell_height),
            np.full(d_id1.size, self.cell_width),
            s_vals,
        ])
        return pd.DataFrame({"id1": id1, "id2": id2, "boundary": boundary})
