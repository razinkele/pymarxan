"""RestorationProblem — restoration data model (grid + habitat/restorable/cost cell states).

Standalone domain model (cf. RiverNetwork / PhylogeneticTree) for restoptr-style restoration:
which restorable cells to convert to habitat, evaluated by ``compute_mesh`` via
``habitat_mask(restored)``. The core + ``from_arrays`` are rasterio-free (inlined
``_nodata_mask``); ``from_rasters`` lazily imports the S2 rasterio helpers from
``pymarxan.spatial.raster``.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

from pymarxan.models.grid import GridGeometry
from pymarxan.restoration.mesh import MeshResult, compute_mesh


def _nodata_mask(arr: np.ndarray, nodata: float | None) -> np.ndarray:
    """True where ``arr`` is nodata (NaN always; ``== nodata`` sentinel). Pure numpy.

    Behaviourally identical to ``pymarxan.spatial.raster._nodata_mask`` (pinned by a regression
    test) — inlined here so the rasterio-free core does not import the rasterio-carrying
    ``spatial.raster`` module.
    """
    m = np.isnan(arr) if np.issubdtype(arr.dtype, np.floating) else np.zeros(arr.shape, dtype=bool)
    if nodata is not None:
        m = m | (arr == nodata)
    return np.asarray(m, dtype=bool)


@dataclass(eq=False)  # numpy fields break the auto __eq__ (repo convention, cf. GridGeometry)
class RestorationProblem:
    """A restoration decision problem over a raster ``GridGeometry``.

    Arrays are ``(grid.n_pu,)`` in the grid's row-major valid-cell (== PU) order — the contract
    ``compute_mesh`` uses. ``grid.mask`` (the study region) is MESH's ``A_total`` denominator.
    """

    grid: GridGeometry
    existing_habitat: np.ndarray
    restorable: np.ndarray
    cost: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.existing_habitat = np.asarray(self.existing_habitat).astype(bool)
        self.restorable = np.asarray(self.restorable).astype(bool)
        if self.cost is None:
            self.cost = np.ones(self.grid.n_pu, dtype=float)
        else:
            self.cost = np.asarray(self.cost).astype(float)

    @property
    def n_pu(self) -> int:
        return self.grid.n_pu

    @property
    def restorable_indices(self) -> np.ndarray:
        """PU indices of restorable cells — the candidate move-set for a restoration optimizer."""
        idx: np.ndarray = np.flatnonzero(self.restorable)
        return idx

    def _check_restored(self, restored: np.ndarray) -> np.ndarray:
        restored = np.asarray(restored).astype(bool)
        if restored.shape != (self.n_pu,):
            raise ValueError(f"restored must have length {self.n_pu}, got {restored.shape}")
        if bool((restored & ~self.restorable).any()):
            raise ValueError("restored contains cells that are not restorable")
        return restored

    def habitat_mask(self, restored: np.ndarray) -> np.ndarray:
        """Post-restoration habitat map ``existing_habitat | restored`` (restored ⊆ restorable).

        The input ``compute_mesh`` scores. A tight optimizer loop can skip the public methods and
        call ``compute_mesh(self.grid, self.existing_habitat | restored)`` directly to avoid the
        per-call subset validation.
        """
        mask: np.ndarray = self.existing_habitat | self._check_restored(restored)
        return mask

    def baseline_mesh(self, **mesh_kwargs) -> MeshResult:
        """MESH of the current (pre-restoration) habitat map."""
        return compute_mesh(self.grid, self.existing_habitat, **mesh_kwargs)

    def restore_mesh(self, restored: np.ndarray, **mesh_kwargs) -> MeshResult:
        """MESH of the post-restoration habitat map (the objective an optimizer maximizes)."""
        return compute_mesh(self.grid, self.habitat_mask(restored), **mesh_kwargs)

    def restoration_cost(self, restored: np.ndarray) -> float:
        """Total cost of a restoration plan."""
        restored = self._check_restored(restored)
        assert self.cost is not None  # set in __post_init__
        return float(self.cost[restored].sum())

    def validate(self) -> list[str]:
        """Return a list of data errors (empty when valid); does not raise."""
        errs: list[str] = []
        n = self.n_pu
        for name, arr in (("existing_habitat", self.existing_habitat),
                          ("restorable", self.restorable), ("cost", self.cost)):
            if arr is None or arr.shape != (n,):
                shape = None if arr is None else arr.shape
                errs.append(f"{name} must have length {n}, got {shape}")
        if (self.existing_habitat.shape == (n,) and self.restorable.shape == (n,)
                and bool((self.existing_habitat & self.restorable).any())):
            errs.append("existing_habitat and restorable must be disjoint "
                        "(an already-habitat cell cannot be restored)")
        if self.cost is not None and self.cost.shape == (n,) and not (
                bool(np.all(np.isfinite(self.cost))) and bool(np.all(self.cost >= 0))):
            errs.append("cost must be finite and >= 0")
        return errs

    @classmethod
    def from_arrays(
        cls,
        existing_habitat: np.ndarray,
        restorable: np.ndarray,
        *,
        cost: np.ndarray | None = None,
        x_min: float,
        y_max: float,
        cell_width: float,
        cell_height: float,
        crs: str | None = None,
        mask_array: np.ndarray | None = None,
        nodata: float | None = None,
    ) -> RestorationProblem:
        """Build from aligned 2-D numpy arrays (pure numpy — no rasterio).

        The **validity mask** (study region) is the explicit ``mask_array`` (non-zero, non-nodata)
        if given, else the non-nodata footprint of ``existing_habitat`` — and it becomes
        ``GridGeometry.mask``, i.e. MESH's ``A_total`` denominator. ``existing_habitat`` /
        ``restorable`` are binarized (``> 0``); ``cost`` defaults to uniform ``1.0``. Restorable /
        cost data on cells outside the study region is dropped (with a warning).
        """
        eh = np.asarray(existing_habitat, dtype=float)
        rs = np.asarray(restorable, dtype=float)
        shape = eh.shape
        if len(shape) != 2:
            raise ValueError(f"arrays must be 2-D, got shape {shape}")
        cost_arr = None if cost is None else np.asarray(cost, dtype=float)
        mask_arr = None if mask_array is None else np.asarray(mask_array, dtype=float)
        for label, a in (("restorable", rs), ("cost", cost_arr), ("mask_array", mask_arr)):
            if a is not None and a.shape != shape:
                raise ValueError(f"{label} shape {a.shape} != existing_habitat shape {shape}")

        # validity precedence: explicit mask -> existing_habitat non-nodata footprint
        if mask_arr is not None:
            valid = (mask_arr != 0) & ~_nodata_mask(mask_arr, nodata)
        else:
            valid = ~_nodata_mask(eh, nodata)
        if not valid.any():
            raise ValueError("no valid cells (the validity mask is empty)")

        # M1: warn when restorable / cost carry real data outside the study region (dropped).
        rest_data = (rs > 0) & ~_nodata_mask(rs, nodata)
        dropped = int((rest_data & ~valid).sum())
        if dropped:
            warnings.warn(
                f"{dropped} restorable cell(s) lie outside the study region "
                "(nodata in the validity layer) and were dropped",
                stacklevel=2,
            )

        rows, cols = np.nonzero(valid)  # row-major == PU order
        grid = GridGeometry(x_min, y_max, cell_width, cell_height, valid, crs)

        existing_vec = (eh[rows, cols] > 0) & ~_nodata_mask(eh, nodata)[rows, cols]
        rest_vec = (rs[rows, cols] > 0) & ~_nodata_mask(rs, nodata)[rows, cols]
        if cost_arr is None:
            cost_vec = np.ones(rows.size, dtype=float)
        else:
            cost_vec = cost_arr[rows, cols]
            nd = _nodata_mask(cost_arr, nodata)[rows, cols]
            if nd.any():
                warnings.warn(
                    f"{int(nd.sum())} valid cell(s) have nodata cost; defaulting to 1.0",
                    stacklevel=2,
                )
            cost_vec = np.where(nd, 1.0, cost_vec)
        return cls(grid, existing_vec, rest_vec, cost_vec)

    @classmethod
    def from_rasters(
        cls,
        existing_habitat: str,
        restorable: str,
        *,
        cost: str | None = None,
        mask_raster: str | None = None,
        band: int = 1,
    ) -> RestorationProblem:
        """Build from aligned single-band north-up rasters (rasterio).

        ``existing_habitat`` defines the reference grid; restorable / cost / mask_raster align to
        it (shape/transform/CRS). Lazily imports the S2 rasterio helpers so the module stays
        rasterio-free at import.
        """
        from pymarxan.spatial.raster import _read, _read_aligned, _require_north_up

        eh, tf, shp, crs = _read(existing_habitat, band)
        _require_north_up(tf)
        rs = _read_aligned(restorable, band, "restorable", tf, shp, crs)
        cost_arr = None if cost is None else _read_aligned(cost, band, "cost", tf, shp, crs)
        mask_arr = (None if mask_raster is None
                    else _read_aligned(mask_raster, band, "mask_raster", tf, shp, crs))
        # north-up transform: x_min = tf.c, y_max = tf.f, cell_width = tf.a, cell_height = -tf.e
        return cls.from_arrays(
            eh, rs, cost=cost_arr,
            x_min=tf.c, y_max=tf.f, cell_width=tf.a, cell_height=-tf.e,
            crs=(crs.to_string() if crs is not None else None),
            mask_array=mask_arr,
        )
