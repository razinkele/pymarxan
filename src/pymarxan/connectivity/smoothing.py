"""Distribution smoothing via a dispersal kernel.

Spread each planning unit's feature amount to nearby units using a
negative-exponential dispersal kernel, so a solver values *being near*
abundance, not only holding it. This is the planning-unit (vector)
analogue of Zonation's distribution smoothing.
"""
from __future__ import annotations

import numpy as np

from pymarxan.connectivity.decay import negative_exponential
from pymarxan.models.grid import GridGeometry


def distance_matrix_from_points(coords: np.ndarray) -> np.ndarray:
    """Euclidean pairwise distance matrix from point coordinates.

    Args:
        coords: ``(n, 2)`` array of planning-unit coordinates (e.g. cell
            centroids), in the units distances should be measured in.

    Returns:
        ``(n, n)`` symmetric matrix of Euclidean distances, zero diagonal.
    """
    from scipy.spatial.distance import cdist

    coords = np.asarray(coords, dtype=float)
    result: np.ndarray = cdist(coords, coords)
    return result


def smooth_distribution(
    amounts: np.ndarray,
    distances: np.ndarray,
    alpha: float,
    *,
    normalize: bool = True,
) -> np.ndarray:
    """Smooth a per-unit feature distribution with a dispersal kernel.

    The kernel is ``K_ij = exp(-alpha * distance_ij)`` (so the diagonal is
    1). With ``normalize=True`` the kernel is column-normalised
    (``K / K.sum(axis=0)``): each source unit's outgoing kernel weights sum
    to 1, so the redistribution conserves total amount
    (``sum(output) == sum(input)``). With ``normalize=False`` the result is
    the raw ``K @ amounts`` accumulation (total grows).

    Args:
        amounts: Length-``n`` array of per-unit amounts for one feature, or an
            ``(n, m)`` array to smooth ``m`` features at once (each column).
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


def smooth_distribution_grid(
    amounts: np.ndarray,
    grid: GridGeometry,
    alpha: float,
    *,
    truncate: float = 1e-9,
) -> np.ndarray:
    """Grid-convolution distribution smoothing (raster analogue of
    :func:`smooth_distribution`).

    Same negative-exponential, mass-conserving (source-normalised) kernel, but
    evaluated as a truncated-window 2-D convolution on the grid — O(cells·log)
    per feature instead of an n×n kernel. Mass is conserved over valid cells
    EXACTLY at any ``truncate``, because the normaliser uses the same truncated
    kernel (each source's outgoing weights sum to 1 over its reachable valid
    cells). Truncation is window-bounded: when the clipped window covers the
    whole grid, results equal the untruncated vector kernel's up to FP
    regrouping (allclose, not bitwise).

    Zonation's own smoothing is unnormalized kernel accumulation (Moilanen
    et al. 2005, doi:10.1098/rspb.2005.3164); this function additionally
    conserves mass per source — an edge-corrected variant whose rankings
    match the unnormalized transform wherever a source's truncated window
    fits inside the valid mask (CAZ/ABF scores are invariant to per-feature
    constant scaling), differing only near edges/holes.

    Common Zonation guidance sets alpha = 2 / (species mean dispersal
    distance) — the mean-dispersal identity of the 2-D kernel (Westwood
    et al. 2020, doi:10.3390/d12020061); alpha is in inverse CRS distance
    units. Zonation 5 itself truncates the kernel tail (Moilanen et al.
    2022, doi:10.1111/2041-210X.13819).

    Args:
        amounts: ``(n_pu,)`` or ``(n_pu, m)`` per-PU amounts, rows in the
            grid's row-major valid-cell order (== CSR row order).
        grid: The problem's :class:`GridGeometry`.
        alpha: Negative-exponential decay rate (> 0), per CRS distance unit.
        truncate: Window cutoff in kernel-value terms, in (0, 1): the window
            radius per axis is ``ceil(-ln(truncate) / (alpha·cell_size))``,
            clipped to the grid extent.

    Returns:
        Smoothed amounts, same shape as ``amounts``.
    """
    from scipy.signal import oaconvolve

    if alpha <= 0:
        raise ValueError(f"alpha must be > 0, got {alpha}")
    if not 0.0 < truncate < 1.0:
        raise ValueError(f"truncate must be in (0, 1), got {truncate}")
    amounts = np.asarray(amounts, dtype=float)
    one_d = amounts.ndim == 1
    cols = amounts[:, None] if one_d else amounts
    mask = grid.mask
    nrows, ncols_g = mask.shape
    flat_valid = np.flatnonzero(mask.reshape(-1))
    if cols.shape[0] != flat_valid.size:
        raise ValueError(
            f"amounts must have {flat_valid.size} rows, got {cols.shape[0]}"
        )

    cw, ch = abs(grid.cell_width), abs(grid.cell_height)
    rx = min(int(np.ceil(-np.log(truncate) / (alpha * cw))), ncols_g - 1)
    ry = min(int(np.ceil(-np.log(truncate) / (alpha * ch))), nrows - 1)
    dy, dx = np.meshgrid(
        np.arange(-ry, ry + 1), np.arange(-rx, rx + 1), indexing="ij"
    )
    kernel = np.exp(-alpha * np.sqrt((dx * cw) ** 2 + (dy * ch) ** 2))

    m = mask.astype(float)
    z = oaconvolve(m, kernel, mode="same")
    box = np.ones(kernel.shape)  # for the reach mask (box conv beats
    # scipy.ndimage.binary_dilation, which is slow for large structures)

    out = np.empty_like(cols)
    for j in range(cols.shape[1]):
        plane = np.zeros(mask.shape)
        plane.reshape(-1)[flat_valid] = cols[:, j]
        ratio = np.zeros_like(plane)
        # Masked division (design §3): at invalid cells far from any valid
        # cell z == 0, and a bare plane/z would seed NaN that the convolution
        # spreads everywhere.
        np.divide(plane, z, out=ratio, where=mask)
        smoothed = oaconvolve(ratio, kernel, mode="same")
        # Restore exact compact support: FFT writes ~1e-16 dust everywhere;
        # true values are zero outside (source support ⊕ window) ∩ mask.
        reach = oaconvolve((plane != 0).astype(float), box, mode="same") > 0.5
        smoothed[~(reach & mask)] = 0.0
        # Clamp FFT roundoff INSIDE the reach: true corner values ~q·truncate^√2
        # can sink below the FFT noise floor (which scales with total landscape
        # mass) and come out negative — and negative amounts would silently
        # invert the warp=1 heap's monotonicity invariant downstream (they
        # bypass raw-amount validation). Design-review HIGH; clipped mass is at
        # noise scale, far inside the conservation tolerance.
        np.maximum(smoothed, 0.0, out=smoothed)
        out[:, j] = smoothed.reshape(-1)[flat_valid]
    return out[:, 0] if one_d else out
