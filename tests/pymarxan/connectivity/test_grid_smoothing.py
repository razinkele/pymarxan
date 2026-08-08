"""Grid-convolution distribution smoothing (design 2026-08-08 §3/§6).

The vector path (smooth_distribution over a dense kernel) is the oracle: on a
grid whose truncation window covers every pairwise offset, the grid kernel is
IDENTICAL to the untruncated vector kernel, so results agree to FP regrouping
(allclose, never bitwise — convolution sums in a different order).
"""
from __future__ import annotations

import numpy as np
import pytest

from pymarxan.connectivity.smoothing import (
    distance_matrix_from_points,
    smooth_distribution,
    smooth_distribution_grid,
)
from pymarxan.models.grid import GridGeometry


def _grid(mask: np.ndarray, cw: float = 1.0, ch: float = 1.0) -> GridGeometry:
    return GridGeometry(
        x_min=0.0, y_max=float(mask.shape[0]) * ch, cell_width=cw, cell_height=ch,
        mask=np.asarray(mask, dtype=bool),
    )


def _vector_oracle(q: np.ndarray, grid: GridGeometry, alpha: float) -> np.ndarray:
    d = distance_matrix_from_points(grid.cell_centroids())
    return smooth_distribution(q, d, alpha)


@pytest.mark.parametrize("cw,ch", [(1.0, 1.0), (2.0, 0.5)])
def test_grid_matches_vector_full_window(cw: float, ch: float) -> None:
    rng = np.random.default_rng(0)
    mask = np.ones((6, 5), dtype=bool)
    grid = _grid(mask, cw, ch)
    q = rng.uniform(0, 3, size=grid.n_pu)
    # alpha small enough that the clipped window covers the whole grid.
    out = smooth_distribution_grid(q, grid, alpha=0.3)
    np.testing.assert_allclose(out, _vector_oracle(q, grid, 0.3), rtol=1e-10, atol=1e-13)


def test_grid_matches_vector_with_holes() -> None:
    rng = np.random.default_rng(1)
    mask = np.ones((7, 6), dtype=bool)
    mask[2:4, 2:4] = False  # interior hole
    mask[0, 0] = False      # corner notch
    grid = _grid(mask)
    q = rng.uniform(0, 2, size=grid.n_pu)
    out = smooth_distribution_grid(q, grid, alpha=0.4)
    np.testing.assert_allclose(out, _vector_oracle(q, grid, 0.4), rtol=1e-10, atol=1e-13)


def test_multi_column_matches_per_column() -> None:
    rng = np.random.default_rng(2)
    grid = _grid(np.ones((5, 5), dtype=bool))
    q = rng.uniform(0, 1, size=(grid.n_pu, 3))
    out = smooth_distribution_grid(q, grid, alpha=0.5)
    assert out.shape == q.shape
    for j in range(3):
        np.testing.assert_array_equal(
            out[:, j], smooth_distribution_grid(q[:, j], grid, alpha=0.5)
        )


@pytest.mark.parametrize("truncate", [1e-9, 1e-2])
def test_mass_conserved_at_any_truncation(truncate: float) -> None:
    rng = np.random.default_rng(3)
    mask = np.ones((10, 9), dtype=bool)
    mask[5, 3:6] = False
    grid = _grid(mask)
    q = rng.uniform(0, 4, size=grid.n_pu)
    out = smooth_distribution_grid(q, grid, alpha=1.2, truncate=truncate)
    # Conservation is exact at ANY truncation: the normalizer uses the same
    # truncated kernel (design §3).
    np.testing.assert_allclose(out.sum(), q.sum(), rtol=1e-12)


def test_single_source_compact_support_exact_zeros() -> None:
    mask = np.ones((15, 15), dtype=bool)
    grid = _grid(mask)
    q = np.zeros(grid.n_pu)
    src = 7 * 15 + 7  # centre cell (full mask: PU index == flat index)
    q[src] = 5.0
    out = smooth_distribution_grid(q, grid, alpha=1.5, truncate=1e-3)
    r = int(np.ceil(-np.log(1e-3) / 1.5))  # window radius in cells
    og = out.reshape(15, 15)
    inside = np.zeros((15, 15), dtype=bool)
    inside[max(0, 7 - r) : 7 + r + 1, max(0, 7 - r) : 7 + r + 1] = True
    # EXACT zeros outside the window (FFT dust masked by support dilation).
    assert (og[~inside] == 0.0).all()
    assert og[7, 7] > 0.0


def test_isolated_invalid_region_no_nan() -> None:
    # Invalid cells far from any valid cell have Z == 0; the masked division
    # must not let 0/0 NaN leak into the convolution (design §3 fix).
    mask = np.zeros((20, 20), dtype=bool)
    mask[:3, :3] = True  # valid island in one corner; rest invalid
    grid = _grid(mask)
    q = np.arange(1.0, grid.n_pu + 1)
    out = smooth_distribution_grid(q, grid, alpha=2.0, truncate=1e-6)
    assert np.isfinite(out).all()
    np.testing.assert_allclose(out.sum(), q.sum(), rtol=1e-12)


def test_huge_alpha_near_identity_and_single_column_grid() -> None:
    grid = _grid(np.ones((4, 4), dtype=bool))
    q = np.arange(1.0, 17.0)
    out = smooth_distribution_grid(q, grid, alpha=50.0)
    np.testing.assert_allclose(out, q, rtol=1e-8)  # self-weight dominates
    col = _grid(np.ones((6, 1), dtype=bool))  # x-radius clips to 0
    qc = np.arange(1.0, 7.0)
    outc = smooth_distribution_grid(qc, col, alpha=0.5)
    np.testing.assert_allclose(outc.sum(), qc.sum(), rtol=1e-12)


def test_nonnegative_under_fft_roundoff() -> None:
    # Design-review HIGH (found independently by two lenses): a high-amplitude
    # blob plus a remote unit source makes true in-window corner values sink
    # below the FFT noise floor — unclamped output goes NEGATIVE inside the
    # reach mask (3,570 cells at default truncate in the review's probe), which
    # would silently break the heap's monotonicity invariant downstream.
    mask = np.ones((256, 256), dtype=bool)
    grid = _grid(mask)
    q = np.zeros(grid.n_pu)
    qg = q.reshape(256, 256)
    qg[40:80, 40:80] = 1e12  # high-amplitude blob
    qg[200, 200] = 1.0       # remote lone unit source
    out = smooth_distribution_grid(qg.reshape(-1), grid, alpha=0.5, truncate=1e-9)
    assert (out >= 0.0).all()
    np.testing.assert_allclose(out.sum(), qg.sum(), rtol=1e-10)


def test_matches_explicit_truncated_kernel_oracle() -> None:
    # Pin the two-convolution formulation against an INDEPENDENTLY built
    # window-bounded column-normalized kernel matrix (stronger than the
    # full-window case: exercises the truncation itself). Both review lenses
    # measured <=6.3e-16.
    rng = np.random.default_rng(5)
    mask = np.ones((10, 9), dtype=bool)
    mask[4, 2:5] = False
    grid = _grid(mask)
    q = rng.uniform(0, 3, size=grid.n_pu)
    alpha, truncate = 1.0, 1e-2
    out = smooth_distribution_grid(q, grid, alpha, truncate=truncate)
    # Explicit oracle: truncated window, column-normalized over valid cells.
    cents = grid.cell_centroids()
    rx = int(np.ceil(-np.log(truncate) / (alpha * 1.0)))
    d = distance_matrix_from_points(cents)
    dxy = np.abs(cents[:, None, :] - cents[None, :, :])
    in_window = (dxy[:, :, 0] <= rx * 1.0 + 1e-12) & (dxy[:, :, 1] <= rx * 1.0 + 1e-12)
    K = np.where(in_window, np.exp(-alpha * d), 0.0)
    Kn = K / K.sum(axis=0)
    np.testing.assert_allclose(out, Kn @ q, rtol=1e-10, atol=1e-13)


def test_validation() -> None:
    grid = _grid(np.ones((3, 3), dtype=bool))
    q = np.ones(9)
    with pytest.raises(ValueError, match="alpha"):
        smooth_distribution_grid(q, grid, alpha=0.0)
    for bad in (0.0, 1.0, -0.5, 2.0):
        with pytest.raises(ValueError, match="truncate"):
            smooth_distribution_grid(q, grid, alpha=1.0, truncate=bad)
    with pytest.raises(ValueError, match="rows"):
        smooth_distribution_grid(np.ones(5), grid, alpha=1.0)
