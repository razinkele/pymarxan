"""Perf budget: raster-scale rank_removal (design §10). Run deliberately:
/opt/micromamba/envs/shiny/bin/pytest tests/benchmarks/bench_zonation.py -m bench -v
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import pytest

from pymarxan.models.grid import GridGeometry
from pymarxan.models.problem import ConservationProblem
from pymarxan.zonation.rank_removal import rank_removal

pytestmark = pytest.mark.bench


def _grid_problem(
    side: int, n_feat: int = 8, seed: int = 0, with_grid: bool = False
) -> ConservationProblem:
    """side x side grid; each feature occupies a random compact block (locality)."""
    rng = np.random.default_rng(seed)
    n_pu = side * side
    pu_ids = np.arange(1, n_pu + 1)
    planning_units = pd.DataFrame(
        {"id": pu_ids, "cost": np.ones(n_pu), "status": np.zeros(n_pu, dtype=int)}
    )
    feat_ids = list(range(1, n_feat + 1))
    features = pd.DataFrame(
        {
            "id": feat_ids,
            "name": [f"f{j}" for j in feat_ids],
            "target": [1.0] * n_feat,
            "spf": [1.0] * n_feat,
        }
    )
    frames = []
    for fid in feat_ids:
        r0, c0 = rng.integers(0, side // 2, size=2)
        h, w_ = rng.integers(side // 4, side // 2, size=2)
        rr, cc = np.meshgrid(
            np.arange(r0, min(r0 + h, side)), np.arange(c0, min(c0 + w_, side)),
            indexing="ij",
        )
        cells = (rr * side + cc).ravel() + 1
        amounts = rng.integers(1, 5, size=cells.size).astype(float)
        frames.append(
            pd.DataFrame({"species": fid, "pu": cells, "amount": amounts})
        )
    grid = None
    if with_grid:
        grid = GridGeometry(
            x_min=0.0,
            y_max=float(side),
            cell_width=1.0,
            cell_height=1.0,
            mask=np.ones((side, side), dtype=bool),
        )
    return ConservationProblem(planning_units, features, pd.concat(frames), grid=grid)


def test_rank_removal_scale_budget() -> None:
    p = _grid_problem(300)  # 90_000 cells
    t0 = time.perf_counter()
    res = rank_removal(p, rule="caz", warp=90)  # n // 1000 batches
    elapsed = time.perf_counter() - t0
    assert len(res.removal_order) == 90_000
    # Generous budget for slow machines; the dense engine takes >> this.
    assert elapsed < 60.0, f"raster-scale rank_removal took {elapsed:.1f}s"


def test_rank_removal_warp1_heap_budget() -> None:
    # The claim to pin: exact warp=1 via the lazy heap is FASTER than the batch
    # path it replaces (the naive per-pop variant did not finish at this size).
    # Absolute budgets proved machine-relative (measured 2026-08-08: heap
    # 102.5s, batch 138.4s on the reference machine; design-review batch
    # reference 120.7s), so assert relative order plus a DNF-catching ceiling.
    p = _grid_problem(300)  # 90_000 cells
    t0 = time.perf_counter()
    res = rank_removal(p, rule="caz", warp=1, curve_every=1000)
    heap_elapsed = time.perf_counter() - t0
    assert len(res.removal_order) == 90_000
    assert heap_elapsed < 300.0, f"warp=1 heap took {heap_elapsed:.1f}s (DNF-class)"
    t0 = time.perf_counter()
    rank_removal(p, rule="caz", warp=1, curve_every=1000, _force_batch=True)
    batch_elapsed = time.perf_counter() - t0
    # 1.05 slack absorbs machine noise; the heap must not be slower than batch.
    assert heap_elapsed < batch_elapsed * 1.05, (
        f"heap {heap_elapsed:.1f}s vs batch {batch_elapsed:.1f}s"
    )


def test_grid_smoothing_rank_removal_budget() -> None:
    from pymarxan.zonation.smoothing import GridSmoothingSpec

    p = _grid_problem(300, with_grid=True)  # 90_000 cells, compact blocks
    # alpha=2.0 -> window radius ~11: small enough that smoothing genuinely
    # preserves sparsity (review #3: alpha=0.5/truncate=1e-9 gives radius 42
    # and 36%-dense output, which cannot witness the sparsity claim).
    spec = GridSmoothingSpec(alpha=2.0)
    t0 = time.perf_counter()
    res = rank_removal(p, rule="caz", warp=90, smoothing=spec)
    elapsed = time.perf_counter() - t0
    assert len(res.removal_order) == 90_000
    assert elapsed < 120.0, f"grid-smoothed rank_removal took {elapsed:.1f}s"
    # Sparsity witness: rebuild the smoothed matrix the way the dispatch does
    # and assert it stays well under dense.
    from pymarxan.connectivity.smoothing import smooth_distribution_grid

    base = p.build_pu_feature_csr()
    csc = base.tocsc()
    nnz = 0
    for j in range(base.shape[1]):
        col = np.zeros(base.shape[0])
        sl = slice(csc.indptr[j], csc.indptr[j + 1])
        col[csc.indices[sl]] = csc.data[sl]
        nnz += int(
            (smooth_distribution_grid(col, p.grid, 2.0) > 0).sum()
        )
    assert nnz < 0.25 * base.shape[0] * base.shape[1], f"smoothed nnz {nnz}"
