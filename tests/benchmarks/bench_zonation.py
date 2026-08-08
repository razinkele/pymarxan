"""Perf budget: raster-scale rank_removal (design §10). Run deliberately:
/opt/micromamba/envs/shiny/bin/pytest tests/benchmarks/bench_zonation.py -m bench -v
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import pytest

from pymarxan.models.problem import ConservationProblem
from pymarxan.zonation.rank_removal import rank_removal

pytestmark = pytest.mark.bench


def _grid_problem(side: int, n_feat: int = 8, seed: int = 0) -> ConservationProblem:
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
    return ConservationProblem(planning_units, features, pd.concat(frames))


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
