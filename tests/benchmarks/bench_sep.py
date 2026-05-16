"""Phase 20 SepState performance benchmark (round-2 CR1 gate).

Pins the round-2 CR1 claim — `SepState.delta_penalty` allocates a per-call
k×k pairwise-distance matrix on the candidate sub-array only (never the
full n_pu×n_pu). Without that fix the per-flip cost blows up on large
problems; with it the cost is bounded by selection footprint.

Run via ``make bench`` / ``pytest tests/benchmarks/bench_sep.py``; not
part of ``make check`` because the runtime is variable across machines.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import pytest

from pymarxan.models.problem import ConservationProblem
from pymarxan.solvers.cache import ProblemCache
from pymarxan.solvers.separation import SepState

pytestmark = pytest.mark.bench


def _make_sep_problem(
    n_pu: int = 500, n_sep_features: int = 5, seed: int = 0,
) -> ConservationProblem:
    """Synthetic sep-active problem with xloc/yloc coords (avoids geopandas).

    Uses a square grid spacing of 100 units; feature targets and
    sepdistance scaled so the constraints actually bind.
    """
    rng = np.random.default_rng(seed)
    side = int(np.ceil(np.sqrt(n_pu)))
    n_pu = side * side
    xs = (np.arange(n_pu) % side) * 100.0
    ys = (np.arange(n_pu) // side) * 100.0

    pu_ids = np.arange(1, n_pu + 1)
    pu = pd.DataFrame({
        "id": pu_ids,
        "cost": rng.uniform(1.0, 10.0, size=n_pu),
        "status": np.zeros(n_pu, dtype=int),
        "xloc": xs,
        "yloc": ys,
    })

    feat_ids = np.arange(1, n_sep_features + 1)
    features = pd.DataFrame({
        "id": feat_ids,
        "name": [f"sep_f{i}" for i in feat_ids],
        "target": [10.0] * n_sep_features,
        "spf": [1.0] * n_sep_features,
        "sepdistance": [200.0] * n_sep_features,  # 2× grid spacing
        "sepnum": [3] * n_sep_features,
    })

    # Each feature is present on roughly 60% of the PUs at amount = 1.
    rows = []
    for fid in feat_ids:
        present = rng.random(n_pu) < 0.6
        for k in range(n_pu):
            if present[k]:
                rows.append({
                    "species": int(fid),
                    "pu": int(pu_ids[k]),
                    "amount": 1.0,
                })
    puvspr = pd.DataFrame(rows)

    return ConservationProblem(
        planning_units=pu, features=features, pu_vs_features=puvspr,
        parameters={"BLM": 0.0},
    )


def test_sepstate_delta_under_500us_at_500pu():
    """Per-flip SepState.delta_penalty median < 500 µs at 500 PU × 5
    sep-active features.

    The original CR1 budget was 200 µs at 5000 PU × 10 features; this
    test uses a smaller problem so it runs cheaply on every bench, but
    it still exercises the k×k pdist path that CR1 fixed."""
    problem = _make_sep_problem(n_pu=500, n_sep_features=5, seed=0)
    cache = ProblemCache.from_problem(problem)
    assert cache.separation_active, "fixture is meant to be sep-active"

    rng = np.random.default_rng(0)
    selected = rng.random(cache.n_pu) > 0.5
    state = SepState.from_selection(cache, selected)

    # 200 flips; measure each individually so we can take a median
    # (avoids skew from any one slow first call).
    samples: list[float] = []
    for _ in range(200):
        idx = int(rng.integers(0, cache.n_pu))
        adding = not bool(state.selected[idx])
        t0 = time.perf_counter()
        state.delta_penalty(cache, idx, adding)
        samples.append(time.perf_counter() - t0)

    median_us = float(np.median(samples)) * 1e6
    assert median_us < 500.0, (
        f"SepState.delta_penalty median per-flip cost: {median_us:.1f} µs "
        "(budget 500 µs). Allocation hot-path likely regressed — verify "
        "count_separation still allocates on the candidate sub-array only."
    )


def test_sepstate_construction_under_50ms_at_500pu():
    """SepState.from_selection runs the full count_separation for every
    sep-active feature once. Budget: 50 ms at 500 PU × 5 features."""
    problem = _make_sep_problem(n_pu=500, n_sep_features=5, seed=0)
    cache = ProblemCache.from_problem(problem)
    rng = np.random.default_rng(0)
    selected = rng.random(cache.n_pu) > 0.5

    t0 = time.perf_counter()
    SepState.from_selection(cache, selected)
    elapsed = time.perf_counter() - t0

    assert elapsed < 0.050, (
        f"SepState.from_selection took {elapsed*1000:.1f} ms "
        "(budget 50 ms). count_separation likely doing too much work."
    )
