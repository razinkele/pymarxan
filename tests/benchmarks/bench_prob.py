"""Phase 18 PROBMODE 3 SA inner-loop performance benchmark.

Pins per-flip cost for the Z-score chance-constraint penalty path —
the most expensive constraint type in the system because it runs a
full ``_compute_zscore_penalty`` evaluation per flip (O(n_feat) with
``scipy.stats.norm.sf`` per active feature).

Run via ``make bench`` / ``pytest tests/benchmarks/bench_prob.py``; not
part of ``make check``.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import pytest

from pymarxan.models.problem import ConservationProblem
from pymarxan.solvers.cache import ProblemCache

pytestmark = pytest.mark.bench


def _make_probmode3_problem(
    n_pu: int = 500, n_features: int = 10, seed: int = 0,
) -> ConservationProblem:
    """Synthetic PROBMODE 3 problem with per-cell Bernoulli probabilities."""
    rng = np.random.default_rng(seed)
    pu_ids = np.arange(1, n_pu + 1)
    pu = pd.DataFrame({
        "id": pu_ids,
        "cost": rng.uniform(1.0, 10.0, size=n_pu),
        "status": np.zeros(n_pu, dtype=int),
    })

    feat_ids = np.arange(1, n_features + 1)
    features = pd.DataFrame({
        "id": feat_ids,
        "name": [f"p_f{i}" for i in feat_ids],
        "target": [10.0] * n_features,
        "spf": [1.0] * n_features,
        "ptarget": [0.9] * n_features,  # every feature is PROBMODE 3 active
    })

    rows = []
    for fid in feat_ids:
        present = rng.random(n_pu) < 0.6
        for k in range(n_pu):
            if present[k]:
                rows.append({
                    "species": int(fid),
                    "pu": int(pu_ids[k]),
                    "amount": float(rng.uniform(0.5, 2.0)),
                    "prob": float(rng.uniform(0.05, 0.3)),
                })
    puvspr = pd.DataFrame(rows)

    return ConservationProblem(
        planning_units=pu, features=features, pu_vs_features=puvspr,
        parameters={"PROBMODE": 3, "BLM": 0.0},
    )


def test_probmode3_delta_under_2500us_at_500pu():
    """Per-flip ``compute_delta_objective`` under PROBMODE 3 must run
    in median < 2.5 ms on a 500-PU × 10-feature problem.

    The current implementation does a full Z-score recomputation
    pre/post flip (cache.py:515-528). The expected-matrix sum is
    O(n_selected · n_feat); the norm.sf loop is O(n_feat). Measured
    median on first install was ~1 ms; the 2.5 ms budget gives ~2.5×
    headroom for slower CI machines while still catching genuine
    regressions (e.g. accidental O(n_pu²) scans)."""
    problem = _make_probmode3_problem(n_pu=500, n_features=10, seed=0)
    cache = ProblemCache.from_problem(problem)
    assert cache.probmode == 3, "fixture is meant to be PROBMODE 3"

    rng = np.random.default_rng(0)
    selected = rng.random(cache.n_pu) > 0.5
    held = cache.compute_held(selected)
    total_cost = float(np.sum(cache.costs[selected]))

    samples: list[float] = []
    for _ in range(200):
        idx = int(rng.integers(0, cache.n_pu))
        t0 = time.perf_counter()
        cache.compute_delta_objective(idx, selected, held, total_cost, blm=0.0)
        samples.append(time.perf_counter() - t0)

    median_us = float(np.median(samples)) * 1e6
    assert median_us < 2500.0, (
        f"PROBMODE 3 delta median per-flip cost: {median_us:.1f} µs "
        "(budget 2500 µs). compute_delta_objective likely regressed — "
        "verify _compute_zscore_penalty still O(n_feat) per flip."
    )


def test_probmode3_full_objective_under_5ms_at_500pu():
    """Full objective recomputation (no incremental state) — budget 5 ms
    at 500 PU × 10 features. Used as a reference for delta speedup."""
    problem = _make_probmode3_problem(n_pu=500, n_features=10, seed=0)
    cache = ProblemCache.from_problem(problem)
    rng = np.random.default_rng(0)
    selected = rng.random(cache.n_pu) > 0.5
    held = cache.compute_held(selected)

    t0 = time.perf_counter()
    cache.compute_full_objective(selected, held, blm=0.0)
    elapsed = time.perf_counter() - t0

    assert elapsed < 0.005, (
        f"PROBMODE 3 full objective took {elapsed*1000:.1f} ms "
        "(budget 5 ms)."
    )
