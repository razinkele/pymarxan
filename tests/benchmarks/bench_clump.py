"""Phase 19 TARGET2 / clumping SA inner-loop performance benchmark.

Pins per-flip cost for the ``ClumpState.delta_penalty`` path — per
affected feature it rebuilds the participant subgraph via
``scipy.sparse.csgraph.connected_components`` (cache.py / clumping.py).
The benchmark catches regressions in the affected-features filter and
in the connected-components recompute.

Run via ``make bench`` / ``pytest tests/benchmarks/bench_clump.py``;
not part of ``make check``.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import pytest

from pymarxan.models.problem import ConservationProblem
from pymarxan.solvers.cache import ProblemCache
from pymarxan.solvers.clumping import ClumpState

pytestmark = pytest.mark.bench


def _make_clumping_problem(
    n_pu: int = 400, n_features: int = 5, seed: int = 0,
) -> ConservationProblem:
    """Square-grid problem with all features marked TARGET2-active. Each
    feature contributes 1 unit per PU; boundary is the standard
    4-neighbour grid so the connected-components recompute has real
    edges to chew on."""
    rng = np.random.default_rng(seed)
    side = int(np.ceil(np.sqrt(n_pu)))
    n_pu = side * side
    pu_ids = np.arange(1, n_pu + 1)
    pu = pd.DataFrame({
        "id": pu_ids,
        "cost": rng.uniform(1.0, 10.0, size=n_pu),
        "status": np.zeros(n_pu, dtype=int),
    })

    feat_ids = np.arange(1, n_features + 1)
    features = pd.DataFrame({
        "id": feat_ids,
        "name": [f"c_f{i}" for i in feat_ids],
        "target": [n_pu * 0.3] * n_features,
        "spf": [1.0] * n_features,
        "target2": [3.0] * n_features,   # every feature is type-4
        "clumptype": [1] * n_features,   # CLUMPTYPE 1 (occ/2)
    })

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

    # 4-neighbour grid boundary.
    boundary_rows = []
    for r in range(side):
        for c in range(side):
            i = r * side + c + 1
            if c + 1 < side:
                boundary_rows.append({"id1": i, "id2": i + 1, "boundary": 1.0})
            if r + 1 < side:
                boundary_rows.append({"id1": i, "id2": i + side, "boundary": 1.0})

    return ConservationProblem(
        planning_units=pu, features=features, pu_vs_features=puvspr,
        boundary=pd.DataFrame(boundary_rows),
        parameters={"BLM": 1.0},
    )


def test_clumpstate_delta_under_5ms_at_400pu():
    """Per-flip ``ClumpState.delta_penalty`` median < 5 ms at 400 PU × 5
    type-4 features. The per-flip cost is O(n_features_with_target2 ·
    edges_in_participant_subgraph); on a 400-PU grid this measures
    ~2 ms; the 5 ms budget gives ~2.5× headroom for slower CI machines
    while catching genuine regressions (the affected-features short-
    circuit or a hidden O(n_pu²) scan)."""
    problem = _make_clumping_problem(n_pu=400, n_features=5, seed=0)
    cache = ProblemCache.from_problem(problem)
    assert cache.clumping_active, "fixture is meant to be clumping-active"

    rng = np.random.default_rng(0)
    selected = rng.random(cache.n_pu) > 0.5
    state = ClumpState.from_selection(cache, selected)

    samples: list[float] = []
    for _ in range(200):
        idx = int(rng.integers(0, cache.n_pu))
        adding = not bool(state.selected[idx])
        t0 = time.perf_counter()
        state.delta_penalty(cache, idx, adding)
        samples.append(time.perf_counter() - t0)

    median_us = float(np.median(samples)) * 1e6
    assert median_us < 5000.0, (
        f"ClumpState.delta_penalty median per-flip cost: {median_us:.1f} µs "
        "(budget 5000 µs / 5 ms). The affected-features short-circuit "
        "or connected-components recompute likely regressed."
    )


def test_clumpstate_construction_under_200ms_at_400pu():
    """``ClumpState.from_selection`` runs the full per-feature component
    build at construction. Budget: 200 ms at 400 PU × 5 features."""
    problem = _make_clumping_problem(n_pu=400, n_features=5, seed=0)
    cache = ProblemCache.from_problem(problem)
    rng = np.random.default_rng(0)
    selected = rng.random(cache.n_pu) > 0.5

    t0 = time.perf_counter()
    ClumpState.from_selection(cache, selected)
    elapsed = time.perf_counter() - t0

    assert elapsed < 0.200, (
        f"ClumpState.from_selection took {elapsed*1000:.1f} ms "
        "(budget 200 ms)."
    )
