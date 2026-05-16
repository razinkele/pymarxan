"""Bedrock + invariant tests for ``SepState``.

Phase 20 Task 11: delta-matches-full over random flips on dense AND sparse
sep-density problems (round-3 H13 parametrization), plus the apply-flip
consistency invariant test (round-2 A4 — pins the v0.3 incremental KD-tree
variant from silently drifting).
"""
from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Point

from pymarxan.models.problem import ConservationProblem
from pymarxan.solvers.cache import ProblemCache
from pymarxan.solvers.separation import (
    SepState,
    compute_sep_penalty_from_scratch,
)


def _make_grid_problem(
    n_pu: int = 9, spacing: float = 100.0,
    sep_density: float = 1.0, sepdistance: float = 150.0,
    sepnum: int = 3,
):
    """3×3 (or n_pu = sqrt(n_pu)²) grid; sep_density of [0,1] determines
    how many features are sep-active (1.0 = all, 0.2 = first 20%)."""
    side = int(np.sqrt(n_pu))
    n_pu = side * side
    coords = [(i * spacing, j * spacing) for i in range(side) for j in range(side)]
    pu = gpd.GeoDataFrame(
        {"id": list(range(1, n_pu + 1)),
         "cost": [1.0] * n_pu, "status": [0] * n_pu},
        geometry=[Point(x, y) for x, y in coords],
        crs="EPSG:3857",
    )
    # 5 features; sep_density fraction get sepdistance + sepnum.
    n_features = 5
    n_active = max(1, int(round(sep_density * n_features)))
    features = pd.DataFrame({
        "id": list(range(1, n_features + 1)),
        "name": [f"f{i}" for i in range(1, n_features + 1)],
        "target": [2.0] * n_features,
        "spf": [1.0] * n_features,
        "sepdistance": [sepdistance if i < n_active else 0.0
                        for i in range(n_features)],
        "sepnum": [sepnum if i < n_active else 1 for i in range(n_features)],
    })
    # Each PU contributes 1 unit of every feature.
    rows = []
    for fid in range(1, n_features + 1):
        for pid in range(1, n_pu + 1):
            rows.append({"species": fid, "pu": pid, "amount": 1.0})
    puvspr = pd.DataFrame(rows)
    return ConservationProblem(
        planning_units=pu, features=features, pu_vs_features=puvspr,
    )


# --- Round-3 H13: parametrize bedrock over density --------------------


@pytest.mark.parametrize("sep_density,seed", [
    (1.0, 1), (1.0, 2),    # dense: every feature is sep-active
    (0.2, 3), (0.2, 4),    # sparse: only 20% sep-active
])
def test_sepstate_delta_matches_full(sep_density, seed):
    """For 200 random flips, ``delta_penalty + previous_total`` must equal
    a from-scratch evaluation on the post-flip selection. Pins the
    v1 full-recompute SepState's delta-correctness contract."""
    p = _make_grid_problem(n_pu=9, sep_density=sep_density)
    cache = ProblemCache.from_problem(p)
    rng = np.random.default_rng(seed)
    selected = rng.random(cache.n_pu) > 0.5
    state = SepState.from_selection(cache, selected)

    # Seed total against compute_sep_penalty_from_scratch (sanity).
    _, expected_initial = compute_sep_penalty_from_scratch(p, selected)
    assert state.penalty_total() == pytest.approx(expected_initial, abs=1e-10)

    for _ in range(200):
        idx = int(rng.integers(0, cache.n_pu))
        adding = not bool(state.selected[idx])
        delta = state.delta_penalty(cache, idx, adding)
        before = state.penalty_total()
        state.apply_flip(cache, idx, adding)
        # apply_flip's new running total should equal before + delta.
        assert state.penalty_total() == pytest.approx(
            before + delta, abs=1e-10,
        ), (
            f"delta mismatch at flip idx={idx} adding={adding}: "
            f"before={before}, delta={delta}, "
            f"after={state.penalty_total()}"
        )
        # And it should match a fresh from-scratch evaluation.
        _, expected_full = compute_sep_penalty_from_scratch(p, state.selected)
        assert state.penalty_total() == pytest.approx(expected_full, abs=1e-9), (
            f"state total {state.penalty_total()} != fresh {expected_full}"
        )


# --- Round-2 A4: apply_flip consistency invariant ---------------------


def test_sepstate_from_selection_after_apply_flip_matches():
    """After a flip sequence + apply_flip chain, ``SepState.from_selection``
    on the live selection must produce the same penalty_total. Trivially
    true for v1 pure-recompute SepState, but pins the contract so the
    v0.3 incremental KD-tree variant cannot silently drift."""
    p = _make_grid_problem(n_pu=9, sep_density=1.0)
    cache = ProblemCache.from_problem(p)
    rng = np.random.default_rng(123)
    selected = rng.random(cache.n_pu) > 0.5
    state = SepState.from_selection(cache, selected)
    for _ in range(50):
        idx = int(rng.integers(0, cache.n_pu))
        adding = not bool(state.selected[idx])
        state.apply_flip(cache, idx, adding)
    rebuilt = SepState.from_selection(cache, state.selected)
    assert state.penalty_total() == pytest.approx(rebuilt.penalty_total(), abs=1e-10)
    np.testing.assert_array_equal(state.sep_counts, rebuilt.sep_counts)


def test_sepstate_inactive_problem_zero_penalty():
    """A non-sep problem produces SepState with zero penalty and zero counts."""
    p = _make_grid_problem(sep_density=0.0001)  # all features sepnum=1 → inactive
    p.features.loc[:, "sepdistance"] = 0.0
    p.features.loc[:, "sepnum"] = 1
    cache = ProblemCache.from_problem(p)
    assert not cache.separation_active
    selected = np.ones(cache.n_pu, dtype=bool)
    state = SepState.from_selection(cache, selected)
    assert state.penalty_total() == 0.0
    assert (state.sep_counts == 0).all()
