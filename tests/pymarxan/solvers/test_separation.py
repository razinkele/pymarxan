"""Tests for ``pymarxan.solvers.separation`` math layer.

Phase 20 Batch 2:
- :func:`compute_sep_penalty` — Marxan hyperbolic curve verbatim from
  ``computation.hpp::computeSepPenalty:15-27``.
- :func:`count_separation` — greedy admission in ascending PU-id order,
  capped at ``sepnum`` (mirrors ``clumping.cpp::CountSeparation2``).
- :func:`compute_sep_penalty_from_scratch` — reference per-feature evaluator.
- :func:`evaluate_solution_separation` — post-hoc Solution attr populator.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pymarxan.models.problem import ConservationProblem
from pymarxan.solvers.separation import (
    compute_sep_penalty,
    compute_sep_penalty_from_scratch,
    count_separation,
    evaluate_solution_separation,
)

# --- Task 6: compute_sep_penalty (hyperbolic curve, parametrized) -------

# Marxan formula: fval = max(count, 1)/sepnum; pen = 1/(7·fval + 0.2) - 1/7.2
# (with count == 0 bumped to fval = 1/sepnum to prevent blow-up).
# Hand-verified expected values rounded to 4 decimals.
_SEP_PENALTY_CASES = [
    # (count, sepnum, expected)  --  exact endpoint or hand computation
    (0, 0, 0.0),                # sepnum<=0 disabled → 0 by guard
    (5, 0, 0.0),                # sepnum<=0 disabled regardless of count
    (1, 1, 0.0),                # fval = 1 (target met) → exactly 0
    (5, 5, 0.0),                # fval = 1
    (0, 3, 1.0 / (7.0 * (1.0 / 3.0) + 0.2) - 1.0 / 7.2),  # count=0 bumps to 1/sepnum
    (1, 3, 1.0 / (7.0 * (1.0 / 3.0) + 0.2) - 1.0 / 7.2),  # same as above
    (2, 3, 1.0 / (7.0 * (2.0 / 3.0) + 0.2) - 1.0 / 7.2),
    (3, 3, 0.0),                # met exactly
    (0, 5, 1.0 / (7.0 * (1.0 / 5.0) + 0.2) - 1.0 / 7.2),
    (4, 5, 1.0 / (7.0 * (4.0 / 5.0) + 0.2) - 1.0 / 7.2),
    (5, 5, 0.0),
]


@pytest.mark.parametrize("count,sepnum,expected", _SEP_PENALTY_CASES)
def test_compute_sep_penalty_table(count, sepnum, expected):
    """Marxan hyperbolic curve verbatim — every boundary case."""
    assert compute_sep_penalty(count, sepnum) == pytest.approx(expected, abs=1e-12)


def test_compute_sep_penalty_monotone_nonincreasing():
    """Penalty must decrease (or stay flat) as ``count`` rises for fixed sepnum."""
    sepnum = 5
    penalties = [compute_sep_penalty(c, sepnum) for c in range(sepnum + 1)]
    for prev, cur in zip(penalties, penalties[1:]):
        assert cur <= prev + 1e-12, f"non-monotone: {penalties}"


def test_compute_sep_penalty_zero_when_target_met():
    """``count >= sepnum`` always returns exactly 0 (Marxan invariant)."""
    for sepnum in (1, 2, 3, 5, 10):
        for over_count in (sepnum, sepnum + 1, sepnum + 100):
            assert compute_sep_penalty(over_count, sepnum) == pytest.approx(
                0.0, abs=1e-12,
            )


# --- Task 7: count_separation (greedy admission) ------------------------


def test_count_separation_linear_arrangement():
    """5 PUs on a line, spacing 100. sepdistance=150 → admit alternating."""
    coords = np.array([[0.0, 0.0], [100.0, 0.0], [200.0, 0.0],
                       [300.0, 0.0], [400.0, 0.0]])
    selected = np.array([True] * 5)
    amounts = np.array([1.0] * 5)
    # PU 0 admitted; PU 1 too close (100); PU 2 ≥ 150 from PU 0 admitted;
    # PU 3 too close to PU 2 (100); PU 4 ≥ 150 from PU 2 admitted. → 3.
    assert count_separation(selected, amounts, coords, 150.0, sepnum=5) == 3


def test_count_separation_no_candidates():
    """Empty selection → 0."""
    coords = np.array([[0.0, 0.0], [10.0, 10.0]])
    selected = np.array([False, False])
    amounts = np.array([1.0, 1.0])
    assert count_separation(selected, amounts, coords, 5.0, sepnum=3) == 0


def test_count_separation_no_amount():
    """Selected PUs with zero amount of the feature aren't candidates."""
    coords = np.array([[0.0, 0.0], [10.0, 10.0]])
    selected = np.array([True, True])
    amounts = np.array([0.0, 0.0])
    assert count_separation(selected, amounts, coords, 5.0, sepnum=3) == 0


def test_count_separation_sepdistance_zero_returns_min_with_sepnum():
    """``sepdistance=0`` → every pair is trivially separated → return
    min(candidates, sepnum). Same as the trivially-satisfied case."""
    coords = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    selected = np.array([True, True, True, True])
    amounts = np.array([1.0, 1.0, 1.0, 1.0])
    assert count_separation(selected, amounts, coords, 0.0, sepnum=3) == 3
    assert count_separation(selected, amounts, coords, 0.0, sepnum=10) == 4


def test_count_separation_caps_at_sepnum():
    """Marxan ``CountSeparation2`` short-circuits at ``sepnum``; pymarxan
    matches. Otherwise we'd waste compute on a plateau-at-zero curve."""
    # 10 PUs widely separated, sepnum=3 — return exactly 3, not 10.
    coords = np.array([[i * 100.0, 0.0] for i in range(10)])
    selected = np.array([True] * 10)
    amounts = np.array([1.0] * 10)
    assert count_separation(selected, amounts, coords, 50.0, sepnum=3) == 3


def test_count_separation_uses_pu_id_order_not_amount():
    """Round-1 C2: Marxan greedy admits in PU-id (insertion) order, NOT
    descending amount. With two equally-close candidates, the smaller-id
    one wins regardless of which has the larger amount."""
    # Three PUs at (0, 0), (10, 0), (20, 0). sepdistance=15 → can admit
    # PU 0 and PU 2 (distance 20); PU 1 (distance 10 from PU 0) is rejected.
    # If we sorted by amount descending, PU 1 (amount 100) would come first,
    # be admitted, and then exclude PU 0 (amount 1). Then PU 2 (amount 1)
    # would be admitted but at distance 10 from PU 1 → rejected → count=1.
    # PU-id order: PU 0 admitted, PU 1 rejected, PU 2 admitted → count=2.
    coords = np.array([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0]])
    selected = np.array([True, True, True])
    amounts = np.array([1.0, 100.0, 1.0])  # PU 1 has biggest amount
    assert count_separation(selected, amounts, coords, 15.0, sepnum=5) == 2


def test_count_separation_square_layout():
    """4 PUs at unit square corners; sepdistance=1.5 → diagonal pair only."""
    coords = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    selected = np.array([True, True, True, True])
    amounts = np.array([1.0, 1.0, 1.0, 1.0])
    # PU 0 admitted; PU 1 dist 1.0 rejected; PU 2 dist 1.0 from PU 0 rejected;
    # PU 3 dist sqrt(2) ≈ 1.414 from PU 0 — under 1.5 → also rejected. → 1.
    assert count_separation(selected, amounts, coords, 1.5, sepnum=5) == 1
    # With sepdistance=1.4 < sqrt(2): PU 0 admit, PU 3 admit (dist sqrt(2)). → 2
    assert count_separation(selected, amounts, coords, 1.4, sepnum=5) == 2


# --- Task 8: compute_sep_penalty_from_scratch + evaluate_solution_separation ----


def _make_separation_problem() -> ConservationProblem:
    """4 PUs at unit-square corners + 2 features; feature 1 sep-active."""
    import geopandas as gpd
    from shapely.geometry import Point

    pu = gpd.GeoDataFrame(
        {"id": [1, 2, 3, 4], "cost": [1.0] * 4, "status": [0] * 4},
        geometry=[Point(0, 0), Point(10, 0), Point(0, 10), Point(10, 10)],
        crs="EPSG:3857",
    )
    features = pd.DataFrame({
        "id": [1, 2],
        "name": ["a", "b"],
        "target": [4.0, 4.0],
        "spf": [1.0, 1.0],
        "sepdistance": [12.0, 0.0],  # only feature 1 is sep-active
        "sepnum": [3, 1],
    })
    # Each PU contributes 1 unit of each feature
    puvspr = pd.DataFrame({
        "species": [1, 1, 1, 1, 2, 2, 2, 2],
        "pu":      [1, 2, 3, 4, 1, 2, 3, 4],
        "amount":  [1.0] * 8,
    })
    return ConservationProblem(
        planning_units=pu, features=features, pu_vs_features=puvspr,
    )


def test_compute_sep_penalty_from_scratch_per_feature():
    """Returns (counts, total_penalty). Counts is per-feature, only
    sep-active features contribute; inactive features get count=0 and
    penalty=0 ignored."""
    p = _make_separation_problem()
    selected = np.array([True, True, True, True])  # all 4 PUs
    counts, total = compute_sep_penalty_from_scratch(p, selected)
    # sepdistance=12 on 10×10 square: diagonal sqrt(200) ≈ 14.14, only
    # diagonal pairs admitted. Greedy from PU 0: admit; reject PU 1 (10);
    # reject PU 2 (10); admit PU 3 (sqrt(200) ≈ 14.14 > 12) → count = 2.
    assert counts[0] == 2
    # sepnum=3, count=2 → non-zero penalty.
    assert total > 0


def test_evaluate_solution_separation_keys_only_sep_active():
    """``evaluate_solution_separation`` returns (shortfalls dict, total).
    Dict keys are feature ids that are sep-active (sepnum>1 AND sepdistance>0)."""
    p = _make_separation_problem()
    selected = np.array([True, True, True, True])
    shortfalls, total = evaluate_solution_separation(p, selected)
    assert set(shortfalls.keys()) == {1}, "only feature 1 is sep-active"
    # sepnum=3, count=2 → shortfall = 1
    assert shortfalls[1] == 1
    assert total > 0


def test_evaluate_solution_separation_zero_when_target_met():
    """If selection achieves count >= sepnum, total penalty is 0."""
    p = _make_separation_problem()
    # Lower sepnum to 2 so it can be met
    p.features.loc[p.features["id"] == 1, "sepnum"] = 2
    selected = np.array([True, True, True, True])
    shortfalls, total = evaluate_solution_separation(p, selected)
    assert shortfalls[1] == 0
    assert total == pytest.approx(0.0, abs=1e-12)


def test_evaluate_solution_separation_empty_selection():
    """Empty selection → shortfall = sepnum (count is 0)."""
    p = _make_separation_problem()
    selected = np.zeros(4, dtype=bool)
    shortfalls, total = evaluate_solution_separation(p, selected)
    assert shortfalls[1] == 3
    assert total > 0


# --- Task 9: build_solution wiring --------------------------------------


def test_build_solution_populates_sep_fields_when_active():
    """``build_solution`` populates ``sep_shortfalls`` and ``sep_penalty``
    on the returned Solution when any feature is sep-active."""
    from pymarxan.solvers.utils import build_solution

    p = _make_separation_problem()
    selected = np.ones(4, dtype=bool)
    sol = build_solution(p, selected, blm=0.0)
    assert sol.sep_shortfalls is not None
    assert sol.sep_penalty is not None
    assert 1 in sol.sep_shortfalls           # feature 1 is sep-active
    assert 2 not in sol.sep_shortfalls       # feature 2 is sep-inactive


def test_build_solution_leaves_sep_fields_none_for_legacy():
    """Round-3 H12 anti-test: a problem with no sep-active features must
    leave ``sep_shortfalls=None`` and ``sep_penalty=None``."""
    from pymarxan.solvers.utils import build_solution

    # Same problem, but mark every feature as sep-disabled (sepnum=1).
    p = _make_separation_problem()
    p.features.loc[:, "sepnum"] = 1
    selected = np.ones(4, dtype=bool)
    sol = build_solution(p, selected, blm=0.0)
    assert sol.sep_shortfalls is None
    assert sol.sep_penalty is None


def test_build_solution_handles_missing_coords_gracefully():
    """Round-3 M8: a heuristic-only user with sepnum>1 but no coords gets
    a warning and ``sep_shortfalls=None`` — not a crash."""
    import warnings

    from pymarxan.solvers.utils import build_solution

    pu = pd.DataFrame({"id": [1, 2], "cost": [1.0, 1.0], "status": [0, 0]})
    features = pd.DataFrame({
        "id": [1], "name": ["a"], "target": [1.0], "spf": [1.0],
        "sepdistance": [10.0], "sepnum": [3],
    })
    puvspr = pd.DataFrame({"species": [1, 1], "pu": [1, 2], "amount": [1.0, 1.0]})
    p = ConservationProblem(
        planning_units=pu, features=features, pu_vs_features=puvspr,
    )
    selected = np.array([True, True])
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always", UserWarning)
        sol = build_solution(p, selected, blm=0.0)
    assert sol.sep_shortfalls is None
    assert sol.sep_penalty is None
    assert any("Separation" in str(w.message) for w in recorded)
