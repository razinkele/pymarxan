"""Per-flip cost of the overall-target term (spec §6).

Comparative, not absolute: under default contributions the contrib_differs gate must make
every zone-to-zone move skip the overall term, so populated overall targets may cost at most
5 % more per flip than zeroed ones. Absolute budgets are machine-relative and live in
bench_zone_sa.py. The zone-target weight multiply (``zone_target_weight`` in
``_penalty_delta``) is unmeasured by design: n_feat-length, paid in both arms.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import pytest

from pymarxan.zones.cache import ZoneProblemCache
from tests.benchmarks.conftest import make_zone_problem

pytestmark = pytest.mark.bench

N_PU, N_FEAT, N_ZONES, N_MOVES = 1000, 50, 3, 20_000


def _per_flip_seconds(problem, moves, assignment) -> float:
    cache = ZoneProblemCache.from_zone_problem(problem)
    held = cache.compute_held(assignment)
    best = float("inf")
    for _ in range(3):
        t0 = time.perf_counter()
        for idx, new_zone in moves:
            cache.compute_delta_zone_objective(
                idx, int(assignment[idx]), new_zone, assignment, held=held, blm=1.0,
            )
        best = min(best, time.perf_counter() - t0)
    return best / len(moves)


def test_overall_term_is_free_under_default_contributions() -> None:
    populated = make_zone_problem(
        n_pu=N_PU, n_feat=N_FEAT, n_zones=N_ZONES, density=0.3, seed=42,
    )
    assert populated.zone_contributions is None
    assert (populated.features["target"] > 0).all()
    zeroed = populated.copy_with(features=populated.features.assign(target=0.0))

    rng = np.random.default_rng(0)
    assignment = rng.integers(1, N_ZONES + 1, size=N_PU)
    moves = [
        (int(rng.integers(N_PU)), int(rng.integers(1, N_ZONES + 1))) for _ in range(N_MOVES)
    ]

    with_targets = _per_flip_seconds(populated, moves, assignment.copy())
    without = _per_flip_seconds(zeroed, moves, assignment.copy())
    ratio = with_targets / without
    assert ratio <= 1.05, (
        f"overall term not gated: {with_targets * 1e6:.2f} µs vs {without * 1e6:.2f} µs "
        f"per flip (ratio {ratio:.3f})"
    )


def test_overall_term_costs_bounded_when_contributions_differ() -> None:
    """Informational upper bound: with a contribution table every move pays the row op."""
    base = make_zone_problem(n_pu=N_PU, n_feat=N_FEAT, n_zones=N_ZONES, density=0.3, seed=42)
    rng = np.random.default_rng(1)
    table = [
        {"feature": f, "zone": z, "contribution": float(rng.uniform(0.2, 1.0))}
        for f in range(1, N_FEAT + 1)
        for z in range(1, N_ZONES + 1)
    ]
    weighted = base.copy_with(zone_contributions=pd.DataFrame(table))
    assignment = rng.integers(1, N_ZONES + 1, size=N_PU)
    moves = [
        (int(rng.integers(N_PU)), int(rng.integers(1, N_ZONES + 1))) for _ in range(N_MOVES)
    ]
    plain = _per_flip_seconds(base, moves, assignment.copy())
    paid = _per_flip_seconds(weighted, moves, assignment.copy())
    # Review M1 measured +109 % ungated; the dense row op should stay well under 3×.
    assert paid / plain <= 3.0, f"ratio {paid / plain:.2f}"
