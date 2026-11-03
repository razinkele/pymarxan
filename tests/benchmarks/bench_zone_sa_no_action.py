"""Per-flip cost of moves to and from "no action" (zone 0) — v0.49.0 R1.

Zone SA proposes zone 0 since v0.49.0. Such moves change the selection, so they pay the
standard-boundary delta; it must stay O(degree), not O(n_pu) (an ``assignment > 0`` mask
per call would be). Comparative, so machine speed cancels out.
"""
from __future__ import annotations

import statistics
import time

import numpy as np
import pytest

from pymarxan.zones.cache import ZoneProblemCache
from pymarxan.zones.model import ZonalProblem
from tests.benchmarks.conftest import make_zone_problem

pytestmark = pytest.mark.bench
N_MOVES = 20_000


def _per_delta(problem: ZonalProblem, *, to_zero: bool) -> float:
    cache = ZoneProblemCache.from_zone_problem(problem)
    rng = np.random.default_rng(0)
    assignment = rng.integers(1, cache.n_zones + 1, size=cache.n_pu)
    held = cache.compute_held(assignment)
    idx = rng.integers(cache.n_pu, size=N_MOVES)
    samples = []
    for _ in range(5):
        t0 = time.perf_counter()
        for i in idx:
            old = int(assignment[i])
            new = 0 if to_zero else (old % cache.n_zones) + 1
            cache.compute_delta_zone_objective(int(i), old, new, assignment, held=held, blm=1.0)
        samples.append((time.perf_counter() - t0) / N_MOVES)
    return statistics.median(samples)


def test_no_action_delta_does_not_scale_with_n_pu() -> None:
    small = make_zone_problem(n_pu=1_000, n_feat=10, n_zones=3, density=0.1, seed=1)
    large = make_zone_problem(n_pu=20_000, n_feat=10, n_zones=3, density=0.1, seed=1)
    ratio = _per_delta(large, to_zero=True) / _per_delta(small, to_zero=True)
    print(f"to-zero per-delta 20k/1k {ratio:.3f}")
    assert ratio <= 1.5, f"to-zero per-delta grows with n_pu: 20k/1k ratio {ratio:.3f}"


def test_no_action_delta_close_to_zone_swap_delta() -> None:
    p = make_zone_problem(n_pu=1_000, n_feat=50, n_zones=3, density=0.3, seed=42)
    ratio = _per_delta(p, to_zero=True) / _per_delta(p, to_zero=False)
    print(f"to-zero / zone-swap per-delta {ratio:.3f}")
    assert ratio <= 1.5, f"to-zero / zone-swap per-delta {ratio:.3f}"
