"""Cost of zone_target_lookup relative to one compute_zone_objective call (#7).

Comparative, not absolute: the lookup runs inside every objective call (heuristic / II
evaluate one per candidate move), so it must stay a small fraction of that call. A pandas
filter + drop_duplicates normalizer measured ~95 % of the call (design review #2); the dict
lookup measured ~4 % in the plan dry run.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import pytest

from pymarxan.zones.objective import compute_zone_objective
from tests.benchmarks.conftest import make_zone_problem

pytestmark = pytest.mark.bench


def _best_of(fn, repeats: int = 5, inner: int = 50) -> float:
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        for _ in range(inner):
            fn()
        best = min(best, time.perf_counter() - t0)
    return best / inner


def test_lookup_is_a_small_fraction_of_the_objective() -> None:
    p = make_zone_problem(n_pu=200, n_feat=20, n_zones=3, density=0.3, seed=7)
    zt = p.zone_targets
    assert zt is not None and len(zt) > 0
    # Duplicates and an unknown id so the lookup does its full work.
    extra = pd.concat(
        [zt, zt.head(5), pd.DataFrame({"zone": [99], "feature": [1], "target": [1.0]})],
        ignore_index=True,
    )
    p = p.copy_with(zone_targets=extra)
    a = np.random.default_rng(0).integers(0, 4, size=200)
    lookup = _best_of(p.zone_target_lookup)
    objective = _best_of(lambda: compute_zone_objective(p, a, 1.0))
    assert lookup <= 0.15 * objective, (
        f"lookup {lookup * 1e6:.1f} µs vs objective {objective * 1e6:.1f} µs"
    )
