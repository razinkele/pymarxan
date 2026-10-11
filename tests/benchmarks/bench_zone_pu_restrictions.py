"""SA per-proposal cost with ~50 % of PUs puzone-restricted vs none (#6 spec §5)."""
from __future__ import annotations

import statistics
import time
from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest

from pymarxan.solvers.base import SolverConfig
from pymarxan.zones.cache import ZoneProblemCache
from pymarxan.zones.model import ZonalProblem, fixed_zones
from pymarxan.zones.solver import ZoneSASolver, _propose
from tests.benchmarks.conftest import make_zone_problem

pytestmark = pytest.mark.bench
N_PU, N_FEAT, N_ZONES, N_MOVES = 1000, 50, 3, 20_000


def _opts(problem: ZonalProblem) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    rules = problem.pu_zone_rules()
    zone_options = np.array(sorted(problem.zone_ids), dtype=int)
    every = frozenset(zone_options.tolist())
    opts = [zone_options if r.allowed == every else np.array(sorted(r.allowed), dtype=int)
            for r in rules]
    fixed = fixed_zones(rules)
    swappable = np.array([i for i in range(len(rules)) if i not in fixed], dtype=int)
    return opts, swappable, zone_options


def _per_evaluated_delta(problem: ZonalProblem) -> float:
    """Seconds per evaluated delta (no-op proposals excluded, so the restricted run's higher
    no-op share cannot hide proposal overhead)."""
    opts, swappable, _ = _opts(problem)
    cache = ZoneProblemCache.from_zone_problem(problem)
    assignment = np.array([o[0] for o in opts])
    held = cache.compute_held(assignment)
    samples = []
    for rep in range(5):
        rng = np.random.default_rng(rep)
        evaluated = 0
        t0 = time.perf_counter()
        for _ in range(N_MOVES):
            idx, new_zone = _propose(rng, swappable, opts)
            old_zone = int(assignment[idx])
            if new_zone != old_zone:
                cache.compute_delta_zone_objective(
                    idx, old_zone, new_zone, assignment, held=held, blm=1.0,
                )
                evaluated += 1
        samples.append((time.perf_counter() - t0) / evaluated)
    return statistics.median(samples)


def _propose_seconds(draw: Callable[[], object]) -> float:
    samples = []
    for _ in range(5):
        t0 = time.perf_counter()
        for _ in range(N_MOVES):
            draw()
        samples.append((time.perf_counter() - t0) / N_MOVES)
    return statistics.median(samples)


def _restricted(base: ZonalProblem) -> ZonalProblem:
    ids = base.planning_units["id"].astype(int).tolist()[::2]
    rows = [(pid, z) for pid in ids for z in (1, 2)]
    return base.copy_with(pu_zones=pd.DataFrame(rows, columns=["pu", "zone"]))


def test_restrictions_cost_at_most_ten_percent_per_delta() -> None:
    base = make_zone_problem(n_pu=N_PU, n_feat=N_FEAT, n_zones=N_ZONES, density=0.3, seed=42)
    ratio = _per_evaluated_delta(_restricted(base)) / _per_evaluated_delta(base)
    print(f"per-delta ratio {ratio:.3f}")
    assert ratio <= 1.10, f"restricted/unrestricted per-delta ratio {ratio:.3f}"


def test_propose_vs_v037_inline_draw() -> None:
    base = make_zone_problem(n_pu=N_PU, n_feat=N_FEAT, n_zones=N_ZONES, density=0.3, seed=42)
    # Time _propose on the RESTRICTED option arrays (half the PUs have 2 zones): the
    # unrestricted arrays never exercise the per-PU options path. _opts still returns the
    # full zone_options, so the v0.37 inline draw below is unchanged.
    opts, swappable, zone_options = _opts(_restricted(base))
    n_sw, n_z = len(swappable), len(zone_options)
    rng_a, rng_b = np.random.default_rng(0), np.random.default_rng(0)

    def v037() -> tuple[int, int]:
        idx = int(swappable[rng_a.integers(n_sw)])
        return idx, int(zone_options[rng_a.integers(n_z)])

    ratio = _propose_seconds(lambda: _propose(rng_b, swappable, opts)) / _propose_seconds(v037)
    print(f"_propose / v0.37 inline draw {ratio:.3f}")
    assert ratio <= 1.10, f"_propose / v0.37 inline draw {ratio:.3f}"


def _sa_seconds_per_iteration(problem: ZonalProblem) -> float:
    problem.parameters["NUMITNS"] = 50_000
    problem.parameters["NUMTEMP"] = 500
    samples = []
    for rep in range(5):
        t0 = time.perf_counter()
        ZoneSASolver().solve(problem, SolverConfig(num_solutions=1, seed=rep))
        samples.append((time.perf_counter() - t0) / 50_000)
    return statistics.median(samples)


def test_whole_sa_loop_restricted_vs_unrestricted() -> None:
    """End-to-end guard on the real SA loop (rule building, option arrays, temperature
    schedule, deltas). Restricted runs skip ~1/12 more deltas (2-zone PUs propose a no-op
    1/2 of the time vs 1/3), which lowers the ratio by ~0.1, so this catches gross loop
    overhead (e.g. per-iteration rule rebuilding); the two tests above bound the fine one."""
    base = make_zone_problem(n_pu=N_PU, n_feat=N_FEAT, n_zones=N_ZONES, density=0.3, seed=42)
    restricted = _restricted(base)
    restricted.parameters = dict(base.parameters)  # own copy: the helper mutates NUMITNS/NUMTEMP
    ratio = _sa_seconds_per_iteration(restricted) / _sa_seconds_per_iteration(base)
    print(f"whole-SA restricted/unrestricted per-iteration {ratio:.3f}")
    assert ratio <= 1.10, f"whole-SA restricted/unrestricted per-iteration {ratio:.3f}"
