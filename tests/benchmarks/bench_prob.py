"""PROBMODE 3 (2D chance term) SA inner-loop performance benchmarks.

v0.47.0 (spec §3.5): the chance term moved out of ``ProblemCache`` into the incremental
``ChanceState`` sidecar, which re-scores only the flipped PU's features. Before, every
flip re-summed dense (n_pu, n_feat) expected / variance matrices (~300× mode 0).

"Per flip" means the delta, plus the commit on an accepted flip, for mode 0 and mode 3
alike. Budgets are comparative (mode 3 ≤ 6× mode 0 on the same machine), except the
two small absolute smoke budgets kept from Phase 18.

Run via ``make bench`` / ``pytest tests/benchmarks/bench_prob.py -m bench``; not part
of ``make check``.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import pytest

from pymarxan.models.problem import ConservationProblem
from pymarxan.solvers.cache import ProblemCache
from pymarxan.solvers.chance_state import ChanceState

pytestmark = pytest.mark.bench

RATIO_BOUND = 6.0


def _make_probmode3_problem(
    n_pu: int = 500, n_features: int = 10, seed: int = 0, risk: str = "dense",
    cells_per_pu: int = 5,
) -> ConservationProblem:
    """Synthetic PROBMODE 3 problem (the spec review's layout: ~``cells_per_pu`` feature
    cells per PU).

    ``risk="dense"``: presence q = 1 − p with loss p ~ U(0, 0.3) on every cell.
    ``risk="sparse"``: 1 % of PUs risky (p = 0.05, q = 0.95); every other cell q = 1.
    """
    rng = np.random.default_rng(seed)
    pu_ids = np.arange(1, n_pu + 1)
    pu = pd.DataFrame({
        "id": pu_ids, "cost": rng.uniform(1.0, 10.0, size=n_pu),
        "status": np.zeros(n_pu, dtype=int),
    })
    k = n_pu * cells_per_pu
    pv = pd.DataFrame({
        "species": rng.integers(1, n_features + 1, k), "pu": rng.integers(1, n_pu + 1, k),
        "amount": rng.uniform(0.1, 3.0, k),
    }).drop_duplicates(["species", "pu"]).reset_index(drop=True)
    if risk == "dense":
        pv["prob"] = 1.0 - rng.uniform(0.0, 0.3, len(pv))
    else:
        risky = rng.choice(pu_ids, max(1, n_pu // 100), replace=False)
        pv["prob"] = np.where(pv["pu"].isin(risky), 0.95, 1.0)
    total = (pv["amount"] * pv["prob"]).groupby(pv["species"]).sum()
    total = total.reindex(range(1, n_features + 1), fill_value=0.0).to_numpy()
    features = pd.DataFrame({
        "id": np.arange(1, n_features + 1),
        "name": [f"p_f{i}" for i in range(1, n_features + 1)],
        "target": 0.4 * total, "spf": 1.0,
        "ptarget": rng.uniform(0.3, 0.95, n_features),
    })
    return ConservationProblem(
        planning_units=pu, features=features, pu_vs_features=pv,
        parameters={"PROBMODE": 3, "BLM": 0.0},
    )


def _per_flip_seconds(
    cache: ProblemCache, chance: bool, sel0: np.ndarray, idxs: np.ndarray,
    accept: np.ndarray,
) -> float:
    """Mean per-flip time: the cache delta (+ ChanceState delta), and on an accepted flip
    the held update (+ ChanceState commit). Best of 3 repeats."""
    best = float("inf")
    for _ in range(3):
        sel = sel0.copy()
        held = cache.compute_held(sel)
        total_cost = float(cache.costs[sel].sum())
        state = ChanceState.from_selection(cache, sel) if chance else None
        t0 = time.perf_counter()
        for i, acc in zip(idxs.tolist(), accept.tolist()):
            adding = not sel[i]
            cache.compute_delta_objective(i, sel, held, total_cost, 0.0)
            if state is not None:
                state.delta_penalty(cache, i, adding)
            if acc:
                sign = 1.0 if adding else -1.0
                sel[i] = adding
                cache.apply_flip_to_held(held, i, sign)
                total_cost += sign * cache.costs[i]
                if state is not None:
                    state.apply_flip(cache, i, adding)
        best = min(best, (time.perf_counter() - t0) / len(idxs))
    return best


@pytest.mark.parametrize("risk", ["dense", "sparse"])
@pytest.mark.parametrize("n_pu, n_feat", [(2000, 20), (10_000, 50)])
def test_probmode3_per_flip_within_6x_mode0(n_pu: int, n_feat: int, risk: str) -> None:
    """Per-flip cost of mode 3 (cache + ChanceState) ≤ 6× mode 0 (cache only)."""
    problem = _make_probmode3_problem(n_pu, n_feat, seed=0, risk=risk)
    cache3 = ProblemCache.from_problem(problem)
    assert cache3.chance_terms == {"2d"}
    cache0 = ProblemCache.from_problem(problem.copy_with(parameters={"PROBMODE": 0}))
    rng = np.random.default_rng(1)
    sel0 = rng.random(n_pu) < 0.5
    idxs = rng.integers(0, n_pu, 6000)
    accept = rng.random(6000) < 0.6
    t0 = _per_flip_seconds(cache0, False, sel0, idxs, accept)
    t3 = _per_flip_seconds(cache3, True, sel0, idxs, accept)
    ratio = t3 / t0
    assert ratio <= RATIO_BOUND, (
        f"PROBMODE 3 per flip {t3 * 1e6:.1f} µs = {ratio:.2f}× mode 0 "
        f"({t0 * 1e6:.1f} µs) at {n_pu}×{n_feat}, {risk} risk (bound {RATIO_BOUND}×)."
    )


def test_probmode3_delta_under_2500us_at_500pu():
    """Per-flip ``ChanceState`` cost (delta + commit) under PROBMODE 3 on a 500-PU ×
    10-feature problem: median < 2.5 ms. Re-pointed (v0.47.0) from the cache's dense
    delta, which no longer carries the chance term (it would pass measuring nothing)."""
    problem = _make_probmode3_problem(n_pu=500, n_features=10, seed=0)
    cache = ProblemCache.from_problem(problem)
    assert cache.chance_terms == {"2d"}, "fixture is meant to be PROBMODE 3"

    rng = np.random.default_rng(0)
    selected = rng.random(cache.n_pu) > 0.5
    state = ChanceState.from_selection(cache, selected)

    samples: list[float] = []
    for _ in range(200):
        idx = int(rng.integers(0, cache.n_pu))
        adding = not state.selected[idx]
        t0 = time.perf_counter()
        state.delta_penalty(cache, idx, adding)
        state.apply_flip(cache, idx, adding)
        samples.append(time.perf_counter() - t0)

    median_us = float(np.median(samples)) * 1e6
    assert median_us < 2500.0, (
        f"PROBMODE 3 ChanceState per-flip median: {median_us:.1f} µs (budget 2500 µs)."
    )


def test_probmode3_full_objective_under_5ms_at_500pu():
    """From-scratch chance term (``ChanceState.from_selection``) — budget 5 ms at 500 PU ×
    10 features. Re-pointed (v0.47.0) from ``compute_full_objective``, which no longer
    includes the chance term."""
    problem = _make_probmode3_problem(n_pu=500, n_features=10, seed=0)
    cache = ProblemCache.from_problem(problem)
    rng = np.random.default_rng(0)
    selected = rng.random(cache.n_pu) > 0.5

    t0 = time.perf_counter()
    ChanceState.from_selection(cache, selected).penalty_total()
    elapsed = time.perf_counter() - t0

    assert elapsed < 0.005, (
        f"PROBMODE 3 from-scratch chance term took {elapsed * 1000:.1f} ms (budget 5 ms)."
    )
