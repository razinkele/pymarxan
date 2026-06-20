"""Barrier-removal optimizers (Phase B): greedy + simulated annealing.

Both maximise the chosen DCI form over binary remove/keep decisions, honour the
budget and the locked-in/locked-out status, and return a ``BarrierSolution``
with ``optimal=False`` (the exact MIP — ``optimal=True`` — lands in Phase C).

The SA loop reuses ``solvers.cooling.CoolingSchedule`` and a flip-and-evaluate
move over the free barriers, with the budget enforced by hard rejection.
"""
from __future__ import annotations

import math

import numpy as np

from pymarxan.rivers.barriers import BarrierProblem, BarrierSolution
from pymarxan.rivers.dci import dci_diadromous, dci_potamodromous
from pymarxan.solvers.cooling import CoolingSchedule

_TOL = 1e-9


def _score(problem: BarrierProblem, removed: set[int]) -> float:
    """DCI of the network with every barrier in ``removed`` made passable."""
    override = {b: 1.0 for b in removed}
    net = problem.network
    if problem.form == "diadromous":
        return dci_diadromous(net, override)
    return dci_potamodromous(net, override)


def _solution(
    problem: BarrierProblem,
    removed: set[int],
    cost_by_id: dict[int, float],
    baseline: float,
    *,
    optimal: bool,
) -> BarrierSolution:
    after = _score(problem, removed)
    cost = sum(cost_by_id[b] for b in removed)
    return BarrierSolution(
        removed=set(removed),
        cost=float(cost),
        dci_before=baseline,
        dci_after=after,
        gain=after - baseline,
        optimal=optimal,
    )


def optimize_barriers_greedy(problem: BarrierProblem) -> BarrierSolution:
    """Greedily add the barrier with the best DCI-gain-per-cost until the budget
    is exhausted or no positive-gain barrier remains. Locked-in barriers are
    pre-included; locked-out are never selected."""
    free, forced, cost = problem._meta()
    baseline = _score(problem, set())

    removed = set(forced)
    cur = _score(problem, removed)
    cur_cost = sum(cost[b] for b in removed)
    remaining = list(free)

    while remaining:
        best_b: int | None = None
        best_ratio = 0.0
        best_score = cur
        best_cost = cur_cost
        for b in remaining:
            new_cost = cur_cost + cost[b]
            if problem.budget is not None and new_cost > problem.budget + _TOL:
                continue
            score = _score(problem, removed | {b})
            gain = score - cur
            if gain <= _TOL:
                continue
            ratio = gain / cost[b] if cost[b] > 0 else math.inf
            if best_b is None or ratio > best_ratio:
                best_b, best_ratio, best_score, best_cost = b, ratio, score, new_cost
        if best_b is None:
            break
        removed.add(best_b)
        remaining.remove(best_b)
        cur = best_score
        cur_cost = best_cost

    return _solution(problem, removed, cost, baseline, optimal=False)


def optimize_barriers_sa(
    problem: BarrierProblem,
    *,
    seed: int = 0,
    num_steps: int = 2000,
    initial_temp: float = 2.0,
    final_temp: float = 0.01,
) -> BarrierSolution:
    """Simulated annealing over the free barriers' remove/keep bits. Budget is
    enforced by hard rejection; locked-in are always removed, locked-out never.
    Deterministic for a given ``seed``."""
    free, forced, cost = problem._meta()
    baseline = _score(problem, set())
    forced_set = set(forced)
    forced_cost = sum(cost[b] for b in forced)
    rng = np.random.default_rng(seed)

    def feasible(chosen: set[int]) -> bool:
        if problem.budget is None:
            return True
        total = forced_cost + sum(cost[b] for b in chosen)
        return total <= problem.budget + _TOL

    def score(chosen: set[int]) -> float:
        return _score(problem, forced_set | chosen)

    current: set[int] = set()
    cur = score(current)
    best_chosen = set(current)
    best_score = cur

    if free:
        schedule = CoolingSchedule.geometric(initial_temp, final_temp, num_steps)
        for step in range(num_steps):
            b = free[int(rng.integers(len(free)))]
            cand = current ^ {b}
            if not feasible(cand):
                continue
            s = score(cand)
            delta = s - cur
            if delta >= 0:
                accept = True
            else:
                t = max(schedule.temperature(step), 1e-12)
                accept = rng.random() < math.exp(delta / t)
            if accept:
                current = cand
                cur = s
                if s > best_score:
                    best_score = s
                    best_chosen = set(cand)

    removed = forced_set | best_chosen
    return _solution(problem, removed, cost, baseline, optimal=False)
