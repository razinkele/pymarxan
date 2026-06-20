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
import pulp

from pymarxan.rivers.barriers import BarrierProblem, BarrierSolution
from pymarxan.rivers.dci import dci_diadromous, dci_potamodromous
from pymarxan.solvers._backends import _make_pulp_solver
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


def optimize_barriers_mip(
    problem: BarrierProblem,
    *,
    mip_backend: str = "auto",
    time_limit: int = 120,
    gap: float = 0.0,
    verbose: bool = False,
) -> BarrierSolution:
    """Exact barrier optimisation for the **binary-passability diadromous**
    case (design §7.3). A segment is connected to the sea iff every blocking
    barrier on its path to the mouth is removed; ``c_i = ∏ y_b`` linearises
    exactly. Returns ``optimal=True``.

    Requires every barrier to be binary-passable (native ``p ∈ {0, 1}``) — for
    partial passability use ``optimize_barriers_sa``. Potamodromous-exact is
    not implemented (the all-pairs product has no compact linearisation).
    """
    if problem.form != "diadromous":
        raise NotImplementedError(
            "exact MIP is implemented for the diadromous form only; "
            "use optimize_barriers_sa for potamodromous"
        )
    net = problem.network
    native = net.barrier_passabilities()
    if any(p not in (0.0, 1.0) for p in native.values()):
        raise ValueError(
            "optimize_barriers_mip requires binary passability (native p in "
            "{0, 1}); use optimize_barriers_sa for partial passability"
        )

    free, forced, cost = problem._meta()
    forced_set = set(forced)
    baseline = _score(problem, set())

    # Only barriers that actually block (native p == 0) gate connectivity.
    blocking = {bid for bid, p in native.items() if p == 0.0}

    model = pulp.LpProblem("rivers_barriers", pulp.LpMaximize)
    y = {bid: pulp.LpVariable(f"y_{bid}", cat="Binary") for bid in free}
    w = net.weights()
    seg_ids = list(w)
    c = {s: pulp.LpVariable(f"c_{s}", lowBound=0.0, upBound=1.0) for s in seg_ids}

    model += pulp.lpSum(w[s] * c[s] for s in seg_ids)

    for s in seg_ids:
        path_blocking = [b for b in net.path_barriers_to_mouth(s) if b in blocking]
        # Effective "open" indicator per blocking barrier on the path:
        #   forced (locked-in) -> always removed (constant 1, no constraint);
        #   free (normal)      -> decision variable y_b;
        #   otherwise (locked-out) -> never removed -> segment unreachable.
        var_terms = []
        unreachable = False
        for b in path_blocking:
            if b in forced_set:
                continue  # constant 1
            if b in y:
                var_terms.append(y[b])
            else:
                unreachable = True
                break
        if unreachable:
            model += c[s] <= 0
            continue
        for v in var_terms:
            model += c[s] <= v
        if var_terms:
            model += c[s] >= pulp.lpSum(var_terms) - (len(var_terms) - 1)
        else:
            model += c[s] >= 1  # no removable blocker on the path -> connected

    if problem.budget is not None:
        forced_cost = sum(cost[b] for b in forced_set)
        model += pulp.lpSum(cost[b] * y[b] for b in free) <= problem.budget - forced_cost

    solver = _make_pulp_solver(
        mip_backend, time_limit=time_limit, gap=gap, verbose=verbose
    )
    model.solve(solver)

    removed = set(forced_set)
    optimal = pulp.LpStatus[model.status] == "Optimal"
    if optimal:
        for bid, var in y.items():
            val = var.value()
            if val is not None and val > 0.5:
                removed.add(bid)
    return _solution(problem, removed, cost, baseline, optimal=optimal)
