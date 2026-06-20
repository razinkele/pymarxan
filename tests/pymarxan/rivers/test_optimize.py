"""Phase B — barrier-decision model + greedy / SA optimizers."""
from __future__ import annotations

import itertools

import pandas as pd
import pytest

from pymarxan.models.problem import (
    STATUS_LOCKED_IN,
    STATUS_LOCKED_OUT,
    STATUS_NORMAL,
)
from pymarxan.rivers import (
    BarrierProblem,
    BarrierSolution,
    RiverNetwork,
    dci_diadromous,
    optimize_barriers_greedy,
    optimize_barriers_sa,
)


def _binary_chain(statuses=None, cost=None) -> RiverNetwork:
    """S1(outlet) <- S2 <- S3; impassable B1 on S2, B2 on S3 (p=0).

    Removing B1 reconnects S2 (and gates S3); removing B2 alone does nothing.
    """
    n = 2
    statuses = statuses or [STATUS_NORMAL] * n
    cost = cost or [1.0] * n
    return RiverNetwork(
        segments=pd.DataFrame(
            {"id": [1, 2, 3], "length": [10.0] * 3, "down_id": [-1, 1, 2]}
        ),
        barriers=pd.DataFrame(
            {
                "id": [1, 2],          # B1 on segment 2, B2 on segment 3
                "segment": [2, 3],
                "pass_up": [0.0, 0.0],
                "pass_down": [0.0, 0.0],
                "removal_cost": cost,
                "status": statuses,
            }
        ),
    )


def _brute_force(problem: BarrierProblem) -> float:
    """Best achievable DCI over all budget-feasible removable subsets,
    using the production DCI function to score (only the search is independent)."""
    net = problem.network
    bar = net.barriers
    forced = {int(r.id) for r in bar.itertuples() if int(r.status) == STATUS_LOCKED_IN}
    free = [int(r.id) for r in bar.itertuples() if int(r.status) == STATUS_NORMAL]
    cost = {int(r.id): float(r.removal_cost) for r in bar.itertuples()}
    best = -1.0
    for k in range(len(free) + 1):
        for combo in itertools.combinations(free, k):
            removed = forced | set(combo)
            total = sum(cost[b] for b in removed)
            if problem.budget is not None and total > problem.budget + 1e-9:
                continue
            score = dci_diadromous(net, {b: 1.0 for b in removed})
            best = max(best, score)
    return best


# --- model -------------------------------------------------------------


def test_solution_baseline_excludes_locked_in():
    """dci_before is the native network (no decision applied); locked-in
    removals count toward dci_after/gain, not the baseline."""
    net = _binary_chain(statuses=[STATUS_LOCKED_IN, STATUS_NORMAL])
    sol = optimize_barriers_greedy(BarrierProblem(net, budget=None))
    assert sol.dci_before == pytest.approx(100.0 / 3, abs=1e-4)   # native, B1 not applied
    assert 1 in sol.removed                                       # locked-in B1 forced
    assert sol.dci_after == pytest.approx(sol.dci_before + sol.gain, abs=1e-6)


# --- greedy ------------------------------------------------------------


def test_greedy_picks_gating_barrier_under_budget():
    sol = optimize_barriers_greedy(BarrierProblem(_binary_chain(), budget=1.0))
    assert isinstance(sol, BarrierSolution)
    assert sol.removed == {1}                       # B1 gates B2's reach
    assert sol.dci_after == pytest.approx(200.0 / 3, abs=1e-4)   # 66.667
    assert sol.cost <= 1.0
    assert sol.optimal is False


def test_greedy_respects_budget():
    sol = optimize_barriers_greedy(BarrierProblem(_binary_chain(), budget=0.0))
    assert sol.removed == set()                     # nothing affordable
    assert sol.dci_after == pytest.approx(sol.dci_before)


def test_greedy_never_removes_locked_out():
    net = _binary_chain(statuses=[STATUS_LOCKED_OUT, STATUS_NORMAL])
    sol = optimize_barriers_greedy(BarrierProblem(net, budget=10.0))
    assert 1 not in sol.removed
    # B1 locked out → S2/S3 stay disconnected → DCId stuck at 33.33
    assert sol.dci_after == pytest.approx(100.0 / 3, abs=1e-4)


def test_greedy_matches_brute_force_on_binary_chain():
    for budget in (0.0, 1.0, 2.0, None):
        problem = BarrierProblem(_binary_chain(), budget=budget)
        sol = optimize_barriers_greedy(problem)
        assert sol.dci_after == pytest.approx(_brute_force(problem), abs=1e-4)


# --- simulated annealing ----------------------------------------------


def test_sa_reaches_brute_force_optimum():
    problem = BarrierProblem(_binary_chain(), budget=1.0)
    sol = optimize_barriers_sa(problem, seed=0, num_steps=2000)
    assert sol.dci_after == pytest.approx(_brute_force(problem), abs=1e-4)
    assert sol.optimal is False


def test_sa_respects_budget_and_locks():
    net = _binary_chain(
        statuses=[STATUS_LOCKED_OUT, STATUS_NORMAL], cost=[1.0, 1.0]
    )
    sol = optimize_barriers_sa(BarrierProblem(net, budget=10.0), seed=1, num_steps=500)
    assert 1 not in sol.removed                      # locked-out never removed
    assert sol.cost <= 10.0


def test_sa_is_deterministic_with_seed():
    problem = BarrierProblem(_binary_chain(), budget=1.0)
    a = optimize_barriers_sa(problem, seed=42, num_steps=500)
    b = optimize_barriers_sa(problem, seed=42, num_steps=500)
    assert a.removed == b.removed


def test_heuristics_never_beat_brute_force():
    # over a few seeded budgets, neither heuristic exceeds the exact optimum
    for budget in (0.0, 1.0, 2.0):
        problem = BarrierProblem(_binary_chain(), budget=budget)
        opt = _brute_force(problem)
        g = optimize_barriers_greedy(problem)
        s = optimize_barriers_sa(problem, seed=7, num_steps=1000)
        assert g.dci_after <= opt + 1e-6
        assert s.dci_after <= opt + 1e-6
