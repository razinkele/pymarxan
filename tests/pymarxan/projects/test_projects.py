"""Project prioritization (oppr-equivalent): model + exact MIP + greedy.

The problem (Joseph et al. 2009; Hanson et al. 2019): fund a set of *actions*
(the things with cost) under a budget; a *project* is completed iff all its
required actions are funded; each completed project secures a set of *features*
with some persistence probability. Maximise the weighted expected persistence
Σ_f w_f · max_{funded p benefiting f} persistence_{f,p}. Shared actions across
projects make simple ranking suboptimal — the exact MIP must beat greedy.
"""
from __future__ import annotations

import itertools

import pandas as pd
import pytest

from pymarxan.projects import (
    ProjectProblem,
    ProjectSolution,
    evaluate_projects,
    prioritize_projects_greedy,
    prioritize_projects_mip,
)


def _knapsack_trap(budget: float = 4.0) -> ProjectProblem:
    """A budgeted instance where ratio-greedy is provably suboptimal.

    Three features, each secured (persistence 1.0) by one single-action
    project. Weights/costs: A=(w3.1,c3), B=(w2,c2), C=(w2,c2). Budget 4.
    Greedy by benefit/cost funds A first (ratio 1.033) then can't afford B or
    C → 3.1. Optimal funds B+C → 4.0.
    """
    features = pd.DataFrame({"id": [1, 2, 3], "weight": [3.1, 2.0, 2.0]})
    actions = pd.DataFrame({"id": [1, 2, 3], "cost": [3.0, 2.0, 2.0]})
    projects = pd.DataFrame({"id": [1, 2, 3]})
    project_actions = pd.DataFrame({"project": [1, 2, 3], "action": [1, 2, 3]})
    project_features = pd.DataFrame(
        {"project": [1, 2, 3], "feature": [1, 2, 3], "persistence": [1.0, 1.0, 1.0]}
    )
    return ProjectProblem(
        features=features,
        actions=actions,
        projects=projects,
        project_actions=project_actions,
        project_features=project_features,
        budget=budget,
    )


def _shared_action() -> ProjectProblem:
    """Two projects share action a1 (complementarity), plus a free baseline."""
    features = pd.DataFrame({"id": [1, 2], "weight": [1.0, 1.0]})
    actions = pd.DataFrame({"id": [1, 2, 3], "cost": [1.0, 1.0, 1.0]})
    projects = pd.DataFrame({"id": [0, 1, 2, 3]})
    # p0 = free baseline (no actions); p1={a1}; p2={a1,a2}; p3={a3}
    project_actions = pd.DataFrame(
        {"project": [1, 2, 2, 3], "action": [1, 1, 2, 3]}
    )
    project_features = pd.DataFrame(
        {
            "project": [0, 0, 1, 2, 3],
            "feature": [1, 2, 1, 2, 2],
            "persistence": [0.1, 0.1, 0.9, 0.9, 0.5],
        }
    )
    return ProjectProblem(
        features=features,
        actions=actions,
        projects=projects,
        project_actions=project_actions,
        project_features=project_features,
        budget=2.0,
    )


def _brute_force(problem: ProjectProblem) -> float:
    """Best weighted benefit over all budget-feasible action subsets, scored
    with the production evaluator (only the search is independent)."""
    actions = list(problem.actions["id"].astype(int))
    cost = dict(zip(problem.actions["id"].astype(int), problem.actions["cost"].astype(float)))
    best = -1.0
    for r in range(len(actions) + 1):
        for combo in itertools.combinations(actions, r):
            c = sum(cost[a] for a in combo)
            if problem.budget is not None and c > problem.budget + 1e-9:
                continue
            best = max(best, evaluate_projects(problem, set(combo)).benefit)
    return best


# --- evaluator ---------------------------------------------------------


def test_evaluate_baseline_and_completion():
    p = _shared_action()
    # nothing funded → only the free baseline project p0 → 0.1 + 0.1
    base = evaluate_projects(p, set())
    assert base.benefit == pytest.approx(0.2)
    assert base.feature_persistence == pytest.approx({1: 0.1, 2: 0.1})
    assert base.cost == 0.0
    # fund a1 → p1 completes (f1 0.9); f2 still baseline 0.1
    s = evaluate_projects(p, {1})
    assert 1 in s.funded_projects
    assert s.feature_persistence[1] == pytest.approx(0.9)
    assert s.feature_persistence[2] == pytest.approx(0.1)
    assert s.benefit == pytest.approx(1.0)


# --- exact MIP ---------------------------------------------------------


def test_mip_matches_brute_force_knapsack():
    for budget in (0.0, 2.0, 3.0, 4.0, None):
        p = _knapsack_trap(budget if budget is not None else 99.0)
        if budget is None:
            p = ProjectProblem(p.features, p.actions, p.projects,
                               p.project_actions, p.project_features, budget=None)
        sol = prioritize_projects_mip(p)
        assert isinstance(sol, ProjectSolution)
        assert sol.optimal is True
        assert sol.benefit == pytest.approx(_brute_force(p), abs=1e-6)


def test_mip_matches_brute_force_shared_action():
    p = _shared_action()
    sol = prioritize_projects_mip(p)
    assert sol.benefit == pytest.approx(_brute_force(p), abs=1e-6)
    assert sol.benefit == pytest.approx(1.8)  # fund a1+a2 → both f1,f2 at 0.9
    assert sol.funded_actions == {1, 2}
    assert sol.cost <= 2.0 + 1e-9


def test_mip_respects_budget():
    p = _knapsack_trap(budget=0.0)
    sol = prioritize_projects_mip(p)
    assert sol.funded_actions == set()
    assert sol.benefit == pytest.approx(0.0)


# --- greedy ------------------------------------------------------------


def test_greedy_runs_and_never_beats_exact():
    for budget in (2.0, 3.0, 4.0):
        p = _knapsack_trap(budget=budget)
        g = prioritize_projects_greedy(p)
        opt = prioritize_projects_mip(p)
        assert g.optimal is False
        assert g.benefit <= opt.benefit + 1e-6
        assert g.cost <= budget + 1e-9


def test_greedy_is_suboptimal_on_the_trap():
    """The knapsack trap: greedy funds the high-ratio action A and stalls;
    the exact MIP funds B+C for a strictly better outcome."""
    p = _knapsack_trap(budget=4.0)
    g = prioritize_projects_greedy(p)
    opt = prioritize_projects_mip(p)
    assert opt.benefit == pytest.approx(4.0)
    assert g.benefit == pytest.approx(3.1)
    assert g.benefit < opt.benefit
