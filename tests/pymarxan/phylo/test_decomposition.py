"""Tests for the branch-as-feature decomposition (PD as an objective)."""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import pytest

from pymarxan.models.problem import ConservationProblem
from pymarxan.phylo.decomposition import phylogenetic_branch_problem
from pymarxan.phylo.diversity import compute_phylogenetic_diversity
from pymarxan.phylo.tree import PhylogeneticTree
from pymarxan.solvers.base import Solution, SolverConfig
from pymarxan.solvers.mip_solver import MIPSolver

TREE = PhylogeneticTree.from_newick("((A:1,B:1):2,C:3);")


def _problem(cost=(1.0, 1.0, 1.0)) -> ConservationProblem:
    planning_units = pd.DataFrame(
        {"id": [1, 2, 3], "cost": list(cost), "status": [0, 0, 0]}
    )
    features = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "name": ["A", "B", "C"],
            "target": [1.0, 1.0, 1.0],
            "spf": [1.0, 1.0, 1.0],
        }
    )
    pu_vs_features = pd.DataFrame(
        {"species": [1, 2, 3], "pu": [1, 2, 3], "amount": [1.0, 1.0, 1.0]}
    )
    return ConservationProblem(planning_units, features, pu_vs_features)


def _solution_from_pus(problem, selected_pu_ids) -> Solution:
    sel = np.array(
        [pid in selected_pu_ids for pid in problem.planning_units["id"]], dtype=bool
    )
    return Solution(selected=sel, cost=0.0, boundary=0.0, objective=0.0, targets_met={})


def test_branch_features_carry_length_as_spf():
    bp = phylogenetic_branch_problem(_problem(), TREE)
    # 4 branches, all representable (A, B, C each occur; internal via A/B).
    assert len(bp.features) == 4
    spf_by_name = dict(zip(bp.features["name"], bp.features["spf"]))
    assert spf_by_name["branch:A"] == pytest.approx(1.0)
    assert spf_by_name["branch:C"] == pytest.approx(3.0)
    # amounts are presence (1.0)
    assert set(bp.pu_vs_features["amount"].unique()) == {1.0}
    # branch feature ids are unique
    assert bp.features["id"].is_unique


def test_unrepresentable_branch_is_dropped_and_min_set_feasible():
    # Remove C from the data → C's pendant branch (length 3) is unrepresentable.
    p = _problem()
    p = p.copy_with(
        pu_vs_features=p.pu_vs_features[p.pu_vs_features["species"] != 3].reset_index(
            drop=True
        )
    )
    bp = phylogenetic_branch_problem(p, TREE)
    names = set(bp.features["name"])
    assert "branch:C" not in names  # dropped
    assert "branch:A" in names and "branch:B" in names
    # min_set stays feasible: cheapest reserve capturing the retained branches.
    sols = MIPSolver(objective="min_set").solve(bp, SolverConfig(num_solutions=1))
    assert len(sols) == 1
    assert sols[0].all_targets_met


def test_min_set_captures_full_representable_pd_and_matches_brute_force():
    p = _problem(cost=(1.0, 1.0, 5.0))  # C's PU is expensive
    bp = phylogenetic_branch_problem(p, TREE)
    sols = MIPSolver(objective="min_set").solve(bp, SolverConfig(num_solutions=1))
    sol = sols[0]
    # realized PD is 100% of representable
    pd_res = compute_phylogenetic_diversity(p, sol, TREE)
    assert pd_res.fraction_pd_representable == pytest.approx(1.0)
    # brute-force min-cost reserve covering every branch
    pu_ids = list(p.planning_units["id"])
    costs = dict(zip(p.planning_units["id"], p.planning_units["cost"]))
    best = None
    for r in range(1, len(pu_ids) + 1):
        for combo in itertools.combinations(pu_ids, r):
            s = _solution_from_pus(p, set(combo))
            if compute_phylogenetic_diversity(p, s, TREE).fraction_pd_representable == 1.0:
                c = sum(costs[i] for i in combo)
                if best is None or c < best:
                    best = c
    assert sol.cost == pytest.approx(best)


def test_max_weighted_features_under_budget_maximizes_pd_vs_brute_force():
    p = _problem()
    bp = phylogenetic_branch_problem(p, TREE)
    bp = bp.copy_with(parameters={**bp.parameters, "COSTBUDGET": 2.0})
    sol = MIPSolver(objective="max_weighted_features").solve(
        bp, SolverConfig(num_solutions=1)
    )[0]
    got_pd = compute_phylogenetic_diversity(p, sol, TREE).pd_represented
    # brute-force max PD achievable for total cost <= 2.0
    pu_ids = list(p.planning_units["id"])
    costs = dict(zip(p.planning_units["id"], p.planning_units["cost"]))
    best_pd = 0.0
    for r in range(0, len(pu_ids) + 1):
        for combo in itertools.combinations(pu_ids, r):
            if sum(costs[i] for i in combo) <= 2.0:
                s = _solution_from_pus(p, set(combo))
                best_pd = max(best_pd, compute_phylogenetic_diversity(p, s, TREE).pd_represented)
    assert got_pd == pytest.approx(best_pd)


def test_all_unrepresentable_yields_empty_feature_set():
    # A tree whose single tip matches no feature → no branches retained.
    tree = PhylogeneticTree.from_newick("(Z:1);")
    bp = phylogenetic_branch_problem(_problem(), tree)
    assert len(bp.features) == 0
    assert len(bp.pu_vs_features) == 0
