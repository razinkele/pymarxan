"""Tests for Faith PD scoring of a solution."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pymarxan.models.problem import ConservationProblem
from pymarxan.phylo.diversity import PDResult, compute_phylogenetic_diversity
from pymarxan.phylo.tree import PhylogeneticTree
from pymarxan.solvers.base import Solution

TREE = PhylogeneticTree.from_newick("((A:1,B:1):2,C:3);")


def _problem() -> ConservationProblem:
    # 3 PUs; feature A only in PU1, B only in PU2, C only in PU3.
    planning_units = pd.DataFrame(
        {"id": [1, 2, 3], "cost": [1.0, 1.0, 1.0], "status": [0, 0, 0]}
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


def _solution(selected: list[bool]) -> Solution:
    return Solution(
        selected=np.array(selected, dtype=bool),
        cost=0.0,
        boundary=0.0,
        objective=0.0,
        targets_met={},
    )


def test_pd_of_A_and_C_reserve():
    # Select PU1 (A) and PU3 (C): PD = A:1 + internal:2 + C:3 = 6; B excluded.
    res = compute_phylogenetic_diversity(_problem(), _solution([True, False, True]), TREE)
    assert res.pd_represented == pytest.approx(6.0)
    assert res.pd_total == pytest.approx(7.0)
    assert res.pd_representable == pytest.approx(7.0)  # all tips occur somewhere
    assert res.fraction_pd_total == pytest.approx(6.0 / 7.0)
    assert res.fraction_pd_representable == pytest.approx(6.0 / 7.0)
    assert res.n_tips == 3
    assert res.n_tips_represented == 2


def test_empty_reserve_has_zero_pd():
    res = compute_phylogenetic_diversity(
        _problem(), _solution([False, False, False]), TREE
    )
    assert res.pd_represented == pytest.approx(0.0)
    assert res.fraction_pd_total == pytest.approx(0.0)
    assert res.fraction_pd_representable == pytest.approx(0.0)


def test_unrepresentable_tip_shrinks_representable_ceiling():
    # Drop C from the data entirely: C's tip can never be represented.
    p = _problem()
    p = p.copy_with(
        pu_vs_features=p.pu_vs_features[p.pu_vs_features["species"] != 3].reset_index(
            drop=True
        )
    )
    # Reserve {A, B}: PD = A:1 + B:1 + internal:2 = 4. Representable excludes
    # C's branch (3) → pd_representable = 4, so representable fraction == 1.0,
    # while total fraction = 4/7 < 1.
    res = compute_phylogenetic_diversity(p, _solution([True, True, False]), TREE)
    assert res.pd_represented == pytest.approx(4.0)
    assert res.pd_representable == pytest.approx(4.0)
    assert res.fraction_pd_representable == pytest.approx(1.0)
    assert res.fraction_pd_total == pytest.approx(4.0 / 7.0)


def test_explicit_tip_feature_map_matches_name_default():
    p = _problem()
    sol = _solution([True, False, True])
    default = compute_phylogenetic_diversity(p, sol, TREE)
    explicit = compute_phylogenetic_diversity(
        p, sol, TREE, tip_feature_map={"A": 1, "B": 2, "C": 3}
    )
    assert explicit.pd_represented == pytest.approx(default.pd_represented)


def test_to_dataframe_columns():
    res = compute_phylogenetic_diversity(_problem(), _solution([True, False, True]), TREE)
    df = res.to_dataframe()
    assert list(df.columns) == ["child_node", "length", "represented"]
    assert isinstance(res, PDResult)


def test_all_unresolved_tips_warns_and_counts():
    # A tree whose tips match no feature name/id → all unresolved, PD 0, warn.
    tree = PhylogeneticTree.from_newick("((X:1,Y:1):2,Z:3);")
    with pytest.warns(UserWarning, match="no tree tip matched"):
        res = compute_phylogenetic_diversity(
            _problem(), _solution([True, True, True]), tree
        )
    assert res.n_tips_unresolved == 3
    assert res.pd_represented == pytest.approx(0.0)
