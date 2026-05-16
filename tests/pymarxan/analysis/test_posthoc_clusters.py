"""Phase 24C: post-hoc cluster diagnostics on a Solution.

``compute_solution_clusters`` partitions the selected PUs of a Solution
into connected components using the problem's boundary graph. Returns
fragmentation diagnostics: cluster count, sizes, max cluster fraction.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pymarxan.analysis.posthoc_clusters import compute_solution_clusters
from pymarxan.models.problem import ConservationProblem
from pymarxan.solvers.base import Solution


def _linear_boundary_problem(n_pu: int = 6):
    """PUs arranged in a line: 1—2—3—4—5—6. Each consecutive pair shares
    a boundary edge. Easy to verify cluster structure by selecting any
    contiguous / disjoint subset."""
    pu = pd.DataFrame({
        "id": list(range(1, n_pu + 1)),
        "cost": [1.0] * n_pu,
        "status": [0] * n_pu,
    })
    features = pd.DataFrame({
        "id": [1], "name": ["a"], "target": [1.0], "spf": [1.0],
    })
    puvspr = pd.DataFrame({
        "species": [1], "pu": [1], "amount": [1.0],
    })
    boundary_rows = [
        {"id1": i, "id2": i + 1, "boundary": 1.0}
        for i in range(1, n_pu)
    ]
    boundary = pd.DataFrame(boundary_rows)
    return ConservationProblem(
        planning_units=pu, features=features, pu_vs_features=puvspr,
        boundary=boundary,
    )


def _sol(selected: list[bool]) -> Solution:
    return Solution(
        selected=np.array(selected, dtype=bool),
        cost=0.0, boundary=0.0, objective=0.0,
        targets_met={1: True},
    )


def test_single_cluster_when_selection_is_contiguous():
    """PUs 1, 2, 3 selected → one cluster of size 3."""
    p = _linear_boundary_problem()
    sol = _sol([True, True, True, False, False, False])
    result = compute_solution_clusters(p, sol)
    assert result["n_clusters"] == 1
    assert result["cluster_sizes"] == [3]


def test_multiple_clusters_when_selection_is_fragmented():
    """PUs 1, 3, 5 selected (non-adjacent) → three singleton clusters."""
    p = _linear_boundary_problem()
    sol = _sol([True, False, True, False, True, False])
    result = compute_solution_clusters(p, sol)
    assert result["n_clusters"] == 3
    assert sorted(result["cluster_sizes"]) == [1, 1, 1]


def test_two_clusters_with_gap():
    """PUs 1, 2 + 5, 6 selected → two clusters of size 2 each."""
    p = _linear_boundary_problem()
    sol = _sol([True, True, False, False, True, True])
    result = compute_solution_clusters(p, sol)
    assert result["n_clusters"] == 2
    assert sorted(result["cluster_sizes"]) == [2, 2]


def test_max_cluster_fraction_reports_largest_share():
    """The biggest cluster's share of the selection is exposed for
    quick "is this a fragmented reserve?" checks."""
    p = _linear_boundary_problem()
    # Cluster sizes: 3 (PUs 1-3) and 1 (PU 5). Max share = 3/4 = 0.75.
    sol = _sol([True, True, True, False, True, False])
    result = compute_solution_clusters(p, sol)
    assert result["n_clusters"] == 2
    assert result["max_cluster_fraction"] == pytest.approx(0.75, abs=1e-9)


def test_empty_selection_returns_zero_clusters():
    """No PUs selected → zero clusters and an empty sizes list."""
    p = _linear_boundary_problem()
    sol = _sol([False] * 6)
    result = compute_solution_clusters(p, sol)
    assert result["n_clusters"] == 0
    assert result["cluster_sizes"] == []
    assert result["max_cluster_fraction"] == 0.0


def test_no_boundary_data_each_selected_pu_is_own_cluster():
    """A problem without boundary data has no edges → every selected PU
    is its own cluster (no adjacency, no merging)."""
    p = _linear_boundary_problem()
    p.boundary = None
    sol = _sol([True, True, True, False, False, False])
    result = compute_solution_clusters(p, sol)
    assert result["n_clusters"] == 3
    assert result["cluster_sizes"] == [1, 1, 1]


def test_pu_to_cluster_mapping_returned():
    """A ``pu_to_cluster`` dict lets users colour-code reserve maps."""
    p = _linear_boundary_problem()
    sol = _sol([True, True, False, False, True, True])
    result = compute_solution_clusters(p, sol)
    mapping = result["pu_to_cluster"]
    # PUs 1, 2 share a cluster id; PUs 5, 6 share a different id;
    # PUs 3, 4 are unselected → not in mapping.
    assert mapping[1] == mapping[2]
    assert mapping[5] == mapping[6]
    assert mapping[1] != mapping[5]
    assert 3 not in mapping
    assert 4 not in mapping
