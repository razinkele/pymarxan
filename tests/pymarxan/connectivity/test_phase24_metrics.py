"""Phase 24A: PageRank, donor, recipient metrics + connectivity_to_boundary.

Extends the existing in/out-degree + betweenness + eigenvector centrality
suite from `pymarxan.connectivity.metrics`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# --- PageRank ----------------------------------------------------------


def test_pagerank_returns_array_of_correct_length():
    from pymarxan.connectivity.metrics import compute_pagerank_centrality
    matrix = np.array([
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
    ])
    scores = compute_pagerank_centrality(matrix)
    assert scores.shape == (3,)


def test_pagerank_scores_sum_to_one():
    """networkx PageRank is normalised; the sum across nodes equals 1."""
    from pymarxan.connectivity.metrics import compute_pagerank_centrality
    matrix = np.array([
        [0.0, 1.0, 0.5],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
    ])
    scores = compute_pagerank_centrality(matrix)
    assert scores.sum() == pytest.approx(1.0, abs=1e-9)


def test_pagerank_empty_graph_returns_uniform():
    """No edges → every node gets 1/n. networkx's default behaviour."""
    from pymarxan.connectivity.metrics import compute_pagerank_centrality
    matrix = np.zeros((4, 4))
    scores = compute_pagerank_centrality(matrix)
    np.testing.assert_allclose(scores, [0.25] * 4, atol=1e-9)


def test_pagerank_central_node_scores_higher():
    """Hub-and-spoke: node 0 receives from all others → highest PageRank."""
    from pymarxan.connectivity.metrics import compute_pagerank_centrality
    matrix = np.array([
        [0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
    ])
    scores = compute_pagerank_centrality(matrix)
    assert scores[0] == max(scores)


# --- Donor + recipient (thresholded in/out-degree) ---------------------


def test_compute_donors_thresholds_out_degree():
    """A donor is a node whose out-degree exceeds the threshold —
    sends more flow than it receives + threshold."""
    from pymarxan.connectivity.metrics import compute_donors
    # Node 0 sends 5+5=10 out, receives 0 in → net +10 (donor).
    # Node 1 receives 5, sends 0 → net -5 (not donor).
    # Node 2 receives 5, sends 0 → net -5 (not donor).
    matrix = np.array([
        [0.0, 5.0, 5.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
    ])
    donors = compute_donors(matrix, threshold=1.0)
    assert donors.dtype == bool
    assert donors[0]
    assert not donors[1]
    assert not donors[2]


def test_compute_recipients_thresholds_in_degree():
    """A recipient is a node whose in-degree exceeds out-degree by
    more than threshold."""
    from pymarxan.connectivity.metrics import compute_recipients
    matrix = np.array([
        [0.0, 5.0, 5.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
    ])
    recipients = compute_recipients(matrix, threshold=1.0)
    assert recipients.dtype == bool
    assert not recipients[0]
    assert recipients[1]
    assert recipients[2]


def test_donor_recipient_complement_at_zero_threshold():
    """With threshold=0 and no equal-flow nodes, donors ∪ recipients
    covers all nodes with any flow."""
    from pymarxan.connectivity.metrics import (
        compute_donors,
        compute_recipients,
    )
    matrix = np.array([
        [0.0, 3.0, 0.0],
        [0.0, 0.0, 7.0],
        [0.0, 0.0, 0.0],
    ])
    donors = compute_donors(matrix, threshold=0.0)
    recipients = compute_recipients(matrix, threshold=0.0)
    # No overlap (strict inequality on either side).
    assert not (donors & recipients).any()


# --- connectivity_to_boundary ------------------------------------------


def test_connectivity_to_boundary_round_trips_to_marxan_format():
    """A connectivity edge list should be convertible to a DataFrame
    with the columns Marxan's bound.dat expects (id1, id2, boundary)."""
    from pymarxan.connectivity.io import connectivity_to_boundary
    conn = pd.DataFrame({
        "id1": [1, 1, 2],
        "id2": [2, 3, 3],
        "value": [10.0, 5.0, 7.0],
    })
    boundary = connectivity_to_boundary(conn)
    assert list(boundary.columns) == ["id1", "id2", "boundary"]
    assert len(boundary) == 3


def test_connectivity_to_boundary_preserves_values():
    """Values map through unchanged (no scaling by default)."""
    from pymarxan.connectivity.io import connectivity_to_boundary
    conn = pd.DataFrame({
        "id1": [1, 2],
        "id2": [2, 3],
        "value": [10.0, 7.0],
    })
    boundary = connectivity_to_boundary(conn)
    np.testing.assert_array_almost_equal(
        boundary["boundary"].values, [10.0, 7.0],
    )


def test_connectivity_to_boundary_accepts_scale_factor():
    """Optional scale lets users convert connectivity units to boundary
    units (e.g. negate so high connectivity = low boundary penalty)."""
    from pymarxan.connectivity.io import connectivity_to_boundary
    conn = pd.DataFrame({
        "id1": [1], "id2": [2], "value": [10.0],
    })
    boundary = connectivity_to_boundary(conn, scale=-1.0)
    assert boundary["boundary"].iloc[0] == -10.0


def test_connectivity_to_boundary_drops_zero_edges():
    """Edges with value=0 are dropped — bound.dat conventions don't
    include null boundaries."""
    from pymarxan.connectivity.io import connectivity_to_boundary
    conn = pd.DataFrame({
        "id1": [1, 2],
        "id2": [2, 3],
        "value": [10.0, 0.0],
    })
    boundary = connectivity_to_boundary(conn)
    assert len(boundary) == 1
    assert boundary["id1"].iloc[0] == 1
