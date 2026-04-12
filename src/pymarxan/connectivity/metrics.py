"""Connectivity metrics for conservation planning.

Computes graph-theoretic metrics from a connectivity (adjacency) matrix.
Compatible with Marxan Connect workflow.
"""
from __future__ import annotations

import numpy as np


def compute_in_degree(matrix: np.ndarray) -> np.ndarray:
    """Compute in-degree (incoming flow) for each node. Sum of each column (excluding diagonal)."""
    m = matrix.copy()
    np.fill_diagonal(m, 0)
    result: np.ndarray = m.sum(axis=0)
    return result


def compute_out_degree(matrix: np.ndarray) -> np.ndarray:
    """Compute out-degree (outgoing flow) for each node. Sum of each row (excluding diagonal)."""
    m = matrix.copy()
    np.fill_diagonal(m, 0)
    result: np.ndarray = m.sum(axis=1)
    return result


def compute_betweenness_centrality(matrix: np.ndarray) -> np.ndarray:
    """Compute betweenness centrality using networkx. Returns normalized [0, 1]."""
    import networkx as nx

    G = nx.from_numpy_array(matrix, create_using=nx.DiGraph)

    bc = nx.betweenness_centrality(G, weight="weight", normalized=True)
    n = matrix.shape[0]
    return np.array([bc.get(i, 0.0) for i in range(n)])


def compute_eigenvector_centrality(matrix: np.ndarray) -> np.ndarray:
    """Compute eigenvector centrality using networkx. Falls back to zeros if convergence fails."""
    import networkx as nx

    n = matrix.shape[0]
    G = nx.from_numpy_array(matrix, create_using=nx.DiGraph)

    if G.number_of_edges() == 0:
        return np.zeros(n)

    try:
        ec = nx.eigenvector_centrality_numpy(G, weight="weight")
        return np.array([ec.get(i, 0.0) for i in range(n)])
    except (nx.NetworkXException, np.linalg.LinAlgError):
        return np.zeros(n)
