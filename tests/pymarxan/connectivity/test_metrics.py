import numpy as np

from pymarxan.connectivity.metrics import (
    compute_betweenness_centrality,
    compute_eigenvector_centrality,
    compute_in_degree,
    compute_out_degree,
)


def _make_matrix():
    """4-node connectivity matrix (asymmetric)."""
    return np.array([
        [0.0, 0.5, 0.0, 0.0],
        [0.1, 0.0, 0.8, 0.0],
        [0.0, 0.2, 0.0, 0.6],
        [0.0, 0.0, 0.3, 0.0],
    ])


class TestInDegree:
    def test_shape(self):
        m = _make_matrix()
        result = compute_in_degree(m)
        assert len(result) == 4

    def test_values(self):
        m = _make_matrix()
        result = compute_in_degree(m)
        np.testing.assert_array_almost_equal(result, [0.1, 0.7, 1.1, 0.6])


class TestOutDegree:
    def test_values(self):
        m = _make_matrix()
        result = compute_out_degree(m)
        np.testing.assert_array_almost_equal(result, [0.5, 0.9, 0.8, 0.3])


class TestBetweennessCentrality:
    def test_shape(self):
        m = _make_matrix()
        result = compute_betweenness_centrality(m)
        assert len(result) == 4

    def test_values_in_range(self):
        m = _make_matrix()
        result = compute_betweenness_centrality(m)
        assert all(0.0 <= v <= 1.0 for v in result)

    def test_middle_nodes_higher(self):
        m = _make_matrix()
        result = compute_betweenness_centrality(m)
        assert result[1] > result[0] or result[2] > result[3]


class TestEigenvectorCentrality:
    def test_shape(self):
        m = _make_matrix()
        result = compute_eigenvector_centrality(m)
        assert len(result) == 4

    def test_values_nonnegative(self):
        m = _make_matrix()
        result = compute_eigenvector_centrality(m)
        assert all(v >= 0 for v in result)

    def test_disconnected_zero(self):
        m = np.zeros((4, 4))
        m[0, 1] = 1.0
        m[1, 0] = 1.0
        result = compute_eigenvector_centrality(m)
        assert result[2] == 0.0
        assert result[3] == 0.0
