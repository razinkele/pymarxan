"""Phase 24B: temporal connectivity + habitat-resistance least-cost-path.

Two new sub-modules under ``pymarxan.connectivity``:
- ``temporal.py::compute_temporal_connectivity`` — summary statistic
  over a stack of per-timestep matrices.
- ``resistance.py::habitat_resistance_to_matrix`` — least-cost-path
  connectivity from a 2D habitat-resistance raster.
"""
from __future__ import annotations

import numpy as np
import pytest

# --- Temporal connectivity ---------------------------------------------


def test_temporal_mean_reduces_stack_to_2d():
    """Mean over a stack of (T, n, n) matrices returns an (n, n) summary."""
    from pymarxan.connectivity.temporal import compute_temporal_connectivity
    stack = np.array([
        [[0.0, 1.0], [1.0, 0.0]],
        [[0.0, 3.0], [3.0, 0.0]],
    ])
    summary = compute_temporal_connectivity(stack, reduction="mean")
    np.testing.assert_array_almost_equal(summary, [[0.0, 2.0], [2.0, 0.0]])


def test_temporal_max_reduces_stack_to_2d():
    """Max reduction picks the highest-connectivity timestep per pair."""
    from pymarxan.connectivity.temporal import compute_temporal_connectivity
    stack = np.array([
        [[0.0, 1.0], [1.0, 0.0]],
        [[0.0, 3.0], [3.0, 0.0]],
        [[0.0, 0.5], [0.5, 0.0]],
    ])
    summary = compute_temporal_connectivity(stack, reduction="max")
    np.testing.assert_array_almost_equal(summary, [[0.0, 3.0], [3.0, 0.0]])


def test_temporal_weighted_uses_weights_array():
    """Weighted reduction takes per-timestep weights and computes
    weighted average."""
    from pymarxan.connectivity.temporal import compute_temporal_connectivity
    stack = np.array([
        [[0.0, 2.0], [2.0, 0.0]],
        [[0.0, 4.0], [4.0, 0.0]],
    ])
    # Weights 1, 3 → weighted avg = (2*1 + 4*3) / 4 = 3.5
    summary = compute_temporal_connectivity(
        stack, reduction="weighted", weights=np.array([1.0, 3.0]),
    )
    np.testing.assert_array_almost_equal(summary, [[0.0, 3.5], [3.5, 0.0]])


def test_temporal_weighted_requires_weights():
    """Reduction='weighted' without weights raises ValueError."""
    from pymarxan.connectivity.temporal import compute_temporal_connectivity
    stack = np.zeros((2, 2, 2))
    with pytest.raises(ValueError, match="weights"):
        compute_temporal_connectivity(stack, reduction="weighted")


def test_temporal_rejects_unknown_reduction():
    """Fail-fast on unknown reduction names."""
    from pymarxan.connectivity.temporal import compute_temporal_connectivity
    stack = np.zeros((2, 2, 2))
    with pytest.raises(ValueError, match="reduction"):
        compute_temporal_connectivity(stack, reduction="median")


def test_temporal_validates_stack_shape():
    """Non-3D input is a programming error; reject loudly."""
    from pymarxan.connectivity.temporal import compute_temporal_connectivity
    with pytest.raises(ValueError, match="3-D"):
        compute_temporal_connectivity(np.zeros((4, 4)), reduction="mean")


# --- Habitat resistance → connectivity matrix --------------------------


def test_resistance_returns_n_by_n_matrix():
    """``habitat_resistance_to_matrix`` builds an (n, n) least-cost matrix
    given a list of (x, y) PU coordinates and a 2D resistance raster."""
    from pymarxan.connectivity.resistance import habitat_resistance_to_matrix
    # 3×3 uniform resistance raster; 3 PUs at corners.
    raster = np.ones((3, 3))
    coords = [(0, 0), (2, 0), (2, 2)]
    matrix = habitat_resistance_to_matrix(raster, coords)
    assert matrix.shape == (3, 3)


def test_resistance_diagonal_is_zero():
    """A PU has zero least-cost-path to itself."""
    from pymarxan.connectivity.resistance import habitat_resistance_to_matrix
    raster = np.ones((4, 4))
    coords = [(0, 0), (3, 3)]
    matrix = habitat_resistance_to_matrix(raster, coords)
    assert matrix[0, 0] == 0.0
    assert matrix[1, 1] == 0.0


def test_resistance_symmetric_undirected_graph():
    """The 4-neighbour grid graph is undirected; matrix is symmetric."""
    from pymarxan.connectivity.resistance import habitat_resistance_to_matrix
    rng = np.random.default_rng(0)
    raster = rng.uniform(1.0, 5.0, size=(5, 5))
    coords = [(0, 0), (4, 4), (2, 2)]
    matrix = habitat_resistance_to_matrix(raster, coords)
    np.testing.assert_allclose(matrix, matrix.T, atol=1e-9)


def test_resistance_higher_for_costly_path():
    """A high-resistance corridor between two PUs raises the LCP cost."""
    from pymarxan.connectivity.resistance import habitat_resistance_to_matrix
    # PUs at (0, 0) and (2, 0); raster row 0 has a high-resistance cell
    # at column 1.
    cheap = np.ones((1, 3))
    expensive = np.array([[1.0, 100.0, 1.0]])
    coords = [(0, 0), (0, 2)]
    cheap_matrix = habitat_resistance_to_matrix(cheap, coords)
    expensive_matrix = habitat_resistance_to_matrix(expensive, coords)
    assert expensive_matrix[0, 1] > cheap_matrix[0, 1]
