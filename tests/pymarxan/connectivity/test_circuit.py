"""Tests for circuit-theory (current-flow) connectivity.

Effective resistance is the circuit-theoretic analogue of least-cost-path
distance: it integrates *all* paths between two cells, not just the
cheapest one, which is why it is the standard Circuitscape connectivity
measure. The expected values below are hand-computable from elementary
resistor-network rules (series add; parallel combine as the harmonic sum).
"""
from __future__ import annotations

import numpy as np
import pytest

from pymarxan.connectivity.circuit import current_flow_to_matrix


def test_single_edge_effective_resistance_equals_edge_resistance():
    # Two adjacent unit-resistance cells -> one resistor of resistance 1.
    raster = np.array([[1.0, 1.0]])
    matrix = current_flow_to_matrix(raster, [(0, 0), (0, 1)])
    assert matrix.shape == (2, 2)
    assert matrix[0, 1] == pytest.approx(1.0)
    assert matrix[1, 0] == pytest.approx(1.0)
    assert matrix[0, 0] == pytest.approx(0.0)


def test_series_resistors_add():
    # Three unit cells in a row: end-to-end is two resistors in series = 2.
    raster = np.array([[1.0, 1.0, 1.0]])
    matrix = current_flow_to_matrix(raster, [(0, 0), (0, 2)])
    assert matrix[0, 1] == pytest.approx(2.0)


def test_square_opposite_corners_is_one_ohm():
    # 2x2 grid of unit resistors (a 4-cycle). Opposite corners: two
    # parallel paths of resistance 2 each -> 2||2 = 1.
    raster = np.ones((2, 2))
    matrix = current_flow_to_matrix(raster, [(0, 0), (1, 1)])
    assert matrix[0, 1] == pytest.approx(1.0)


def test_square_adjacent_corners_is_three_quarter_ohm():
    # Adjacent corners of the unit 4-cycle: direct resistor (1) in parallel
    # with the way around (3) -> 1||3 = 0.75.
    raster = np.ones((2, 2))
    matrix = current_flow_to_matrix(raster, [(0, 0), (0, 1)])
    assert matrix[0, 1] == pytest.approx(0.75)


def test_matrix_is_symmetric_with_zero_diagonal():
    raster = np.array([[1.0, 2.0, 1.0], [2.0, 1.0, 3.0]])
    coords = [(0, 0), (1, 2), (0, 2)]
    matrix = current_flow_to_matrix(raster, coords)
    assert np.allclose(matrix, matrix.T)
    assert np.allclose(np.diag(matrix), 0.0)


def test_higher_resistance_raises_effective_resistance():
    # A high-resistance middle cell makes the ends less connected (larger
    # effective resistance) than an all-low-resistance row.
    low = current_flow_to_matrix(np.array([[1.0, 1.0, 1.0]]), [(0, 0), (0, 2)])
    high = current_flow_to_matrix(np.array([[1.0, 9.0, 1.0]]), [(0, 0), (0, 2)])
    assert high[0, 1] > low[0, 1]


def test_single_planning_unit_returns_zero_matrix():
    matrix = current_flow_to_matrix(np.ones((2, 2)), [(0, 0)])
    assert matrix.shape == (1, 1)
    assert matrix[0, 0] == pytest.approx(0.0)


def test_non_2d_raster_raises():
    with pytest.raises(ValueError, match="2-D"):
        current_flow_to_matrix(np.ones(4), [(0, 0)])


def test_coord_out_of_bounds_raises():
    with pytest.raises(ValueError, match="outside raster"):
        current_flow_to_matrix(np.ones((2, 2)), [(0, 0), (5, 5)])


def test_non_positive_resistance_raises():
    with pytest.raises(ValueError, match="positive"):
        current_flow_to_matrix(np.array([[1.0, 0.0]]), [(0, 0), (0, 1)])


def test_non_finite_resistance_raises():
    with pytest.raises(ValueError, match="finite"):
        current_flow_to_matrix(np.array([[1.0, np.inf]]), [(0, 0), (0, 1)])


def test_two_planning_units_in_same_cell_have_zero_resistance():
    # Co-located PUs (same grid cell) are perfectly connected.
    matrix = current_flow_to_matrix(np.ones((2, 2)), [(0, 0), (0, 0)])
    assert matrix.shape == (2, 2)
    assert matrix[0, 1] == pytest.approx(0.0)
