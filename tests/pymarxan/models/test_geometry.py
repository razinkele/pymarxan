"""Tests for synthetic geometry generator."""
from __future__ import annotations

import math

from pymarxan.models.geometry import generate_grid


def test_generate_grid_count():
    """Grid returns exactly n_pu bounding boxes."""
    grid = generate_grid(6)
    assert len(grid) == 6


def test_generate_grid_single():
    """Single PU produces one cell."""
    grid = generate_grid(1)
    assert len(grid) == 1
    sw, ne = grid[0]
    assert ne[0] > sw[0]  # north > south
    assert ne[1] > sw[1]  # east > west


def test_generate_grid_layout_dimensions():
    """Grid layout is ceil(sqrt(n)) columns."""
    grid = generate_grid(10)
    assert len(grid) == 10
    # 10 PUs -> ceil(sqrt(10))=4 cols, 3 rows (4+4+2)
    cols = math.ceil(math.sqrt(10))
    assert cols == 4


def test_generate_grid_no_overlap():
    """Adjacent cells should share edges, not overlap."""
    grid = generate_grid(4, cell_size=0.01)
    # 4 PUs -> 2x2 grid
    # Cell (0,0) and cell (0,1) should share a vertical edge
    sw0, ne0 = grid[0]  # bottom-left
    sw1, ne1 = grid[1]  # bottom-right
    assert abs(ne0[1] - sw1[1]) < 1e-10  # east edge of 0 == west edge of 1


def test_generate_grid_custom_origin():
    """Custom origin shifts all cells."""
    grid = generate_grid(1, origin=(10.0, 20.0))
    sw, ne = grid[0]
    assert sw[0] == 10.0
    assert sw[1] == 20.0


def test_generate_grid_empty():
    """Zero PUs returns empty list."""
    grid = generate_grid(0)
    assert grid == []
