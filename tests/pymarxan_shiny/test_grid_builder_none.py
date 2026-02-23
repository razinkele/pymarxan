"""Tests for grid builder None-input guard."""
from __future__ import annotations

import pytest

from pymarxan.spatial.grid import generate_planning_grid


class TestGridBuilderInputValidation:
    def test_none_cell_size_raises(self):
        """Passing None as cell_size should raise TypeError or ValueError."""
        with pytest.raises((TypeError, ValueError)):
            generate_planning_grid(
                bounds=(0.0, 0.0, 1.0, 1.0),
                cell_size=None,
            )

    def test_zero_cell_size_raises(self):
        """cell_size=0 should raise ValueError."""
        with pytest.raises(ValueError):
            generate_planning_grid(
                bounds=(0.0, 0.0, 1.0, 1.0),
                cell_size=0.0,
            )
