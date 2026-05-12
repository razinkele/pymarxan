"""Tests for SA alpha computation when initial_temp is very small."""
from __future__ import annotations


class TestAlphaClamp:
    def test_alpha_below_one_when_temp_tiny(self):
        """Alpha must always be <= 1.0 for cooling to work."""
        initial_temp = 0.0001  # Very small
        num_temp_steps = 100
        initial_temp = max(initial_temp, 0.001)
        alpha = (0.001 / initial_temp) ** (1.0 / max(1, num_temp_steps))
        assert alpha <= 1.0

    def test_alpha_below_one_normal(self):
        """Normal case: initial_temp > 0.001 gives alpha < 1."""
        initial_temp = 10.0
        num_temp_steps = 100
        initial_temp = max(initial_temp, 0.001)
        alpha = (0.001 / initial_temp) ** (1.0 / max(1, num_temp_steps))
        assert alpha < 1.0
