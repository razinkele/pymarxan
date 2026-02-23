"""Tests for network_view edge cap."""
from __future__ import annotations


class TestEdgeCap:
    def test_max_edges_constant_exists(self):
        """Module should define MAX_EDGES constant."""
        from pymarxan_shiny.modules.mapping import network_view
        assert hasattr(network_view, "MAX_EDGES")
        assert network_view.MAX_EDGES <= 5000
