"""Zonation-style rank-removal prioritization for pymarxan.

Ranks every planning unit by iterative backward removal (Moilanen et al. 2005):
the whole landscape is stripped one cell at a time, removing the least-valuable
cell each step; the removal order is a continuous 0-1 priority map. See
``docs/plans/2026-07-14-zonation-phase-a-design.md``.
"""
from __future__ import annotations

from pymarxan.zonation.rank_removal import rank_removal
from pymarxan.zonation.result import ZonationResult
from pymarxan.zonation.smoothing import SmoothingSpec

__all__ = ["SmoothingSpec", "ZonationResult", "rank_removal"]
