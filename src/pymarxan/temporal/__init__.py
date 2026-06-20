"""Dynamic / multi-period reserve design.

Reserve selection that unfolds over time rather than all at once:

- :func:`dynamic_reserve_greedy` — Costello–Polasky informed-myopic scheduling
  (protect by value × loss-risk over a per-period budget).
- :func:`two_stage_reserve_mip` — Snyder–Haight–ReVelle two-stage stochastic
  maximal-coverage MIP (act now vs. recourse under future scenarios).
"""
from __future__ import annotations

from pymarxan.temporal.dynamic import DynamicScheduleResult, dynamic_reserve_greedy
from pymarxan.temporal.two_stage import TwoStageResult, two_stage_reserve_mip

__all__ = [
    "DynamicScheduleResult",
    "TwoStageResult",
    "dynamic_reserve_greedy",
    "two_stage_reserve_mip",
]
