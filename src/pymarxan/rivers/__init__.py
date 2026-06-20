"""River-network restoration: dendritic connectivity (DCI) + barriers.

Phase A ships the read-only metrics: the ``RiverNetwork`` model and the DCI
computations (``dci_diadromous``, ``dci_potamodromous``, ``segment_connectivity``).
Barrier-decision optimization (greedy / SA / MIP) lands in later phases.
"""
from __future__ import annotations

from pymarxan.rivers.dci import (
    dci_diadromous,
    dci_potamodromous,
    segment_connectivity,
)
from pymarxan.rivers.network import RiverNetwork

__all__ = [
    "RiverNetwork",
    "dci_diadromous",
    "dci_potamodromous",
    "segment_connectivity",
]
