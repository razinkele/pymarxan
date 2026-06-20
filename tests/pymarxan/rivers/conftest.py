"""Shared river-network builders for Phase A tests.

Fixtures return ``RiverNetwork`` objects for the hand-computed topologies
pinned in ``docs/plans/2026-06-20-phase19-rivers-aquatic-restoration-design.md``
§5. Barrier ids/segments follow the design's convention that a barrier sits at
the *downstream end* of its segment (the link ``segment -> down_id``).
"""
from __future__ import annotations

import pandas as pd
import pytest

from pymarxan.rivers import RiverNetwork


def _segments(ids, down_ids, length=10.0):
    return pd.DataFrame(
        {"id": ids, "length": [length] * len(ids), "down_id": down_ids}
    )


def _barriers(ids, segments, pass_up):
    return pd.DataFrame(
        {
            "id": ids,
            "segment": segments,
            "pass_up": pass_up,
            "pass_down": pass_up,  # symmetric
        }
    )


def make_chain() -> RiverNetwork:
    """S1 —[B1]— S2 —[B2]— S3, S1 the outlet, no mouth barrier.

    down: S2->S1, S3->S2. B1 (id 1) on segment S2; B2 (id 2) on segment S3.
    p(B1)=p(B2)=0.5, lengths 10.
    """
    return RiverNetwork(
        segments=_segments([1, 2, 3], [-1, 1, 2]),
        barriers=_barriers([1, 2], [2, 3], [0.5, 0.5]),
    )


def make_ytree() -> RiverNetwork:
    """Y: headwaters S2, S3 join at confluence/outlet S1 (no mouth barrier).

    down: S2->S1, S3->S1. B2 (id 2) on segment S2; B3 (id 3) on segment S3.
    p=0.5, lengths 10. LCA(S2, S3) is the interior segment S1.
    """
    return RiverNetwork(
        segments=_segments([1, 2, 3], [-1, 1, 1]),
        barriers=_barriers([2, 3], [2, 3], [0.5, 0.5]),
    )


def make_ytree_mouth_blocked() -> RiverNetwork:
    """Y-tree plus a fully-blocking mouth barrier B1 (p=0) on the outlet S1."""
    return RiverNetwork(
        segments=_segments([1, 2, 3], [-1, 1, 1]),
        barriers=_barriers([1, 2, 3], [1, 2, 3], [0.0, 0.5, 0.5]),
    )


def make_single() -> RiverNetwork:
    """One segment, no barriers."""
    return RiverNetwork(
        segments=_segments([1], [-1]),
        barriers=pd.DataFrame(
            {"id": [], "segment": [], "pass_up": [], "pass_down": []}
        ),
    )


@pytest.fixture
def chain() -> RiverNetwork:
    return make_chain()


@pytest.fixture
def ytree() -> RiverNetwork:
    return make_ytree()


@pytest.fixture
def ytree_mouth_blocked() -> RiverNetwork:
    return make_ytree_mouth_blocked()


@pytest.fixture
def single() -> RiverNetwork:
    return make_single()
