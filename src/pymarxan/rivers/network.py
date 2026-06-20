"""River-network model for the rivers / barrier-restoration feature (Phase A).

``RiverNetwork`` encodes a rooted river tree with the **downstream-pointer**
convention (every segment names its single downstream neighbour via ``down_id``;
the unique outlet has no downstream). A barrier sits at the *downstream end* of
its ``segment`` — i.e. on the link ``segment -> down_id`` — so a barrier on the
outlet segment models a barrier at the river mouth.

Topology is held in plain NumPy arrays / dicts — **no ``networkx`` dependency**
(it is only an optional extra in this project, and the rooted-tree walks here
need nothing more). See
``docs/plans/2026-06-20-phase19-rivers-aquatic-restoration-design.md`` §4.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

# Outlet sentinel: a segment whose down_id is NA or <= 0 drains to the sea.


def _is_outlet_value(x: Any) -> bool:
    if pd.isna(x):
        return True
    try:
        return int(x) <= 0
    except (TypeError, ValueError):
        return False


@dataclass
class RiverNetwork:
    """A rooted river network of segments and barriers.

    Parameters
    ----------
    segments
        Columns: ``id``, ``length``, ``down_id`` (NA or <= 0 for the outlet),
        optional ``weight`` (defaults to ``length``).
    barriers
        Columns: ``id``, ``segment``, ``pass_up`` (∈ [0, 1]), optional
        ``pass_down`` (defaults to ``pass_up`` — symmetric). For Phase A the
        scalar passability used by DCI is ``pass_up``.
    """

    segments: pd.DataFrame
    barriers: pd.DataFrame

    # Cached topology (built in __post_init__).
    _seg_ids: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int), repr=False)
    _seg_idx: dict[int, int] = field(default_factory=dict, repr=False)
    _parent: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int), repr=False)
    _depth: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int), repr=False)
    _weight: np.ndarray = field(default_factory=lambda: np.zeros(0), repr=False)
    _outlet_idx: int = field(default=-1, repr=False)
    _seg_barriers: dict[int, list[int]] = field(default_factory=dict, repr=False)
    _barrier_pass: dict[int, float] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        seg = self.segments
        bar = self.barriers

        seg_ids = seg["id"].astype(int).to_numpy()
        if len(np.unique(seg_ids)) != len(seg_ids):
            raise ValueError("segment ids must be unique")
        self._seg_ids = seg_ids
        self._seg_idx = {int(s): i for i, s in enumerate(seg_ids)}
        n = len(seg_ids)

        # weights default to length
        if "weight" in seg.columns:
            self._weight = seg["weight"].astype(float).to_numpy()
        else:
            self._weight = seg["length"].astype(float).to_numpy()

        # parent array (segment index of downstream neighbour, -1 for outlet)
        parent = np.full(n, -1, dtype=int)
        down = seg["down_id"].tolist()
        outlets = 0
        for i, d in enumerate(down):
            if _is_outlet_value(d):
                outlets += 1
                continue
            di = self._seg_idx.get(int(d))
            if di is None:
                raise ValueError(f"down_id {d!r} does not resolve to a segment")
            parent[i] = di
        if outlets != 1:
            raise ValueError(f"network must have exactly one outlet (found {outlets})")
        self._parent = parent
        self._outlet_idx = int(np.where(parent == -1)[0][0])

        # acyclicity + connectivity: every node must reach the outlet within n hops
        depth = np.full(n, -1, dtype=int)
        for i in range(n):
            steps = 0
            cur = i
            while parent[cur] != -1:
                cur = parent[cur]
                steps += 1
                if steps > n:
                    raise ValueError("network is not a tree (cycle detected)")
            depth[i] = steps
        self._depth = depth

        # barriers
        self._seg_barriers = {i: [] for i in range(n)}
        self._barrier_pass = {}
        if len(bar) > 0:
            bids = bar["id"].astype(int).to_numpy()
            if len(np.unique(bids)) != len(bids):
                raise ValueError("barrier ids must be unique")
            pass_up = bar["pass_up"].astype(float).to_numpy()
            pass_down = (
                bar["pass_down"].astype(float).to_numpy()
                if "pass_down" in bar.columns
                else pass_up
            )
            for k in range(len(bar)):
                p = float(pass_up[k])
                pd_ = float(pass_down[k])
                if not (0.0 <= p <= 1.0) or not (0.0 <= pd_ <= 1.0):
                    raise ValueError("barrier passabilities must lie in [0, 1]")
                seg_id = int(bar["segment"].iloc[k])
                si = self._seg_idx.get(seg_id)
                if si is None:
                    raise ValueError(
                        f"barrier {int(bids[k])} is on segment {seg_id}, which does not exist"
                    )
                bid = int(bids[k])
                self._seg_barriers[si].append(bid)
                self._barrier_pass[bid] = p  # symmetric: pass_up is the scalar p

    # --- basic accessors ------------------------------------------------

    @property
    def n_segments(self) -> int:
        return len(self._seg_ids)

    @property
    def n_barriers(self) -> int:
        return len(self._barrier_pass)

    @property
    def outlet(self) -> int:
        """Segment id of the unique outlet."""
        return int(self._seg_ids[self._outlet_idx])

    def weights(self) -> dict[int, float]:
        """Per-segment weight w_i = weight_i / Σ weight (sums to 1)."""
        total = float(self._weight.sum())
        return {
            int(self._seg_ids[i]): float(self._weight[i] / total)
            for i in range(self.n_segments)
        }

    def barrier_passabilities(
        self, override: dict[int, float] | None = None
    ) -> dict[int, float]:
        """Effective passability per barrier: ``override`` value if present
        (e.g. a removed barrier → 1.0), else the native ``pass_up``."""
        if not override:
            return dict(self._barrier_pass)
        return {
            bid: float(override.get(bid, native))
            for bid, native in self._barrier_pass.items()
        }

    # --- topology -------------------------------------------------------

    def _seg_passability(self, si: int, passmap: dict[int, float]) -> float:
        p = 1.0
        for bid in self._seg_barriers[si]:
            p *= passmap[bid]
        return p

    def root_products(
        self, override: dict[int, float] | None = None
    ) -> dict[int, float]:
        """For each segment, the product of barrier passabilities from that
        segment down to the mouth (= ``c_i`` for the diadromous DCI)."""
        passmap = self.barrier_passabilities(override)
        n = self.n_segments
        rp = np.ones(n)
        # process parents before children (ascending depth)
        for i in np.argsort(self._depth):
            seg_p = self._seg_passability(int(i), passmap)
            par = self._parent[i]
            rp[i] = seg_p if par == -1 else seg_p * rp[par]
        return {int(self._seg_ids[i]): float(rp[i]) for i in range(n)}

    def lca(self, a: int, b: int) -> int:
        """Lowest common ancestor of segments ``a`` and ``b`` (segment id)."""
        ai, bi = self._seg_idx[int(a)], self._seg_idx[int(b)]
        while self._depth[ai] > self._depth[bi]:
            ai = self._parent[ai]
        while self._depth[bi] > self._depth[ai]:
            bi = self._parent[bi]
        while ai != bi:
            ai = self._parent[ai]
            bi = self._parent[bi]
        return int(self._seg_ids[ai])

    def path_barriers(self, a: int, b: int) -> list[int]:
        """Barrier ids on the undirected path between segments ``a`` and ``b``.

        Walks up from each endpoint to (but not including) their LCA, so
        barriers *below* the confluence are correctly excluded — the path
        from ``a`` to ``b`` does not cross them.
        """
        li = self._seg_idx[self.lca(a, b)]
        out: list[int] = []
        for start in (self._seg_idx[int(a)], self._seg_idx[int(b)]):
            cur = start
            while cur != li:
                out.extend(self._seg_barriers[cur])
                cur = self._parent[cur]
        return out

    def path_barriers_to_mouth(self, a: int) -> list[int]:
        """Barrier ids from segment ``a`` down to the mouth (inclusive of the
        outlet's barrier, i.e. a mouth barrier)."""
        out: list[int] = []
        cur = self._seg_idx[int(a)]
        while cur != -1:
            out.extend(self._seg_barriers[cur])
            cur = self._parent[cur]
        return out
