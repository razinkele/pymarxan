"""Result container for a Zonation rank-removal run."""
from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd


@dataclass
class ZonationResult:
    """Priority ranking of every planning unit, with performance curves.

    ``priority_rank`` maps PU id -> rank in (0, 1], where 1.0 is the
    highest-priority cell (removed last). Ranks are unique by construction (each
    removal position gets a distinct ``(k+1)/n``), so ``top_fraction`` is
    deterministic. ``removal_order`` lists PU ids first-removed (lowest priority)
    first. ``performance_curves`` is wide form: ``prop_landscape_remaining`` and
    ``prop_cost_remaining`` columns plus one ``feat_<id>`` column per feature
    (retained proportion), one row per recorded step.
    """

    priority_rank: dict[int, float]
    removal_order: list[int]
    performance_curves: pd.DataFrame
    rule: str

    def top_fraction(self, f: float) -> set[int]:
        """Return the PU ids in the top ``f`` share by priority rank."""
        if not 0.0 < f <= 1.0:
            raise ValueError(f"f must be in (0, 1], got {f}")
        n = len(self.priority_rank)
        k = math.ceil(f * n)
        ordered = sorted(
            self.priority_rank, key=lambda pu: self.priority_rank[pu], reverse=True
        )
        return set(ordered[:k])

    def to_dataframe(self) -> pd.DataFrame:
        position = {pu: i for i, pu in enumerate(self.removal_order)}
        pus = list(self.priority_rank)
        return pd.DataFrame(
            {
                "pu_id": pus,
                "priority_rank": [self.priority_rank[p] for p in pus],
                "removal_position": [position[p] for p in pus],
            }
        )
