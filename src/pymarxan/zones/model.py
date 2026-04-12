"""Zonal conservation problem data model for MarZone-style multi-zone planning."""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from pymarxan.models.problem import ConservationProblem


@dataclass
class ZonalProblem(ConservationProblem):
    zones: pd.DataFrame = field(default_factory=lambda: pd.DataFrame())
    zone_costs: pd.DataFrame = field(default_factory=lambda: pd.DataFrame())
    zone_contributions: pd.DataFrame | None = None
    zone_targets: pd.DataFrame | None = None
    zone_boundary_costs: pd.DataFrame | None = None

    @property
    def n_zones(self) -> int:
        return len(self.zones)

    @property
    def zone_ids(self) -> set:
        return set(self.zones["id"])

    def get_zone_cost(self, pu_id: int, zone_id: int) -> float:
        row = self.zone_costs[
            (self.zone_costs["pu"] == pu_id)
            & (self.zone_costs["zone"] == zone_id)
        ]
        if len(row) == 0:
            return 0.0
        return float(row.iloc[0]["cost"])

    def get_contribution(self, feature_id: int, zone_id: int) -> float:
        if self.zone_contributions is None:
            return 1.0
        row = self.zone_contributions[
            (self.zone_contributions["feature"] == feature_id)
            & (self.zone_contributions["zone"] == zone_id)
        ]
        if len(row) == 0:
            return 1.0
        return float(row.iloc[0]["contribution"])

    def validate(self) -> list[str]:
        errors = super().validate()

        if not {"id", "name"}.issubset(set(self.zones.columns)):
            errors.append("zones missing columns: id, name")

        if not {"pu", "zone", "cost"}.issubset(set(self.zone_costs.columns)):
            errors.append("zone_costs missing columns: pu, zone, cost")
        else:
            pu_ids = set(self.planning_units["id"])
            z_ids = self.zone_ids
            # Build set of existing (pu, zone) pairs in O(R) for O(1) lookup
            existing_pairs = set(
                zip(self.zone_costs["pu"].values, self.zone_costs["zone"].values)
            )
            for pid in pu_ids:
                for zid in z_ids:
                    if (pid, zid) not in existing_pairs:
                        errors.append(
                            f"zone_costs missing entry for PU {pid}, zone {zid}"
                        )
                        break
                if errors and "zone_costs missing entry" in errors[-1]:
                    break

        if self.zone_contributions is not None:
            req = {"feature", "zone", "contribution"}
            if not req.issubset(set(self.zone_contributions.columns)):
                errors.append(
                    f"zone_contributions missing columns: "
                    f"{sorted(req - set(self.zone_contributions.columns))}"
                )

        if self.zone_targets is not None:
            req = {"zone", "feature", "target"}
            if not req.issubset(set(self.zone_targets.columns)):
                errors.append(
                    f"zone_targets missing columns: "
                    f"{sorted(req - set(self.zone_targets.columns))}"
                )

        if self.zone_boundary_costs is not None:
            req = {"zone1", "zone2", "cost"}
            if not req.issubset(set(self.zone_boundary_costs.columns)):
                errors.append(
                    f"zone_boundary_costs missing columns: "
                    f"{sorted(req - set(self.zone_boundary_costs.columns))}"
                )

        return errors
