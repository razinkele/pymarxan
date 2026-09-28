"""Zonal conservation problem data model for MarZone-style multi-zone planning."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
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

    # ------------------------------------------------------------------
    # Index conventions shared by objective.py, cache.py, mip_solver.py, writers.py
    # ------------------------------------------------------------------

    def zone_index(self) -> dict[int, int]:
        """Zone id -> matrix row: rank in ``sorted(zone_ids)`` + 1 (row 0 = unassigned)."""
        return {int(zid): k + 1 for k, zid in enumerate(sorted(self.zone_ids))}

    def feature_index(self) -> dict[int, int]:
        """Feature id -> matrix column, in ``features["id"]`` order."""
        return {int(fid): j for j, fid in enumerate(self.features["id"].values)}

    # ------------------------------------------------------------------
    # Contributions (MarZone zones.hpp:619-627 with a table, :651-668 without)
    # ------------------------------------------------------------------

    def contribution_default(self) -> float:
        """Contribution of an unlisted (feature, zone) pair.

        MarZone zero-fills the contribution array whenever a contribution file is
        supplied and only sets the listed pairs (``zones.hpp:619``); the all-ones
        default applies only when no file exists (``zones.hpp:651``).
        """
        return 1.0 if self.zone_contributions is None else 0.0

    def contribution_lookup(self) -> dict[tuple[int, int], float]:
        """Listed contributions keyed ``(feature_id, zone_id)``. Unlisted pairs are absent."""
        if self.zone_contributions is None:
            return {}
        zc = self.zone_contributions
        return {
            (int(f), int(z)): float(c)
            for f, z, c in zip(
                zc["feature"].values, zc["zone"].values, zc["contribution"].values,
                strict=True,
            )
        }

    def get_contribution(self, feature_id: int, zone_id: int) -> float:
        return self.contribution_lookup().get(
            (int(feature_id), int(zone_id)), self.contribution_default(),
        )

    def contribution_matrix(self) -> np.ndarray:
        """(n_zones + 1, n_feat) contribution per (zone row, feature column); row 0 is zero."""
        zidx = self.zone_index()
        fidx = self.feature_index()
        m = np.zeros((self.n_zones + 1, self.n_features), dtype=np.float64)
        m[1:, :] = self.contribution_default()
        for (fid, zid), c in self.contribution_lookup().items():
            row = zidx.get(zid)
            col = fidx.get(fid)
            if row is not None and col is not None:
                m[row, col] = c
        return m

    def contribution_gaps(self) -> list[tuple[int, int]]:
        """Unlisted (feature, zone) pairs of a supplied table, in (feature, sorted zone) order.

        Advisory only: the pairs default to ``contribution_default()`` (0.0), ``validate()``
        does not report them, and ``load_zone_project`` warns once when a table is partial.
        ``[]`` when ``zone_contributions`` is ``None``.
        """
        if self.zone_contributions is None:
            return []
        lookup = self.contribution_lookup()
        z_sorted = sorted(int(z) for z in self.zone_ids)
        return [
            (int(fid), z)
            for fid in self.features["id"].values
            for z in z_sorted
            if (int(fid), z) not in lookup
        ]

    # ------------------------------------------------------------------
    # Zone-target weighting (pymarxan extension; MarZone always uses raw amounts)
    # ------------------------------------------------------------------

    def zone_target_contrib(self) -> int:
        """``ZONETARGETCONTRIB`` parameter as 0 or 1; ``ValueError`` for anything else."""
        raw = self.parameters.get("ZONETARGETCONTRIB", 0)
        if isinstance(raw, bool) or isinstance(raw, str) or raw not in (0, 1):
            raise ValueError(
                f"ZONETARGETCONTRIB must be 0 (raw zone targets, MarZone) or 1 "
                f"(contribution-weighted zone targets, pymarxan <= 0.35); got {raw!r}"
            )
        return int(raw)

    def zone_target_weight_matrix(self) -> np.ndarray:
        """Weight applied to amounts when accumulating zone targets.

        All ones (rows 1..n_zones) by default — MarZone ``reserve.hpp:164`` accumulates
        raw amounts — or the contribution matrix when ``ZONETARGETCONTRIB == 1``.
        """
        if self.zone_target_contrib() == 1:
            return self.contribution_matrix()
        w = np.zeros((self.n_zones + 1, self.n_features), dtype=np.float64)
        w[1:, :] = 1.0
        return w

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
            else:
                feat_ids = [int(f) for f in self.features["id"].values]
                lookup = self.contribution_lookup()
                unknown = [
                    (f, z) for (f, z) in lookup
                    if z not in self.zone_ids or f not in feat_ids
                ]
                if unknown:
                    shown = "; ".join(f"feature {f}, zone {z}" for f, z in unknown[:5])
                    more = " …" if len(unknown) > 5 else ""
                    errors.append(
                        f"zone_contributions references {len(unknown)} unknown "
                        f"(feature, zone) pair(s): {shown}{more}"
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

        try:
            self.zone_target_contrib()
        except ValueError as exc:
            errors.append(str(exc))

        if "target2" in self.features.columns:
            t2 = self.features["target2"].fillna(0.0).astype(float)
            if (t2 > 0).any():
                errors.append(
                    "target2 (clumping) is not supported in zone problems: "
                    f"{int((t2 > 0).sum())} feature(s) have target2 > 0"
                )

        return errors
