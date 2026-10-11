"""Zonal conservation problem data model for MarZone-style multi-zone planning."""
from __future__ import annotations

import warnings
from collections import Counter
from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np
import pandas as pd

from pymarxan.models.problem import (
    STATUS_INITIAL_INCLUDE,
    STATUS_LOCKED_IN,
    STATUS_LOCKED_OUT,
    ConservationProblem,
)


def _first_five(items: list[str]) -> str:
    """'a; b; c; d; e …' — the first five items, with an ellipsis when more exist."""
    return "; ".join(items[:5]) + (" …" if len(items) > 5 else "")


def _known_id(value: object, known: set[int]) -> int | None:
    """The integer id ``value`` names when it is integral and in ``known``; else ``None``.

    Accepts ``2`` and ``2.0``; rejects NaN, inf, ``2.5`` and non-numeric values (never raises).
    """
    try:
        as_float = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if not as_float.is_integer():
        return None
    ident = int(as_float)
    return ident if ident in known else None


class PUZoneRule(NamedTuple):
    """Zones a planning unit may take (``ZonalProblem.pu_zone_rules``)."""

    allowed: frozenset[int]
    may_be_unassigned: bool
    start: int | None

    @property
    def forced(self) -> bool:
        """Exactly one allowed zone and the PU must be assigned."""
        return len(self.allowed) == 1 and not self.may_be_unassigned


def rule_violations(rules: list[PUZoneRule], assignment: np.ndarray) -> list[int]:
    """Indices of PUs whose zone in ``assignment`` breaks their rule."""
    bad: list[int] = []
    for i, (rule, zone) in enumerate(zip(rules, assignment, strict=True)):
        z = int(zone)
        if (z == 0 and not rule.may_be_unassigned) or (z != 0 and z not in rule.allowed):
            bad.append(i)
    return bad


def fixed_zones(rules: list[PUZoneRule]) -> dict[int, int]:
    """PU index -> the zone it must take (forced) or 0 (no allowed zone); the rest move."""
    fixed: dict[int, int] = {}
    for i, rule in enumerate(rules):
        if rule.forced:
            (fixed[i],) = rule.allowed
        elif not rule.allowed:
            fixed[i] = 0
    return fixed


def _only(zones: frozenset[int]) -> int | None:
    return next(iter(zones)) if len(zones) == 1 else None


@dataclass
class ZonalProblem(ConservationProblem):
    zones: pd.DataFrame = field(default_factory=lambda: pd.DataFrame())
    zone_costs: pd.DataFrame = field(default_factory=lambda: pd.DataFrame())
    zone_contributions: pd.DataFrame | None = None
    zone_targets: pd.DataFrame | None = None
    zone_boundary_costs: pd.DataFrame | None = None
    pu_locks: pd.DataFrame | None = field(default=None, kw_only=True)
    pu_zones: pd.DataFrame | None = field(default=None, kw_only=True)

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

    # ------------------------------------------------------------------
    # Zone targets (MarZone zones.hpp BuildZoneTarget; reserve.hpp target > 0 gate)
    # ------------------------------------------------------------------

    def zone_target_lookup(
        self, *, include_inert: bool = False,
    ) -> dict[tuple[int, int], float]:
        """Zone targets keyed ``(zone_id, feature_id)`` under MarZone load semantics.

        Rows naming a zone not in ``zone_ids`` or a feature not in ``features["id"]``
        (or a non-integral id) are skipped; a repeated pair keeps its LAST row
        (``zones.hpp`` BuildZoneTarget). Pairs whose surviving target is not > 0 are
        inert (``reserve.hpp`` applies zone targets only when > 0) and are dropped
        unless ``include_inert`` — met-flags and the summary keep them, as met.
        Insertion order is each kept pair's first appearance. ``{}`` when
        ``zone_targets`` is ``None`` or lacks ``zone``/``feature``/``target``.
        Never warns. Last-wins on ``(zone, feature)`` is exact because the reader
        rejects occurrence targettypes 2/3; supporting them makes the key per field.
        """
        zt = self.zone_targets
        if zt is None or not {"zone", "feature", "target"}.issubset(zt.columns):
            return {}
        zones = {int(z) for z in self.zone_ids}
        feats = {int(f) for f in self.features["id"].values}
        out: dict[tuple[int, int], float] = {}
        for z, f, t in zip(
            zt["zone"].tolist(), zt["feature"].tolist(), zt["target"].tolist(), strict=True,
        ):
            zid, fid = _known_id(z, zones), _known_id(f, feats)
            if zid is not None and fid is not None:
                out[(zid, fid)] = float(t)
        if include_inert:
            return out
        return {key: t for key, t in out.items() if t > 0}

    def zone_target_issues(
        self,
    ) -> tuple[list[tuple[int, object, object]], list[tuple[int, int]]]:
        """(unknown rows as ``(position, zone, feature)``, duplicate ``(zone, feature)`` keys).

        The rows ``zone_target_lookup`` skips or collapses; shared by ``validate()`` and
        ``warn_zone_target_issues``. ``([], [])`` when ``zone_targets`` is ``None`` or
        lacks ``zone``/``feature``/``target``.
        """
        zt = self.zone_targets
        if zt is None or not {"zone", "feature", "target"}.issubset(zt.columns):
            return [], []
        zones = {int(z) for z in self.zone_ids}
        feats = {int(f) for f in self.features["id"].values}
        unknown: list[tuple[int, object, object]] = []
        seen: set[tuple[int, int]] = set()
        dups: list[tuple[int, int]] = []
        dup_set: set[tuple[int, int]] = set()
        for pos, (z, f) in enumerate(
            zip(zt["zone"].tolist(), zt["feature"].tolist(), strict=True),
        ):
            zid, fid = _known_id(z, zones), _known_id(f, feats)
            if zid is None or fid is None:
                unknown.append((pos, z, f))
                continue
            key = (zid, fid)
            if key in seen and key not in dup_set:
                dup_set.add(key)
                dups.append(key)
            seen.add(key)
        return unknown, dups

    # ------------------------------------------------------------------
    # PU locks / zone restrictions (MarZone pu.hpp LoadPuLock / PersistPuLock / PersistPuZone)
    # ------------------------------------------------------------------

    def _pu_table_rows(self, table: pd.DataFrame | None) -> list[tuple[object, object]]:
        if table is None or not {"pu", "zone"}.issubset(table.columns):
            return []
        return list(zip(table["pu"].tolist(), table["zone"].tolist(), strict=True))

    def pu_lock_lookup(self) -> dict[int, int]:
        """PU id -> locked zone. Last row per known PU wins, THEN a lock naming an unknown
        zone is dropped and the PU falls through to lower rules (MarZone pu.hpp:483-496)."""
        pus = {int(p) for p in self.planning_units["id"].values}
        zones = {int(z) for z in self.zone_ids}
        last: dict[int, object] = {}
        for p, z in self._pu_table_rows(self.pu_locks):
            pid = _known_id(p, pus)
            if pid is not None:
                last[pid] = z
        out: dict[int, int] = {}
        for pid, z in last.items():
            zid = _known_id(z, zones)
            if zid is not None:
                out[pid] = zid
        return out

    def pu_zone_lookup(self) -> dict[int, frozenset[int]]:
        """PU id -> distinct known zones it is restricted to. A PU whose rows name only
        unknown zones is omitted (unrestricted, MarZone numZones == 0)."""
        pus = {int(p) for p in self.planning_units["id"].values}
        zones = {int(z) for z in self.zone_ids}
        acc: dict[int, set[int]] = {}
        for p, z in self._pu_table_rows(self.pu_zones):
            pid, zid = _known_id(p, pus), _known_id(z, zones)
            if pid is not None and zid is not None:
                acc.setdefault(pid, set()).add(zid)
        return {pid: frozenset(zs) for pid, zs in acc.items()}

    def pu_zone_rules(self) -> list[PUZoneRule]:
        """One rule per PU in ``planning_units`` order; within pymarxan.zones the only place
        that interprets status, pu_locks and pu_zones (spec §2 table, first match wins)."""
        every = frozenset(int(z) for z in self.zone_ids)
        first = min(every) if every else None
        locks = self.pu_lock_lookup()
        restrict = self.pu_zone_lookup()
        pu = self.planning_units
        statuses = (
            pu["status"].values.astype(int) if "status" in pu.columns
            else np.zeros(len(pu), dtype=int)
        )
        rules: list[PUZoneRule] = []
        for pid_raw, status in zip(pu["id"].values, statuses, strict=True):
            pid, s = int(pid_raw), int(status)
            if pid in locks:
                z = locks[pid]
                rule = PUZoneRule(allowed=frozenset({z}), may_be_unassigned=False, start=z)
            elif s == STATUS_LOCKED_OUT:
                rule = PUZoneRule(allowed=frozenset(), may_be_unassigned=True, start=0)
            elif pid in restrict:  # status 2 within puzone, or plain puzone
                zs = restrict[pid]
                rule = PUZoneRule(allowed=zs, may_be_unassigned=False, start=_only(zs))
            elif s == STATUS_LOCKED_IN and first is not None:
                rule = PUZoneRule(allowed=frozenset({first}), may_be_unassigned=False, start=first)
            else:
                start = first if s == STATUS_INITIAL_INCLUDE else None
                rule = PUZoneRule(allowed=every, may_be_unassigned=True, start=start)
            rules.append(rule)
        if any(not r.allowed and not r.may_be_unassigned for r in rules):
            raise AssertionError("pu_zone_rules: a PU with no zone that must be assigned")
        return rules

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
                pairs = list(zip(self.zone_contributions["feature"].tolist(),
                                 self.zone_contributions["zone"].tolist(), strict=True))
                dup_pairs = [p for p, n in Counter(pairs).items() if n > 1]
                if dup_pairs:
                    errors.append(
                        f"zone_contributions has {len(dup_pairs)} duplicate (feature, zone) "
                        "pair(s), resolved last-wins: "
                        + _first_five([f"feature {f}, zone {z}" for f, z in dup_pairs])
                    )

        if self.zone_targets is not None:
            req = {"zone", "feature", "target"}
            if not req.issubset(set(self.zone_targets.columns)):
                errors.append(
                    f"zone_targets missing columns: "
                    f"{sorted(req - set(self.zone_targets.columns))}"
                )
            else:
                unknown_rows, dup_keys = self.zone_target_issues()
                if unknown_rows:
                    errors.append(
                        f"zone_targets references {len(unknown_rows)} row(s) with unknown "
                        "zone/feature ids (skipped by the solvers): "
                        + _first_five([f"zone {z}, feature {f}" for _, z, f in unknown_rows])
                    )
                if dup_keys:
                    errors.append(
                        f"zone_targets has {len(dup_keys)} duplicate (zone, feature) "
                        "pair(s), resolved last-wins by the solvers: "
                        + _first_five([f"zone {z}, feature {f}" for z, f in dup_keys])
                    )
                zt = self.zone_targets
                vals = pd.to_numeric(zt["target"], errors="coerce").to_numpy(dtype=float)
                bad_pos = np.flatnonzero(~np.isfinite(vals))
                if len(bad_pos):
                    zs, fs = zt["zone"].tolist(), zt["feature"].tolist()
                    errors.append(
                        f"zone_targets has {len(bad_pos)} row(s) with a non-finite target "
                        "(treated as inert): "
                        + _first_five([f"zone {zs[i]}, feature {fs[i]}" for i in bad_pos])
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

        if self.zone_targets is not None and "targettype" in self.zone_targets.columns:
            bad = self.zone_targets["targettype"].fillna(0).astype(int) != 0
            if bad.any():
                errors.append(
                    f"zone_targets has {int(bad.sum())} row(s) with unresolved targettype "
                    "!= 0; pass the frame through resolve_zone_target_types(zone_targets, "
                    "pu_vs_features) or use load_zone_project"
                )

        if "target2" in self.features.columns:
            t2 = self.features["target2"].fillna(0.0).astype(float)
            if (t2 > 0).any():
                errors.append(
                    "target2 (clumping) is not supported in zone problems: "
                    f"{int((t2 > 0).sum())} feature(s) have target2 > 0"
                )

        return errors


class ZoneTargetWarning(UserWarning):
    """``zone_targets`` rows the zone solvers skip (unknown ids) or collapse (duplicates)."""


def warn_zone_target_issues(problem: ZonalProblem, solver_name: str) -> None:
    """Emit one ``ZoneTargetWarning`` when ``problem.zone_target_issues()`` is non-empty.

    Called once at the entry of every zone-solver call (``solve`` and
    ``ZoneIISolver.improve``); nothing downstream warns.
    """
    unknown, dups = problem.zone_target_issues()
    if not unknown and not dups:
        return
    parts: list[str] = []
    if unknown:
        parts.append(
            f"{len(unknown)} row(s) with unknown zone/feature ids (skipped, as MarZone "
            "does): " + _first_five([f"zone {z}, feature {f}" for _, z, f in unknown])
        )
    if dups:
        parts.append(
            f"{len(dups)} duplicate (zone, feature) pair(s) (last row wins): "
            + _first_five([f"zone {z}, feature {f}" for z, f in dups])
        )
    warnings.warn(
        f"{solver_name}: zone_targets has " + "; and ".join(parts),
        ZoneTargetWarning,
        stacklevel=3,
    )
