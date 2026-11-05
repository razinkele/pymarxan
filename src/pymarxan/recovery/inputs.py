"""Inputs of a nature recovery plan: where habitat exists, where it could be restored, how
likely restoration is to succeed and what each action costs (spec §3.1)."""
from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from pymarxan.models.problem import ConservationProblem

DEFAULT_RESTORE_COST_FACTOR = 2.5
DEFAULT_PROTECT_PROP = 0.3
DEFAULT_RESTORE_PROP = 0.2
DEFAULT_PASSIVE_EFFICACY = 0.5
DEFAULT_PASSIVE_COST_FACTOR = 1.0
PLACEHOLDER = "placeholder"
USER = "user"


def _is_finite_number(x: object) -> bool:
    return isinstance(x, int | float | np.integer | np.floating) and not isinstance(
        x, bool
    ) and math.isfinite(float(x))


def _first(items: Sequence[object], n: int = 5) -> str:
    return ", ".join(str(i) for i in items[:n]) + (" …" if len(items) > n else "")


@dataclass(frozen=True)
class PassiveAction:
    """Passive recovery (natural regeneration): success = η_h × active p_success, cost =
    ``cost_factor × unit cost``. Defaults are ROWER placeholders (η 0.5, cost factor 1.0)."""

    efficacy: float = DEFAULT_PASSIVE_EFFICACY
    efficacy_by_habitat: Mapping[int, float] | None = None
    cost_factor: float = DEFAULT_PASSIVE_COST_FACTOR

    def efficacy_for(self, habitat: int) -> float:
        if self.efficacy_by_habitat is not None and habitat in self.efficacy_by_habitat:
            return float(self.efficacy_by_habitat[habitat])
        return float(self.efficacy)


@dataclass(eq=False)
class RecoveryInputs:
    """Everything ``build_recovery_problem`` needs (spec §3.1). Not solvable on its own.

    ``base`` holds planning units (id, cost, status), habitats as features and the
    **existing** habitat amount per unit in ``pu_vs_features``. ``potential`` (columns
    ``pu, feature, amount``) is the **additional** amount restoration would add to a unit.
    ``p_success`` is one probability or a Series indexed by unit id. ``restore_cost`` (a
    Series indexed by unit id) overrides ``restore_cost_factor × cost``;
    ``restore_cost_by_habitat`` adds a per-habitat implementation cost per unit of
    potential to the active restore cost. Targets: ``protect_prop`` of each habitat's
    existing amount; ``restore_target`` (absolute, expected units) or ``restore_prop`` of
    the maximum expected restorable amount. ``spf`` ``None`` = scaled per feature.
    """

    base: ConservationProblem
    potential: pd.DataFrame
    p_success: float | pd.Series
    restore_cost: pd.Series | None = None
    restore_cost_factor: float = DEFAULT_RESTORE_COST_FACTOR
    restore_cost_by_habitat: Mapping[int, float] | None = None
    passive: PassiveAction | None = None
    maturity: Mapping[int, float] | None = None
    horizon_years: float | None = None
    protect_prop: float | Mapping[int, float] = DEFAULT_PROTECT_PROP
    restore_prop: float | Mapping[int, float] = DEFAULT_RESTORE_PROP
    restore_target: Mapping[int, float] | None = None
    spf: float | None = None

    # ------------------------------------------------------------------
    # Per-habitat / per-unit views (assume validate() == [])
    # ------------------------------------------------------------------

    def habitat_ids(self) -> list[int]:
        """Habitat (feature) ids in ascending order."""
        return sorted(int(f) for f in self.base.features["id"].values)

    def habitat_names(self) -> dict[int, str]:
        f = self.base.features
        names = f["name"].astype(str).tolist() if "name" in f.columns else [
            str(int(i)) for i in f["id"].values
        ]
        return {int(i): n for i, n in zip(f["id"].values, names, strict=True)}

    def unit_ids(self) -> list[int]:
        """Unit ids in ``base.planning_units`` order."""
        return [int(i) for i in self.base.planning_units["id"].values]

    def unit_costs(self) -> np.ndarray:
        return np.asarray(self.base.planning_units["cost"].values, dtype=float)

    def unit_statuses(self) -> np.ndarray:
        pu = self.base.planning_units
        if "status" not in pu.columns:
            return np.zeros(len(pu), dtype=int)
        return np.asarray(pu["status"].values, dtype=int)

    def p_by_unit(self) -> np.ndarray:
        if isinstance(self.p_success, pd.Series):
            return np.array([float(self.p_success[i]) for i in self.unit_ids()], dtype=float)
        return np.full(len(self.unit_ids()), float(self.p_success), dtype=float)

    def potential_matrix(self) -> np.ndarray:
        """(n_units, n_habitats) raw potential, habitats in ``habitat_ids()`` order."""
        units = {u: k for k, u in enumerate(self.unit_ids())}
        habs = {h: k for k, h in enumerate(self.habitat_ids())}
        out = np.zeros((len(units), len(habs)), dtype=float)
        pot = self.potential
        for u, h, a in zip(pot["pu"].values, pot["feature"].values, pot["amount"].values,
                           strict=True):
            out[units[int(u)], habs[int(h)]] += float(a)
        return out

    def existing_matrix(self) -> np.ndarray:
        """(n_units, n_habitats) existing amount from ``base.pu_vs_features``."""
        units = {u: k for k, u in enumerate(self.unit_ids())}
        habs = {h: k for k, h in enumerate(self.habitat_ids())}
        out = np.zeros((len(units), len(habs)), dtype=float)
        pv = self.base.pu_vs_features
        for u, h, a in zip(pv["pu"].values, pv["species"].values, pv["amount"].values,
                           strict=True):
            if int(u) in units and int(h) in habs:
                out[units[int(u)], habs[int(h)]] += float(a)
        return out

    def restore_cost_by_unit(self) -> np.ndarray:
        """Active restore cost per unit, implementation costs included."""
        if self.restore_cost is not None:
            rc = np.array([float(self.restore_cost[i]) for i in self.unit_ids()], dtype=float)
        else:
            rc = self.restore_cost_factor * self.unit_costs()
        if self.restore_cost_by_habitat:
            per = np.array(
                [float(self.restore_cost_by_habitat.get(h, 0.0)) for h in self.habitat_ids()]
            )
            rc = rc + self.potential_matrix() @ per
        return rc

    def passive_cost_by_unit(self) -> np.ndarray | None:
        if self.passive is None:
            return None
        return self.passive.cost_factor * self.unit_costs()

    def maturity_for(self, habitat: int) -> float:
        if self.maturity is not None and habitat in self.maturity:
            return float(self.maturity[habitat])
        return 1.0

    def protect_prop_for(self, habitat: int) -> float:
        if isinstance(self.protect_prop, Mapping):
            return float(self.protect_prop.get(habitat, DEFAULT_PROTECT_PROP))
        return float(self.protect_prop)

    def restore_prop_for(self, habitat: int) -> float:
        if isinstance(self.restore_prop, Mapping):
            return float(self.restore_prop.get(habitat, DEFAULT_RESTORE_PROP))
        return float(self.restore_prop)

    def provenance(self) -> dict[str, str]:
        """``"placeholder"`` for every ecological/economic value left at its default."""
        out = {
            "restore_cost": USER if self.restore_cost is not None or (
                self.restore_cost_factor != DEFAULT_RESTORE_COST_FACTOR
            ) else PLACEHOLDER,
            "protect_prop": USER if self.protect_prop != DEFAULT_PROTECT_PROP else PLACEHOLDER,
            "restore_target": USER if self.restore_target is not None or (
                self.restore_prop != DEFAULT_RESTORE_PROP
            ) else PLACEHOLDER,
            "maturity": USER if self.maturity else PLACEHOLDER,
        }
        if self.passive is not None:
            out["passive_efficacy"] = USER if (
                self.passive.efficacy != DEFAULT_PASSIVE_EFFICACY
                or self.passive.efficacy_by_habitat
            ) else PLACEHOLDER
            out["passive_cost_factor"] = USER if (
                self.passive.cost_factor != DEFAULT_PASSIVE_COST_FACTOR
            ) else PLACEHOLDER
        return out

    # ------------------------------------------------------------------
    # Validation (errors only; warnings come from check_recovery_inputs)
    # ------------------------------------------------------------------

    def validate(self) -> list[str]:
        errors = [f"base: {e}" for e in self.base.validate()]
        pu = self.base.planning_units
        unit_ids = [int(i) for i in pu["id"].values]
        units = set(unit_ids)
        habs: set[int] = set()
        for h in self.base.features["id"].values:
            if not _is_finite_number(h) or float(h) != int(h) or int(h) < 1:
                errors.append(
                    f"habitat id {h!r} must be a positive integer (it maps to feature ids "
                    "2h-1 and 2h)"
                )
            else:
                habs.add(int(h))
        if errors:
            return errors

        cost = np.asarray(pu["cost"].values, dtype=float)
        if (cost < 0).any() or not np.isfinite(cost).all():
            bad = [u for u, c in zip(unit_ids, cost, strict=True) if not c >= 0]
            errors.append(f"unit costs must be finite and >= 0: units {_first(bad)}")
        pv = self.base.pu_vs_features
        ex = np.asarray(pv["amount"].values, dtype=float)
        ex_ok = np.isfinite(ex) & (ex >= 0)
        if not ex_ok.all():
            bad_ex = sorted({int(u) for u in pv["pu"].values[~ex_ok]})
            errors.append(
                f"existing habitat amounts must be finite and >= 0: units {_first(bad_ex)}"
            )

        pot = self.potential
        missing = {"pu", "feature", "amount"} - set(pot.columns)
        if missing:
            errors.append(f"potential is missing columns {sorted(missing)}")
        else:
            bad_u = sorted({int(u) for u in pot["pu"].values} - units)
            bad_h = sorted({int(h) for h in pot["feature"].values} - habs)
            if bad_u:
                errors.append(f"potential names unknown units {_first(list(bad_u))}")
            if bad_h:
                errors.append(f"potential names unknown habitats {_first(list(bad_h))}")
            amt = np.asarray(pot["amount"].values, dtype=float)
            if (amt < 0).any() or not np.isfinite(amt).all():
                errors.append("potential amounts must be finite and >= 0")
            if pot.duplicated(["pu", "feature"]).any():
                errors.append("potential has duplicate (pu, feature) rows")

        errors += self._check_unit_series("p_success", self.p_success, units, 0.0, 1.0)
        if self.restore_cost is not None:
            errors += self._check_unit_series(
                "restore_cost", self.restore_cost, units, 0.0, math.inf,
            )
        elif not (_is_finite_number(self.restore_cost_factor) and self.restore_cost_factor > 0):
            errors.append("restore_cost_factor must be > 0")
        errors += self._check_habitat_map(
            "restore_cost_by_habitat", self.restore_cost_by_habitat, habs, 0.0, math.inf,
            low_open=False,
        )
        errors += self._check_habitat_map("maturity", self.maturity, habs, 0.0, 1.0)
        if self.maturity and any(float(g) != 1.0 for g in self.maturity.values()):
            if self.horizon_years is None:
                errors.append(
                    "horizon_years must be stated when a maturity factor other than 1 is "
                    "given (the factor is judged at that horizon)"
                )
        if self.horizon_years is not None and not (
            _is_finite_number(self.horizon_years) and self.horizon_years > 0
        ):
            errors.append("horizon_years must be > 0")
        for name, value in (("protect_prop", self.protect_prop),
                            ("restore_prop", self.restore_prop)):
            if isinstance(value, Mapping):
                errors += self._check_habitat_map(name, value, habs, 0.0, 1.0, low_open=False)
            elif not (_is_finite_number(value) and 0.0 <= float(value) <= 1.0):
                errors.append(f"{name} must be in [0, 1]")
        errors += self._check_habitat_map(
            "restore_target", self.restore_target, habs, 0.0, math.inf, low_open=False,
        )
        if self.spf is not None and not (_is_finite_number(self.spf) and self.spf > 0):
            errors.append("spf must be > 0 (or None for the scaled default)")
        if self.passive is not None:
            pa = self.passive
            if not (_is_finite_number(pa.efficacy) and 0.0 < pa.efficacy <= 1.0):
                errors.append("passive efficacy must be in (0, 1]")
            errors += self._check_habitat_map(
                "passive efficacy_by_habitat", pa.efficacy_by_habitat, habs, 0.0, 1.0,
            )
            if not (_is_finite_number(pa.cost_factor) and pa.cost_factor > 0):
                errors.append("passive cost_factor must be > 0")
        if errors:
            return errors

        rc = self.restore_cost_by_unit()
        bad = [u for u, c in zip(unit_ids, rc, strict=True) if not c > 0]
        if bad:
            errors.append(
                f"restore cost must be > 0 in every unit (so restoring a unit without "
                f"potential is never better than no action): units {_first(bad)}"
            )
        pc = self.passive_cost_by_unit()
        if pc is not None:
            bad = [u for u, c in zip(unit_ids, pc, strict=True) if not c > 0]
            if bad:
                errors.append(f"passive cost must be > 0 in every unit: units {_first(bad)}")
            never = [u for u, a, p in zip(unit_ids, rc, pc, strict=True) if p >= a]
            if never:
                errors.append(
                    f"passive cost >= active restore cost in {len(never)} unit(s), so passive "
                    f"recovery is never chosen there: units {_first(never)}"
                )
        return errors

    @staticmethod
    def _check_unit_series(
        name: str, value: object, units: set[int], lo: float, hi: float,
    ) -> list[str]:
        if isinstance(value, pd.Series):
            if not all(_is_finite_number(i) and float(i) == int(i) for i in value.index):
                return [f"{name} index must hold integer unit ids"]
            ids_list = [int(i) for i in value.index]
            counts = Counter(ids_list)
            dups = sorted(i for i, c in counts.items() if c > 1)
            if dups:
                return [f"{name} has duplicate unit ids {_first(dups)}"]
            ids = set(ids_list)
            out = []
            if units - ids:
                out.append(f"{name} has no value for units {_first(sorted(units - ids))}")
            if ids - units:
                out.append(f"{name} names unknown units {_first(sorted(ids - units))}")
            vals = np.asarray(value.values, dtype=float)
            if not (np.isfinite(vals).all() and (vals >= lo).all() and (vals <= hi).all()):
                out.append(f"{name} values must be in [{lo:g}, {hi:g}]")
            return out
        if not (_is_finite_number(value) and lo <= float(value) <= hi):  # type: ignore[arg-type]
            return [f"{name} must be in [{lo:g}, {hi:g}]"]
        return []

    @staticmethod
    def _check_habitat_map(
        name: str, value: Mapping[int, float] | None, habs: set[int], lo: float, hi: float,
        *, low_open: bool = True,
    ) -> list[str]:
        if value is None:
            return []
        out = []
        unknown = sorted(int(k) for k in value if int(k) not in habs)
        if unknown:
            out.append(f"{name} names unknown habitats {_first(list(unknown))}")
        for k, v in value.items():
            ok = _is_finite_number(v) and (lo < v if low_open else lo <= v) and v <= hi
            if not ok:
                bracket = "(" if low_open else "["
                out.append(f"{name}[{k}] = {v!r} must be in {bracket}{lo:g}, {hi:g}]")
        return out
