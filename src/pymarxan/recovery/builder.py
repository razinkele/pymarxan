"""Build a nature recovery plan as an ordinary Marxan-with-Zones problem (spec §3.2–3.3).

Zones: unassigned (0) = no action, 1 = protect, 2 = restore (active), 3 = restore
(passive, only with ``RecoveryInputs.passive``). Each habitat ``h`` becomes two features:
``2h-1`` = ``<name>_existing`` (credited 1 under every action) and ``2h`` =
``<name>_potential`` (amount = potential × p_success; credited 0 / g_h / η_h·g_h under
protect / active / passive). All targets are MarZone overall targets in expected units.
"""
from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np
import pandas as pd

from pymarxan.recovery.inputs import RecoveryInputs
from pymarxan.zones.diagnostics import diagnose_zone_infeasibility
from pymarxan.zones.model import ZonalProblem

ZONE_NO_ACTION = 0
ZONE_PROTECT = 1
ZONE_RESTORE_ACTIVE = 2
ZONE_RESTORE_PASSIVE = 3
ACTION_NAMES: dict[int, str] = {
    ZONE_NO_ACTION: "no_action",
    ZONE_PROTECT: "protect",
    ZONE_RESTORE_ACTIVE: "restore_active",
    ZONE_RESTORE_PASSIVE: "restore_passive",
}
# Scaled species penalty factor: SPF_SCALE x the dearest cost per unit of a feature any
# (unit, action) supplies, so one unit of shortfall costs 10x more than buying it the most
# expensive way. Integrality means no finite factor guarantees heuristics meet every target
# (one missing 0.2 can need a whole unit), so solvers still report the shortfall; on the
# anchors and random 7-8 unit instances SA + II met the targets in ~95 % of runs at 10
# (50 % at 1, 67 % at 2) and always reached the MIP optimum when they did.
SPF_SCALE = 10.0
# Parameters copied from the base project; everything else is set by the builder.
_COPIED_PARAMETERS = ("BLM", "NUMITNS", "NUMTEMP", "COOLING", "MIP_TIME_LIMIT", "MIP_GAP")


def existing_feature_id(habitat: int) -> int:
    return 2 * int(habitat) - 1


def potential_feature_id(habitat: int) -> int:
    return 2 * int(habitat)


class RecoveryInputError(ValueError):
    """``build_recovery_problem`` refused the inputs; ``errors`` lists every reason."""

    def __init__(self, errors: list[str]) -> None:
        self.errors = list(errors)
        super().__init__("; ".join(self.errors))


class RecoveryInputWarning(UserWarning):
    """A recovery input is legal but probably not what the planner meant."""


class RecoveryCheck(NamedTuple):
    errors: list[str]
    warnings: list[str]


@dataclass(eq=False, frozen=True)
class RecoveryMapping:
    """How the built ``ZonalProblem`` maps back to habitats, actions and units.

    ``habitats``: habitat, name, existing_feature, potential_feature, existing_total,
    existing_target, restore_max (most active restoration could deliver, expected units),
    restore_target (expected units), maturity, efficacy (NaN without passive).
    ``units``: pu, cost, status (as given), p_success, restore_cost, passive_cost (NaN
    without passive), restorable (any expected potential > 0).
    ``potential``: pu, habitat, amount — raw potential (for re-scoring at other p).
    """

    habitats: pd.DataFrame
    units: pd.DataFrame
    potential: pd.DataFrame
    zones: dict[int, str]
    passive: bool
    horizon_years: float | None
    spf_scaled: bool
    provenance: dict[str, str] = field(default_factory=dict)

    def action_names(self, assignment: np.ndarray) -> list[str]:
        return [self.zones[int(z)] for z in assignment]


def scaled_spf(problem: ZonalProblem) -> dict[int, float]:
    """Feature id -> ``SPF_SCALE × max_{unit, zone} zone_cost / (amount × contribution)``
    over the (unit, zone) pairs that supply the feature; 1.0 for a feature nothing supplies.
    """
    zc = {(int(p), int(z)): float(c) for p, z, c in zip(
        problem.zone_costs["pu"], problem.zone_costs["zone"], problem.zone_costs["cost"],
        strict=True)}
    contrib = problem.contribution_lookup()
    zones = sorted(int(z) for z in problem.zone_ids)
    dearest: dict[int, float] = {}
    pv = problem.pu_vs_features
    for f, u, a in zip(pv["species"], pv["pu"], pv["amount"], strict=True):
        for z in zones:
            supply = float(a) * contrib.get((int(f), z), 0.0)
            if supply > 0:
                ratio = zc.get((int(u), z), 0.0) / supply
                dearest[int(f)] = max(dearest.get(int(f), 0.0), ratio)
    return {
        int(f): SPF_SCALE * dearest[int(f)] if dearest.get(int(f), 0.0) > 0 else 1.0
        for f in problem.features["id"]
    }


def _pu_vs_features(inputs: RecoveryInputs) -> pd.DataFrame:
    units = inputs.unit_ids()
    habs = inputs.habitat_ids()
    existing = inputs.existing_matrix()
    expected = inputs.potential_matrix() * inputs.p_by_unit()[:, None]
    rows: list[tuple[int, int, float]] = []
    for k, h in enumerate(habs):
        for i, u in enumerate(units):
            if existing[i, k] > 0:
                rows.append((existing_feature_id(h), u, float(existing[i, k])))
        for i, u in enumerate(units):
            if expected[i, k] > 0:
                rows.append((potential_feature_id(h), u, float(expected[i, k])))
    df = pd.DataFrame(rows, columns=["species", "pu", "amount"])
    return df.sort_values(["species", "pu"], kind="stable").reset_index(drop=True)


def _assemble(inputs: RecoveryInputs) -> tuple[ZonalProblem, RecoveryMapping]:
    """The problem and mapping, assuming ``inputs.validate() == []``."""
    units = inputs.unit_ids()
    habs = inputs.habitat_ids()
    names = inputs.habitat_names()
    cost = inputs.unit_costs()
    status = inputs.unit_statuses()
    p = inputs.p_by_unit()
    pot = inputs.potential_matrix()
    existing = inputs.existing_matrix()
    expected = pot * p[:, None]
    restorable = expected.sum(axis=1) > 0
    actable = status != 3
    passive = inputs.passive
    rc = inputs.restore_cost_by_unit()
    pc = inputs.passive_cost_by_unit()

    zone_ids = [ZONE_PROTECT, ZONE_RESTORE_ACTIVE] + (
        [ZONE_RESTORE_PASSIVE] if passive is not None else []
    )
    zones = pd.DataFrame({"id": zone_ids, "name": [ACTION_NAMES[z] for z in zone_ids]})
    if min(zone_ids) != ZONE_PROTECT:  # status-2 PUs land in min(zone_ids)
        raise AssertionError("protect must be the lowest zone id")

    # Restoring a unit without potential would only re-buy its existing habitat; charging
    # protect + restore there keeps it strictly worse than protection (and than no action).
    surcharge = np.where(restorable, 0.0, cost)
    zc_rows: list[tuple[int, int, float]] = []
    for i, u in enumerate(units):
        zc_rows.append((u, ZONE_PROTECT, float(cost[i])))
        zc_rows.append((u, ZONE_RESTORE_ACTIVE, float(rc[i] + surcharge[i])))
        if pc is not None:
            zc_rows.append((u, ZONE_RESTORE_PASSIVE, float(pc[i] + surcharge[i])))
    zone_costs = pd.DataFrame(zc_rows, columns=["pu", "zone", "cost"])

    spf = 1.0 if inputs.spf is None else float(inputs.spf)
    feat_rows: list[tuple[int, str, float, float]] = []
    contrib_rows: list[tuple[int, int, float]] = []
    hab_rows: list[dict[str, object]] = []
    for k, h in enumerate(habs):
        fe, fp = existing_feature_id(h), potential_feature_id(h)
        g = inputs.maturity_for(h)
        eta = passive.efficacy_for(h) if passive is not None else float("nan")
        existing_total = float(existing[:, k].sum())
        existing_target = inputs.protect_prop_for(h) * existing_total
        restore_max = float(expected[actable, k].sum()) * g
        if inputs.restore_target is not None and h in inputs.restore_target:
            restore_target = float(inputs.restore_target[h])
        else:
            restore_target = inputs.restore_prop_for(h) * restore_max
        feat_rows.append((fe, f"{names[h]}_existing", existing_target, spf))
        feat_rows.append((fp, f"{names[h]}_potential", restore_target, spf))
        contrib_rows += [(fe, ZONE_PROTECT, 1.0), (fe, ZONE_RESTORE_ACTIVE, 1.0),
                         (fp, ZONE_PROTECT, 0.0), (fp, ZONE_RESTORE_ACTIVE, g)]
        if passive is not None:
            contrib_rows += [(fe, ZONE_RESTORE_PASSIVE, 1.0), (fp, ZONE_RESTORE_PASSIVE, eta * g)]
        hab_rows.append({
            "habitat": h, "name": names[h], "existing_feature": fe, "potential_feature": fp,
            "existing_total": existing_total, "existing_target": existing_target,
            "restore_max": restore_max, "restore_target": restore_target,
            "maturity": g, "efficacy": eta,
        })
    features = pd.DataFrame(feat_rows, columns=["id", "name", "target", "spf"])
    contributions = pd.DataFrame(contrib_rows, columns=["feature", "zone", "contribution"])

    # Status: 3 stays (no action); 1 is treated as 0; 2 is written as explicit rules.
    built_status = np.where(status == 1, 0, status)
    locks = [(u, ZONE_PROTECT) for i, u in enumerate(units)
             if status[i] == 2 and not restorable[i]]
    restrict = [(u, z) for i, u in enumerate(units) if status[i] == 2 and restorable[i]
                for z in zone_ids]
    planning_units = pd.DataFrame({"id": units, "cost": cost, "status": built_status})

    params: dict[str, object] = {
        k: inputs.base.parameters[k] for k in _COPIED_PARAMETERS if k in inputs.base.parameters
    }
    params.setdefault("BLM", 0.0)
    params.update({"MANDATORYALLOCATION": 0, "MISSLEVEL": 1.0})

    problem = ZonalProblem(
        planning_units=planning_units,
        features=features,
        pu_vs_features=_pu_vs_features(inputs),
        boundary=None if inputs.base.boundary is None else inputs.base.boundary.copy(),
        parameters=params,
        zones=zones,
        zone_costs=zone_costs,
        zone_contributions=contributions,
        pu_locks=pd.DataFrame(locks, columns=["pu", "zone"]) if locks else None,
        pu_zones=pd.DataFrame(restrict, columns=["pu", "zone"]) if restrict else None,
    )
    if inputs.spf is None:
        spf_of = scaled_spf(problem)
        problem.features["spf"] = [spf_of[int(f)] for f in problem.features["id"]]
    potential_long = inputs.potential[["pu", "feature", "amount"]].rename(
        columns={"feature": "habitat"}
    ).astype({"pu": int, "habitat": int, "amount": float})
    mapping = RecoveryMapping(
        habitats=pd.DataFrame(hab_rows),
        units=pd.DataFrame({
            "pu": units, "cost": cost, "status": status, "p_success": p,
            "restore_cost": rc,
            "passive_cost": pc if pc is not None else np.full(len(units), np.nan),
            "restorable": restorable,
        }),
        potential=potential_long.sort_values(["habitat", "pu"]).reset_index(drop=True),
        zones={z: ACTION_NAMES[z] for z in [ZONE_NO_ACTION, *zone_ids]},
        passive=passive is not None,
        horizon_years=None if inputs.horizon_years is None else float(inputs.horizon_years),
        spf_scaled=inputs.spf is None,
        provenance=inputs.provenance(),
    )
    return problem, mapping


def _habitat_messages(messages: list[str], mapping: RecoveryMapping) -> list[str]:
    """Name the habitat and target kind in ``diagnose_zone_infeasibility`` messages."""
    label: dict[int, str] = {}
    hab = mapping.habitats
    for name, fe, fp in zip(hab["name"], hab["existing_feature"], hab["potential_feature"],
                            strict=True):
        label[int(fe)] = f"protect target for habitat {name!r}"
        label[int(fp)] = f"restore target for habitat {name!r} (expected units)"
    out = []
    for msg in messages:
        m = re.search(r"feature (\d+)", msg)
        prefix = label.get(int(m.group(1))) if m else None
        out.append(f"{prefix}: {msg}" if prefix else msg)
    return out


def check_recovery_inputs(inputs: RecoveryInputs) -> RecoveryCheck:
    """Errors (``build_recovery_problem`` refuses) and warnings (it builds and warns).

    Errors: everything ``RecoveryInputs.validate`` reports, then targets no plan can meet
    (``diagnose_zone_infeasibility`` on the built problem, in expected units). Warnings:
    habitats without restoration potential; units where a restore action costs less than
    protection while holding existing habitat (restoring is then the cheaper way to keep
    that habitat, so the plan may label such units "restore").
    """
    errors = inputs.validate()
    if errors:
        return RecoveryCheck(errors, [])
    problem, mapping = _assemble(inputs)
    errors = _habitat_messages(diagnose_zone_infeasibility(problem), mapping)
    warns: list[str] = []
    for name, restore_max in zip(mapping.habitats["name"], mapping.habitats["restore_max"],
                                 strict=True):
        if float(restore_max) <= 0:
            warns.append(
                f"habitat {name!r} has no restoration potential (with p_success > 0 in a "
                "unit that may act); its restore target is 0"
            )
    units = mapping.units
    holds = inputs.existing_matrix().sum(axis=1) > 0
    cheaper = units["restore_cost"].to_numpy() < units["cost"].to_numpy()
    if mapping.passive:
        cheaper |= units["passive_cost"].to_numpy() < units["cost"].to_numpy()
    cheap_units = units["pu"].to_numpy()[cheaper & holds & units["restorable"].to_numpy()]
    if len(cheap_units):
        warns.append(
            f"a restore action costs less than protection in {len(cheap_units)} unit(s) that "
            "hold existing habitat; restoring is then the cheaper way to keep that habitat, "
            "so the plan may label them 'restore': units "
            + ", ".join(str(int(u)) for u in cheap_units[:5])
            + (" …" if len(cheap_units) > 5 else "")
        )
    return RecoveryCheck(errors, warns)


def build_recovery_problem(inputs: RecoveryInputs) -> tuple[ZonalProblem, RecoveryMapping]:
    """The recovery plan as a ``ZonalProblem`` (every zone solver works on it) plus the
    mapping back to habitats and actions. Raises ``RecoveryInputError`` listing every
    error; emits one ``RecoveryInputWarning`` per warning."""
    check = check_recovery_inputs(inputs)
    if check.errors:
        raise RecoveryInputError(check.errors)
    for msg in check.warnings:
        warnings.warn(msg, RecoveryInputWarning, stacklevel=2)
    problem, mapping = _assemble(inputs)
    post = problem.validate()
    if post:
        raise AssertionError(f"built recovery problem fails validate(): {post}")
    return problem, mapping
